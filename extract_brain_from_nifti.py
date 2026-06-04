from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure


INSTALL_HINT = (
    "Missing dependency. Install requirements with:\n"
    "  python -m pip install hd-bet nibabel numpy scipy scikit-image pillow\n"
)


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def require_hdbet() -> str:
    exe = shutil.which("hd-bet")
    if exe is None:
        raise RuntimeError(INSTALL_HINT)
    return exe


def ensure_nii_gz(input_path: Path, work_dir: Path, prefix: str) -> Path:
    if input_path.name.endswith(".nii.gz"):
        return input_path
    out_path = work_dir / f"{prefix}_input.nii.gz"
    img = nib.load(str(input_path))
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {img.shape}")
    nib.save(img, str(out_path))
    return out_path


def run_hdbet(input_gz: Path, brain_image: Path, device: str, disable_tta: bool) -> Path:
    hdbet = require_hdbet()
    cmd = [
        hdbet,
        "-i",
        str(input_gz),
        "-o",
        str(brain_image),
        "-device",
        device,
        "--save_bet_mask",
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    subprocess.run(cmd, check=True)

    mask_path = brain_image.parent / f"{nii_stem(brain_image)}_bet.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"HD-BET finished but mask was not found: {mask_path}")
    return mask_path


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 0
    normals[ok] /= lengths[ok, None]
    normals[~ok] = 0
    return normals.astype(np.float32)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    normals = vertex_normals(vertices, faces)
    with path.open("wb") as f:
        f.write(b"hdbet_brain_surface".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for normal, face in zip(normals, faces):
            pts = vertices[face]
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *pts[0]))
            f.write(struct.pack("<3f", *pts[1]))
            f.write(struct.pack("<3f", *pts[2]))
            f.write(struct.pack("<H", 0))


def make_standard_mesh(mask: np.ndarray, img: nib.Nifti1Image, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    volume = ndi.gaussian_filter(mask.astype(np.float32), sigma=smoothing)
    verts_mm, faces, _normals, _values = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=tuple(float(z) for z in zooms),
        step_size=1,
        allow_degenerate=False,
    )
    scale = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    index_to_world_no_spacing = img.affine @ np.linalg.inv(scale)
    hom = np.c_[verts_mm, np.ones(len(verts_mm), dtype=np.float32)]
    world = (index_to_world_no_spacing @ hom.T).T[:, :3].astype(np.float32)
    return world, faces.astype(np.uint32)


def make_highres_mesh(
    mask: np.ndarray,
    img: nib.Nifti1Image,
    factor: float,
    margin_vox: int,
) -> tuple[np.ndarray, np.ndarray]:
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    coords = np.array(np.nonzero(mask)).T
    start = np.maximum(coords.min(axis=0) - margin_vox, 0)
    stop = np.minimum(coords.max(axis=0) + margin_vox + 1, np.array(mask.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    cropped = mask[slices]

    inside = ndi.distance_transform_edt(cropped, sampling=zooms)
    outside = ndi.distance_transform_edt(~cropped, sampling=zooms)
    sdf = (inside - outside).astype(np.float32)
    sdf = ndi.zoom(sdf, zoom=factor, order=3, mode="nearest", prefilter=True).astype(np.float32)

    highres_spacing = tuple(float(z / factor) for z in zooms)
    verts_mm, faces, _normals, _values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=highres_spacing,
        step_size=1,
        allow_degenerate=False,
    )
    voxel_indices = start.astype(np.float32)[None, :] + verts_mm / zooms[None, :]
    hom = np.c_[voxel_indices, np.ones(len(voxel_indices), dtype=np.float32)]
    world = (img.affine @ hom.T).T[:, :3].astype(np.float32)
    return world, faces.astype(np.uint32)


def make_stl(
    mask_path: Path,
    reference_mri_path: Path,
    out_stl: Path,
    stl_factor: float,
    smoothing: float,
) -> tuple[int, int]:
    img = nib.as_closest_canonical(nib.load(str(reference_mri_path)))
    mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
    mask = np.asanyarray(mask_img.dataobj) > 0
    if mask.shape != img.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match reference MRI shape {img.shape}")
    if stl_factor <= 1.0:
        vertices, faces = make_standard_mesh(mask, img, smoothing=smoothing)
    else:
        vertices, faces = make_highres_mesh(mask, img, factor=stl_factor, margin_vox=8)
    write_binary_stl(out_stl, vertices, faces)
    return len(vertices), len(faces)


def make_overlay(mri_path: Path, mask_path: Path, out_path: Path) -> None:
    mri = nib.as_closest_canonical(nib.load(str(mri_path)))
    mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
    data = np.asanyarray(mri.dataobj).astype(np.float32)
    mask = np.asanyarray(mask_img.dataobj) > 0
    vals = data[data > 0]
    lo, hi = np.percentile(vals, [1.0, 99.5])

    def scaled(arr: np.ndarray) -> np.ndarray:
        arr = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        return (arr * 255).astype(np.uint8)

    panels = []
    for name, axis in [("sagittal", 0), ("coronal", 1), ("axial", 2)]:
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        positions = np.linspace(occupied[0], occupied[-1], 5) if occupied.size else np.linspace(0, data.shape[axis] - 1, 5)
        for pos in positions:
            idx = int(round(pos))
            panels.append((scaled(np.rot90(np.take(data, idx, axis=axis))), np.rot90(np.take(mask, idx, axis=axis)), f"{name} {idx}"))

    tile_h = max(p[0].shape[0] for p in panels)
    tile_w = max(p[0].shape[1] for p in panels)
    pad = 18
    canvas = Image.new("RGB", (5 * tile_w, 3 * (tile_h + pad)), (0, 0, 0))
    for i, (gray, msk, label) in enumerate(panels):
        r, c = divmod(i, 5)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        edge = msk ^ ndi.binary_erosion(msk)
        rgb[msk] = (0.62 * rgb[msk] + np.array([25, 165, 255]) * 0.38).astype(np.uint8)
        rgb[edge] = np.array([255, 70, 15], dtype=np.uint8)
        im = Image.fromarray(rgb)
        tile = Image.new("RGB", (tile_w, tile_h + pad), (0, 0, 0))
        tile.paste(im, ((tile_w - im.width) // 2, pad + (tile_h - im.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 2), label, fill=(255, 255, 255))
        canvas.paste(tile, (c * tile_w, r * (tile_h + pad)))
    canvas.save(out_path)


def make_surface_preview(mask_path: Path, out_path: Path) -> None:
    img = nib.as_closest_canonical(nib.load(str(mask_path)))
    mask = np.asanyarray(img.dataobj) > 0

    def render(label: str, axis: int, reverse: bool) -> Image.Image:
        arr = np.flip(mask, axis=axis) if reverse else mask
        hit = arr.any(axis=axis)
        depth = np.argmax(arr, axis=axis).astype(np.float32)
        depth[~hit] = np.nan
        finite = np.isfinite(depth)
        h, w = depth.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if finite.any():
            lo, hi = np.nanpercentile(depth, [1, 99])
            norm = 1.0 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            filled = np.nan_to_num(depth, nan=np.nanmean(depth))
            gy, gx = np.gradient(filled)
            relief = np.clip(norm - 0.025 * gx + 0.025 * gy, 0, 1)
            base = np.array([214, 188, 160], dtype=np.float32)
            rgb[finite] = np.clip(base * (0.45 + 0.55 * relief[finite, None]), 0, 255).astype(np.uint8)
        panel = Image.fromarray(np.rot90(rgb))
        panel.thumbnail((360, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 360), (0, 0, 0))
        canvas.paste(panel, ((360 - panel.width) // 2, (360 - panel.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), label, fill=(255, 255, 255))
        return canvas

    panels = [
        render("left", 0, False),
        render("right", 0, True),
        render("front", 1, True),
        render("top", 2, True),
    ]
    canvas = Image.new("RGB", (720, 720), (0, 0, 0))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 2) * 360, (i // 2) * 360))
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skull-strip a 3D anatomical NIfTI with HD-BET and optionally export an STL brain surface."
    )
    parser.add_argument("input", type=Path, help="Input 3D NIfTI file (.nii or .nii.gz).")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="Output folder. Default: input file folder.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Default: input filename without .nii/.nii.gz.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="HD-BET device. Default: cpu.")
    parser.add_argument("--tta", action="store_true", help="Enable HD-BET test-time augmentation. Slower, sometimes slightly better.")
    parser.add_argument("--no-stl", action="store_true", help="Only write skull-stripped NIfTI and mask.")
    parser.add_argument("--stl-factor", type=float, default=1.0, help="STL upsampling factor. Use 2, 3, or 4 for denser meshes.")
    parser.add_argument("--stl-smoothing", type=float, default=0.35, help="Gaussian smoothing for factor 1 STL. Default: 0.35.")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA overlay PNG.")
    parser.add_argument("--no-preview", action="store_true", help="Skip surface preview PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = (args.out_dir or input_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or nii_stem(input_path)

    work_dir = out_dir / f"{prefix}_hdbet_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_gz = ensure_nii_gz(input_path, work_dir, prefix)
    hdbet_brain = work_dir / f"{prefix}_hdbet_brain.nii.gz"
    hdbet_mask = run_hdbet(input_gz, hdbet_brain, device=args.device, disable_tta=not args.tta)

    final_brain = out_dir / f"{prefix}_brain.nii.gz"
    final_mask = out_dir / f"{prefix}_brain_mask.nii.gz"
    shutil.copy2(hdbet_brain, final_brain)
    shutil.copy2(hdbet_mask, final_mask)

    print(f"wrote: {final_brain}")
    print(f"wrote: {final_mask}")

    if not args.no_qa:
        qa_path = out_dir / f"{prefix}_brain_QA.png"
        make_overlay(input_path, final_mask, qa_path)
        print(f"wrote: {qa_path}")

    if not args.no_preview:
        preview_path = out_dir / f"{prefix}_brain_surface_preview.png"
        make_surface_preview(final_mask, preview_path)
        print(f"wrote: {preview_path}")

    if not args.no_stl:
        stl_path = out_dir / f"{prefix}_brain.stl"
        vertices, faces = make_stl(
            final_mask,
            input_path,
            stl_path,
            stl_factor=args.stl_factor,
            smoothing=args.stl_smoothing,
        )
        print(f"mesh vertices: {vertices}")
        print(f"mesh faces: {faces}")
        print(f"wrote: {stl_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

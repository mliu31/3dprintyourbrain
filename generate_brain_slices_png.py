from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def load_canonical(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(img.dataobj)
    return img, data


def scale_to_u8(data: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    values = data[np.isfinite(data) & (data > 0)]
    if values.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(values.astype(np.float32), [lower_pct, upper_pct])
    scaled = np.clip((data.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def choose_positions(volume_shape: tuple[int, int, int], mask: np.ndarray | None, axis: int, n_slices: int) -> np.ndarray:
    if mask is not None and mask.any():
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        if occupied.size:
            return np.linspace(occupied[0], occupied[-1], n_slices)
    return np.linspace(0, volume_shape[axis] - 1, n_slices + 2)[1:-1]


def make_slices_png(
    nifti_path: Path,
    out_path: Path,
    mask_path: Path | None,
    slices_per_axis: int,
    lower_pct: float,
    upper_pct: float,
) -> None:
    _img, data = load_canonical(nifti_path)
    data = data.astype(np.float32)
    gray_volume = scale_to_u8(data, lower_pct=lower_pct, upper_pct=upper_pct)
    mask = None
    if mask_path is not None:
        _mask_img, mask_data = load_canonical(mask_path)
        mask = mask_data > 0
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match NIfTI shape {data.shape}")

    panels: list[tuple[np.ndarray, np.ndarray | None, str]] = []
    for name, axis in [("sagittal", 0), ("coronal", 1), ("axial", 2)]:
        for pos in choose_positions(data.shape, mask, axis, slices_per_axis):
            idx = int(round(pos))
            gray = np.rot90(np.take(gray_volume, idx, axis=axis))
            msk = np.rot90(np.take(mask, idx, axis=axis)) if mask is not None else None
            panels.append((gray, msk, f"{name} {idx}"))

    tile_h = max(p[0].shape[0] for p in panels)
    tile_w = max(p[0].shape[1] for p in panels)
    label_h = 18
    canvas = Image.new("RGB", (slices_per_axis * tile_w, 3 * (tile_h + label_h)), (0, 0, 0))

    for i, (gray, msk, label) in enumerate(panels):
        row, col = divmod(i, slices_per_axis)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        if msk is not None:
            edge = msk ^ ndi.binary_erosion(msk)
            rgb[msk] = (0.62 * rgb[msk] + np.array([25, 165, 255]) * 0.38).astype(np.uint8)
            rgb[edge] = np.array([255, 70, 15], dtype=np.uint8)

        panel = Image.fromarray(rgb)
        tile = Image.new("RGB", (tile_w, tile_h + label_h), (0, 0, 0))
        tile.paste(panel, ((tile_w - panel.width) // 2, label_h + (tile_h - panel.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 2), label, fill=(255, 255, 255))
        canvas.paste(tile, (col * tile_w, row * (tile_h + label_h)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PNG montage of sagittal/coronal/axial NIfTI slices.")
    parser.add_argument("nifti", type=Path, help="Input NIfTI file.")
    parser.add_argument("-m", "--mask", type=Path, default=None, help="Optional mask NIfTI to overlay.")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--slices-per-axis", type=int, default=5)
    parser.add_argument("--lower-pct", type=float, default=1.0)
    parser.add_argument("--upper-pct", type=float, default=99.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.nifti.with_name(f"{args.nifti.name.replace('.nii.gz', '').replace('.nii', '')}_slices.png")
    make_slices_png(
        nifti_path=args.nifti.resolve(),
        out_path=out.resolve(),
        mask_path=args.mask.resolve() if args.mask else None,
        slices_per_axis=args.slices_per_axis,
        lower_pct=args.lower_pct,
        upper_pct=args.upper_pct,
    )


if __name__ == "__main__":
    main()

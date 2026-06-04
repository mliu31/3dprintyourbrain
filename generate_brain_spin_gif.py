from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image


STL_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ]
)


def read_stl_sample(path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    with path.open("rb") as f:
        f.seek(80)
        triangle_count = struct.unpack("<I", f.read(4))[0]

    step = max(1, int(np.ceil(triangle_count / max_faces)))
    records = np.memmap(path, dtype=STL_DTYPE, mode="r", offset=84, shape=(triangle_count,))
    sample = records[::step]
    vertices = sample["vertices"].astype(np.float32)
    centers = vertices.mean(axis=1)
    normals = sample["normal"].astype(np.float32)

    lengths = np.linalg.norm(normals, axis=1)
    bad = lengths < 1e-7
    if np.any(bad):
        tri = vertices[bad]
        normals[bad] = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-7)
    return centers, normals, triangle_count, step


def render_frame(
    centers: np.ndarray,
    normals: np.ndarray,
    angle: float,
    size: int,
    scale: float,
    splat_radius: int,
) -> Image.Image:
    c = float(np.cos(angle))
    s = float(np.sin(angle))

    x = c * centers[:, 0] - s * centers[:, 1]
    y = s * centers[:, 0] + c * centers[:, 1]
    z = centers[:, 2]

    nx = c * normals[:, 0] - s * normals[:, 1]
    ny = s * normals[:, 0] + c * normals[:, 1]
    nz = normals[:, 2]
    n_view = np.stack([nx, ny, nz], axis=1)

    light = np.array([0.35, 0.55, 0.76], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = 0.34 + 0.66 * np.clip(np.abs(n_view @ light), 0.0, 1.0)
    depth = y.astype(np.float32)

    px = np.rint(x * scale + size / 2).astype(np.int32)
    py = np.rint(size / 2 - z * scale).astype(np.int32)
    offsets = [
        (dx, dy)
        for dy in range(-splat_radius, splat_radius + 1)
        for dx in range(-splat_radius, splat_radius + 1)
    ]

    zbuf = np.full(size * size, -np.inf, dtype=np.float32)
    for dx, dy in offsets:
        xx = px + dx
        yy = py + dy
        valid = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        idx = yy[valid] * size + xx[valid]
        np.maximum.at(zbuf, idx, depth[valid])

    img = np.zeros((size * size, 3), dtype=np.uint8)
    base = np.array([218, 190, 158], dtype=np.float32)
    color = np.clip(base[None, :] * shade[:, None], 0, 255).astype(np.uint8)
    for dx, dy in offsets:
        xx = px + dx
        yy = py + dy
        valid = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        idx = yy[valid] * size + xx[valid]
        visible = depth[valid] >= zbuf[idx] - 1e-4
        img[idx[visible]] = color[valid][visible]

    return Image.fromarray(img.reshape(size, size, 3), mode="RGB")


def default_out_path(stl: Path) -> Path:
    return stl.with_name(f"{stl.stem}_spin_z.gif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a z-axis spin GIF from a binary STL mesh.")
    parser.add_argument("stl", type=Path, help="Input binary STL file.")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--max-faces", type=int, default=520_000, help="Maximum sampled faces for rendering.")
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument("--splat-radius", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl = args.stl.resolve()
    out = (args.out or default_out_path(stl)).resolve()
    centers, normals, triangle_count, step = read_stl_sample(stl, args.max_faces)

    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    centers = centers - ((bbox_min + bbox_max) / 2)[None, :]
    xy_radius = np.linalg.norm(centers[:, :2], axis=1).max()
    z_radius = np.abs(centers[:, 2]).max()
    scale = (args.size * 0.38) / max(float(xy_radius), float(z_radius), 1.0)

    frames = [
        render_frame(centers, normals, 2 * np.pi * i / args.frames, args.size, scale, args.splat_radius)
        for i in range(args.frames)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )

    print(f"source triangles: {triangle_count}")
    print(f"sample step: {step}")
    print(f"rendered faces: {len(centers)}")
    print(f"frames: {args.frames}")
    print(f"size: {args.size}x{args.size}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()

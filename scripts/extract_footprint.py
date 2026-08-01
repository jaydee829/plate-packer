"""Prototype: extract 2D footprint masks from supported STL/OBJ files.

Projects every triangle of the mesh straight down onto XY and rasterizes the
union into a binary mask (the "vertical shadow"), then dilates by the minimum
spacing margin. Saves a PNG per input for eyeballing and prints timing data,
including an fftconvolve benchmark against a full-plate mask, to answer the
seed doc's open questions 1-2 (resolution vs. cost, fillPoly robustness).

Usage:
    python scripts/extract_footprint.py example_stls [--res 0.1] [--spacing 0.5]
    python scripts/extract_footprint.py model.stl --res 0.05
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.signal import fftconvolve

PLATE_MM = (200.0, 130.0)  # benchmark plate size (typical mid-size MSLA)


def extract_footprint(mesh: trimesh.Trimesh, res_mm: float, spacing_mm: float):
    """Return (mask, origin_mm, stats). Mask is uint8 {0,1}, row 0 = min Y."""
    t0 = time.perf_counter()
    tris = mesh.triangles  # (n, 3, 3) mm
    # Real-world pre-supported STLs can contain NaN vertices (seen in the wild
    # at ~0.015% of triangles); they poison the bounds and canvas size.
    finite = np.isfinite(tris).all(axis=(1, 2))
    n_bad = int((~finite).sum())
    if n_bad:
        tris = tris[finite]
    if len(tris) == 0:
        raise ValueError("mesh has no finite triangles")
    tris_xy = tris[:, :, :2]
    # Pad the canvas by the dilation radius so the spacing margin isn't
    # clipped at the borders; origin shifts accordingly.
    r = max(1, round(spacing_mm / res_mm)) if spacing_mm > 0 else 0
    origin = tris_xy.reshape(-1, 2).min(axis=0) - r * res_mm
    size_px = np.ceil((tris_xy.reshape(-1, 2).max(axis=0) - origin) / res_mm).astype(int) + 1 + r

    mask = np.zeros((size_px[1], size_px[0]), dtype=np.uint8)
    tris_px = np.round((tris_xy - origin) / res_mm).astype(np.int32)
    # One fillPoly call per triangle: a single multi-polygon call uses the
    # even-odd rule, so overlapping triangles cancel instead of unioning.
    for tri in tris_px:
        cv2.fillConvexPoly(mask, tri, 1)
    t_raster = time.perf_counter() - t0

    if r > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kernel)

    stats = {
        "triangles": len(mesh.triangles),
        "dropped_nonfinite": n_bad,
        "z_height_mm": float(tris[:, :, 2].max() - tris[:, :, 2].min()),
        "footprint_mm": tuple(
            (tris_xy.reshape(-1, 2).max(axis=0) - tris_xy.reshape(-1, 2).min(axis=0)).round(1)
        ),
        "mask_px": mask.shape,
        "coverage": float(mask.mean()),
        "raster_s": t_raster,
    }
    return mask, origin, stats


def benchmark_convolution(mask: np.ndarray, res_mm: float) -> float:
    """Time one fftconvolve of this footprint against a full plate mask."""
    plate = np.zeros((int(PLATE_MM[1] / res_mm), int(PLATE_MM[0] / res_mm)), dtype=np.float32)
    t0 = time.perf_counter()
    fftconvolve(plate, mask[::-1, ::-1].astype(np.float32), mode="valid")
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="STL/OBJ file or folder of them")
    ap.add_argument("--res", type=float, default=0.1, help="raster resolution, mm/px")
    ap.add_argument("--spacing", type=float, default=0.5, help="min spacing margin, mm")
    ap.add_argument("--out", type=Path, default=Path("footprints"), help="PNG output dir")
    args = ap.parse_args()

    files = (
        sorted(p for p in args.path.rglob("*") if p.suffix.lower() in {".stl", ".obj"})
        if args.path.is_dir()
        else [args.path]
    )
    if not files:
        sys.exit(f"no STL/OBJ files found in {args.path}")
    args.out.mkdir(exist_ok=True)

    for f in files:
        # Flatten the relative path into the PNG name so same-named files in
        # different subfolders don't collide.
        stem = (
            "_".join(f.relative_to(args.path).with_suffix("").parts)
            if args.path.is_dir()
            else f.stem
        )
        t0 = time.perf_counter()
        mesh = trimesh.load_mesh(f, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        t_load = time.perf_counter() - t0

        mask, _origin, s = extract_footprint(mesh, args.res, args.spacing)
        t_conv = benchmark_convolution(mask, args.res)

        png = args.out / f"{stem}_{args.res}mm.png"
        cv2.imwrite(str(png), mask * 255)

        print(f"\n{f.name}")
        if s["dropped_nonfinite"]:
            print(f"  WARNING: dropped {s['dropped_nonfinite']:,} non-finite triangles")
        print(
            f"  triangles={s['triangles']:,}  z_height={s['z_height_mm']:.1f}mm  "
            f"footprint={s['footprint_mm'][0]}x{s['footprint_mm'][1]}mm"
        )
        print(
            f"  mask={s['mask_px'][1]}x{s['mask_px'][0]}px @ {args.res}mm/px  "
            f"coverage={s['coverage']:.1%}"
        )
        print(
            f"  load={t_load:.2f}s  raster={s['raster_s']:.3f}s  "
            f"fftconvolve_vs_{int(PLATE_MM[0])}x{int(PLATE_MM[1])}plate={t_conv:.3f}s"
        )
        print(f"  -> {png}")


if __name__ == "__main__":
    main()

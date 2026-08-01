"""Footprint extraction: mesh -> undilated vertical-shadow mask (ADR-009).

Spacing dilation is applied at load time (plate_packer.loading), never here:
cached footprints are content-addressed by file hash alone, so they may only
contain data intrinsic to the STL.
"""

import time

import cv2
import numpy as np
import trimesh


def extract_footprint(mesh: trimesh.Trimesh, res_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mask, origin_mm, stats). Mask is uint8 {0,1}, row 0 = min Y,
    canvas is the exact triangle bounds. origin_mm = XY of pixel (0,0)."""
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

    flat = tris_xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1

    mask = np.zeros((size_px[1], size_px[0]), dtype=np.uint8)
    tris_px = np.round((tris_xy - origin) / res_mm).astype(np.int32)
    # One fill call per triangle: a single multi-polygon fillPoly call uses
    # the even-odd rule, so overlapping triangles cancel instead of unioning.
    for tri in tris_px:
        cv2.fillConvexPoly(mask, tri, 1)
    t_raster = time.perf_counter() - t0

    stats = {
        "triangles": len(mesh.triangles),
        "dropped_nonfinite": n_bad,
        "z_height_mm": float(tris[:, :, 2].max() - tris[:, :, 2].min()),
        "footprint_mm": tuple((flat.max(axis=0) - flat.min(axis=0)).round(1)),
        "mask_px": mask.shape,
        "coverage": float(mask.mean()),
        "raster_s": t_raster,
    }
    return mask, origin, stats

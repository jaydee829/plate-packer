"""Footprint extraction: mesh -> undilated vertical-shadow mask (ADR-009).

Spacing dilation is applied at load time (plate_packer.loading), never here:
cached footprints are content-addressed by file hash alone, so they may only
contain data intrinsic to the STL.
"""

import time

import cv2
import numpy as np
import trimesh

BAND_MM = 0.25  # Z bin height for the horizontal-cap scan
HORIZ_NZ = 0.9  # |unit normal z| above this = near-horizontal (cap) face
MIN_BASE_FRAC = 0.10  # a cap must cover >= this fraction of the footprint to be a raft
DETECTOR_VERSION = 1  # bump when any detector constant changes (invalidates body masks)


def _raster(tri_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill each (k, 3, 2) int32 triangle into a `shape` uint8 {0,1} canvas.
    One fillConvexPoly per triangle: a batched fillPoly XORs overlaps (bugs.md)."""
    canvas = np.zeros(shape, np.uint8)
    for tri in tri_px:
        cv2.fillConvexPoly(canvas, tri, 1)
    return canvas


def detect_base_cut(tris: np.ndarray, res_mm: float, cap_mm: float) -> float:
    """Offset above the mesh base below which geometry is raft/support base.

    STL meshes are hollow shells, so a slab has no interior triangles at
    mid-height -- only its flat top/bottom *cap* faces carry area. A raft is such
    a big flat horizontal surface, so we detect it directly: bin near-horizontal
    faces by height and cut at the HIGHEST band within [z0, z0 + cap_mm] whose cap
    shadow covers at least MIN_BASE_FRAC of the full footprint (the raft's top
    surface). Returns 0.0 (no cut, the safe default) when no such cap exists in
    the window -- so a raftless model, a wide solid box (top cap out of window),
    a too-small foot, or a base taller than the cap all leave model_body == full.
    """
    if cap_mm <= 0 or len(tris) == 0:
        return 0.0
    xy = tris[:, :, :2]
    z = tris[:, :, 2]
    z0, z1 = float(z.min()), float(z.max())
    if z1 - z0 <= BAND_MM:
        return 0.0
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / res_mm).astype(np.int32)
    full_area = int(_raster(tri_px, shape).sum())
    if full_area == 0:
        return 0.0
    # Near-horizontal (cap) faces: |unit normal_z| > HORIZ_NZ, excluding degenerate.
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    horiz = (norm > 0) & (np.abs(normals[:, 2]) > HORIZ_NZ * norm)
    tri_z = z.mean(axis=1)
    top = min(z0 + cap_mm, z1)
    n_bands = int((top - z0) / BAND_MM)
    cut = 0.0
    for i in range(n_bands):
        band_lo = z0 + i * BAND_MM
        band_hi = band_lo + BAND_MM
        sel = horiz & (tri_z >= band_lo) & (tri_z < band_hi)
        if sel.any() and int(_raster(tri_px[sel], shape).sum()) >= MIN_BASE_FRAC * full_area:
            cut = band_lo - z0  # loop ascends; the highest qualifying cap wins
    return cut


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

"""Footprint extraction: mesh -> undilated vertical-shadow mask (ADR-009).

Spacing dilation is applied at load time (plate_packer.loading), never here:
cached footprints are content-addressed by file hash alone, so they may only
contain data intrinsic to the STL.
"""

import time

import cv2
import numpy as np
import trimesh

BAND_MM = 0.25  # Z step for the cut-depth sweep
MIN_REDUCTION = 0.05  # min footprint drop (fraction) worth excluding a base for
FLAT_EPS = 0.01  # plateau tolerance (fraction of footprint) for the knee
DETECT_RES_MM = 0.2  # coarse raster res for detection (area ratios are scale-tolerant)
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

    Detected by the footprint-area knee: as the cut depth d rises, the model-body
    footprint (shadow of triangles with max Z > z0 + d) shrinks sharply through
    the base then plateaus. We cut at the plateau's start. Computed in a single
    raster pass: paint each pixel with the highest max-Z of the triangles covering
    it (fill in ascending max-Z so the top wins), then area(d) is a threshold on
    that top-reach map. Returns 0.0 (no cut, the safe default) when the footprint
    never drops by MIN_REDUCTION inside [z0, z0 + cap_mm] -- so a raftless model, a
    wide solid box, or a base taller than the cap all leave model_body == full.
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
    tri_zmax = z.max(axis=1)
    # Per-pixel top-reach map: each pixel keeps the tallest (max_Z - z0) of the
    # triangles covering it (fill in ascending order so the tallest paints last).
    # float64 with NO additive offset: a large offset in float32 rounds away the
    # sub-ULP fraction at band boundaries, so reach would drop a pixel one band
    # before the float-compared body mask does (real-STL bug). area(d) = reach > d
    # then matches the body rule max_Z > z0 + d exactly. Empty and base-only pixels
    # stay 0 and are correctly excluded without an offset.
    reach = np.zeros(shape, np.float64)
    for i in np.argsort(tri_zmax):
        cv2.fillConvexPoly(reach, tri_px[i], float(tri_zmax[i] - z0))

    def area(d):  # pixels whose top reach is above cut depth d
        return int((reach > d).sum())

    a_full = area(0.0)
    if a_full == 0:
        return 0.0
    n_bands = int(cap_mm / BAND_MM)
    areas = [area(i * BAND_MM) for i in range(n_bands + 1)]
    a_min = min(areas)
    if a_full - a_min < MIN_REDUCTION * a_full:
        return 0.0  # base not worth excluding
    thresh = a_min + FLAT_EPS * a_full
    for i in range(n_bands + 1):
        if areas[i] <= thresh:  # first depth on the plateau = the knee
            return float(i * BAND_MM)
    return 0.0


def extract_footprints(mesh, res_mm: float, cut_cap_mm: float):
    """Return (full_mask, body_mask, origin_mm, cut_mm, stats).

    full_mask is the full vertical shadow (identical to extract_footprint).
    body_mask, on the same canvas/origin, is the shadow of triangles reaching
    above the auto-detected base cut (max Z > z0 + cut_mm); when cut_mm == 0 it
    is a copy of full_mask. stats mirrors extract_footprint's, plus 'cut_mm'.
    """
    t0 = time.perf_counter()
    tris = mesh.triangles
    finite = np.isfinite(tris).all(axis=(1, 2))
    n_bad = int((~finite).sum())
    if n_bad:
        tris = tris[finite]
    if len(tris) == 0:
        raise ValueError("mesh has no finite triangles")
    xy = tris[:, :, :2]
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / res_mm).astype(np.int32)
    full_mask = _raster(tri_px, shape)

    cut_mm = detect_base_cut(tris, DETECT_RES_MM, cut_cap_mm)
    if cut_mm <= 0:
        body_mask = full_mask.copy()
    else:
        z0 = float(tris[:, :, 2].min())
        keep = tris[:, :, 2].max(axis=1) > z0 + cut_mm
        body_mask = _raster(tri_px[keep], shape)
        if not body_mask.any():  # nothing survives the cut -> don't cut
            body_mask = full_mask.copy()
            cut_mm = 0.0
    t_raster = time.perf_counter() - t0

    stats = {
        "triangles": len(mesh.triangles),
        "dropped_nonfinite": n_bad,
        "z_height_mm": float(tris[:, :, 2].max() - tris[:, :, 2].min()),
        "footprint_mm": tuple((flat.max(axis=0) - flat.min(axis=0)).round(1)),
        "mask_px": full_mask.shape,
        "coverage": float(full_mask.mean()),
        "raster_s": t_raster,
        "cut_mm": cut_mm,
    }
    return full_mask, body_mask, (float(origin[0]), float(origin[1])), cut_mm, stats


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

"""Shape-aware rotation candidates: lay convex-hull edges parallel to the plate
axes (spec 2026-08-06, ADR-012). Resolution-independent — computed once per
piece from its mask hull and reused at every raster resolution."""

import math

import cv2
import numpy as np

_DEDUP_DEG = 2.0  # merge angles within this many degrees
_CIRCLE_RATIO = 0.90  # hull_area / min-enclosing-circle area above this => round


def _analytic_aabb_area(hull_pts: np.ndarray, angle_deg: float) -> float:
    """Axis-aligned bbox area of the hull points rotated by angle_deg. Analytic
    (no rasterization) so it is free of warpAffine interpolation growth."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = hull_pts @ np.array([[c, -s], [s, c]])
    return float((np.ptp(rot[:, 0])) * (np.ptp(rot[:, 1])))


def angle_candidates(
    mask: np.ndarray, cap: int = 12, min_edge_frac: float = 0.1, safety_grid: int = 0
) -> list[float]:
    """Rotation angles (deg, in [0,180)) that lay long hull edges parallel to a
    plate axis. 0.0 is always included; circle-like hulls collapse to [0.0]
    (plus any safety_grid angles). Sorted most-compact-first, capped at cap."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    pts = np.argwhere(mask > 0)[:, ::-1].astype(np.int32)  # (x=col, y=row)
    if len(pts) == 0:
        return [0.0]
    hull = cv2.convexHull(pts)[:, 0, :].astype(np.float64)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    perimeter = float(lengths.sum())

    hull_area = cv2.contourArea(hull.astype(np.int32))
    (_, _), radius = cv2.minEnclosingCircle(pts)
    circle_area = math.pi * radius * radius
    is_circle = circle_area > 0 and hull_area / circle_area > _CIRCLE_RATIO

    angles = {0.0}
    if not is_circle:
        for (dx, dy), length in zip(edges, lengths, strict=True):
            if length < min_edge_frac * perimeter:
                continue
            # theta that lays edge (dx, dy) parallel to an axis under
            # rotate_mask's convention (x'=cos*x+sin*y, y'=-sin*x+cos*y):
            # -dx*sin+dy*cos=0 => theta = atan2(dy, dx). (No negation -- the
            # mirror sign left generic edges un-aligned; see bugs.md 2026-08-07.)
            base = math.degrees(math.atan2(dy, dx)) % 180
            angles.add(round(base, 6))
            angles.add(round((base + 90) % 180, 6))
    if safety_grid > 0:
        for i in range(safety_grid):
            angles.add(round((i * 360.0 / safety_grid) % 180, 6))

    deduped: list[float] = []
    for a in sorted(angles):
        if not deduped or abs(a - deduped[-1]) > _DEDUP_DEG:
            deduped.append(a)
    deduped.sort(key=lambda a: (_analytic_aabb_area(hull, a), a))
    result = deduped[:cap]
    if 0.0 not in result:
        # 0.0 (the lossless un-rotated path) must always remain available; drop
        # the least-compact candidate for it, then re-sort so 0.0 sits at its
        # true compactness rank (and wins score ties by the ascending-value key,
        # as the lossless orientation should) rather than always landing last.
        result = sorted([*deduped[: cap - 1], 0.0], key=lambda a: (_analytic_aabb_area(hull, a), a))
    return result

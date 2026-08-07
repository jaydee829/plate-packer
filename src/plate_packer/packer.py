"""Greedy plate packing: FFT-correlation legality, pluggable placement, spillover."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import fftconvolve

# Correlation of two 0/1 masks is >= 1 wherever they truly overlap; FFT noise
# is orders of magnitude below 0.5.
_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True)
class Placement:
    piece: int  # index into the input piece list
    plate: int  # 0-based plate number
    row: int  # anchor (top-left) of the rotated mask on the plate
    col: int
    angle: float  # degrees CCW
    contact: float = 0.0  # chosen anchor's contact score (0.0 under bottom_left)


def pack(
    pieces,
    plate_shape,
    rotations=1,
    plate_mask=None,
    choose=None,
    prerotated=None,
    order=None,
    validate=True,
    edge_weight=1.0,
):
    """Greedy-pack piece masks onto plates; spill to a new plate when full.

    plate_mask pre-encodes unusable plate regions as occupied pixels.
    prerotated (list of {angle: mask}) skips per-call rotation; order
    overrides the largest-area-first insertion order; validate=False skips
    the every-piece-fits-an-empty-plate check (improve() validates once).
    edge_weight scales the plate-border contact score (see contact_map).
    Raises ValueError if a piece cannot fit an empty plate at any rotation.
    """
    choose = choose or contact_first
    empty = plate_mask.copy() if plate_mask is not None else np.zeros(plate_shape, np.uint8)
    if prerotated is None:
        angles = [i * 360.0 / rotations for i in range(rotations)]
        prerotated = [{a: rotate_mask(p, a)[0] for a in angles} for p in pieces]
    rings = (
        [{a: contact_ring(m) for a, m in variants.items()} for variants in prerotated]
        if getattr(choose, "uses_contact", False)
        else None
    )

    if validate:
        for i, variants in enumerate(prerotated):
            if not any(_fits(empty, m) for m in variants.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")

    if order is None:
        order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True)
    plates: list[np.ndarray] = []
    placements: list[Placement] = []
    for i in order:
        target = plate_idx = None
        piece_rings = rings[i] if rings is not None else None
        for idx, occupancy in enumerate(plates):
            target = _best_spot(occupancy, prerotated[i], piece_rings, choose, edge_weight)
            if target:
                plate_idx = idx
                break
        if target is None:
            plates.append(empty.copy())
            plate_idx = len(plates) - 1
            target = _best_spot(plates[plate_idx], prerotated[i], piece_rings, choose, edge_weight)
        if target is None:
            # Reachable only with validate=False (validate=True raises up front):
            # honor the documented contract instead of failing on the unpack.
            raise ValueError(f"piece {i} does not fit an empty plate at any rotation")
        (row, col), angle, score = target
        mask = prerotated[i][angle]
        plates[plate_idx][row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
        placements.append(Placement(i, plate_idx, row, col, angle, score))
    return sorted(placements, key=lambda p: p.piece)


def _elongation(mask: np.ndarray) -> float:
    """long_side / short_side of the min-area bounding box (>= 1)."""
    pts = np.argwhere(mask > 0)[:, ::-1].astype(np.int32)
    (_, (w, h), _) = cv2.minAreaRect(pts)
    lo, hi = min(w, h), max(w, h)
    return hi / lo if lo > 0 else 1.0


def seed_order(pieces, ordering: str = "difficulty") -> list[int]:
    """Greedy insertion seed order. 'difficulty' = area * elongation descending
    (a long thin piece needs a long channel that only exists early); 'area' =
    legacy largest-area-first."""
    if ordering == "area":
        key = lambda i: float(pieces[i].sum())  # noqa: E731
    else:
        key = lambda i: float(pieces[i].sum()) * _elongation(pieces[i])  # noqa: E731
    return sorted(range(len(pieces)), key=key, reverse=True)


def _fits(plate, piece):
    return (
        piece.shape[0] <= plate.shape[0]
        and piece.shape[1] <= plate.shape[1]
        and legal_placement_map(plate, piece).any()
    )


def _best_spot(occupancy, variants, rings, choose, edge_weight=1.0):
    """Best (anchor, angle, contact) across rotations: highest contact, then
    lowest row/col; ties beyond that keep the earliest angle."""
    best = None  # (sort_key, anchor, angle, score)
    for angle, mask in variants.items():
        if mask.shape[0] > occupancy.shape[0] or mask.shape[1] > occupancy.shape[1]:
            continue
        legal = legal_placement_map(occupancy, mask)
        contact = (
            contact_map(occupancy, rings[angle], edge_weight)
            if rings is not None
            else np.zeros(legal.shape)
        )
        anchor = choose(legal, contact)
        if anchor is None:
            continue
        score = float(contact[anchor])
        key = (-score, anchor[0], anchor[1])
        if best is None or key < best[0]:
            best = (key, anchor, angle, score)
    if best is None:
        return None
    return best[1], best[2], best[3]


def legal_placement_map(plate: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Boolean map of anchor positions (top-left) where piece fits on plate.

    Shape is (plate_h - piece_h + 1, plate_w - piece_w + 1): only placements
    fully inside the plate are considered.
    """
    overlap = fftconvolve(plate.astype(np.float32), piece[::-1, ::-1].astype(np.float32), "valid")
    return overlap < _OVERLAP_THRESHOLD


def contact_ring(mask: np.ndarray) -> np.ndarray:
    """1-px halo around a tight-cropped mask; shape (h+2, w+2)."""
    padded = np.pad(mask, 1).astype(np.uint8)
    return cv2.dilate(padded, np.ones((3, 3), np.uint8)) - padded


def contact_map(plate: np.ndarray, ring: np.ndarray, edge_weight: float = 1.0) -> np.ndarray:
    """Contact score at every anchor: halo pixels touching occupancy or the
    plate border. The border frame is weighted by edge_weight (occupancy stays
    weight 1). Same anchor coordinates/shape as legal_placement_map. Rounding to
    2 decimals collapses FFT float noise (measured <3e-4 on full-size plates, so
    2-decimal rounding is ~18x-safe) for stable ties, while preserving fractional
    edge_weight signal -- plain np.rint would quantize e.g. edge_weight=0.1 to 0."""
    attraction = np.pad(plate.astype(np.float32), 1, constant_values=edge_weight)
    raw = fftconvolve(attraction, ring[::-1, ::-1].astype(np.float32), "valid")
    return np.round(raw, 2)


def contact_first(legal: np.ndarray, contact: np.ndarray) -> tuple[int, int] | None:
    """Legal anchor with the highest contact score; ties resolve bottom-left
    (argmax first-occurrence in row-major order IS lowest row, then col)."""
    if not legal.any():
        return None
    r, c = np.unravel_index(int(np.argmax(np.where(legal, contact, -1.0))), legal.shape)
    return int(r), int(c)


contact_first.uses_contact = True


def _crop_to_content_bbox(binary: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop a binary mask to its content bounding box.

    Returns (cropped, r0, c0) where r0, c0 are the offsets of the crop.
    """
    if not binary.any():
        raise ValueError("cannot crop an empty mask")
    rows, cols = binary.any(axis=1), binary.any(axis=0)
    r0, c0 = int(np.argmax(rows)), int(np.argmax(cols))
    cropped = binary[
        r0 : len(rows) - np.argmax(rows[::-1]),
        c0 : len(cols) - np.argmax(cols[::-1]),
    ]
    return cropped, r0, c0


def rotate_mask(mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a binary mask about its center, expanding the canvas and cropping
    to the content bbox. Returns (rotated, affine): affine is the 2x3 map from
    input px (x=col, y=row) to output px, crop included.

    The linear part is [[cos, sin], [-sin, cos]] in (col,row) coords on BOTH
    paths (rot90 and warpAffine agree). Export derives world rotation from this
    affine; the nominal angle's sign convention is never trusted downstream."""
    angle_deg %= 360
    h, w = mask.shape
    if angle_deg % 90 == 0:
        k = int(angle_deg // 90)
        # Exact and lossless at right angles; warpAffine clips edge pixels.
        rotated = np.ascontiguousarray(np.rot90(mask, k))
        # Crop to content bbox even for right angles to ensure tight output
        cropped, r0, c0 = _crop_to_content_bbox(rotated)
        # Compute affine for this right angle with crop adjustment
        affines_base = {
            0: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            1: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0]]),
            2: np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0]]),
            3: np.array([[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0]]),
        }
        aff = affines_base[k].copy().astype(np.float64)
        aff[0, 2] -= c0
        aff[1, 2] -= r0
        return cropped, aff
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(np.ceil(w * cos + h * sin)) + 2
    new_h = int(np.ceil(w * sin + h * cos)) + 2
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    # Linear interpolation + any-touched-pixel threshold: rotation may only
    # GROW the footprint. Losing boundary pixels would create false free
    # space in the collision map.
    rotated = cv2.warpAffine(mask.astype(np.float32), m, (new_w, new_h), flags=cv2.INTER_LINEAR)
    binary = (rotated > 0).astype(np.uint8)
    cropped, r0, c0 = _crop_to_content_bbox(binary)
    m[0, 2] -= c0
    m[1, 2] -= r0
    return cropped, m


def bottom_left(legal: np.ndarray, contact: np.ndarray | None = None) -> tuple[int, int] | None:
    """Pick the legal anchor with the lowest row, then lowest column.

    The contact argument is ignored; it exists to match the contact_first signature.
    """
    flat = np.flatnonzero(legal)
    if len(flat) == 0:
        return None
    r, c = np.unravel_index(flat[0], legal.shape)
    return int(r), int(c)


CHOOSERS = {"contact": contact_first, "bottom_left": bottom_left}

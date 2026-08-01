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


def pack(pieces, plate_shape, rotations=1, plate_mask=None, choose=None):
    """Greedy-pack piece masks onto plates; spill to a new plate when full.

    plate_mask pre-encodes unusable plate regions as occupied pixels.
    Raises ValueError if a piece cannot fit an empty plate at any rotation.
    """
    choose = choose or bottom_left
    empty = plate_mask.copy() if plate_mask is not None else np.zeros(plate_shape, np.uint8)
    angles = [i * 360.0 / rotations for i in range(rotations)]
    # Pre-rotate every piece once; rotation choice is per-placement.
    rotated = [{a: rotate_mask(p, a)[0] for a in angles} for p in pieces]

    # Validation before any packing: every piece must fit an empty plate.
    for i, variants in enumerate(rotated):
        if not any(_fits(empty, m) for m in variants.values()):
            raise ValueError(f"piece {i} does not fit an empty plate at any rotation")

    order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True)
    plates: list[np.ndarray] = []
    placements: list[Placement] = []
    for i in order:
        target = plate_idx = None
        for idx, occupancy in enumerate(plates):
            target = _best_spot(occupancy, rotated[i], choose)
            if target:
                plate_idx = idx
                break
        if target is None:
            plates.append(empty.copy())
            plate_idx = len(plates) - 1
            target = _best_spot(plates[plate_idx], rotated[i], choose)
        (row, col), angle = target
        mask = rotated[i][angle]
        plates[plate_idx][row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
        placements.append(Placement(i, plate_idx, row, col, angle))
    return sorted(placements, key=lambda p: p.piece)


def _fits(plate, piece):
    return (
        piece.shape[0] <= plate.shape[0]
        and piece.shape[1] <= plate.shape[1]
        and legal_placement_map(plate, piece).any()
    )


def _best_spot(occupancy, variants, choose):
    """Best (anchor, angle) across rotations by the placement heuristic."""
    best = None
    for angle, mask in variants.items():
        if mask.shape[0] > occupancy.shape[0] or mask.shape[1] > occupancy.shape[1]:
            continue
        anchor = choose(legal_placement_map(occupancy, mask))
        if anchor and (best is None or anchor < best[0]):
            best = (anchor, angle)
    return best


def legal_placement_map(plate: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Boolean map of anchor positions (top-left) where piece fits on plate.

    Shape is (plate_h - piece_h + 1, plate_w - piece_w + 1): only placements
    fully inside the plate are considered.
    """
    overlap = fftconvolve(plate.astype(np.float32), piece[::-1, ::-1].astype(np.float32), "valid")
    return overlap < _OVERLAP_THRESHOLD


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
        binary = rotated.astype(np.uint8)
        rows, cols = binary.any(axis=1), binary.any(axis=0)
        if rows.any() and cols.any():
            r0, c0 = int(np.argmax(rows)), int(np.argmax(cols))
            cropped = binary[
                r0 : len(rows) - np.argmax(rows[::-1]),
                c0 : len(cols) - np.argmax(cols[::-1]),
            ]
        else:
            r0, c0 = 0, 0
            cropped = binary
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
    rows, cols = binary.any(axis=1), binary.any(axis=0)
    r0, c0 = int(np.argmax(rows)), int(np.argmax(cols))
    cropped = binary[
        r0 : len(rows) - np.argmax(rows[::-1]),
        c0 : len(cols) - np.argmax(cols[::-1]),
    ]
    m[0, 2] -= c0
    m[1, 2] -= r0
    return cropped, m


def bottom_left(legal: np.ndarray) -> tuple[int, int] | None:
    """Pick the legal anchor with the lowest row, then lowest column."""
    flat = np.flatnonzero(legal)
    if len(flat) == 0:
        return None
    r, c = np.unravel_index(flat[0], legal.shape)
    return int(r), int(c)

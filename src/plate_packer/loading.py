"""Cache doc -> pack-ready mask: conservative downsample + spacing dilation.

The single place where packer/printer config (working resolution, minimum
spacing) is applied to intrinsic cached footprints (ADR-009).
"""

import math

import cv2
import numpy as np

from plate_packer.footprint_io import RES_RATIO_TOL, FootprintDoc


def conservative_downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Block max: any occupied source pixel marks the coarse cell."""
    if factor == 1:
        return mask.copy()
    h, w = mask.shape
    ph, pw = -h % factor, -w % factor
    padded = np.pad(mask, ((0, ph), (0, pw)))
    return (
        padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
        .max(axis=(1, 3))
        .astype(np.uint8)
    )


def dilate(mask: np.ndarray, r_px: int) -> np.ndarray:
    """Dilate with an elliptical kernel; canvas pre-padded so the margin
    is never clipped at the borders."""
    padded = np.pad(mask, r_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    return cv2.dilate(padded, kernel)


def dilation_radius_px(spacing_mm: float, working_res_mm: float) -> int:
    """Per-piece dilation radius. Pieces pack dilated-vs-dilated, so each side
    contributes half of spacing_mm, the TRUE minimum inter-piece gap (ADR-010)."""
    if spacing_mm <= 0:
        return 0
    return math.ceil(spacing_mm / 2 / working_res_mm)


def prepare_mask(
    doc: FootprintDoc, spacing_mm: float, working_res_mm: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Returns (mask, origin_mm): origin_mm is the XY of the prepared mask's
    pixel (0,0). Dilation pads all sides, shifting it by -r_px*res per axis;
    conservative downsample keeps blocks anchored at pixel 0 (no shift)."""
    ratio = working_res_mm / doc.res_mm_per_px
    if abs(ratio - round(ratio)) > RES_RATIO_TOL or ratio < 1:
        raise ValueError(
            f"working res {working_res_mm} must be an integer multiple "
            f"of canonical res {doc.res_mm_per_px}"
        )
    mask = conservative_downsample(doc.masks[0], round(ratio))
    origin = (float(doc.origin_mm[0]), float(doc.origin_mm[1]))
    r_px = dilation_radius_px(spacing_mm, working_res_mm)
    if r_px:
        mask = dilate(mask, r_px)
        origin = (origin[0] - r_px * working_res_mm, origin[1] - r_px * working_res_mm)
    return mask, origin

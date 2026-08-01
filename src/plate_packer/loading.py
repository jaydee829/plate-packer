"""Cache doc -> pack-ready mask: conservative downsample + spacing dilation.

The single place where packer/printer config (working resolution, minimum
spacing) is applied to intrinsic cached footprints (ADR-009).
"""

import cv2
import numpy as np

from plate_packer.footprint_io import FootprintDoc

_RATIO_TOL = 1e-6


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


def prepare_mask(doc: FootprintDoc, spacing_mm: float, working_res_mm: float) -> np.ndarray:
    ratio = working_res_mm / doc.res_mm_per_px
    if abs(ratio - round(ratio)) > _RATIO_TOL or ratio < 1:
        raise ValueError(
            f"working res {working_res_mm} must be an integer multiple "
            f"of canonical res {doc.res_mm_per_px}"
        )
    mask = conservative_downsample(doc.masks[0], round(ratio))
    if spacing_mm > 0:
        mask = dilate(mask, max(1, round(spacing_mm / working_res_mm)))
    return mask

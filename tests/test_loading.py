"""Load-step tests: conservative downsample + spacing dilation (ADR-009)."""

import numpy as np
import pytest

from plate_packer.footprint_io import CANONICAL_RES_MM, FootprintDoc
from plate_packer.loading import conservative_downsample, dilate, prepare_mask


def make_doc(mask):
    return FootprintDoc(
        sha="a" * 64,
        res_mm_per_px=CANONICAL_RES_MM,
        origin_mm=(0.0, 0.0),
        z_height_mm=5.0,
        triangles=2,
        dropped_nonfinite=0,
        masks=[mask.astype(np.uint8)],
    )


def test_downsample_lone_pixel_survives():
    """A single occupied pixel must mark its coarse cell (naive averaging drops it)."""
    mask = np.zeros((8, 8), np.uint8)
    mask[3, 5] = 1
    out = conservative_downsample(mask, 2)
    assert out.shape == (4, 4)
    assert out[1, 2] == 1
    assert out.sum() == 1


def test_downsample_solid_stays_solid():
    out = conservative_downsample(np.ones((10, 10), np.uint8), 2)
    assert out.shape == (5, 5)
    assert out.all()


def test_downsample_pads_ragged_edges():
    """Sizes not divisible by factor pad with zeros; edge content survives."""
    mask = np.zeros((5, 7), np.uint8)
    mask[4, 6] = 1
    out = conservative_downsample(mask, 2)
    assert out.shape == (3, 4)
    assert out[2, 3] == 1


@pytest.mark.parametrize(
    ("mask_shape", "r"),
    [
        pytest.param((10, 20), 5, id="rect-r5"),
        pytest.param((1, 1), 3, id="single-pixel-r3"),
    ],
)
def test_dilate_pads_canvas_so_margin_never_clips(mask_shape, r):
    mask = np.ones(mask_shape, np.uint8)
    out = dilate(mask, r)
    assert out.shape == (mask_shape[0] + 2 * r, mask_shape[1] + 2 * r)
    # cardinal extremes of the elliptical kernel reach the canvas edge
    assert out[0, r + mask_shape[1] // 2] == 1
    assert out[out.shape[0] // 2, 0] == 1


def test_prepare_mask_downsamples_then_dilates():
    """0.05 doc at working res 0.1 with 0.5mm spacing: 2x downsample + 5px pad."""
    doc = make_doc(np.ones((40, 20), np.uint8))  # 2.0 x 1.0 mm at 0.05
    out = prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.1)
    assert out.shape == (20 + 10, 10 + 10)  # halved, then +5px each side
    assert out[15, 10] == 1


def test_prepare_mask_zero_spacing_skips_dilation():
    doc = make_doc(np.ones((40, 20), np.uint8))
    out = prepare_mask(doc, spacing_mm=0.0, working_res_mm=0.1)
    assert out.shape == (20, 10)


def test_prepare_mask_non_integer_ratio_raises():
    doc = make_doc(np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.075)

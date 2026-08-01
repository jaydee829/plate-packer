"""Load-step tests: conservative downsample + spacing dilation (ADR-009)."""

import numpy as np
import pytest

from plate_packer.footprint_io import CANONICAL_RES_MM, FootprintDoc
from plate_packer.loading import (
    conservative_downsample,
    dilate,
    dilation_radius_px,
    prepare_mask,
)


def make_doc(mask, origin_mm=(0.0, 0.0), res=CANONICAL_RES_MM):
    return FootprintDoc(
        sha="a" * 64,
        res_mm_per_px=res,
        origin_mm=origin_mm,
        z_height_mm=5.0,
        triangles=2,
        dropped_nonfinite=0,
        masks=[mask.astype(np.uint8)],
    )


@pytest.mark.parametrize(
    ("spacing", "res", "expected_r"),
    [(2.0, 0.1, 10), (1.0, 0.1, 5), (0.5, 0.1, 3), (0.0, 0.1, 0), (2.0, 0.05, 20), (-1.0, 0.1, 0)],
)
def test_dilation_radius_px(spacing, res, expected_r):
    assert dilation_radius_px(spacing, res) == expected_r


@pytest.mark.parametrize(
    ("spacing", "res", "doc_origin", "expected_origin"),
    [
        (2.0, 0.1, (1.0, 2.0), (0.0, 1.0)),  # r=10 px -> -1.0 mm both axes
        (1.0, 0.05, (0.0, 0.0), (-0.5, -0.5)),  # r=10 px at 0.05 -> -0.5 mm
        (0.0, 0.1, (3.5, -2.0), (3.5, -2.0)),  # no dilation -> unchanged
    ],
)
def test_prepare_mask_origin(spacing, res, doc_origin, expected_origin):
    doc = make_doc(np.ones((20, 20), np.uint8), origin_mm=doc_origin)
    _, origin = prepare_mask(doc, spacing, res)
    assert origin == pytest.approx(expected_origin)


def test_prepare_mask_downsample_alone_keeps_origin():
    doc = make_doc(np.ones((20, 20), np.uint8), origin_mm=(1.5, 2.5))
    _, origin = prepare_mask(doc, 0.0, 0.1)  # 2x downsample, no dilation
    assert origin == pytest.approx((1.5, 2.5))


def test_prepare_mask_dilated_bbox_growth():
    # content bbox must grow by exactly r_px per side
    inner = np.zeros((40, 40), np.uint8)
    inner[10:30, 10:30] = 1
    doc = make_doc(inner)
    mask, _ = prepare_mask(doc, 1.0, 0.05)  # r = ceil(0.5/0.05) = 10 px
    rows = np.flatnonzero(mask.any(axis=1))
    assert rows[-1] - rows[0] + 1 == 20 + 2 * 10


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
    """0.05 doc at working res 0.1 with 0.5mm spacing: 2x downsample + 3px pad."""
    doc = make_doc(np.ones((40, 20), np.uint8))  # 2.0 x 1.0 mm at 0.05
    out, _ = prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.1)
    assert out.shape == (20 + 6, 10 + 6)  # halved, then +3px each side
    assert out[9, 6] == 1


def test_prepare_mask_zero_spacing_skips_dilation():
    doc = make_doc(np.ones((40, 20), np.uint8))
    out, _ = prepare_mask(doc, spacing_mm=0.0, working_res_mm=0.1)
    assert out.shape == (20, 10)


def test_prepare_mask_non_integer_ratio_raises():
    doc = make_doc(np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.075)


def test_prepare_mask_never_aliases_the_cached_doc():
    """Mutating the returned mask must not corrupt the cached doc (no-op path)."""
    mask = np.ones((4, 4), np.uint8)
    doc = make_doc(mask)
    out, _ = prepare_mask(doc, spacing_mm=0.0, working_res_mm=CANONICAL_RES_MM)
    out[0, 0] = 0
    assert doc.masks[0][0, 0] == 1

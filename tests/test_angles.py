"""Tests for shape-aware angle candidates (spec 2026-08-06, ADR-012)."""

import math

import cv2
import numpy as np
import pytest

from plate_packer.angles import _analytic_aabb_area, angle_candidates
from plate_packer.packer import rotate_mask


def solid(h, w):
    return np.ones((h, w), np.uint8)


def _circle(r):
    m = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(m, (r, r), r, 1, -1)
    return m


def _l_shape():
    m = np.zeros((10, 10), np.uint8)
    m[:, :4] = 1
    m[6:, :] = 1
    return m


def _tilted_rect():
    base = np.ones((3, 12), np.uint8)
    m = cv2.getRotationMatrix2D((6, 1.5), 30, 1.0)
    m[0, 2] += 30 / 2 - 6
    m[1, 2] += 30 / 2 - 1.5
    return (cv2.warpAffine(base.astype(np.float32), m, (30, 30)) > 0).astype(np.uint8)


@pytest.mark.parametrize(
    "mask, expected",
    [
        pytest.param(solid(4, 4), [0.0, 90.0], id="square-axis-aligned"),
        pytest.param(solid(3, 12), [0.0, 90.0], id="long-rect-long-edge-on-axis"),
        pytest.param(_circle(10), [0.0], id="circle-single-angle"),
    ],
)
def test_angle_candidates_known_shapes(mask, expected):
    assert angle_candidates(mask) == expected


def test_angle_candidates_l_shape_finds_axes_and_diagonal():
    # An L's convex hull has the two axis edges plus a diagonal across the
    # concavity, giving axis angles and their 45deg diagonal partners.
    assert set(angle_candidates(_l_shape())) == {0.0, 45.0, 90.0, 135.0}


def test_angle_candidates_always_includes_zero():
    assert 0.0 in angle_candidates(_l_shape())
    assert 0.0 in angle_candidates(solid(3, 12))


def test_angle_candidates_cap_limits_count():
    result = angle_candidates(_l_shape(), cap=2)
    assert len(result) == 2
    assert result == [0.0, 90.0]  # the two most compact orientations


def test_angle_candidates_min_edge_frac_filters_short_edges():
    # min_edge_frac=0.5 exceeds every edge fraction of a 3x12 rectangle
    # (longest edge is 12/30 = 0.4 of the perimeter), so only the always-on
    # 0.0 survives.
    assert angle_candidates(solid(3, 12), min_edge_frac=0.5) == [0.0]


def test_angle_candidates_safety_grid_unions_uniform_angles():
    # safety_grid=8 -> {0,45,90,135} mod 180; union with the square's {0,90}.
    assert set(angle_candidates(solid(4, 4), safety_grid=8)) == {0.0, 45.0, 90.0, 135.0}


def test_angle_candidates_safety_grid_applies_to_circle():
    # Circle collapses shape-derived angles to {0}, but safety_grid still unions.
    assert set(angle_candidates(_circle(10), safety_grid=4)) == {0.0, 90.0}


def test_angle_candidates_sorted_by_compactness_deskews_tilt():
    # A rectangle rasterized at +30deg: the top candidate must de-skew it to a
    # tighter analytic bbox than leaving it at 0deg.
    tilted = _tilted_rect()

    def analytic_aabb(mask, angle):
        pts = cv2.convexHull(np.argwhere(mask > 0)[:, ::-1].astype(np.int32))[:, 0, :].astype(float)
        rad = np.radians(angle)
        rot = pts @ np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        return (np.ptp(rot[:, 0])) * (np.ptp(rot[:, 1]))

    cands = angle_candidates(tilted)
    assert analytic_aabb(tilted, cands[0]) < analytic_aabb(tilted, 0.0)


def test_analytic_aabb_area_matches_rotate_mask_convention():
    # The compactness key must rotate hull points the SAME way rotate_mask does
    # (x,y) -> (x*cos + y*sin, -x*sin + y*cos), so ranking reflects the real
    # placed bbox. Verified on an asymmetric (scalene) point set where the
    # rotation sign matters (regression: PR #6 Important #1).
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 7.0]])
    a = 30.0
    rad = math.radians(a)
    c, s = math.cos(rad), math.sin(rad)
    xs = pts[:, 0] * c + pts[:, 1] * s
    ys = -pts[:, 0] * s + pts[:, 1] * c
    expected = float(np.ptp(xs) * np.ptp(ys))
    assert _analytic_aabb_area(pts, a) == pytest.approx(expected)


@pytest.mark.parametrize("cap", [1, 2, 3], ids=["cap-1", "cap-2", "cap-3"])
def test_angle_candidates_keeps_zero_under_small_cap(cap):
    # 0.0 (the lossless un-rotated path) must survive cap truncation even when
    # it is not among the most compact orientations of a tilted piece.
    result = angle_candidates(_tilted_rect(), cap=cap)
    assert 0.0 in result
    assert len(result) == cap


def test_angle_candidates_cap_one_tilted_is_zero():
    assert angle_candidates(_tilted_rect(), cap=1) == [0.0]


def test_angle_candidates_deskews_generic_tilted_edge():
    # A long rectangle tilted to a generic (non-axis, non-45deg) orientation: a
    # shape-aware candidate must actually axis-align it -- rotate_mask by the
    # best candidate yields a much tighter bbox than leaving it at 0deg.
    # Regression: the per-edge angle formula had an inverted sign that left
    # generic hull edges un-aligned (PR #6 review, bugs.md 2026-08-07).
    base = np.zeros((7, 28), np.uint8)
    base[1:6, 2:26] = 1
    tilted, _ = rotate_mask(base, 25.0)

    def bbox_area(a):
        m, _ = rotate_mask(tilted, a)
        rr, cc = np.nonzero(m)
        return (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)

    cands = angle_candidates(tilted)
    assert min(bbox_area(a) for a in cands) < 0.75 * bbox_area(0.0)


@pytest.mark.parametrize("cap", [0, -1], ids=["cap-0", "cap-neg"])
def test_angle_candidates_rejects_cap_below_one(cap):
    with pytest.raises(ValueError, match="cap"):
        angle_candidates(solid(4, 4), cap=cap)

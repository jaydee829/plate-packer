"""Tests for shape-aware angle candidates (spec 2026-08-06, ADR-012)."""

import cv2
import numpy as np
import pytest

from plate_packer.angles import angle_candidates


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
    base = solid(3, 12)
    m = cv2.getRotationMatrix2D((6, 1.5), 30, 1.0)
    m[0, 2] += 30 / 2 - 6
    m[1, 2] += 30 / 2 - 1.5
    tilted = (cv2.warpAffine(base.astype(np.float32), m, (30, 30)) > 0).astype(np.uint8)

    def analytic_aabb(mask, angle):
        pts = cv2.convexHull(np.argwhere(mask > 0)[:, ::-1].astype(np.int32))[:, 0, :].astype(float)
        rad = np.radians(angle)
        rot = pts @ np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        return (np.ptp(rot[:, 0])) * (np.ptp(rot[:, 1]))

    cands = angle_candidates(tilted)
    assert analytic_aabb(tilted, cands[0]) < analytic_aabb(tilted, 0.0)

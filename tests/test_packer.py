"""Packer tests: FFT legality, placement heuristic, greedy spillover."""

import numpy as np
import pytest

from plate_packer.packer import (
    bottom_left,
    legal_placement_map,
    pack,
    rotate_mask,
)


def occupancy_from(placements, pieces, plate_shape):
    """Rebuild per-plate occupancy from placements; asserts in-bounds."""
    n_plates = max(p.plate for p in placements) + 1
    plates = np.zeros((n_plates, *plate_shape), np.uint8)
    for p in placements:
        m, _ = rotate_mask(pieces[p.piece], p.angle)
        plates[p.plate, p.row : p.row + m.shape[0], p.col : p.col + m.shape[1]] += m
    return plates


def solid(h, w):
    return np.ones((h, w), dtype=np.uint8)


def test_empty_plate_every_anchor_is_legal():
    legal = legal_placement_map(np.zeros((10, 10), np.uint8), solid(3, 3))
    assert legal.shape == (8, 8)
    assert legal.all()


def test_full_plate_has_no_legal_anchor():
    legal = legal_placement_map(solid(10, 10), solid(3, 3))
    assert not legal.any()


@pytest.mark.parametrize(
    ("legal_positions", "expected"),
    [
        pytest.param([(0, 0), (0, 5), (3, 2)], (0, 0), id="origin-wins"),
        pytest.param([(2, 7), (2, 3), (5, 0)], (2, 3), id="lowest-row-then-lowest-col"),
        pytest.param([(4, 4)], (4, 4), id="single-option"),
        pytest.param([], None, id="no-legal-placement"),
    ],
)
def test_bottom_left_picks_lowest_row_then_col(legal_positions, expected):
    legal = np.zeros((8, 8), bool)
    for r, c in legal_positions:
        legal[r, c] = True
    assert bottom_left(legal) == expected


@pytest.mark.parametrize(
    ("angle", "expected_shape"),
    [
        pytest.param(0, (2, 5), id="0deg-unchanged"),
        pytest.param(90, (5, 2), id="90deg-swaps-axes"),
        pytest.param(180, (2, 5), id="180deg-keeps-shape"),
    ],
)
def test_rotate_mask_right_angles(angle, expected_shape):
    rotated, _ = rotate_mask(solid(2, 5), angle)
    assert rotated.shape == expected_shape
    assert rotated.all()  # solid rectangle stays solid at right angles


def test_rotate_mask_45deg_preserves_area_and_stays_binary():
    mask = solid(10, 10)
    rotated, _ = rotate_mask(mask, 45)
    assert set(np.unique(rotated)) <= {0, 1}
    # Conservative binarization: rotation may grow the footprint (collision
    # safety) but must never lose pixels; growth is bounded by the perimeter.
    assert 100 <= int(rotated.sum()) <= 160
    # canvas must expand to hold the diagonal, and content is cropped tight
    assert max(rotated.shape) >= 13
    assert rotated.any(axis=1).all() and rotated.any(axis=0).all()


def test_pack_places_two_pieces_on_one_plate_without_overlap():
    pieces = [solid(3, 3), solid(3, 3)]
    placements = pack(pieces, (10, 10))
    assert [p.plate for p in placements] == [0, 0]
    occ = occupancy_from(placements, pieces, (10, 10))
    assert occ.max() == 1  # no pixel claimed twice


def test_pack_spills_to_second_plate_when_first_is_full():
    pieces = [solid(6, 6), solid(6, 6)]
    placements = pack(pieces, (10, 10))
    assert sorted(p.plate for p in placements) == [0, 1]


def test_pack_sorts_largest_area_first():
    pieces = [solid(2, 2), solid(5, 5)]  # small listed first
    placements = pack(pieces, (10, 10))
    big = next(p for p in placements if p.piece == 1)
    assert (big.row, big.col) == (0, 0)  # largest got first pick


def test_pack_rotation_enables_fit():
    # 8x3 piece cannot fit a 4x10 plate upright, but fits rotated 90 degrees
    placements = pack([solid(8, 3)], (4, 10), rotations=4)
    assert placements[0].angle % 180 == 90


@pytest.mark.parametrize(
    ("piece_shape", "rotations"),
    [
        pytest.param((12, 12), 1, id="too-big-any-way-up"),
        pytest.param((8, 3), 1, id="fits-only-rotated-but-rotation-disabled"),
    ],
)
def test_pack_rejects_piece_that_cannot_fit_an_empty_plate(piece_shape, rotations):
    with pytest.raises(ValueError, match="does not fit"):
        pack([solid(*piece_shape)], (4, 10), rotations=rotations)


def test_obstacle_blocks_exactly_the_overlapping_anchors():
    plate = np.zeros((10, 10), np.uint8)
    plate[4, 4] = 1  # single occupied pixel
    legal = legal_placement_map(plate, solid(3, 3))
    # A 3x3 piece anchored at (r, c) covers rows r..r+2, cols c..c+2; it hits
    # (4,4) exactly when the anchor is in rows/cols 2..4.
    blocked = np.zeros((8, 8), bool)
    blocked[2:5, 2:5] = True
    assert (legal == ~blocked).all()


def _asym_mask():
    m = np.zeros((15, 25), np.uint8)
    m[2:13, 3:22] = 1
    m[2:5, 3:8] = 0  # notch: no rotational symmetry
    return m


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, 270.0, 30.0, 45.0, 137.5])
def test_rotate_affine_maps_content_into_content(angle):
    mask = _asym_mask()
    rotated, aff = rotate_mask(mask, angle)
    rr, cc = np.nonzero(mask)
    pts = aff @ np.vstack([cc, rr, np.ones_like(cc)])  # (x', y') columns
    xs, ys = np.round(pts[0]).astype(int), np.round(pts[1]).astype(int)
    assert (xs >= 0).all() and (ys >= 0).all()
    assert (xs < rotated.shape[1]).all() and (ys < rotated.shape[0]).all()
    # binarization only GROWS the mask: each mapped point must hit an on-pixel
    # within its 3x3 neighborhood (exact pixel for right angles)
    if angle % 90 == 0:
        assert rotated[ys, xs].all()
    else:
        padded = np.pad(rotated, 1)
        hit = np.zeros(len(xs), bool)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                hit |= padded[ys + dy, xs + dx].astype(bool)
        assert hit.all()


@pytest.mark.parametrize("angle", [0.0, 90.0, 30.0, 137.5])
def test_rotate_affine_is_rigid(angle):
    _, aff = rotate_mask(_asym_mask(), angle)
    lin = aff[:, :2]
    np.testing.assert_allclose(lin @ lin.T, np.eye(2), atol=1e-9)
    assert np.linalg.det(lin) > 0


@pytest.mark.parametrize("angle", [90.0, 30.0])
def test_rotate_output_bbox_is_tight(angle):
    rotated, _ = rotate_mask(_asym_mask(), angle)
    assert rotated[0].any() and rotated[-1].any()
    assert rotated[:, 0].any() and rotated[:, -1].any()

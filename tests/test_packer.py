"""Packer tests: FFT legality, placement heuristic, greedy spillover."""

import numpy as np
import pytest

from plate_packer.packer import (
    Placement,
    bottom_left,
    contact_first,
    contact_map,
    contact_ring,
    legal_placement_map,
    pack,
    rotate_mask,
    rotate_pair,
    seed_order,
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


def test_pack_validate_false_still_raises_valueerror_on_unfittable_piece():
    # validate=False skips the up-front fit check, but a piece that fits no
    # plate must still fail with the documented ValueError, not a raw
    # TypeError from unpacking a None placement.
    with pytest.raises(ValueError, match="does not fit"):
        pack([solid(12, 12)], (4, 10), validate=False)


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


# --- contact scoring (Task 1) ---

RING_1PX = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
RING_2X2 = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], np.uint8)
# L-shape [[1,0],[1,1]] padded to 4x4; ring = dilation minus mask.
RING_L = np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], np.uint8)


@pytest.mark.parametrize(
    "mask, expected",
    [
        (np.ones((1, 1), np.uint8), RING_1PX),
        (np.ones((2, 2), np.uint8), RING_2X2),
        (np.array([[1, 0], [1, 1]], np.uint8), RING_L),
    ],
    ids=["single-pixel", "square-2x2", "L-shape"],
)
def test_contact_ring_known_shapes(mask, expected):
    np.testing.assert_array_equal(contact_ring(mask), expected)


@pytest.mark.parametrize(
    "anchor, expected",
    [((0, 0), 5), ((0, 1), 3), ((1, 1), 0), ((0, 3), 5), ((3, 3), 5)],
    ids=["corner-tl", "top-edge", "interior", "corner-tr", "corner-br"],
)
def test_contact_map_empty_plate_border_contact(anchor, expected):
    plate = np.zeros((4, 4), np.uint8)
    ring = contact_ring(np.ones((1, 1), np.uint8))
    cmap = contact_map(plate, ring)
    assert cmap.shape == (4, 4)
    assert cmap[anchor] == expected


@pytest.mark.parametrize(
    "anchor, expected",
    [((1, 1), 1), ((2, 1), 1), ((2, 2), 0), ((0, 0), 5)],
    ids=["diagonal-neighbour", "side-neighbour", "on-top-center-excluded", "far-corner"],
)
def test_contact_map_single_occupied_pixel(anchor, expected):
    plate = np.zeros((5, 5), np.uint8)
    plate[2, 2] = 1
    ring = contact_ring(np.ones((1, 1), np.uint8))
    assert contact_map(plate, ring)[anchor] == expected


def test_contact_first_picks_max_contact():
    legal = np.ones((2, 2), bool)
    contact = np.array([[0.0, 1.0], [2.0, 0.0]])
    assert contact_first(legal, contact) == (1, 0)


def test_contact_first_tie_breaks_bottom_left():
    legal = np.ones((2, 2), bool)
    contact = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert contact_first(legal, contact) == (0, 0)


def test_contact_first_ignores_illegal_high_scores():
    legal = np.array([[False, True], [True, False]])
    contact = np.array([[9.0, 1.0], [2.0, 0.0]])
    assert contact_first(legal, contact) == (1, 0)


def test_contact_first_returns_none_when_nothing_legal():
    assert contact_first(np.zeros((3, 3), bool), np.ones((3, 3))) is None


def test_contact_first_declares_uses_contact():
    assert getattr(contact_first, "uses_contact", False) is True


def test_bottom_left_ignores_contact_argument():
    legal = np.ones((2, 2), bool)
    contact = np.array([[0.0, 0.0], [9.0, 9.0]])
    assert bottom_left(legal, contact) == (0, 0)


# --- scored placement integration (Task 2) ---


def test_placement_contact_defaults_to_zero():
    p = Placement(0, 0, 1, 2, 90.0)
    assert p.contact == 0.0


def test_pack_default_choose_records_contact_scores():
    placements = pack([solid(2, 2), solid(2, 2)], (6, 6))
    by_piece = {p.piece: p for p in placements}
    assert by_piece[0].contact == 7.0  # 2x2 in a corner: 7 halo px on the border frame
    # A free corner (2 border edges = 7) always outscores a mid-edge slot
    # adjacent to piece 0 (1 border edge + 1 neighbour = 4 + 2 = 6), so the
    # second piece takes the next free corner, not the adjacent slot.
    assert by_piece[1].contact == 7.0


def test_pack_contact_places_second_piece_in_a_free_corner():
    placements = pack([solid(2, 2), solid(2, 2)], (6, 6))
    by_piece = {p.piece: p for p in placements}
    assert (by_piece[0].row, by_piece[0].col) == (0, 0)
    assert (by_piece[1].row, by_piece[1].col) == (0, 4)


def test_pack_bottom_left_contact_is_zero():
    placements = pack([solid(2, 2)], (6, 6), choose=bottom_left)
    assert placements[0].contact == 0.0


def test_pack_rotation_chosen_for_snugness():
    # 8x8 plate; cols 0..2 occupied except a 1-wide, 4-deep slot at col 1.
    # A 1x4 bar fits the slot only rotated (4x1); the slot's 3-sided contact
    # (14) beats any open-area placement (8), so scoring must pick 90 deg
    # even though 0 deg is legal elsewhere.
    plate_mask = np.zeros((8, 8), np.uint8)
    plate_mask[:, :3] = 1
    plate_mask[:4, 1] = 0
    placements = pack([solid(1, 4)], (8, 8), rotations=4, plate_mask=plate_mask)
    p = placements[0]
    assert p.angle == 90.0
    assert (p.row, p.col) == (0, 1)
    assert p.contact == 14.0


def test_pack_prerotated_and_order_match_defaults():
    pieces = [solid(2, 2), solid(3, 3), solid(1, 2)]
    angles = [0.0, 90.0, 180.0, 270.0]
    prerotated = [{a: rotate_mask(p, a)[0] for a in angles} for p in pieces]
    order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True)
    default = pack(pieces, (8, 8), rotations=4)
    explicit = pack(pieces, (8, 8), rotations=4, prerotated=prerotated, order=order, validate=False)
    assert default == explicit


def test_pack_validate_true_rejects_oversized_piece():
    with pytest.raises(ValueError, match="does not fit"):
        pack([solid(9, 9)], (4, 4))


def test_pack_custom_order_is_respected():
    # Two pieces, reversed order: the SMALL piece is placed first and takes
    # the (0, 0) corner.
    pieces = [solid(3, 3), solid(2, 2)]
    placements = pack(pieces, (8, 8), order=[1, 0])
    by_piece = {p.piece: p for p in placements}
    assert (by_piece[1].row, by_piece[1].col) == (0, 0)


# --- edge_weight contact scaling (Task 2, ADR-012) ---


@pytest.mark.parametrize(
    "edge_weight, corner, top, interior",
    [(1.0, 5, 3, 0), (2.0, 10, 6, 0), (0.0, 0, 0, 0)],
    ids=["ew-1", "ew-2-doubles-border", "ew-0-zeroes-border"],
)
def test_contact_map_edge_weight_scales_border(edge_weight, corner, top, interior):
    plate = np.zeros((4, 4), np.uint8)
    ring = contact_ring(np.ones((1, 1), np.uint8))
    cmap = contact_map(plate, ring, edge_weight)
    assert cmap[0, 0] == corner
    assert cmap[0, 1] == top
    assert cmap[1, 1] == interior


@pytest.mark.parametrize("edge_weight", [1.0, 3.0], ids=["ew-1", "ew-3"])
def test_contact_map_edge_weight_leaves_piece_contact_unchanged(edge_weight):
    plate = np.zeros((5, 5), np.uint8)
    plate[2, 2] = 1  # one occupied interior pixel
    ring = contact_ring(np.ones((1, 1), np.uint8))
    cmap = contact_map(plate, ring, edge_weight)
    assert cmap[2, 1] == 1  # side neighbour of the occupied pixel: piece-piece only
    assert cmap[1, 1] == 1  # diagonal neighbour: piece-piece only


def test_contact_map_edge_weight_defaults_to_one():
    plate = np.zeros((4, 4), np.uint8)
    ring = contact_ring(np.ones((1, 1), np.uint8))
    np.testing.assert_array_equal(contact_map(plate, ring), contact_map(plate, ring, 1.0))


def test_contact_map_fractional_edge_weight_not_truncated():
    # A fractional edge_weight must actually take effect, not be coerced to 0 by
    # padding a uint8 array (regression: PR #6 Critical).
    plate = np.zeros((4, 4), np.uint8)
    ring = contact_ring(np.ones((1, 1), np.uint8))
    zero = contact_map(plate, ring, 0.0)[0, 0]
    half = contact_map(plate, ring, 0.5)[0, 0]
    full = contact_map(plate, ring, 1.0)[0, 0]
    assert zero == 0
    assert half > zero  # 0.5 must NOT truncate to 0
    assert half < full  # and stays below full weight


@pytest.mark.parametrize(
    "edge_weight, corner",
    [(0.1, 0.5), (0.2, 1.0), (0.3, 1.5)],
    ids=["ew-0.1", "ew-0.2", "ew-0.3"],
)
def test_contact_map_preserves_small_fractional_edge_weight(edge_weight, corner):
    # np.rint would quantize the 5-border-cell corner score to whole numbers
    # (0.1 -> rint(0.5) -> 0, erasing the knob); rounding to 2 decimals keeps the
    # fractional border signal (PR #6 review). corner = 5 border cells * weight.
    plate = np.zeros((4, 4), np.uint8)
    ring = contact_ring(np.ones((1, 1), np.uint8))
    assert contact_map(plate, ring, edge_weight)[0, 0] == pytest.approx(corner)


# --- difficulty ordering (Task 2, ADR-012) ---


@pytest.mark.parametrize(
    "ordering, expected",
    [("difficulty", [1, 0]), ("area", [0, 1])],
    ids=["difficulty-elongated-first", "area-equal-keeps-index-order"],
)
def test_seed_order_equal_area_bar_vs_blob(ordering, expected):
    # bar and blob both have area 36; difficulty (area*elongation) puts the
    # elongated bar first, area treats them as a tie broken by index.
    pieces = [solid(6, 6), solid(2, 18)]
    assert seed_order(pieces, ordering) == expected


@pytest.mark.parametrize(
    "ordering, expected",
    [("difficulty", [1, 0]), ("area", [0, 1])],
    ids=["difficulty-elongation-dominates", "area-bigger-blob-first"],
)
def test_seed_order_elongation_can_outrank_area(ordering, expected):
    # big blob has more area (49 > 36) but the bar's elongation dominates the
    # difficulty product.
    pieces = [solid(7, 7), solid(2, 18)]
    assert seed_order(pieces, ordering) == expected


def test_seed_order_area_matches_legacy_largest_first():
    pieces = [solid(2, 2), solid(5, 5), solid(3, 3)]
    assert seed_order(pieces, "area") == sorted(
        range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True
    )


# --- shared-canvas rotation (Task 8) ---


def test_rotate_pair_degenerate_full_equals_body():
    full = np.ones((10, 12), np.uint8)
    fr, br, _aff = rotate_pair(full, full, 37.0)
    assert br.shape == fr.shape
    assert (br == fr).all()


def test_rotate_pair_angle0_reconstructs_body():
    full = np.ones((12, 12), np.uint8)
    body = np.zeros((12, 12), np.uint8)
    body[2:10, 4:8] = 1  # narrower body, smaller bbox than full
    fr, br, _aff = rotate_pair(full, body, 0.0)
    assert (fr == full).all()
    assert (br == body).all()  # body placed back at its full-frame position


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 150.0], ids=lambda a: f"deg{a:g}")
def test_rotate_pair_body_subset_same_shape(angle):
    full = np.ones((12, 12), np.uint8)
    body = full.copy()
    body[3:9, 3:9] = 0  # interior hole (same bbox as full)
    fr, br, _aff = rotate_pair(full, body, angle)
    assert br.shape == fr.shape
    assert (br & ~fr).sum() == 0  # body_rot is a subset of full_rot
    assert br.sum() < fr.sum()  # the hole survives


# --- two-mask packing (Task 9) ---


def _paired_variants(full, body, angles):
    """Build (body_variants, full_variants) dicts sharing a canvas per angle."""
    bvar, fvar = {}, {}
    for a in angles:
        fr, br, _ = rotate_pair(full, body, a)
        fvar[a], bvar[a] = fr, br
    return bvar, fvar


def test_boundary_rafts_may_overlap_same_plate():
    # full = 20x20 raft; body = central 6-wide column (bodies stay disjoint,
    # but the wide rafts overlap). Two pieces should share ONE plate.
    full = np.ones((20, 20), np.uint8)
    body = np.zeros((20, 20), np.uint8)
    body[:, 7:13] = 1
    b, f = _paired_variants(full, body, [0.0])
    placements = pack([body], (20, 40), prerotated=[b], boundary=[f], order=[0], validate=False)
    # pack a second identical piece by passing two
    placements = pack(
        [body, body],
        (20, 40),
        prerotated=[b, b],
        boundary=[f, f],
        order=[0, 1],
        validate=False,
    )
    assert max(p.plate for p in placements) == 0  # both on plate 0


def test_boundary_full_kept_on_plate():
    # A piece whose body would fit flush at the right edge but whose full shadow
    # (same shared shape, wider content) must stay within the plate: with a
    # bordered plate the full may not overlap the border.
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[:, :4] = 1  # body content only on the left of the shared canvas
    b, f = _paired_variants(full, body, [0.0])
    border = np.zeros((10, 30), np.uint8)
    border[:, :2] = border[:, -2:] = 1  # 2px dead margins left/right
    placements = pack(
        [body],
        (10, 30),
        plate_mask=border,
        prerotated=[b],
        boundary=[f],
        order=[0],
        validate=False,
    )
    (pl,) = placements
    # full (all 10 cols occupied) must sit clear of both 2px borders
    assert pl.col >= 2 and pl.col + 10 <= 28


def test_boundary_empty_plate_fit_uses_full():
    # body fits a tiny plate but the full shadow does not -> rejected.
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[:4, :4] = 1
    b, f = _paired_variants(full, body, [0.0])
    with pytest.raises(ValueError, match="does not fit"):
        pack([body], (6, 6), prerotated=[b], boundary=[f], order=[0], validate=True)


def test_boundary_raft_may_not_overlap_another_body():
    # B: solid, no cut (body == full). A: cut, body is the RIGHT strip; raft = left.
    B_full = np.ones((20, 20), np.uint8)
    A_full = np.ones((20, 20), np.uint8)
    A_body = np.zeros((20, 20), np.uint8)
    A_body[:, 16:] = 1
    plate = (20, 40)
    prerot = [{0.0: B_full}, {0.0: A_body}]
    bound = [{0.0: B_full}, {0.0: A_full}]
    placements = pack(
        [B_full, A_body], plate, prerotated=prerot, boundary=bound, order=[0, 1], validate=False
    )
    occ_body_B = np.zeros(plate, np.uint8)
    occ_full_A = np.zeros(plate, np.uint8)
    for pl in placements:
        m = bound[pl.piece][0.0] if pl.piece == 1 else B_full  # A: full; B: body(==full)
        tgt = occ_full_A if pl.piece == 1 else occ_body_B
        tgt[pl.row : pl.row + 20, pl.col : pl.col + 20] |= m
    assert (occ_full_A & occ_body_B).sum() == 0  # A's raft must not sit on B's body


def test_boundary_body_may_not_overlap_another_raft():
    # B: cut, body is the LEFT strip; raft = right. A: solid (body == full).
    B_full = np.ones((20, 20), np.uint8)
    B_body = np.zeros((20, 20), np.uint8)
    B_body[:, :4] = 1
    A_full = np.ones((20, 20), np.uint8)
    plate = (20, 40)
    prerot = [{0.0: B_body}, {0.0: A_full}]
    bound = [{0.0: B_full}, {0.0: A_full}]
    placements = pack(
        [B_body, A_full], plate, prerotated=prerot, boundary=bound, order=[0, 1], validate=False
    )
    B_raft = B_full & ~B_body
    occ_raft_B = np.zeros(plate, np.uint8)
    occ_body_A = np.zeros(plate, np.uint8)
    for pl in placements:
        if pl.piece == 0:
            occ_raft_B[pl.row : pl.row + 20, pl.col : pl.col + 20] |= B_raft
        else:
            occ_body_A[pl.row : pl.row + 20, pl.col : pl.col + 20] |= A_full
    assert (occ_body_A & occ_raft_B).sum() == 0  # A's body must not sit on B's raft

"""Tests for the improvement loop: fitness, moves, ILS (spec 2026-08-05)."""

from itertools import pairwise

import numpy as np
import pytest

from plate_packer.angles import angle_candidates
from plate_packer.improve import (
    ImproveResult,
    _move_random_reinsert,
    _move_random_swap,
    _move_targeted_reinsert,
    _move_targeted_swap,
    _move_window_shuffle,
    _prerotate_multi_res,
    _reinsert,
    _update_beam,
    falkenauer,
    improve,
    perturb,
    plate_fills,
    shake,
)
from plate_packer.packer import Placement, pack, rotate_mask, seed_order


def solid(h, w):
    return np.ones((h, w), np.uint8)


@pytest.mark.parametrize(
    "fills, expected",
    [
        ([1.0], 1.0),
        ([0.5, 0.5], 0.25),
        ([0.6, 0.6], 0.36),
        ([1.0, 0.1, 0.1], 0.34),
    ],
    ids=["one-full", "two-half", "two-0.6", "concentrated-three"],
)
def test_falkenauer_known_values(fills, expected):
    assert falkenauer(fills) == pytest.approx(expected)


def test_falkenauer_prefers_concentration():
    assert falkenauer([0.75, 0.74, 0.73, 0.11]) > falkenauer([0.62, 0.61, 0.54, 0.56])


def test_falkenauer_prefers_fewer_plates_same_area():
    # total fill 1.2 spread over 2 vs 3 plates
    assert falkenauer([0.6, 0.6]) > falkenauer([1.0, 0.19, 0.01])


def test_plate_fills_sums_piece_pixels_per_plate():
    placements = [
        Placement(0, 0, 0, 0, 0.0),
        Placement(1, 0, 0, 5, 0.0),
        Placement(2, 1, 0, 0, 0.0),
    ]
    piece_px = [10, 20, 5]
    fills = plate_fills(placements, piece_px, usable_px=100)
    assert fills == [pytest.approx(0.30), pytest.approx(0.05)]


def test_improve_result_is_frozen():
    r = ImproveResult([], 1, 0, 0.5, 0.5)
    with pytest.raises(AttributeError):
        r.evaluations = 2


def _fixture_placements():
    # plate 0 (fill 0.9): pieces 0,1 (contacts 8, 2); plate 1 (fill 0.1): piece 2
    return [
        Placement(0, 0, 0, 0, 0.0, contact=8.0),
        Placement(1, 0, 0, 5, 0.0, contact=2.0),
        Placement(2, 1, 0, 0, 0.0, contact=1.0),
    ]


FILLS = [0.9, 0.1]
ORDER = [0, 1, 2]


@pytest.mark.parametrize(
    "move",
    [
        _move_targeted_reinsert,
        _move_targeted_swap,
        _move_random_swap,
        _move_random_reinsert,
        _move_window_shuffle,
    ],
    ids=["t-reinsert", "t-swap", "r-swap", "r-reinsert", "window"],
)
def test_moves_return_valid_permutations(move):
    rng = np.random.default_rng(42)
    result = move(ORDER, _fixture_placements(), FILLS, rng)
    assert sorted(result) == ORDER
    assert ORDER == [0, 1, 2]  # input never mutated


def test_reinsert_moves_element():
    assert _reinsert([0, 1, 2, 3], 3, 0) == [3, 0, 1, 2]


def test_targeted_reinsert_moves_min_fill_plate_piece_earlier():
    rng = np.random.default_rng(0)
    result = _move_targeted_reinsert(ORDER, _fixture_placements(), FILLS, rng)
    # piece 2 (only piece on the min-fill plate) must move earlier than pos 2
    assert result.index(2) < 2


def test_targeted_swap_picks_lowest_contact_of_sample():
    # SAMPLE (5) >= population (3): the sample is the whole population, so the
    # lowest-contact piece (2) is always one side of the swap.
    rng = np.random.default_rng(0)
    result = _move_targeted_swap(ORDER, _fixture_placements(), FILLS, rng)
    assert result.index(2) != ORDER.index(2)


def test_targeted_reinsert_single_plate_falls_back():
    placements = [Placement(0, 0, 0, 0, 0.0), Placement(1, 0, 0, 5, 0.0)]
    rng = np.random.default_rng(1)
    result = _move_targeted_reinsert([0, 1], placements, [0.5], rng)
    assert sorted(result) == [0, 1]


def test_window_shuffle_short_order_falls_back():
    rng = np.random.default_rng(2)
    result = _move_window_shuffle([0, 1], _fixture_placements()[:2], [0.5], rng)
    assert sorted(result) == [0, 1]


def test_perturb_is_deterministic_per_seed():
    a = perturb(ORDER, _fixture_placements(), FILLS, np.random.default_rng(7))
    b = perturb(ORDER, _fixture_placements(), FILLS, np.random.default_rng(7))
    assert a == b


def test_shake_returns_valid_permutation():
    rng = np.random.default_rng(3)
    result = shake(ORDER, _fixture_placements(), FILLS, rng)
    assert sorted(result) == ORDER


PIECES = [solid(2, 2), solid(2, 2), solid(2, 2)]


def test_improve_budget_zero_equals_greedy_seed_pack():
    pieces = PIECES
    res = improve(pieces, (6, 6), budget_s=0.0)
    angles = [angle_candidates(p) for p in pieces]
    prerot = [{a: rotate_mask(p, a)[0] for a in ang} for p, ang in zip(pieces, angles, strict=True)]
    order = seed_order(pieces, "difficulty")
    expected = pack(pieces, (6, 6), prerotated=prerot, order=order, validate=False)
    assert res.placements == expected
    assert res.evaluations == 1
    assert res.fitness_initial == res.fitness_final


def test_improve_same_seed_same_result():
    kwargs = dict(budget_s=0.5, patience=10, min_improvement=0.0, seed=7)
    a = improve(PIECES, (6, 6), **kwargs)
    b = improve(PIECES, (6, 6), **kwargs)
    assert a.placements == b.placements
    assert (a.evaluations, a.improvements) == (b.evaluations, b.improvements)


def test_improve_stall_stop_counts_evaluations():
    # min_improvement can never be met -> marker never resets -> exactly
    # patience loop evaluations after the initial one.
    res = improve(PIECES, (6, 6), budget_s=60.0, patience=3, min_improvement=10.0)
    assert res.evaluations == 4  # 1 initial + 3 stalled


def test_improve_budget_stop_terminates_before_any_iteration(monkeypatch):
    # Budget already exhausted when the first while-guard is checked: the loop
    # must not run, leaving only the initial evaluation. _FakeClock(0) fails the
    # guard immediately. patience is unreachable so ONLY the budget can stop it.
    monkeypatch.setattr("plate_packer.improve.time.monotonic", _FakeClock(0))
    res = improve(PIECES, (6, 6), budget_s=0.2, patience=10**9, min_improvement=0.0)
    assert res.evaluations == 1


def test_improve_fitness_never_worsens():
    res = improve(PIECES, (6, 6), budget_s=0.5, patience=20, min_improvement=0.0)
    assert res.fitness_final >= res.fitness_initial


def test_improve_on_improve_reports_increasing_fitness():
    seen = []
    improve(
        [solid(3, 3), solid(3, 3), solid(2, 2), solid(2, 2)],
        (7, 7),
        budget_s=1.0,
        patience=50,
        min_improvement=0.0,
        seed=3,
        on_improve=lambda evals, plates, fit: seen.append((evals, plates, fit)),
    )
    fits = [f for _, _, f in seen]
    assert fits == sorted(fits)
    assert all(f2 > f1 for f1, f2 in pairwise(fits))


class _FakeClock:
    """monotonic() returns 0.0 for the first (allowed_iters + 1) calls -- the
    `start` capture plus that many passing while-guard checks -- then a large
    value that fails the guard. Emulates a machine that fits exactly
    `allowed_iters` loop evaluations into the budget, with no real-time jitter.
    """

    def __init__(self, allowed_iters):
        self._remaining = allowed_iters + 1

    def __call__(self):
        if self._remaining > 0:
            self._remaining -= 1
            return 0.0
        return 9999.0


# Diverse pieces so different insertion orders yield different packings.
_WALL_PIECES = [solid(3, 3), solid(2, 4), solid(4, 2), solid(2, 2), solid(3, 2)]


@pytest.mark.parametrize(
    "allowed_iters, expected_evaluations",
    [(3, 4), (7, 8)],
    ids=["fast-machine-3", "slow-machine-7"],
)
def test_improve_wall_clock_eval_count_follows_time_schedule(
    monkeypatch, allowed_iters, expected_evaluations
):
    # Same seed, different clock speeds -> different evaluation counts. This is
    # the budget-bounded path that is machine-dependent (see improve() docstring):
    # patience is set unreachably high so ONLY the wall clock stops the loop.
    monkeypatch.setattr("plate_packer.improve.time.monotonic", _FakeClock(allowed_iters))
    res = improve(_WALL_PIECES, (8, 8), budget_s=1.0, patience=10**9, min_improvement=0.0, seed=5)
    assert res.evaluations == expected_evaluations


def test_improve_deterministic_for_fixed_eval_count(monkeypatch):
    # With the evaluation count pinned (fixed clock schedule) the run is fully
    # reproducible per seed -- the property the docstring actually guarantees.
    def run():
        monkeypatch.setattr("plate_packer.improve.time.monotonic", _FakeClock(6))
        return improve(
            _WALL_PIECES, (8, 8), budget_s=1.0, patience=10**9, min_improvement=0.0, seed=5
        )

    a, b = run(), run()
    assert a.placements == b.placements
    assert (a.evaluations, a.improvements) == (b.evaluations, b.improvements)


def test_update_beam_keeps_top_k_by_fitness():
    beam = []
    for order, fit in ([0, 1, 2], 0.3), ([2, 1, 0], 0.5), ([1, 0, 2], 0.4):
        beam = _update_beam(beam, order, fit, k=2)
    assert [f for f, _ in beam] == [0.5, 0.4]
    assert [o for _, o in beam] == [[2, 1, 0], [1, 0, 2]]


def test_update_beam_dedupes_identical_orderings():
    beam = _update_beam([], [0, 1, 2], 0.3, k=5)
    beam = _update_beam(beam, [0, 1, 2], 0.3, k=5)
    assert len(beam) == 1


def test_update_beam_keeps_best_fitness_for_repeated_ordering():
    beam = _update_beam([], [0, 1, 2], 0.3, k=5)
    beam = _update_beam(beam, [0, 1, 2], 0.7, k=5)
    assert beam == [(0.7, [0, 1, 2])]


def test_update_beam_breaks_fitness_ties_by_ordering():
    beam = _update_beam([], [2, 1, 0], 0.5, k=5)
    beam = _update_beam(beam, [0, 1, 2], 0.5, k=5)
    assert [o for _, o in beam] == [[0, 1, 2], [2, 1, 0]]  # ascending tuple order


def test_update_beam_does_not_mutate_input():
    original = [(0.5, [2, 1, 0])]
    _update_beam(original, [0, 1, 2], 0.9, k=5)
    assert original == [(0.5, [2, 1, 0])]


def test_prerotate_multi_res_keys_match_angles_at_both_resolutions():
    pieces = [solid(3, 5)]
    fine, coarse = _prerotate_multi_res(pieces, [[0.0, 90.0]], factor=2)
    assert set(fine[0]) == {0.0, 90.0}
    assert set(coarse[0]) == {0.0, 90.0}


@pytest.mark.parametrize("angle", [0.0, 90.0, 30.0], ids=["0deg", "90deg", "30deg"])
def test_prerotate_coarse_is_superset_of_fine(angle):
    # Every ON pixel of the fine rotated mask must fall in an ON coarse cell
    # (block-max grows-never-shrinks) -> coarse-legal implies fine-legal.
    piece = np.zeros((6, 10), np.uint8)
    piece[1:5, 2:8] = 1
    factor = 3
    fine, coarse = _prerotate_multi_res([piece], [[angle]], factor)
    fm, cm = fine[0][angle], coarse[0][angle]
    fr, fc = np.nonzero(fm)
    assert cm[fr // factor, fc // factor].all()


def test_prerotate_factor_one_coarse_equals_fine():
    pieces = [solid(4, 4)]
    fine, coarse = _prerotate_multi_res(pieces, [[0.0]], factor=1)
    np.testing.assert_array_equal(fine[0][0.0], coarse[0][0.0])


def test_improve_result_has_beam_field():
    res = improve(PIECES, (6, 6), budget_s=0.0)
    assert isinstance(res.beam, list)
    for _coarse_fit, _fine_fit, n_plates in res.beam:
        assert isinstance(n_plates, int)


def test_improve_returns_best_fine_not_best_coarse():
    # fitness_final is a FINE fitness and never below the seed's fine fitness.
    res = improve(_WALL_PIECES, (10, 10), budget_s=0.4, patience=40, min_improvement=0.0, seed=5)
    assert res.fitness_final >= res.fitness_initial


def test_improve_coarse_legal_orderings_pack_legally_at_fine():
    # Every returned placement must be collision-free at fine resolution
    # (coarse-legal => fine-legal).
    pieces = [solid(3, 3), solid(2, 4), solid(4, 2), solid(2, 2)]
    res = improve(pieces, (8, 8), budget_s=0.3, patience=30, min_improvement=0.0, seed=1)
    occ = {}
    for p in res.placements:
        m, _ = rotate_mask(pieces[p.piece], p.angle)
        plate = occ.setdefault(p.plate, np.zeros((8, 8), np.uint8))
        plate[p.row : p.row + m.shape[0], p.col : p.col + m.shape[1]] += m
    assert all(plate.max() <= 1 for plate in occ.values())


def test_improve_beam_size_bounded_by_beam_param():
    res = improve(
        _WALL_PIECES, (10, 10), budget_s=0.3, patience=40, min_improvement=0.0, seed=5, beam=3
    )
    assert len(res.beam) <= 3

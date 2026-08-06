"""Iterated local search over the greedy insertion order (spec 2026-08-05).

Fitness is Falkenauer's grouping objective: mean squared plate fill. With
total piece area fixed, concentrating area on fewer/fuller plates always
scores higher, so plate count needs no separate objective term.
"""

import time
from dataclasses import dataclass

import numpy as np

from plate_packer.packer import Placement, _fits, contact_first, pack, rotate_mask

SHAKE_AFTER = 20
SHAKE_MOVES = 4
SAMPLE = 5
WINDOW = 3


@dataclass(frozen=True)
class ImproveResult:
    placements: list[Placement]
    evaluations: int
    improvements: int
    fitness_initial: float
    fitness_final: float


def plate_fills(placements, piece_px, usable_px) -> list[float]:
    """Per-plate fill fractions from dilated-mask pixel sums (placements never
    overlap, so fills are additive)."""
    n = max(p.plate for p in placements) + 1
    totals = [0] * n
    for p in placements:
        totals[p.plate] += piece_px[p.piece]
    return [t / usable_px for t in totals]


def falkenauer(fills) -> float:
    """Mean squared fill -- maximize."""
    return float(sum(f * f for f in fills) / len(fills))


def _update_beam(beam, order, fitness, k):
    """Return a new beam of up to k (fitness, order) pairs, best fitness first,
    orderings distinct, ties broken by ordering. Input beam is not mutated."""
    best = {tuple(o): f for f, o in beam}
    key = tuple(order)
    if key not in best or fitness > best[key]:
        best[key] = fitness
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(f, list(o)) for o, f in ranked[:k]]


def _reinsert(order, i_pos, j_pos):
    order = list(order)
    piece = order.pop(i_pos)
    order.insert(j_pos, piece)
    return order


def _move_random_swap(order, placements, fills, rng):
    if len(order) < 2:
        return list(order)
    i = int(rng.integers(0, len(order)))
    j = int((i + 1 + rng.integers(0, len(order) - 1)) % len(order))
    order = list(order)
    order[i], order[j] = order[j], order[i]
    return order


def _move_random_reinsert(order, placements, fills, rng):
    if len(order) < 2:
        return list(order)
    i = int(rng.integers(0, len(order)))
    j = int(rng.integers(0, len(order)))
    return _reinsert(order, i, j)


def _move_window_shuffle(order, placements, fills, rng):
    if len(order) < WINDOW:
        return _move_random_swap(order, placements, fills, rng)
    start = int(rng.integers(0, len(order) - WINDOW + 1))
    order = list(order)
    window = [order[start + k] for k in rng.permutation(WINDOW)]
    order[start : start + WINDOW] = window
    return order


def _move_targeted_reinsert(order, placements, fills, rng):
    """Random piece from the min-fill plate -> random earlier order position."""
    if len(fills) < 2:
        return _move_random_reinsert(order, placements, fills, rng)
    donor = int(np.argmin(fills))
    candidates = [p.piece for p in placements if p.plate == donor]
    piece = int(rng.choice(candidates))
    i = order.index(piece)
    if i == 0:
        return _move_random_reinsert(order, placements, fills, rng)
    j = int(rng.integers(0, i))
    return _reinsert(order, i, j)


def _move_targeted_swap(order, placements, fills, rng):
    """Lowest-contact piece of a random sample <-> random other position."""
    if len(order) < 2:
        return list(order)
    k = min(SAMPLE, len(placements))
    sample = rng.choice(len(placements), size=k, replace=False)
    worst = min((placements[int(s)] for s in sample), key=lambda p: p.contact).piece
    i = order.index(worst)
    j = int((i + 1 + rng.integers(0, len(order) - 1)) % len(order))
    order = list(order)
    order[i], order[j] = order[j], order[i]
    return order


_MOVES = [
    (0.45, _move_targeted_reinsert),
    (0.25, _move_targeted_swap),
    (0.15, _move_random_swap),
    (0.10, _move_random_reinsert),
    (0.05, _move_window_shuffle),
]
_RANDOM_MOVES = [_move_random_swap, _move_random_reinsert, _move_window_shuffle]


def perturb(order, placements, fills, rng):
    """One weighted-random move applied to a copy of order."""
    r = float(rng.random())
    acc = 0.0
    for prob, move in _MOVES:
        acc += prob
        if r < acc:
            return move(order, placements, fills, rng)
    return _MOVES[-1][1](order, placements, fills, rng)


def shake(order, placements, fills, rng):
    """SHAKE_MOVES stacked uniform-random moves -- the ILS escape hatch."""
    for _ in range(SHAKE_MOVES):
        move = _RANDOM_MOVES[int(rng.integers(0, len(_RANDOM_MOVES)))]
        order = move(order, placements, fills, rng)
    return order


def improve(
    pieces,
    plate_shape,
    rotations=1,
    plate_mask=None,
    choose=None,
    budget_s=2700.0,
    min_improvement=0.005,
    patience=30,
    seed=0,
    validate=True,
    on_improve=None,
):
    """Iterated local search over the greedy insertion order.

    Anytime: every evaluation is a complete valid packing, so both stop
    conditions (wall-clock budget, stall: `patience` evaluations without
    `min_improvement` cumulative fitness gain) return the best found.
    budget_s=0 returns the plain greedy pack.

    Determinism: the search is deterministic per seed *for a fixed number of
    evaluations* -- each evaluation consumes the shared RNG, so identical draw
    sequences require identical evaluation counts. The stall stop is itself
    deterministic (it fires at a fixed evaluation count for given inputs), but
    the wall-clock budget is NOT: how many evaluations fit in `budget_s`
    depends on machine speed and load, so a budget-bounded run can yield a
    different layout for the same seed on different hardware. For a fully
    reproducible run set `budget_s` high enough that the stall condition
    always fires first (see key_facts.md).

    on_improve(evaluations, n_plates, fitness) fires at each new best.
    """
    choose = choose or contact_first
    rng = np.random.default_rng(seed)
    angles = [i * 360.0 / rotations for i in range(rotations)]
    prerotated = [{a: rotate_mask(p, a)[0] for a in angles} for p in pieces]
    piece_px = [int(p.sum()) for p in pieces]
    usable_px = plate_shape[0] * plate_shape[1] - (
        int(plate_mask.sum()) if plate_mask is not None else 0
    )
    start = time.monotonic()

    def evaluate(order):
        result = pack(
            pieces,
            plate_shape,
            plate_mask=plate_mask,
            choose=choose,
            prerotated=prerotated,
            order=order,
            validate=False,
        )
        return result, falkenauer(plate_fills(result, piece_px, usable_px))

    best_order = sorted(range(len(pieces)), key=lambda i: piece_px[i], reverse=True)
    if validate:
        # One up-front validation; every repack then runs with validate=False.
        empty = plate_mask.copy() if plate_mask is not None else np.zeros(plate_shape, np.uint8)
        for i, variants in enumerate(prerotated):
            if not any(_fits(empty, m) for m in variants.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")
    best, best_fit = evaluate(best_order)
    fitness_initial = best_fit
    evaluations, improvements = 1, 0
    marker, evals_since_marker, fails = best_fit, 0, 0
    incumbent = best_order

    while time.monotonic() - start < budget_s:
        fills = plate_fills(best, piece_px, usable_px)
        candidate = perturb(incumbent, best, fills, rng)
        result, fit = evaluate(candidate)
        evaluations += 1
        evals_since_marker += 1
        if fit > best_fit:
            best, best_order, best_fit = result, candidate, fit
            incumbent = candidate
            improvements += 1
            fails = 0
            if on_improve is not None:
                on_improve(evaluations, max(p.plate for p in best) + 1, best_fit)
            if best_fit - marker >= min_improvement:
                marker, evals_since_marker = best_fit, 0
        else:
            fails += 1
            if fails >= SHAKE_AFTER:
                incumbent = shake(best_order, best, fills, rng)
                fails = 0
        if evals_since_marker >= patience:
            break

    return ImproveResult(best, evaluations, improvements, fitness_initial, best_fit)

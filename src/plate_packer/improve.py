"""Iterated local search over the greedy insertion order (spec 2026-08-05).

Fitness is Falkenauer's grouping objective: mean squared plate fill. With
total piece area fixed, concentrating area on fewer/fuller plates always
scores higher, so plate count needs no separate objective term.
"""

from dataclasses import dataclass

import numpy as np

from plate_packer.packer import Placement

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

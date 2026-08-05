"""Iterated local search over the greedy insertion order (spec 2026-08-05).

Fitness is Falkenauer's grouping objective: mean squared plate fill. With
total piece area fixed, concentrating area on fewer/fuller plates always
scores higher, so plate count needs no separate objective term.
"""

from dataclasses import dataclass

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

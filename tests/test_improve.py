"""Tests for the improvement loop: fitness, moves, ILS (spec 2026-08-05)."""

import numpy as np
import pytest

from plate_packer.improve import ImproveResult, falkenauer, plate_fills
from plate_packer.packer import Placement


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

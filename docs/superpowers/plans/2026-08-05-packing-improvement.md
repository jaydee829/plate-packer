# Packing Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contact-scored placement + targeted-move iterated local search over insertion order, maximizing Falkenauer fitness, per `docs/superpowers/specs/2026-08-05-packing-improvement-design.md`.

**Architecture:** `packer.py` gains a contact-scoring kernel (ring + FFT contact map) and a scored chooser; `pack()` gains `prerotated`/`order`/`validate` params for cheap repacks. New module `improve.py` holds fitness, moves, and the ILS loop. `config.py`/`cli.py` wire the knobs.

**Tech Stack:** numpy, scipy.signal.fftconvolve, cv2 (dilate), typer, pytest.

## Global Constraints

- All case-driven tests are parametrized (`pytest.mark.parametrize`) — one named case per test, never loops inside a test body. (User's global rule.)
- Conservative-coverage invariant: masks may GROW, never shrink. Scoring must never affect the binary legality gate.
- Fitness = `(1/n)·Σ fillᵢ²`, fillᵢ from **dilated prepared-mask pixel sums** / usable plate pixels. Plate count is NOT a separate objective term.
- Contact map values pass through `np.rint` (FFT-noise-proof ties).
- All randomness through one `numpy.random.default_rng(seed)`; no `time`-seeded or global RNG. Wall clock via `time.monotonic()`.
- Move probabilities exactly: 0.45 targeted-reinsert, 0.25 targeted-swap, 0.15 random-swap, 0.10 random-reinsert, 0.05 window-shuffle. Constants: `SHAKE_AFTER = 20`, `SHAKE_MOVES = 4`, `SAMPLE = 5`, `WINDOW = 3`.
- `budget_s = 0` ⇒ result identical to plain greedy `pack()` (evaluations = 1, improvements = 0).
- Config knob names/defaults exactly: `improve_budget_s = 2700.0`, `min_improvement = 0.005`, `patience = 30`, `seed = 0`, `placement = "contact"`.
- ruff check + format clean; run `uv run ruff check . && uv run ruff format --check .` before each commit. Full suite: `uv run pytest -q`.

---

### Task 1: Contact-scoring kernel (`contact_ring`, `contact_map`, choosers)

**Files:**
- Modify: `src/plate_packer/packer.py`
- Test: `tests/test_packer.py` (append)

**Interfaces:**
- Produces: `contact_ring(mask: np.ndarray) -> np.ndarray` — 1-px halo, shape `(h+2, w+2)`, uint8.
- Produces: `contact_map(plate: np.ndarray, ring: np.ndarray) -> np.ndarray` — float array, shape `(H-h+1, W-w+1)` (same as `legal_placement_map`), values `np.rint`-ed.
- Produces: `contact_first(legal, contact) -> tuple[int, int] | None` with attribute `contact_first.uses_contact = True`.
- Produces: `bottom_left(legal, contact=None) -> tuple[int, int] | None` (signature gains ignored 2nd arg; behavior unchanged).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_packer.py`:

```python
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
```

Add `contact_first, contact_map, contact_ring` to the existing `from plate_packer.packer import ...` block.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_packer.py -q`
Expected: ImportError (`contact_ring` etc. not defined).

- [ ] **Step 3: Implement** — in `src/plate_packer/packer.py`, after `legal_placement_map`:

```python
def contact_ring(mask: np.ndarray) -> np.ndarray:
    """1-px halo around a tight-cropped mask; shape (h+2, w+2)."""
    padded = np.pad(mask, 1).astype(np.uint8)
    return cv2.dilate(padded, np.ones((3, 3), np.uint8)) - padded


def contact_map(plate: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Contact score at every anchor: halo pixels touching occupancy or the
    plate border. Same anchor coordinates and shape as legal_placement_map;
    np.rint collapses FFT noise so score ties are exact."""
    attraction = np.pad(plate, 1, constant_values=1)
    raw = fftconvolve(attraction.astype(np.float32), ring[::-1, ::-1].astype(np.float32), "valid")
    return np.rint(raw)


def contact_first(legal: np.ndarray, contact: np.ndarray) -> tuple[int, int] | None:
    """Legal anchor with the highest contact score; ties resolve bottom-left
    (argmax first-occurrence in row-major order IS lowest row, then col)."""
    if not legal.any():
        return None
    r, c = np.unravel_index(int(np.argmax(np.where(legal, contact, -1.0))), legal.shape)
    return int(r), int(c)


contact_first.uses_contact = True
```

Change `bottom_left` signature to `def bottom_left(legal: np.ndarray, contact: np.ndarray | None = None) -> tuple[int, int] | None:` (body unchanged; docstring notes the ignored argument).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_packer.py -q` — all pass. Then full suite `uv run pytest -q` (the old `choose(legal)`-style internal call in `_best_spot` still works because `contact` is optional on `bottom_left`).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/packer.py tests/test_packer.py
git commit -m "feat: contact-scoring kernel (ring, FFT contact map, scored chooser)"
```

---

### Task 2: Scored placement integration (`pack` params, `Placement.contact`, `_best_spot`)

**Files:**
- Modify: `src/plate_packer/packer.py`
- Test: `tests/test_packer.py` (append)

**Interfaces:**
- Consumes: Task 1's `contact_ring`, `contact_map`, `contact_first`.
- Produces: `Placement` gains `contact: float = 0.0` (appended, default — existing positional constructions stay valid).
- Produces: `pack(pieces, plate_shape, rotations=1, plate_mask=None, choose=None, prerotated=None, order=None, validate=True)`; default `choose` becomes `contact_first`.
- Produces: `CHOOSERS = {"contact": contact_first, "bottom_left": bottom_left}` (module-level, for config wiring).
- Produces: `_best_spot(occupancy, variants, rings, choose)` returning `(anchor, angle, score) | None`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_packer.py`:

```python
# --- scored placement integration (Task 2) ---


def test_placement_contact_defaults_to_zero():
    p = Placement(0, 0, 1, 2, 90.0)
    assert p.contact == 0.0


def test_pack_default_choose_records_contact_scores():
    placements = pack([solid(2, 2), solid(2, 2)], (6, 6))
    by_piece = {p.piece: p for p in placements}
    assert by_piece[0].contact == 7.0  # 2x2 in a corner: 7 halo px on the border frame
    assert by_piece[1].contact == 6.0  # 4 border px + 2 px against piece 0


def test_pack_contact_places_second_piece_adjacent():
    placements = pack([solid(2, 2), solid(2, 2)], (6, 6))
    by_piece = {p.piece: p for p in placements}
    assert (by_piece[0].row, by_piece[0].col) == (0, 0)
    assert (by_piece[1].row, by_piece[1].col) == (0, 2)


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
```

Add `Placement` to the packer import block if missing.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_packer.py -q`
Expected: TypeError (`pack` has no `prerotated`/`order`/`validate`), AttributeError (`contact` field), assertion failures.

- [ ] **Step 3: Implement** — replace `Placement`, `pack`, and `_best_spot` in `src/plate_packer/packer.py`:

```python
@dataclass(frozen=True)
class Placement:
    piece: int  # index into the input piece list
    plate: int  # 0-based plate number
    row: int  # anchor (top-left) of the rotated mask on the plate
    col: int
    angle: float  # degrees CCW
    contact: float = 0.0  # chosen anchor's contact score (0.0 under bottom_left)
```

```python
def pack(
    pieces,
    plate_shape,
    rotations=1,
    plate_mask=None,
    choose=None,
    prerotated=None,
    order=None,
    validate=True,
):
    """Greedy-pack piece masks onto plates; spill to a new plate when full.

    plate_mask pre-encodes unusable plate regions as occupied pixels.
    prerotated (list of {angle: mask}) skips per-call rotation; order
    overrides the largest-area-first insertion order; validate=False skips
    the every-piece-fits-an-empty-plate check (improve() validates once).
    Raises ValueError if a piece cannot fit an empty plate at any rotation.
    """
    choose = choose or contact_first
    empty = plate_mask.copy() if plate_mask is not None else np.zeros(plate_shape, np.uint8)
    if prerotated is None:
        angles = [i * 360.0 / rotations for i in range(rotations)]
        prerotated = [{a: rotate_mask(p, a)[0] for a in angles} for p in pieces]
    rings = (
        [{a: contact_ring(m) for a, m in variants.items()} for variants in prerotated]
        if getattr(choose, "uses_contact", False)
        else None
    )

    if validate:
        for i, variants in enumerate(prerotated):
            if not any(_fits(empty, m) for m in variants.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")

    if order is None:
        order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True)
    plates: list[np.ndarray] = []
    placements: list[Placement] = []
    for i in order:
        target = plate_idx = None
        piece_rings = rings[i] if rings is not None else None
        for idx, occupancy in enumerate(plates):
            target = _best_spot(occupancy, prerotated[i], piece_rings, choose)
            if target:
                plate_idx = idx
                break
        if target is None:
            plates.append(empty.copy())
            plate_idx = len(plates) - 1
            target = _best_spot(plates[plate_idx], prerotated[i], piece_rings, choose)
        (row, col), angle, score = target
        mask = prerotated[i][angle]
        plates[plate_idx][row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
        placements.append(Placement(i, plate_idx, row, col, angle, score))
    return sorted(placements, key=lambda p: p.piece)
```

```python
def _best_spot(occupancy, variants, rings, choose):
    """Best (anchor, angle, contact) across rotations: highest contact, then
    lowest row/col; ties beyond that keep the earliest angle."""
    best = None  # (sort_key, anchor, angle, score)
    for angle, mask in variants.items():
        if mask.shape[0] > occupancy.shape[0] or mask.shape[1] > occupancy.shape[1]:
            continue
        legal = legal_placement_map(occupancy, mask)
        contact = (
            contact_map(occupancy, rings[angle]) if rings is not None else np.zeros(legal.shape)
        )
        anchor = choose(legal, contact)
        if anchor is None:
            continue
        score = float(contact[anchor])
        key = (-score, anchor[0], anchor[1])
        if best is None or key < best[0]:
            best = (key, anchor, angle, score)
    if best is None:
        return None
    return best[1], best[2], best[3]
```

Add after the chooser definitions:

```python
CHOOSERS = {"contact": contact_first, "bottom_left": bottom_left}
```

Note the definition-order constraint: `contact_first`/`bottom_left` must be defined before `CHOOSERS` and before `pack`'s default resolution (which happens at call time, so `pack` may stay above them in the file — keep the existing file layout and put `CHOOSERS` at the bottom).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q` — full suite. Pre-existing packer tests assert invariants (no overlap, spillover, sort order, rotation-enables-fit) and must pass unchanged under the new default chooser. If one fails, inspect whether it encodes bottom-left-specific anchors; only such a test may be updated (pass `choose=bottom_left` explicitly) — invariant tests must pass as-is.

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/packer.py tests/test_packer.py
git commit -m "feat: contact-scored placement; pack() prerotated/order/validate params"
```

---

### Task 3: Fitness (`improve.py`: `plate_fills`, `falkenauer`, `ImproveResult`)

**Files:**
- Create: `src/plate_packer/improve.py`
- Create: `tests/test_improve.py`

**Interfaces:**
- Consumes: `Placement` from `plate_packer.packer`.
- Produces: `plate_fills(placements, piece_px, usable_px) -> list[float]`.
- Produces: `falkenauer(fills: list[float]) -> float`.
- Produces: `ImproveResult` frozen dataclass: `placements: list[Placement]`, `evaluations: int`, `improvements: int`, `fitness_initial: float`, `fitness_final: float`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_improve.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -q`
Expected: ModuleNotFoundError (`plate_packer.improve`).

- [ ] **Step 3: Implement** — create `src/plate_packer/improve.py`:

```python
"""Iterated local search over the greedy insertion order (spec 2026-08-05).

Fitness is Falkenauer's grouping objective: mean squared plate fill. With
total piece area fixed, concentrating area on fewer/fuller plates always
scores higher, so plate count needs no separate objective term.
"""

import time
from dataclasses import dataclass

import numpy as np

from plate_packer.packer import Placement, contact_first, pack, rotate_mask

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
```

(`time`, `contact_first`, `pack`, `rotate_mask`, and the constants are consumed by Tasks 4-5; ruff will flag them as unused here — add `# noqa: F401` markers ONLY if ruff fails, and remove them in Task 5 when the symbols are used.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_improve.py -q` then `uv run ruff check .` (fix unused-import complaints per the note above).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: Falkenauer fitness and ImproveResult"
```

---

### Task 4: Moves (`perturb`, `shake`, and the five move functions)

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py` (append)

**Interfaces:**
- Consumes: Task 3's module; `Placement.contact` from Task 2.
- Produces: `perturb(order, placements, fills, rng) -> list[int]` and `shake(order, placements, fills, rng) -> list[int]`; both return NEW lists (never mutate the input).
- Internal: `_reinsert`, `_move_targeted_reinsert`, `_move_targeted_swap`, `_move_random_swap`, `_move_random_reinsert`, `_move_window_shuffle` — each `(order, placements, fills, rng) -> list[int]` except `_reinsert(order, i_pos, j_pos)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_improve.py`:

```python
from plate_packer.improve import (
    _move_random_reinsert,
    _move_random_swap,
    _move_targeted_reinsert,
    _move_targeted_swap,
    _move_window_shuffle,
    _reinsert,
    perturb,
    shake,
)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -q`
Expected: ImportError on the move functions.

- [ ] **Step 3: Implement** — append to `src/plate_packer/improve.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_improve.py -q && uv run ruff check .`

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: targeted and random ILS moves, perturb dispatcher, shake"
```

---

### Task 5: The ILS loop (`improve()`)

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py` (append)

**Interfaces:**
- Consumes: everything above; `pack(prerotated=, order=, validate=)` from Task 2.
- Produces: `improve(pieces, plate_shape, rotations=1, plate_mask=None, choose=None, budget_s=2700.0, min_improvement=0.005, patience=30, seed=0, validate=True, on_improve=None) -> ImproveResult`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_improve.py`:

```python
from plate_packer.improve import improve
from plate_packer.packer import pack

PIECES = [solid(2, 2), solid(2, 2), solid(2, 2)]


def test_improve_budget_zero_equals_plain_greedy():
    res = improve(PIECES, (6, 6), budget_s=0.0)
    assert res.placements == pack(PIECES, (6, 6))
    assert res.evaluations == 1
    assert res.improvements == 0
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


def test_improve_budget_stop_terminates():
    res = improve(PIECES, (6, 6), budget_s=0.2, patience=10**9, min_improvement=0.0)
    assert res.evaluations >= 1


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
    assert all(f2 > f1 for f1, f2 in zip(fits, fits[1:], strict=False))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -q`
Expected: ImportError (`improve` not defined).

- [ ] **Step 3: Implement** — append to `src/plate_packer/improve.py`:

```python
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
    budget_s=0 returns the plain greedy pack. Deterministic per seed.
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
```

The validation branch uses `_fits` — extend the module-top import to `from plate_packer.packer import Placement, _fits, contact_first, pack, rotate_mask`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_improve.py -q && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: improve() iterated local search with budget and stall stops"
```

---

### Task 6: Config knobs + CLI wiring + report line

**Files:**
- Modify: `src/plate_packer/config.py`
- Modify: `src/plate_packer/cli.py`
- Test: `tests/test_config.py` (append), `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `improve`, `ImproveResult` from Task 5; `CHOOSERS` from Task 2.
- Produces: `PackConfig` fields `improve_budget_s: float = 2700.0`, `min_improvement: float = 0.005`, `patience: int = 30`, `seed: int = 0`, `placement: str = "contact"` (all under `[packing]` in TOML).
- Produces: CLI `pack` options `--budget` (float, default None → config) and `--seed` (int, default None → config).

- [ ] **Step 1: Write the failing config tests** — append to `tests/test_config.py` (match the file's existing style for writing TOML to tmp_path):

```python
def test_improvement_defaults():
    cfg = PackConfig()
    assert cfg.improve_budget_s == 2700.0
    assert cfg.min_improvement == 0.005
    assert cfg.patience == 30
    assert cfg.seed == 0
    assert cfg.placement == "contact"


@pytest.mark.parametrize(
    "toml_body, bad_key",
    [
        ("[packing]\nimprove_budget_s = -1", "improve_budget_s"),
        ("[packing]\nmin_improvement = -0.1", "min_improvement"),
        ("[packing]\npatience = 0", "patience"),
        ('[packing]\nplacement = "wizard"', "placement"),
    ],
    ids=["negative-budget", "negative-min-improvement", "zero-patience", "unknown-placement"],
)
def test_improvement_knob_validation(tmp_path, toml_body, bad_key):
    p = tmp_path / "config.toml"
    p.write_text(toml_body, encoding="utf-8")
    with pytest.raises(ValueError, match=bad_key):
        load_config(p)


def test_improvement_knobs_load_from_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[packing]\nimprove_budget_s = 60\nmin_improvement = 0.01\n"
        'patience = 5\nseed = 42\nplacement = "bottom_left"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.improve_budget_s == 60.0
    assert cfg.min_improvement == 0.01
    assert cfg.patience == 5
    assert cfg.seed == 42
    assert cfg.placement == "bottom_left"
```

- [ ] **Step 2: Write the failing CLI tests** — append to `tests/test_cli.py`:

```python
def test_pack_budget_zero_is_plain_greedy(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "0"])
    assert result.exit_code == 0
    assert "improve:" not in result.output
    assert "improvement:" not in result.output


def test_pack_improvement_summary_in_report(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "5", "--seed", "1"])
    assert result.exit_code == 0
    assert "improvement:" in result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "improvement:" in report
    assert "evaluations" in report
```

(The default-config path also exercises `improve` — the stall stop keeps two-box runs to ~31 tiny evaluations, so no test-duration concern.)

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -q`
Expected: AttributeError / assertion failures on the new fields, unknown `--budget` option.

- [ ] **Step 4: Implement config** — in `src/plate_packer/config.py`, add fields to `PackConfig`:

```python
    improve_budget_s: float = 2700.0
    min_improvement: float = 0.005
    patience: int = 30
    seed: int = 0
    placement: str = "contact"
```

In `load_config`, extend the `PackConfig(...)` construction:

```python
improve_budget_s = (float(packing.get("improve_budget_s", PackConfig.improve_budget_s)),)
min_improvement = (float(packing.get("min_improvement", PackConfig.min_improvement)),)
patience = (int(packing.get("patience", PackConfig.patience)),)
seed = (int(packing.get("seed", PackConfig.seed)),)
placement = (str(packing.get("placement", PackConfig.placement)),)
```

In `_validate`, append:

```python
    if cfg.improve_budget_s < 0:
        raise ValueError("packing.improve_budget_s must be >= 0")
    if cfg.min_improvement < 0:
        raise ValueError("packing.min_improvement must be >= 0")
    if cfg.patience < 1:
        raise ValueError("packing.patience must be >= 1")
    if cfg.placement not in ("contact", "bottom_left"):
        raise ValueError('packing.placement must be "contact" or "bottom_left"')
```

- [ ] **Step 5: Implement CLI** — in `src/plate_packer/cli.py`:

Imports: add `from plate_packer.improve import improve` and extend the packer import to `from plate_packer.packer import CHOOSERS, legal_placement_map, pack, rotate_mask`.

New options on `pack_command` (after `force`):

```python
budget: float = (
    typer.Option(
        None, "--budget", help="improvement budget in seconds (0 = plain greedy; default: config)"
    ),
)
seed: int = (typer.Option(None, help="improvement search RNG seed (default: config)"),)
```

Replace Stage 3 (`placements = pack(...)`) with:

```python
    # Stage 3: pack (placements come back sorted by piece index). budget > 0
    # wraps greedy in the improvement search; per-piece validation already ran.
    choose = CHOOSERS[cfg.placement]
    budget_s = cfg.improve_budget_s if budget is None else budget
    seed_val = cfg.seed if seed is None else seed
    improve_line = None
    if budget_s > 0:
        res = improve(
            masks,
            plate_shape,
            rotations=cfg.rotations,
            plate_mask=plate_mask,
            choose=choose,
            budget_s=budget_s,
            min_improvement=cfg.min_improvement,
            patience=cfg.patience,
            seed=seed_val,
            validate=False,
            on_improve=lambda evals, plates, fit: typer.echo(
                f"  improve: eval {evals}: {plates} plate(s), fitness {fit:.4f}"
            ),
        )
        placements = res.placements
        improve_line = (
            f"improvement: {res.evaluations} evaluations, {res.improvements} improvements, "
            f"fitness {res.fitness_initial:.4f} -> {res.fitness_final:.4f}"
        )
    else:
        placements = pack(
            masks,
            plate_shape,
            rotations=cfg.rotations,
            plate_mask=plate_mask,
            choose=choose,
            validate=False,
        )
```

In Stage 5, after `lines.append(f"{len(piece_files)} pieces -> {len(plate_files)} plate(s)")` add:

```python
    if improve_line:
        lines.append(improve_line)
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 7: Commit**

```bash
git add src/plate_packer/config.py src/plate_packer/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: improvement config knobs and pack CLI wiring (--budget, --seed)"
```

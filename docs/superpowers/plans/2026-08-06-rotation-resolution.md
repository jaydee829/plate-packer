# Rotation & Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep contact scoring as the default while raising effective rotation granularity, by generating shape-aware angle candidates and running the ILS at a coarse resolution with only a small beam of orderings refined at fine resolution.

**Architecture:** A new `angles.py` derives per-piece rotation candidates from the convex hull (resolution-independent, computed once). `improve.py` is restructured into coarse-to-fine beam search: prerotate each piece at its shape-aware angles at two resolutions (fine + a block-max-downsampled coarse superset), run the existing targeted-move ILS at coarse resolution keeping a beam of the top-K distinct orderings, then fine-pack the beam and return the best fine result. Ordering seed becomes difficulty-first (area × elongation). Contact weighting gains an `edge_weight` knob.

**Tech Stack:** Python 3.11+, numpy, scipy (`signal.fftconvolve`), opencv-python (`cv2.convexHull`, `cv2.minAreaRect`, `cv2.minEnclosingCircle`). `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-08-06-rotation-resolution-design.md` (ADR-012). Read it for rationale; this plan is the build order.

## Global Constraints

Every task's requirements implicitly include this section.

- **Conservative-coverage invariant is sacred.** `rotate_mask` grows-never-shrinks; `conservative_downsample` (block max) grows-never-shrinks; the binary FFT legality gate (`legal_placement_map`) is unchanged. No change may introduce false free space. Coarse masks MUST be supersets of fine masks so coarse-legal ⇒ fine-legal.
- **Angles are resolution-independent** — computed once per piece from the mask hull and reused at both resolutions.
- **Tests are parametrized/atomic** (repo rule + user global rule): each case is its own named test via `pytest.mark.parametrize`, never a loop inside one test body. A failure must pinpoint the exact case.
- **Determinism** holds per seed *for a fixed coarse-evaluation count*; the wall-clock budget stays machine-dependent. Wall-clock-path tests use the existing `_FakeClock` in `tests/test_improve.py`.
- **New config knobs are validated** exactly like the existing ones in `config._validate`.
- **`ruff check` and `ruff format --check` must pass**; run the full `pytest` suite green before each commit.
- Run Python via `uv run` (e.g. `uv run pytest ...`, `uv run ruff ...`).
- Angle values are floats in `[0.0, 180.0)`; `0.0` is always present in a candidate list.

---

### Task 1: `angle_candidates` (new `angles.py`)

**Files:**
- Create: `src/plate_packer/angles.py`
- Test: `tests/test_angles.py`

**Interfaces:**
- Consumes: nothing from other tasks (`cv2`, `numpy`, `math` only).
- Produces: `angle_candidates(mask: np.ndarray, cap: int = 12, min_edge_frac: float = 0.1, safety_grid: int = 0) -> list[float]` — a de-duplicated, compactness-sorted list of rotation angles in degrees, `0.0` always first-or-present, capped at `cap`. Consumed by `improve()` (Task 5).

**Design notes (read before writing):**
- The sort key is the **analytic** axis-aligned bbox area of the hull points *rotated by the candidate angle* — NOT the bbox of a rasterized rotated mask. warpAffine interpolation inflates a re-rotated tilted raster and would make `0.0` win falsely; the analytic key correctly ranks the de-skewing angle as most compact. This was verified empirically (tilted rect: analytic AABB 47.9 at the de-skew angle vs 99.0 at 0°).
- Circle detection: hull area ≈ its min-enclosing-circle area (ratio > 0.90) ⇒ shape-derived angles collapse to just `{0.0}`. `safety_grid`, if set, still unions its uniform angles on top (so the mechanism is available even for round pieces).
- Angle for a hull edge with delta `(dx, dy)`: `(-degrees(atan2(dy, dx))) % 180`, plus that `+ 90) % 180` (flat side seats against bottom OR left border).
- Dedup: after collecting into a set (rounded to 6 dp), walk the value-sorted angles and drop any within `2.0°` of the previously kept one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_angles.py
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
        return (rot[:, 0].ptp()) * (rot[:, 1].ptp())

    cands = angle_candidates(tilted)
    assert analytic_aabb(tilted, cands[0]) < analytic_aabb(tilted, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_angles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plate_packer.angles'`.

- [ ] **Step 3: Write the implementation**

```python
# src/plate_packer/angles.py
"""Shape-aware rotation candidates: lay convex-hull edges parallel to the plate
axes (spec 2026-08-06, ADR-012). Resolution-independent — computed once per
piece from its mask hull and reused at every raster resolution."""

import math

import cv2
import numpy as np

_DEDUP_DEG = 2.0  # merge angles within this many degrees
_CIRCLE_RATIO = 0.90  # hull_area / min-enclosing-circle area above this => round


def _analytic_aabb_area(hull_pts: np.ndarray, angle_deg: float) -> float:
    """Axis-aligned bbox area of the hull points rotated by angle_deg. Analytic
    (no rasterization) so it is free of warpAffine interpolation growth."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = hull_pts @ np.array([[c, -s], [s, c]]).T
    return float((rot[:, 0].ptp()) * (rot[:, 1].ptp()))


def angle_candidates(
    mask: np.ndarray, cap: int = 12, min_edge_frac: float = 0.1, safety_grid: int = 0
) -> list[float]:
    """Rotation angles (deg, in [0,180)) that lay long hull edges parallel to a
    plate axis. 0.0 is always included; circle-like hulls collapse to [0.0]
    (plus any safety_grid angles). Sorted most-compact-first, capped at cap."""
    pts = np.argwhere(mask > 0)[:, ::-1].astype(np.int32)  # (x=col, y=row)
    if len(pts) == 0:
        return [0.0]
    hull = cv2.convexHull(pts)[:, 0, :].astype(np.float64)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    perimeter = float(lengths.sum())

    hull_area = cv2.contourArea(hull.astype(np.int32))
    (_, _), radius = cv2.minEnclosingCircle(pts)
    circle_area = math.pi * radius * radius
    is_circle = circle_area > 0 and hull_area / circle_area > _CIRCLE_RATIO

    angles = {0.0}
    if not is_circle:
        for (dx, dy), length in zip(edges, lengths, strict=True):
            if length < min_edge_frac * perimeter:
                continue
            base = (-math.degrees(math.atan2(dy, dx))) % 180
            angles.add(round(base, 6))
            angles.add(round((base + 90) % 180, 6))
    if safety_grid > 0:
        for i in range(safety_grid):
            angles.add(round((i * 360.0 / safety_grid) % 180, 6))

    deduped: list[float] = []
    for a in sorted(angles):
        if not deduped or abs(a - deduped[-1]) > _DEDUP_DEG:
            deduped.append(a)
    deduped.sort(key=lambda a: (_analytic_aabb_area(hull, a), a))
    return deduped[:cap]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_angles.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/angles.py tests/test_angles.py
uv run ruff format src/plate_packer/angles.py tests/test_angles.py
git add src/plate_packer/angles.py tests/test_angles.py
git commit -m "feat: shape-aware angle candidates (angles.py, ADR-012)"
```

---

### Task 2: `edge_weight` contact scaling + difficulty ordering (packer.py)

**Files:**
- Modify: `src/plate_packer/packer.py`
- Test: `tests/test_packer.py`

**Interfaces:**
- Consumes: existing `contact_map`, `contact_ring`, `_best_spot`, `pack`, `rotate_mask` from `packer.py`.
- Produces:
  - `contact_map(plate, ring, edge_weight=1.0)` — the border frame is padded with `constant_values=edge_weight` instead of `1`; occupancy pixels stay `1`.
  - `pack(..., edge_weight=1.0)` — threaded down to `contact_map` via `_best_spot`.
  - `seed_order(pieces, ordering="difficulty") -> list[int]` — insertion order; `"difficulty"` sorts by `area * elongation` descending, `"area"` reproduces the legacy largest-area-first order. Consumed by `improve()` (Task 5) and the CLI (Task 7).

**Design notes:**
- `elongation` = `long_side / short_side` of the piece's min-area bounding box (`cv2.minAreaRect`), `>= 1`, measured at the piece's canonical orientation (rotation-stable).
- `_best_spot` currently calls `contact_map(occupancy, rings[angle])`; add an `edge_weight` parameter to `_best_spot` and thread it into that call. `pack` gains `edge_weight=1.0` and passes it to `_best_spot`.
- Verified literals: with a single occupied pixel and a 1x1 ring, `edge_weight=1.0` gives corner=5 / top-edge=3 / interior=0; `edge_weight=2.0` gives corner=10 / top=6 / interior=0; a piece-piece side-neighbour contact stays `1` at any `edge_weight`. `seed_order`: `solid(2,18)` (elongation 17) precedes `solid(6,6)` (elongation 1) under `"difficulty"` and follows it under `"area"`; `solid(2,18)` precedes `solid(7,7)` (area 49 > 36) under `"difficulty"` because elongation dominates.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_packer.py

# --- edge_weight contact scaling (Task 2, ADR-012) ---

from plate_packer.packer import seed_order  # noqa: E402  (add to the top import block)


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_packer.py -k "edge_weight or seed_order" -v`
Expected: FAIL — `seed_order` import error and `contact_map()` rejecting the third positional arg.

- [ ] **Step 3: Write the implementation**

In `src/plate_packer/packer.py`:

Update `contact_map`:

```python
def contact_map(plate: np.ndarray, ring: np.ndarray, edge_weight: float = 1.0) -> np.ndarray:
    """Contact score at every anchor: halo pixels touching occupancy or the
    plate border. The border frame is weighted by edge_weight (occupancy stays
    weight 1). Same anchor coordinates/shape as legal_placement_map; np.rint
    collapses FFT noise so score ties are exact."""
    attraction = np.pad(plate, 1, constant_values=edge_weight)
    raw = fftconvolve(attraction.astype(np.float32), ring[::-1, ::-1].astype(np.float32), "valid")
    return np.rint(raw)
```

Thread `edge_weight` through `_best_spot` and `pack` (add the parameter to both signatures; pass it from `pack`'s loop into `_best_spot`, and from `_best_spot` into the `contact_map(occupancy, rings[angle], edge_weight)` call). Keep the default `1.0` so all existing callers are unaffected.

Add `seed_order` (place it near `pack`, before or after is fine):

```python
def _elongation(mask: np.ndarray) -> float:
    """long_side / short_side of the min-area bounding box (>= 1)."""
    pts = np.argwhere(mask > 0)[:, ::-1].astype(np.int32)
    (_, (w, h), _) = cv2.minAreaRect(pts)
    lo, hi = min(w, h), max(w, h)
    return hi / lo if lo > 0 else 1.0


def seed_order(pieces, ordering: str = "difficulty") -> list[int]:
    """Greedy insertion seed order. 'difficulty' = area * elongation descending
    (a long thin piece needs a long channel that only exists early); 'area' =
    legacy largest-area-first."""
    if ordering == "area":
        key = lambda i: float(pieces[i].sum())  # noqa: E731
    else:
        key = lambda i: float(pieces[i].sum()) * _elongation(pieces[i])  # noqa: E731
    return sorted(range(len(pieces)), key=key, reverse=True)
```

- [ ] **Step 4: Run the full packer suite to verify pass**

Run: `uv run pytest tests/test_packer.py -v`
Expected: PASS (new cases pass; existing contact/pack tests unaffected since `edge_weight` defaults to 1.0).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/packer.py tests/test_packer.py
uv run ruff format src/plate_packer/packer.py tests/test_packer.py
git add src/plate_packer/packer.py tests/test_packer.py
git commit -m "feat: contact_map edge_weight + difficulty seed_order (ADR-012)"
```

---

### Task 3: beam bookkeeping helper `_update_beam` (improve.py)

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py`

**Interfaces:**
- Produces: `_update_beam(beam, order, fitness, k) -> list[tuple[float, list[int]]]` — returns a NEW list of up to `k` `(fitness, order)` pairs, highest fitness first, ties broken deterministically by the ordering tuple, orderings kept distinct (a permutation seen twice keeps only its best fitness). Does not mutate `beam`. Consumed by `improve()` (Task 5).

**Design notes:**
- `order` is a list of piece indices (a permutation). Distinctness is by the permutation tuple.
- Re-inserting an existing ordering with a higher fitness updates it; with an equal/lower fitness it is a no-op on that ordering.
- Sort key `(-fitness, tuple(order))` — fitness descending, then ordering ascending for a total, deterministic order.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_improve.py
from plate_packer.improve import _update_beam  # noqa: E402 (add to top import block)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -k update_beam -v`
Expected: FAIL — `_update_beam` does not exist.

- [ ] **Step 3: Write the implementation**

Add to `src/plate_packer/improve.py`:

```python
def _update_beam(beam, order, fitness, k):
    """Return a new beam of up to k (fitness, order) pairs, best fitness first,
    orderings distinct, ties broken by ordering. Input beam is not mutated."""
    best = {tuple(o): f for f, o in beam}
    key = tuple(order)
    if key not in best or fitness > best[key]:
        best[key] = fitness
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(f, list(o)) for o, f in ranked[:k]]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_improve.py -k update_beam -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/improve.py tests/test_improve.py
uv run ruff format src/plate_packer/improve.py tests/test_improve.py
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: top-K distinct-ordering beam helper (ADR-012)"
```

---

### Task 4: two-resolution prerotation helper `_prerotate_multi_res` (improve.py)

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py`

**Interfaces:**
- Consumes: `rotate_mask` (packer.py), `conservative_downsample` (loading.py).
- Produces: `_prerotate_multi_res(pieces, piece_angles, factor) -> tuple[list[dict], list[dict]]` — `(fine_prerotated, coarse_prerotated)`, each a list (one dict per piece) mapping `angle -> mask`. Fine masks are `rotate_mask(piece, angle)[0]`; coarse masks are `conservative_downsample(fine_mask, factor)` — block-max supersets, guaranteeing coarse-legal ⇒ fine-legal. Consumed by `improve()` (Task 5).

**Design notes:**
- Coarse variants are the block-max downsample of the **already-rotated fine** mask, not a rotate of a downsampled mask — this is what guarantees the superset relation per angle.
- `factor` is `round(coarse_res_mm / working_res_mm)`, an integer `>= 1`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_improve.py
from plate_packer.improve import _prerotate_multi_res  # noqa: E402 (top import block)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -k prerotate -v`
Expected: FAIL — `_prerotate_multi_res` does not exist.

- [ ] **Step 3: Write the implementation**

Add the import and function to `src/plate_packer/improve.py`:

```python
from plate_packer.loading import conservative_downsample
```

```python
def _prerotate_multi_res(pieces, piece_angles, factor):
    """Per-piece {angle: mask} at fine resolution and at a block-max-downsampled
    coarse resolution. Coarse masks are supersets of fine (coarse-legal =>
    fine-legal)."""
    fine, coarse = [], []
    for piece, angles in zip(pieces, piece_angles, strict=True):
        fvar = {a: rotate_mask(piece, a)[0] for a in angles}
        cvar = {a: conservative_downsample(m, factor) for a, m in fvar.items()}
        fine.append(fvar)
        coarse.append(cvar)
    return fine, coarse
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_improve.py -k prerotate -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/improve.py tests/test_improve.py
uv run ruff format src/plate_packer/improve.py tests/test_improve.py
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: two-resolution superset prerotation helper (ADR-012)"
```

---

### Task 5: coarse-to-fine `improve()` restructure (improve.py)

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py`

**Interfaces:**
- Consumes: `angle_candidates` (Task 1), `seed_order`/`pack`/`contact_first`/`_fits`/`rotate_mask` (packer.py), `_update_beam` (Task 3), `_prerotate_multi_res` (Task 4), `conservative_downsample` (loading.py), existing move/`perturb`/`shake`/`falkenauer`/`plate_fills`.
- Produces: restructured `improve(...)` and `ImproveResult` with a new `beam` field.

**New `improve` signature** (replaces the current one; `rotations` is dropped — the search now uses shape-aware angles):

```python
def improve(
    pieces,
    plate_shape,
    plate_mask=None,
    choose=None,
    budget_s=2700.0,
    min_improvement=0.005,
    patience=30,
    seed=0,
    working_res_mm=0.1,
    coarse_res_mm=0.4,
    beam=5,
    angle_cap=12,
    min_edge_frac=0.1,
    safety_grid=0,
    edge_contact_weight=1.0,
    ordering="difficulty",
    validate=True,
    on_improve=None,
):
```

**New `ImproveResult`:**

```python
@dataclass(frozen=True)
class ImproveResult:
    placements: list[Placement]
    evaluations: int  # coarse evaluations (the search effort)
    improvements: int
    fitness_initial: float  # fine fitness of the difficulty-seed order
    fitness_final: float  # best fine fitness among {seed} ∪ beam
    beam: list  # (coarse_fitness, fine_fitness, n_plates) per beam survivor, best fine first
```

**Algorithm (spec §2):**
1. `factor = round(coarse_res_mm / working_res_mm)`.
2. `piece_angles = [angle_candidates(p, cap=angle_cap, min_edge_frac=min_edge_frac, safety_grid=safety_grid) for p in pieces]`.
3. `fine_prerot, coarse_prerot = _prerotate_multi_res(pieces, piece_angles, factor)`.
4. Build fine and coarse plate masks: `empty_fine = plate_mask.copy() or zeros(plate_shape)`; `coarse_plate_mask = conservative_downsample(empty_fine, factor)`; `coarse_shape = coarse_plate_mask.shape`. Pass `coarse_plate_mask` as the coarse `plate_mask` (an all-zero mask behaves as None).
5. Fine metrics: `fine_piece_px = [int(p.sum()) for p in pieces]`; `fine_usable = plate_shape[0]*plate_shape[1] - int(empty_fine.sum())`. Coarse metrics: `coarse_piece_px = [int(coarse_prerot[i][a].sum()) for i, a in first-angle]` — use the piece's 0.0 coarse mask if present, else the first angle (area is angle-invariant for fill bias per ADR-011). Simplest correct choice: `coarse_piece_px = [int(next(iter(coarse_prerot[i].values())).sum()) for i in range(n)]`; `coarse_usable = coarse_shape[0]*coarse_shape[1] - int(coarse_plate_mask.sum())`.
6. If `validate`: run the one-time fit check against `empty_fine` using the **fine** variants (raise the documented `ValueError` if a piece fits no empty plate at any candidate angle).
7. Coarse evaluation closure:

```python
def eval_coarse(order):
    result = pack(pieces, coarse_shape, plate_mask=coarse_plate_mask, choose=choose,
                  prerotated=coarse_prerot, order=order, validate=False,
                  edge_weight=edge_contact_weight)
    return result, falkenauer(plate_fills(result, coarse_piece_px, coarse_usable))
```

8. Seed order `seed_ord = seed_order(pieces, ordering)`. Run the **existing ILS loop** on `eval_coarse` (identical structure to the current one: initial eval, `while time.monotonic() - start < budget_s`, `perturb`, accept-if-better, `shake` after `SHAKE_AFTER` fails, stall stop via `patience`/`min_improvement`). Track `evaluations`, `improvements`. On every accepted best AND on the initial eval, update the beam: `beam_list = _update_beam(beam_list, best_order, best_fit, beam)`. (Updating only on new bests is sufficient and keeps the beam to the best distinct orderings.)
9. Fine refinement closure:

```python
def fine_pack(order):
    result = pack(pieces, plate_shape, plate_mask=plate_mask, choose=choose,
                  prerotated=fine_prerot, order=order, validate=False,
                  edge_weight=edge_contact_weight)
    fit = falkenauer(plate_fills(result, fine_piece_px, fine_usable))
    return result, fit
```

10. Baseline: `seed_result, seed_fit = fine_pack(seed_ord)`; `fitness_initial = seed_fit`.
11. Fine-pack each beam ordering; assemble `candidates = [(seed coarse-fit-or-None, seed_fit, seed_result)] + [(coarse_fit, fine_fit, result) for each beam member]`. Pick `best` by max fine fitness (so `fitness_final >= fitness_initial` always — anytime guarantee holds at fine resolution). `placements = best.result`.
12. `beam` field = `[(coarse_fit, fine_fit, n_plates) for each beam member]` sorted by fine fitness descending (best first); `n_plates = max(p.plate for p in result) + 1`.
13. `on_improve(evaluations, n_plates, fitness)` still fires on each new coarse best (unchanged behaviour, coarse fitness).

**Existing-test updates (in the same commit):** the semantics of `improve()` change (coarse-to-fine, shape-aware angles, no `rotations`), so update these existing tests to the new contract while preserving their intent:
- `test_improve_budget_zero_equals_plain_greedy`: budget 0 now returns the fine pack of the difficulty-seed order with shape-aware angles. Replace the equality target with an explicit fine pack:

```python
def test_improve_budget_zero_equals_greedy_seed_pack():
    pieces = PIECES
    res = improve(pieces, (6, 6), budget_s=0.0)
    angles = [angle_candidates(p) for p in pieces]
    prerot = [{a: rotate_mask(p, a)[0] for a in ang} for p, ang in zip(pieces, angles)]
    order = seed_order(pieces, "difficulty")
    expected = pack(pieces, (6, 6), prerotated=prerot, order=order, validate=False)
    assert res.placements == expected
    assert res.evaluations == 1
    assert res.fitness_initial == res.fitness_final
```

- `test_improve_budget_stop_terminates_before_any_iteration`, `test_improve_wall_clock_eval_count_follows_time_schedule`, `test_improve_stall_stop_counts_evaluations`, `test_improve_deterministic_for_fixed_eval_count`, `test_improve_same_seed_same_result`, `test_improve_fitness_never_worsens`, `test_improve_on_improve_reports_increasing_fitness`: keep as-is except remove any `rotations=` argument (none currently pass it) and confirm they still assert on `evaluations`/determinism/monotonicity, which are preserved (1 initial + N loop coarse evals). Add the imports `angle_candidates`, `seed_order`, `rotate_mask` where needed. These tests must still pass unchanged in count logic.

- [ ] **Step 1: Write the new/updated failing tests**

```python
# Append to tests/test_improve.py
from plate_packer.angles import angle_candidates  # noqa: E402 (top import block)
from plate_packer.packer import rotate_mask, seed_order  # noqa: E402 (extend existing import)


def test_improve_result_has_beam_field():
    res = improve(PIECES, (6, 6), budget_s=0.0)
    assert isinstance(res.beam, list)
    for coarse_fit, fine_fit, n_plates in res.beam:
        assert isinstance(n_plates, int)


def test_improve_returns_best_fine_not_best_coarse():
    # fitness_final is a FINE fitness and never below the seed's fine fitness.
    res = improve(_WALL_PIECES, (10, 10), budget_s=0.4, patience=40,
                  min_improvement=0.0, seed=5)
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
    res = improve(_WALL_PIECES, (10, 10), budget_s=0.3, patience=40,
                  min_improvement=0.0, seed=5, beam=3)
    assert len(res.beam) <= 3
```

Also apply the existing-test updates described above (rewrite `test_improve_budget_zero_equals_plain_greedy` as `test_improve_budget_zero_equals_greedy_seed_pack`, drop any `rotations=` kwargs).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve.py -v`
Expected: FAIL — `ImproveResult` has no `beam`; the new signature/semantics not yet present.

- [ ] **Step 3: Write the implementation**

Restructure `improve()` and `ImproveResult` per the Algorithm section above. Preserve the ILS loop body verbatim (moves, accept, shake, stall) but drive it through `eval_coarse`, and update the beam on each new best. Keep the docstring's determinism paragraph (per seed for a fixed coarse-evaluation count; wall-clock machine-dependent), updating "evaluations" to say coarse evaluations and documenting that `budget_s=0` returns the fine pack of the difficulty-seed order.

- [ ] **Step 4: Run the full improve suite to verify pass**

Run: `uv run pytest tests/test_improve.py -v`
Expected: PASS (new + updated cases).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/improve.py tests/test_improve.py
uv run ruff format src/plate_packer/improve.py tests/test_improve.py
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: coarse-to-fine beam search in improve() (ADR-012)"
```

---

### Task 6: config knobs (config.py)

**Files:**
- Modify: `src/plate_packer/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces new `PackConfig` fields (all under `[packing]`): `coarse_res_mm: float = 0.4`, `beam: int = 5`, `angle_cap: int = 12`, `min_edge_frac: float = 0.1`, `safety_grid: int = 0`, `edge_contact_weight: float = 1.0`, `ordering: str = "difficulty"`. Consumed by the CLI (Task 7).

**Validation constraints (add to `_validate`):**
- `coarse_res_mm >= working_res_mm` AND `coarse_res_mm / working_res_mm` is (within `RES_RATIO_TOL`) an integer. Message: `packing.coarse_res_mm must be an integer multiple of working_res_mm and >= it`.
- `beam >= 1`; `angle_cap >= 1`; `0 < min_edge_frac <= 1`; `safety_grid >= 0`; `edge_contact_weight >= 0`; `ordering in {"difficulty", "area"}`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_config.py  (match the file's existing import/write-config style)


def test_config_defaults_include_coarse_to_fine_knobs(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.coarse_res_mm == 0.4
    assert cfg.beam == 5
    assert cfg.angle_cap == 12
    assert cfg.min_edge_frac == 0.1
    assert cfg.safety_grid == 0
    assert cfg.edge_contact_weight == 1.0
    assert cfg.ordering == "difficulty"


def test_config_reads_coarse_to_fine_knobs(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[packing]\n"
        "coarse_res_mm = 0.5\n"
        "beam = 8\n"
        "angle_cap = 6\n"
        "min_edge_frac = 0.2\n"
        "safety_grid = 12\n"
        "edge_contact_weight = 2.0\n"
        'ordering = "area"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert (cfg.coarse_res_mm, cfg.beam, cfg.angle_cap) == (0.5, 8, 6)
    assert (cfg.min_edge_frac, cfg.safety_grid, cfg.edge_contact_weight) == (0.2, 12, 2.0)
    assert cfg.ordering == "area"


@pytest.mark.parametrize(
    "key, match",
    [
        ("coarse_res_mm = 0.05", "coarse_res_mm"),  # below working_res_mm (0.1)
        ("coarse_res_mm = 0.35", "coarse_res_mm"),  # not an integer multiple of 0.1
        ("beam = 0", "beam"),
        ("angle_cap = 0", "angle_cap"),
        ("min_edge_frac = 0", "min_edge_frac"),
        ("min_edge_frac = 1.5", "min_edge_frac"),
        ("safety_grid = -1", "safety_grid"),
        ("edge_contact_weight = -0.5", "edge_contact_weight"),
        ('ordering = "spiral"', "ordering"),
    ],
    ids=[
        "coarse-below-working", "coarse-not-multiple", "beam-zero", "cap-zero",
        "edge-frac-zero", "edge-frac-gt-one", "safety-negative",
        "edge-weight-negative", "ordering-unknown",
    ],
)
def test_config_rejects_invalid_coarse_to_fine_knobs(tmp_path, key, match):
    p = tmp_path / "config.toml"
    p.write_text(f"[packing]\n{key}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(p)
```

(If `tests/test_config.py` does not already import `pytest`/`Path`/`load_config`, add them to match the existing header.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -k "coarse_to_fine or knobs" -v`
Expected: FAIL — fields absent, no validation.

- [ ] **Step 3: Write the implementation**

Add the seven fields to the `PackConfig` dataclass, read them in `load_config` from the `packing` table (using `PackConfig.<field>` as the default, casting like the neighbours), and add the validation block to `_validate`:

```python
    ratio = cfg.coarse_res_mm / cfg.working_res_mm
    if cfg.coarse_res_mm < cfg.working_res_mm or abs(ratio - round(ratio)) > RES_RATIO_TOL:
        raise ValueError(
            "packing.coarse_res_mm must be an integer multiple of working_res_mm and >= it"
        )
    if cfg.beam < 1:
        raise ValueError("packing.beam must be >= 1")
    if cfg.angle_cap < 1:
        raise ValueError("packing.angle_cap must be >= 1")
    if not (0 < cfg.min_edge_frac <= 1):
        raise ValueError("packing.min_edge_frac must be in (0, 1]")
    if cfg.safety_grid < 0:
        raise ValueError("packing.safety_grid must be >= 0")
    if cfg.edge_contact_weight < 0:
        raise ValueError("packing.edge_contact_weight must be >= 0")
    if cfg.ordering not in ("difficulty", "area"):
        raise ValueError('packing.ordering must be "difficulty" or "area"')
```

`RES_RATIO_TOL` is already imported in `config.py`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/config.py tests/test_config.py
uv run ruff format src/plate_packer/config.py tests/test_config.py
git add src/plate_packer/config.py tests/test_config.py
git commit -m "feat: coarse-to-fine config knobs (ADR-012)"
```

---

### Task 7: CLI wiring (cli.py)

**Files:**
- Modify: `src/plate_packer/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `angle_candidates` (Task 1), `seed_order` (Task 2), the restructured `improve()` (Task 5), the new config fields (Task 6).
- Produces: `pack` CLI gains `--coarse-res` and `--beam` options; the improvement call passes the new knobs; the Stage-2 fit-check and the `budget == 0` greedy path both use shape-aware angles.

**Changes:**
1. Import `angle_candidates` from `plate_packer.angles`; import `seed_order` from `plate_packer.packer` (extend the existing import line).
2. Add two options to `pack_command`:

```python
    coarse_res: float = typer.Option(
        None, "--coarse-res", help="coarse search resolution mm/px (default: config)"
    ),
    beam: int = typer.Option(None, "--beam", help="fine-refinement beam width (default: config)"),
```

3. **Stage 2 fit-check** currently uses a uniform `angles` grid. Replace the per-piece check so it uses shape-aware candidates:

```python
    cand = angle_candidates(
        mask, cap=cfg.angle_cap, min_edge_frac=cfg.min_edge_frac, safety_grid=cfg.safety_grid
    )
    fits = any(_fits(plate_mask, rotate_mask(mask, a)[0]) for a in cand)
```

Remove the now-unused module-level `angles = [...]` uniform grid (Stage 1) if nothing else consumes it.

4. **Stage 3 improve call**: pass the new knobs and drop `rotations`:

```python
        res_improve = improve(
            masks,
            plate_shape,
            plate_mask=plate_mask,
            choose=choose,
            budget_s=budget_s,
            min_improvement=cfg.min_improvement,
            patience=cfg.patience,
            seed=seed_val,
            working_res_mm=res,
            coarse_res_mm=(cfg.coarse_res_mm if coarse_res is None else coarse_res),
            beam=(cfg.beam if beam is None else beam),
            angle_cap=cfg.angle_cap,
            min_edge_frac=cfg.min_edge_frac,
            safety_grid=cfg.safety_grid,
            edge_contact_weight=cfg.edge_contact_weight,
            ordering=cfg.ordering,
            validate=False,
            on_improve=lambda evals, plates, fit: typer.echo(
                f"  improve: eval {evals}: {plates} plate(s), fitness {fit:.4f}"
            ),
        )
```

5. **Stage 3 greedy path** (`budget_s <= 0`): build shape-aware prerotated variants and the difficulty seed order, then call `pack` with them (so the plain-greedy path matches the search's angle set and ordering):

```python
    else:
        prerot = [
            {
                a: rotate_mask(masks[i], a)[0]
                for a in angle_candidates(
                    masks[i], cap=cfg.angle_cap, min_edge_frac=cfg.min_edge_frac,
                    safety_grid=cfg.safety_grid,
                )
            }
            for i in range(len(masks))
        ]
        placements = pack(
            masks,
            plate_shape,
            plate_mask=plate_mask,
            choose=choose,
            prerotated=prerot,
            order=seed_order(masks, cfg.ordering),
            validate=False,
            edge_weight=cfg.edge_contact_weight,
        )
```

Stage 4/5/6 are unchanged (they re-derive transforms via `rotate_mask(masks[pl.piece], pl.angle)`, which works for any angle the packer chose). The `res` variable in Stages 4/5 (`cfg.working_res_mm`) is untouched — do not shadow it.

- [ ] **Step 1: Write the failing test**

Extend `tests/test_cli.py`, reusing its module-level `runner` (`CliRunner()`) and `_setup(tmp_path, monkeypatch)` helper (writes `config.toml` + two 10×10×5 boxes under `tmp_path/models`, chdirs into `tmp_path`; plates land in `tmp_path/plates`). Add:

```python
def test_pack_cli_greedy_uses_shape_aware_angles(tmp_path, monkeypatch):
    # A budget-0 pack still succeeds end-to-end (report written, verify ok) with
    # the shape-aware greedy path.
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "0"])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "plate(s)" in report


def test_pack_cli_accepts_coarse_res_and_beam_options(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["pack", str(src), "--budget", "1", "--coarse-res", "0.4", "--beam", "2"]
    )
    assert result.exit_code == 0, result.output
```

The existing `test_pack_budget_zero_is_plain_greedy` and `test_pack_improvement_summary_in_report` must still pass unchanged (`_setup`'s `config.toml` sets `rotations = 1`, but the coarse-to-fine path ignores `rotations`; `coarse_res_mm` defaults to 0.4 → factor 4 over the 0.1 mm working res).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -k "shape_aware or coarse_res" -v`
Expected: FAIL — options not defined / greedy path not yet shape-aware.

- [ ] **Step 3: Implement the CLI changes** per the Changes list above.

- [ ] **Step 4: Run the full suite to verify pass**

Run: `uv run pytest -v`
Expected: PASS (entire suite green).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/plate_packer/cli.py tests/test_cli.py
uv run ruff format src/plate_packer/cli.py tests/test_cli.py
git add src/plate_packer/cli.py tests/test_cli.py
git commit -m "feat: wire shape-aware angles + coarse-to-fine into the pack CLI (ADR-012)"
```

---

## Self-Review (completed against the spec)

- **§1 shape-aware angles** → Task 1 (all bullets: hull edges, +90, dedup, compactness sort, circle collapse, always-0, safety_grid, resolution-independent). Sort key corrected to analytic hull AABB (verified) so "compact orientations first" is real, not a rasterization artifact.
- **§2 coarse-to-fine beam** → Tasks 3 (beam), 4 (two-res superset prerotation), 5 (restructure, `ImproveResult.beam`, observability, coarse-legal⇒fine-legal, best-fine return, determinism).
- **§3 difficulty ordering** → Task 2 `seed_order`, wired in Tasks 5 & 7.
- **§4 contact weighting** → Task 2 `contact_map(edge_weight)`, threaded through `pack` and into `improve`/CLI.
- **§5 packer changes** → Task 2 (`edge_weight` only; `pack` already accepts `prerotated`/`order`).
- **§6 config surface** → Task 6 (all seven knobs + constraints); CLI `--coarse-res`/`--beam` → Task 7.
- **§7 deferred** → not built (correct).
- **§8 success criteria** → unit/synthetic coverage lands here; the empirical ≤4-plates / fitness ≥0.4785 measurement is a post-merge benchmark on the user's machine (documented; cannot run in this environment).
- **§9 testing** → every listed test type has a home (angle shapes, block-max growth, coarse-legal⇒fine-legal, beam top-K distinct + ties, difficulty vs area, `edge_weight` scaling, determinism with `_FakeClock`, E2E via CLI verify).

**Type consistency:** `angle_candidates(mask, cap, min_edge_frac, safety_grid)`, `seed_order(pieces, ordering)`, `contact_map(plate, ring, edge_weight)`, `pack(..., edge_weight)`, `_update_beam(beam, order, fitness, k)`, `_prerotate_multi_res(pieces, piece_angles, factor)`, and the new `improve` signature are used identically everywhere they appear across tasks.

**All test literals were verified against real cv2/numpy/scipy output via a probe before writing** (square/rect/circle/L angle sets; cap; min_edge_frac; safety_grid union; edge_weight border-vs-piece scaling; difficulty vs area orderings; block-max superset).

# Packing Improvement Design — Contact-Scored Placement + Targeted-Move ILS

**Date:** 2026-08-05
**Research basis:** `docs/research/2026-08-05-packing-methods-survey.md` (tiers 1+2)
**Baseline:** greedy bottom-left first-fit, 54–62% occupancy on the 30-piece
Tome of Demons shakedown.

## 1. Objective: Falkenauer fitness (THE objective, not a tie-break)

```
fitness = (1/n) · Σᵢ fillᵢ²          maximize
fillᵢ   = (Σ dilated-mask pixels of pieces on plate i) / usable_px
usable_px = plate_h·plate_w − plate_mask.sum()   (working-resolution pixels)
```

- Piece pixel counts are exact sums of the prepared (dilated) masks — placements
  never overlap, so per-plate fill is additive; no occupancy grids needed to
  score a candidate.
- Fewer plates dominates: with total piece area fixed and greedy never opening
  an unneeded plate, dropping a plate always raises fitness. Plate count is
  reported to the user but is **not** a separate objective term.
- Fitness is dimensionless and piece-count independent — comparable across runs.

## 2. Contact-scored placement (constructor upgrade, always on by default)

Legality is untouched: the binary FFT legality map still gates every placement
(conservative-coverage invariant preserved; merged-shadow self-check unaffected).
Scoring only ranks the provably legal anchors.

**Contact ring.** For a rotated piece mask `m` (uint8, tight-cropped):

```python
padded = np.pad(m, 1)  # (h+2, w+2)
ring = cv2.dilate(padded, np.ones((3, 3), np.uint8)) - padded
```

**Contact map.** With plate occupancy `occ` (H, W) — which already includes the
plate_mask's unusable regions as occupied:

```python
attraction = np.pad(occ, 1, constant_values=1)  # 1px frame = plate-edge contact
raw = fftconvolve(attraction.astype(np.float32), ring[::-1, ::-1].astype(np.float32), "valid")
contact = np.rint(raw)  # (H-h+1, W-w+1) — same shape as legality map
```

`contact[r, c]` = count of the piece's 1-px halo pixels touching occupied pixels
or the plate border when anchored at (r, c). Alignment check: anchor (r, c)
places the piece at rows r..r+h−1; its halo spans r−1..r+h, which is exactly
rows r..r+h+1 of the padded attraction — the "valid" correlation lines up with
the legality map's anchor coordinates one-for-one. `np.rint` collapses FFT noise
(~1e-7) so score ties are exact and platform-stable.

Because prepared masks are dilated by spacing/2 per ADR-010, "halos touching"
means pieces sit at exactly the minimum legal gap.

**Chooser interface change.** `choose(legal) -> anchor | None` becomes
`choose(legal, contact) -> anchor | None`:

- `contact_first` (new default): among legal anchors, maximize contact;
  tie-break lowest row, then lowest column. Implementation:
  `np.argmax(np.where(legal, contact, -1.0))` — argmax's first-occurrence rule
  in row-major order IS the bottom-left tie-break.
- `bottom_left` (kept, config-selectable): ignores `contact`, current behavior.

**Rotation choice.** `_best_spot` compares candidates across rotations by
`(-contact_at_anchor, row, col)` lexicographic minimum; ties beyond that keep
the earliest angle in the angle list (deterministic). This makes rotation
selection snugness-driven, not just position-driven.

**Cost:** one extra fftconvolve per (rotation × placement attempt) — roughly 2×
pack time, reclaimed by rotation caching (§3).

## 3. Search: targeted-move iterated local search over insertion order

**State** = a permutation of piece indices (the greedy insertion order).
**Evaluator** = the greedy packer itself, unchanged semantics.

### pack() refactor (enables cheap repacks)

`pack(pieces, plate_shape, rotations=1, plate_mask=None, choose=None,
prerotated=None, order=None, validate=True)`:

- `prerotated`: `list[dict[angle, mask]]` — pass to skip per-call rotation.
  `None` → computed internally (current behavior).
- `order`: explicit insertion order; `None` → area-sorted descending (current
  behavior).
- `validate`: the every-piece-fits-empty-plate check; `improve()` validates once
  and passes `False` for repacks (the check is FFT-heavy).
- `Placement` gains `contact: float = 0.0` — the chosen anchor's contact score
  (0.0 under `bottom_left`). Appended field with default: existing positional
  constructions stay valid.

### ILS loop (`improve()` in new module `src/plate_packer/improve.py`)

```
incumbent_order = area-sorted           # evaluated first = today's baseline
best = pack(incumbent_order); marker = fitness(best); evals_since_marker = 0
while elapsed < budget_s:
    candidate = perturb(incumbent_order, best_result, rng)
    result = pack(candidate, prerotated=..., validate=False)
    evals_since_marker += 1
    if fitness(result) > fitness(best):
        best, incumbent_order = result, candidate
        fails = 0
        if fitness(best) − marker ≥ min_improvement:
            marker = fitness(best); evals_since_marker = 0
    else:
        fails += 1
        if fails ≥ SHAKE_AFTER (20): incumbent_order = shake(best_order, rng); fails = 0
    if evals_since_marker ≥ patience: break        # stall stop
return ImproveResult(placements=best, evaluations, improvements,
                     fitness_initial, fitness_final)
```

**Moves** (probabilities fixed in v1; rng-driven):

| p | move | detail |
|---|------|--------|
| 0.45 | targeted reinsert | random piece from the min-fill plate → uniformly random earlier position in the order |
| 0.25 | targeted swap | lowest-contact piece among a 5-piece random sample ↔ uniformly random other piece |
| 0.15 | random swap | two uniform random positions |
| 0.10 | random reinsert | uniform random piece → uniform random position |
| 0.05 | window shuffle | shuffle a random 3-length contiguous window |

Targeted moves use the *best result's* placements (min-fill plate membership,
per-placement contact scores). Degenerate cases (1 plate → no "min-fill
donor" distinct from others; sample larger than piece count) fall back to the
corresponding random move.

**Shake** = 4 stacked random-uniform moves applied to the best-known order.

**Stopping:** whichever comes first — wall-clock budget (`time.monotonic`),
or stall (`patience` evaluations without `min_improvement` cumulative fitness
gain since the marker). `budget_s = 0` skips the loop entirely: the result is
the plain greedy pack (identical placements, evaluations = 1).

**Determinism:** all randomness from `numpy.random.default_rng(seed)`. Same
inputs + config ⇒ identical output **for a fixed evaluation count**. The stall
stop is deterministic; the wall-clock budget is not (eval count is
machine-dependent), so a budget-bounded run can differ across hardware for the
same seed. Reproducible runs set `budget_s` high so the stall stop wins.

**Observability:** optional `on_improve(evaluations, n_plates, fitness)`
callback fired at each new best; CLI echoes these live. Final report gains:
`improvement: E evaluations, I improvements, fitness F0 -> F1`.

## 4. Config & CLI

`PackConfig` additions (validated like existing knobs):

| knob | default | constraint |
|------|---------|------------|
| `improve_budget_s` | 2700.0 | ≥ 0 |
| `min_improvement` | 0.005 | ≥ 0 |
| `patience` | 30 | ≥ 1 |
| `seed` | 0 | int |
| `placement` | `"contact"` | `{"contact", "bottom_left"}` |

CLI `pack` gains `--budget` (seconds, overrides config) and `--seed`. Flow:
after validation, `improve()` when effective budget > 0, else `pack()`. Verify
stage unchanged (re-rotates masks for occupancy exactly as today).

## 5. Rejected alternatives (recorded so we don't relitigate)

- **A\*/branch-and-bound:** requires an admissible bound on completions of a
  partial layout; the only easy one (area bound, 100%-density completion) is
  far too loose for irregular shapes to prune anything — the tree stays ~n!
  wide. Exact-methods literature stalls at ~10–27 polygons/hour; we have 30+.
  Falkenauer fitness is a quality score, not a bound, and cannot make A*
  admissible.
- **BRKGA (PAMPA-faithful):** wants thousands of evaluations; a 45-min budget
  at ~1 min/repack yields 50–200. Revisit if the inner loop gets ~10× faster.
- **Beam search constructor:** partial-layout fitness is myopic, memory grows
  with beam width, and the contact-scored constructor is already strong.
- **Overlap-minimizing layout search (tier 3):** state of the art but a
  different architecture (penetration maps, incremental moves); deferred to
  v2+ per the survey.

## 6. Tests (all case-driven tests parametrized/atomic per repo rule)

- **Ring:** known shapes (single pixel, 2×2 square, L-shape) → exact expected
  ring masks.
- **Contact map:** tiny hand-computed cases — empty plate (border-only contact),
  one placed piece (adjacency counts), values at specific anchors.
- **Chooser:** `contact_first` picks max-contact anchor; bottom-left tie-break
  among equal scores; returns None when nothing legal; `bottom_left` ignores
  contact.
- **Rotation choice:** a piece whose 90° rotation nestles better is placed
  rotated even when 0° is legal.
- **Fitness:** hand-computed fills → exact value; concentration property
  ((.75,.74,.73,.11) beats (.62,.61,.54,.56)); fewer plates beats more for the
  same pieces.
- **pack() params:** `prerotated`/`order`/`validate=False` produce identical
  results to the defaults they bypass.
- **Moves:** each move type yields a valid permutation; targeted moves pick from
  the documented populations; degenerate fallbacks.
- **ILS:** budget 0 ⇒ result identical to plain `pack()`; same seed ⇒ identical
  result; best fitness monotonically non-decreasing across improvements; stall
  stop fires (tiny patience, min_improvement > any possible gain); budget stop
  fires.
- **Config/CLI:** knob validation errors name the bad key; `--budget 0` runs
  pure greedy; report contains the improvement summary line.
- **E2E:** small synthetic piece set — improved fitness ≥ greedy fitness, all
  placements pass the merged-shadow self-check.

## 7. Out of scope (v1 of this milestone)

Shape-aware angle pruning (compute-only win — later), LAHC acceptance,
adaptive move probabilities, coarse-resolution search evaluations,
parallel evaluations (ADR-008 territory), tier-3 layout search.

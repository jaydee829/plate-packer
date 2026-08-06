# Rotation & Resolution Design — Shape-Aware Angles + Coarse-to-Fine Beam Search

**Date:** 2026-08-06
**Builds on:** ADR-011 (contact-scored placement + targeted-move ILS)
**Motivation (empirical):** On the 30-piece Tome of Demons set, contact-scored
greedy at 8 rotations gave **5 plates / fitness 0.3685**; at **16 rotations it
gave 4 plates / 0.4785**, edging out bottom-left greedy (4 / 0.4765). The
regression at 8 rotations was a rotation-granularity artifact: at 45° steps,
irregular hull edges almost never lie parallel to a neighbor or the plate, so
contact scoring cannot find long shared boundaries. Each full repack costs
~105-196s, so simply cranking a uniform rotation grid starves the ILS search.
This milestone makes many rotations *affordable* (coarse-resolution search) and
*targeted* (shape-aware angle candidates).

## Goal

Keep contact scoring as the default and raise effective rotation granularity
without starving the search, by: (1) generating shape-aware angle candidates
that lay hull edges parallel to plate axes; (2) running the ILS at a coarse
resolution and fine-packing only a small beam of the best orderings.

## 1. Shape-aware angle candidates (`angles.py`, new)

`angle_candidates(mask, cap=12, min_edge_frac=0.1, safety_grid=0) -> list[float]`

- Convex-hull the mask's occupied pixels once per piece (`cv2.convexHull` over
  `np.argwhere` boundary points).
- For each hull edge whose length exceeds `min_edge_frac × hull_perimeter`,
  emit the rotation angle that lays that edge parallel to the plate x-axis, plus
  that angle `+ 90°` (so the flat side can seat against the bottom *or* the left
  border). Angle for an edge with delta `(dx, dy)` is `(-degrees(atan2(dy, dx)))
  % 180`.
- Deduplicate angles within ~2°, sort ascending by the resulting axis-aligned
  bounding-box area (compact orientations first), cap at `cap`.
- Circle-like hulls (eccentricity below a threshold, i.e. hull area ≈ its
  min-enclosing-circle area) collapse to `[0.0]` — rotation is pointless.
- Always include `0.0` in the candidate set so the un-rotated footprint is
  available (right-angle lossless path).
- `safety_grid`: if `> 0`, union the shape-aware angles with a uniform grid of
  `safety_grid` angles (`i·360/safety_grid`). Default `0` — the mechanism exists
  but contributes nothing unless configured.
- Angles are **resolution-independent** (hull orientation does not change with
  raster resolution), so they are computed once per piece and reused at both
  coarse and fine resolution.

**Conservative-coverage note:** angle selection only chooses *which* rotations
to try; `rotate_mask` still grows-never-shrinks each rotated mask, and the
binary FFT legality gate is unchanged.

## 2. Coarse-to-fine beam search (`improve.py`, restructured)

`improve()` gains `coarse_res_mm=0.4`, `beam=5` (plus the existing budget /
patience / seed knobs). New pipeline:

1. **Prepare masks at two resolutions.** The working (fine) mask at
   `working_res_mm` (0.1) and a coarse mask block-max-downsampled to
   `coarse_res_mm` (0.4). Block-max **grows** the mask (any occupied fine
   sub-pixel → occupied coarse pixel), preserving the conservative-coverage
   invariant.
2. **Prerotate** each piece at its `angle_candidates` at *both* resolutions.
3. **Coarse ILS.** Run the existing targeted-move ILS loop over insertion order
   at coarse resolution — cheap evaluations (≈16× fewer pixels ⇒ ≈16× cheaper
   FFTs). Maximize coarse Falkenauer fitness. Maintain a bounded **beam of the
   top-K distinct orderings** seen (by coarse fitness); distinct = different
   insertion-order permutation.
4. **Fine refinement.** After the stop condition (budget or stall), fine-pack
   each of the K beam orderings at `working_res_mm` with shape-aware angles.
   Return the `ImproveResult` of the best **fine** fitness.

**Correctness — coarse-legal ⇒ fine-legal.** Coarse masks are supersets of the
fine masks (block-max growth). Non-overlap of supersets implies non-overlap of
subsets, so every ordering the coarse search deems legal packs validly at fine
resolution; fine placement can only seat equal-or-tighter. Coarse search may
*miss* some fine-legal tight placements (it is conservative), which is
acceptable — it ranks orderings; fine is the source of truth.

**Observability.** `improve()` also returns, per beam member, its coarse fitness
and its fine fitness — the coarse↔fine correlation data that decides whether the
deferred adaptive/successive-halving refinement (§7) is worth building.

**`ImproveResult`** gains `beam` (list of `(coarse_fitness, fine_fitness,
n_plates)` for the K survivors, best first). `placements`/`evaluations`/
`improvements`/`fitness_initial`/`fitness_final` refer to the winning fine pack;
`evaluations` counts coarse evaluations (the search effort).

**Determinism** is unchanged from ADR-011: deterministic per seed for a fixed
coarse-evaluation count; the wall-clock budget remains machine-dependent.

## 3. Difficulty-first ordering (the seed)

Replace largest-area-first with `difficulty = area × elongation`, where
`elongation = long_side / short_side` of the piece's min-area bounding box
(≥ 1). A long thin piece sorts ahead of an equal-area blob because it needs a
long channel that only exists early in the pack. Elongation is measured at a
canonical orientation, so it is rotation-stable. This is only the ILS **seed**
order; the search optimizes order from there, so its main effect is faster
convergence. Config `ordering = "difficulty" | "area"`, default `"difficulty"`.

## 4. Contact weighting

Plate-edge and piece-piece contact stay **equally weighted**
(`edge_contact_weight = 1.0`). Implementation: the contact-map's border frame is
padded with `constant_values=edge_contact_weight` instead of `1`; occupancy
pixels stay `1`. Rationale: border contact is already free and abundant, so
pieces frame the perimeter without an explicit boost, and over-boosting risks
wall-hugging at the expense of interior fills. The knob exists so a benchmark
showing loose interiors can raise it without code change.

## 5. Packer changes

Minimal. `pack()` already accepts per-piece `prerotated` dicts `{angle: mask}`
and a `plate_mask`; it receives shape-aware angle sets instead of a uniform
grid and is otherwise unchanged. `contact_map` gains the `edge_weight`
parameter (default 1.0) threaded from `edge_contact_weight`.

## 6. Config surface

New `[packing]` knobs (validated like existing ones):

| knob | default | constraint |
|------|---------|-----------|
| `coarse_res_mm` | 0.4 | ≥ `working_res_mm`, integer multiple of it |
| `beam` | 5 | ≥ 1 |
| `angle_cap` | 12 | ≥ 1 |
| `min_edge_frac` | 0.1 | 0 < x ≤ 1 |
| `safety_grid` | 0 | ≥ 0 |
| `edge_contact_weight` | 1.0 | ≥ 0 |
| `ordering` | `"difficulty"` | `{"difficulty", "area"}` |

Existing `rotations` is retained only for the legacy uniform-grid path and as
the `safety_grid` density reference. CLI `pack` gains `--coarse-res` and
`--beam` overrides.

## 7. Out of scope (deferred; recorded so we don't relitigate)

- **Adaptive / successive-halving refinement** — fine-pack top-1, then refine
  further beam members only if their coarse fitness can plausibly overtake the
  current best fine result. Build after the fixed-beam version yields
  coarse↔fine correlation data (§2 observability).
- **Cluster / pairwise nesting** — pre-nest two complementary pieces (e.g.
  mirror wings) into a super-piece before packing. Different architecture.
- **Delta / incremental evaluation** — re-pack only from the first changed piece
  in an ILS move rather than a full repack. The next perf lever after this.
- **Height-aware Z-banding** — v2.

## 8. Success criteria

- `angle_candidates` returns axis-aligning angles on known shapes (long
  rectangle → its long edge parallel to an axis; circle → single angle).
- Coarse-to-fine `improve()` on the 30-piece set matches or beats the
  16-rotation contact-greedy result (**≤ 4 plates**, fitness ≥ 0.4785) at a
  wall-clock cost below the single-resolution 8-rotation search, validated
  empirically after implementation.
- All existing invariants hold: conservative-coverage, merged-shadow self-check,
  determinism per seed for fixed coarse-eval count.

## 9. Testing

Parametrized/atomic per the repo rule:

- `angle_candidates`: square, long rectangle (long edge aligned to axis), circle
  (1 angle), L-shape, `safety_grid` union, `cap` enforcement, `min_edge_frac`
  filtering of short edges.
- Block-max downsample grows (coarse ⊇ fine), and the coarse-legal ⇒ fine-legal
  property on a hand-built case.
- Beam retains the top-K *distinct* orderings by coarse fitness; ties handled
  deterministically.
- Difficulty ordering puts an elongated piece ahead of an equal-area blob;
  `ordering="area"` reproduces the old seed.
- `contact_map` `edge_weight` scales border contact but not piece-piece contact.
- Coarse-to-fine determinism per seed for a fixed coarse-eval count (fake-clock
  as in ADR-011 tests); best returned result is the best *fine* fitness among
  the beam.
- E2E on a small synthetic set: fine fitness ≥ single-resolution greedy fitness;
  all placements pass the merged-shadow self-check.

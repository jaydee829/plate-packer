# Packing Methods Survey — Beating Greedy Bottom-Left

**Date:** 2026-08-05
**Method:** Deep-research workflow — 5 search angles, 20 sources fetched, 93 claims
extracted, top 25 adversarially verified (3 independent votes each; 25 confirmed,
0 refuted). Findings below are the synthesized, deduplicated claims with vote
counts and sources.

**Question:** What algorithms should plate_packer consider to beat greedy
bottom-left first-fit (54–62% plate occupancy on the 30-piece Tome of Demons
shakedown)? Scope: 2D irregular nesting (raster/NFP, metaheuristics, placement
heuristics) and 3D packing with a fixed face on a common plane (2.5D), free
Z-rotation or finer rotation discretization.

---

## Headline conclusions

1. **The raster + FFT architecture is validated, not a compromise.** Density
   gains come from *search layered on top of* the existing kernel, not from
   replacing it.
2. **Avoiding NFPs remains correct** (ADR-002 holds). Robust NFP generation for
   general non-convex polygons is still an open problem in 2025; recent SOTA
   solvers avoid NFPs too.
3. **The upgrade ladder, cheapest first:** (a) scored placement instead of
   first-legal-pixel; (b) metaheuristic search over piece *sequence* wrapping
   the existing placer; (c) overlap-minimizing layout search (SOTA, raster-native
   implementations exist). Rotation granularity is a compute knob, not a density
   lever.
4. **Exact/MIP methods can be ignored** for this workload (optimality tops out
   at ~10–27 polygonal pieces).

## Verified findings

### F1. Raster representation is competitive with vector/NFP state of the art
*(confidence: high; votes 3-0 ×4)*

Raster masks handle arbitrary non-convex shapes uniformly and reduce overlap
detection to cell counting, at the cost of memory, inexact non-orthogonal edges,
and a resolution/accuracy tradeoff (Bennell & Oliveira, EJOR 2008 — the canonical
geometry tutorial). Raster methods are not second-class: Sato et al.'s raster
penetration-map algorithm (EJOR 279, 2019) matched or beat best-known (largely
vector/NFP) compaction on 9 of 15 standard irregular strip packing instances and
improved average density on 13; a 2025 SOTA paper cites it as representative
state of the art.

- https://gent.cs.kuleuven.be/vakgroepit/sites/gent.cs.kuleuven.be/files/nesting_problems_tutorialEJOR-184-2008.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0377221719304837
- https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/full

### F2. NFPs: mainstream, exact, and still the wrong choice for us
*(confidence: high; votes 3-0 ×2)*

NFP gives O(n) overlap tests after precomputation, but building a robust NFP
generator for general non-convex polygons remains a known open engineering
problem (Bennell & Oliveira 2008; Frontiers 2022 review; still treated as open
by Rocha 2019, Gardeyn 2025). Per-rotation precomputation clashes with fine
rotation discretization. 2025 SOTA iterative solvers ("sparrow") use NFP-free
collision engines. **If an improvement loop is added, stay raster.**

### F3. Iterative search needs a fast incremental geometry kernel
*(confidence: high; vote 3-0)*

Exact trigonometric overlap testing (D-functions) is unsuitable for iterative
search — feasibility must be recomputed from scratch every move; it only works
inside one-pass constructive algorithms. Our FFT kernel satisfies the
constructive phase; an improvement loop wants per-move overlap *deltas*, not
full-plate reconvolution (a real design question for tier 3).

### F4. Metaheuristics dominate; exact MIP is irrelevant at our scale
*(confidence: high; votes 3-0, 2-1)*

Heuristics/metaheuristics dominate nesting in literature and practice (Leão et
al., EJOR 2020). SOTA exact MIP proves optimality only on ~10–27 polygonal
pieces within an hour (Lastra-Díaz & Ortuño 2024) and assumes polygonal
geometry. Matheuristics (MIP-assisted heuristics) exist but are polygon-bound —
a practical mismatch with raster masks (this sub-claim carried the one 2-1 vote).

### F5. Sequence search wrapping a BL placer is a proven, low-disruption win
*(confidence: high; votes 3-0 ×2)*

PAMPA (pixel-based BL + biased random-key genetic algorithm over placement
sequence; Computers & OR 2023, additive-manufacturing build plates — our problem
class) beat the prior pixel-based constructive method on 15/16 ESICUP instances
(13 statistically significant, Wilcoxon 5%) with ~20% less runtime. Ablation
attributes the gain to sequence search + placement scoring, **not** angle
selection. Corroborating lineage: Burke et al.'s bottom-left-fill + hill
climbing/tabu over the packing order produced 25 new best solutions on 26
benchmarks (Oper. Res. 2006). Maps directly onto plate_packer: keep the FFT/BL
placer as the inner decoder, wrap it in sequence search.

- https://optimization-online.org/wp-content/uploads/2022/08/An-Efficient-Pixel_based-Packing-Algorithm-for-Additive-Manufacturing-Production-Planning.pdf
- https://dl.acm.org/doi/abs/10.1287/opre.1060.0293

### F6. Rotation discretization is a compute knob, not a density lever
*(confidence: high; votes 3-0 ×2)*

PAMPA's shape-aware angle selection (circle-like → 0 rotations;
bounding-box-like → 2; hull≈bbox → 4; else hull-edge-aligned angles capped)
produced statistically identical density to a fixed 4-angle grid (0.584 vs
0.583; 19/20 instances indistinguishable) at ~16–20% less runtime. Caveat:
ablation ran on the authors' generated instances. Implication: don't chase
density through finer rotation grids or free rotation; use shape-aware angle
candidates to *save* compute if rotation cost becomes a bottleneck.

### F7. SOTA density = "search over the layout" with temporary overlap
*(confidence: high; votes 3-0 ×6 merged claims)*

The best-known benchmark results all come from methods that permit overlapping
intermediate layouts and then minimize/separate overlap under metaheuristic
guidance — not constructive placement. Two lineages: (a) Gomes & Oliveira's
simulated annealing + LP compaction/separation (+8.84% avg over prior best);
(b) guided local search over penetration depth (Umetani & Murakami coordinate
descent; Sato's ROMA full grid search over raster penetration maps). Every
best-result holder from 2006 through 2025 ("sparrow") is a layout-search method.
Caveats: mostly strip packing (minimize length), and Gomes-Oliveira's LP models
are polygon-based.

- https://www.sciencedirect.com/science/article/abs/pii/S0377221704005879
- https://ar5iv.labs.arxiv.org/html/2104.04525
- https://www.sciencedirect.com/science/article/abs/pii/S0377221719304837

### F8. Layout search exists raster-natively — no need to leave pixel space
*(confidence: high; votes 3-0 ×3)*

Umetani & Murakami's coordinate-descent GLS runs directly on rasterized shapes
(alternating horizontal/vertical line searches, corner detection), made
tractable at high resolution by a "double scanline" run-length compression of
masks. Sato's ROMA restricts placement to a rectangular grid and searches
preprocessed raster penetration maps — a direct analogue of our
FFT-correlation-over-raster approach that scores overlap *depth* rather than
binary feasibility. Caveats: Umetani used 0/180° rotations only; scanline
machinery would replace rather than accelerate fftconvolve; ROMA's penetration
maps are pairwise structures, not one per-plate map.

### F9. Concrete placement-scoring rules, all pluggable into `choose`
*(confidence: high; votes 3-0 ×3)*

Established alternatives to bottom-left: **maximum-utilization** (maximize area
utilization in the earliest bin — directly aligned with plate-count
minimization), **minimum-length** (minimize enclosing rectangle of partial
layout), **lowest-gravity-center**, and **hole-filling** via inner-fit polygon.
PAMPA's tie-breaker is **attachment value** (boundary contact with container +
placed parts), credited for denser layouts. A 2025 raster method (Meng et al.,
Sci Rep 15:12320) ranks candidates by cross-correlation score descending —
trivially available from our existing fftconvolve output — and masks out voids
smaller than the current piece to eliminate dead space (note: "filling" there
means marking too-small voids occupied, not placing pieces into them).

- https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2022.966691/full
- https://www.nature.com/articles/s41598-025-97202-0

### F10. Physics-style post-passes work but transfer only analogically
*(confidence: medium; vote 3-0)*

Simulated container shaking + vacant-space filling after heuristic placement
measurably compacts 3D layouts (Zhuang et al., Computers & Graphics 2024) and
beat prior 3D irregular packing (54.47% vs ~51.3% on a dental dataset). The
gravity-driven 3D mechanism is only an analogy for our fixed-base 2.5D setting;
the paper itself notes initialization contributes most of the density.

### Bonus (from fetch phase): FFT correlation validated at scale

Cui et al., "Dense, Interlocking-Free and Scalable Spectral Packing of Generic
3D Objects" (SIGGRAPH 2023) — the closest published match to our architecture
(voxel masks + FFT correlation enumerating all collision-free offsets per
rotation) — measured FFT ~3000× faster than brute-force sliding (0.003s vs 9.4s
on a ~3M-cell grid). Validates ADR-003 as scalable.

- https://dspace.mit.edu/bitstream/handle/1721.1/152166/3592126.pdf

## Cross-cutting caveats

- **Benchmark transfer:** nearly all quantitative results are strip packing
  (minimize length) or single-bin density, not plate-count minimization on
  fixed plates. Density gains should translate to fewer plates, but no surveyed
  source measures our exact objective — validate empirically on Tome of Demons.
- **Novelty gap:** no single verified source demonstrates free-rotation +
  penetration-depth raster layout search combined. Tier 3 with fine rotation
  would be lightly novel.
- **2.5D framing:** the fixed-base problem was answered almost entirely by 2D
  nesting literature; the AM papers (PAMPA, Meng 2025) treat build-plate packing
  the same way, confirming the 2D reduction is the standard model.
- **Frontier moved in 2025:** "sparrow" (NFP-free collision engines) surpasses
  ROMA by ~5% on some instances; cited historical results are accurate but not
  the frontier.

## Open questions for the improvement milestone

1. How well do strip-packing density gains translate to plate-count reduction
   at our scale (dozens of pieces, 54–62% baseline)? Empirical validation only.
2. Can penetration-depth layout search use FFT cross-correlation as its kernel
   (overlap counts are already a soft score) — and how does per-move incremental
   update compare to full-plate reconvolution?
3. Best rotation strategy inside a layout-search loop — e.g., coarse-to-fine
   refinement around promising placements?
4. For a fixed time budget, which layer buys the most occupancy per CPU-second:
   sequence search, placement scoring, or full layout search?

## Recommendation

Build **tier 1 (scored placement)** and **tier 2 (time-budgeted sequence
search)** together as the improvement milestone: tier 1 is nearly free and
strengthens tier 2's inner loop. Defer **tier 3 (overlap-minimizing layout
search)** to v2+. Spend nothing on rotation granularity for density; consider
shape-aware angle pruning later purely as a speed optimization.

# Architectural Decision Records

This file documents key architectural decisions, their context, and trade-offs.

## Templates

### ADR-XXX: Decision Title (YYYY-MM-DD)

**Context:**
- Why the decision was needed
- What problem it solves

**Decision:**
- What was chosen

**Alternatives Considered:**
- Option 1 -> Why rejected
- Option 2 -> Why rejected

**Consequences:**
- Benefits
- Trade-offs

## Decisions

### ADR-001: Raster masks instead of vector geometry / No-Fit Polygon (2026-08-01)

**Context:**
- Core problem is 2D irregular polygon nesting (NP-hard). Prior art (Deepnest) uses NFP + metaheuristics.
- Input meshes are often non-manifold garbage from Patreon/Kickstarter files.

**Decision:**
- Rasterize footprints into binary numpy masks (~0.05–0.1 mm/px, configurable). Union = pixel OR.

**Alternatives Considered:**
- NFP / shapely boolean unions -> polygon-clipping robustness bugs on ugly meshes; requires topological validity we can't guarantee.

**Consequences:**
- Pros: Arbitrary concavity for free; ugly meshes rasterize fine; rotation search is cheap.
- Trade-offs: Resolution-dependent accuracy; memory/compute scale with plate size and resolution.

### ADR-002: Footprint = vertical shadow of the full supported mesh (2026-08-01)

**Context:**
- Need a collision-safe 2D footprint covering model + supports + raft, including support branches wider than the raft.

**Decision:**
- Project every triangle straight down onto XY; footprint is the union. Non-overlapping shadows provably cannot collide at any height.

**Alternatives Considered:**
- Raft outline only -> misses flared supports and overhangs.
- Height-aware Z-banded masks -> correct but complex; deferred to v2, not v1.

**Consequences:**
- Pros: Simple and provably correct.
- Trade-offs: Conservative — forbids tall-over-short interleaving until v2.

### ADR-003: Collision detection via FFT cross-correlation (2026-08-01)

**Context:**
- Need overlap tests for every candidate translation at many rotations, fast.

**Decision:**
- `scipy.signal.fftconvolve` plate occupancy against candidate mask; zero pixels in the result are legal placements. One convolution per rotation (~36–72 steps). CPU first, cupy later if needed.

**Alternatives Considered:**
- Per-position mask AND tests -> orders of magnitude slower.
- NFP-based placement -> see ADR-001.

**Consequences:**
- Pros: All translations evaluated at once; "no zeros across all rotations" is a *proof* the piece doesn't fit → clean spillover to next plate, no timeouts.
- Trade-offs: FFT output needs a numerical zero-tolerance; cost scales with mask size.

### ADR-004: Open formats only; v1 exports one merged STL per plate (2026-08-01)

**Context:**
- Proprietary slicer projects (.lys, .chitubox) are closed, unparseable, version-fragile.
- Lychee "Export 3D Asset" bakes supports into plain STL — the sanctioned interchange point.

**Decision:**
- Input: supported STL/OBJ. Output v1: one merged STL per plate (`trimesh.util.concatenate`); 3MF per plate in v1.5 after round-trip testing.

**Alternatives Considered:**
- Parse/write .lys -> quarantined to speculative v3, never a core dependency.
- 3MF immediately -> slicer transform-import behavior unverified.

**Consequences:**
- Pros: Slicer sees one object and can't auto-arrange it out from under us; format stability.
- Trade-offs: No per-object identity in the slicer for v1 (acceptable — everything is pre-supported).

### ADR-005: Greedy packing first; improvement loop is an optional bolt-on layer (2026-08-01)

**Context:**
- Real objective is minimizing plate count; need a working end-to-end pipeline before optimizing density.

**Decision:**
- Largest-area-first greedy with spillover, pluggable placement heuristic (bottom-left initially). Time-budgeted anytime metaheuristic (annealing/restarts) added later as a separate layer.

**Alternatives Considered:**
- Metaheuristic-first -> delays a working pipeline; greedy alone is already competitive with slicer auto-layout.

**Consequences:**
- Pros: Every improvement-loop iteration holds a complete valid solution — early stopping costs quality, never correctness.
- Trade-offs: v1 density is heuristic-limited.

### ADR-006: Open source on GitHub; uv + ruff + GitHub Actions CI (2026-08-01)

**Context:**
- Project will be published as open source; needs standard, low-friction contributor tooling.

**Decision:**
- Host on GitHub with GitHub Actions CI (tests + lint on push/PR).
- `uv` for packaging/dependency/env management (pyproject.toml).
- `ruff` for linting and formatting, enforced in CI.

**Alternatives Considered:**
- pip + venv -> works (current interim setup) but slower, no lockfile story.
- black/flake8/isort -> ruff replaces all three with one tool.

**Consequences:**
- Pros: Fast installs and reproducible envs via uv lock; single-tool lint/format; CI mirrors local commands.
- Trade-offs: Existing pip `.venv` must be migrated to uv when the package is formalized.

### ADR-007: v1 interfaces must not block v2 Z-banding (2026-08-01)

**Context:**
- v2 replaces one shadow mask per piece with K per-band masks to allow tall-over-short overhang interleaving. Designed now so v1 code doesn't paint us into a corner.

**Decision:**
- Extraction returns an opaque per-piece footprint object the packer doesn't introspect; v2 swaps "one mask" for "K masks + band edges" touching only extraction and the legality check.
- Legality stays "zeros in a non-negative correlation map": band maps are summed (in frequency domain — one inverse FFT per rotation), so zero-sum = legal in every band. Greedy loop, spillover proof, and export are untouched.
- Banding gives overhang interleaving only — pieces always sit at Z=0; no stacking.
- Physical clearance handled at extraction as dilation knobs: per-band XY margin (removal/drainage under canopies) and Z-margin at band edges.
- Rasterization invariant: raster resolution <= min_spacing/4, so dilation swamps sub-pixel aliasing on thin support slivers; draw triangle edges (polylines) in addition to fills as dropout insurance.

**Alternatives Considered:**
- Full 3D voxel collision -> massive cost; banding captures the mini-shaped win (wide base/thin stem/wide canopy) with 4-8 bands.
- Sparse point-set collision for near-empty upper bands -> optimization only if K-band FFT ever measures as bottleneck.

**Consequences:**
- Pros: v2 is an extraction+legality change, not a rewrite; correctness argument (conservative per-band shadows) carries over.
- Trade-offs: conservative within each band; per-triangle band assignment duplicates boundary-spanning triangles.

### ADR-008: Rasterization loop stays serial until measured; escalation is render, not threads (2026-08-01)

**Context:**
- Per-triangle fillConvexPoly measured 2-3.5s on 1.7M-triangle real meshes, once per piece (rotation sweep rotates the mask, not the mesh).

**Decision:**
- Keep the serial loop. If extraction measures as the bottleneck: (1) piece-level multiprocessing pool, (2) orthographic top-down depth render (trimesh/pyrender) which also yields Z-band masks cheaply.

**Alternatives Considered:**
- Thread-parallelizing the inner loop -> concurrent OpenCV writes to one buffer are unsafe; per-thread masks + OR is a worse version of the process pool.

**Consequences:**
- Pros: simple, obviously correct code while the pipeline is built around it.
- Trade-offs: extraction of a 20-piece job spends ~40s serial today.

### ADR-009: Adopt stl_curator footprint contract; footprint docs store intrinsic data only (2026-08-01)

**Context:**
- stl_curator's M1 design (its docs/superpowers/specs/2026-08-01-stl-curator-m1-design.md §4) defines the normative interface: footprints are content-addressed by STL SHA-256 at `footprints/<first-2-hex>/<hash>.json` under a shared `footprints_dir`; one STL → one JSON doc → many footprints (z-slices); plate_packer owns and versions the JSON internals; curator treats it as opaque.
- Content-addressing by hash alone means the document can only contain data *intrinsic to the STL*. Our current extraction bakes in the min-spacing dilation — but spacing (and target resolution) are packer/printer config, not properties of the file. Baked-in dilation would silently serve wrong margins when config changes.

**Decision:**
- Accept the contract as written; no changes requested to stl_curator.
- Footprint documents store only intrinsic data: UNDILATED shadow mask(s) at a recorded canonical resolution, origin, z-height, band edges (v2), triangle count, dropped-nonfinite count, schema version. Masks embedded as base64 PNG (binary masks compress ~100x).
- Spacing dilation and any resolution downsampling move from extraction time to packer load time (one cv2.dilate per piece — measured trivial). The packer core stays margin-unaware; the loading layer applies config.
- Supersedes the "dilate once at extraction" clause of the pipeline design (seed doc §1); everything else stands.

**Alternatives Considered:**
- Key cache by hash+params -> violates the contract (curator derives location from hash alone) and explodes cache variants.
- Ask curator to add params to the path -> unnecessary; the fix is cleanly on our side and the contract explicitly gives us schema ownership.

**Consequences:**
- Pros: one cached footprint serves any spacing/resolution config; contract untouched; cache is pure function of file content.
- Trade-offs: extraction output is no longer directly packable — a thin load step (dilate + optional downsample) sits between cache and packer.

### ADR-010: spacing_mm is the true inter-piece gap; per-piece dilation = spacing/2 (2026-08-01)

**Context:** The packer packs dilated masks against dilated occupancy, so dilating each
piece by d enforces a 2d gap between true footprints. Dilating by the full spacing
silently doubled the user's requested gap.

**Decision:** `spacing_mm` (config) is the minimum gap between placed pieces. Per-piece
dilation radius = `ceil((spacing_mm/2) / working_res_mm)` px (`loading.dilation_radius_px`).
Side effect (accepted): pieces keep >= spacing/2 clearance from plate edges because the
dilated mask must fit on-plate; `edge_margin_mm` adds an explicit dead band on top.

**Alternatives:** Keep full-spacing dilation and document gap = 2x spacing -> rejected,
violates least surprise. Dilate candidate only, keep occupancy undilated -> rejected,
requires storing both mask variants and changes packer internals for no precision gain.

**Consequences:** Halved dilation radii vs. the pre-ADR behavior; prepare_mask returns
(mask, origin_mm) so export can compose exact transforms.

**Update (2026-08-07): `spacing_mm` default lowered 2.0 → 1.0mm** (user-approved). On
the 30-piece benchmark, 1mm keeps the same 4-plate floor (area-bound for this set) but
packs the used plates denser (raw occupancy up on 3 of 4). 1mm is a reasonable baseline
for pre-supported minis; bump for beefier supports. Purely the *default* — the knob and
its spacing/2 semantics are unchanged, and it's applied at load time (undilated cache
reused, so re-runs at other values are cheap). See key_facts "Real-World Benchmarks".

### ADR-011: Falkenauer fitness + contact-scored placement + targeted-move ILS (2026-08-05)

**Context:** Greedy bottom-left first-fit reached 54-62% occupancy on the 30-piece
shakedown. A verified deep-research survey (docs/research/2026-08-05-packing-methods-survey.md)
found density gains come from search layered on a fast geometry kernel, not smarter
one-pass rules; raster/FFT was validated as competitive with vector/NFP state of the art.

**Decision:** (1) Objective is Falkenauer's grouping fitness `mean(fill^2)` — THE single
objective, no separate plate-count term (fewer plates dominates it naturally). (2) Placement
chooser scores legal anchors by boundary contact (1px-halo ring correlated against occupancy
+ plate border via a second fftconvolve), max contact, bottom-left tie-break; default on,
`placement = "bottom_left"` config fallback. (3) Improvement = iterated local search over
the greedy insertion order: 70% targeted moves (reinsert from min-fill plate, swap
lowest-contact piece), 30% random; shake after 20 fails; stops at wall-clock budget
(`improve_budget_s`, default 2700) or stall (`patience` evals without `min_improvement`
gain). Budget 0 = plain greedy. Deterministic per `seed` **for a fixed evaluation count**:
the stall stop is deterministic, but the wall-clock budget is not (eval count depends on
machine speed), so a budget-bounded run can differ across hardware for the same seed. For a
reproducible run, set `improve_budget_s` high enough that the stall always fires first.

**Alternatives:** A*/branch-and-bound -> rejected, no usable admissible bound for irregular
nesting (area bound prunes nothing; exact methods stall at ~10-27 polygons). BRKGA
(PAMPA-style) -> deferred, needs thousands of evaluations vs our 50-200 per 45-min budget.
Beam-search constructor -> rejected, myopic partial fitness. Overlap-minimizing layout
search (SOTA) -> deferred to v2+ (different architecture: penetration maps, incremental moves).

**Consequences:** `pack()` gained prerotated/order/validate params (rotation cached across
repacks); `Placement.contact` records the chosen anchor's score; contact adds a second FFT
per placement attempt (~2x constructor cost, reclaimed by rotation caching in the loop).

### ADR-012: Shape-aware angles + coarse-to-fine beam search (2026-08-06)

**Context:** Benchmark on the 30-piece set (see key_facts "Real-World
Benchmarks") showed contact-scored greedy regressed to 5 plates at 8 rotations
but recovered to 4 plates (beating bottom-left greedy) at 16 rotations — the
regression was a 45°-granularity artifact (irregular hull edges rarely lie
parallel to anything). But each repack costs ~105-196s, so a uniform fine
rotation grid starves the ILS search. Keeps contact as default (revises the
post-ADR-011 "flip to bottom-left" idea — with enough rotations contact wins).

**Decision:** (1) **Shape-aware angle candidates** (`angles.py`): lay convex-hull
edges parallel to plate axes, capped, circle-like → 1 angle; a uniform
safety-grid union is available but off by default. (2) **Coarse-to-fine beam
search**: run the ILS at a coarse resolution (0.4mm, ~16x cheaper evals),
keep the top-K orderings, fine-pack only those at 0.1mm, return the best fine
result. Coarse-legal ⇒ fine-legal (block-max downsample grows masks, so coarse
masks are supersets). (3) **Difficulty-first ordering** seed (area × elongation)
replaces largest-area-first. Fixed beam-K now; adaptive successive-halving
deferred until coarse↔fine correlation data justifies it.

**Alternatives:** uniform fine grid (no shape-awareness) → wasteful, starves
search. Adaptive/hierarchical beam from the start → YAGNI, K is small so batch
fine-refine is cheap; build after fixed-beam yields correlation data. Cluster
nesting, delta-evaluation → deferred.

**Consequences:** `improve()` restructured (coarse/fine masks, beam);
`contact_map` gains `edge_weight` (plate-edge vs piece-piece contact weight,
default 1.0 = equal). Spec: docs/superpowers/specs/2026-08-06-rotation-resolution-design.md.

**Retrospective (2026-08-07, validated on the 30-piece set):** final result
**4 plates / fine fitness 0.4801**, beating bottom-left (0.4765) and 16-rot
contact (0.4785) — but only after two corrections and a default change:

1. **Shape-aware angles alone under-deliver — the "targeted angles replace
   uniform grids" premise only half-held.** With `safety_grid=0` the pack came
   out at **5 plates / 0.3497**: the 30 pieces averaged only 4.7 shape-aware
   angles each (5 circular bases got just 1), far short of the ~16 rotations
   contact scoring needs. The uniform `safety_grid` backstop does the heavy
   lifting; shape-aware angles are a *supplement*, not a replacement.
   **`safety_grid` default flipped 0 → 16** so the tool performs out of the box
   (with `angle_cap=12`, safety_grid=16 → ~8 distinct mod-180 uniform angles
   unioned with shape-aware, capped to 12). This partly re-opens the ADR-011
   "contact vs bottom-left" question: contact only wins with enough rotations.
   **CAVEAT (2026-08-07, review round 2): this "shape-aware under-delivers"
   conclusion was CONFOUNDED by a bug** — `angle_candidates` emitted the mirror
   angle, so the shape-aware angles were generically *wrong* (bugs.md). With that
   fixed, shape-aware angles may now contribute meaningfully and `safety_grid`
   might be reducible. **Re-benchmark before trusting the default-16 value** and
   re-assess whether shape-aware angles earn their keep vs a pure uniform grid.
2. **The fine stage lost a plate by re-packing instead of realizing the coarse
   layout** (coarse fit 4, fine re-pack spilled to 5). Fixed: the fine stage now
   also realizes the coarse layout (`_scale_placements`, anchors × factor —
   collision-free/in-bounds by the block-max superset property + a coarse-plate
   padding guard) and keeps the better of {fine re-pack, realized-coarse}. See
   bugs.md 2026-08-07. **General lesson:** a multi-resolution search must
   *realize* the coarse solution, not rank orderings and re-solve at fine.
3. **Export OOMed at production scale** (trimesh float64 caches on a ~3.9M-tri
   plate) — rewrote `export_plates` to stream the binary STL per piece. bugs.md.

Deferred/next: shape-aware angles' marginal value over pure uniform is now
questionable — a future pass could measure whether they add anything beyond the
safety grid, and revisit `angle_cap` given safety_grid is on by default.

### ADR-013: Support-aware footprints (base-layer exclusion) (2026-08-07)

**Context:**
- Pre-supported models include rafts and tree supports; on resin plates these rigs occupy material budget but are disposable post-print.
- Collision detection at v1 conservatively uses the full vertical shadow of the supported mesh (ADR-002), forbidding raft overlap — two rafts cannot coexist on the same plate.
- Removing base-layer support via footprint-area knee detection allows raft interior to overlap freely while keeping the outline on-plate, freeing interior space for adjacent pieces.

**Decision:**
- Opt-in `support_aware` config knob (default false). When enabled, extract a `model_body` footprint via **footprint-area knee detection**: measure the projected area of the shadow at each Z-level; find the height `cut_z_mm` where the area stops shrinking within the `support_cut_cap_mm` window (interior cavity base); `model_body` is the conservative superset at Z ≥ cut_z_mm. No cut unless area drops ≥ `MIN_REDUCTION` (5%).
- **Two-mask collision**: the *only* permitted overlap is raft-on-raft (bases fuse). The packer tracks per plate the union of placed **bodies** and of placed **fulls**; a placement is legal iff the candidate **full** clears placed **bodies** AND the candidate **body** clears placed **fulls** AND the full stays within the plate/margins. Both directions are required — a single full-vs-body check misses body-on-raft (a body carries low-Z support material, so a body over another's raft is a real collision), which is the common mixed cut / no-cut case. (Corrected 2026-08-07 after PR #8 review found the original body-vs-body-only check silently allowed raft-on-body collisions.) Implemented via `rotate_pair` (shared canvas → same-shape AND legality).
- Cache schema v2 adds optional `model_body` entry (SHA-256 keyed per-STL); reads accept v1 or v2 (falls back to full shadow gracefully). Rafts from the detected cut may fuse in the merged output; acceptable trade-off.

**Alternatives Considered:**
- Fixed-mm cut (e.g., "cut 1 mm above model base") -> inflexible, meshes vary widely; some have no raft or shallow rafts.
- Per-piece numeric `cut_z_mm` input -> defers the hard problem to the user; defeats the benefit.
- Area-cliff detector (jump in area between z-levels) -> meshes are hollow shells, not solid objects; cliffs rarely sharp; rejected in testing.
- Horizontal-cap detector (find flat z-span) -> same issue; raft caps often sloped or layered; cuts too shallow, 0% gain in real tests.
- Band-stack alternative (parallel band height evaluation) -> deferred to v2; too complex for v1.
- Single-mask packing (raft off-plate edge) -> violates the print contract; raft must stay on-plate.
- Crop-offset verify (separate interior/boundary checks on clipped regions) -> superseded by shared-canvas two-mask approach.

**Consequences:**
- Pros: 14–32% real measured footprint reduction on `*_supported.stl` from the Tome of Demons corpus; interior concavity gain (real rafts hug outline at 0.0 mm outer flare). Raft fusion acceptable; pieces still print successfully.
- Trade-offs: Requires correct STL mesh connectivity (non-manifold rafts may fail silently). v2 will add per-band exclusion for deeper overhangs. Detector evolution was hypothesis-driven (area-cliff, horizontal-cap tried first); see bugs.md 2026-08-07 for float32 precision discovery. Spec: `docs/superpowers/specs/2026-08-07-support-aware-footprints-design.md`.

### ADR-014: Raft-signature gate (band-dominance acceptance) (2026-08-08)

**Context:**
- The two-mask collision model (ADR-013) confines overlap to raft∩raft, but is only as safe as classification: the area-knee detector fires on any mesh whose shadow shrinks ≥5% within the cap and plateaus — including 13 unsupported corpus meshes (wings on a tip, merged cloth; all cap-adjacent knees) and hypothetical integral plinths. A bogus cut opts model geometry into raft fusion.

**Decision:**
- Accept a knee only if the geometry just above it looks like a support forest: rasterize triangles straddling `z0 + cut + BAND_MM` (cross-section outlines) and require largest-component/band-area ≤ `RAFT_BAND_DOMINANCE_MAX = 0.35`; empty band ⇒ reject. Module constant, not config (ADR-009 hash-addressability). `DETECTOR_VERSION` 1→2. `detect_base_cut(..., gated=False)` exposes the raw knee for probing (`tools/probe_raft_gate.py`).
- Corpus calibration (Tome of Demons, 229 STLs): true rafts dominance 0.018–0.208 (n=101, knees 0.25–1.25 mm), bogus cuts 0.556–1.0 (n=13, knees 4.5–5.0 mm) — 2.7× gap around 0.35. Gate: 101 accepts / 13 rejects, exact.

**Alternatives Considered:**
- Band *area fraction* (straddle area / footprint) → does NOT separate: meshes are hollow shells, so a wing's wall ring is as sparse by area (0.02–0.08) as a support forest. Dominance measures connectedness — the actual physical difference — and is scale-free.
- `raft_window_mm` accept-window on knee depth → redundant (deep knees are exactly the high-dominance ones) and cannot catch a shallow plinth.
- Filename allowlist (`*supported*`) → corpus has 7 supported exports missing the suffix; naming is unreliable.

**Consequences:**
- False rejects (few-pillar minis; corpus min is 22 components) cost density only, never correctness. Remaining false-accept shape — a field of separate thin spikes off the plate — is physically raft-like; accepted risk.
- Underlying mechanism of both failure directions: the band is the *XY shadow* of straddling triangles, not a true planar cross-section. Sloped geometry (tapered support necks, diagonal struts) inflates/merges components → false reject, the safe direction; a near-horizontal patch that happens to straddle the plane (spread hand, crown tips) contributes an isolated small blob instead of a slice → the concrete false-accept path. Well-separated on Tome of Demons' vertical pillar forests (2.7× gap); recheck via `tools/probe_raft_gate.py` on corpora with more organic/sloped supports. Opt-in regression: `test_raft_gate_verdict_on_real_corpus` (`-m example_stls`) pins 2 accept + 2 reject verdicts on named corpus files.
- Smooth synthetic tapers (e.g. `trimesh.creation.cone`) never fire the knee at all (every side triangle reaches the apex, so the reach map never drops); fine-tessellated real tapers do. Synthetic taper tests must use stacked shrinking boxes.

### ADR-015: Raft-fusion packing (gated body-over-raft nesting) (2026-08-08)

**Context:**
- Strict two-mask packing (raft∩raft only, ADR-013) measured ~no density gain: rafts hug outlines (0 flare), so raft-only regions are interior concavities unreachable without a body crossing the other full outline. The valuable move — body nesting over a neighbor's raft — was forbidden because a 2D shadow can't prove the body column has no low-Z material.
- Physical audit of body-over-raft for a *gate-accepted* piece: its below-raft-top material at body pixels is its own raft + pillar feet — disposable, same class as the already-permitted raft-raft fusion. The residual hazard (model surface dipping below a neighbor's ~1 mm raft top on a piece that still carries an accepted raft) is rare on Lychee-style exports.

**Decision:**
- **Fused ⇔ body ≠ full** (gate-accepted cut), derived inside the packer by mask comparison — no new parameters, no CLI/improve changes. Fused pieces may nest bodies over each other's rafts and fuse raft-with-raft; their bodies never overlap (spacing intact). A non-fused piece keeps strict full-shadow collision both ways. Plate boundary always uses the full. Per-plate grids: fused bodies ∪, non-fused fulls ∪, all fulls ∪.
- `support_cut_cap_mm` default 5.0 → **3.0**: costs nothing (all accepted corpus knees ≤ 1.25 mm), independently kills deep bogus knees, and guarantees fusion can never touch geometry above 3 mm. No `DETECTOR_VERSION` bump (cap is config, not part of the cache key; cached cuts unaffected).
- Coarse-res collapse (thin raft ring vanishing under block-max downsample → coarse body == full → treated non-fused) is strictly conservative, preserving improve()'s coarse-legal ⇒ fine-legal invariant.

**Alternatives Considered:**
- Keep strict two-mask → provably safe but delivers no density; that safety is preserved per-piece for anything the gate rejects.
- Per-pixel min-Z clearance mask ("underpass-safe") → converts the accepted risk into a proof; **tabled by explicit user decision (YAGNI), designated v2 hardening**.
- Full Z-banded 2.5D collision → tabled with it.

**Consequences:**
- Accepted risk: fusion confined to z < 3 mm guaranteed, < ~1.25 mm in practice, on gate-verified pieces only; a model dipping below a neighbor's raft top is the silent failure case — revisit the clearance mask if it ever bites.
- Spec: `docs/superpowers/specs/2026-08-08-raft-fusion-packing-design.md` (includes the permitted-overlap matrix).

## Usage Tips

- Check this file **before** proposing an architectural change. If the proposal
  conflicts with an existing ADR, acknowledge the prior decision and explain why
  revisiting it is warranted.
- ADRs are lightweight and historical — keep all of them.
- Find decisions about a topic with
  `Grep(pattern="^### ADR-", path="docs/project_notes/decisions.md")` or a keyword search.

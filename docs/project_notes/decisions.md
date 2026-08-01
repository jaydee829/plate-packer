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

## Usage Tips

- Check this file **before** proposing an architectural change. If the proposal
  conflicts with an existing ADR, acknowledge the prior decision and explain why
  revisiting it is warranted.
- ADRs are lightweight and historical — keep all of them.
- Find decisions about a topic with
  `Grep(pattern="^### ADR-", path="docs/project_notes/decisions.md")` or a keyword search.

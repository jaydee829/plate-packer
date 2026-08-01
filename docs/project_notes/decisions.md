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

## Usage Tips

- Check this file **before** proposing an architectural change. If the proposal
  conflicts with an existing ADR, acknowledge the prior decision and explain why
  revisiting it is warranted.
- ADRs are lightweight and historical — keep all of them.
- Find decisions about a topic with
  `Grep(pattern="^### ADR-", path="docs/project_notes/decisions.md")` or a keyword search.

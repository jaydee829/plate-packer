# Work Log (Issues)

This file tracks work history and ticket references.

## Templates

### YYYY-MM-DD - TICKET-ID: Brief Description
- **Status**: Completed / In Progress / Blocked
- **Description**: 1-2 line summary
- **URL**: Link to ticket or PR
- **Notes**: Any important context

## Log

### 2026-08-01 - FOOTPRINT-IO: Content-addressed cache + CLI (PR #1)
- **Status**: Completed (PR open, CI green)
- **Description**: Implemented ADR-009 via subagent-driven development: package extraction (undilated), footprint_io contract docs (schema v1, 0.05mm/px, atomic writes), dilate-on-load, `plate-packer footprints` CLI. 53 tests.
- **URL**: https://github.com/jaydee829/plate-packer/pull/1
- **Notes**: Two instructive bugs caught by review/CI: cached-array aliasing on the no-op path, and typer's single-command collapse (see bugs.md). Export-milestone TODO recorded in key_facts.md (prepare_mask origin offset).

### 2026-08-01 - PACKER: Greedy packing engine (v1 core)
- **Status**: Completed
- **Description**: TDD-built src/plate_packer/packer.py — FFT legality map, bottom-left heuristic (pluggable), lossless right-angle + conservative arbitrary-angle mask rotation, largest-first greedy with spillover, pre-pack validation. Repo flipped to src/ package layout (hatchling).
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: End-to-end on real Archvillain footprints: 8 pieces -> 1 plate (200x130mm), 56% occupied, pack=14s at 8 rotations. Rotation must never LOSE pixels (false free space) — right angles use np.rot90, arbitrary angles use linear-interp + any-touched threshold. Next: export (transform + merged STL) and the merged-shadow self-check.

### 2026-08-01 - SETUP: Tooling, prototype, and GitHub publish
- **Status**: Completed
- **Description**: uv-managed pyproject + ruff + pytest; footprint-extraction prototype with tests (2 rasterization bugs found/fixed, see bugs.md); published public repo with MIT license and GitHub Actions CI.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: CI matrix is Python 3.11/3.14 with opencv-python-headless (avoids libGL on Linux runners). Awaiting real STLs in example_stls/ (gitignored — copyrighted content).

### 2026-08-01 - INIT: Project bootstrap
- **Status**: Completed
- **Description**: Created CLAUDE.md from the seed design doc and initialized the project memory system (docs/project_notes/ + memory protocols in CLAUDE.md, GEMINI.md, AGENTS.md).
- **URL**: N/A
- **Notes**: Repo is pre-code. Next planned step per PLATEPACKER_SEED.md: prototype footprint extraction against real supported STLs in `example_stls/` (currently empty — needs files).

## Usage Tips

- Log completed work with a ticket/PR id, date, and link so history stays traceable.
- Keep descriptions to 1-2 lines; put longer context in the **Notes** field.
- Archive entries older than ~3 months by manual cleanup; this log is not automated.

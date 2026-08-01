# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Memory System

Institutional knowledge lives in `docs/project_notes/` (protocols shared with other AI tools via `AGENTS.md`/`GEMINI.md`):

- **decisions.md** — ADRs. Check **before** proposing architectural changes; the five settled v1 decisions below are recorded there as ADR-001..005. If a proposal conflicts, acknowledge the ADR and justify revisiting it.
- **bugs.md** — bug log with root causes and fixes. Search it when an error feels familiar; add an entry when a recurring/instructive bug is resolved.
- **key_facts.md** — project constants, conventions, config surface. Prefer documented facts over assumptions.
- **issues.md** — work log. Add an entry when completing a piece of work.

Keep entries concise, dated, bullet-listed.

## Pre-Merge Checklist (required)

When work is complete and you are about to ask the user whether to merge, update BOTH memory systems FIRST, in the same turn as the merge prompt — the user merges and then compacts the session before starting a new task, so anything not written down is lost:

1. **Project memory** (`docs/project_notes/`): work-log entry in `issues.md`; any instructive bugs to `bugs.md`; new/changed decisions to `decisions.md`; new constants, contracts, or TODOs to `key_facts.md`. Commit these to the branch being merged.
2. **Claude's persistent memory** (the per-project memory directory + `MEMORY.md` index): anything cross-session that doesn't belong in the repo — user preferences and feedback, workflow lessons, cross-project context (e.g. stl_curator coordination state).

Only then present the merge options.

## Project Status

Working package (`src/plate_packer/`, uv-managed, hatchling src layout) with pytest suite and GitHub Actions CI. Implemented: footprint extraction (`footprint.py`), greedy packer (`packer.py`), content-addressed footprint cache per the stl_curator contract (`footprint_io.py`, ADR-009), dilate-on-load with origin tracking (`loading.py`, ADR-010 spacing semantics), config (`config.py`), exact-transform export + runtime merged-shadow self-check (`export.py`), and the CLI (`cli.py`: `footprints`, `pack`). The v1 loop is complete: folder of STLs → `plate_NN.stl` + report. Next candidates: real-pack validation on Tome of Demons, PNG layout previews / 3MF (v1.5), improvement loop. The original design doc is `PLATEPACKER_SEED.md`; `example_stls/` is a junction to real supported STLs (gitignored — copyrighted).

## What This Is

A tool that packs pre-supported resin (MSLA) models onto build plates using true irregular 2D footprints with free rotation, minimizing plate count. Input: supported STL/OBJ files (e.g. Lychee "Export 3D Asset"). Output: one merged STL per plate (v1), later 3MF.

## Settled Design Decisions (do not relitigate for v1)

These were deliberately chosen in `PLATEPACKER_SEED.md`; don't propose alternatives unless the user reopens them:

1. **Footprint = vertical shadow** — union of all mesh triangles projected straight down to XY. Conservative but provably collision-free. Height-aware Z-banded masks are v2.
2. **Raster, not vector** — footprints are binary numpy masks (~0.05–0.1 mm/px, configurable). No shapely/NFP/polygon clipping; non-manifold meshes must rasterize fine because topological validity is never required.
3. **Collision via FFT cross-correlation** — `scipy.signal.fftconvolve` of plate occupancy vs. candidate mask gives overlap at every translation at once; zero pixels = legal placements. One convolution per rotation step.
4. **"Doesn't fit" is provable, not a timeout** — no zeros across all rotations means the piece spills to a new plate. The objective is minimizing plate count.
5. **Open formats only** — never parse or write proprietary slicer projects (.lys, .chitubox). Merged STL per plate is the v1 interchange.

## Pipeline Architecture

```
validate pieces → extract footprints (dilated by min-spacing margin)
  → sort largest-area first → greedy pack with spillover
  → [optional, later] time-budgeted improvement loop
  → export plates (rigid transform + concatenate) → layout report
```

Key structural rules baked into the design:

- **Margin dilation happens at load time, not extraction** (ADR-009 superseded the seed doc here): cached footprints are undilated/intrinsic at canonical 0.05 mm/px; `loading.prepare_mask` applies spacing + working resolution. The packer itself stays margin-unaware.
- **Validation runs before packing**: pieces taller than build volume or too big for an empty plate at any rotation are hard per-piece errors, never mysterious spillover.
- **Plate masks pre-encode unusable regions** (dead margins, chamfered corners), so every zero pixel is a printable placement.
- **Placement heuristic is pluggable** (start with bottom-left).
- **The improvement loop is an optional layer bolted onto greedy** — every iteration holds a complete valid solution, so early stopping is always safe. Build greedy first.

## The Coordinate Landmine

Packer works in pixel space; slicers expect millimeters with origin at plate center. Export must convert pixel → mm → recenter about plate midpoint. Rotation is about Z only, XY translation only, Z untouched. This conversion is the most likely source of subtle bugs — the pixel↔mm↔pixel round-trip is a priority test target.

## End-to-End Self-Check

Rasterize the merged output mesh's shadow and diff against the predicted occupancy mask. A match validates extraction, packing, transform, and export in one step. Implement this early and keep it as a test.

## Tech Stack & Tooling

- Python 3.11+, `trimesh` (mesh IO, transforms, export), `numpy`, `scipy` (fftconvolve), `opencv-python` (rasterization — one `fillConvexPoly` call per triangle; a batched `fillPoly` call XORs overlapping polygons, see bugs.md)
- **Project will be open-sourced on GitHub** with GitHub Actions CI (tests + lint on push/PR).
- **`uv`** for packaging/deps/env (pyproject.toml); the pip-created `.venv` is interim until migration. **`ruff`** for lint + format, enforced in CI. (ADR-006)
- `pytest` for tests. Priority targets: rasterization correctness on known shapes, coordinate round-trip, spillover behavior, merged-shadow self-check.
- Config: single dataclass or TOML — plate dims + unusable-region mask, build volume Z, raster resolution, min spacing, rotation steps, placement heuristic, improvement time budget.
- Later maybe: `cupy` (GPU convolution), `typer`/`argparse` CLI.

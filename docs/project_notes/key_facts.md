# Key Project Facts

This file tracks important project configuration, constants, and environment details.

## Project Overview
- **Project Name**: Plate Packer (Resin Plate Packer)
- **Description**: Packs pre-supported resin (MSLA) models onto build plates using true irregular 2D footprints with free rotation, minimizing plate count. Design doc: `PLATEPACKER_SEED.md`.

## Local Development
- **OS / Runtime**: Windows 11, Python 3.11+
- **Primary Workflow**: CLI tool (planned): folder of supported STLs + printer config in → `plate_01.stl`, `plate_02.stl`, … + layout report out
- **Setup**: `uv sync` from repo root (src-layout package, hatchling). Test STL inputs in `example_stls/` (junction → `C:\dev\stl_curator\example_stls`).

## Tooling & Distribution (decided 2026-08-01)
- **Repo**: https://github.com/jaydee829/plate-packer (public, MIT license)
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — ruff check + format-check + pytest on Python 3.11 and 3.14, on push to main and PRs.
- **Never commit `example_stls/`** — pre-supported Patreon/Kickstarter models are copyrighted third-party content (gitignored).
- **Package/env management**: `uv` (pyproject.toml-based). The current pip-created `.venv` is interim scaffolding — migrate to `uv` when the package is formalized.
- **Lint/format**: `ruff` (linter + formatter), enforced in CI.

## Technology Stack
- **Core Libraries**: `trimesh` (mesh IO, transforms, 3MF/scene export), `numpy`, `scipy` (`signal.fftconvolve`), `opencv-python` (`cv2.fillPoly` rasterization)
- **Later / Optional**: `cupy` (GPU convolution), `typer` or `argparse` (CLI)
- **Testing**: `pytest`. Priority targets: rasterization correctness on known shapes, pixel↔mm↔pixel round-trip, spillover behavior, merged-shadow end-to-end self-check.

## stl_curator Interface (normative — ADR-009)
- **Contract location**: `C:\dev\stl_curator\docs\superpowers\specs\2026-08-01-stl-curator-m1-design.md` §4. Changes require updating both projects and that section.
- **Footprint docs**: `footprints/<first-2-hex>/<sha256-hex>.json`, shared `footprints_dir` config. One STL → one doc → many footprints (z-slices). We own/version the JSON internals; curator only records existence.
- **Docs hold intrinsic data only** (undilated masks, canonical res); spacing/res applied at packer load time.

## Deferred Hardening (do when ADR-008 escalates extraction to multiprocessing)
- `footprint_io.save_doc` uses a deterministic tmp filename (`<sha>.json.tmp`). Two workers racing on the same input could collide mid-write. Switch to a process-unique tmp name (tempfile.mkstemp-style, same directory) BEFORE parallelizing extraction. (PR #1 review, 2026-08-01 — not a live issue while extraction is serial.)

## Export-Milestone TODO (from footprint-io final review, 2026-08-01)
- `prepare_mask` returns only the mask and discards the origin shift its dilation introduces (pads all sides → origin moves by −r·working_res; ragged-edge downsample pads bottom/right only → origin unchanged). The export milestone must have `prepare_mask` return the origin offset rather than callers re-deriving it (see the shim in `scripts/extract_footprint.py` for intended semantics).

## Domain Constants & Conventions
- **Raster resolution**: ~0.05–0.1 mm/px, config knob; tune empirically (0.1 mm/px on a ~200×130 mm plate → ~2000×1300 masks).
- **Rotation steps**: ~36–72, config knob.
- **Coordinate convention (landmine)**: packer works in pixel space; slicers expect millimeters with origin at plate center. Export = pixel → mm → recenter about plate midpoint. Rotation about Z only; Z never touched.
- **Margin handling (ADR-009)**: cached footprints are undilated at canonical 0.05 mm/px; spacing dilation + conservative downsample happen at load time (`plate_packer.loading.prepare_mask`); packer core is margin-unaware.
- **Config surface**: plate dims (mm) + optional unusable-region mask, build volume Z, raster resolution, min spacing, rotation steps, placement heuristic, improvement time budget — single dataclass or TOML.

## Usage Tips
- Organize facts by category; prefer bullet lists over tables for easy editing.
- Prefer documented facts here over assumptions when looking up config.

## SECURITY — What NOT to Store

This file is committed to version control. **Never** put secrets here:

- ❌ Passwords, API keys, tokens, private keys, connection strings with embedded credentials
- ❌ `.env` file contents, OAuth client secrets, signing keys, certificates
- ❌ Anything you would not paste into a public PR

Instead, store:

- ✅ The **name/location** of a secret and how to obtain it
- ✅ Non-secret config: ports, hostnames, public URLs, project IDs

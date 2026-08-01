# Key Project Facts

This file tracks important project configuration, constants, and environment details.

## Project Overview
- **Project Name**: Plate Packer (Resin Plate Packer)
- **Description**: Packs pre-supported resin (MSLA) models onto build plates using true irregular 2D footprints with free rotation, minimizing plate count. Design doc: `PLATEPACKER_SEED.md`.

## Local Development
- **OS / Runtime**: Windows 11, Python 3.11+
- **Primary Workflow**: CLI tool (planned): folder of supported STLs + printer config in → `plate_01.stl`, `plate_02.stl`, … + layout report out
- **Setup**: No package config yet (pre-code). Test STL inputs go in `example_stls/`.

## Tooling & Distribution (decided 2026-08-01)
- **Open source**: project will be published on GitHub.
- **CI**: GitHub Actions (test + lint on push/PR) once the repo is on GitHub.
- **Package/env management**: `uv` (pyproject.toml-based). The current pip-created `.venv` is interim scaffolding — migrate to `uv` when the package is formalized.
- **Lint/format**: `ruff` (linter + formatter), enforced in CI.

## Technology Stack
- **Core Libraries**: `trimesh` (mesh IO, transforms, 3MF/scene export), `numpy`, `scipy` (`signal.fftconvolve`), `opencv-python` (`cv2.fillPoly` rasterization)
- **Later / Optional**: `cupy` (GPU convolution), `typer` or `argparse` (CLI)
- **Testing**: `pytest`. Priority targets: rasterization correctness on known shapes, pixel↔mm↔pixel round-trip, spillover behavior, merged-shadow end-to-end self-check.

## Domain Constants & Conventions
- **Raster resolution**: ~0.05–0.1 mm/px, config knob; tune empirically (0.1 mm/px on a ~200×130 mm plate → ~2000×1300 masks).
- **Rotation steps**: ~36–72, config knob.
- **Coordinate convention (landmine)**: packer works in pixel space; slicers expect millimeters with origin at plate center. Export = pixel → mm → recenter about plate midpoint. Rotation about Z only; Z never touched.
- **Margin handling**: minimum-spacing dilation applied once at footprint extraction; packer is margin-unaware.
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

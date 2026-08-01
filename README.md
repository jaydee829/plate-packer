# Plate Packer

Packs pre-supported resin (MSLA) models onto build plates using **true irregular footprints** with free rotation, minimizing plate count. Existing slicer auto-layout tools use bounding boxes and waste plate area; resin print time depends only on Z-height, so every extra model packed onto a plate is free throughput.

**Status: early prototype.** The design is documented in [PLATEPACKER_SEED.md](PLATEPACKER_SEED.md).

## Approach

- **Footprint = vertical shadow**: every triangle of the supported mesh (model + supports + raft) projected onto XY — provably collision-safe.
- **Raster, not vector**: footprints are binary masks; no polygon clipping, so non-manifold meshes are fine.
- **Collision via FFT cross-correlation**: one convolution per rotation evaluates every translation at once; "doesn't fit" is provable, not a timeout.
- **Open formats only**: input is supported STL/OBJ (e.g. Lychee "Export 3D Asset"); output is one merged STL per plate.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # create env and install deps
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format

# prototype: extract footprint masks from a folder of supported STLs
uv run python scripts/extract_footprint.py example_stls --res 0.1 --spacing 0.5
```

Project knowledge (ADRs, bug log, key facts) lives in [docs/project_notes/](docs/project_notes/).

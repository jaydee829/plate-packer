# Resin Plate Packer — Project Seed

## Problem Statement

Resin (MSLA) print time depends on Z-height, not object count — every additional model packed onto a build plate is essentially free throughput. Existing slicer auto-layout tools (Chitubox, Lychee) use bounding boxes or crude margins and waste significant plate area. This tool packs pre-supported models onto resin build plates using true irregular footprints with free rotation, then exports an arrangement the slicer can consume directly.

The core problem is 2D irregular polygon nesting (NP-hard). Prior art: Deepnest (laser cutting, No-Fit Polygon + metaheuristic approach). We are deliberately taking the **raster/GPU-friendly approach instead of NFP** — it handles arbitrary concavity for free, sidesteps polygon-clipping robustness bugs on ugly meshes, and makes rotation search cheap.

## Core Design Decisions (settled — don't relitigate in v1)

1. **Footprint = vertical shadow.** Project every triangle of the supported mesh (model + supports + raft) straight down onto XY; the footprint is the union. This automatically captures support branches that flare wider than the raft and model overhangs wider than the supports. If two shadows don't overlap, the models cannot collide at any height. Conservative (forbids tall-over-short interleaving) but *correct*. Height-aware Z-banded masks are a v2 extension, not part of v1.

2. **Raster, not vector.** Rasterize footprints into binary numpy masks (~0.05–0.1 mm/px; make it a config knob and determine empirically). Union = OR-ing pixels. No shapely boolean unions — non-manifold garbage meshes from Kickstarters rasterize fine because we never need topological validity, only pixel coverage.

3. **Collision via FFT cross-correlation.** Convolving the plate occupancy mask with a candidate's mask computes overlap for every translation simultaneously; zero-valued pixels in the result are legal placements. One convolution per rotation (~36–72 rotation steps). `scipy.signal.fftconvolve` on CPU first; cupy later if needed.

4. **"Doesn't fit" is detectable, not a timeout.** If every rotation's correlation map has no zeros, the piece provably cannot be placed on this plate → it **spills over** to start the next plate. The real objective is minimizing plate count (2D bin packing), since the print queue is measured in plates.

5. **Input contract = supported STL/OBJ. Output = open formats.** The tool stays deliberately ignorant of proprietary slicer project formats (.lys, .chitubox — closed, unparseable, version-fragile). Lychee's "Export 3D Asset" bakes supports/raft into a plain STL; that's the sanctioned interchange point. Pre-supported Patreon/Kickstarter files already ship as supported STLs.

## Pipeline

```
validate pieces → extract footprints → sort (largest-area first)
  → greedy pack with spillover → [optional] improvement loop (time-budgeted)
  → export plates → report layouts + warnings
```

### 1. Footprint extraction
- Load mesh with `trimesh`.
- Project all triangles to XY, fill into a binary mask via `cv2.fillPoly` (or an orthographic top-down depth render via trimesh/pyrender — pick whichever proves more robust on real files).
- Dilate the mask by a configurable **minimum spacing** margin (parts must not fuse; resin needs drainage paths). Doing the dilation once at extraction time keeps the packer itself margin-unaware.
- Record mesh Z-height for build-volume validation.

### 2. Validation (before any packing)
- Piece taller than build volume → hard error, report per-piece.
- Piece footprint doesn't fit an *empty* plate at any tested rotation → hard error ("doesn't fit your printer, period"), so it never surfaces as mysterious spillover.

### 3. Greedy packing
- Sort by footprint area, descending (standard heuristic).
- For each piece: for each rotation, fftconvolve against current plate occupancy; collect zero-pixels as legal placements; choose by placement heuristic (start with bottom-left; make it pluggable).
- No legal placement on any open plate → open a new plate.
- Plate mask can pre-encode unusable regions (dead margins, chamfered corners) so all placements are guaranteed printable.

### 4. Improvement loop (optional layer, bolts on later — build greedy first)
- Anytime metaheuristic (simulated annealing / random restarts): perturb insertion order or a rotation, re-run greedy, keep if plate count drops or first-plate density improves.
- Time/iteration budget; every iteration holds a complete valid solution, so early stopping costs quality, never correctness.

### 5. Export
- Apply each placement as a 4×4 rigid transform (rotation about Z + XY translation; Z untouched): `trimesh.transformations` + `mesh.apply_transform`.
- **v1: one merged STL per plate** (`trimesh.util.concatenate`). Slicer sees one object and can't auto-arrange it out from under us. Since everything is pre-supported, per-object identity in the slicer isn't needed.
- **v1.5: 3MF per plate** via `trimesh.Scene` — preserves per-object transforms and identity. Needs round-trip testing against the actual slicer version before trusting.
- **Coordinate convention is the landmine:** packer works in pixel space; slicers expect millimeters with origin at plate center. Export = pixel → mm → recenter about plate midpoint. Get this wrong and everything imports shifted half a plate.

### Validation trick (end-to-end self-check)
Rasterize the *merged output mesh's* shadow and diff it against the predicted occupancy mask. A match proves the whole chain — extraction, packing, transform, export — is consistent.

## Tech Stack

- Python 3.11+
- `trimesh` (mesh IO, transforms, scene/3MF export), `numpy`, `scipy` (fftconvolve), `opencv-python` (fillPoly rasterization)
- Later maybe: `cupy` (GPU convolution), CLI via `typer` or `argparse`
- Tests: `pytest`. Priority test targets: rasterization correctness on known shapes, coordinate round-trip (pixel↔mm↔pixel), spillover behavior, the merged-shadow validation check.

## Config (single dataclass or TOML)

- Plate dimensions (mm) + optional unusable-region mask; build volume Z
- Raster resolution (mm/px)
- Minimum spacing (mm)
- Rotation steps
- Placement heuristic
- Improvement-loop time budget

## Roadmap

- **v1:** shadow extraction → greedy pack + spillover → merged STL export → CLI taking a folder of supported STLs + printer config, emitting `plate_01.stl`, `plate_02.stl`, … plus a text/PNG layout report.
- **v1.5:** 3MF export; layout preview images (matplotlib render of masks with labels).
- **v2:** Z-banded masks (per-height-band collision → allows tall-over-short interleaving); annealing improvement layer; edge-avoidance weighting for heavy models (peel forces are worst near plate edges).
- **v3 (speculative):** `.lyt` writer if that format ever gets decoded (references original STLs + placements → arrangement reopens natively in Lychee with editable supports). Quarantined as an optional helper, never a core dependency. Integration with a companion STL library/curator tool.

## Open Questions (resolve empirically, early)

1. What raster resolution balances accuracy vs. convolution cost on real ~200×130 mm plates? (0.1 mm/px → ~2000×1300 masks — measure fftconvolve timing.)
2. Do typical pre-supported Patreon files rasterize cleanly via fillPoly, or does the depth-render path prove more robust?
3. How many rotation steps before returns diminish? (Symmetric bases may need very few.)
4. Does the target slicer's 3MF import honor transforms correctly, or is merged STL the permanent answer?

## First Session Goal

Prototype footprint extraction (~20 lines: trimesh + OpenCV), run it against 2–3 real supported STL exports, eyeball the masks, and answer open questions 1–2. The extractor is the gating step; everything downstream consumes its output.

# Bug Log

This file tracks project bugs, their root causes, solutions, and prevention strategies.

## Templates

### YYYY-MM-DD - Brief Bug Description
- **Issue**: What went wrong (the observable symptom)
- **Root Cause**: Why it happened (the underlying cause, not the symptom)
- **Solution**: How it was fixed (specific change, file, or approach)
- **Prevention**: How to avoid it in the future

## Log

<!-- Add new entries below in reverse-chronological order (newest first). -->

### 2026-08-07 - MemoryError exporting a large merged plate (trimesh float64 caches)
- **Issue**: `pack` of the 30-piece selection at `--budget 3600` searched fine (4 plates, stall-stopped at eval 21) but crashed in `export_plates` → `merged.export()` with `MemoryError: Unable to allocate 89.7 MiB for (3918332, 3) float64`. A cousin of the 2026-08-01 verify OOM, but on the export path (untouched since v1).
- **Root Cause**: `export_plates` loaded every piece of a plate as a `trimesh.Trimesh`, `trimesh.util.concatenate`d them into one ~3.9M-face mesh, then `.export()`. Binary-STL export lazily builds several full-size **float64** caches (`self.triangles` (n,3,3)=282 MiB, `triangles_cross`, `face_normals`) on top of holding every piece mesh — ~1 GB peak for one plate. The 89.7 MiB was just the allocation that tipped an already-exhausted process.
- **Solution**: rewrote `export_plates` to stream the binary STL with numpy — for each piece, `read_stl_triangles` (lean, no mesh object) → `_transform_triangles` (float32 4×4) → `_stl_records` (float32 unit normals) → write records to the open file, backfilling the count. Peak memory is bounded to a **single piece** (~150 MB), never a whole plate, and never trimesh's float64 caches (commit on feat/rotation-resolution). Also echo the improvement summary before export so the fitness result survives a downstream failure.
- **Prevention**: never round-trip our own plate geometry through a full mesh loader/exporter when a triangle soup suffices — the same lesson as read_stl_triangles for verify. Peak memory must scale with one piece, not one plate. Production-scale plates (millions of triangles) are the only place this shows; unit-test boxes never will, so reason about the allocation, not just correctness.

### 2026-08-06 - Fractional edge_contact_weight silently truncated (uint8 pad before float cast)
- **Issue**: `contact_map`'s `edge_weight` knob (ADR-012) silently truncated any fractional value — `edge_contact_weight=0.5` disabled border attraction entirely (→ 0), `1.7`→`1`, `>255` wrapped mod 256. Only integer weights ≤255 worked. Found by the PR #6 GitHub review.
- **Root Cause**: `np.pad(plate, 1, constant_values=edge_weight)` runs on a `uint8` `plate`; `np.pad` preserves the input dtype, so `constant_values` is cast to uint8 BEFORE the later `.astype(np.float32)` — pure order-of-operations. Every plate/occupancy array in the codebase is uint8.
- **Solution**: cast to float first — `np.pad(plate.astype(np.float32), 1, constant_values=edge_weight)` (commit 59ed8dc). Regression: `test_contact_map_fractional_edge_weight_not_truncated` (0 < weight-0.5 border < weight-1.0 border).
- **Prevention**: when a float constant meets an integer array, cast the array first. Tests for a numeric knob must include a FRACTIONAL value — the existing edge_weight tests used only whole numbers (0/1/2/3), which round-trip losslessly through uint8 and hid the bug. Same lesson caught two sibling PR #6 findings: an angle-sort key that used the mirror rotation sign vs `rotate_mask` (invisible to symmetric-shape tests) and an unvalidated CLI `--coarse-res` override that could `round()` the coarse factor to 0 → raw `ZeroDivisionError`.

### 2026-08-06 - Coarse-to-fine search crashed on a fine-fitting piece with edge_margin_mm > 0
- **Issue**: With `edge_margin_mm > 0`, `improve()` could raise an uncaught, MISLEADING `ValueError("piece i does not fit an empty plate at any rotation")` for a piece that provably fits the fine plate — a hard crash, violating ADR-004 ("doesn't fit is provable, not an artifact"). Found by the ADR-012 final whole-branch review; default config (`edge_margin_mm=0.0`) is immune.
- **Root Cause**: Validation (CLI fit-check + improve's up-front check) runs at FINE resolution, but the coarse ILS packs against block-max-downsampled masks. Block-max grows BOTH the piece and the plate's margin frame, so a near-plate-spanning piece that fits the fine empty plate can fail to seat the coarse empty plate; `pack()` then hits its `target is None` contract-raise even under `validate=False`, and it propagates out of the coarse search.
- **Solution**: In `improve()`, after building coarse artifacts, check `all(any(_fits(coarse_plate_mask, m) for m in variants.values()) for variants in coarse_prerot)`; if any piece can't seat the coarse empty plate, fall back to fine resolution for the coarse phase (reassign the five `coarse_*` names to their fine equivalents). No crash, correct result, only the coarse speedup is lost for that run. No-op on the all-zero default coarse plate (commit c5554c2). Regression: `test_improve_survives_coarse_growth_of_margin_frame`.
- **Prevention**: Any lossy/conservative coarsening that grows masks can make a fine-legal piece coarse-illegal — validate at the SAME resolution you pack at, or degrade gracefully; never let a resolution artifact surface as a "doesn't fit" (ADR-004). When two code paths use different resolutions, test the seam with a piece sized to the boundary.

### 2026-08-01 - MemoryError in verify stage on first real-world pack
- **Issue**: `pack` of 30 Tome of Demons pieces crashed with `MemoryError: Unable to allocate 382 MiB` while verifying plate_01.stl (16.7M vertices, 266MB file); plates were written but report/verify never ran.
- **Root Cause**: verify reloaded merged plates via `trimesh.load_mesh`, which builds a Scene then deep-copies the mesh in `to_mesh()` (~3-4x the mesh in RAM), on top of all pack-stage arrays still alive in the process.
- **Solution**: `export.read_stl_triangles` reads binary STL triangle records directly with numpy (no mesh object, no copy, float32); CLI builds/writes the report before verification and frees mask arrays first (commit bd56cef).
- **Prevention**: never round-trip our own exported plates through a full mesh loader when only the triangle soup is needed; test with production-scale inputs early — the unit-test boxes could never surface allocation-order issues.
- **Issue**: `ruff format --check .` (the CI gate) failed on `docs/superpowers/plans/2026-08-01-export.md` — a docs-only file.
- **Root Cause**: ruff formats fenced python code blocks in .md files; the plan's hand-written snippets weren't format-clean.
- **Solution**: `uv run ruff format <file>` on the plan doc (commit 68b6cc7).
- **Prevention**: Run the format check before committing ANY file containing python fences, including docs/ markdown; or format plan docs at write time.

### 2026-08-01 - Typer single-command app made tests pass locally, fail on CI
- **Issue**: All 5 CLI tests exited 2 (usage error) on CI while passing locally; e2e had "verified" the CLI.
- **Root Cause**: `typer.Typer()` with exactly one registered command collapses it into the root command, so the token `footprints` was parsed as the first `paths` argument. Locally it passed by accident: the gitignored `./footprints/` output dir existed in cwd, satisfying `exists=True`.
- **Solution**: Added a no-op `@app.callback()` to preserve subcommand structure, plus a regression test that chdirs to a clean tmp cwd (commit 8f378e9).
- **Prevention**: Tests that touch CLI parsing must run from a cwd without repo artifacts (monkeypatch.chdir(tmp_path)). Distrust green tests whose inputs coincide with gitignored local state — CI's clean checkout is the arbiter.

### 2026-08-01 - NaN vertices in real pre-supported STLs crash extraction
- **Issue**: `ValueError: negative dimensions are not allowed` sizing the mask canvas for a real Archvillain STL (Decataur Pose 1 body).
- **Root Cause**: 192 of 1.27M triangles (0.015%) had all-NaN vertices; NaN propagated through bounds → canvas size cast to a garbage negative int. `mesh.bounds` was poisoned the same way.
- **Solution**: Filter non-finite triangles before computing bounds/z-height in `extract_footprint`; report a dropped-triangle count in stats and a WARNING in output.
- **Prevention**: `test_nonfinite_triangles_are_dropped` (NaN + inf cases). Assume every field of a Kickstarter mesh can be garbage — validate at load, never trust `mesh.bounds` on unprocessed meshes.

### 2026-08-01 - fillPoly multi-polygon call cancels overlapping triangles
- **Issue**: Footprint mask of a simple box had 37% coverage instead of ~100%; overlapping projected triangles produced holes.
- **Root Cause**: `cv2.fillPoly(mask, list_of_polys, 1)` applies the even-odd fill rule across the whole batch — overlapping polygons XOR out instead of unioning.
- **Solution**: Fill one triangle per call (`cv2.fillConvexPoly` in a loop) in `scripts/extract_footprint.py` so each fill ORs onto the mask.
- **Prevention**: Never batch overlapping polygons into one fillPoly call. The merged-shadow self-check (seed doc) would catch regressions; keep a known-shape rasterization test (box → ~100% coverage) in the pytest suite.

### 2026-08-01 - Spacing dilation clipped at mask borders
- **Issue**: Dilated margin couldn't extend past the canvas, silently truncating the minimum-spacing zone at footprint edges.
- **Root Cause**: Mask canvas was sized to the undilated footprint bounds; `cv2.dilate` keeps image size.
- **Solution**: Pad the canvas by the dilation radius on all sides before rasterizing (origin shifts by -r·res).
- **Prevention**: Test that a dilated mask's bounding box equals footprint + 2×spacing in both axes.

## Usage Tips

- Log bugs that are **recurring or instructive**, not every trivial typo.
- Keep each field to 1-3 lines. Link to the PR or commit that fixed it when useful.
- Search this file before debugging a familiar-feeling error
  (`Grep(pattern="connection refused", path="docs/project_notes/bugs.md", -i=true)`).
- Always lead with a date so entries stay temporally ordered.
- Likely future entry categories for this project: coordinate/offset bugs at export
  (pixel vs. mm, plate-center origin), rasterization artifacts on non-manifold meshes,
  fftconvolve numerical-tolerance issues (near-zero vs. zero pixels).

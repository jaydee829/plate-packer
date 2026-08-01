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

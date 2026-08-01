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

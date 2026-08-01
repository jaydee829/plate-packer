# Export Milestone Design — placements → merged plate STLs + runtime self-check

Date: 2026-08-01
Status: Approved (brainstorm with user; approach A — exact transform tracking)

## Goal

Complete the v1 loop from the seed doc: `plate-packer pack <folder>` takes supported
STL/OBJ files plus a printer config and emits `plate_01.stl`, `plate_02.stl`, … with a
text layout report, verifying every plate with the merged-shadow self-check at runtime.

## Non-Goals (v1)

- 3MF export, PNG layout previews (v1.5 per seed doc).
- `--copies` / quantity support — each discovered file is exactly one piece.
- Improvement loop; packing stays greedy.
- Changing the footprint cache contract (ADR-009 docs are untouched).

## 1. Config module — `src/plate_packer/config.py` (new)

Frozen dataclass `PackConfig`, loaded by `load_config(path: Path | None) -> PackConfig`.
Every field has a default (Anycubic Photon Mono X 6K); a missing file or missing keys
are fine. Unknown keys are ignored.

```toml
[printer]                       # defaults: Anycubic Photon Mono X 6K
plate_mm = [197.0, 122.0]       # X (width), Y (depth)
build_height_mm = 245.0

[packing]
working_res_mm = 0.1            # must be an integer multiple of canonical 0.05
spacing_mm = 2.0                # minimum gap between placed pieces (true gap, see ADR)
edge_margin_mm = 0.0            # extra dead band at plate borders
rotations = 8

[paths]
footprints_dir = "footprints"
output_dir = "plates"
```

Validation at load: plate dims, build height, working_res > 0; spacing, edge_margin ≥ 0;
rotations ≥ 1 (int); working_res an integer multiple of `CANONICAL_RES_MM` (same check
and error wording as `prepare_mask`). Violations raise `ValueError` naming the key.

`cli._default_footprints_dir` is deleted; both `footprints` and `pack` commands read
paths through `load_config`. CLI `--footprints-dir` still overrides.

### Spacing semantics (new ADR at implementation time)

The packer packs dilated-vs-dilated masks, so a per-piece dilation of `d` enforces a
`2d` gap between true footprints. Therefore:

- **`spacing_mm` is defined as the true minimum gap between placed pieces.**
- Dilation radius applied per piece: `r_px = ceil((spacing_mm / 2) / working_res_mm)`
  (ceil = conservative; 0 when spacing_mm == 0).
- Side effect (documented, accepted): each piece keeps ≥ spacing/2 clearance from plate
  edges because the dilated mask must fit on-plate.
- `edge_margin_mm` adds an occupied border band of `ceil(edge_margin_mm / working_res_mm)`
  pixels to the plate mask handed to `pack()` (0 px when 0.0).

## 2. Exact transform tracking

### 2.1 Coordinate conventions (pinned)

- Masks are indexed `[row, col]`; **row 0 = min Y**, so +row = +Y, +col = +X.
- Pixel coordinates are `(x=col, y=row)`. Continuous piece frame:
  `p_px = (p_mm − origin_mm) / res` — pixel index *i* sits at `origin + i·res`,
  matching `extract_footprint`'s `round()` rasterization.
- Plate frame: plate pixel `(r, c)` sits at `(c·res, r·res)` mm from the plate's
  min-X/min-Y corner. Slicer frame = plate frame recentered:
  `(x, y)_slicer = (x, y)_corner − (plate_w/2, plate_h/2)`.
- Any residual half-pixel convention error shifts the *whole layout* uniformly by
  ≤ res/2 relative to the plate edge; it cannot create inter-piece collision.

### 2.2 `loading.prepare_mask` — signature change

`prepare_mask(doc, spacing_mm, working_res_mm) -> tuple[np.ndarray, tuple[float, float]]`

Returns `(mask, origin_mm)` where `origin_mm` is the XY of the prepared mask's pixel
(0, 0): `doc.origin_mm` shifted by `−r_px · working_res_mm` per axis when dilation
padded the canvas (r_px per §1), unchanged otherwise. Conservative downsample never
moves the origin (blocks are anchored at pixel 0). Retires the Export-Milestone TODO
in key_facts.md. Internal dilation radius switches from `round(spacing/res)` to the
§1 formula.

### 2.3 `packer.rotate_mask` — signature change

`rotate_mask(mask, angle_deg) -> tuple[np.ndarray, np.ndarray]`

Returns `(rotated_mask, affine)` where `affine` is a 2×3 float64 matrix mapping input
pixel coords `(x, y, 1)` to output pixel coords, **including** the canvas expansion and
the data-dependent content crop:

- Right-angle path (`np.rot90`): exact permutation affine derived from k and the input
  shape (e.g. k=1: `(x, y) → (y, w−1−x)` in whatever orientation matches rot90's array
  semantics — derived and locked by the invariant test, not by trusting documentation).
- Arbitrary path: the adjusted `cv2.getRotationMatrix2D` matrix minus the final crop
  offset `(col_min, row_min)`.

Invariant (tested): for every input content pixel, `affine · (x, y, 1)` lands within
the output mask's content (allowing the ≤1 px growth binarization introduces — the
affine maps geometry exactly; the mask may only be larger).

`pack()` internals unpack the tuple; `Placement` is unchanged. Export re-derives the
affine by calling `rotate_mask` again for the placed angle (deterministic).

### 2.4 `export.placement_transform`

```python
def placement_transform(
    prepared_origin_mm: tuple[float, float],
    rotation_affine: np.ndarray,      # 2x3, from rotate_mask at placement.angle
    row: int, col: int,               # placement anchor, plate px
    working_res_mm: float,
    plate_mm: tuple[float, float],
) -> np.ndarray:                      # 4x4 rigid transform for trimesh
```

Composes, entirely as 2D affines, then lifts to 4×4 (Z untouched):

1. mesh mm → prepared px: `(p − prepared_origin) / res`
2. prepared px → rotated px: `rotation_affine`
3. rotated px → plate px: `+ (col, row)`
4. plate px → corner mm: `× res`
5. corner mm → slicer mm: `− (plate_w/2, plate_h/2)`

The composite is exactly rigid (unit scale: the two `res` scalings cancel; the rotation
part is orthonormal). The world rotation falls out of the composed matrix — the nominal
angle's sign convention is never trusted. Assert orthonormality (`R·Rᵀ ≈ I`,
`det ≈ +1`) and raise if violated (catches accidental mirroring).

### 2.5 `export.export_plates`

```python
def export_plates(
    files: list[Path],                # piece index -> source file
    placements: list[Placement],
    transforms: list[np.ndarray],     # piece index -> 4x4
    output_dir: Path,
) -> list[Path]                       # written plate files, plate order
```

Groups placements by plate; per plate, loads each source mesh
(`trimesh.load_mesh(process=False)`, Scene-concatenate like the CLI), applies its 4×4,
`trimesh.util.concatenate`, writes binary `plate_01.stl` (1-based, zero-padded to 2)
into `output_dir` (created if missing), then frees that plate's meshes — memory stays
bounded to one plate. Z is never modified: pre-supported rafts keep their source Z.

## 3. Runtime self-check — `export.verify_plate`

```python
def verify_plate(
    plate_mesh: trimesh.Trimesh,      # the merged mesh just written
    occupancy: np.ndarray,            # predicted plate occupancy, uint8 {0,1}
    working_res_mm: float,
    plate_mm: tuple[float, float],
    spacing_mm: float,
) -> int                              # count of violating pixels (0 = pass)
```

- Rasterizes the merged mesh's shadow directly into a plate-shaped canvas in the plate
  frame (§2.1), reusing the per-triangle `fillConvexPoly` approach (non-finite triangles
  filtered as in `extract_footprint`). Triangles extending outside the canvas count as
  violations (clipped pixels are compared where representable; a vertex outside plate
  bounds is a failure by construction since occupancy is all-inside).
- Assertion: **actual shadow ⊆ predicted occupancy** (subset, not equality — occupancy
  legitimately contains spacing margins). When `spacing_mm == 0`, the predicted mask is
  dilated by 1 px first as rounding tolerance.
- The `occupancy` argument is reconstructed by the caller (the CLI) from placements +
  rotated masks (deterministic re-placement, same `|=` loop as `pack`) — `pack()`'s API
  stays unchanged.

CLI behavior: verify every plate after writing (default on; `--no-verify` skips).
Failure: report plate filename + violating pixel count, keep all output files on disk,
exit 1.

## 4. CLI — `plate-packer pack`

```
plate-packer pack PATHS... [--config config.toml] [--out DIR] [--footprints-dir DIR]
                           [--no-verify] [--force]
```

- `--config` default `./config.toml` (missing file → pure defaults). `--out` /
  `--footprints-dir` override the config's paths.
- `--force` regenerates footprint docs even if current (passed through to the cache step).

Flow:

1. **Discover**: same recursive `.stl`/`.obj` discovery + resolved-path dedupe as
   `footprints` (shared helper). Empty → "no STL/OBJ files found", exit 0.
2. **Cache**: for each file, hash; generate the footprint doc if missing/stale (inline,
   same code path as the `footprints` command). A file that fails extraction is a hard
   error (collected, see step 3) — pack needs every piece.
3. **Load + validate (before packing)**: `load_doc` → `prepare_mask`. Collect *all*
   per-piece errors: extraction/load failures; `z_height_mm > build_height_mm`;
   no legal placement on an empty plate at any rotation (checked via
   `legal_placement_map` over the prepared plate mask). If any: print every error with
   filename and reason, exit 1. No silent skipping, no partial packs.
4. **Pack**: `pack(masks, plate_shape, rotations, plate_mask)` — plate shape =
   `(round(plate_y/res), round(plate_x/res))`, plate mask carries the edge-margin band.
5. **Export**: build transforms (§2.4), `export_plates` (§2.5).
6. **Verify**: §3 unless `--no-verify`.
7. **Report**: console + `<out>/report.txt` — per plate: filename, piece count,
   occupancy % (true footprint pixels ÷ usable plate pixels); per piece: source file
   name, final X/Y mm in slicer frame (piece footprint centroid), rotation in degrees
   CCW (as reported by the composed transform, not the nominal angle). Summary line:
   N pieces → M plates.

## 5. Testing

Parametrized, atomic tests throughout (no loops inside test bodies).

- **`config`**: defaults with no file; TOML overrides; each validation error
  (parametrized key × bad value); unknown keys ignored.
- **`prepare_mask` origin**: dilation shifts origin by exactly −r_px·res per axis;
  spacing=0 and factor=1 shift nothing; r_px matches the ceil(spacing/2/res) formula
  (parametrized spacing × res).
- **`rotate_mask` affine invariant**: for asymmetric masks × angles
  {0, 90, 180, 270, 30, 45, 137.5}: every input content pixel maps into output content;
  output content bbox is tight (crop included in affine).
- **Round-trip (seed-doc priority)**: `placement_transform` on synthetic shapes —
  transform a piece-frame pixel's mm position, convert back to plate px, assert it lands
  at anchor+rotated position, parametrized angles × anchors × resolutions.
- **Orthonormality**: composed transforms satisfy R·Rᵀ ≈ I, det ≈ +1 (parametrized
  angles); mirror injection raises.
- **Absolute coordinates**: a known box at a known placement → assert exported vertex
  mm values (plate-center origin, Z untouched). Catches sign/mirror/centering errors
  subset checks can structurally miss.
- **`verify_plate`**: passing case (exported synthetic plate); failing cases — mesh
  shifted by > spacing, mesh outside plate bounds; spacing=0 tolerance path.
- **End-to-end CLI**: tiny synthetic STLs → plates written, valid STLs, report content,
  self-check passes; failure paths — too-tall piece, doesn't-fit piece (both listed,
  exit 1), `--no-verify` skips verification; clean-cwd regression discipline
  (monkeypatch.chdir per bugs.md).

## Deferred / recorded

- The §1 spacing redefinition becomes a new ADR in decisions.md when implemented.
- key_facts.md Export-Milestone TODO is deleted when §2.2 lands.
- Deferred hardening (unique tmp filename) remains deferred — extraction is still serial.

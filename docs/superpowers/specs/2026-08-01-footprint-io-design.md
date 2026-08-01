# Footprint I/O & CLI — Design

Date: 2026-08-01
Status: Approved in brainstorming session (this doc is the written record)
Implements: ADR-009 (adopt stl_curator contract, intrinsic-only cache docs)
Contract: `C:\dev\stl_curator\docs\superpowers\specs\2026-08-01-stl-curator-m1-design.md` §4 (normative)

## 1. Scope

Make footprints a shared, content-addressed cache per the stl_curator contract,
and expose generation through the first real CLI entry point:

- Move extraction into the package, **undilated** (ADR-009).
- New footprint document I/O: hashing, contract paths, versioned JSON schema.
- New load step: cache doc → pack-ready mask (dilate + conservative downsample).
- `plate-packer footprints` CLI subcommand (typer).

Out of scope: pack/export CLI (v1 roadmap), Z-band extraction (v2), any
stl_curator-side changes (none needed).

## 2. Decisions made in this session

- **Canonical cache resolution: fixed 0.05 mm/px.** Measured on the largest
  real model (1.67M triangles): extraction time is identical to 0.1 (the
  per-triangle loop dominates, not pixel fill) and PNGs are 18 KB vs 7 KB.
  0.05 keeps the tight-spacing door open: safety invariant res <= spacing/4
  allows spacing down to 0.2 mm without regenerating the library. Schema is
  versioned, so a future resolution change is a clean regeneration
  (footprints are cache-tier, regenerable — contract §4.6).
- **CLI subcommand now**, not library-only: starts the v1 CLI skeleton and
  gives stl_curator something runnable today.
- Packing continues at a coarser **working resolution** (0.1 mm/px today,
  config); the load step downsamples conservatively.

## 3. Units

| Unit | Responsibility | Depends on |
|---|---|---|
| `src/plate_packer/footprint.py` | mesh → undilated mask + stats (extraction, moved from scripts/) | trimesh, cv2, numpy |
| `src/plate_packer/footprint_io.py` | SHA-256, contract path derivation, save/load of versioned JSON docs (base64 PNG masks, atomic writes) | cv2, numpy (no trimesh) |
| `src/plate_packer/loading.py` | doc → pack-ready mask: conservative downsample to working res, then spacing dilation | footprint_io, cv2 |
| `src/plate_packer/cli.py` | `plate-packer footprints PATHS...` (typer) | all above |
| `scripts/extract_footprint.py` | thin wrapper kept for mask eyeballing/timing; delegates to the package | package |

The contract lives entirely in `footprint_io.py`; extraction and packing can
change without touching it, and vice versa.

## 4. Document schema (v1)

Path: `<footprints_dir>/<sha256[:2]>/<sha256>.json`, written atomically
(tmp file + `os.replace`) so concurrent readers never see torn JSON.

```json
{
  "schema_version": 1,
  "generator": "plate-packer 0.1.0",
  "stl_sha256": "<64 hex chars>",
  "res_mm_per_px": 0.05,
  "origin_mm": [x_min, y_min],
  "z_height_mm": 96.3,
  "triangles": 1274046,
  "dropped_nonfinite": 192,
  "footprints": [
    {"kind": "full_shadow", "z_band_mm": [0.0, null], "mask_png_b64": "<base64>"}
  ]
}
```

- Masks are **undilated** and at canonical resolution only.
- `footprints` is an array so v2 Z-band entries
  (`{"kind": "z_band", "z_band_mm": [lo, hi], ...}`) append without a schema
  break; v1 writes exactly one `full_shadow` entry with band `[0.0, null]`
  (null = top of model).
- Unknown keys are ignored on read (forward compatibility).
- A doc whose `schema_version` != current is treated as absent (regenerate).

## 5. Data flow

Generate: discover STL/OBJ recursively → stream SHA-256 → skip if doc exists
at current schema_version (idempotent; `--force` overrides) → load mesh →
extract undilated mask at 0.05 → write doc → per-file status line + summary
(N written / skipped / failed).

Load for packing: `load_piece(sha_or_path, spacing_mm, working_res_mm)` →
decode PNG → if working res coarser: conservative downsample (any occupied
source pixel marks the target pixel — coverage can grow, never shrink;
working res must be an integer multiple of canonical res, else ValueError) →
dilate by `round(spacing_mm / working_res_mm)` (elliptical kernel, canvas
pre-padded so the margin is never clipped). Raises on unknown schema_version.

## 6. Configuration

`footprints_dir` resolution order: CLI `--footprints-dir` flag →
`config.toml` (`[paths] footprints_dir`, gitignored; `config.example.toml`
committed, mirroring stl_curator) → default `footprints/`. No other config
keys in this milestone.

## 7. Error handling

- Per-file failures (unreadable file, corrupt mesh, zero finite triangles)
  are caught, reported in the summary, and do not halt the batch. Exit code
  1 if any file failed, 0 otherwise.
- Extraction keeps the NaN/inf triangle filtering and dropped-count
  reporting already in place.

## 8. Testing (parametrized, atomic cases)

1. Path derivation: known hashes → contract paths (`ab…/abcd….json`).
2. Round-trip: save → load yields pixel-identical mask and all metadata.
3. Curator opacity: doc parses with plain `json`, required keys present,
   mask field is valid base64 (no plate_packer imports needed).
4. Load-step dilation: the existing extraction-dilation test cases migrate
   here (mask spans footprint + 2×spacing; margin never clipped).
5. Conservative downsample: constructed masks where naive area-averaging
   would drop lone pixels — they must survive; solid shapes stay solid.
6. Idempotency: second run skips; `--force` rewrites; stale schema_version
   regenerates.
7. Extraction (moved tests): undilated box masks have exact dimensions;
   NaN/inf filtering unchanged.
8. CLI smoke: typer runner over a tmp tree with a synthetic STL — creates
   doc at contract path, correct summary, exit codes for success/failure.

## 9. Open questions

None. (Resolution and entry point resolved in this session; contract
questions were resolved by ADR-009.)

# Footprint I/O & CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Content-addressed footprint cache per the stl_curator contract (ADR-009), with extraction moved into the package (undilated), a dilate-on-load step, and a `plate-packer footprints` CLI.

**Architecture:** Five thin units: `footprint.py` (mesh → undilated mask), `footprint_io.py` (hashing + contract paths + versioned JSON docs — the *only* file that knows the curator contract), `loading.py` (doc → pack-ready mask: conservative downsample + spacing dilation), `cli.py` (typer entry point), and `scripts/extract_footprint.py` reduced to a wrapper.

**Tech Stack:** Python 3.11+, trimesh, numpy, opencv (`cv2`), typer, pytest. All commands run via `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-01-footprint-io-design.md`

## Global Constraints

- Canonical cache resolution: `CANONICAL_RES_MM = 0.05` (fixed; recorded in every doc).
- Schema version: `SCHEMA_VERSION = 1`. A doc with any other version is treated as absent.
- Doc path (contract, normative): `<footprints_dir>/<sha256[:2]>/<sha256>.json`.
- Masks in docs are UNDILATED, uint8 {0,1}, base64 PNG. Dilation happens only in `loading.py`.
- Doc writes are atomic: write `<name>.json.tmp` in the same directory, then `os.replace`.
- Working resolution must be an integer multiple of canonical resolution, else `ValueError`.
- Conservative rule everywhere: coverage may grow, never shrink (downsample = block max; rotation and dilation already comply).
- Tests: parametrized, one behavior per named case (global CLAUDE.md rule).
- Lint/format must pass: `uv run ruff check .` and `uv run ruff format --check .`
- Commit format: imperative summary + trailer lines:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01BroNsgeTx9fQGSExjHXHWN`

---

### Task 1: Package extraction module (undilated)

**Files:**
- Create: `src/plate_packer/footprint.py`
- Modify: `tests/test_footprint.py` (rewrite imports + dilation-dependent cases)
- Modify: `scripts/extract_footprint.py` (temporarily inline the dilation it loses — final wrapper form happens in Task 5)

**Interfaces:**
- Consumes: nothing new.
- Produces: `extract_footprint(mesh: trimesh.Trimesh, res_mm: float) -> tuple[np.ndarray, np.ndarray, dict]` — returns `(mask, origin_mm, stats)`. `mask` is uint8 {0,1}, row 0 = min Y, canvas is the exact triangle bounds (no margin padding). `origin_mm` is `np.ndarray` shape (2,): the XY position (mm, mesh coords) of pixel (0,0). `stats` keys: `triangles` (int), `dropped_nonfinite` (int), `z_height_mm` (float), `footprint_mm` (tuple of 2 floats), `mask_px` (tuple), `coverage` (float), `raster_s` (float). Raises `ValueError("mesh has no finite triangles")`.

- [ ] **Step 1: Write the failing tests** — rewrite `tests/test_footprint.py` to import from the package and expect exact undilated dimensions:

```python
"""Rasterization correctness tests on known shapes (seed doc priority target)."""

import numpy as np
import pytest
import trimesh

from plate_packer.footprint import extract_footprint

RES = 0.1


@pytest.mark.parametrize(
    "extents",
    [
        pytest.param((20, 10, 5), id="box-20x10"),
        pytest.param((5, 5, 50), id="tall-thin-box"),
        pytest.param((100, 3, 2), id="long-sliver"),
    ],
)
def test_box_mask_has_exact_undilated_dimensions(extents):
    """Canvas spans exactly the footprint: extents/res + 1 boundary pixel."""
    mesh = trimesh.creation.box(extents=extents)
    mask, _origin, _stats = extract_footprint(mesh, RES)
    assert mask.shape == (round(extents[1] / RES) + 1, round(extents[0] / RES) + 1)


def test_box_mask_is_solid():
    """Overlapping projected triangles must union, not cancel (fillPoly even-odd bug)."""
    mask, _origin, _stats = extract_footprint(trimesh.creation.box(extents=(20, 10, 5)), RES)
    assert mask.mean() > 0.97


def test_origin_is_min_corner_in_mesh_coords():
    """trimesh boxes are centered on the origin, so min corner = -extents/2."""
    mesh = trimesh.creation.box(extents=(20, 10, 5))
    _mask, origin, _stats = extract_footprint(mesh, RES)
    assert origin == pytest.approx([-10.0, -5.0])


@pytest.mark.parametrize(
    "corrupt",
    [pytest.param(float("nan"), id="nan-vertex"), pytest.param(float("inf"), id="inf-vertex")],
)
def test_nonfinite_triangles_are_dropped(corrupt):
    """NaN/inf vertices (seen in real pre-supported STLs) must not poison the mask."""
    mesh = trimesh.creation.box(extents=(20, 10, 5))
    vertices = np.vstack([mesh.vertices, [[corrupt, corrupt, corrupt]] * 3])
    n = len(mesh.vertices)
    faces = np.vstack([mesh.faces, [[n, n + 1, n + 2]]])
    corrupted = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mask, _origin, stats = extract_footprint(corrupted, RES)
    assert stats["dropped_nonfinite"] == 1
    assert stats["z_height_mm"] == pytest.approx(5.0)
    assert mask.mean() > 0.97


def test_all_nonfinite_mesh_raises():
    nan = float("nan")
    mesh = trimesh.Trimesh(vertices=[[nan, nan, nan]] * 3, faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError, match="no finite triangles"):
        extract_footprint(mesh, RES)


def test_stacked_boxes_shadow_unions():
    """Two stacked boxes (same shadow) must still produce a solid union."""
    a = trimesh.creation.box(extents=(10, 10, 2))
    b = trimesh.creation.box(extents=(10, 10, 2))
    b.apply_translation([0, 0, 5])
    mask, _origin, _stats = extract_footprint(trimesh.util.concatenate([a, b]), RES)
    assert mask.mean() > 0.97
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_footprint.py -q`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'plate_packer.footprint'`

- [ ] **Step 3: Create `src/plate_packer/footprint.py`** — the extraction logic from `scripts/extract_footprint.py`, minus all `spacing_mm`/dilation code:

```python
"""Footprint extraction: mesh -> undilated vertical-shadow mask (ADR-009).

Spacing dilation is applied at load time (plate_packer.loading), never here:
cached footprints are content-addressed by file hash alone, so they may only
contain data intrinsic to the STL.
"""

import time

import cv2
import numpy as np
import trimesh


def extract_footprint(mesh: trimesh.Trimesh, res_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mask, origin_mm, stats). Mask is uint8 {0,1}, row 0 = min Y,
    canvas is the exact triangle bounds. origin_mm = XY of pixel (0,0)."""
    t0 = time.perf_counter()
    tris = mesh.triangles  # (n, 3, 3) mm
    # Real-world pre-supported STLs can contain NaN vertices (seen in the wild
    # at ~0.015% of triangles); they poison the bounds and canvas size.
    finite = np.isfinite(tris).all(axis=(1, 2))
    n_bad = int((~finite).sum())
    if n_bad:
        tris = tris[finite]
    if len(tris) == 0:
        raise ValueError("mesh has no finite triangles")
    tris_xy = tris[:, :, :2]

    flat = tris_xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1

    mask = np.zeros((size_px[1], size_px[0]), dtype=np.uint8)
    tris_px = np.round((tris_xy - origin) / res_mm).astype(np.int32)
    # One fill call per triangle: a single multi-polygon fillPoly call uses
    # the even-odd rule, so overlapping triangles cancel instead of unioning.
    for tri in tris_px:
        cv2.fillConvexPoly(mask, tri, 1)
    t_raster = time.perf_counter() - t0

    stats = {
        "triangles": len(mesh.triangles),
        "dropped_nonfinite": n_bad,
        "z_height_mm": float(tris[:, :, 2].max() - tris[:, :, 2].min()),
        "footprint_mm": tuple((flat.max(axis=0) - flat.min(axis=0)).round(1)),
        "mask_px": mask.shape,
        "coverage": float(mask.mean()),
        "raster_s": t_raster,
    }
    return mask, origin, stats
```

- [ ] **Step 4: Keep the script importable** — in `scripts/extract_footprint.py`, delete the local `extract_footprint` function and replace with an import plus a local dilation shim so the script's `--spacing` flag still works until Task 5:

```python
import cv2  # keep existing imports; remove the now-unused ones flagged by ruff
from plate_packer.footprint import extract_footprint as _extract


def extract_footprint(mesh, res_mm, spacing_mm):
    mask, origin, stats = _extract(mesh, res_mm)
    if spacing_mm > 0:
        r = max(1, round(spacing_mm / res_mm))
        mask = np.pad(mask, r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kernel)
        origin = origin - r * res_mm
    return mask, origin, stats
```

- [ ] **Step 5: Run tests, lint**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check .`
Expected: all tests PASS (including untouched `tests/test_packer.py`), lint clean.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: move extraction into package, undilated (ADR-009)"
```
(with the standard trailer lines from Global Constraints)

---

### Task 2: Footprint document I/O (the contract file)

**Files:**
- Create: `src/plate_packer/footprint_io.py`
- Create: `tests/test_footprint_io.py`

**Interfaces:**
- Consumes: nothing from other tasks (works on numpy masks + plain data).
- Produces:
  - `CANONICAL_RES_MM: float = 0.05`, `SCHEMA_VERSION: int = 1`
  - `file_sha256(path: Path) -> str` (64 lowercase hex chars, streamed 1 MiB chunks)
  - `doc_path(footprints_dir: Path, sha: str) -> Path`
  - `save_doc(footprints_dir: Path, sha: str, mask: np.ndarray, origin_mm, stats: dict) -> Path` — atomic write; `stats` is the dict from `extract_footprint`
  - `load_doc(footprints_dir: Path, sha: str) -> FootprintDoc` — raises `ValueError` on unknown `schema_version`, `FileNotFoundError` if absent
  - `has_current_doc(footprints_dir: Path, sha: str) -> bool` — False when absent OR stale version
  - `@dataclass FootprintDoc`: `sha: str`, `res_mm_per_px: float`, `origin_mm: tuple[float, float]`, `z_height_mm: float`, `triangles: int`, `dropped_nonfinite: int`, `masks: list[np.ndarray]` (decoded, uint8 {0,1}; v1 always length 1)

- [ ] **Step 1: Write the failing tests** — `tests/test_footprint_io.py`:

```python
"""Contract-path, round-trip, and curator-opacity tests for footprint docs."""

import base64
import json

import numpy as np
import pytest

from plate_packer.footprint_io import (
    CANONICAL_RES_MM,
    SCHEMA_VERSION,
    doc_path,
    file_sha256,
    has_current_doc,
    load_doc,
    save_doc,
)

SHA_A = "a" * 64
STATS = {
    "triangles": 12,
    "dropped_nonfinite": 0,
    "z_height_mm": 5.0,
    "footprint_mm": (20.0, 10.0),
    "mask_px": (101, 201),
    "coverage": 0.99,
    "raster_s": 0.01,
}


def checker_mask():
    mask = np.zeros((8, 6), np.uint8)
    mask[::2, ::2] = 1
    return mask


@pytest.mark.parametrize(
    ("sha", "expected_parts"),
    [
        pytest.param("ab" + "0" * 62, ("ab", "ab" + "0" * 62 + ".json"), id="ab-prefix"),
        pytest.param("7f" + "e" * 62, ("7f", "7f" + "e" * 62 + ".json"), id="7f-prefix"),
    ],
)
def test_doc_path_follows_contract(tmp_path, sha, expected_parts):
    p = doc_path(tmp_path, sha)
    assert p == tmp_path / expected_parts[0] / expected_parts[1]


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "x.stl"
    f.write_bytes(b"solid x" * 1000)
    assert file_sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_round_trip_preserves_mask_and_metadata(tmp_path):
    mask = checker_mask()
    save_doc(tmp_path, SHA_A, mask, (-10.0, -5.0), STATS)
    doc = load_doc(tmp_path, SHA_A)
    assert doc.sha == SHA_A
    assert doc.res_mm_per_px == CANONICAL_RES_MM
    assert doc.origin_mm == (-10.0, -5.0)
    assert doc.z_height_mm == 5.0
    assert doc.triangles == 12
    assert doc.dropped_nonfinite == 0
    assert len(doc.masks) == 1
    assert (doc.masks[0] == mask).all()


def test_doc_is_plain_json_per_curator_contract(tmp_path):
    """The curator must be able to treat the doc as opaque JSON."""
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    raw = json.loads(doc_path(tmp_path, SHA_A).read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["stl_sha256"] == SHA_A
    assert raw["res_mm_per_px"] == CANONICAL_RES_MM
    assert raw["footprints"][0]["kind"] == "full_shadow"
    assert raw["footprints"][0]["z_band_mm"] == [0.0, None]
    base64.b64decode(raw["footprints"][0]["mask_png_b64"], validate=True)


def test_no_tmp_file_left_behind(tmp_path):
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param("none", False, id="absent"),
        pytest.param("current", True, id="current-version"),
        pytest.param("stale", False, id="stale-version-treated-as-absent"),
    ],
)
def test_has_current_doc(tmp_path, setup, expected):
    if setup in ("current", "stale"):
        save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    if setup == "stale":
        p = doc_path(tmp_path, SHA_A)
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["schema_version"] = 0
        p.write_text(json.dumps(raw), encoding="utf-8")
    assert has_current_doc(tmp_path, SHA_A) is expected


def test_load_doc_stale_version_raises(tmp_path):
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    p = doc_path(tmp_path, SHA_A)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_doc(tmp_path, SHA_A)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_footprint_io.py -q`
Expected: ERROR — `ModuleNotFoundError: No module named 'plate_packer.footprint_io'`

- [ ] **Step 3: Implement `src/plate_packer/footprint_io.py`:**

```python
"""Footprint cache documents: the stl_curator interface contract (ADR-009).

This is the ONLY module that knows the contract: content-addressed paths
(<dir>/<sha[:2]>/<sha>.json), the versioned JSON schema, and atomic writes.
Docs hold intrinsic data only — masks are undilated, at canonical resolution.
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CANONICAL_RES_MM = 0.05
SCHEMA_VERSION = 1
_GENERATOR = "plate-packer 0.1.0"


@dataclass(frozen=True)
class FootprintDoc:
    sha: str
    res_mm_per_px: float
    origin_mm: tuple[float, float]
    z_height_mm: float
    triangles: int
    dropped_nonfinite: int
    masks: list  # list[np.ndarray], uint8 {0,1}; v1 always length 1


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def doc_path(footprints_dir: Path, sha: str) -> Path:
    return Path(footprints_dir) / sha[:2] / f"{sha}.json"


def save_doc(footprints_dir, sha, mask, origin_mm, stats) -> Path:
    ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "stl_sha256": sha,
        "res_mm_per_px": CANONICAL_RES_MM,
        "origin_mm": [float(origin_mm[0]), float(origin_mm[1])],
        "z_height_mm": stats["z_height_mm"],
        "triangles": stats["triangles"],
        "dropped_nonfinite": stats["dropped_nonfinite"],
        "footprints": [
            {
                "kind": "full_shadow",
                "z_band_mm": [0.0, None],
                "mask_png_b64": base64.b64encode(png.tobytes()).decode(),
            }
        ],
    }
    path = doc_path(footprints_dir, sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)  # readers never see a torn doc
    return path


def load_doc(footprints_dir, sha) -> FootprintDoc:
    raw = json.loads(doc_path(footprints_dir, sha).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    masks = []
    for fp in raw["footprints"]:
        buf = np.frombuffer(base64.b64decode(fp["mask_png_b64"]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        masks.append((img > 0).astype(np.uint8))
    return FootprintDoc(
        sha=raw["stl_sha256"],
        res_mm_per_px=raw["res_mm_per_px"],
        origin_mm=tuple(raw["origin_mm"]),
        z_height_mm=raw["z_height_mm"],
        triangles=raw["triangles"],
        dropped_nonfinite=raw["dropped_nonfinite"],
        masks=masks,
    )


def has_current_doc(footprints_dir, sha) -> bool:
    p = doc_path(footprints_dir, sha)
    if not p.exists():
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return raw.get("schema_version") == SCHEMA_VERSION
```

- [ ] **Step 4: Run tests, lint**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check .`
Expected: all PASS, lint clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: footprint cache documents per stl_curator contract"
```

---

### Task 3: Load step (downsample + dilate)

**Files:**
- Create: `src/plate_packer/loading.py`
- Create: `tests/test_loading.py`

**Interfaces:**
- Consumes: `FootprintDoc` (Task 2).
- Produces:
  - `prepare_mask(doc: FootprintDoc, spacing_mm: float, working_res_mm: float) -> np.ndarray` — pack-ready uint8 {0,1} mask.
  - `conservative_downsample(mask: np.ndarray, factor: int) -> np.ndarray` (block max; pads bottom/right with zeros to a multiple of factor).
  - `dilate(mask: np.ndarray, r_px: int) -> np.ndarray` (pads canvas by r_px on all sides, elliptical kernel 2r+1).

- [ ] **Step 1: Write the failing tests** — `tests/test_loading.py`:

```python
"""Load-step tests: conservative downsample + spacing dilation (ADR-009)."""

import numpy as np
import pytest

from plate_packer.footprint_io import CANONICAL_RES_MM, FootprintDoc
from plate_packer.loading import conservative_downsample, dilate, prepare_mask


def make_doc(mask):
    return FootprintDoc(
        sha="a" * 64,
        res_mm_per_px=CANONICAL_RES_MM,
        origin_mm=(0.0, 0.0),
        z_height_mm=5.0,
        triangles=2,
        dropped_nonfinite=0,
        masks=[mask.astype(np.uint8)],
    )


def test_downsample_lone_pixel_survives():
    """A single occupied pixel must mark its coarse cell (naive averaging drops it)."""
    mask = np.zeros((8, 8), np.uint8)
    mask[3, 5] = 1
    out = conservative_downsample(mask, 2)
    assert out.shape == (4, 4)
    assert out[1, 2] == 1
    assert out.sum() == 1


def test_downsample_solid_stays_solid():
    out = conservative_downsample(np.ones((10, 10), np.uint8), 2)
    assert out.shape == (5, 5)
    assert out.all()


def test_downsample_pads_ragged_edges():
    """Sizes not divisible by factor pad with zeros; edge content survives."""
    mask = np.zeros((5, 7), np.uint8)
    mask[4, 6] = 1
    out = conservative_downsample(mask, 2)
    assert out.shape == (3, 4)
    assert out[2, 3] == 1


@pytest.mark.parametrize(
    ("mask_shape", "r"),
    [
        pytest.param((10, 20), 5, id="rect-r5"),
        pytest.param((1, 1), 3, id="single-pixel-r3"),
    ],
)
def test_dilate_pads_canvas_so_margin_never_clips(mask_shape, r):
    mask = np.ones(mask_shape, np.uint8)
    out = dilate(mask, r)
    assert out.shape == (mask_shape[0] + 2 * r, mask_shape[1] + 2 * r)
    # cardinal extremes of the elliptical kernel reach the canvas edge
    assert out[0, r + mask_shape[1] // 2] == 1
    assert out[out.shape[0] // 2, 0] == 1


def test_prepare_mask_downsamples_then_dilates():
    """0.05 doc at working res 0.1 with 0.5mm spacing: 2x downsample + 5px pad."""
    doc = make_doc(np.ones((40, 20), np.uint8))  # 2.0 x 1.0 mm at 0.05
    out = prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.1)
    assert out.shape == (20 + 10, 10 + 10)  # halved, then +5px each side
    assert out[15, 10] == 1


def test_prepare_mask_zero_spacing_skips_dilation():
    doc = make_doc(np.ones((40, 20), np.uint8))
    out = prepare_mask(doc, spacing_mm=0.0, working_res_mm=0.1)
    assert out.shape == (20, 10)


def test_prepare_mask_non_integer_ratio_raises():
    doc = make_doc(np.ones((4, 4), np.uint8))
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.075)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_loading.py -q`
Expected: ERROR — `ModuleNotFoundError: No module named 'plate_packer.loading'`

- [ ] **Step 3: Implement `src/plate_packer/loading.py`:**

```python
"""Cache doc -> pack-ready mask: conservative downsample + spacing dilation.

The single place where packer/printer config (working resolution, minimum
spacing) is applied to intrinsic cached footprints (ADR-009).
"""

import cv2
import numpy as np

from plate_packer.footprint_io import FootprintDoc

_RATIO_TOL = 1e-6


def conservative_downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Block max: any occupied source pixel marks the coarse cell."""
    if factor == 1:
        return mask
    h, w = mask.shape
    ph, pw = -h % factor, -w % factor
    padded = np.pad(mask, ((0, ph), (0, pw)))
    return (
        padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
        .max(axis=(1, 3))
        .astype(np.uint8)
    )


def dilate(mask: np.ndarray, r_px: int) -> np.ndarray:
    """Dilate with an elliptical kernel; canvas pre-padded so the margin
    is never clipped at the borders."""
    padded = np.pad(mask, r_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    return cv2.dilate(padded, kernel)


def prepare_mask(doc: FootprintDoc, spacing_mm: float, working_res_mm: float) -> np.ndarray:
    ratio = working_res_mm / doc.res_mm_per_px
    if abs(ratio - round(ratio)) > _RATIO_TOL or ratio < 1:
        raise ValueError(
            f"working res {working_res_mm} must be an integer multiple "
            f"of canonical res {doc.res_mm_per_px}"
        )
    mask = conservative_downsample(doc.masks[0], round(ratio))
    if spacing_mm > 0:
        mask = dilate(mask, max(1, round(spacing_mm / working_res_mm)))
    return mask
```

- [ ] **Step 4: Run tests, lint**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check .`
Expected: all PASS, lint clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: load step - conservative downsample + spacing dilation"
```

---

### Task 4: CLI (`plate-packer footprints`)

**Files:**
- Create: `src/plate_packer/cli.py`
- Create: `tests/test_cli.py`
- Create: `config.example.toml`
- Modify: `pyproject.toml` (add typer dep, `[project.scripts]`)
- Modify: `.gitignore` (add `config.toml`)

**Interfaces:**
- Consumes: `extract_footprint` (Task 1), `file_sha256`/`save_doc`/`has_current_doc` (Task 2).
- Produces: console script `plate-packer` → `plate_packer.cli:app` (typer). Command `footprints PATHS... [--footprints-dir DIR] [--force]`. `footprints_dir` resolution: flag → `config.toml` `[paths] footprints_dir` → default `Path("footprints")`.

- [ ] **Step 1: Add dependency and entry point**

Run: `uv add typer`
Then add to `pyproject.toml`:

```toml
[project.scripts]
plate-packer = "plate_packer.cli:app"
```

- [ ] **Step 2: Write the failing tests** — `tests/test_cli.py`:

```python
"""CLI smoke tests over a tmp tree with synthetic STLs."""

import json

import pytest
import trimesh
from typer.testing import CliRunner

from plate_packer.cli import app
from plate_packer.footprint_io import SCHEMA_VERSION, doc_path, file_sha256

runner = CliRunner()


@pytest.fixture
def stl_tree(tmp_path):
    root = tmp_path / "models" / "nested"
    root.mkdir(parents=True)
    trimesh.creation.box(extents=(4, 2, 1)).export(root / "brick.stl")
    trimesh.creation.box(extents=(2, 2, 2)).export(tmp_path / "models" / "cube.stl")
    return tmp_path / "models"


def test_footprints_writes_contract_docs(stl_tree, tmp_path):
    out = tmp_path / "fp"
    result = runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    assert result.exit_code == 0
    sha = file_sha256(stl_tree / "cube.stl")
    doc = json.loads(doc_path(out, sha).read_text(encoding="utf-8"))
    assert doc["schema_version"] == SCHEMA_VERSION
    assert "2 written" in result.output


def test_footprints_skips_cached_unless_forced(stl_tree, tmp_path):
    out = tmp_path / "fp"
    runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    second = runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    assert "2 skipped" in second.output
    forced = runner.invoke(
        app, ["footprints", str(stl_tree), "--footprints-dir", str(out), "--force"]
    )
    assert "2 written" in forced.output


def test_footprints_reports_failures_and_continues(stl_tree, tmp_path):
    bad = stl_tree / "corrupt.stl"
    bad.write_bytes(b"this is not an stl")
    result = runner.invoke(
        app, ["footprints", str(stl_tree), "--footprints-dir", str(tmp_path / "fp")]
    )
    assert result.exit_code == 1
    assert "2 written" in result.output
    assert "1 failed" in result.output
    assert "corrupt.stl" in result.output
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -q`
Expected: ERROR — `ModuleNotFoundError: No module named 'plate_packer.cli'`

- [ ] **Step 4: Implement `src/plate_packer/cli.py`:**

```python
"""plate-packer CLI (typer). First subcommand: footprints (generate cache docs)."""

import tomllib
from pathlib import Path

import trimesh
import typer

from plate_packer.footprint import extract_footprint
from plate_packer.footprint_io import (
    CANONICAL_RES_MM,
    file_sha256,
    has_current_doc,
    save_doc,
)

app = typer.Typer(no_args_is_help=True)

_EXTENSIONS = {".stl", ".obj"}


def _default_footprints_dir() -> Path:
    cfg = Path("config.toml")
    if cfg.exists():
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        value = data.get("paths", {}).get("footprints_dir")
        if value:
            return Path(value)
    return Path("footprints")


def _discover(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(q for q in sorted(p.rglob("*")) if q.suffix.lower() in _EXTENSIONS)
        elif p.suffix.lower() in _EXTENSIONS:
            files.append(p)
    return files


@app.command()
def footprints(
    paths: list[Path] = typer.Argument(..., exists=True),
    footprints_dir: Path = typer.Option(
        None, help="cache dir (default: config.toml or ./footprints)"
    ),
    force: bool = typer.Option(False, help="regenerate even if a current doc exists"),
):
    """Generate footprint cache documents for STL/OBJ files."""
    out_dir = footprints_dir or _default_footprints_dir()
    written = skipped = 0
    failures: list[tuple[Path, str]] = []
    for f in _discover(paths):
        try:
            sha = file_sha256(f)
            if not force and has_current_doc(out_dir, sha):
                skipped += 1
                continue
            mesh = trimesh.load_mesh(f, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            mask, origin, stats = extract_footprint(mesh, CANONICAL_RES_MM)
            save_doc(out_dir, sha, mask, origin, stats)
            written += 1
            typer.echo(f"  {f.name}: ok ({stats['mask_px'][1]}x{stats['mask_px'][0]}px)")
        except Exception as e:  # per-file failures never halt the batch
            failures.append((f, str(e)))
            typer.echo(f"  {f.name}: FAILED ({e})")
    typer.echo(f"{written} written, {skipped} skipped, {len(failures)} failed -> {out_dir}")
    if failures:
        raise typer.Exit(code=1)
```

- [ ] **Step 5: Create `config.example.toml`** (committed; real `config.toml` gitignored):

```toml
# Copy to config.toml and adjust. All keys optional.
[paths]
# Shared with stl_curator (interface contract §4.2).
footprints_dir = "footprints"
```

Append `config.toml` to `.gitignore` under the "Tool output" section.

- [ ] **Step 6: Run tests, lint**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check .`
Expected: all PASS, lint clean.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: plate-packer CLI with footprints subcommand"
```

---

### Task 5: Script wrapper + end-to-end verification

**Files:**
- Modify: `scripts/extract_footprint.py` (final thin-wrapper form)
- Modify: `README.md` (CLI usage)

**Interfaces:**
- Consumes: `extract_footprint` (Task 1), `dilate` (Task 3).
- Produces: nothing new (script is a dev utility; CLI is the product path).

- [ ] **Step 1: Reduce the script to a thin wrapper** — replace the Task 1 shim: keep argparse/PNG/timing/report code, import extraction from the package and dilation from `plate_packer.loading`:

```python
from plate_packer.footprint import extract_footprint as _extract
from plate_packer.loading import dilate


def extract_footprint(mesh, res_mm, spacing_mm):
    mask, origin, stats = _extract(mesh, res_mm)
    if spacing_mm > 0:
        r = max(1, round(spacing_mm / res_mm))
        mask = dilate(mask, r)
        origin = origin - r * res_mm
    return mask, origin, stats
```

Delete the numpy/cv2 dilation shim added in Task 1 and any imports ruff flags as unused.

- [ ] **Step 2: Full suite + lint**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check .`
Expected: all PASS, lint clean.

- [ ] **Step 3: End-to-end on real files** — run the CLI against two real STLs and verify a doc round-trips into the packer:

```bash
uv run plate-packer footprints \
  "example_stls/Archvillain Games - Tome of Demons Volume 1/Armaros, Chaos Incarnate/STL_Armaros_Head_Supported.stl" \
  --footprints-dir footprints
```

Expected: `1 written, 0 skipped, 0 failed`. Re-run → `0 written, 1 skipped`. Then:

```bash
uv run python -c "
from pathlib import Path
from plate_packer.footprint_io import file_sha256, load_doc
from plate_packer.loading import prepare_mask
from plate_packer.packer import pack
f = Path('example_stls/Archvillain Games - Tome of Demons Volume 1/Armaros, Chaos Incarnate/STL_Armaros_Head_Supported.stl')
doc = load_doc(Path('footprints'), file_sha256(f))
piece = prepare_mask(doc, spacing_mm=0.5, working_res_mm=0.1)
placements = pack([piece], (1300, 2000))
print('e2e ok:', piece.shape, placements[0])
"
```

Expected: prints `e2e ok:` with a Placement on plate 0.

- [ ] **Step 4: Update README** — replace the `scripts/extract_footprint.py` usage line in the Development section with:

```sh
# generate footprint cache docs (shared with stl_curator)
uv run plate-packer footprints example_stls --footprints-dir footprints
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: finish footprint I/O milestone - wrapper, README, e2e verified"
```

---

## Self-Review Notes

- Spec §3 units ↔ Tasks 1–5: all covered; `origin_mm` flows extraction → doc → (unused by packer yet, consumed by export milestone).
- Types consistent: `extract_footprint` 2-arg in package (Task 1), 3-arg only in the script wrapper (Tasks 1/5); `save_doc(dir, sha, mask, origin_mm, stats)` matches CLI usage (Task 4).
- No placeholders; every step has runnable code/commands.

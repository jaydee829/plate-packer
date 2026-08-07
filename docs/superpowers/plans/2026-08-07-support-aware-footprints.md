# Support-Aware Footprints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in mode that packs pre-supported models on a base-excluded
"model body" footprint (full shadow minus an auto-detected raft/support base),
recovering concavities for denser plates, while leaving default behavior
byte-identical.

**Architecture:** `footprint.py` gains footprint-area-knee base detection and a two-mask
extractor (full + body). The cache doc (schema v2) stores both; `loading` selects
which the packer consumes. When `support_aware` is on, the packer uses a two-mask
collision — body vs pieces (rafts overlap), full vs the plate boundary — with body
and full rotated onto one shared canvas (`rotate_pair`), so verify ORs the full
shadow at each anchor. All new behavior is gated by config; when off, existing code
paths run unchanged.

**Tech Stack:** Python 3.11+, trimesh, numpy, opencv-python-headless, typer,
pytest. Managed with `uv`; run tests via `uv run pytest`, lint via
`uv run ruff check` and `uv run ruff format`.

## Global Constraints

- **Off = unchanged.** `support_aware` defaults `False`; with it off, packing and
  verification use the existing `full_shadow` code path, byte-identical to today.
- **Conservative coverage above the cut.** `model_body` = shadow of triangles with
  **max Z `> z0 + cut`** (a straddling triangle is kept whole) — it can never
  invent free space above the cut.
- **Cap is the search window.** Detection sweeps cut depth only within
  `[z0, z0 + cap]`; footprint doesn't drop by `MIN_REDUCTION` → cut = 0. Never cut
  deeper than the cap.
- **Detector constants** live in `footprint.py`: `BAND_MM = 0.25`,
  `MIN_REDUCTION = 0.05`, `FLAT_EPS = 0.01`, `DETECT_RES_MM = 0.2`,
  `DETECTOR_VERSION = 1`. Only `support_aware` and `support_cut_cap_mm`
  (default `5.0`) are user config.
- **Canonical resolution** stays `CANONICAL_RES_MM = 0.05`; both masks share one
  extraction canvas/origin.
- **Cache schema** bumps to `SCHEMA_VERSION = 2`; reads accept `{1, 2}`. A v1 doc
  (or a v2 doc without a body mask) reads with `body_mask = None`.
- **Tests are parametrized and atomic** — one named case per input (project rule).
  Never a loop of asserts inside one test body.

---

### Task 1: Base-cut detection (`detect_base_cut`)

**Files:**
- Modify: `src/plate_packer/footprint.py`
- Test: `tests/test_footprint.py`

**Interfaces:**
- Produces:
  - `BAND_MM = 0.25`, `MIN_REDUCTION = 0.05`, `FLAT_EPS = 0.01`,
    `DETECT_RES_MM = 0.2`, `DETECTOR_VERSION = 1` (module constants).
  - `_raster(tri_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray` — fill
    each `(k, 3, 2)` int32 triangle into a `shape` uint8 {0,1} canvas.
  - `detect_base_cut(tris: np.ndarray, res_mm: float, cap_mm: float) -> float` —
    offset above the mesh base below which geometry is raft/support base, found by
    the footprint-area knee; `0.0` when the footprint doesn't drop by at least
    `MIN_REDUCTION` within `[z0, z0 + cap_mm]`. `tris` is an `(n, 3, 3)` finite
    triangle array.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_footprint.py`:

```python
from plate_packer.footprint import BAND_MM, detect_base_cut


def _box(xy, z_lo, z_hi):
    """Axis-aligned box with XY extents `xy`, spanning Z [z_lo, z_hi]."""
    b = trimesh.creation.box(extents=(xy[0], xy[1], z_hi - z_lo))
    b.apply_translation([0, 0, (z_lo + z_hi) / 2])
    return b


def _tris(*boxes):
    return trimesh.util.concatenate(list(boxes)).triangles


@pytest.mark.parametrize(
    ("tris", "expected"),
    [
        pytest.param(_tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12)), 2.0, id="raft-then-pillar"),
        pytest.param(_tris(_box((1, 1), 0, 12)), 0.0, id="pillar-only-no-base"),
        pytest.param(_tris(_box((20, 20), 0, 12)), 0.0, id="wide-solid-no-drop"),
        pytest.param(
            _tris(_box((20, 20), 0, 8), _box((1, 1), 8, 18)), 0.0, id="base-past-cap-window"
        ),
        pytest.param(
            _tris(_box((2, 2), 0, 0.5), _box((1, 1), 0.5, 8), _box((20, 20), 8, 10)),
            0.0,
            id="tiny-foot-below-min-base-frac",
        ),
    ],
)
def test_detect_base_cut(tris, expected):
    assert detect_base_cut(tris, 0.1, 5.0) == pytest.approx(expected, abs=BAND_MM)


def test_detect_base_cut_zero_cap_returns_zero():
    tris = _tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12))
    assert detect_base_cut(tris, 0.1, 0.0) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_footprint.py -k detect_base_cut -v`
Expected: FAIL — `ImportError: cannot import name 'BAND_MM'` / `detect_base_cut`.

- [ ] **Step 3: Implement**

Add to `src/plate_packer/footprint.py` (after the imports and before
`extract_footprint`):

```python
BAND_MM = 0.25  # Z step for the cut-depth sweep
MIN_REDUCTION = 0.05  # min footprint drop (fraction) worth excluding a base for
FLAT_EPS = 0.01  # plateau tolerance (fraction of footprint) for the knee
DETECT_RES_MM = 0.2  # coarse raster res for detection (area ratios are scale-tolerant)
DETECTOR_VERSION = 1  # bump when any detector constant changes (invalidates body masks)


def _raster(tri_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill each (k, 3, 2) int32 triangle into a `shape` uint8 {0,1} canvas.
    One fillConvexPoly per triangle: a batched fillPoly XORs overlaps (bugs.md)."""
    canvas = np.zeros(shape, np.uint8)
    for tri in tri_px:
        cv2.fillConvexPoly(canvas, tri, 1)
    return canvas


def detect_base_cut(tris: np.ndarray, res_mm: float, cap_mm: float) -> float:
    """Offset above the mesh base below which geometry is raft/support base.

    Detected by the footprint-area knee: as the cut depth d rises, the model-body
    footprint (shadow of triangles with max Z > z0 + d) shrinks sharply through
    the base then plateaus. We cut at the plateau's start. Computed in a single
    raster pass: paint each pixel with the highest max-Z of the triangles covering
    it (fill in ascending max-Z so the top wins), then area(d) is a threshold on
    that top-reach map. Returns 0.0 (no cut, the safe default) when the footprint
    never drops by MIN_REDUCTION inside [z0, z0 + cap_mm] -- so a raftless model, a
    wide solid box, or a base taller than the cap all leave model_body == full.
    """
    if cap_mm <= 0 or len(tris) == 0:
        return 0.0
    xy = tris[:, :, :2]
    z = tris[:, :, 2]
    z0, z1 = float(z.min()), float(z.max())
    if z1 - z0 <= BAND_MM:
        return 0.0
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / res_mm).astype(np.int32)
    tri_zmax = z.max(axis=1)
    # Per-pixel top-reach map: each pixel keeps the tallest (max_Z - z0) of the
    # triangles covering it (fill in ascending order so the tallest paints last).
    # float64 with NO additive offset: a large offset in float32 rounds away the
    # sub-ULP fraction at band boundaries, so reach would drop a pixel one band
    # before the float-compared body mask does (real-STL bug). area(d) = reach > d
    # then matches the body rule max_Z > z0 + d exactly. Empty and base-only pixels
    # stay 0 and are correctly excluded without an offset.
    reach = np.zeros(shape, np.float64)
    for i in np.argsort(tri_zmax):
        cv2.fillConvexPoly(reach, tri_px[i], float(tri_zmax[i] - z0))

    def area(d):  # pixels whose top reach is above cut depth d
        return int((reach > d).sum())

    a_full = area(0.0)
    if a_full == 0:
        return 0.0
    n_bands = int(cap_mm / BAND_MM)
    areas = [area(i * BAND_MM) for i in range(n_bands + 1)]
    a_min = min(areas)
    if a_full - a_min < MIN_REDUCTION * a_full:
        return 0.0  # base not worth excluding
    thresh = a_min + FLAT_EPS * a_full
    for i in range(n_bands + 1):
        if areas[i] <= thresh:  # first depth on the plateau = the knee
            return float(i * BAND_MM)
    return 0.0
```

Note `area(0.0)` uses `reach > 0` (strictly above `z0`), so an empty pixel or one
whose tallest covering triangle sits exactly at `z0` is excluded — consistent with
the `max Z > z0 + cut` body rule. The float64/no-offset choice is load-bearing:
with a float32 `+1.0` offset the reach map dropped raft-top pixels a band early
(their fraction rounded off near 1.5), cutting too shallow for 0% reduction on
real STLs while the synthetic slab tests still passed.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_footprint.py -k detect_base_cut -v`
Expected: PASS (all 6 cases).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/footprint.py tests/test_footprint.py
git commit -m "feat: footprint-area-knee base-cut detection for support-aware footprints"
```

---

### Task 2: Two-mask extractor (`extract_footprints`)

**Files:**
- Modify: `src/plate_packer/footprint.py`
- Test: `tests/test_footprint.py`

**Interfaces:**
- Consumes: `detect_base_cut`, `_raster`, `DETECTOR_VERSION` (Task 1).
- Produces:
  - `extract_footprints(mesh, res_mm: float, cut_cap_mm: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, dict]`
    returning `(full_mask, body_mask, origin_mm, cut_mm, stats)`. `full_mask` is
    identical to `extract_footprint`'s mask; `body_mask` shares its shape/origin;
    `stats` has the same keys as `extract_footprint`'s plus `"cut_mm"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_footprint.py`:

```python
from plate_packer.footprint import extract_footprints


def test_extract_footprints_full_matches_extract_footprint():
    mesh = trimesh.creation.box(extents=(20, 10, 5))
    ref, _o, _s = extract_footprint(mesh, RES)
    full, _body, _origin, _cut, _stats = extract_footprints(mesh, RES, 5.0)
    assert full.shape == ref.shape
    assert (full == ref).all()


def test_extract_footprints_body_subset_of_full_and_smaller():
    mesh = trimesh.util.concatenate([_box((20, 20), 0, 2), _box((1, 1), 2, 12)])
    full, body, _origin, cut, stats = extract_footprints(mesh, RES, 5.0)
    assert full.shape == body.shape
    assert (body & ~full).sum() == 0  # body is a subset of full
    assert 0 < body.sum() < full.sum()  # raft slab removed
    assert cut == pytest.approx(2.0, abs=BAND_MM)
    assert stats["cut_mm"] == cut


def test_extract_footprints_no_cut_gives_identical_body():
    mesh = _box((1, 1), 0, 12)  # pillar only -> cut 0
    full, body, _origin, cut, _stats = extract_footprints(mesh, RES, 5.0)
    assert cut == 0.0
    assert (body == full).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_footprint.py -k extract_footprints -v`
Expected: FAIL — `cannot import name 'extract_footprints'`.

- [ ] **Step 3: Implement**

Add to `src/plate_packer/footprint.py` (after `detect_base_cut`):

```python
def extract_footprints(mesh, res_mm: float, cut_cap_mm: float):
    """Return (full_mask, body_mask, origin_mm, cut_mm, stats).

    full_mask is the full vertical shadow (identical to extract_footprint).
    body_mask, on the same canvas/origin, is the shadow of triangles reaching
    above the auto-detected base cut (max Z > z0 + cut_mm); when cut_mm == 0 it
    is a copy of full_mask. stats mirrors extract_footprint's, plus 'cut_mm'.
    """
    t0 = time.perf_counter()
    tris = mesh.triangles
    finite = np.isfinite(tris).all(axis=(1, 2))
    n_bad = int((~finite).sum())
    if n_bad:
        tris = tris[finite]
    if len(tris) == 0:
        raise ValueError("mesh has no finite triangles")
    xy = tris[:, :, :2]
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / res_mm).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / res_mm).astype(np.int32)
    full_mask = _raster(tri_px, shape)

    # Detection runs on a coarse grid (area ratios are scale-tolerant); the body
    # mask below is still rasterized at the caller's res_mm.
    cut_mm = detect_base_cut(tris, DETECT_RES_MM, cut_cap_mm)
    if cut_mm <= 0:
        body_mask = full_mask.copy()
    else:
        z0 = float(tris[:, :, 2].min())
        keep = tris[:, :, 2].max(axis=1) > z0 + cut_mm
        body_mask = _raster(tri_px[keep], shape)
    t_raster = time.perf_counter() - t0

    stats = {
        "triangles": len(mesh.triangles),
        "dropped_nonfinite": n_bad,
        "z_height_mm": float(tris[:, :, 2].max() - tris[:, :, 2].min()),
        "footprint_mm": tuple((flat.max(axis=0) - flat.min(axis=0)).round(1)),
        "mask_px": full_mask.shape,
        "coverage": float(full_mask.mean()),
        "raster_s": t_raster,
        "cut_mm": cut_mm,
    }
    return full_mask, body_mask, (float(origin[0]), float(origin[1])), cut_mm, stats
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_footprint.py -v`
Expected: PASS (new cases plus all existing footprint tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/footprint.py tests/test_footprint.py
git commit -m "feat: extract_footprints returns full + body masks with base cut"
```

---

### Task 3: Gated integration test on a real STL (lands early)

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_support_integration.py`

**Interfaces:**
- Consumes: `extract_footprints` (Task 2), `load_piece_mesh`
  (`plate_packer.export`), `CANONICAL_RES_MM` (`plate_packer.footprint_io`).

**Why now:** validate the detector against *real* pre-supported STLs and see the
true footprint reduction before building the rest on top. Only the
`*_supported.stl` files carry rafts (the raw kit parts float at assembly Z with no
base and correctly yield cut 0), so the test targets those. Measured benefit on
this corpus: wings/tails/bodies drop **−14% to −32%**. The test is deselected by
default and skipped when the `example_stls` junction is absent, so CI and a plain
`pytest` run both skip it.

- [ ] **Step 1: Register the marker and deselect it by default**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not example_stls'"
markers = [
    "example_stls: integration tests needing the gitignored example_stls junction (opt in with -m example_stls)",
]
```

- [ ] **Step 2: Write the test (it is the failing artifact)**

Create `tests/test_support_integration.py`:

```python
"""Gated integration: base-cut detection on real pre-supported STLs.

Deselected by default (see pyproject addopts). Run explicitly with real assets:
    uv run pytest -m example_stls -s
"""

from pathlib import Path

import pytest

from plate_packer.export import load_piece_mesh
from plate_packer.footprint import extract_footprints
from plate_packer.footprint_io import CANONICAL_RES_MM

pytestmark = pytest.mark.example_stls

EXAMPLES = Path("example_stls")


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_stls junction absent")
def test_detection_fires_on_real_supports(capsys):
    # Only the pre-supported exports carry rafts; match case-insensitively
    # ("_Supported" / "_supported" both occur in the corpus).
    stls = [p for p in sorted(EXAMPLES.rglob("*.stl")) if "_supported" in p.stem.lower()][:8]
    if not stls:
        pytest.skip("no *_supported.stl files under example_stls")
    results = []
    for stl in stls:
        full, body, _origin, cut, _stats = extract_footprints(
            load_piece_mesh(stl), CANONICAL_RES_MM, 5.0
        )
        reduction = 1 - body.sum() / full.sum() if full.sum() else 0.0
        results.append((stl.name, cut, reduction))
    with capsys.disabled():
        print("\nbase-cut detection on real supported STLs:")
        for name, cut, reduction in results:
            print(f"  {name}: cut={cut:.2f}mm  footprint area -{reduction:.1%}")
    assert any(cut > 0 for _name, cut, _r in results), "detector never fired on real supports"
    assert any(r > 0.05 for _name, _c, r in results), "no meaningful footprint reduction"
```

- [ ] **Step 3: Confirm it is skipped in the default run**

Run: `uv run pytest tests/test_support_integration.py -v`
Expected: 1 deselected (no failures) — the `-m 'not example_stls'` addopts filters it.

- [ ] **Step 4: Confirm it runs when opted in (if assets present)**

Run: `uv run pytest tests/test_support_integration.py -m example_stls -s`
Expected: PASS with a printed cut/area-reduction line per supported STL (real cuts
and reductions ≳ 14%), or SKIPPED if the junction is absent. Note in the report
which happened and paste the printed reductions.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_support_integration.py
git commit -m "test: gated integration for base-cut detection on real STLs"
```

---

### Task 4: Cache schema v2 — store and load both masks

**Files:**
- Modify: `src/plate_packer/footprint_io.py`
- Test: `tests/test_footprint_io.py`

**Interfaces:**
- Consumes: `DETECTOR_VERSION` is passed in by callers (not imported here, to
  avoid a footprint_io -> footprint dependency); this task treats it as an opaque
  int argument.
- Produces:
  - `SCHEMA_VERSION = 2`.
  - `save_doc(footprints_dir, sha, mask, origin_mm, stats, res_mm_per_px=CANONICAL_RES_MM, body_mask=None, cut_z_mm=None, detector_version=None) -> Path`
    — writes a `model_body` footprint entry and doc-level `cut_z_mm` /
    `detector_version` when `body_mask` is given.
  - `FootprintDoc` gains `body_mask: np.ndarray | None = None`,
    `cut_z_mm: float | None = None`, `detector_version: int | None = None`;
    `masks` stays `[full_shadow]`.
  - `load_doc` / `has_current_doc` accept `schema_version` in `{1, 2}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_footprint_io.py`:

```python
def body_mask():
    m = np.zeros((8, 6), np.uint8)
    m[2:6, 1:5] = 1
    return m


def test_round_trip_preserves_body_mask_and_cut(tmp_path):
    full, body = checker_mask(), body_mask()
    save_doc(
        tmp_path,
        SHA_A,
        full,
        (-10.0, -5.0),
        STATS,
        body_mask=body,
        cut_z_mm=2.5,
        detector_version=1,
    )
    doc = load_doc(tmp_path, SHA_A)
    assert (doc.masks[0] == full).all()
    assert doc.body_mask is not None
    assert (doc.body_mask == body).all()
    assert doc.cut_z_mm == 2.5
    assert doc.detector_version == 1


def test_body_mask_written_as_model_body_entry(tmp_path):
    save_doc(
        tmp_path,
        SHA_A,
        checker_mask(),
        (0.0, 0.0),
        STATS,
        body_mask=body_mask(),
        cut_z_mm=2.5,
        detector_version=1,
    )
    raw = json.loads(doc_path(tmp_path, SHA_A).read_text(encoding="utf-8"))
    kinds = [fp["kind"] for fp in raw["footprints"]]
    assert kinds == ["full_shadow", "model_body"]
    assert raw["footprints"][1]["z_band_mm"] == [2.5, None]


def test_doc_without_body_has_none_body_fields(tmp_path):
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)  # no body
    doc = load_doc(tmp_path, SHA_A)
    assert doc.body_mask is None
    assert doc.cut_z_mm is None
    assert doc.detector_version is None


def test_v1_doc_reads_with_no_body(tmp_path):
    """A schema-1 doc (e.g. from stl_curator) loads; body is absent."""
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    p = doc_path(tmp_path, SHA_A)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    raw["footprints"] = [raw["footprints"][0]]  # full_shadow only
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert has_current_doc(tmp_path, SHA_A) is True
    doc = load_doc(tmp_path, SHA_A)
    assert doc.body_mask is None
    assert len(doc.masks) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_footprint_io.py -k "body or v1_doc" -v`
Expected: FAIL — `save_doc()` got an unexpected keyword `body_mask`.

- [ ] **Step 3: Implement**

In `src/plate_packer/footprint_io.py`:

Change the schema constant and add a supported set:

```python
SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA = {1, 2}
```

Add body fields to the dataclass:

```python
@dataclass(frozen=True)
class FootprintDoc:
    sha: str
    res_mm_per_px: float
    origin_mm: tuple[float, float]
    z_height_mm: float
    triangles: int
    dropped_nonfinite: int
    masks: list  # list[np.ndarray]; masks[0] is always full_shadow
    body_mask: object = None  # np.ndarray | None (model_body), None if absent
    cut_z_mm: float | None = None
    detector_version: int | None = None
```

Replace `save_doc` with:

```python
def _png_b64(mask) -> str:
    ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def save_doc(
    footprints_dir,
    sha,
    mask,
    origin_mm,
    stats,
    res_mm_per_px=CANONICAL_RES_MM,
    body_mask=None,
    cut_z_mm=None,
    detector_version=None,
) -> Path:
    footprints = [{"kind": "full_shadow", "z_band_mm": [0.0, None], "mask_png_b64": _png_b64(mask)}]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "stl_sha256": sha,
        "res_mm_per_px": res_mm_per_px,
        "origin_mm": [float(origin_mm[0]), float(origin_mm[1])],
        "z_height_mm": stats["z_height_mm"],
        "triangles": stats["triangles"],
        "dropped_nonfinite": stats["dropped_nonfinite"],
        "footprints": footprints,
    }
    if body_mask is not None:
        footprints.append(
            {
                "kind": "model_body",
                "z_band_mm": [float(cut_z_mm) if cut_z_mm is not None else 0.0, None],
                "mask_png_b64": _png_b64(body_mask),
            }
        )
        doc["cut_z_mm"] = float(cut_z_mm) if cut_z_mm is not None else 0.0
        doc["detector_version"] = detector_version
    path = doc_path(footprints_dir, sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)  # readers never see a torn doc
    return path
```

Replace `load_doc` with:

```python
def _decode_mask(fp, sha) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(fp["mask_png_b64"]), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"corrupt mask PNG in doc {sha}")
    return (img > 0).astype(np.uint8)


def load_doc(footprints_dir, sha) -> FootprintDoc:
    raw = json.loads(doc_path(footprints_dir, sha).read_text(encoding="utf-8"))
    if raw.get("schema_version") not in _SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    full_masks, body_mask, cut_z_mm = [], None, None
    for fp in raw["footprints"]:
        if fp.get("kind") == "model_body":
            body_mask = _decode_mask(fp, raw["stl_sha256"])
            cut_z_mm = fp["z_band_mm"][0]
        else:
            full_masks.append(_decode_mask(fp, raw["stl_sha256"]))
    return FootprintDoc(
        sha=raw["stl_sha256"],
        res_mm_per_px=raw["res_mm_per_px"],
        origin_mm=tuple(raw["origin_mm"]),
        z_height_mm=raw["z_height_mm"],
        triangles=raw["triangles"],
        dropped_nonfinite=raw["dropped_nonfinite"],
        masks=full_masks,
        body_mask=body_mask,
        cut_z_mm=cut_z_mm,
        detector_version=raw.get("detector_version"),
    )
```

In `has_current_doc`, change the final line from
`return raw.get("schema_version") == SCHEMA_VERSION` to:

```python
    return raw.get("schema_version") in _SUPPORTED_SCHEMA
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_footprint_io.py -v`
Expected: PASS (new cases plus all existing footprint_io tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/footprint_io.py tests/test_footprint_io.py
git commit -m "feat: cache schema v2 stores model_body mask + cut metadata"
```

---

### Task 5: Config knobs (`support_aware`, `support_cut_cap_mm`)

**Files:**
- Modify: `src/plate_packer/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `PackConfig.support_aware: bool = False`,
  `PackConfig.support_cut_cap_mm: float = 5.0`; `_validate` rejects
  `support_cut_cap_mm <= 0`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_support_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.support_aware is False
    assert cfg.support_cut_cap_mm == 5.0


def test_support_knobs_load_from_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[packing]\nsupport_aware = true\nsupport_cut_cap_mm = 3.0\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.support_aware is True
    assert cfg.support_cut_cap_mm == 3.0


@pytest.mark.parametrize("value", [pytest.param("0", id="zero"), pytest.param("-1", id="negative")])
def test_support_cut_cap_must_be_positive(tmp_path, value):
    p = tmp_path / "config.toml"
    p.write_text(f"[packing]\nsupport_cut_cap_mm = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="support_cut_cap_mm"):
        load_config(p)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_config.py -k support -v`
Expected: FAIL — `PackConfig` has no attribute `support_aware`.

- [ ] **Step 3: Implement**

In `src/plate_packer/config.py`, add to `PackConfig` (after `ordering`):

```python
    support_aware: bool = False  # opt-in: pack on base-excluded model_body footprint
    support_cut_cap_mm: float = 5.0  # search-window / max base-cut height (mm)
```

In `load_config`, add to the `PackConfig(...)` construction:

```python
support_aware = (bool(packing.get("support_aware", PackConfig.support_aware)),)
support_cut_cap_mm = (float(packing.get("support_cut_cap_mm", PackConfig.support_cut_cap_mm)),)
```

In `_validate`, add:

```python
    if cfg.support_cut_cap_mm <= 0:
        raise ValueError("packing.support_cut_cap_mm must be > 0")
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (new cases plus all existing config tests).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/config.py tests/test_config.py
git commit -m "feat: support_aware + support_cut_cap_mm config knobs"
```

---

### Task 6: Mask selection in `prepare_mask`

**Files:**
- Modify: `src/plate_packer/loading.py`
- Test: `tests/test_loading.py`

**Interfaces:**
- Consumes: `FootprintDoc.body_mask` (Task 4).
- Produces: `prepare_mask(doc, spacing_mm, working_res_mm, kind="full_shadow") -> tuple[np.ndarray, tuple[float, float]]`
  — `kind="model_body"` prepares `doc.body_mask`; raises `ValueError` if it is
  absent. `kind` defaults to `"full_shadow"`, so every existing caller is
  unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_loading.py` (create the file if it does not exist, mirroring
the imports of the other tests — it needs `numpy as np`, `pytest`, and
`from plate_packer.footprint_io import FootprintDoc`, plus
`from plate_packer.loading import prepare_mask`):

```python
def _doc(full, body=None):
    return FootprintDoc(
        sha="a" * 64,
        res_mm_per_px=0.05,
        origin_mm=(-1.0, -2.0),
        z_height_mm=5.0,
        triangles=1,
        dropped_nonfinite=0,
        masks=[full],
        body_mask=body,
        cut_z_mm=2.0 if body is not None else None,
        detector_version=1 if body is not None else None,
    )


def test_prepare_mask_defaults_to_full_shadow():
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[3:7, 3:7] = 1
    mask, _origin = prepare_mask(_doc(full, body), 0.0, 0.1)  # kind defaults full
    assert mask.sum() == full[::2, ::2].sum()  # downsample of full, not body


def test_prepare_mask_selects_body():
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[2:8, 2:8] = 1
    full_mask, origin_full = prepare_mask(_doc(full, body), 0.0, 0.1, kind="full_shadow")
    body_mask, origin_body = prepare_mask(_doc(full, body), 0.0, 0.1, kind="model_body")
    assert body_mask.sum() < full_mask.sum()
    assert origin_full == origin_body  # same origin regardless of kind


def test_prepare_mask_body_absent_raises():
    with pytest.raises(ValueError, match="model_body"):
        prepare_mask(_doc(np.ones((10, 10), np.uint8), body=None), 0.0, 0.1, kind="model_body")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_loading.py -k prepare_mask -v`
Expected: FAIL — `prepare_mask()` got an unexpected keyword `kind`.

- [ ] **Step 3: Implement**

In `src/plate_packer/loading.py`, change `prepare_mask`'s signature and the line
that reads the source mask:

```python
def prepare_mask(
    doc: FootprintDoc, spacing_mm: float, working_res_mm: float, kind: str = "full_shadow"
) -> tuple[np.ndarray, tuple[float, float]]:
```

Replace the body-selecting line (currently `mask = conservative_downsample(doc.masks[0], round(ratio))`)
with:

```python
    if kind == "model_body":
        if doc.body_mask is None:
            raise ValueError("doc has no model_body mask")
        source = doc.body_mask
    else:
        source = doc.masks[0]
    mask = conservative_downsample(source, round(ratio))
```

Update the docstring's first line to note the `kind` selector.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_loading.py -v`
Expected: PASS. Also run `uv run pytest tests/ -q` to confirm no existing caller
of `prepare_mask` broke.

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/loading.py tests/test_loading.py
git commit -m "feat: prepare_mask selects full_shadow or model_body by kind"
```

---

### Task 7: Full-shadow occupancy placement helper

**Files:**
- Modify: `src/plate_packer/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `rotate_mask` (`plate_packer.packer`) in tests only.
- Produces:
  - `occupancy_from_full(occ: np.ndarray, full_rot: np.ndarray, body_aff: np.ndarray, full_aff: np.ndarray, row: int, col: int) -> None`
    — OR a rotated full-shadow mask into `occ` at the world position of a body
    placed at `(row, col)`. Both masks share the un-cropped rotation canvas, so
    the full anchor is `(row + Δr, col + Δc)` with
    `Δr = body_aff[1,2] - full_aff[1,2]`, `Δc = body_aff[0,2] - full_aff[0,2]`.
    Clipped to `occ` bounds; mutates `occ` in place.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export.py`:

```python
from plate_packer.export import occupancy_from_full
from plate_packer.packer import rotate_mask


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 137.0], ids=lambda a: f"deg{a:g}")
def test_occupancy_from_full_covers_body_placement(angle):
    """A full mask (>= body) placed via occupancy_from_full must cover the body
    placed the normal way at the same anchor, for any rotation."""
    full = np.zeros((30, 30), np.uint8)
    full[5:25, 5:25] = 1  # full footprint
    body = np.zeros((30, 30), np.uint8)
    body[10:25, 10:25] = 1  # body = subset (base cleared), SAME canvas

    body_rot, body_aff = rotate_mask(body, angle)
    full_rot, full_aff = rotate_mask(full, angle)

    row, col = 40, 50
    occ_body = np.zeros((120, 120), np.uint8)
    occ_body[row : row + body_rot.shape[0], col : col + body_rot.shape[1]] |= body_rot

    occ_full = np.zeros((120, 120), np.uint8)
    occupancy_from_full(occ_full, full_rot, body_aff, full_aff, row, col)

    # full-shadow occupancy is a superset of the body placement (alignment holds)
    assert (occ_body.astype(bool) & ~occ_full.astype(bool)).sum() == 0
    assert occ_full.sum() >= occ_body.sum()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_export.py -k occupancy_from_full -v`
Expected: FAIL — `cannot import name 'occupancy_from_full'`.

- [ ] **Step 3: Implement**

Add to `src/plate_packer/export.py`:

```python
def occupancy_from_full(
    occ: np.ndarray,
    full_rot: np.ndarray,
    body_aff: np.ndarray,
    full_aff: np.ndarray,
    row: int,
    col: int,
) -> None:
    """OR a rotated full-shadow mask into `occ` at the world position of a body
    placed at (row, col). The body and full prepared masks share one un-cropped
    rotation canvas (identical extraction canvas -> downsample -> dilation), so
    the full anchor is the body anchor plus the affines' crop-translation
    difference. Clipped to occ bounds; mutates occ in place."""
    fr = int(round(row + (body_aff[1, 2] - full_aff[1, 2])))
    fc = int(round(col + (body_aff[0, 2] - full_aff[0, 2])))
    h, w = full_rot.shape
    r0, c0 = max(fr, 0), max(fc, 0)
    r1, c1 = min(fr + h, occ.shape[0]), min(fc + w, occ.shape[1])
    if r1 <= r0 or c1 <= c0:
        return
    occ[r0:r1, c0:c1] |= full_rot[r0 - fr : r1 - fr, c0 - fc : c1 - fc]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_export.py -k occupancy_from_full -v`
Expected: PASS (all 4 angle cases).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/export.py tests/test_export.py
git commit -m "feat: occupancy_from_full places full shadow at body world anchor"
```

---

> **Task 7 note (superseded):** `occupancy_from_full` (crop-offset placement) is
> replaced by the shared-canvas approach below — `rotate_pair` makes body/full
> share a canvas, so verify is a plain OR and no crop-offset math remains. Task 11
> removes `occupancy_from_full` and its test.

### Task 8: Shared-canvas rotation (`rotate_pair`)

**Files:**
- Modify: `src/plate_packer/packer.py`
- Test: `tests/test_packer.py`

**Interfaces:**
- Produces:
  - `_paste(dst, src, r, c) -> None` — OR `src` into `dst` at `(r, c)`, clipped to `dst` bounds.
  - `rotate_pair(full, body, angle_deg) -> (full_rot, body_rot, affine)` — rotate
    `full` and `body` (with `body ⊆ full`, same input canvas) onto one shared
    canvas cropped to the full mask's content bbox. `full_rot` equals
    `rotate_mask(full, angle_deg)[0]`; `body_rot` has the same shape with the
    body content at its position in the full frame; `affine` is full's 2×3 map.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_packer.py`:

```python
from plate_packer.packer import rotate_pair


def test_rotate_pair_degenerate_full_equals_body():
    full = np.ones((10, 12), np.uint8)
    fr, br, _aff = rotate_pair(full, full, 37.0)
    assert br.shape == fr.shape
    assert (br == fr).all()


def test_rotate_pair_angle0_reconstructs_body():
    full = np.ones((12, 12), np.uint8)
    body = np.zeros((12, 12), np.uint8)
    body[2:10, 4:8] = 1  # narrower body, smaller bbox than full
    fr, br, _aff = rotate_pair(full, body, 0.0)
    assert (fr == full).all()
    assert (br == body).all()  # body placed back at its full-frame position


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 150.0], ids=lambda a: f"deg{a:g}")
def test_rotate_pair_body_subset_same_shape(angle):
    full = np.ones((12, 12), np.uint8)
    body = full.copy()
    body[3:9, 3:9] = 0  # interior hole (same bbox as full)
    fr, br, _aff = rotate_pair(full, body, angle)
    assert br.shape == fr.shape
    assert (br & ~fr).sum() == 0  # body_rot is a subset of full_rot
    assert br.sum() < fr.sum()  # the hole survives
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_packer.py -k rotate_pair -v`
Expected: FAIL — `cannot import name 'rotate_pair'`.

- [ ] **Step 3: Implement**

Add to `src/plate_packer/packer.py` (after `rotate_mask`):

```python
def _paste(dst: np.ndarray, src: np.ndarray, r: int, c: int) -> None:
    """OR `src` into `dst` at top-left (r, c), clipped to `dst` bounds."""
    h, w = src.shape
    r0, c0 = max(r, 0), max(c, 0)
    r1, c1 = min(r + h, dst.shape[0]), min(c + w, dst.shape[1])
    if r1 <= r0 or c1 <= c0:
        return
    dst[r0:r1, c0:c1] |= src[r0 - r : r1 - r, c0 - c : c1 - c]


def rotate_pair(full: np.ndarray, body: np.ndarray, angle_deg: float):
    """Rotate `full` and `body` (body ⊆ full, same input canvas) onto ONE shared
    canvas cropped to the full mask's content bbox.

    Returns (full_rot, body_rot, affine): full_rot == rotate_mask(full, ·)[0];
    body_rot has the same shape, with the body content placed at its position in
    the full frame; affine is full's 2×3 map (export/verify use it). Because both
    outputs share shape and anchor, downstream legality/verify need no crop-offset
    arithmetic. The body's crop origin sits (aff_full - aff_body) below/right of
    full's, so the body is pasted at that offset."""
    full_rot, aff_full = rotate_mask(full, angle_deg)
    body_own, aff_body = rotate_mask(body, angle_deg)
    dr = int(round(aff_full[1, 2] - aff_body[1, 2]))
    dc = int(round(aff_full[0, 2] - aff_body[0, 2]))
    body_rot = np.zeros_like(full_rot)
    _paste(body_rot, body_own, dr, dc)
    return full_rot, body_rot, aff_full
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_packer.py -k rotate_pair -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/packer.py tests/test_packer.py
git commit -m "feat: rotate_pair rotates body and full onto a shared canvas"
```

---

### Task 9: Two-mask packing (`boundary`)

**Files:**
- Modify: `src/plate_packer/packer.py`
- Test: `tests/test_packer.py`

**Interfaces:**
- Consumes: `legal_placement_map`, `contact_map`, `contact_ring`, `_fits`, `Placement`, `contact_first`, `rotate_mask` (existing).
- Produces:
  - `pack(..., boundary=None)` — new keyword. `boundary` is a per-piece list of
    `{angle: full_rot}` **parallel to `prerotated`** (`{angle: body_rot}`), the two
    masks sharing a canvas (same shape per angle). When `boundary` is None the
    existing single-mask path runs unchanged. When given: inter-piece collision
    uses the body, plate-boundary/dead-margin uses the full, empty-plate fit uses
    the full.
  - `_best_spot_bounded(pieces_occ, border, variants, fullvars, rings, choose, edge_weight)`
    and `_pack_bounded(...)` internal helpers.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_packer.py`:

```python
from plate_packer.packer import pack, rotate_pair, contact_ring


def _paired_variants(full, body, angles):
    """Build (body_variants, full_variants) dicts sharing a canvas per angle."""
    bvar, fvar = {}, {}
    for a in angles:
        fr, br, _ = rotate_pair(full, body, a)
        fvar[a], bvar[a] = fr, br
    return bvar, fvar


def test_boundary_rafts_may_overlap_same_plate():
    # full = 20x20 raft; body = central 6-wide column (bodies stay disjoint,
    # but the wide rafts overlap). Two pieces should share ONE plate.
    full = np.ones((20, 20), np.uint8)
    body = np.zeros((20, 20), np.uint8)
    body[:, 7:13] = 1
    b, f = _paired_variants(full, body, [0.0])
    placements = pack([body], (20, 40), prerotated=[b], boundary=[f], order=[0], validate=False)
    # pack a second identical piece by passing two
    placements = pack(
        [body, body],
        (20, 40),
        prerotated=[b, b],
        boundary=[f, f],
        order=[0, 1],
        validate=False,
    )
    assert max(p.plate for p in placements) == 0  # both on plate 0


def test_boundary_full_kept_on_plate():
    # A piece whose body would fit flush at the right edge but whose full shadow
    # (same shared shape, wider content) must stay within the plate: with a
    # bordered plate the full may not overlap the border.
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[:, :4] = 1  # body content only on the left of the shared canvas
    b, f = _paired_variants(full, body, [0.0])
    border = np.zeros((10, 30), np.uint8)
    border[:, :2] = border[:, -2:] = 1  # 2px dead margins left/right
    placements = pack(
        [body],
        (10, 30),
        plate_mask=border,
        prerotated=[b],
        boundary=[f],
        order=[0],
        validate=False,
    )
    (pl,) = placements
    # full (all 10 cols occupied) must sit clear of both 2px borders
    assert pl.col >= 2 and pl.col + 10 <= 28


def test_boundary_empty_plate_fit_uses_full():
    # body fits a tiny plate but the full shadow does not -> rejected.
    full = np.ones((10, 10), np.uint8)
    body = np.zeros((10, 10), np.uint8)
    body[:4, :4] = 1
    b, f = _paired_variants(full, body, [0.0])
    with pytest.raises(ValueError, match="does not fit"):
        pack([body], (6, 6), prerotated=[b], boundary=[f], order=[0], validate=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_packer.py -k boundary -v`
Expected: FAIL — `pack()` got an unexpected keyword `boundary`.

- [ ] **Step 3: Implement**

In `src/plate_packer/packer.py`, add the `boundary=None` parameter to `pack`'s
signature (last param) and, immediately after `rings = (...)` is built and before
the `if validate:` block, insert an early delegation:

```python
    if boundary is not None:
        return _pack_bounded(
            pieces, empty, prerotated, boundary, rings, choose, order, validate, edge_weight
        )
```

(`empty` is the border base already computed at the top of `pack`.) Leave the
entire existing single-mask body below unchanged.

Then add the two helpers (near `_best_spot`):

```python
def _best_spot_bounded(pieces_occ, border, variants, fullvars, rings, choose, edge_weight=1.0):
    """Best (anchor, angle, contact) under two-mask legality: the body must not
    overlap placed bodies (pieces_occ), and the full shadow must clear the plate
    border/margins. body_rot and full_rot share a canvas (same shape), so the two
    legality maps AND directly."""
    best = None
    for angle, body in variants.items():
        full = fullvars[angle]
        if body.shape[0] > pieces_occ.shape[0] or body.shape[1] > pieces_occ.shape[1]:
            continue
        legal = legal_placement_map(pieces_occ, body) & legal_placement_map(border, full)
        contact = (
            contact_map(pieces_occ | border, rings[angle], edge_weight)
            if rings is not None
            else np.zeros(legal.shape)
        )
        anchor = choose(legal, contact)
        if anchor is None:
            continue
        score = float(contact[anchor])
        key = (-score, anchor[0], anchor[1])
        if best is None or key < best[0]:
            best = (key, anchor, angle, score)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _pack_bounded(
    pieces, border, prerotated, boundary, rings, choose, order, validate, edge_weight
):
    """Two-mask greedy pack: bodies collide with bodies (rafts overlap freely),
    full shadows stay on-plate. Plates track pieces-only occupancy."""
    plate_shape = border.shape
    if validate:
        for i, fullvars in enumerate(boundary):
            if not any(_fits(border, m) for m in fullvars.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")
    if order is None:
        order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].sum()), reverse=True)
    plates: list[np.ndarray] = []
    placements: list[Placement] = []
    for i in order:
        piece_rings = rings[i] if rings is not None else None
        target = plate_idx = None
        for idx, pocc in enumerate(plates):
            target = _best_spot_bounded(
                pocc, border, prerotated[i], boundary[i], piece_rings, choose, edge_weight
            )
            if target:
                plate_idx = idx
                break
        if target is None:
            plates.append(np.zeros(plate_shape, np.uint8))
            plate_idx = len(plates) - 1
            target = _best_spot_bounded(
                plates[plate_idx],
                border,
                prerotated[i],
                boundary[i],
                piece_rings,
                choose,
                edge_weight,
            )
        if target is None:
            raise ValueError(f"piece {i} does not fit an empty plate at any rotation")
        (row, col), angle, score = target
        body = prerotated[i][angle]
        plates[plate_idx][row : row + body.shape[0], col : col + body.shape[1]] |= body
        placements.append(Placement(i, plate_idx, row, col, angle, score))
    return sorted(placements, key=lambda p: p.piece)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_packer.py -v`
Expected: PASS (new boundary cases plus all existing packer tests — the
single-mask path is untouched).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/packer.py tests/test_packer.py
git commit -m "feat: two-mask packing (body vs pieces, full vs plate boundary)"
```

---

### Task 10: Thread `boundary` through `improve`

**Files:**
- Modify: `src/plate_packer/improve.py`
- Test: `tests/test_improve.py`

**Interfaces:**
- Consumes: `rotate_pair`, `pack(..., boundary=)` (Tasks 8-9); `conservative_downsample`.
- Produces: `improve(..., boundary_pieces=None)` — when given (a per-piece list of
  full-shadow prepared masks parallel to `pieces`, the body masks), the coarse
  and fine packs run two-mask. Body and full variants are built with `rotate_pair`
  (shared canvas) at fine and block-max-downsampled coarse resolution, so the
  coarse phase ANDs the same-shape masks. When None, behavior is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_improve.py`:

```python
from plate_packer.improve import improve


def test_improve_boundary_keeps_full_on_plate():
    # body: narrow central column; full: full-width raft (shared 12x12 canvas
    # after prep). With boundary on, the returned layout's full shadows must all
    # fit the plate (a smoke check that boundary is honored end to end).
    full = np.ones((12, 12), np.uint8)
    body = np.zeros((12, 12), np.uint8)
    body[:, 4:8] = 1
    res = improve(
        [body, body],
        (12, 48),
        boundary_pieces=[full, full],
        budget_s=0.0,
        angle_cap=1,
        min_edge_frac=0.5,
        safety_grid=0,
        validate=True,
    )
    from plate_packer.packer import rotate_mask

    for pl in res.placements:
        fr, _ = rotate_mask(full, pl.angle)
        assert pl.row + fr.shape[0] <= 12
        assert pl.col + fr.shape[1] <= 48
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_improve.py -k boundary -v`
Expected: FAIL — `improve()` got an unexpected keyword `boundary_pieces`.

- [ ] **Step 3: Implement**

In `src/plate_packer/improve.py`:

Add a paired prerotation helper:

```python
def _prerotate_paired(pieces, fulls, piece_angles, factor):
    """Fine + coarse {angle: mask} for body and full, on a shared canvas per
    angle (rotate_pair). Coarse = block-max downsample of each (supersets =>
    coarse-legal implies fine-legal)."""
    from plate_packer.packer import rotate_pair

    fine_b, coarse_b, fine_f, coarse_f = [], [], [], []
    for body, full, angles in zip(pieces, fulls, piece_angles, strict=True):
        bvar, fvar = {}, {}
        for a in angles:
            fr, br, _ = rotate_pair(full, body, a)
            fvar[a], bvar[a] = fr, br
        fine_b.append(bvar)
        fine_f.append(fvar)
        coarse_b.append({a: conservative_downsample(m, factor) for a, m in bvar.items()})
        coarse_f.append({a: conservative_downsample(m, factor) for a, m in fvar.items()})
    return fine_b, coarse_b, fine_f, coarse_f
```

Add `boundary_pieces=None` to `improve`'s signature (after `ordering`). After
`piece_angles` is computed, branch the prerotation and set fine/coarse boundary:

```python
    if boundary_pieces is None:
        fine_prerot, coarse_prerot = _prerotate_multi_res(pieces, piece_angles, factor)
        fine_bound = coarse_bound = None
    else:
        fine_prerot, coarse_prerot, fine_bound, coarse_bound = _prerotate_paired(
            pieces, boundary_pieces, piece_angles, factor
        )
```

The `coarse_seats_all` fallback must also drop the coarse boundary to fine when
it fires (keep the two consistent):

```python
    if not coarse_seats_all:
        coarse_prerot = fine_prerot
        coarse_bound = fine_bound
        coarse_plate_mask = empty_fine
        coarse_shape = plate_shape
        coarse_piece_px = fine_piece_px
        coarse_usable = fine_usable
        realize_scale = 1
```

The up-front validation and the `coarse_seats_all` check must use the FULL masks
when boundary is on (full is the binding fit constraint). Replace the two `_fits`
uses so they test `fine_bound`/`coarse_bound` when present:

```python
    fit_fine = fine_bound if fine_bound is not None else fine_prerot
    fit_coarse = coarse_bound if coarse_bound is not None else coarse_prerot
    coarse_seats_all = all(
        any(_fits(coarse_plate_mask, m) for m in variants.values()) for variants in fit_coarse
    )
    ...
    if validate:
        for i, variants in enumerate(fit_fine):
            if not any(_fits(empty_fine, m) for m in variants.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")
```

(Note: `fit_coarse` must be recomputed after the `coarse_seats_all` fallback
reassigns `coarse_bound`/`coarse_prerot`; compute `coarse_seats_all` from the
pre-fallback `fit_coarse`, then let the fallback reassign as above.)

Pass `boundary=` to both pack calls:

```python
def eval_coarse(order):
    result = pack(
        pieces,
        coarse_shape,
        plate_mask=coarse_plate_mask,
        choose=choose,
        prerotated=coarse_prerot,
        boundary=coarse_bound,
        order=order,
        validate=False,
        edge_weight=edge_contact_weight,
    )
    return result, falkenauer(plate_fills(result, coarse_piece_px, coarse_usable))


def fine_pack(order):
    result = pack(
        pieces,
        plate_shape,
        plate_mask=plate_mask,
        choose=choose,
        prerotated=fine_prerot,
        boundary=fine_bound,
        order=order,
        validate=False,
        edge_weight=edge_contact_weight,
    )
    return result, falkenauer(plate_fills(result, fine_piece_px, fine_usable))
```

`fine_piece_px`/`coarse_piece_px` stay as the BODY areas (`pieces[i].sum()` and
the coarse body variant), since the packed footprint is the body.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_improve.py -v`
Expected: PASS (new boundary case plus all existing improve tests — `boundary_pieces=None` leaves them unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/improve.py tests/test_improve.py
git commit -m "feat: thread boundary (full-shadow) masks through improve"
```

---

### Task 11: CLI wiring + full-shadow verify

**Files:**
- Modify: `src/plate_packer/cli.py`, `src/plate_packer/export.py`
- Test: `tests/test_cli.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `extract_footprints`, `DETECTOR_VERSION`, `rotate_pair`, `pack(..., boundary=)`,
  `improve(..., boundary_pieces=)`, `prepare_mask(kind=)`, `save_doc(body_mask=,…)`,
  `cfg.support_aware`, `cfg.support_cut_cap_mm`.
- Removes: `export.occupancy_from_full` and its test (superseded by shared-canvas verify).

**Behavior:**
1. `footprints` command: extract both masks, `save_doc(..., body_mask=, cut_z_mm=, detector_version=DETECTOR_VERSION)`.
2. `pack` command: on miss/`--force` extract+save both. When `cfg.support_aware`
   and the doc lacks a current body mask (`doc.body_mask is None` or
   `doc.detector_version != DETECTOR_VERSION`), re-extract that piece (fallback to
   full shadow, logged, if the STL read raises). Prepare `mask = prepare_mask(doc,
   spacing, res, kind="model_body")` and `full = prepare_mask(doc, spacing, res,
   kind="full_shadow")` when using body; else both are the full mask. Validate the
   FULL mask fits (`angle_candidates(full)` + `_fits`). Pack passes `boundary` =
   per-piece full variants when support-aware.
3. Placement→transform and verify use `rotate_mask(full, angle)` at `(row, col)`
   (the anchor is already in the full frame). Verify ORs `rotate_mask(full,
   angle)[0]` into the plate occupancy at `(row, col)` — plain placement. Off path
   unchanged (uses the packing mask as today).

- [ ] **Step 1: Remove `occupancy_from_full`**

Delete `occupancy_from_full` from `src/plate_packer/export.py` and its test
`test_occupancy_from_full_covers_body_placement` from `tests/test_export.py`
(the shared-canvas verify below replaces it).

- [ ] **Step 2: Write the failing CLI tests**

Add to `tests/test_cli.py`:

```python
import trimesh


def _fused_piece():
    """Solid raft (full outline) + a narrower body with the SAME outer bbox but a
    hollow centre — realistic: outer extent matches, interior differs. Rafts of
    neighbours overlap; nothing hangs off the plate."""
    raft = trimesh.creation.box(extents=(20, 20, 2))
    raft.apply_translation([0, 0, 1])
    left = trimesh.creation.box(extents=(4, 20, 20))
    left.apply_translation([-8, 0, 12])
    right = trimesh.creation.box(extents=(4, 20, 20))
    right.apply_translation([8, 0, 12])
    return trimesh.util.concatenate([raft, left, right])


def _write_pieces(stl_dir, n):
    stl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _fused_piece().export(str(stl_dir / f"piece_{i}.stl"))


def _cfg(tmp_path, support_aware):
    p = tmp_path / "config.toml"
    p.write_text(
        f"[packing]\nsupport_aware = {'true' if support_aware else 'false'}\n", encoding="utf-8"
    )
    return p


@pytest.mark.parametrize("support_aware", [False, True], ids=["off", "on"])
def test_pack_self_check_passes(tmp_path, support_aware):
    stl_dir = tmp_path / "stls"
    _write_pieces(stl_dir, 2)
    result = CliRunner().invoke(
        app,
        [
            "pack",
            str(stl_dir),
            "--config",
            str(_cfg(tmp_path, support_aware)),
            "--footprints-dir",
            str(tmp_path / "fp"),
            "--out",
            str(tmp_path / "out"),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verify" in result.output
    assert "FAILED" not in result.output
    assert (tmp_path / "out" / "plate_01.stl").exists()


def test_pack_support_aware_writes_body_mask(tmp_path):
    stl_dir = tmp_path / "stls"
    _write_pieces(stl_dir, 1)
    fp = tmp_path / "fp"
    CliRunner().invoke(
        app,
        [
            "pack",
            str(stl_dir),
            "--config",
            str(_cfg(tmp_path, True)),
            "--footprints-dir",
            str(fp),
            "--out",
            str(tmp_path / "out"),
            "--budget",
            "0",
        ],
    )
    from plate_packer.footprint_io import file_sha256, load_doc

    doc = load_doc(fp, file_sha256(next(stl_dir.glob("*.stl"))))
    assert doc.body_mask is not None
    assert doc.detector_version == 1
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "self_check or body_mask" -v`
Expected: FAIL (support-aware pack not wired; body mask not written).

- [ ] **Step 4: Implement the CLI**

Follow the interface/behavior list above. Concretely in `src/plate_packer/cli.py`:

- Imports: `from plate_packer.footprint import DETECTOR_VERSION, extract_footprints`
  (remove the old `extract_footprint` import — unused, ruff F401); add `rotate_pair`
  to the `plate_packer.packer` import; remove `occupancy_from_full` from imports.
- `footprints` command: compute `cap = load_config(None).support_cut_cap_mm` once,
  extract both, `save_doc(..., body_mask=body, cut_z_mm=cut, detector_version=DETECTOR_VERSION)`.
- `pack` piece loop: build `masks` (packing = body when support-aware, else full),
  `full_masks` (prepared full, == packing mask when off), `origins`, `areas`
  (undilated of the packed kind), plus the re-extract-when-stale logic. Validate
  the FULL mask fits an empty plate.
- Build per-piece boundary variants only when support-aware: for the budget>0
  path, `improve(masks, ..., boundary_pieces=full_masks)`; for `--budget 0`, build
  `prerot` (body) and `bound` (full) via `rotate_pair(full_masks[i], masks[i], a)`
  for each `a in angle_candidates(full_masks[i], ...)` and call `pack(masks, ...,
  prerotated=prerot, boundary=bound, ...)`. When off, keep today's `prerot`/`pack`
  with `boundary=None`.
- Transforms: `_, aff = rotate_mask(full_masks[pl.piece] if cfg.support_aware else masks[pl.piece], pl.angle)`.
- Verify: build each plate occupancy by ORing `rotate_mask(full_masks[pl.piece] if
  cfg.support_aware else masks[pl.piece], pl.angle)[0]` at `(pl.row, pl.col)`.
  Free `masks` and `full_masks` before the reload.

The full concrete pack-loop and verify code is in the brief's appendix (the
implementer should follow the interface list; report any ambiguity before coding).

- [ ] **Step 5: Run tests + full suite + lint**

Run: `uv run pytest tests/test_cli.py tests/test_export.py -v && uv run pytest tests/ -q && uv run ruff check && uv run ruff format --check`
Expected: new CLI tests pass (off AND on), `occupancy_from_full` test gone, full
suite green, lint clean.

- [ ] **Step 6: Commit**

```bash
git add src/plate_packer/cli.py src/plate_packer/export.py tests/test_cli.py tests/test_export.py
git commit -m "feat: CLI packs body with full-shadow plate boundary + verify"
```

---
### Task 12: Record decisions and key facts

**Files:**
- Modify: `docs/project_notes/decisions.md`
- Modify: `docs/project_notes/key_facts.md`

**Interfaces:** none (documentation only). `issues.md` and Claude's persistent
memory are updated at the merge prompt per the pre-merge checklist, not here.

- [ ] **Step 1: Add the ADR**

Append to `docs/project_notes/decisions.md` a new dated entry **ADR-013:
Support-aware footprints (base-layer exclusion)** capturing: opt-in
`support_aware` packs on a base-excluded `model_body` footprint; **footprint-area
knee** detection (cut where the projected footprint stops shrinking, within the
cap window; no cut unless it drops ≥ `MIN_REDUCTION`); `model_body` keyed on max-Z
(conservative superset above the cut). **Two-mask collision:** inter-piece uses
the body (rafts overlap freely — the whole point), the **plate boundary uses the
full shadow** (raft must stay on-plate), implemented via `rotate_pair` (body+full
share a canvas, so legality is a same-shape AND and verify is a plain OR of the
full mask); schema v2 with v1 read fallback. Note the detector evolution
(area-cliff and horizontal-cap tried and rejected — shells have no solid
cross-section, and caps cut too shallow), the measured −14% to −32% reduction on
real `*_supported.stl`, and that real rafts hug the model outline (outer flare
0.0 mm) so the gain is interior concavity. Alternatives rejected: fixed-mm cut,
per-piece numeric cut, band-stack, single-mask packing (raft off plate edge),
crop-offset verify (superseded by shared-canvas). Reference the spec path.

- [ ] **Step 2: Add the bug entry**

Append to `docs/project_notes/bugs.md` a dated entry: **detector reach map lost a
band to float32 + offset.** The single-pass top-reach map first stored
`max_Z − z0 + 1.0` in a float32 canvas; near ~1.5 the float32 ULP rounded off the
sub-mm fraction, so `area(d)` dropped raft-top pixels one `BAND_MM` early and the
cut landed too shallow — 0% real reduction on `*_supported.stl` while the
synthetic slab unit tests still passed. **Fix:** store `max_Z − z0` (no offset) in
a **float64** canvas; `area(d) = reach > d` then matches the float-compared body
mask exactly. Caught by the gated real-STL integration test, not the unit tests —
note that as the lesson.

- [ ] **Step 3: Add key facts**

Append to `docs/project_notes/key_facts.md`: the new config knobs
(`support_aware=false`, `support_cut_cap_mm=5.0`); detector constants
(`BAND_MM=0.25`, `MIN_REDUCTION=0.05`, `FLAT_EPS=0.01`, `DETECT_RES_MM=0.2`,
`DETECTOR_VERSION=1`, in `footprint.py`); the footprint-area-knee detector and the
measured −14% to −32% reduction on the Tome-of-Demons `*_supported.stl` corpus;
cache `SCHEMA_VERSION=2` with `model_body` entry + `cut_z_mm` /
`detector_version` metadata, reads accept `{1,2}`; the **two-mask collision**
(body vs pieces, full vs plate boundary) via `pack(..., boundary=)` /
`improve(..., boundary_pieces=)` / `rotate_pair`, with verify ORing the full mask
at each anchor; the `example_stls` pytest marker (deselected by default); and the
**stl_curator coordination item** — the contract now has an optional `model_body`
band that curator should eventually emit; plate_packer falls back gracefully until
it does.

- [ ] **Step 4: Commit**

```bash
git add docs/project_notes/decisions.md docs/project_notes/key_facts.md docs/project_notes/bugs.md
git commit -m "docs: ADR-013 support-aware footprints + key facts + bug"
```

---

## Notes for the executor

- **Run order matters for the safety net:** Tasks 1→2→3 land detection and the
  real-STL check before any cache/CLI wiring. Do not reorder.
- **Never weaken the off path.** In the packer (Task 9), `improve` (Task 10), and
  the CLI (Task 11), `boundary=None` / `support_aware=false` must run the exact
  pre-feature code so the default stays byte-identical. The two-mask path is a
  separate branch, never a rewrite of the single-mask path.
- **Coordinate landmine.** `rotate_pair` (Task 8) concentrates all crop-offset
  math in one place; every downstream step (pack AND, verify OR, export transform)
  works in the shared full frame with no offsets. The `rotate_pair` sign is locked
  by `test_rotate_pair_angle0_reconstructs_body` — do not "simplify" it away.
- **`example_stls` is a gitignored junction to copyrighted STLs.** Task 3's test
  must stay deselected by default and skip cleanly when it is absent.
- **`DETECTOR_VERSION` bump discipline:** any change to `BAND_MM`,
  `MIN_REDUCTION`, `FLAT_EPS`, or `DETECT_RES_MM` must bump `DETECTOR_VERSION` so
  support-aware runs re-extract stale body masks automatically.

# Raft-Signature Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a band-dominance acceptance gate to the base-cut detector so only true support rafts (pillar forests) opt into raft merging; model walls, plinths, and taper false-knees are rejected.

**Architecture:** One new private helper `_band_dominance` in `src/plate_packer/footprint.py`, called at the end of `detect_base_cut` after the area-knee is found. It rasterizes triangles straddling the plane `z0 + cut + BAND_MM` on the detection canvas and computes largest-component / total band area. Cut accepted iff dominance ≤ 0.35. Everything downstream (two-mask packer, cache, CLI) is unchanged except a `DETECTOR_VERSION` bump that invalidates cached body masks.

**Tech Stack:** Python 3.11+, numpy, opencv (`cv2.connectedComponentsWithStats`), trimesh (test meshes), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-08-raft-signature-gate-design.md` — read it first.

## Global Constraints

- `RAFT_BAND_DOMINANCE_MAX = 0.35` — module constant in `footprint.py`, NOT config (cache stays addressable by STL hash alone, ADR-009).
- `DETECTOR_VERSION` bumps 1 → 2 (Task 2, not Task 1 — tasks stay independently green).
- `support_cut_cap_mm` (5.0) and `MIN_REDUCTION` (0.05) are unchanged.
- Tests are parametrized and atomic (`pytest.mark.parametrize`, one named case each) — never loops inside a test body (user's global testing preference).
- All expected values below were verified by probe script against the real detector before writing this plan — do not "fix" them if a test fails; debug the implementation instead.
- Run tests with `uv run pytest ...`. Lint/format with `uv run ruff check .` and `uv run ruff format .` before every commit.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Band-dominance gate in `detect_base_cut`

**Files:**
- Modify: `src/plate_packer/footprint.py` (constants block ~line 14, `detect_base_cut` ~line 30)
- Modify: `tests/test_footprint.py` (helpers ~line 74, `test_detect_base_cut` parametrize ~line 85, `test_extract_footprints_body_subset_of_full_and_smaller` ~line 118, fallback test docstring ~line 135)
- Modify: `tests/test_cli.py` (`_fused_piece` ~line 276)

**Interfaces:**
- Consumes: existing `detect_base_cut(tris, res_mm, cap_mm)`, `_raster(tri_px, shape)`, `BAND_MM`.
- Produces: `detect_base_cut(tris, res_mm, cap_mm, *, gated: bool = True) -> float` (keyword `gated=False` returns the raw pre-gate knee — Task 3's probe tool uses it); module constant `RAFT_BAND_DOMINANCE_MAX = 0.35`; private `_band_dominance(z, tri_px, shape, z0, cut_mm) -> float`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_footprint.py`, replace the existing `_box` helper (line 74) with a version that takes an XY center, and add a pillar-layout constant right after `_tris`:

```python
def _box(xy, z_lo, z_hi, center=(0, 0)):
    """Axis-aligned box with XY extents `xy`, spanning Z [z_lo, z_hi]."""
    b = trimesh.creation.box(extents=(xy[0], xy[1], z_hi - z_lo))
    b.apply_translation([center[0], center[1], (z_lo + z_hi) / 2])
    return b


def _tris(*boxes):
    return trimesh.util.concatenate(list(boxes)).triangles


# 8 pillar positions inside a 20x20 raft: a synthetic support forest whose
# straddle band is 8 equal rings (dominance 1/8, well under the gate).
PILLARS = [(-7, -7), (-7, 0), (-7, 7), (0, -7), (0, 7), (7, -7), (7, 0), (7, 7)]
```

Replace the whole `test_detect_base_cut` parametrize list (lines 85–102) with:

```python
@pytest.mark.parametrize(
    ("tris", "expected"),
    [
        pytest.param(
            _tris(_box((20, 20), 0, 2), *[_box((1, 1), 2, 12, c) for c in PILLARS]),
            2.0,
            id="raft-then-pillar-forest",
        ),
        pytest.param(
            _tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12)),
            0.0,
            id="single-pillar-band-is-one-component-gate-rejects",
        ),
        pytest.param(
            _tris(_box((30, 30), 0, 2), _box((10, 10), 2, 10)),
            0.0,
            id="plinth-wall-ring-gate-rejects",
        ),
        pytest.param(
            _tris(*[_box((30 - 2 * k, 30 - 2 * k), k, k + 1) for k in range(8)]),
            0.0,
            id="staircase-taper-knee-at-cap-gate-rejects",
        ),
        pytest.param(
            _tris(_box((20, 20), 0, 1), _box((10, 10), 5, 8)),
            0.0,
            id="floating-body-empty-band-gate-rejects",
        ),
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


@pytest.mark.parametrize(
    ("tris", "expected_knee"),
    [
        pytest.param(
            _tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12)), 2.0, id="single-pillar"
        ),
        pytest.param(
            _tris(_box((30, 30), 0, 2), _box((10, 10), 2, 10)), 2.0, id="plinth"
        ),
        pytest.param(
            _tris(*[_box((30 - 2 * k, 30 - 2 * k), k, k + 1) for k in range(8)]),
            5.0,
            id="staircase-taper",
        ),
        pytest.param(
            _tris(_box((20, 20), 0, 1), _box((10, 10), 5, 8)), 1.0, id="floating-body"
        ),
    ],
)
def test_detect_base_cut_ungated_still_finds_knee(tris, expected_knee):
    """gated=False exposes the raw area-knee: proves the gate (not MIN_REDUCTION)
    is what rejects these shapes in test_detect_base_cut above."""
    assert detect_base_cut(tris, 0.1, 5.0, gated=False) == pytest.approx(
        expected_knee, abs=BAND_MM
    )
```

Update `test_extract_footprints_body_subset_of_full_and_smaller` (line 118) — its single-pillar mesh is now gate-rejected, so give it the forest (cut stays 2.0; verified at the internal `DETECT_RES_MM=0.2` too):

```python
def test_extract_footprints_body_subset_of_full_and_smaller():
    mesh = trimesh.util.concatenate(
        [_box((20, 20), 0, 2)] + [_box((1, 1), 2, 12, c) for c in PILLARS]
    )
    full, body, _origin, cut, stats = extract_footprints(mesh, RES, 5.0)
    assert full.shape == body.shape
    assert (body & ~full).sum() == 0  # body is a subset of full
    assert 0 < body.sum() < full.sum()  # raft slab removed
    assert cut == pytest.approx(2.0, abs=BAND_MM)
    assert stats["cut_mm"] == cut
```

Update the docstring of `test_extract_footprints_falls_back_when_cut_would_empty_body` (line 135) — the gate now rejects this mesh before the fallback (nothing straddles above a raft-only model), but the fallback stays as defense-in-depth. Replace the docstring only; keep the body unchanged:

```python
def test_extract_footprints_falls_back_when_cut_would_empty_body():
    """Raft-only input (whole model shorter than cut_cap_mm): the band gate
    rejects the knee first (nothing straddles above the model's own top), so
    detect returns 0.0 and body == full. The empty-body fallback inside
    extract_footprints stays as defense-in-depth behind the gate; this pins
    the observable contract: no cut, non-empty body identical to full."""
```

In `tests/test_cli.py`, replace `_fused_piece` (lines 276–286) — its two 4×20 slab walls are one band ring each (dominance 0.5, gate-rejected), which would silently degrade the CLI's two-mask coverage to body == full. Use a forest so the cut keeps firing (verified: cut 2.0, dominance 0.125–0.136 at both 0.05 and 0.2 mm/px):

```python
def _fused_piece():
    """Solid raft (full outline) + 8 support pillars + a narrower body slab —
    realistic: outer extent matches the raft, body shadow is smaller. Rafts of
    neighbours overlap; nothing hangs off the plate. The pillar forest keeps
    the band-dominance gate accepting the cut (8 rings, dominance 1/8)."""
    pillars = [(-7, -7), (-7, 0), (-7, 7), (0, -7), (0, 7), (7, -7), (7, 0), (7, 7)]
    parts = [trimesh.creation.box(extents=(20, 20, 2))]
    parts[0].apply_translation([0, 0, 1])
    for x, y in pillars:
        p = trimesh.creation.box(extents=(1, 1, 6))
        p.apply_translation([x, y, 5])
        parts.append(p)
    body = trimesh.creation.box(extents=(10, 10, 4))
    body.apply_translation([0, 0, 10])
    parts.append(body)
    return trimesh.util.concatenate(parts)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_footprint.py -v`
Expected: `test_detect_base_cut_ungated_still_finds_knee` cases ERROR with `TypeError: detect_base_cut() got an unexpected keyword argument 'gated'`; gate-reject cases in `test_detect_base_cut` FAIL (detector still returns 2.0/5.0/1.0 where 0.0 is expected). `raft-then-pillar-forest` and the untouched cases PASS.

- [ ] **Step 3: Implement the gate**

In `src/plate_packer/footprint.py`, add one constant after `DETECT_RES_MM` (line 17):

```python
RAFT_BAND_DOMINANCE_MAX = 0.35  # accept a cut iff largest band component <= 35% of band
```

Add the helper directly above `detect_base_cut`:

```python
def _band_dominance(z, tri_px, shape, z0, cut_mm) -> float:
    """Largest-connected-component share of the straddle band just above the cut.

    Rasterizes the triangles crossing the plane z0 + cut_mm + BAND_MM (outlines,
    since meshes are hollow shells): a support forest is many small pillar rings
    (low share), a model wall is one large ring (high share). Corpus-calibrated:
    true rafts 0.018-0.208, bogus cuts 0.556-1.0 (see 2026-08-08 spec). Returns
    1.0 when nothing straddles the plane -- no support evidence, caller rejects."""
    plane = z0 + cut_mm + BAND_MM
    straddle = (z.min(axis=1) < plane) & (z.max(axis=1) > plane)
    if not straddle.any():
        return 1.0
    band = _raster(tri_px[straddle], shape)
    if not band.any():
        return 1.0
    _n, _labels, comp_stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    sizes = comp_stats[1:, cv2.CC_STAT_AREA]  # row 0 is background
    return float(sizes.max() / sizes.sum())
```

Change `detect_base_cut`'s signature (line 30) to:

```python
def detect_base_cut(
    tris: np.ndarray, res_mm: float, cap_mm: float, *, gated: bool = True
) -> float:
```

Append to its docstring (after the existing "Returns 0.0 ..." sentence):

```
    A knee is then ACCEPTED only if the geometry just above it looks like a
    support forest (band dominance <= RAFT_BAND_DOMINANCE_MAX); model walls,
    plinths, and taper false-knees return 0.0. gated=False skips the gate and
    returns the raw knee (probe/diagnostic use).
```

Replace the final knee loop (lines 77–81):

```python
    thresh = a_min + FLAT_EPS * a_full
    for i in range(n_bands + 1):
        if areas[i] <= thresh:  # first depth on the plateau = the knee
            cut = float(i * BAND_MM)
            if not gated or _band_dominance(z, tri_px, shape, z0, cut) <= RAFT_BAND_DOMINANCE_MAX:
                return cut
            return 0.0
    return 0.0
```

(`z`, `z0`, `tri_px`, `shape` are already in scope in `detect_base_cut`; do not rebuild them.)

- [ ] **Step 4: Run the footprint tests**

Run: `uv run pytest tests/test_footprint.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: all PASS (CLI + integration tests use the updated `_fused_piece`; `tests/test_support_integration.py` is gated on `example_stls/` presence and still fires on real supported STLs — run it if the junction exists).

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/plate_packer/footprint.py tests/test_footprint.py tests/test_cli.py
git commit -m "feat: band-dominance gate rejects non-raft base cuts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Bump DETECTOR_VERSION to 2

**Files:**
- Modify: `src/plate_packer/footprint.py:18` (`DETECTOR_VERSION`)
- Modify: `tests/test_cli.py:355` (stale-version comment) and `tests/test_cli.py:404-408` (`test_pack_support_aware_writes_body_mask`)

**Interfaces:**
- Consumes: `DETECTOR_VERSION` (currently 1), CLI re-extraction on version mismatch (`cli.py:202`).
- Produces: `DETECTOR_VERSION = 2`. No signature changes.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, `test_pack_support_aware_writes_body_mask` (line 385): make the assert version-agnostic AND pin the bump. Replace the last three lines (406–408) with:

```python
    from plate_packer.footprint import DETECTOR_VERSION
    from plate_packer.footprint_io import file_sha256, load_doc

    doc = load_doc(fp, file_sha256(next(stl_dir.glob("*.stl"))))
    assert doc.body_mask is not None
    assert doc.detector_version == DETECTOR_VERSION
    assert DETECTOR_VERSION == 2  # gate added 2026-08-08: stale v1 body masks must regenerate
```

Also update the comment on `tests/test_cli.py:355` from `# stale: real DETECTOR_VERSION is 1` to `# stale: any value != footprint.DETECTOR_VERSION`.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_pack_support_aware_writes_body_mask -v`
Expected: FAIL on `assert DETECTOR_VERSION == 2` (it is 1).

- [ ] **Step 3: Bump the constant**

In `src/plate_packer/footprint.py` line 18, change:

```python
DETECTOR_VERSION = 2  # bump when any detector constant changes (v2: band-dominance gate)
```

- [ ] **Step 4: Run the suite**

Run: `uv run pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/plate_packer/footprint.py tests/test_cli.py
git commit -m "feat: DETECTOR_VERSION 2 invalidates pre-gate cached body masks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Corpus probe tool + regression run

**Files:**
- Create: `tools/probe_raft_gate.py` (dev tool, not part of the package — `tools/` is a new top-level dir)

**Interfaces:**
- Consumes: `detect_base_cut(..., gated=False)` from Task 1, `_band_dominance` internals reimplemented locally is FORBIDDEN — import `BAND_MM`, `DETECT_RES_MM`, `_raster`, `detect_base_cut` from `plate_packer.footprint` and `load_piece_mesh` from `plate_packer.export`.
- Produces: TSV on stdout: `path  knee_mm  dominance  n_comp  verdict  secs`. Verdict is `accept` / `reject` / `no-knee`.

- [ ] **Step 1: Write the tool**

```python
"""Corpus probe for the raft-signature gate (not part of the package).

For every STL under a root: raw area-knee (gated=False), band dominance at the
knee, and the gate verdict. Used to calibrate/recheck RAFT_BAND_DOMINANCE_MAX
against a real corpus (see docs/superpowers/specs/2026-08-08-raft-signature-
gate-design.md). Runtime is a few seconds per multi-million-triangle STL; use
start/count to chunk long runs.

Usage: uv run python tools/probe_raft_gate.py <root_dir> [start] [count]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from plate_packer.export import load_piece_mesh
from plate_packer.footprint import (
    BAND_MM,
    DETECT_RES_MM,
    RAFT_BAND_DOMINANCE_MAX,
    _raster,
    detect_base_cut,
)


def probe(path: Path) -> tuple[float, float, int, str]:
    tris = np.asarray(load_piece_mesh(path).triangles)
    tris = tris[np.isfinite(tris).all(axis=(1, 2))]
    knee = detect_base_cut(tris, DETECT_RES_MM, 5.0, gated=False)
    if knee <= 0:
        return 0.0, 0.0, 0, "no-knee"
    xy = tris[:, :, :2]
    z = tris[:, :, 2]
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / DETECT_RES_MM).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / DETECT_RES_MM).astype(np.int32)
    plane = float(z.min()) + knee + BAND_MM
    straddle = (z.min(axis=1) < plane) & (z.max(axis=1) > plane)
    band = _raster(tri_px[straddle], shape) if straddle.any() else np.zeros(shape, np.uint8)
    if not band.any():
        return knee, 1.0, 0, "reject"
    _n, _labels, comp_stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    sizes = comp_stats[1:, cv2.CC_STAT_AREA]
    dom = float(sizes.max() / sizes.sum())
    verdict = "accept" if dom <= RAFT_BAND_DOMINANCE_MAX else "reject"
    return knee, dom, len(sizes), verdict


def main() -> None:
    root = Path(sys.argv[1])
    paths = sorted(root.rglob("*.stl"), key=lambda p: str(p).lower())
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else len(paths)
    for path in paths[start : start + count]:
        t0 = time.perf_counter()
        try:
            knee, dom, n_comp, verdict = probe(path)
        except Exception as e:  # noqa: BLE001 - probe must survive bad meshes
            print(f"{path}\tERROR\t{e}", flush=True)
            continue
        print(
            f"{path}\t{knee:.2f}\t{dom:.3f}\t{n_comp}\t{verdict}"
            f"\t{time.perf_counter() - t0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
```

Note: the canvas/dominance recomputation here intentionally mirrors `_band_dominance` at probe scope (the helper needs internals `detect_base_cut` already built; the tool rebuilds them once per file). Keep the reject rule identical: empty band ⇒ reject.

- [ ] **Step 2: Smoke-run on one folder**

Run: `uv run python "tools/probe_raft_gate.py" "example_stls/Archvillain Games - Tome of Demons Volume 1/Armaros, Chaos Incarnate" 2>&1 | head -25`
Expected: TSV rows; `*Supported*` files mostly `accept` with dominance < 0.21; unsupported files `no-knee`.

- [ ] **Step 3: Full corpus regression (chunked, ~5 s/file, 229 files — 4 chunks of ~60 stay under 10 min each)**

```bash
# SCRATCH = the session scratchpad directory from your context (never the repo)
PYTHONUNBUFFERED=1 uv run python tools/probe_raft_gate.py example_stls 0 60   | tee "$SCRATCH/probe_1.tsv" | tail -1
PYTHONUNBUFFERED=1 uv run python tools/probe_raft_gate.py example_stls 60 60  | tee "$SCRATCH/probe_2.tsv" | tail -1
PYTHONUNBUFFERED=1 uv run python tools/probe_raft_gate.py example_stls 120 60 | tee "$SCRATCH/probe_3.tsv" | tail -1
PYTHONUNBUFFERED=1 uv run python tools/probe_raft_gate.py example_stls 180 60 | tee "$SCRATCH/probe_4.tsv" | tail -1
cat "$SCRATCH"/probe_*.tsv | cut -f5 | sort | uniq -c
```
Expected counts: `accept` = 101, `reject` = 13, `no-knee` = 115, ERROR = 0. These must match the calibration exactly; any drift is a bug in Task 1 — stop and debug, do not adjust the threshold.

- [ ] **Step 4: Lint, commit**

```bash
uv run ruff check tools/ && uv run ruff format tools/
git add tools/probe_raft_gate.py
git commit -m "feat: corpus probe tool for the raft-signature gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Project notes

**Files:**
- Modify: `docs/project_notes/decisions.md` (append ADR-014)
- Modify: `docs/project_notes/key_facts.md`
- Modify: `docs/project_notes/bugs.md`
- Modify: `docs/project_notes/issues.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Append ADR-014 to decisions.md** (before the `## Usage Tips` section, matching existing ADR format):

```markdown
### ADR-014: Raft-signature gate (band-dominance acceptance) (2026-08-08)

**Context:**
- The two-mask collision model (ADR-013) confines overlap to raft∩raft, but is only as safe as classification: the area-knee detector fires on any mesh whose shadow shrinks ≥5% within the cap and plateaus — including 13 unsupported corpus meshes (wings on a tip, merged cloth; all cap-adjacent knees) and hypothetical integral plinths. A bogus cut opts model geometry into raft fusion.

**Decision:**
- Accept a knee only if the geometry just above it looks like a support forest: rasterize triangles straddling `z0 + cut + BAND_MM` (cross-section outlines) and require largest-component/band-area ≤ `RAFT_BAND_DOMINANCE_MAX = 0.35`; empty band ⇒ reject. Module constant, not config (ADR-009 hash-addressability). `DETECTOR_VERSION` 1→2. `detect_base_cut(..., gated=False)` exposes the raw knee for probing (`tools/probe_raft_gate.py`).
- Corpus calibration (Tome of Demons, 229 STLs): true rafts dominance 0.018–0.208 (n=101, knees 0.25–1.25 mm), bogus cuts 0.556–1.0 (n=13, knees 4.5–5.0 mm) — 2.7× gap around 0.35. Gate: 101 accepts / 13 rejects, exact.

**Alternatives Considered:**
- Band *area fraction* (straddle area / footprint) → does NOT separate: meshes are hollow shells, so a wing's wall ring is as sparse by area (0.02–0.08) as a support forest. Dominance measures connectedness — the actual physical difference — and is scale-free.
- `raft_window_mm` accept-window on knee depth → redundant (deep knees are exactly the high-dominance ones) and cannot catch a shallow plinth.
- Filename allowlist (`*supported*`) → corpus has 7 supported exports missing the suffix; naming is unreliable.

**Consequences:**
- False rejects (few-pillar minis; corpus min is 22 components) cost density only, never correctness. Remaining false-accept shape — a field of separate thin spikes off the plate — is physically raft-like; accepted risk.
- Smooth synthetic tapers (e.g. `trimesh.creation.cone`) never fire the knee at all (every side triangle reaches the apex, so the reach map never drops); fine-tessellated real tapers do. Synthetic taper tests must use stacked shrinking boxes.
```

- [ ] **Step 2: key_facts.md** — add under the detector/support-aware section (match existing bullet style):

```markdown
- Raft-signature gate (ADR-014): `RAFT_BAND_DOMINANCE_MAX = 0.35` in `footprint.py`; `DETECTOR_VERSION = 2`; recalibrate with `tools/probe_raft_gate.py <root>` (expects 101 accept / 13 reject / 115 no-knee on the Tome corpus).
```

- [ ] **Step 3: bugs.md** — append:

```markdown
## 2026-08-08 — corpus labels and band-area dead end (raft-signature gate)
- Filename-derived labels lied: 7 publisher supported exports lack "supported" in the name (`STL_Pose1_Body.stl` etc.). Population splits by filename need signature-level verification.
- Band *area fraction* failed as a raft discriminator (hollow shells: wall rings as sparse as pillar forests); component *dominance* separates 2.7×. Lesson: calibrate on the real corpus before trusting a physically-plausible metric.
- `trimesh.creation.cone` can't test taper false-knees: every side triangle reaches the apex so the reach map never drops. Use stacked shrinking boxes.
```

- [ ] **Step 4: issues.md** — append a dated work-log entry:

```markdown
## 2026-08-08 — raft-signature gate (ADR-014)
- Band-dominance acceptance gate in `detect_base_cut`; `DETECTOR_VERSION` 2; `tools/probe_raft_gate.py`; corpus-calibrated 0.35 (101/13 exact split). Spec: `docs/superpowers/specs/2026-08-08-raft-signature-gate-design.md`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/project_notes/
git commit -m "docs: ADR-014 raft-signature gate + key facts, bugs, work log

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

# Support-Aware Footprints — Base-Layer Exclusion Design

**Date:** 2026-08-07
**Builds on:** ADR-009 (content-addressed footprint cache), ADR-010 (spacing
semantics), the merged-shadow self-check (`export.verify_plate`).
**Motivation:** Every footprint today is the full vertical shadow of *all* mesh
triangles (`footprint.extract_footprint` discards Z at line 28 and unions every
triangle in XY). That includes the raft / support base at the bottom of a
pre-supported model. The base structure is roughly convex and sits under the
whole support forest, so it fills the concavities of wings, tails, and cradles —
exactly the geometry that would otherwise nest well. For a user who accepts that
**rafts may overlap and fuse** (snipped apart in post), excluding the base band
from the collision footprint recovers those concavities and packs denser.

## Goal

Add an opt-in mode that packs on a **base-excluded ("model body") footprint** —
the full shadow minus the geometry below an auto-detected cut height — while
leaving the default (full-shadow) behavior byte-for-byte unchanged. The cut is
detected per model from mesh geometry, clamped by a safety cap, and cached
alongside the full shadow.

## Physical model (the assumption this rests on)

All input models are pre-supported and carry a raft / support base on their
first few millimeters. Below the cut height, **any overlap between pieces is
permitted** — their bases fuse into shared cured resin at the plate, and the
user separates them in post. Above the cut, the ordinary conservative-shadow
collision guarantee is unchanged: if two body shadows are disjoint, the pieces
cannot intersect at any Z above the cut. The feature never weakens collision
above the cut; it only *frees* the base band.

Consequence, stated plainly: turning this on produces plates where rafts touch
and fuse. That is the intended trade for density, and why the mode is opt-in.

## 1. Two masks per footprint (`footprint.py`, `footprint_io.py`)

`extract_footprint` stops discarding Z. It rasterizes **two masks on one shared
canvas / origin**:

- **`full_shadow`** — every finite triangle projected to XY. Identical to today's
  output, same origin, same canvas size.
- **`model_body`** — the shadow of every triangle that **reaches above** the cut
  (max Z `> cut_z`), painted onto the *same* canvas. Cleared out are only
  triangles lying **entirely below** the cut (max Z `≤ cut_z`) — the base band.
  Inclusion keys on *max* Z, not min Z: a triangle straddling the cut is kept in
  full, so `model_body` is a conservative **superset** of the true above-cut
  cross-section and can never invent free space above the cut. Its origin and
  shape match `full_shadow` exactly, so every downstream transform is identical.

Sharing the canvas is load-bearing: the export transform in `cli.pack_command`
derives placement from the prepared mask's `origin_mm`. Packing on `model_body`
must place the *whole* STL (raft included) correctly, so `model_body` must not be
re-cropped to a tighter bbox — it keeps `full_shadow`'s origin.

Both masks are stored in the cache doc (see §5). Because the cut is a pure
function of mesh geometry (no user parameter enters extraction except the global
cap constant, versioned into the doc), the doc stays addressable by STL hash
alone — ADR-009 holds.

## 2. Auto-detection: the raft's flat top (`footprint.py`)

STL meshes are hollow **shells**, not solids: a slab has no triangles across its
interior at mid-height — only its flat top and bottom **cap** faces carry the
filled area (an interior Z-slab intersects only the vertical side walls, which
project to a thin perimeter, not the slab). A raft / support base is exactly such
a big flat horizontal surface, so we detect it **directly by its cap**, not by an
area profile. **The safety cap is the search window** — we look only within
`[z0, z0 + support_cut_cap_mm]`, never scan the whole model, never cut deeper:

1. Let `z0 = min triangle Z`, `z1 = max triangle Z`. If `z1 − z0 ≤ BAND_MM` →
   cut = 0. Let `A_full` = the full-shadow occupied-pixel area.
2. **Near-horizontal faces** are those whose unit normal has `|n_z| > HORIZ_NZ`
   (initial `0.9`, ~within 25° of flat); degenerate zero-area faces are excluded.
   Bin each horizontal face by its **mean Z** into bands of height `BAND_MM`
   (initial `0.25`).
3. For each band within `[z0, min(z0 + support_cut_cap_mm, z1)]`, rasterize the
   shadow of that band's horizontal faces on the *full-shadow canvas/origin* and
   record the cap area `A_cap`.
4. The cut is the **highest** band whose `A_cap ≥ MIN_BASE_FRAC × A_full` — the
   top-most substantial flat shelf, i.e. the raft's top surface. No qualifying
   cap in the window → cut = 0. `MIN_BASE_FRAC` starts **sensitive** (initial
   `0.10` — a flat shelf covering ≥10% of the footprint counts as a raft, so
   Lychee's thin-strip bases are caught); walk it **up** once functionality is
   proven, re-packing to see how aggressive we can safely be.
5. `model_body` = shadow of triangles with max Z `> z0 + cut` (§1); when
   cut = 0, `model_body` is a copy of `full_shadow`.

Fail-safe by construction — every ambiguous case degrades to "no cut," i.e.
`model_body == full_shadow`, which is exactly today's behavior:

- **No base** (supports printed straight to the plate) → no substantial flat cap
  → cut = 0.
- **Wide solid box / bust on a plinth** — its only big caps are the bottom
  (z0, a no-op) and the very top (outside the cap window) → highest in-window cap
  is at z0 → cut = 0.
- **Cap too small to be a raft** (`A_cap` below `MIN_BASE_FRAC × A_full`) → no
  qualifying cap → cut = 0.
- **Tall solid base** (base persists past the cap window, so its top cap is above
  the window) → no qualifying cap inside the window → cut = 0. No benefit for that
  piece, and **nothing fuses** — the safe outcome. (The only residual risk is a
  *real* model with a big flat horizontal shelf inside its first
  `support_cut_cap_mm` covering ≥ `MIN_BASE_FRAC` of the footprint; bounded by the
  cap, covered by the deferred per-piece disable.)

`BAND_MM`, `HORIZ_NZ`, and `MIN_BASE_FRAC` are **module constants**, not config —
detector internals, tuned against real STLs and pinned by tests. Only the enable
flag and the cap are user-facing.

### Lychee reality (why §6 lands early)

Lychee "Export 3D Asset" often uses a **skate / interface base**: thin flat
strips at z ≈ 0 connecting the pillar feet, *not* a solid slab. Those strips are
still flat-topped, so they present a horizontal cap — but one covering only a
fraction of the footprint. `MIN_BASE_FRAC` must be low enough to catch a strip
lattice, and the true density gain on strip-style bases is an open empirical
question. The detector's core assumption — that a real pre-supported base shows a
detectable horizontal cap — is therefore verified against a real `example_stls`
file **early** (§6), before anything is built on top of it.

## 3. Self-check stays honest (`cli.py`, `export.verify_plate`)

`verify_plate` rasterizes the merged output STL and asserts its shadow is a
**subset** of the predicted plate occupancy. The merged STL still contains full
rafts, so if the prediction were built from `model_body` the raft pixels would
fall outside it and every plate would fail verification.

Resolution: **pack on `model_body`, but build the verification occupancy from
`full_shadow`.** The merged output (rafts included) is then a subset of the
prediction, and the extraction → pack → transform → export round-trip stays fully
validated. The only assertion deliberately dropped is raft-vs-raft
non-overlap — precisely the freedom this feature introduces. Both masks are
cached, so `full_shadow` is on hand at verify time regardless of the packing
mask.

Placing `full_shadow` at the *same world location* as the packed `model_body`
needs no new coordinate math: the two prepared masks share an identical
un-cropped rotation canvas (same extraction canvas → same downsample → same
dilation), so a piece placed at body anchor `(row, col)` has its full-shadow
anchor at `row + (aff_body[1,2] − aff_full[1,2])`, `col + (aff_body[0,2] −
aff_full[0,2])`, where `aff_*` are the 2×3 affines `rotate_mask` returns for the
two masks at the piece's angle. Pure integer placement (clipped to the plate),
reusing `rotate_mask` — no resampling.

When `support_aware` is off, packing and verification both use `full_shadow` by
the existing code path, so behavior is byte-identical.

## 4. Config surface & defaults (`config.py`)

```toml
[packing]
support_aware = false        # opt-in; off = today's behavior exactly
support_cut_cap_mm = 5.0     # safety clamp on the auto-detected cut
```

- **`support_aware: bool = False`.** Off by default: fusing rafts is a workflow
  choice, so the default output stays physically conventional and the change is
  fully reversible.
- **`support_cut_cap_mm: float = 5.0`.** Sized for a typical 2–4 mm raft + burn-in
  + start of the support segment, with headroom, while still clamping a genuinely
  tall base. Validated `> 0`.
- **Detector internals are not config** (see §2).
- **Per-piece override is deferred to a fast-follow.** v1 is a global on/off.
  Because both masks are always cached, a later "use full shadow for these
  stems" exclusion is trivial and needs no re-extraction. Not load-bearing for
  the density win.

Mask selection lives in `loading.prepare_mask`, which gains a parameter naming
which cached mask to prepare (`full_shadow` default; `model_body` when
`support_aware` and the doc carries a body mask). `cli.pack_command` passes the
body mask to the packer and the full mask to verification.

## 5. Cache schema & stl_curator coordination (`footprint_io.py`)

Bump `SCHEMA_VERSION` 1 → 2:

- The `footprints` list gains a second entry, `kind: "model_body"`, with
  `z_band_mm: [cut_z_mm, null]` and its own PNG.
- Doc-level detection metadata: `cut_z_mm`, `cap_mm`, `detector_version` (an int
  bumped when detector constants change — see the re-extraction rule below).
- `FootprintDoc` keeps `masks` = `[full_shadow]` (unchanged for existing
  consumers) and gains optional `body_mask`, `cut_z_mm`, `detector_version`
  fields, populated from the `model_body` entry when present.

**Reads accept both v1 and v2.** plate_packer's own extraction always writes v2.
A v1 doc — e.g. one written by stl_curator, which does not know about body
masks — reads cleanly with no `model_body`.

Re-extraction is scoped so the upgrade cost falls only on opt-in users, once:

- `has_current_doc` still treats **both v1 and v2 as current** for the *default*
  path, so support-off users never re-extract and see no behavior change.
- When `support_aware` is on, pack additionally requires a *current* body mask:
  if the loaded doc lacks `model_body` (a v1 doc, or a curator-written v2 without
  it) **or** its `detector_version` differs from the code's current
  `DETECTOR_VERSION`, that piece is re-extracted to v2 so the body mask exists and
  is up to date. Bumping `DETECTOR_VERSION` when you retune `MIN_BASE_FRAC`
  therefore
  auto-invalidates stale body masks on the next support-aware run — no `--force`
  needed. A piece whose STL is unavailable at pack time falls back to
  `full_shadow` and logs it.

This gives opt-in users immediate benefit while imposing nothing on everyone
else, and it self-heals if stl_curator later overwrites a doc with v1 (content is
addressed by STL hash; v2 is a strict superset, re-derived on the next
support-aware run).

This is a contract change stl_curator should eventually mirror (emit the
`model_body` band). Recorded as a coordination item in `key_facts.md` /
persistent memory; no code dependency on curator adopting it.

## 6. Testing (TDD)

Parametrized, atomic — one named case per input, per the global rule.

**Unit — detection (synthetic meshes, no I/O):**
- Slab raft + thin pillars → cut at the raft top.
- No base (pillars to plate) → cut = 0.
- Wide solid box (only z0 + top caps, top outside window) → cut = 0.
- Small foot cap below `MIN_BASE_FRAC × A_full` → cut = 0.
- Tall solid base persisting past the cap window (top cap above window) → cut = 0.

**Unit — masks & doc:**
- `model_body ⊆ full_shadow`, identical origin/shape, base pixels cleared.
- Doc round-trips both masks + metadata; `FootprintDoc` exposes them by kind.
- v1 doc (no body mask) reads cleanly; body absent → fallback path.

**Unit — self-check:**
- Two pieces whose bodies nest but whose rafts overlap → `verify_plate` passes
  when occupancy is built from `full_shadow`.

**Unit — toggle:**
- `support_aware = false` ⇒ pack + verify identical to pre-feature behavior on a
  fixed fixture (guards the "off = unchanged" promise).

**Config:**
- Defaults present (`support_aware=False`, `support_cut_cap_mm=5.0`); TOML
  override; validation (`support_cut_cap_mm > 0`).

**Integration — real STL, gated, landed early:**
- Runs the detector on a known `example_stls` file; asserts a sensible cut is
  found (past the z=0 strip band, within the cap) and reports the **area
  reduction** `1 − area(model_body)/area(full_shadow)` so the real-world benefit
  is visible.
- Gated by a `pytest` marker `example_stls`, **deselected by default** in
  `pyproject.toml` (`addopts = "-m 'not example_stls'"` or equivalent), and
  additionally `skipif` the `example_stls` junction is absent — so CI and a plain
  `pytest` run both skip it; opt in locally with `pytest -m example_stls`.
- Lands immediately after the detector exists, before mask/doc/pack wiring, so
  the core assumption is validated against reality first.

## Out of scope (v1)

- Per-piece override (fast-follow; both masks already cached).
- stl_curator emitting body masks (coordination item; graceful fallback covers
  the gap).
- Numeric per-piece cut heights / a multi-band mask stack.
- Exposing detector internals (`BAND_MM`, `HORIZ_NZ`, `MIN_BASE_FRAC`) as
  config.
- Any change to default (support-off) behavior.

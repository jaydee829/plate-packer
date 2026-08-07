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

## 2. Auto-detection: the footprint-area knee (`footprint.py`)

Excluding the base is only worthwhile where it actually **shrinks the packed
footprint**, so we detect the cut from the projected-area profile itself. Sweep
the cut depth and watch the model-body footprint (shadow of triangles reaching
above the cut) shrink: on a real pre-supported model the area drops sharply
through the raft / support base in the first 1–2 mm, then goes **dead flat** — a
clean knee (empirically −14% to −32% on Tome-of-Demons wings/tails/bodies). Cut
at the knee. **The safety cap is the search window** — depths only within
`[z0, z0 + support_cut_cap_mm]`, never deeper.

Computed in a **single raster pass**, no per-depth re-rasterization: paint each
pixel with `max Z − z0` of the tallest triangle covering it (fill in ascending
`max Z` so the tallest paints last) → a per-pixel **top-reach** map, in **float64
with no additive offset**. Then `area(d)` = pixels whose reach `> d`, read straight
off that map for every depth. (The offset/precision matters: a float32 map with a
`+1` offset rounds off the sub-ULP fraction at band boundaries and drops raft-top
pixels a band early — cutting too shallow on real STLs while synthetic slab tests
still pass.) The reach map is rasterized at a coarse `DETECT_RES_MM` (initial
`0.2`) — area *ratios* are scale-tolerant, and the coarser grid keeps the fill
cheap.

1. `z0 = min Z`. `A_full = area(0)`. If `z1 − z0 ≤ BAND_MM` → cut = 0.
2. Over depths `d = 0, BAND_MM, 2·BAND_MM, …` within `[0, support_cut_cap_mm]`,
   let `A_min = min area(d)` — the most the footprint shrinks inside the window.
3. If `A_full − A_min < MIN_REDUCTION × A_full`, the base isn't worth excluding →
   cut = 0.
4. Otherwise cut at the **smallest** `d` with `area(d) ≤ A_min + FLAT_EPS × A_full`
   — the start of the plateau (the knee).
5. `model_body` = shadow of triangles with max Z `> z0 + cut` (§1); when
   cut = 0, `model_body` is a copy of `full_shadow`.

Fail-safe by construction — any model whose footprint doesn't drop by at least
`MIN_REDUCTION` inside the window degrades to "no cut" (`model_body ==
full_shadow`, today's behavior):

- **No base / supports straight to plate** — footprint stable → cut = 0.
- **Wide solid box / bust on a plinth** — footprint doesn't shrink in the window
  → cut = 0.
- **Tall solid base** (persists past the cap window) — no plateau reached inside
  the window → cut = 0. No benefit, and **nothing fuses** — the safe outcome.
  (Residual risk: a *real* model whose footprint genuinely collapses ≥
  `MIN_REDUCTION` within its first `support_cut_cap_mm`; bounded by the cap,
  covered by the deferred per-piece disable.)

`BAND_MM`, `MIN_REDUCTION`, `FLAT_EPS`, and `DETECT_RES_MM` are **module
constants**, not config — detector internals, tuned against real STLs and pinned
by tests. `MIN_REDUCTION` (initial `0.05`) is the sensitivity knob — how much
footprint gain justifies fusing bases; **start sensitive (0.05) and raise** it as
we tighten. Only the enable flag and the cap are user-facing.

### Real-STL findings (why §6 landed early)

Probing the real Tome-of-Demons `*_supported.stl` exports settled the detector.
The **raw kit parts have no rafts** (they float at assembly Z, no flat base), so
they correctly yield cut = 0 — support-aware needs the *supported* files. On the
supported files the footprint drops sharply through the base in the first 1–2 mm
then plateaus flat to 12 mm, giving **−14% to −32%** footprint reduction (wings,
tails, winged body). This is what motivated the area-knee detector over earlier
cap-based ideas, and it is the payoff the feature exists to capture. The
integration test (§6) runs on the `*_supported.stl` corpus and asserts a real cut
and reduction, landed **early** so the detector was validated before the rest was
built on top of it.

## 3. Two-mask packing: body for pieces, full for the plate

Packing only on `model_body` is unsafe at the **plate boundary**: the piece is
placed by its (narrower) body, but the (wider) full shadow — the raft — can then
hang off the plate edge, which cannot print. Real Lychee rafts hug the model
outline (measured outer flare **0.0 mm** on the Tome-of-Demons corpus, so the
gain is entirely *interior* concavity), but a raft that flares even slightly past
the body would overhang, so the packer enforces it explicitly rather than relying
on that.

**Collision model (support-aware):** a placement is legal iff
- the **body** does not overlap already-placed **bodies** (rafts freely overlap —
  the feature's whole point), **and**
- the **full shadow** lies within the plate and clear of its dead margins.

**Shared-canvas mechanic.** For each piece/angle, the body and full masks are
rotated onto **one shared canvas** (`rotate_pair`: rotate the full mask with
`rotate_mask`, then place the body on the full's cropped canvas), so `body_rot`
and `full_rot` have **identical shape and anchor**. Legality is then a plain
same-shape AND of two `legal_placement_map` calls — `legal_placement_map(pieces,
body_rot) & legal_placement_map(plate_border, full_rot)` — with no crop-offset
arithmetic anywhere. Placement `(row, col, angle)` lives in this shared (full)
frame. Block-max downsampling to the coarse resolution keeps both masks the same
shape, so the coarse phase ANDs identically. The empty-plate fit check (ADR-004)
uses the **full** mask (the binding constraint).

**Self-check.** `verify_plate` asserts the merged output shadow ⊆ predicted
occupancy. The output contains full rafts, so the prediction is built from the
**full** shadow: OR each piece's `rotate_mask(full, angle)` into the plate
occupancy at its `(row, col)` — a plain placement, since the anchor is already in
the full frame. The extraction → pack → transform → export round-trip stays fully
validated; the only assertion dropped is raft-vs-raft non-overlap. The export
transform likewise derives from `rotate_mask(full, angle)` + `(row, col)`.

**Off path unchanged.** When `support_aware` is off, `boundary` is absent, the
packer runs its existing single-mask path (`full_shadow` vs a merged
border+pieces occupancy), and verification uses the existing occupancy loop —
byte-identical to today.

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
  is up to date. Bumping `DETECTOR_VERSION` when you retune `MIN_REDUCTION`
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
- Slab raft + thin pillars → cut at the raft top (footprint collapses there).
- No base / pillar-only (footprint stable) → cut = 0.
- Wide solid box (footprint stable) → cut = 0.
- Tiny foot under a big body (drop below `MIN_REDUCTION`) → cut = 0.
- Base taller than the cap window (no plateau in window) → cut = 0.

**Unit — masks & doc:**
- `model_body ⊆ full_shadow`, identical origin/shape, base pixels cleared.
- Doc round-trips both masks + metadata; `FootprintDoc` exposes them by kind.
- v1 doc (no body mask) reads cleanly; body absent → fallback path.

**Unit — shared-canvas rotation (`rotate_pair`):**
- `rotate_pair(full, full, angle)` → the pair is identical (degenerate case).
- `body ⊆ full` ⇒ `body_rot ⊆ full_rot`, same shape, at several angles.

**Unit — two-mask packing:**
- Rafts overlap: two pieces whose bodies fit disjoint but whose full shadows
  overlap → both placed on one plate (body-vs-body legality only).
- Plate boundary: a piece whose body fits flush at the edge but whose full shadow
  would overhang is pushed inward (full-vs-border legality) — no off-plate raft.
- Empty-plate fit uses the full mask (a piece whose full exceeds the plate is
  rejected even if its body fits).

**Unit — self-check:**
- Two pieces whose bodies nest but whose rafts overlap → `verify_plate` passes
  (occupancy ORed from `full_shadow` at each anchor).

**Unit — toggle:**
- `support_aware = false` ⇒ `boundary` unused; pack + verify identical to
  pre-feature behavior on a fixed fixture (guards the "off = unchanged" promise).

**Config:**
- Defaults present (`support_aware=False`, `support_cut_cap_mm=5.0`); TOML
  override; validation (`support_cut_cap_mm > 0`).

**Integration — real STL, gated, landed early:**
- Runs `extract_footprints` on real **`*_supported.stl`** files under
  `example_stls` (the raw kit parts have no rafts, so the test targets the
  supported exports). Asserts at least one has `cut > 0` and a real footprint
  reduction, and prints each file's cut + `1 − area(model_body)/area(full_shadow)`
  so the benefit is visible (`-s`).
- Gated by a `pytest` marker `example_stls`, **deselected by default** in
  `pyproject.toml` (`addopts = "-m 'not example_stls'"`), and additionally
  `skipif` the `example_stls` junction is absent — so CI and a plain `pytest` run
  both skip it; opt in locally with `pytest -m example_stls -s`.
- Landed immediately after the detector existed, before mask/doc/pack wiring — it
  is what surfaced the raw-vs-supported distinction and drove the area-knee
  detector.

## Out of scope (v1)

- Per-piece override (fast-follow; both masks already cached).
- stl_curator emitting body masks (coordination item; graceful fallback covers
  the gap).
- Numeric per-piece cut heights / a multi-band mask stack.
- Exposing detector internals (`BAND_MM`, `MIN_REDUCTION`, `FLAT_EPS`,
  `DETECT_RES_MM`) as
  config.
- Any change to default (support-off) behavior.

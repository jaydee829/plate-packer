# Raft-Signature Gate — Band-Dominance Acceptance for the Base-Cut Detector

**Date:** 2026-08-08
**Builds on:** ADR-013 (support-aware footprints, two-mask collision), the
area-knee detector (`footprint.detect_base_cut`), cache detector versioning
(`DETECTOR_VERSION`).
**Status:** designed; calibrated on the Tome of Demons corpus (229 STLs).

## Problem

The two-mask collision model (ADR-013, corrected after PR #8 review) is sound:
permitted overlap is provably confined to raft-region ∩ raft-region. But that
guarantee is only as good as the masks — it assumes every detected "raft" band
(`full − body`) contains only disposable support material. The area-knee
detector cannot promise that: it fires on *any* mesh whose shadow shrinks ≥ 5%
within `support_cut_cap_mm` (5 mm) of its base and then plateaus. Measured on
the corpus, 13 **unsupported** meshes (wings resting on a tip, merged cloth)
get a bogus cut — all cap-adjacent (4.5–5.0 mm), the no-plateau taper failure
mode. A hypothetical integral plinth or chamfered base would produce a shallow
bogus cut the same way. Any such piece silently opts its own model geometry
into raft fusion: a neighbor's raft may legally overlap — and cure into —
model material.

## Decision

Add a single **acceptance gate** after knee detection, inside
`detect_base_cut`: the geometry crossing a plane just above the proposed cut
must look like a support forest (many small pillar cross-sections), not a model
wall (one large connected ring). If it does not, return 0.0 — the piece packs
on its full shadow and never participates in raft merging.

Concretely, after a knee at depth `cut` is found:

1. Select triangles that **straddle** the plane `z0 + cut + BAND_MM`
   (`min Z < plane < max Z`). Straddle selection is non-manifold-safe — no
   topology, no `mesh.section`.
2. Rasterize them at `DETECT_RES_MM` onto the detection canvas (the `tri_px` /
   `shape` already computed there). This yields the band mask — cross-section
   *outlines*, since meshes are hollow shells: pillar rings for supports, a
   long wall ring for solid geometry.
3. Compute connected components (`cv2.connectedComponentsWithStats`,
   8-connectivity). **Dominance** = largest component area / total band area.
4. Accept the cut iff the band is non-empty and
   `dominance <= RAFT_BAND_DOMINANCE_MAX`; otherwise return 0.0.

New module constant in `footprint.py` (detector-intrinsic, not config — keeps
the cache addressable by STL hash alone, ADR-009):

```python
RAFT_BAND_DOMINANCE_MAX = 0.35  # accept cut iff largest band component <= 35% of band
```

Bump `DETECTOR_VERSION` 1 → 2 so cached body masks regenerate (the CLI already
re-extracts on version mismatch).

## Calibration evidence (why 0.35)

Probe over every corpus STL: run the knee detector, then measure the band at
`cut + BAND_MM`. Population labels corrected for 7 publisher supported-exports
missing the "supported" suffix (identified by `STL_` naming and confirmed by
their support-forest signature):

| population | n with cut | dominance | components | knee depth |
| --- | --- | --- | --- | --- |
| supported (true rafts) | 101 | 0.018 – **0.208** | 22 – 226 | 0.25 – 1.25 mm |
| unsupported (bogus cuts) | 13 | **0.556** – 1.000 | 1 – 13 | 4.5 – 5.0 mm |

A 2.7× gap separates the populations; 0.35 sits between them with margin both
ways. On this corpus the gate rejects all 13 bogus cuts and accepts all 101
real rafts.

Band *area fraction* (tried first) does **not** separate: hollow shells make a
wing's wall ring as sparse by area (0.02–0.08 of footprint) as a support
forest. Dominance measures connectedness, which is the actual physical
difference, and is scale-free — no dependence on model size or resolution.

## What deliberately does not change

- The two-mask collision core, `rotate_pair`, cache schema v2, CLI/improve
  threading: untouched. The fix is classification-only.
- `support_cut_cap_mm` stays 5 mm and stays the only depth knob. It is the
  *search* range; the gate replaces the need for a separate accept-window
  (`raft_window_mm` was considered and dropped — the deep-knee failures it
  would catch are exactly the high-dominance ones, and it cannot catch a
  shallow plinth, which dominance does).
- `MIN_REDUCTION` stays 5%: with the gate carrying safety, small real gains
  remain worth taking (the user's corpus is predominantly pre-supported;
  integral plinths on-plate are rare in resin due to elephant foot).

## Failure directions

- **False reject (raft treated as no-raft):** costs density only, never
  correctness. Expected for tiny pieces with very few pillars (a 3–4-pillar
  band can exceed 0.35 dominance; corpus minimum is 22 components). Acceptable.
- **False accept (model band treated as raft):** requires model geometry whose
  cross-section outline at the cut fragments into many small pieces, none over
  35% — e.g. a dense field of separate thin spikes growing straight from the
  plate. Physically raft-like enough that fusion damage is marginal; noted,
  not defended against in v1.

## Testing

Parametrized, atomic cases (global testing preference):

- **Synthetic support forest** (raft slab + ≥ 8 thin pillars + body slab
  above): knee found, gate accepts, `body_mask` excludes the slab.
- **Synthetic taper** (hollow pyramid shell, no plateau): knee lands at cap,
  gate rejects (dominance ≈ 1), cut = 0.
- **Synthetic plinth** (wide box base + narrow box body, shallow knee): wall
  ring is one component, gate rejects, cut = 0.
- **Empty band** (nothing straddles the plane above the knee): gate rejects.
- Hand-computed expectations verified by probe script before writing asserts
  (SDD lesson: plan-literal errors).
- Existing `extract_footprints` / packer / cache tests unchanged and green;
  `DETECTOR_VERSION` bump asserted in the cache-invalidation test.
- Regression evidence: re-run the corpus probe post-implementation; expect
  101 accepts / 13 rejects exactly. The probe script is committed as
  `tools/probe_raft_gate.py` so recalibration on future corpora is
  reproducible (it is not part of the package).

## Open questions

None blocking. If future corpora surface true rafts above 0.35 dominance
(sparse-pillar minis), revisit with the same probe rather than guessing.

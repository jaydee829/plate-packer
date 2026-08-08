# Raft-Fusion Packing — Gated Body-Over-Raft Nesting (ADR-015)

**Date:** 2026-08-08
**Builds on:** ADR-013 (two-mask collision), ADR-014 (raft-signature gate).
**Status:** user-approved design ("YAGNI zone"); per-pixel Z-clearance and full
2.5D Z-banding explicitly tabled.

## Problem

Strict two-mask packing (only raft∩raft overlap) yields ~no density gain:
rafts hug model outlines (0 flare), so raft-only regions are interior
concavities unreachable without a body crossing the other piece's full
outline. The valuable move — nesting a body over a neighbor's raft — was
forbidden because a 2D shadow cannot prove the body column has no low-Z
material.

## Decision (accepted-risk YAGNI version)

Allow body-over-raft overlap, but **only between gate-accepted pieces**
("fused" pieces, ADR-014). Risk analysis of what physically occupies a fused
piece's body column below a neighbor's raft top: its own raft and support-
pillar feet — disposable material, same acceptability class as the raft-raft
fusion already permitted. The remaining hazard (model surface dipping below a
neighbor's ~1 mm raft top on a piece that nonetheless carries a gate-accepted
raft) is accepted as rare on Lychee-style pre-supported exports; the per-pixel
min-Z clearance mask is the designated hardening if it ever bites.

Second knob: `support_cut_cap_mm` default 5.0 → **3.0**. Every accepted knee
on the calibration corpus sits at 0.25–1.25 mm; a 3 mm cap costs nothing,
independently kills the deep bogus knees (defense in depth with the gate),
and guarantees fusion can never touch geometry above 3 mm. No
`DETECTOR_VERSION` bump (user decision). Ceiling enforcement at load instead:
the pack CLI treats a cached doc as stale when `cut_z_mm >
support_cut_cap_mm` and re-extracts under the active cap, so legacy caches
from the 5 mm era cannot defeat the guarantee (a cached cut ≤ cap stays
valid regardless of the cap it was found under).

## Collision semantics

Per piece, **fused ⇔ body ≠ full** (a gate-rejected or raftless piece has
body == full). Derived inside the packer by mask comparison — no new
parameters, no CLI/improve threading changes.

Per plate, three occupancy grids:

- `hard_body` — union of placed **fused bodies**
- `full_nf` — union of placed **non-fused fulls**
- `full_all` — union of **all** placed fulls

Legality (all candidates additionally keep full-within-border):

- **Fused candidate:** `body` clears `hard_body` AND `full` clears `full_nf`.
  Its body may sit over fused rafts (the nesting win); its raft may sit under
  fused bodies and over fused rafts. Body-body contact between fused pieces
  stays forbidden with spacing intact (bodies are spacing-dilated).
- **Non-fused candidate:** `full` clears `full_all`. Strict full-shadow
  collision in both directions — it neither sits on rafts nor gets nested on
  (its low-Z material is unknown).

Permitted-overlap matrix (F = fused, N = non-fused):

| candidate \ placed | F body | F raft | N full |
| --- | --- | --- | --- |
| **F body** | ✗ | ✓ (nest) | ✗ |
| **F raft** | ✓ | ✓ (fuse) | ✗ |
| **N full** | ✗ | ✗ | ✗ |

Coarse-resolution note (improve loop): block-max downsampling can collapse a
thin raft ring so coarse body == coarse full → the piece is treated as
non-fused at coarse res. Strictly conservative, so "coarse-legal ⇒ fine-legal"
still holds; the fine repack recovers the fusion freedom. The invariant also
needs the converse direction, which holds by determinism: fine body == fine
full block-maxes to equal coarse masks, so coarse-fused ⇒ fine-fused — a
nested coarse layout can never scale to a fine layout where the nesting
partner turns out non-fused.

Contact scoring attracts to `hard_body | full_nf` plus the border (rafts are
free-fire, not attractors).

## What does not change

- Plate boundary: the full shadow must stay within plate/margins (print
  contract — rafts stay on-plate).
- Export, verify (merged-shadow self-check ORs fulls; overlap-absorbing),
  cache schema, CLI, improve signatures, the gate itself.
- Existing strict-mode expectations for mixed fused/non-fused pairs: the two
  ADR-013 regression tests (raft-may-not-overlap-body in both directions with
  a solid piece) keep passing because a solid piece is non-fused by
  derivation.

## Testing

- New packer tests (synthetic masks, parametrized/atomic): fused-fused
  nesting shares a plate where strict mode would spill; fused-fused body-body
  still separates; existing mixed-pair strictness tests unchanged and green.
- Config default test updated to 3.0.
- Full suite green; corpus baseline (the point of this change) run by the
  user post-merge.

## Follow-ups (tabled by explicit user decision)

- Per-pixel min-Z clearance mask ("underpass-safe") — converts the accepted
  risk into a proof; designated v2 hardening.
- Full Z-banded (2.5D) collision — see ADR-013 limitation note.

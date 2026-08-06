# Work Log (Issues)

This file tracks work history and ticket references.

## Templates

### YYYY-MM-DD - TICKET-ID: Brief Description
- **Status**: Completed / In Progress / Blocked
- **Description**: 1-2 line summary
- **URL**: Link to ticket or PR
- **Notes**: Any important context

## Log

### 2026-08-06 - ROTRES: shape-aware angles + coarse-to-fine beam search (ADR-012)
- **Status**: Completed (branch feat/rotation-resolution, awaiting merge)
- **Description**: 7-task SDD of ADR-012. New `angles.py::angle_candidates` (hull-edge-parallel angles, circle→[0.0], 0.0 pinned through cap, analytic-hull-AABB compactness sort). `improve()` restructured to coarse-to-fine beam search: coarse ILS on block-max-superset masks → top-K distinct orderings (`_update_beam`) → fine-pack survivors (`_prerotate_multi_res`) → return best FINE; `ImproveResult.beam` observability. `seed_order` difficulty ordering (area×elongation); `contact_map` `edge_weight`; 7 config knobs; CLI shape-aware fit-check + greedy path + `--coarse-res`/`--beam`. 263 tests.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: Final whole-branch review (opus) verified the hard invariants hold: never-worsens (difficulty-seed's fine pack always in the best-fine candidate set), angle-set consistency across fit-check/greedy/improve, no `res` shadowing, conservative-coverage e2e, determinism per seed for fixed coarse-eval count. One Important fixed on-branch: coarse-search crashed on a fine-fitting piece when `edge_margin_mm>0` (block-max grows margin+piece) → fall back to fine resolution for the coarse phase (see bugs.md, commit c5554c2). Four minors shipped (deferred): no deterministic never-worsens safety-net test; seed fine-packed twice when also a beam survivor; `angle_candidates` recomputed on the budget=0 path; `cfg.rotations` now legacy/unused in the pack flow (spec §6 retains it). **NEXT: user runs the real 30-piece benchmark via `!` terminal** to validate the §8 success criterion (≤4 plates / fitness ≥0.4785 below the single-res 8-rot cost) — could not run here (machine kills heavy compute).

### 2026-08-06 - BENCH+ROT: benchmark findings + rotation/resolution milestone spec'd
- **Status**: Spec complete (branch feat/rotation-resolution), plan+SDD next
- **Description**: Benchmarked ADR-011 on the 30-piece set (no STL export, direct improve() probes). Found: (a) contact-scored greedy REGRESSED to 5 plates vs bottom-left's 4 at 8 rotations; (b) at 16 rotations contact greedy recovers to 4 plates / fitness 0.4785, beating bottom-left (0.4765); (c) each repack costs ~105-196s so the ILS is starved (5 evals → 0 improvements). Concluded: raise rotations but make them affordable (coarse-res search) and targeted (shape-aware angles). Spec: ADR-012 / docs/superpowers/specs/2026-08-06-rotation-resolution-design.md.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: The 60-min background pack could not run — this machine kills detached heavy compute (confirmed; see [[background-heavy-compute-killed]] memory). Ran foreground probes ≤10min instead. plates/ (original 4-plate greedy output) preserved for eyeball. Full-length runs must go via the user's own terminal (`!` prefix).

### 2026-08-05 - IMPROVE: contact-scored placement + targeted-move ILS (ADR-011)
- **Status**: Completed (branch feat/packing-improvement, awaiting merge)
- **Description**: Full cycle: deep-research survey (docs/research/) → spec → 6-task SDD. Contact-scoring kernel (ring + FFT contact map), scored default chooser, `pack()` prerotated/order/validate params, Falkenauer fitness, targeted/random ILS moves, `improve()` with budget+stall stops, config knobs + CLI `--budget`/`--seed` + report line. 206 tests.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: Local fable whole-branch review READY TO MERGE (0 Critical/Important). GitHub PR #5 review then caught two real items, both fixed on-branch: an Important claim-accuracy bug (`improve()` "deterministic per seed" overclaimed the wall-clock-budget path; scoped to fixed eval counts + reproducible-mode note + `_FakeClock` tests) and a contract gap (`pack(validate=False)` raised a raw `TypeError` instead of the documented `ValueError`; guarded). Two plan-literal errors caught by implementers during SDD (corner-contact arithmetic; `res` shadowing in cli.py) — both independently verified. Next: re-run the 30-piece benchmark with a high budget (so the deterministic stall stop decides it) to measure occupancy gain vs the 54-62% baseline.

### 2026-08-01 - SHAKEDOWN: first real-world pack (30 Tome of Demons pieces)
- **Status**: Completed (branch fix/verify-oom, awaiting merge)
- **Description**: Packed 8 Armaros parts + Decataur P1 + Tamareth P2 + Vulduk P2 + Kabeiroth P2 + Muzulk P2/P3 + Vanguard P1 (all supported, bases included). 30 pieces → 4 plates at 54–62% occupancy, all plates verify ok, mixed 45°-family rotations used. Timing on user's machine: extraction 204s (29 meshes), pack+export+verify+report 477s.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: Found + fixed verify-stage OOM (see bugs.md). Plates in ./plates/ for user eyeball; selection list in pack_selection.txt.

### 2026-08-01 - EXPORT: v1 loop complete — pack CLI + exact transforms + runtime self-check
- **Status**: Completed (merged, PR #3)
- **Description**: SDD execution of the export milestone: `config.py` (TOML, Photon Mono X 6K defaults), `prepare_mask` returns origin (ADR-010 spacing/2 dilation), `rotate_mask` returns its px→px affine incl. crop, `export.py` (placement_transform 4×4 composition, merged plate STLs, verify_plate subset self-check), `plate-packer pack` command. 141 tests.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: Final review (fable): coordinate chain verified analytically + adversarial probe at 6 angles — READY TO MERGE. Known self-check blind spots documented in key_facts.md. `--copies` will break transforms-per-piece indexing (see key_facts TODO).

### 2026-08-01 - FOOTPRINT-IO: Content-addressed cache + CLI (PR #1)
- **Status**: Completed (PR open, CI green)
- **Description**: Implemented ADR-009 via subagent-driven development: package extraction (undilated), footprint_io contract docs (schema v1, 0.05mm/px, atomic writes), dilate-on-load, `plate-packer footprints` CLI. 53 tests.
- **URL**: https://github.com/jaydee829/plate-packer/pull/1
- **Notes**: Two instructive bugs caught by review/CI: cached-array aliasing on the no-op path, and typer's single-command collapse (see bugs.md). Export-milestone TODO recorded in key_facts.md (prepare_mask origin offset).

### 2026-08-01 - PACKER: Greedy packing engine (v1 core)
- **Status**: Completed
- **Description**: TDD-built src/plate_packer/packer.py — FFT legality map, bottom-left heuristic (pluggable), lossless right-angle + conservative arbitrary-angle mask rotation, largest-first greedy with spillover, pre-pack validation. Repo flipped to src/ package layout (hatchling).
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: End-to-end on real Archvillain footprints: 8 pieces -> 1 plate (200x130mm), 56% occupied, pack=14s at 8 rotations. Rotation must never LOSE pixels (false free space) — right angles use np.rot90, arbitrary angles use linear-interp + any-touched threshold. Next: export (transform + merged STL) and the merged-shadow self-check.

### 2026-08-01 - SETUP: Tooling, prototype, and GitHub publish
- **Status**: Completed
- **Description**: uv-managed pyproject + ruff + pytest; footprint-extraction prototype with tests (2 rasterization bugs found/fixed, see bugs.md); published public repo with MIT license and GitHub Actions CI.
- **URL**: https://github.com/jaydee829/plate-packer
- **Notes**: CI matrix is Python 3.11/3.14 with opencv-python-headless (avoids libGL on Linux runners). Awaiting real STLs in example_stls/ (gitignored — copyrighted content).

### 2026-08-01 - INIT: Project bootstrap
- **Status**: Completed
- **Description**: Created CLAUDE.md from the seed design doc and initialized the project memory system (docs/project_notes/ + memory protocols in CLAUDE.md, GEMINI.md, AGENTS.md).
- **URL**: N/A
- **Notes**: Repo is pre-code. Next planned step per PLATEPACKER_SEED.md: prototype footprint extraction against real supported STLs in `example_stls/` (currently empty — needs files).

## Usage Tips

- Log completed work with a ticket/PR id, date, and link so history stays traceable.
- Keep descriptions to 1-2 lines; put longer context in the **Notes** field.
- Archive entries older than ~3 months by manual cleanup; this log is not automated.

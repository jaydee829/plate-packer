# Key Project Facts

This file tracks important project configuration, constants, and environment details.

## Project Overview
- **Project Name**: Plate Packer (Resin Plate Packer)
- **Description**: Packs pre-supported resin (MSLA) models onto build plates using true irregular 2D footprints with free rotation, minimizing plate count. Design doc: `PLATEPACKER_SEED.md`.

## Local Development
- **OS / Runtime**: Windows 11, Python 3.11+
- **Primary Workflow**: CLI tool (planned): folder of supported STLs + printer config in → `plate_01.stl`, `plate_02.stl`, … + layout report out
- **Setup**: `uv sync` from repo root (src-layout package, hatchling). Test STL inputs in `example_stls/` (junction → `C:\dev\stl_curator\example_stls`).

## Tooling & Distribution (decided 2026-08-01)
- **Repo**: https://github.com/jaydee829/plate-packer (public, MIT license)
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — ruff check + format-check + pytest on Python 3.11 and 3.14, on push to main and PRs.
- **Never commit `example_stls/`** — pre-supported Patreon/Kickstarter models are copyrighted third-party content (gitignored).
- **Package/env management**: `uv` (pyproject.toml-based). The current pip-created `.venv` is interim scaffolding — migrate to `uv` when the package is formalized.
- **Lint/format**: `ruff` (linter + formatter), enforced in CI.

## Technology Stack
- **Core Libraries**: `trimesh` (mesh IO, transforms, 3MF/scene export), `numpy`, `scipy` (`signal.fftconvolve`), `opencv-python` (`cv2.fillPoly` rasterization)
- **Later / Optional**: `cupy` (GPU convolution), `typer` or `argparse` (CLI)
- **Testing**: `pytest`. Priority targets: rasterization correctness on known shapes, pixel↔mm↔pixel round-trip, spillover behavior, merged-shadow end-to-end self-check.

## stl_curator Interface (normative — ADR-009)
- **Contract location**: `C:\dev\stl_curator\docs\superpowers\specs\2026-08-01-stl-curator-m1-design.md` §4. Changes require updating both projects and that section.
- **Footprint docs**: `footprints/<first-2-hex>/<sha256-hex>.json`, shared `footprints_dir` config. One STL → one doc → many footprints (z-slices). We own/version the JSON internals; curator only records existence.
- **Docs hold intrinsic data only** (undilated masks, canonical res); spacing/res applied at packer load time.

## Deferred Hardening (do when ADR-008 escalates extraction to multiprocessing)
- `footprint_io.save_doc` uses a deterministic tmp filename (`<sha>.json.tmp`). Two workers racing on the same input could collide mid-write. Switch to a process-unique tmp name (tempfile.mkstemp-style, same directory) BEFORE parallelizing extraction. (PR #1 review, 2026-08-01 — not a live issue while extraction is serial.)

## Export & Self-Check (v1, 2026-08-01)
- **CLI**: `plate-packer pack PATHS... [--config] [--out] [--footprints-dir] [--no-verify] [--force]` → `plate_NN.stl` + `report.txt`. Default printer config: Anycubic Photon Mono X 6K (197×122 mm, Z 245) — all knobs in `config.toml`.
- **Self-check sensitivity floor**: verify_plate asserts shadow ⊆ *dilated* occupancy, so misplacements under spacing/2 mm pass silently; an entirely missing piece also passes (empty shadow). "verify ok" ≠ sub-millimeter placement proof.
- **`--copies` landmine (future)**: cli/export index `transforms[placement.piece]`, valid only while pack() emits exactly one placement per piece. Copies require per-placement transforms.
- **Report rotation values** are true world rotation from the composed transform; nominal `Placement.angle` 90° prints as 270° (rot90/warpAffine rotate clockwise in the row-0=min-Y frame). Intentional — do not "fix".

## Improvement Search (v1.1, 2026-08-05, ADR-011)
- **Knobs** (`[packing]`): `improve_budget_s = 2700`, `min_improvement = 0.005`, `patience = 30`, `seed = 0`, `placement = "contact" | "bottom_left"`. CLI: `--budget`, `--seed` override config (no re-validation — negative `--budget` just means plain greedy). Budget 0 = plain greedy (still contact-scored).
- **Fitness** = `mean(fill²)` over plates, fills from dilated-mask px sums / usable px. Reported in the `improvement:` report line; also echoed live per new best.
- **Determinism / reproducible runs**: deterministic per `seed` only for a fixed evaluation count. The stall stop is deterministic; the wall-clock budget is NOT (how many evals fit in `improve_budget_s` depends on machine speed). At the default 2700s budget, realistic piece counts hit the timer, not the stall, so the same seed can give different layouts on different hardware. **For a reproducible run (e.g. before/after benchmarking), set `improve_budget_s` high** so the `patience` stall determines the stop. (PR #5 review finding, 2026-08-06.)
- `pack(validate=False)` on an unfittable piece raises the documented `ValueError` (guarded against the `None`-unpack; PR #5 review). Remaining perf follow-ups (ride): skip the contact FFT when `legal.any()` is False (~1.3-1.5x waste on failed probes); add a `rings` param to `pack()` to stop per-repack ring rebuilds (<1% cost today).
- **rotations>1 note**: plate fills use unrotated dilated piece_px — identical bias across candidates, ranking unaffected.

## Coarse-to-Fine Search (v1.2, 2026-08-06, ADR-012)
- **Shape-aware angles** (`angles.py`): `angle_candidates(mask, cap=12, min_edge_frac=0.1, safety_grid=0) -> list[float]` — convex-hull edges laid parallel to plate axes (angle `degrees(atan2(dy,dx)) % 180` plus `+90`; the `+atan2` sign matches `rotate_mask` — an earlier mirror-sign bug left generic edges un-aligned, see bugs.md 2026-08-07), circle-like hull → `[0.0]`, `0.0` always present (pinned even when `cap` truncates; `cap<1` raises), sorted by ANALYTIC hull-rotation AABB area (NOT rasterized — warpAffine interpolation would inflate a re-rotated tilted raster and corrupt the order). Resolution-independent; computed once per piece, reused at both resolutions. `safety_grid>0` unions a uniform grid (**config default 16**, function-param default 0).
- **`improve()` restructured** (coarse-to-fine beam): signature dropped `rotations`; gained `working_res_mm=0.1, coarse_res_mm=0.4, beam=5, angle_cap=12, min_edge_frac=0.1, safety_grid=0, edge_contact_weight=1.0, ordering="difficulty"`. Pipeline: coarse ILS at `factor=round(coarse_res_mm/working_res_mm)` (block-max superset masks, ~16× cheaper FFTs) → beam of top-K distinct orderings (`_update_beam`) → fine-pack survivors → return best FINE. `ImproveResult` gained `beam: list = field(default_factory=list)` = `(coarse_fitness, fine_fitness, n_plates)` best-first. `evaluations` counts COARSE evals.
- **`budget_s` bounds only the COARSE loop** — the fine-refinement runs up to `beam+1` unbudgeted full fine packs after (seed + each beam survivor). Total wall-clock ≈ `budget_s + (beam+1) × fine-pack cost`; a single fine pack is minutes on the real set. Documented in the `improve()` docstring + CLI `--budget` help (PR #6 review).
- **`contact_map` rounds to 2 decimals** (`np.round(raw, 2)`, not `np.rint`) so fractional `edge_contact_weight` survives; FFT noise <3e-4 on full plates so 2-decimal rounding is ~18×-safe for stable ties (PR #6 review; earlier `np.rint` crushed e.g. 0.1 → 0).
- **Never-worsens is by construction**: the fine candidate set is `{difficulty-seed order} ∪ {beam orderings}`; the seed's fine pack is ALWAYS in the `max`, so `fitness_final >= fitness_initial` holds even when the seed is evicted from the beam. `fitness_initial` = fine fitness of the difficulty-seed order.
- **Dual fine candidate per ordering (2026-08-07 fix)**: each ordering yields TWO fine candidates — a fresh fine re-pack AND the coarse layout scaled to fine (`_scale_placements`, anchors × factor) — and the better fitness wins. The re-pack alone can LOSE a plate to greedy myopia (coarse fit 4 plates but the fine re-pack spilled to 5; see bugs.md 2026-08-07): the fine contact-greedy is less regularized than the coarse grid. The scaled-coarse candidate is collision-free/in-bounds by the block-max superset property + a coarse-plate padding guard (partial edge cells marked occupied so scaled anchors stay in-bounds), so it preserves the coarse plate count. Scaled-coarse has ≤ factor-1 px (~0.4mm) grid slop, so it's looser — more rotations (`safety_grid`) let the fine re-pack itself hit the low plate count *densely* and beat it.
- **Coarse-legal ⇒ fine-legal**: coarse masks are block-max downsamples of the FINE rotated masks (`_prerotate_multi_res` rotates-then-downsamples → supersets). Returned placements come from the fine pack's own legal map, so output is fine-legal by construction.
- **Coarse-plate fallback (edge_margin landmine)**: block-max grows the margin frame, so with `edge_margin_mm>0` a fine-fitting near-plate-spanning piece can fail the coarse empty plate. `improve()` guards: if any piece can't seat the coarse empty plate, it falls back to fine resolution for the coarse phase (no crash, no speedup). No-op on the default `edge_margin_mm=0.0`. See bugs.md 2026-08-06.
- **`seed_order(pieces, ordering)`** (`packer.py`): `"difficulty"` = area×elongation desc (elongation = long/short of `cv2.minAreaRect`, rotation-stable); `"area"` = legacy largest-first. **`contact_map(plate, ring, edge_weight=1.0)`**: border frame padded with `constant_values=edge_weight` (occupancy stays weight 1); threaded through both `_best_spot` calls in `pack(..., edge_weight=1.0)` and both coarse+fine `pack` calls in `improve`.
- **New config knobs** (`[packing]`): `coarse_res_mm=0.4` (≥ working_res, integer multiple), `beam=5` (≥1), `angle_cap=12` (≥1), `min_edge_frac=0.1` (0<x≤1), `safety_grid=16` (≥0), `edge_contact_weight=1.0` (≥0), `ordering="difficulty"|"area"`. CLI `pack` gained `--coarse-res`/`--beam`. **`rotations` is now legacy** — retained/validated in config but unused in the pack flow (spec §6 keeps it as a safety_grid density reference).
- **`safety_grid` config default is 16, NOT 0 (2026-08-07)** — shape-aware angles alone (mean 4.7/piece on the 30-piece set) starve contact scoring (5 plates); the uniform backstop is essential (see ADR-012 retrospective). Note the config default (16) differs from the `angle_candidates`/`improve()` *function-parameter* default (0), which unit tests still use; the CLI passes `cfg.safety_grid`.
- **Deferred (ADR-012 §7)**: adaptive/successive-halving beam refinement, cluster/pairwise nesting, delta/incremental evaluation, Z-banding.

## Real-World Benchmarks (user's machine, 30 Tome of Demons pieces)
- **2026-08-01** (0.1mm/px, 8 rot): extraction/caching 204s; pack+export+verify+report 477s; → 4 plates at 54–62% occupancy, ~200–270MB merged STL per plate.
- **2026-08-06** greedy comparison (budget 0, 0.1mm/px), single repack cost + result:
  - bottom-left, 8 rot: **4 plates**, fitness 0.4765, ~105s/eval, occ [62.3, 60.9, 53.8, 56.3%]
  - contact, 8 rot: **5 plates**, fitness 0.3685, ~196s/eval, occ [64.9, 60.4, 54.2, 49.4, 4.4%] — contact REGRESSED at 8 rot
  - contact, 16 rot: **4 plates**, fitness 0.4785, mean contact 77px, ~381s/eval — recovers, beats bottom-left
- **Search is eval-cost-bound**: ILS at ~105-196s/eval → 60-min budget ≈ 19-34 evals; 5 bottom-left evals gave 0 improvements. Per-eval cost, not budget, is the bottleneck → motivates coarse-to-fine (ADR-012).
- **2026-08-07** coarse-to-fine (ADR-012), contact, `coarse_res=0.4`, `beam=5`:
  - `safety_grid=0` (shape-aware only): **5 plates**, fitness **0.3497**, initial==final (search added nothing at fine) — fine re-pack spilled 2 pieces the coarse pass fit; ~4.7 angles/piece. FAILED the bar.
  - `safety_grid=16` + realize-coarse fix: **4 plates**, fitness **0.4779 → 0.4801** (34 evals, 2 improvements), occ [59.3, 60.1, 65.4, 48.6%], all verify ok. **Beats bottom-left (0.4765) and 16-rot contact (0.4785) → §8 bar cleared.** Search stall-stopped at 34 evals (~minutes), so `safety_grid=16`'s pricier evals are still affordable. Drove the `safety_grid` default 0→16.
- **Ops**: this machine kills detached heavy compute (background pack killed mid-run). Foreground probes must be ≤10min (600s Bash cap); full-length runs go through the user's terminal via `!`. See [[background-heavy-compute-killed]].

## Domain Constants & Conventions
- **Raster resolution**: ~0.05–0.1 mm/px, config knob; tune empirically (0.1 mm/px on a ~200×130 mm plate → ~2000×1300 masks).
- **Rotation steps**: ~36–72, config knob.
- **Coordinate convention (landmine)**: packer works in pixel space; slicers expect millimeters with origin at plate center. Export = pixel → mm → recenter about plate midpoint. Rotation about Z only; Z never touched.
- **Margin handling (ADR-009)**: cached footprints are undilated at canonical 0.05 mm/px; spacing dilation + conservative downsample happen at load time (`plate_packer.loading.prepare_mask`); packer core is margin-unaware.
- **`spacing_mm` default is 1.0mm (2026-08-07, was 2.0mm; user-approved)** — the true inter-piece gap (ADR-010); 1mm packs denser on the benchmark set while keeping the 4-plate floor. Tunable per support geometry; applied at load time so re-runs at other values reuse the undilated cache.
- **Config surface**: plate dims (mm) + optional unusable-region mask, build volume Z, raster resolution, min spacing, rotation steps, placement heuristic, improvement time budget — single dataclass or TOML.

## Usage Tips
- Organize facts by category; prefer bullet lists over tables for easy editing.
- Prefer documented facts here over assumptions when looking up config.

## SECURITY — What NOT to Store

This file is committed to version control. **Never** put secrets here:

- ❌ Passwords, API keys, tokens, private keys, connection strings with embedded credentials
- ❌ `.env` file contents, OAuth client secrets, signing keys, certificates
- ❌ Anything you would not paste into a public PR

Instead, store:

- ✅ The **name/location** of a secret and how to obtain it
- ✅ Non-secret config: ports, hostnames, public URLs, project IDs

"""Iterated local search over the greedy insertion order (spec 2026-08-05).

Fitness is Falkenauer's grouping objective: mean squared plate fill. With
total piece area fixed, concentrating area on fewer/fuller plates always
scores higher, so plate count needs no separate objective term.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from plate_packer.angles import angle_candidates
from plate_packer.loading import conservative_downsample
from plate_packer.packer import Placement, _fits, contact_first, pack, rotate_mask, seed_order

SHAKE_AFTER = 20
SHAKE_MOVES = 4
SAMPLE = 5
WINDOW = 3


@dataclass(frozen=True)
class ImproveResult:
    placements: list[Placement]
    evaluations: int  # coarse evaluations (the search effort)
    improvements: int
    fitness_initial: float  # fine fitness of the difficulty-seed order
    fitness_final: float  # best fine fitness among {seed} U beam
    beam: list = field(default_factory=list)  # (coarse_fit, fine_fit, n_plates) per survivor


def plate_fills(placements, piece_px, usable_px) -> list[float]:
    """Per-plate fill fractions from dilated-mask pixel sums (placements never
    overlap, so fills are additive)."""
    n = max(p.plate for p in placements) + 1
    totals = [0] * n
    for p in placements:
        totals[p.plate] += piece_px[p.piece]
    return [t / usable_px for t in totals]


def falkenauer(fills) -> float:
    """Mean squared fill -- maximize."""
    return float(sum(f * f for f in fills) / len(fills))


def _prerotate_multi_res(pieces, piece_angles, factor):
    """Per-piece {angle: mask} at fine resolution and at a block-max-downsampled
    coarse resolution. Coarse masks are supersets of fine (coarse-legal =>
    fine-legal)."""
    fine, coarse = [], []
    for piece, angles in zip(pieces, piece_angles, strict=True):
        fvar = {a: rotate_mask(piece, a)[0] for a in angles}
        cvar = {a: conservative_downsample(m, factor) for a, m in fvar.items()}
        fine.append(fvar)
        coarse.append(cvar)
    return fine, coarse


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


def _scale_placements(placements, scale):
    """Scale a coarse layout's anchors to fine resolution. Coarse masks are
    block-max supersets of the fine masks, so the fine masks at the scaled
    anchors are collision-free and in-bounds -- this preserves the coarse plate
    count, which a fresh fine re-pack can lose to greedy myopia (the coarse grid
    regularizes placement; the fine contact-greedy chases local maxima and can
    spill an extra plate)."""
    return [
        Placement(p.piece, p.plate, p.row * scale, p.col * scale, p.angle, p.contact)
        for p in placements
    ]


def _update_beam(beam, order, fitness, k):
    """Return a new beam of up to k (fitness, order) pairs, best fitness first,
    orderings distinct, ties broken by ordering. Input beam is not mutated."""
    best = {tuple(o): f for f, o in beam}
    key = tuple(order)
    if key not in best or fitness > best[key]:
        best[key] = fitness
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(f, list(o)) for o, f in ranked[:k]]


def _reinsert(order, i_pos, j_pos):
    order = list(order)
    piece = order.pop(i_pos)
    order.insert(j_pos, piece)
    return order


def _move_random_swap(order, placements, fills, rng):
    if len(order) < 2:
        return list(order)
    i = int(rng.integers(0, len(order)))
    j = int((i + 1 + rng.integers(0, len(order) - 1)) % len(order))
    order = list(order)
    order[i], order[j] = order[j], order[i]
    return order


def _move_random_reinsert(order, placements, fills, rng):
    if len(order) < 2:
        return list(order)
    i = int(rng.integers(0, len(order)))
    j = int(rng.integers(0, len(order)))
    return _reinsert(order, i, j)


def _move_window_shuffle(order, placements, fills, rng):
    if len(order) < WINDOW:
        return _move_random_swap(order, placements, fills, rng)
    start = int(rng.integers(0, len(order) - WINDOW + 1))
    order = list(order)
    window = [order[start + k] for k in rng.permutation(WINDOW)]
    order[start : start + WINDOW] = window
    return order


def _move_targeted_reinsert(order, placements, fills, rng):
    """Random piece from the min-fill plate -> random earlier order position."""
    if len(fills) < 2:
        return _move_random_reinsert(order, placements, fills, rng)
    donor = int(np.argmin(fills))
    candidates = [p.piece for p in placements if p.plate == donor]
    piece = int(rng.choice(candidates))
    i = order.index(piece)
    if i == 0:
        return _move_random_reinsert(order, placements, fills, rng)
    j = int(rng.integers(0, i))
    return _reinsert(order, i, j)


def _move_targeted_swap(order, placements, fills, rng):
    """Lowest-contact piece of a random sample <-> random other position."""
    if len(order) < 2:
        return list(order)
    k = min(SAMPLE, len(placements))
    sample = rng.choice(len(placements), size=k, replace=False)
    worst = min((placements[int(s)] for s in sample), key=lambda p: p.contact).piece
    i = order.index(worst)
    j = int((i + 1 + rng.integers(0, len(order) - 1)) % len(order))
    order = list(order)
    order[i], order[j] = order[j], order[i]
    return order


_MOVES = [
    (0.45, _move_targeted_reinsert),
    (0.25, _move_targeted_swap),
    (0.15, _move_random_swap),
    (0.10, _move_random_reinsert),
    (0.05, _move_window_shuffle),
]
_RANDOM_MOVES = [_move_random_swap, _move_random_reinsert, _move_window_shuffle]


def perturb(order, placements, fills, rng):
    """One weighted-random move applied to a copy of order."""
    r = float(rng.random())
    acc = 0.0
    for prob, move in _MOVES:
        acc += prob
        if r < acc:
            return move(order, placements, fills, rng)
    return _MOVES[-1][1](order, placements, fills, rng)


def shake(order, placements, fills, rng):
    """SHAKE_MOVES stacked uniform-random moves -- the ILS escape hatch."""
    for _ in range(SHAKE_MOVES):
        move = _RANDOM_MOVES[int(rng.integers(0, len(_RANDOM_MOVES)))]
        order = move(order, placements, fills, rng)
    return order


def improve(
    pieces,
    plate_shape,
    plate_mask=None,
    choose=None,
    budget_s=2700.0,
    min_improvement=0.005,
    patience=30,
    seed=0,
    working_res_mm=0.1,
    coarse_res_mm=0.4,
    beam=5,
    angle_cap=12,
    min_edge_frac=0.1,
    safety_grid=0,
    edge_contact_weight=1.0,
    ordering="difficulty",
    boundary_pieces=None,
    validate=True,
    on_improve=None,
):
    """Coarse-to-fine iterated local search over the greedy insertion order
    (ADR-012).

    The ILS loop searches at a coarse (downsampled) raster resolution, which
    is cheap enough to explore many orderings; the best `beam` orderings found
    (plus the difficulty-seed baseline) are then repacked once each at full
    (fine) resolution, and the best-fitness fine pack is returned. Because
    coarse masks are conservative supersets of the fine masks (block-max
    downsample), a coarse-legal pack is always fine-legal, so every fine
    repack succeeds.

    Anytime: every coarse evaluation is a complete valid packing, so both stop
    conditions (wall-clock budget, stall: `patience` evaluations without
    `min_improvement` cumulative coarse-fitness gain) return the best found.
    budget_s=0 returns the fine pack of the difficulty-seed order (no search).

    NOTE: `budget_s` bounds only the COARSE search loop. After it stops, the
    fine-refinement stage runs up to `beam + 1` full fine-resolution packs (the
    seed plus each beam survivor), which are NOT budgeted. On large inputs a
    single fine pack can cost minutes, so total wall-clock is roughly
    `budget_s + (beam + 1) x fine-pack cost` -- set `budget_s`/`beam` with that
    in mind for time-boxed runs.

    Determinism: the search is deterministic per seed *for a fixed number of
    coarse evaluations* -- each evaluation consumes the shared RNG, so
    identical draw sequences require identical evaluation counts. The stall
    stop is itself deterministic (it fires at a fixed evaluation count for
    given inputs), but the wall-clock budget is NOT: how many evaluations fit
    in `budget_s` depends on machine speed and load, so a budget-bounded run
    can yield a different layout for the same seed on different hardware. For
    a fully reproducible run set `budget_s` high enough that the stall
    condition always fires first (see key_facts.md).

    fitness_final is always >= fitness_initial: each ordering (the
    difficulty-seed and every beam member) yields two fine candidates -- a fresh
    fine re-pack and the coarse layout scaled to fine resolution -- and the best
    fine fitness over all of them wins. The seed's fine re-pack is always in the
    set, so the anytime guarantee holds. The scaled-coarse candidate preserves
    the coarse plate count, which a fresh fine re-pack can otherwise lose to
    greedy myopia.

    on_improve(evaluations, n_plates, fitness) fires at each new coarse best
    (coarse evaluations/fitness -- the search's internal bookkeeping).

    boundary_pieces (per-piece list of full-shadow prepared masks, parallel to
    `pieces` which are the body masks) switches both the coarse and fine packs
    to two-mask collision (body-body for inter-piece, full-plate for the
    border/margins), same as `pack(..., boundary=)`. When None (default),
    behavior is single-mask and unchanged.
    """
    choose = choose or contact_first
    rng = np.random.default_rng(seed)

    factor = round(coarse_res_mm / working_res_mm)
    if factor < 1:
        raise ValueError(
            f"coarse_res_mm ({coarse_res_mm}) must be >= working_res_mm ({working_res_mm})"
        )
    piece_angles = [
        angle_candidates(p, cap=angle_cap, min_edge_frac=min_edge_frac, safety_grid=safety_grid)
        for p in pieces
    ]
    if boundary_pieces is None:
        fine_prerot, coarse_prerot = _prerotate_multi_res(pieces, piece_angles, factor)
        fine_bound = coarse_bound = None
    else:
        fine_prerot, coarse_prerot, fine_bound, coarse_bound = _prerotate_paired(
            pieces, boundary_pieces, piece_angles, factor
        )

    empty_fine = plate_mask.copy() if plate_mask is not None else np.zeros(plate_shape, np.uint8)
    coarse_plate_mask = conservative_downsample(empty_fine, factor)
    # Coarse cells whose fine footprint runs past the real plate (when
    # plate_shape is not a multiple of factor) are marked occupied, so a coarse
    # placement always scales back in-bounds at fine resolution.
    coarse_plate_mask[plate_shape[0] // factor :, :] = 1
    coarse_plate_mask[:, plate_shape[1] // factor :] = 1
    coarse_shape = coarse_plate_mask.shape
    realize_scale = factor

    fine_piece_px = [int(p.sum()) for p in pieces]
    fine_usable = plate_shape[0] * plate_shape[1] - int(empty_fine.sum())
    coarse_piece_px = [int(coarse_prerot[i][0.0].sum()) for i in range(len(pieces))]
    coarse_usable = coarse_shape[0] * coarse_shape[1] - int(coarse_plate_mask.sum())

    # A piece that fits the fine empty plate can fail the block-max-grown
    # coarse plate (both the piece and the margin frame grow). Rather than
    # raise a misleading "does not fit" (ADR-004) or crash mid-search, fall
    # back to fine resolution for the coarse phase -- still correct, just
    # without the coarse-search speedup. Immune to this on the default
    # edge_margin_mm=0.0 path (all-zero coarse plate seats every piece).
    # When boundary is on, the full shadow is the binding fit constraint (it,
    # not the body, must clear the plate border/margins), so fit checks use it.
    fit_fine = fine_bound if fine_bound is not None else fine_prerot
    fit_coarse = coarse_bound if coarse_bound is not None else coarse_prerot
    coarse_seats_all = all(
        any(_fits(coarse_plate_mask, m) for m in variants.values()) for variants in fit_coarse
    )
    if not coarse_seats_all:
        coarse_prerot = fine_prerot
        coarse_bound = fine_bound
        coarse_plate_mask = empty_fine
        coarse_shape = plate_shape
        coarse_piece_px = fine_piece_px
        coarse_usable = fine_usable
        realize_scale = 1  # coarse phase now runs at fine res; anchors are already fine

    if validate:
        # One up-front validation at fine resolution; every repack then runs
        # with validate=False (coarse-legal => fine-legal covers the rest).
        for i, variants in enumerate(fit_fine):
            if not any(_fits(empty_fine, m) for m in variants.values()):
                raise ValueError(f"piece {i} does not fit an empty plate at any rotation")

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

    start = time.monotonic()
    seed_ord = seed_order(pieces, ordering)

    best_order = seed_ord
    best, best_fit = eval_coarse(best_order)
    evaluations, improvements = 1, 0
    marker, evals_since_marker, fails = best_fit, 0, 0
    incumbent = best_order
    beam_list = _update_beam([], best_order, best_fit, beam)

    while time.monotonic() - start < budget_s:
        fills = plate_fills(best, coarse_piece_px, coarse_usable)
        candidate = perturb(incumbent, best, fills, rng)
        result, fit = eval_coarse(candidate)
        evaluations += 1
        evals_since_marker += 1
        if fit > best_fit:
            best, best_order, best_fit = result, candidate, fit
            incumbent = candidate
            improvements += 1
            fails = 0
            beam_list = _update_beam(beam_list, best_order, best_fit, beam)
            if on_improve is not None:
                on_improve(evaluations, max(p.plate for p in best) + 1, best_fit)
            if best_fit - marker >= min_improvement:
                marker, evals_since_marker = best_fit, 0
        else:
            fails += 1
            if fails >= SHAKE_AFTER:
                incumbent = shake(best_order, best, fills, rng)
                fails = 0
        if evals_since_marker >= patience:
            break

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

    def fine_candidates(order):
        # Two ways to realize an ordering at fine resolution: a fresh fine
        # re-pack (tighter positions when it doesn't spill), and the coarse
        # layout scaled up (can't lose plates to greedy myopia). Return both as
        # (placements, fine_fitness); the caller keeps whichever scores higher.
        repack_res, repack_fit = fine_pack(order)
        coarse_res, _ = eval_coarse(order)
        realized = _scale_placements(coarse_res, realize_scale)
        realized_fit = falkenauer(plate_fills(realized, fine_piece_px, fine_usable))
        return [(repack_res, repack_fit), (realized, realized_fit)]

    seed_cands = fine_candidates(seed_ord)
    fitness_initial = seed_cands[0][1]  # seed fine re-pack: the never-worsens baseline

    all_cands = list(seed_cands)
    beam_out = []
    for coarse_fit, order in beam_list:
        # The seed is pre-seeded into the beam, so reuse its already-computed
        # fine candidates instead of re-running a (minutes-long) fine pack.
        if order == seed_ord:
            cands = seed_cands
        else:
            cands = fine_candidates(order)
            all_cands.extend(cands)
        best_res, best_fit = max(cands, key=lambda c: c[1])
        beam_out.append((coarse_fit, best_fit, max(p.plate for p in best_res) + 1))

    placements, fitness_final = max(all_cands, key=lambda c: c[1])
    beam_out.sort(key=lambda t: -t[1])

    return ImproveResult(
        placements, evaluations, improvements, fitness_initial, fitness_final, beam_out
    )

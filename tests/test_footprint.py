"""Rasterization correctness tests on known shapes (seed doc priority target)."""

import numpy as np
import pytest
import trimesh

from plate_packer.footprint import BAND_MM, detect_base_cut, extract_footprint, extract_footprints

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


@pytest.mark.parametrize(
    ("tris", "expected"),
    [
        pytest.param(
            _tris(_box((20, 20), 0, 2), *[_box((1, 1), 2, 12, c) for c in PILLARS]),
            2.0,
            id="raft-then-pillar-forest",
        ),
    ],
)
def test_detect_base_cut(tris, expected):
    assert detect_base_cut(tris, 0.1, 5.0) == pytest.approx(expected, abs=BAND_MM)


@pytest.mark.parametrize(
    "tris",
    [
        pytest.param(
            _tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12)),
            id="single-pillar-band-is-one-component-gate-rejects",
        ),
        pytest.param(
            _tris(_box((30, 30), 0, 2), _box((10, 10), 2, 10)),
            id="plinth-wall-ring-gate-rejects",
        ),
        pytest.param(
            _tris(*[_box((30 - 2 * k, 30 - 2 * k), k, k + 1) for k in range(8)]),
            id="staircase-taper-knee-at-cap-gate-rejects",
        ),
        pytest.param(
            _tris(_box((20, 20), 0, 1), _box((10, 10), 5, 8)),
            id="floating-body-empty-band-gate-rejects",
        ),
        pytest.param(_tris(_box((1, 1), 0, 12)), id="pillar-only-no-base"),
        pytest.param(_tris(_box((20, 20), 0, 12)), id="wide-solid-no-drop"),
        pytest.param(_tris(_box((20, 20), 0, 8), _box((1, 1), 8, 18)), id="base-past-cap-window"),
        pytest.param(
            _tris(_box((2, 2), 0, 0.5), _box((1, 1), 0.5, 8), _box((20, 20), 8, 10)),
            id="tiny-foot-below-min-base-frac",
        ),
    ],
)
def test_detect_base_cut_rejects(tris):
    assert detect_base_cut(tris, 0.1, 5.0) == 0.0


@pytest.mark.parametrize(
    ("tris", "expected_knee"),
    [
        pytest.param(_tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12)), 2.0, id="single-pillar"),
        pytest.param(_tris(_box((30, 30), 0, 2), _box((10, 10), 2, 10)), 2.0, id="plinth"),
        pytest.param(
            _tris(*[_box((30 - 2 * k, 30 - 2 * k), k, k + 1) for k in range(8)]),
            5.0,
            id="staircase-taper",
        ),
        pytest.param(_tris(_box((20, 20), 0, 1), _box((10, 10), 5, 8)), 1.0, id="floating-body"),
    ],
)
def test_detect_base_cut_ungated_still_finds_knee(tris, expected_knee):
    """gated=False exposes the raw area-knee: proves the gate (not MIN_REDUCTION)
    is what rejects these shapes in test_detect_base_cut above."""
    assert detect_base_cut(tris, 0.1, 5.0, gated=False) == pytest.approx(expected_knee, abs=BAND_MM)


def test_detect_base_cut_zero_cap_returns_zero():
    tris = _tris(_box((20, 20), 0, 2), _box((1, 1), 2, 12))
    assert detect_base_cut(tris, 0.1, 0.0) == 0.0


def test_extract_footprints_full_matches_extract_footprint():
    mesh = trimesh.creation.box(extents=(20, 10, 5))
    ref, _o, _s = extract_footprint(mesh, RES)
    full, _body, _origin, _cut, _stats = extract_footprints(mesh, RES, 5.0)
    assert full.shape == ref.shape
    assert (full == ref).all()


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


def test_extract_footprints_no_cut_gives_identical_body():
    mesh = _box((1, 1), 0, 12)  # pillar only -> cut 0
    full, body, _origin, cut, _stats = extract_footprints(mesh, RES, 5.0)
    assert cut == 0.0
    assert (body == full).all()


def test_extract_footprints_falls_back_when_cut_would_empty_body():
    """Raft-only input (whole model shorter than cut_cap_mm): the band gate
    rejects the knee first (nothing straddles above the model's own top), so
    detect returns 0.0 and body == full. The empty-body fallback inside
    extract_footprints stays as defense-in-depth behind the gate; this pins
    the observable contract: no cut, non-empty body identical to full."""
    mesh = _box((20, 20), 0, 2)  # no body above the raft at all
    full, body, _origin, cut, stats = extract_footprints(mesh, RES, 5.0)
    assert cut == 0.0
    assert stats["cut_mm"] == 0.0
    assert body.any()
    assert (body == full).all()

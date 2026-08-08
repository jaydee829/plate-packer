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


def test_extract_footprints_falls_back_when_cut_would_empty_body():
    """Raft-only input (whole model shorter than cut_cap_mm): detect_base_cut fires
    a cut equal to the model's own top (its footprint-area knee sits at its own
    apex, since nothing exists above it), which would leave zero kept triangles
    and an empty body_mask. extract_footprints must fall back to no cut instead
    of propagating an empty mask (downstream rotate_mask cannot crop one)."""
    mesh = _box((20, 20), 0, 2)  # no body above the raft at all
    full, body, _origin, cut, stats = extract_footprints(mesh, RES, 5.0)
    assert cut == 0.0
    assert stats["cut_mm"] == 0.0
    assert body.any()
    assert (body == full).all()

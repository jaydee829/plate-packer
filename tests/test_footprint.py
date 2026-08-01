"""Rasterization correctness tests on known shapes (seed doc priority target)."""

import numpy as np
import pytest
import trimesh

from plate_packer.footprint import extract_footprint

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

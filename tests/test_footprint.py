"""Rasterization correctness tests on known shapes (seed doc priority target)."""

import sys
from pathlib import Path

import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_footprint import extract_footprint

RES = 0.1


@pytest.mark.parametrize(
    ("extents", "spacing"),
    [
        pytest.param((20, 10, 5), 0.5, id="box-20x10-margin"),
        pytest.param((20, 10, 5), 0.0, id="box-20x10-no-margin"),
        pytest.param((5, 5, 50), 1.0, id="tall-thin-box"),
        pytest.param((100, 3, 2), 0.5, id="long-sliver"),
    ],
)
def test_box_mask_dimensions(extents, spacing):
    """A box's shadow mask must span footprint + 2*spacing in each axis."""
    mesh = trimesh.creation.box(extents=extents)
    mask, _origin, _stats = extract_footprint(mesh, RES, spacing)
    expected_w = round((extents[0] + 2 * spacing) / RES)
    expected_h = round((extents[1] + 2 * spacing) / RES)
    # +-2px tolerance: rounding at both borders plus the ceil in canvas sizing
    assert abs(mask.shape[1] - expected_w) <= 2
    assert abs(mask.shape[0] - expected_h) <= 2


@pytest.mark.parametrize(
    ("extents", "spacing"),
    [
        pytest.param((20, 10, 5), 0.5, id="box-with-margin"),
        pytest.param((20, 10, 5), 0.0, id="box-no-margin"),
    ],
)
def test_box_mask_is_solid(extents, spacing):
    """Overlapping projected triangles must union, not cancel (fillPoly even-odd bug)."""
    mesh = trimesh.creation.box(extents=extents)
    mask, _origin, _stats = extract_footprint(mesh, RES, spacing)
    assert mask.mean() > 0.97


@pytest.mark.parametrize(
    "corrupt",
    [pytest.param(float("nan"), id="nan-vertex"), pytest.param(float("inf"), id="inf-vertex")],
)
def test_nonfinite_triangles_are_dropped(corrupt):
    """NaN/inf vertices (seen in real pre-supported STLs) must not poison the mask."""
    import numpy as np

    mesh = trimesh.creation.box(extents=(20, 10, 5))
    vertices = np.vstack([mesh.vertices, [[corrupt, corrupt, corrupt]] * 3])
    n = len(mesh.vertices)
    faces = np.vstack([mesh.faces, [[n, n + 1, n + 2]]])
    corrupted = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mask, _origin, stats = extract_footprint(corrupted, RES, 0.0)
    assert stats["dropped_nonfinite"] == 1
    assert stats["z_height_mm"] == pytest.approx(5.0)
    assert mask.mean() > 0.97


def test_disjoint_scene_masks_do_not_cancel():
    """Two stacked boxes (same shadow) must still produce a solid union."""
    a = trimesh.creation.box(extents=(10, 10, 2))
    b = trimesh.creation.box(extents=(10, 10, 2))
    b.apply_translation([0, 0, 5])
    mesh = trimesh.util.concatenate([a, b])
    mask, _origin, _stats = extract_footprint(mesh, RES, 0.0)
    assert mask.mean() > 0.97

"""Export: transform composition, plate STL writing, merged-shadow self-check."""

import numpy as np
import pytest
import trimesh

from plate_packer.export import export_plates, placement_transform, verify_plate
from plate_packer.footprint import extract_footprint
from plate_packer.packer import Placement, rotate_mask


def _mask():
    m = np.zeros((20, 30), np.uint8)
    m[2:18, 3:27] = 1
    m[2:6, 3:9] = 0
    return m


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, 270.0, 30.0, 137.5])
@pytest.mark.parametrize(("row", "col"), [(0, 0), (7, 13)])
def test_round_trip_pixel_mm_pixel(angle, row, col):
    """Seed-doc priority: a piece-frame pixel's mm position, pushed through the
    4x4, lands exactly on its predicted plate pixel."""
    mask = _mask()
    _, aff = rotate_mask(mask, angle)
    res, origin, plate = 0.1, (5.0, -3.0), (100.0, 60.0)
    t4 = placement_transform(origin, aff, row, col, res, plate)
    r, c = 2, 3  # a content pixel of the input mask
    p = np.array([origin[0] + c * res, origin[1] + r * res, 0.0, 1.0])
    q = t4 @ p
    got_px = (q[:2] + np.array(plate) / 2) / res  # slicer mm -> plate px (x, y)
    expected = (aff @ np.array([c, r, 1.0])) + np.array([col, row])
    np.testing.assert_allclose(got_px, expected, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 90.0, 45.0, 200.0])
def test_transform_is_proper_rigid(angle):
    _, aff = rotate_mask(_mask(), angle)
    t4 = placement_transform((0.0, 0.0), aff, 0, 0, 0.1, (100.0, 60.0))
    lin = t4[:3, :3]
    np.testing.assert_allclose(lin @ lin.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(lin) == pytest.approx(1.0)
    assert t4[2, 3] == 0.0  # Z untouched


def test_mirror_affine_rejected():
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    with pytest.raises(ValueError, match="proper rotation"):
        placement_transform((0.0, 0.0), mirror, 0, 0, 0.1, (100.0, 60.0))


RES = 0.1
PLATE = (100.0, 60.0)
PLATE_SHAPE = (round(PLATE[1] / RES), round(PLATE[0] / RES))


def _box(w=10.0, d=10.0, h=5.0):
    b = trimesh.creation.box(extents=(w, d, h))
    b.apply_translation((w / 2, d / 2, h / 2))  # min corner at (0,0,0)
    return b


def _identity_aff():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_absolute_coordinates_known_box():
    """A box placed at anchor (0,0), angle 0: pixel (0,0) = mask origin lands at
    the plate corner, i.e. slicer (-50, -30). Z untouched."""
    box = _box()
    _mask, origin, _ = extract_footprint(box, RES)
    t4 = placement_transform(
        (float(origin[0]), float(origin[1])), _identity_aff(), 0, 0, RES, PLATE
    )
    moved = box.copy()
    moved.apply_transform(t4)
    np.testing.assert_allclose(moved.bounds[0], [-50.0, -30.0, 0.0], atol=RES)
    np.testing.assert_allclose(moved.bounds[1], [-40.0, -20.0, 5.0], atol=RES)


def test_export_plates_writes_one_file_per_plate(tmp_path):
    box = _box()
    f = tmp_path / "piece.stl"
    box.export(f)
    _mask, origin, _ = extract_footprint(box, RES)
    t = placement_transform((float(origin[0]), float(origin[1])), _identity_aff(), 0, 0, RES, PLATE)
    placements = [Placement(0, 0, 0, 0, 0.0), Placement(1, 1, 0, 0, 0.0)]
    out = export_plates([f, f], placements, [t, t], tmp_path / "plates")
    assert [p.name for p in out] == ["plate_01.stl", "plate_02.stl"]
    reloaded = trimesh.load_mesh(out[0], process=False)
    np.testing.assert_allclose(reloaded.bounds[0], [-50.0, -30.0, 0.0], atol=RES)


def _shadow_setup(shift_mm=(0.0, 0.0)):
    """Box exported at anchor (100, 100); occupancy = its true placed mask."""
    box = _box()
    mask, origin, _ = extract_footprint(box, RES)
    row = col = 100
    t4 = placement_transform(
        (float(origin[0]), float(origin[1])), _identity_aff(), row, col, RES, PLATE
    )
    moved = box.copy()
    moved.apply_transform(t4)
    moved.apply_translation((shift_mm[0], shift_mm[1], 0.0))
    occ = np.zeros(PLATE_SHAPE, np.uint8)
    occ[row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
    return moved, occ


def test_verify_plate_passes_for_faithful_export():
    mesh, occ = _shadow_setup()
    assert verify_plate(mesh, occ, RES, PLATE, 0.0) == 0


@pytest.mark.parametrize("shift", [(5.0, 0.0), (0.0, -5.0)])
def test_verify_plate_catches_shifted_mesh(shift):
    mesh, occ = _shadow_setup(shift_mm=shift)
    assert verify_plate(mesh, occ, RES, PLATE, 0.0) > 0


def test_verify_plate_catches_out_of_bounds():
    # box sits at x [-40,-30]; +85mm puts it at [45,55], past the +50 plate edge
    mesh, occ = _shadow_setup(shift_mm=(85.0, 0.0))
    assert verify_plate(mesh, occ, RES, PLATE, 0.0) > 0

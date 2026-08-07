"""Export: transform composition, plate STL writing, merged-shadow self-check."""

import numpy as np
import pytest
import trimesh

from plate_packer.export import (
    _stl_records,
    _transform_triangles,
    export_plates,
    occupancy_from_full,
    placement_transform,
    read_stl_triangles,
    verify_plate,
)
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


def test_transform_triangles_applies_rigid_4x4():
    tris = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], np.float32)
    t4 = np.eye(4)
    t4[0, 3], t4[1, 3] = 10.0, -5.0  # pure translation
    out = _transform_triangles(tris, t4)
    np.testing.assert_allclose(
        out[0], [[10.0, -5.0, 0.0], [11.0, -5.0, 0.0], [10.0, -4.0, 0.0]], atol=1e-5
    )


def test_transform_triangles_leaves_z_untouched_under_z_rotation():
    # 90deg about Z: (x,y,z) -> (-y, x, z); Z coordinate must be preserved.
    tris = np.array([[[1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [1.0, 1.0, 7.0]]], np.float32)
    t4 = np.eye(4)
    t4[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    out = _transform_triangles(tris, t4)
    np.testing.assert_allclose(out[0, :, 2], [3.0, 3.0, 7.0], atol=1e-5)  # Z unchanged


def test_stl_records_unit_normal_and_verts():
    tris = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], np.float32)
    rec = _stl_records(tris)
    assert rec.shape == (1,)
    np.testing.assert_allclose(rec["normal"][0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(rec["verts"][0], tris[0], atol=1e-6)


def test_stl_records_degenerate_triangle_gets_zero_normal():
    tris = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], np.float32)
    rec = _stl_records(tris)
    np.testing.assert_allclose(rec["normal"][0], [0.0, 0.0, 0.0], atol=1e-6)


def test_export_plates_streams_multiple_pieces_into_one_plate(tmp_path):
    box = _box()
    f = tmp_path / "piece.stl"
    box.export(f)
    _mask, origin, _ = extract_footprint(box, RES)
    t0 = placement_transform(
        (float(origin[0]), float(origin[1])), _identity_aff(), 0, 0, RES, PLATE
    )
    t1 = placement_transform(
        (float(origin[0]), float(origin[1])), _identity_aff(), 0, 200, RES, PLATE
    )
    placements = [Placement(0, 0, 0, 0, 0.0), Placement(1, 0, 0, 200, 0.0)]
    out = export_plates([f, f], placements, [t0, t1], tmp_path / "plates")
    assert len(out) == 1
    reloaded = trimesh.load_mesh(out[0], process=False)
    assert len(reloaded.triangles) == 2 * len(box.triangles)  # both pieces present
    np.testing.assert_allclose(reloaded.bounds[0], [-50.0, -30.0, 0.0], atol=RES)
    np.testing.assert_allclose(reloaded.bounds[1], [-20.0, -20.0, 5.0], atol=RES)


def test_export_plates_writes_exact_binary_stl(tmp_path):
    box = _box()
    f = tmp_path / "p.stl"
    box.export(f)
    _mask, origin, _ = extract_footprint(box, RES)
    t = placement_transform((float(origin[0]), float(origin[1])), _identity_aff(), 0, 0, RES, PLATE)
    out = export_plates([f], [Placement(0, 0, 0, 0, 0.0)], [t], tmp_path / "plates")
    data = out[0].read_bytes()
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    assert n == len(box.triangles)
    assert len(data) == 84 + 50 * n  # exact binary-STL size, no ASCII fallback
    assert read_stl_triangles(out[0]).shape == box.triangles.shape  # round-trips


def test_export_plates_output_passes_verify_self_check(tmp_path):
    # Full chain the CLI relies on: streaming export -> read_stl_triangles ->
    # merged-shadow self-check must agree (0 violations) for a faithful placement.
    box = _box()
    f = tmp_path / "piece.stl"
    box.export(f)
    mask, origin, _ = extract_footprint(box, RES)
    row = col = 100
    t = placement_transform(
        (float(origin[0]), float(origin[1])), _identity_aff(), row, col, RES, PLATE
    )
    out = export_plates([f], [Placement(0, 0, row, col, 0.0)], [t], tmp_path / "plates")
    occ = np.zeros(PLATE_SHAPE, np.uint8)
    occ[row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
    assert verify_plate(read_stl_triangles(out[0]), occ, RES, PLATE, 0.0) == 0


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


def _lshape_mesh():
    """Asymmetric L-shaped extrusion (no rotational symmetry) -- regression
    fixture for the rotated-placement e2e self-check (final review finding:
    verify_plate was never exercised at angle != 0 with a real mesh)."""
    a = trimesh.creation.box(extents=(12.0, 4.0, 5.0))
    a.apply_translation((6.0, 2.0, 2.5))
    b = trimesh.creation.box(extents=(4.0, 10.0, 5.0))
    b.apply_translation((2.0, 5.0, 2.5))
    m = trimesh.util.concatenate([a, b])
    m.apply_translation((3.3, -7.1, 1.0))  # arbitrary offset incl. Z
    return m


@pytest.mark.parametrize("angle", [90.0, 45.0])
def test_verify_plate_passes_for_rotated_placement(angle):
    """Merged-shadow property at angle != 0: shadow of a correctly-placed,
    non-rotationally-symmetric mesh is a subset of the predicted occupancy."""
    mesh = _lshape_mesh()
    mask, origin, _ = extract_footprint(mesh, RES)
    rmask, aff = rotate_mask(mask, angle)
    row, col = 50, 80
    t4 = placement_transform((float(origin[0]), float(origin[1])), aff, row, col, RES, PLATE)
    moved = mesh.copy()
    moved.apply_transform(t4)
    occ = np.zeros(PLATE_SHAPE, np.uint8)
    occ[row : row + rmask.shape[0], col : col + rmask.shape[1]] |= rmask
    assert verify_plate(moved, occ, RES, PLATE, 2.0) == 0


@pytest.mark.parametrize("angle", [90.0, 45.0])
def test_verify_plate_catches_shifted_rotated_placement(angle):
    """Companion failing case: shifting the rotated mesh beyond the spacing
    margin must be caught by the self-check."""
    mesh = _lshape_mesh()
    mask, origin, _ = extract_footprint(mesh, RES)
    rmask, aff = rotate_mask(mask, angle)
    row, col = 50, 80
    t4 = placement_transform((float(origin[0]), float(origin[1])), aff, row, col, RES, PLATE)
    moved = mesh.copy()
    moved.apply_transform(t4)
    moved.apply_translation((3.0, 0.0, 0.0))  # > 2.0mm spacing margin
    occ = np.zeros(PLATE_SHAPE, np.uint8)
    occ[row : row + rmask.shape[0], col : col + rmask.shape[1]] |= rmask
    assert verify_plate(moved, occ, RES, PLATE, 2.0) > 0


def test_read_stl_triangles_matches_trimesh(tmp_path):
    """Binary fast path returns the same triangle soup trimesh would."""
    box = _box()
    f = tmp_path / "box.stl"
    box.export(f)  # trimesh writes binary STL for .stl
    tris = read_stl_triangles(f)
    assert tris.shape == box.triangles.shape
    got = np.sort(tris.reshape(-1, 9), axis=0)
    expected = np.sort(np.asarray(box.triangles, dtype=np.float32).reshape(-1, 9), axis=0)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_read_stl_triangles_ascii_fallback(tmp_path):
    box = _box()
    f = tmp_path / "box_ascii.stl"
    f.write_bytes(trimesh.exchange.stl.export_stl_ascii(box).encode())
    tris = read_stl_triangles(f)
    assert tris.shape == box.triangles.shape


def test_verify_plate_accepts_raw_triangle_array():
    """The CLI verify path feeds raw (n,3,3) triangles, not a Trimesh."""
    mesh, occ = _shadow_setup()
    assert verify_plate(np.asarray(mesh.triangles), occ, RES, PLATE, 0.0) == 0


def test_read_stl_triangles_empty_file_returns_no_triangles(tmp_path):
    f = tmp_path / "empty.stl"
    f.write_bytes(b"")
    assert read_stl_triangles(f).shape == (0, 3, 3)


def test_read_stl_triangles_truncated_binary_is_not_misparsed(tmp_path):
    """A binary header claiming 1000 triangles over a 10-byte body must fail
    the size check and route to the fallback loader -- never be parsed as
    1000 garbage triangles by the fast path."""
    f = tmp_path / "trunc.stl"
    f.write_bytes(b"\x00" * 80 + np.uint32(1000).tobytes() + b"\x00" * 10)
    try:
        tris = read_stl_triangles(f)
    except Exception:  # the fallback loader may raise; that is acceptable
        return
    assert len(tris) == 0


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 137.0], ids=lambda a: f"deg{a:g}")
def test_occupancy_from_full_covers_body_placement(angle):
    """A full mask (>= body) placed via occupancy_from_full must cover the body
    placed the normal way at the same anchor, for any rotation."""
    full = np.zeros((30, 30), np.uint8)
    full[5:25, 5:25] = 1  # full footprint
    body = np.zeros((30, 30), np.uint8)
    body[10:25, 10:25] = 1  # body = subset (base cleared), SAME canvas

    body_rot, body_aff = rotate_mask(body, angle)
    full_rot, full_aff = rotate_mask(full, angle)

    row, col = 40, 50
    occ_body = np.zeros((120, 120), np.uint8)
    occ_body[row : row + body_rot.shape[0], col : col + body_rot.shape[1]] |= body_rot

    occ_full = np.zeros((120, 120), np.uint8)
    occupancy_from_full(occ_full, full_rot, body_aff, full_aff, row, col)

    # full-shadow occupancy is a superset of the body placement (alignment holds)
    assert (occ_body.astype(bool) & ~occ_full.astype(bool)).sum() == 0
    assert occ_full.sum() >= occ_body.sum()

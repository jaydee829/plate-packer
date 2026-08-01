"""Export: transform composition, plate STL writing, merged-shadow self-check."""

import numpy as np
import pytest

from plate_packer.export import placement_transform
from plate_packer.packer import rotate_mask


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

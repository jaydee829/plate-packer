"""Export: placements -> exact rigid transforms -> merged plate STLs + self-check.

Composes the full pixel chain as 2D affines (mesh mm -> prepared px -> rotated
px -> plate px -> slicer mm with origin at plate center), so the mesh lands
exactly where the packer reserved pixels. The world rotation comes out of the
composed matrix -- nominal angle sign conventions are never trusted.
"""

import numpy as np


def placement_transform(
    prepared_origin_mm: tuple[float, float],
    rotation_affine: np.ndarray,
    row: int,
    col: int,
    working_res_mm: float,
    plate_mm: tuple[float, float],
) -> np.ndarray:
    """4x4 rigid transform (rotation about Z + XY translation, Z identity)."""
    res = working_res_mm
    ox, oy = prepared_origin_mm
    to_px = np.array([[1 / res, 0, -ox / res], [0, 1 / res, -oy / res], [0, 0, 1]])
    rot = np.vstack([rotation_affine, [0.0, 0.0, 1.0]])
    anchor = np.array([[1, 0, col], [0, 1, row], [0, 0, 1]], dtype=float)
    to_mm = np.array([[res, 0, 0], [0, res, 0], [0, 0, 1]])
    recenter = np.array(
        [[1, 0, -plate_mm[0] / 2], [0, 1, -plate_mm[1] / 2], [0, 0, 1]], dtype=float
    )
    m = recenter @ to_mm @ anchor @ rot @ to_px
    lin = m[:2, :2]
    if not np.allclose(lin @ lin.T, np.eye(2), atol=1e-6) or np.linalg.det(lin) < 0:
        raise ValueError("composed transform is not a proper rotation (mirroring or scale)")
    t4 = np.eye(4)
    t4[:2, :2] = lin
    t4[:2, 3] = m[:2, 2]
    return t4

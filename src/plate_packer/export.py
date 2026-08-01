"""Export: placements -> exact rigid transforms -> merged plate STLs + self-check.

Composes the full pixel chain as 2D affines (mesh mm -> prepared px -> rotated
px -> plate px -> slicer mm with origin at plate center), so the mesh lands
exactly where the packer reserved pixels. The world rotation comes out of the
composed matrix -- nominal angle sign conventions are never trusted.
"""

from pathlib import Path

import cv2
import numpy as np
import trimesh


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


def load_piece_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh


_STL_RECORD = np.dtype([("normal", "<f4", 3), ("verts", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl_triangles(path: Path) -> np.ndarray:
    """(n, 3, 3) float32 triangle soup from an STL, without building a mesh.

    Binary STLs (everything export_plates writes) are read directly with
    numpy: trimesh's loader materializes a Scene and deep-copies the merged
    mesh, which doubles peak memory on multi-hundred-MB plates and OOMed the
    first real-world verify run. Non-binary files fall back to trimesh.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fh.seek(80)
        count_bytes = fh.read(4)
    if len(count_bytes) == 4:
        n = int(np.frombuffer(count_bytes, "<u4")[0])
        if 84 + 50 * n == size:  # authoritative binary-STL signature
            return np.fromfile(path, dtype=_STL_RECORD, count=n, offset=84)["verts"]
    return np.asarray(load_piece_mesh(path).triangles, dtype=np.float32)


def export_plates(files, placements, transforms, output_dir: Path) -> list[Path]:
    """One merged binary STL per plate; meshes loaded and freed plate-by-plate
    so memory stays bounded to a single plate. Z is never modified."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_plates = max(p.plate for p in placements) + 1
    written: list[Path] = []
    for plate_idx in range(n_plates):
        parts = []
        for pl in placements:
            if pl.plate != plate_idx:
                continue
            mesh = load_piece_mesh(files[pl.piece])
            mesh.apply_transform(transforms[pl.piece])
            parts.append(mesh)
        merged = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
        path = output_dir / f"plate_{plate_idx + 1:02d}.stl"
        merged.export(path)
        written.append(path)
    return written


def _rasterize_plate_shadow(mesh, working_res_mm, plate_mm, tol_px):
    """Shadow of a slicer-frame mesh on the plate pixel grid. Returns
    (canvas, n_oob_vertices): vertices beyond the plate (+/- tol_px slack)."""
    h_px = round(plate_mm[1] / working_res_mm)
    w_px = round(plate_mm[0] / working_res_mm)
    tris3 = mesh if isinstance(mesh, np.ndarray) else mesh.triangles
    tris = tris3[:, :, :2]
    finite = np.isfinite(tris).all(axis=(1, 2))
    tris = tris[finite]
    tris_px = np.round((tris + np.array(plate_mm) / 2) / working_res_mm).astype(np.int64)
    oob = (
        (tris_px[..., 0] < -tol_px)
        | (tris_px[..., 0] > w_px - 1 + tol_px)
        | (tris_px[..., 1] < -tol_px)
        | (tris_px[..., 1] > h_px - 1 + tol_px)
    )
    canvas = np.zeros((h_px, w_px), np.uint8)
    clipped = np.clip(tris_px, [0, 0], [w_px - 1, h_px - 1]).astype(np.int32)
    # One fill per triangle: batched fillPoly XORs overlaps (see bugs.md).
    for tri in clipped:
        cv2.fillConvexPoly(canvas, tri, 1)
    return canvas, int(oob.sum())


def verify_plate(plate_mesh, occupancy, working_res_mm, plate_mm, spacing_mm) -> int:
    """Merged-shadow self-check: count actual-shadow pixels outside the
    predicted occupancy (subset assertion -- occupancy legitimately includes
    spacing margins). 0 = pass. spacing == 0 gets 1 px rounding tolerance.

    plate_mesh: a Trimesh, or a raw (n, 3, 3) triangle array (see
    read_stl_triangles -- the CLI feeds triangles directly to keep peak
    memory bounded on large merged plates)."""
    tol_px = 0 if spacing_mm > 0 else 1
    shadow, n_oob = _rasterize_plate_shadow(plate_mesh, working_res_mm, plate_mm, tol_px)
    predicted = occupancy
    if tol_px:
        predicted = cv2.dilate(occupancy, np.ones((3, 3), np.uint8))
    violations = int((shadow.astype(bool) & ~predicted.astype(bool)).sum())
    return violations + n_oob

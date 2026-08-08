"""Corpus probe for the raft-signature gate (not part of the package).

For every STL under a root: raw area-knee (gated=False), band dominance at the
knee, and the gate verdict. Used to calibrate/recheck RAFT_BAND_DOMINANCE_MAX
against a real corpus (see docs/superpowers/specs/2026-08-08-raft-signature-
gate-design.md). Runtime is a few seconds per multi-million-triangle STL; use
start/count to chunk long runs.

Usage: uv run python tools/probe_raft_gate.py <root_dir> [start] [count]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from plate_packer.export import load_piece_mesh
from plate_packer.footprint import (
    BAND_MM,
    DETECT_RES_MM,
    RAFT_BAND_DOMINANCE_MAX,
    _raster,
    detect_base_cut,
)


def probe(path: Path) -> tuple[float, float, int, str]:
    tris = np.asarray(load_piece_mesh(path).triangles)
    tris = tris[np.isfinite(tris).all(axis=(1, 2))]
    # keep in sync with PackConfig.support_cut_cap_mm default
    knee = detect_base_cut(tris, DETECT_RES_MM, 5.0, gated=False)
    if knee <= 0:
        return 0.0, 0.0, 0, "no-knee"
    xy = tris[:, :, :2]
    z = tris[:, :, 2]
    flat = xy.reshape(-1, 2)
    origin = flat.min(axis=0)
    size_px = np.round((flat.max(axis=0) - origin) / DETECT_RES_MM).astype(int) + 1
    shape = (int(size_px[1]), int(size_px[0]))
    tri_px = np.round((xy - origin) / DETECT_RES_MM).astype(np.int32)
    plane = float(z.min()) + knee + BAND_MM
    straddle = (z.min(axis=1) < plane) & (z.max(axis=1) > plane)
    band = _raster(tri_px[straddle], shape) if straddle.any() else np.zeros(shape, np.uint8)
    if not band.any():
        return knee, 1.0, 0, "reject"
    _n, _labels, comp_stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    sizes = comp_stats[1:, cv2.CC_STAT_AREA]
    dom = float(sizes.max() / sizes.sum())
    verdict = "accept" if dom <= RAFT_BAND_DOMINANCE_MAX else "reject"
    return knee, dom, len(sizes), verdict


def main() -> None:
    root = Path(sys.argv[1])
    paths = sorted(root.rglob("*.stl"), key=lambda p: str(p).lower())
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else len(paths)
    for path in paths[start : start + count]:
        t0 = time.perf_counter()
        try:
            knee, dom, n_comp, verdict = probe(path)
        except Exception as e:
            print(f"{path}\t-\t-\t-\tERROR\t{e}", flush=True)
            continue
        print(
            f"{path}\t{knee:.2f}\t{dom:.3f}\t{n_comp}\t{verdict}\t{time.perf_counter() - t0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

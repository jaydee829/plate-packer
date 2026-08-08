"""Gated integration: base-cut detection on real pre-supported STLs.

Deselected by default (see pyproject addopts). Run explicitly with real assets:
    uv run pytest -m example_stls -s
"""

from pathlib import Path

import numpy as np
import pytest

from plate_packer.export import load_piece_mesh
from plate_packer.footprint import DETECT_RES_MM, detect_base_cut, extract_footprints
from plate_packer.footprint_io import CANONICAL_RES_MM

pytestmark = pytest.mark.example_stls

EXAMPLES = Path("example_stls")


def _finite_tris(path):
    tris = np.asarray(load_piece_mesh(path).triangles)
    return tris[np.isfinite(tris).all(axis=(1, 2))]


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_stls junction absent")
def test_detection_fires_on_real_supports(capsys):
    # Only the pre-supported exports carry rafts; match case-insensitively
    # ("_Supported" / "_supported" both occur in the corpus).
    stls = [p for p in sorted(EXAMPLES.rglob("*.stl")) if "_supported" in p.stem.lower()][:8]
    if not stls:
        pytest.skip("no *_supported.stl files under example_stls")
    results = []
    for stl in stls:
        full, body, _origin, cut, _stats = extract_footprints(
            load_piece_mesh(stl), CANONICAL_RES_MM, 5.0
        )
        reduction = 1 - body.sum() / full.sum() if full.sum() else 0.0
        results.append((stl.name, cut, reduction))
    with capsys.disabled():
        print("\nbase-cut detection on real supported STLs:")
        for name, cut, reduction in results:
            print(f"  {name}: cut={cut:.2f}mm  footprint area -{reduction:.1%}")
    assert any(cut > 0 for _name, cut, _r in results), "detector never fired on real supports"
    assert any(r > 0.05 for _name, _c, r in results), "no meaningful footprint reduction"


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_stls junction absent")
@pytest.mark.parametrize(
    ("name", "accepted"),
    [
        pytest.param("STL_Armaros_Body_Wingless_Supported.stl", True, id="supported-body-accept"),
        pytest.param("STL_Armaros_LeftWing_supported.stl", True, id="supported-wing-accept"),
        pytest.param("Pose2_Wing_R.stl", False, id="unsupported-wing-reject"),
        pytest.param("pose01-wing1.stl", False, id="unsupported-taper-reject"),
    ],
)
def test_raft_gate_verdict_on_real_corpus(name, accepted):
    """Pins the ADR-014 corpus calibration on a fixed sample: the raw knee fires
    on every one of these files (gated=False), and the band-dominance gate
    accepts only the truly supported ones. If a detector-constant change flips
    a verdict here, rerun tools/probe_raft_gate.py over the full corpus and
    revisit RAFT_BAND_DOMINANCE_MAX from the measured split -- never by hand."""
    matches = sorted(EXAMPLES.rglob(name))
    if not matches:
        pytest.skip(f"{name} not found under example_stls")
    tris = _finite_tris(matches[0])
    assert detect_base_cut(tris, DETECT_RES_MM, 5.0, gated=False) > 0, "raw knee must fire"
    gated_cut = detect_base_cut(tris, DETECT_RES_MM, 5.0)
    assert (gated_cut > 0) == accepted

"""Gated integration: base-cut detection on real pre-supported STLs.

Deselected by default (see pyproject addopts). Run explicitly with real assets:
    uv run pytest -m example_stls -s
"""

from pathlib import Path

import pytest

from plate_packer.export import load_piece_mesh
from plate_packer.footprint import extract_footprints
from plate_packer.footprint_io import CANONICAL_RES_MM

pytestmark = pytest.mark.example_stls

EXAMPLES = Path("example_stls")


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_stls junction absent")
def test_detection_fires_on_real_supports(capsys):
    stls = sorted(EXAMPLES.rglob("*.stl"))[:8]
    if not stls:
        pytest.skip("no STL files under example_stls")
    results = []
    for stl in stls:
        full, body, _origin, cut, _stats = extract_footprints(
            load_piece_mesh(stl), CANONICAL_RES_MM, 5.0
        )
        reduction = 1 - body.sum() / full.sum() if full.sum() else 0.0
        results.append((stl.name, cut, reduction))
    with capsys.disabled():
        print("\nbase-cut detection on real STLs:")
        for name, cut, reduction in results:
            print(f"  {name}: cut={cut:.2f}mm  footprint area -{reduction:.1%}")
    assert any(cut > 0 for _name, cut, _r in results), "detector never fired on real supports"

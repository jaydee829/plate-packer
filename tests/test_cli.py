"""CLI smoke tests over a tmp tree with synthetic STLs."""

import json

import pytest
import trimesh
from typer.testing import CliRunner

from plate_packer.cli import app
from plate_packer.footprint_io import SCHEMA_VERSION, doc_path, file_sha256

runner = CliRunner()


@pytest.fixture
def stl_tree(tmp_path):
    root = tmp_path / "models" / "nested"
    root.mkdir(parents=True)
    trimesh.creation.box(extents=(4, 2, 1)).export(root / "brick.stl")
    trimesh.creation.box(extents=(2, 2, 2)).export(tmp_path / "models" / "cube.stl")
    return tmp_path / "models"


def test_footprints_writes_contract_docs(stl_tree, tmp_path):
    out = tmp_path / "fp"
    result = runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    assert result.exit_code == 0
    sha = file_sha256(stl_tree / "cube.stl")
    doc = json.loads(doc_path(out, sha).read_text(encoding="utf-8"))
    assert doc["schema_version"] == SCHEMA_VERSION
    assert "2 written" in result.output


def test_footprints_skips_cached_unless_forced(stl_tree, tmp_path):
    out = tmp_path / "fp"
    runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    second = runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    assert "2 skipped" in second.output
    forced = runner.invoke(
        app, ["footprints", str(stl_tree), "--footprints-dir", str(out), "--force"]
    )
    assert "2 written" in forced.output


def test_footprints_reports_failures_and_continues(stl_tree, tmp_path):
    bad = stl_tree / "corrupt.stl"
    bad.write_bytes(b"this is not an stl")
    result = runner.invoke(
        app, ["footprints", str(stl_tree), "--footprints-dir", str(tmp_path / "fp")]
    )
    assert result.exit_code == 1
    assert "2 written" in result.output
    assert "1 failed" in result.output
    assert "corrupt.stl" in result.output

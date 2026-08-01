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


def test_footprints_dedupes_overlapping_inputs(stl_tree, tmp_path):
    """When both a directory and a file inside it are passed, dedupe while preserving order."""
    out = tmp_path / "fp"
    # Pass both the directory and one specific file inside it
    cube_file = stl_tree / "cube.stl"
    # Use --force to prevent caching from masking the duplicate
    result = runner.invoke(
        app, ["footprints", str(stl_tree), str(cube_file), "--footprints-dir", str(out), "--force"]
    )
    assert result.exit_code == 0
    # Should only write 2 files (cube and brick), not 3 (cube processed twice + brick)
    assert "2 written" in result.output
    # Verify the exact count to ensure no double-processing
    assert "3 written" not in result.output


def test_footprints_no_files_found_message(tmp_path):
    """When no STL/OBJ files are found, output should say so."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out = tmp_path / "fp"
    result = runner.invoke(app, ["footprints", str(empty_dir), "--footprints-dir", str(out)])
    assert result.exit_code == 0
    assert "no STL/OBJ files found" in result.output


def test_footprints_subcommand_works_without_local_footprints_dir(stl_tree, tmp_path, monkeypatch):
    """Regression: with a single-command typer app, 'footprints' was parsed as a
    path argument and only worked when ./footprints happened to exist locally."""
    workdir = tmp_path / "clean-cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    out = tmp_path / "fp"
    result = runner.invoke(app, ["footprints", str(stl_tree), "--footprints-dir", str(out)])
    assert result.exit_code == 0
    assert "2 written" in result.output

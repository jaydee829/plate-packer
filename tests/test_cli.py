"""CLI smoke tests over a tmp tree with synthetic STLs."""

import json

import pytest
import trimesh
from typer.testing import CliRunner

from plate_packer.cli import app
from plate_packer.footprint_io import SCHEMA_VERSION, doc_path, file_sha256

runner = CliRunner()

TEST_CONFIG = """
[printer]
plate_mm = [40.0, 30.0]
build_height_mm = 50.0

[packing]
working_res_mm = 0.1
spacing_mm = 1.0
rotations = 1
"""


def _write_box(path, w=10.0, d=10.0, h=5.0):
    b = trimesh.creation.box(extents=(w, d, h))
    b.apply_translation((w / 2, d / 2, h / 2))
    b.export(path)


def _setup(tmp_path, monkeypatch, boxes=((10, 10, 5), (10, 10, 5))):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "models"
    src.mkdir()
    for i, (w, d, h) in enumerate(boxes):
        _write_box(src / f"box{i}.stl", w, d, h)
    (tmp_path / "config.toml").write_text(TEST_CONFIG, encoding="utf-8")
    return src


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


def test_pack_writes_plates_and_report(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "plates" / "plate_01.stl").exists()
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "plate_01.stl" in report
    assert "box0.stl" in report and "box1.stl" in report


def test_pack_verifies_by_default(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src)])
    assert "verify" in result.output.lower()


def test_pack_no_verify_skips(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--no-verify"])
    assert result.exit_code == 0, result.output
    assert "verify" not in result.output.lower()


def test_pack_too_tall_piece_listed_and_aborts(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch, boxes=((10, 10, 5), (10, 10, 60)))
    result = runner.invoke(app, ["pack", str(src)])
    assert result.exit_code == 1
    assert "box1.stl" in result.output and "height" in result.output.lower()
    assert not (tmp_path / "plates").exists()


def test_pack_oversized_piece_listed_and_aborts(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch, boxes=((10, 10, 5), (60, 40, 5)))
    result = runner.invoke(app, ["pack", str(src)])
    assert result.exit_code == 1
    assert "box1.stl" in result.output and "fit" in result.output.lower()


def test_pack_collects_all_errors(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch, boxes=((10, 10, 60), (60, 40, 5)))
    result = runner.invoke(app, ["pack", str(src)])
    assert result.exit_code == 1
    assert "box0.stl" in result.output and "box1.stl" in result.output


def test_pack_empty_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "models"
    empty.mkdir()
    result = runner.invoke(app, ["pack", str(empty)])
    assert result.exit_code == 0
    assert "no STL/OBJ files found" in result.output


def test_pack_spillover_to_second_plate(tmp_path, monkeypatch):
    # Four 18x18 pieces + 1mm gaps cannot all fit a 40x30 plate.
    src = _setup(tmp_path, monkeypatch, boxes=((18, 18, 5),) * 4)
    result = runner.invoke(app, ["pack", str(src)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "plates" / "plate_02.stl").exists()


def test_pack_budget_zero_is_plain_greedy(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "0"])
    assert result.exit_code == 0
    assert "improve:" not in result.output
    assert "improvement:" not in result.output


def test_pack_improvement_summary_in_report(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "5", "--seed", "1"])
    assert result.exit_code == 0
    assert "improvement:" in result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "improvement:" in report
    assert "evaluations" in report

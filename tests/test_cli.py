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


def test_pack_cli_rejects_invalid_coarse_res(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--coarse-res", "0.04"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ZeroDivisionError)


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


def test_pack_from_file_packs_listed_files(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sel = tmp_path / "sel.txt"
    sel.write_text("models/box0.stl\nmodels/box1.stl\n", encoding="utf-8")
    result = runner.invoke(app, ["pack", "--from-file", str(sel)])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "box0.stl" in report and "box1.stl" in report


def test_pack_from_file_ignores_blanks_and_comments(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sel = tmp_path / "sel.txt"
    sel.write_text(
        "# selection\n\nmodels/box0.stl\n   \n# trailing comment\nmodels/box1.stl\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["pack", "--from-file", str(sel)])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "box0.stl" in report and "box1.stl" in report


def test_pack_from_file_unions_with_positional_paths(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sel = tmp_path / "sel.txt"
    sel.write_text("models/box1.stl\n", encoding="utf-8")
    result = runner.invoke(app, ["pack", "models/box0.stl", "--from-file", str(sel)])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "box0.stl" in report and "box1.stl" in report


def test_pack_from_file_missing_entry_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sel = tmp_path / "sel.txt"
    sel.write_text("models/box0.stl\nmodels/ghost.stl\n", encoding="utf-8")
    result = runner.invoke(app, ["pack", "--from-file", str(sel)])
    assert result.exit_code != 0
    assert "ghost.stl" in result.output


def test_pack_from_file_missing_list_file_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", "--from-file", str(tmp_path / "nope.txt")])
    assert result.exit_code != 0
    assert "nope.txt" in result.output


def test_pack_cli_greedy_uses_shape_aware_angles(tmp_path, monkeypatch):
    # A budget-0 pack still succeeds end-to-end (report written, verify ok) with
    # the shape-aware greedy path.
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["pack", str(src), "--budget", "0"])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "plates" / "report.txt").read_text(encoding="utf-8")
    assert "plate(s)" in report


def test_pack_cli_accepts_coarse_res_and_beam_options(tmp_path, monkeypatch):
    src = _setup(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["pack", str(src), "--budget", "1", "--coarse-res", "0.4", "--beam", "2"]
    )
    assert result.exit_code == 0, result.output


def _fused_piece():
    """Solid raft (full outline) + a narrower body with the SAME outer bbox but a
    hollow centre — realistic: outer extent matches, interior differs. Rafts of
    neighbours overlap; nothing hangs off the plate."""
    raft = trimesh.creation.box(extents=(20, 20, 2))
    raft.apply_translation([0, 0, 1])
    left = trimesh.creation.box(extents=(4, 20, 20))
    left.apply_translation([-8, 0, 12])
    right = trimesh.creation.box(extents=(4, 20, 20))
    right.apply_translation([8, 0, 12])
    return trimesh.util.concatenate([raft, left, right])


def _write_pieces(stl_dir, n):
    stl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _fused_piece().export(str(stl_dir / f"piece_{i}.stl"))


def _cfg(tmp_path, support_aware):
    p = tmp_path / "config.toml"
    p.write_text(
        f"[packing]\nsupport_aware = {'true' if support_aware else 'false'}\n", encoding="utf-8"
    )
    return p


@pytest.mark.parametrize("support_aware", [False, True], ids=["off", "on"])
def test_pack_self_check_passes(tmp_path, support_aware):
    stl_dir = tmp_path / "stls"
    _write_pieces(stl_dir, 2)
    result = CliRunner().invoke(
        app,
        [
            "pack",
            str(stl_dir),
            "--config",
            str(_cfg(tmp_path, support_aware)),
            "--footprints-dir",
            str(tmp_path / "fp"),
            "--out",
            str(tmp_path / "out"),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verify" in result.output
    assert "FAILED" not in result.output
    assert (tmp_path / "out" / "plate_01.stl").exists()


def test_pack_support_aware_stale_detector_version_falls_back_to_full_shadow(tmp_path, monkeypatch):
    """Regression: when a doc is stale (detector_version mismatch) but still has
    an old body_mask, and re-extraction fails, the piece must fall back to the
    full shadow rather than silently packing the stale body mask."""
    import plate_packer.cli as cli_mod
    from plate_packer.footprint import extract_footprints
    from plate_packer.footprint_io import file_sha256, save_doc

    stl_dir = tmp_path / "stls"
    _write_pieces(stl_dir, 1)
    piece = next(stl_dir.glob("*.stl"))
    fp = tmp_path / "fp"
    sha = file_sha256(piece)

    # Pre-seed a doc that has_current_doc() considers current (schema_version
    # only) but whose detector_version is stale, with an old body_mask still
    # present -- the exact condition the review finding was about.
    full, body, origin, cut, stats = extract_footprints(cli_mod.load_piece_mesh(piece), 0.05, 4.0)
    save_doc(
        fp,
        sha,
        full,
        origin,
        stats,
        res_mm_per_px=0.05,
        body_mask=body,
        cut_z_mm=cut,
        detector_version=0,  # stale: real DETECTOR_VERSION is 1
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(cli_mod, "extract_footprints", _boom)

    result = CliRunner().invoke(
        app,
        [
            "pack",
            str(stl_dir),
            "--config",
            str(_cfg(tmp_path, True)),
            "--footprints-dir",
            str(fp),
            "--out",
            str(tmp_path / "out"),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "body-mask extraction failed" in result.output
    assert "using full shadow" in result.output
    assert "verify" in result.output
    assert "FAILED" not in result.output


def test_pack_support_aware_writes_body_mask(tmp_path):
    stl_dir = tmp_path / "stls"
    _write_pieces(stl_dir, 1)
    fp = tmp_path / "fp"
    CliRunner().invoke(
        app,
        [
            "pack",
            str(stl_dir),
            "--config",
            str(_cfg(tmp_path, True)),
            "--footprints-dir",
            str(fp),
            "--out",
            str(tmp_path / "out"),
            "--budget",
            "0",
        ],
    )
    from plate_packer.footprint_io import file_sha256, load_doc

    doc = load_doc(fp, file_sha256(next(stl_dir.glob("*.stl"))))
    assert doc.body_mask is not None
    assert doc.detector_version == 1

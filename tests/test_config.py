"""Config loading: defaults, TOML overrides, validation."""

from pathlib import Path

import pytest

from plate_packer.config import PackConfig, load_config

FULL_TOML = """
[printer]
plate_mm = [218.88, 122.88]
build_height_mm = 260.0

[packing]
working_res_mm = 0.05
spacing_mm = 1.0
edge_margin_mm = 2.0
rotations = 36

[paths]
footprints_dir = "cache/fp"
output_dir = "out"
"""


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("plate_mm", (197.0, 122.0)),
        ("build_height_mm", 245.0),
        ("working_res_mm", 0.1),
        ("spacing_mm", 2.0),
        ("edge_margin_mm", 0.0),
        ("rotations", 8),
        ("footprints_dir", Path("footprints")),
        ("output_dir", Path("plates")),
    ],
)
def test_defaults_without_file(tmp_path, field, expected):
    cfg = load_config(tmp_path / "missing.toml")
    assert getattr(cfg, field) == expected


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("plate_mm", (218.88, 122.88)),
        ("build_height_mm", 260.0),
        ("working_res_mm", 0.05),
        ("spacing_mm", 1.0),
        ("edge_margin_mm", 2.0),
        ("rotations", 36),
        ("footprints_dir", Path("cache/fp")),
        ("output_dir", Path("out")),
    ],
)
def test_toml_overrides(tmp_path, field, expected):
    p = tmp_path / "config.toml"
    p.write_text(FULL_TOML, encoding="utf-8")
    assert getattr(load_config(p), field) == expected


def test_partial_toml_keeps_other_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[packing]\nspacing_mm = 0.5\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.spacing_mm == 0.5
    assert cfg.plate_mm == (197.0, 122.0)


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[printer]\nnozzle = 0.4\n[extra]\nfoo = 1\n", encoding="utf-8")
    assert load_config(p) == PackConfig()


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        ("[printer]\nplate_mm = [197.0]\n", "plate_mm"),
        ("[printer]\nplate_mm = [0.0, 122.0]\n", "plate_mm"),
        ("[printer]\nbuild_height_mm = -1\n", "build_height_mm"),
        ("[packing]\nworking_res_mm = 0.0\n", "working_res_mm"),
        ("[packing]\nworking_res_mm = 0.07\n", "integer multiple"),
        ("[packing]\nworking_res_mm = 0.025\n", "integer multiple"),
        ("[packing]\nspacing_mm = -0.1\n", "spacing_mm"),
        ("[packing]\nedge_margin_mm = -1\n", "edge_margin_mm"),
        ("[packing]\nrotations = 0\n", "rotations"),
    ],
)
def test_validation_errors(tmp_path, toml, match):
    p = tmp_path / "config.toml"
    p.write_text(toml, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(p)


def test_working_res_triple_of_canonical_ok(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[packing]\nworking_res_mm = 0.15\n", encoding="utf-8")
    assert load_config(p).working_res_mm == 0.15

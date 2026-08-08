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
        ("spacing_mm", 1.0),
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
    p.write_text("[packing]\nworking_res_mm = 0.15\ncoarse_res_mm = 0.30\n", encoding="utf-8")
    assert load_config(p).working_res_mm == 0.15


def test_improvement_defaults():
    cfg = PackConfig()
    assert cfg.improve_budget_s == 2700.0
    assert cfg.min_improvement == 0.005
    assert cfg.patience == 30
    assert cfg.seed == 0
    assert cfg.placement == "contact"


@pytest.mark.parametrize(
    "toml_body, bad_key",
    [
        ("[packing]\nimprove_budget_s = -1", "improve_budget_s"),
        ("[packing]\nmin_improvement = -0.1", "min_improvement"),
        ("[packing]\npatience = 0", "patience"),
        ('[packing]\nplacement = "wizard"', "placement"),
    ],
    ids=["negative-budget", "negative-min-improvement", "zero-patience", "unknown-placement"],
)
def test_improvement_knob_validation(tmp_path, toml_body, bad_key):
    p = tmp_path / "config.toml"
    p.write_text(toml_body, encoding="utf-8")
    with pytest.raises(ValueError, match=bad_key):
        load_config(p)


def test_improvement_knobs_load_from_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[packing]\nimprove_budget_s = 60\nmin_improvement = 0.01\n"
        'patience = 5\nseed = 42\nplacement = "bottom_left"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.improve_budget_s == 60.0
    assert cfg.min_improvement == 0.01
    assert cfg.patience == 5
    assert cfg.seed == 42
    assert cfg.placement == "bottom_left"


def test_config_defaults_include_coarse_to_fine_knobs(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.coarse_res_mm == 0.4
    assert cfg.beam == 5
    assert cfg.angle_cap == 12
    assert cfg.min_edge_frac == 0.1
    assert cfg.safety_grid == 16  # uniform-rotation backstop on by default (2026-08-07 benchmark)
    assert cfg.edge_contact_weight == 1.0
    assert cfg.ordering == "difficulty"


def test_config_reads_coarse_to_fine_knobs(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[packing]\n"
        "coarse_res_mm = 0.5\n"
        "beam = 8\n"
        "angle_cap = 6\n"
        "min_edge_frac = 0.2\n"
        "safety_grid = 12\n"
        "edge_contact_weight = 2.0\n"
        'ordering = "area"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert (cfg.coarse_res_mm, cfg.beam, cfg.angle_cap) == (0.5, 8, 6)
    assert (cfg.min_edge_frac, cfg.safety_grid, cfg.edge_contact_weight) == (0.2, 12, 2.0)
    assert cfg.ordering == "area"


@pytest.mark.parametrize(
    "key, match",
    [
        ("coarse_res_mm = 0.05", "coarse_res_mm"),  # below working_res_mm (0.1)
        ("coarse_res_mm = 0.35", "coarse_res_mm"),  # not an integer multiple of 0.1
        ("beam = 0", "beam"),
        ("angle_cap = 0", "angle_cap"),
        ("min_edge_frac = 0", "min_edge_frac"),
        ("min_edge_frac = 1.5", "min_edge_frac"),
        ("safety_grid = -1", "safety_grid"),
        ("edge_contact_weight = -0.5", "edge_contact_weight"),
        ('ordering = "spiral"', "ordering"),
    ],
    ids=[
        "coarse-below-working",
        "coarse-not-multiple",
        "beam-zero",
        "cap-zero",
        "edge-frac-zero",
        "edge-frac-gt-one",
        "safety-negative",
        "edge-weight-negative",
        "ordering-unknown",
    ],
)
def test_config_rejects_invalid_coarse_to_fine_knobs(tmp_path, key, match):
    p = tmp_path / "config.toml"
    p.write_text(f"[packing]\n{key}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(p)


def test_support_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.support_aware is False
    assert cfg.support_cut_cap_mm == 3.0  # ADR-015: hard ceiling on the fusion zone


def test_support_knobs_load_from_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[packing]\nsupport_aware = true\nsupport_cut_cap_mm = 3.0\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.support_aware is True
    assert cfg.support_cut_cap_mm == 3.0


@pytest.mark.parametrize("value", [pytest.param("0", id="zero"), pytest.param("-1", id="negative")])
def test_support_cut_cap_must_be_positive(tmp_path, value):
    p = tmp_path / "config.toml"
    p.write_text(f"[packing]\nsupport_cut_cap_mm = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="support_cut_cap_mm"):
        load_config(p)

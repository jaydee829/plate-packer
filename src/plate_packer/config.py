"""Pack configuration: printer + packing + paths, from config.toml (all optional)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from plate_packer.footprint_io import CANONICAL_RES_MM

_RATIO_TOL = 1e-6


@dataclass(frozen=True)
class PackConfig:
    plate_mm: tuple[float, float] = (197.0, 122.0)  # Anycubic Photon Mono X 6K
    build_height_mm: float = 245.0
    working_res_mm: float = 0.1
    spacing_mm: float = 2.0  # true minimum gap between pieces (ADR-010)
    edge_margin_mm: float = 0.0
    rotations: int = 8
    footprints_dir: Path = Path("footprints")
    output_dir: Path = Path("plates")


def load_config(path: Path | None = None) -> PackConfig:
    """Missing file or missing keys fall back to defaults; unknown keys ignored."""
    cfg_path = Path(path) if path else Path("config.toml")
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    printer = data.get("printer", {})
    packing = data.get("packing", {})
    paths = data.get("paths", {})
    cfg = PackConfig(
        plate_mm=tuple(printer.get("plate_mm", PackConfig.plate_mm)),
        build_height_mm=float(printer.get("build_height_mm", PackConfig.build_height_mm)),
        working_res_mm=float(packing.get("working_res_mm", PackConfig.working_res_mm)),
        spacing_mm=float(packing.get("spacing_mm", PackConfig.spacing_mm)),
        edge_margin_mm=float(packing.get("edge_margin_mm", PackConfig.edge_margin_mm)),
        rotations=int(packing.get("rotations", PackConfig.rotations)),
        footprints_dir=Path(paths.get("footprints_dir", PackConfig.footprints_dir)),
        output_dir=Path(paths.get("output_dir", PackConfig.output_dir)),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: PackConfig) -> None:
    if len(cfg.plate_mm) != 2 or any(v <= 0 for v in cfg.plate_mm):
        raise ValueError("printer.plate_mm must be two positive numbers")
    if cfg.build_height_mm <= 0:
        raise ValueError("printer.build_height_mm must be positive")
    if cfg.working_res_mm <= 0:
        raise ValueError("packing.working_res_mm must be positive")
    ratio = cfg.working_res_mm / CANONICAL_RES_MM
    if abs(ratio - round(ratio)) > _RATIO_TOL or ratio < 1:
        raise ValueError(
            f"packing.working_res_mm {cfg.working_res_mm} must be an integer multiple "
            f"of canonical res {CANONICAL_RES_MM}"
        )
    if cfg.spacing_mm < 0:
        raise ValueError("packing.spacing_mm must be >= 0")
    if cfg.edge_margin_mm < 0:
        raise ValueError("packing.edge_margin_mm must be >= 0")
    if cfg.rotations < 1:
        raise ValueError("packing.rotations must be >= 1")

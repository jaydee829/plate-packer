"""plate-packer CLI (typer). Subcommands: footprints (generate cache docs), pack."""

import math
from pathlib import Path

import numpy as np
import typer

from plate_packer.config import load_config
from plate_packer.export import (
    export_plates,
    load_piece_mesh,
    placement_transform,
    verify_plate,
)
from plate_packer.footprint import extract_footprint
from plate_packer.footprint_io import (
    CANONICAL_RES_MM,
    file_sha256,
    has_current_doc,
    load_doc,
    save_doc,
)
from plate_packer.loading import prepare_mask
from plate_packer.packer import legal_placement_map, pack, rotate_mask

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """plate-packer: pack pre-supported resin models onto build plates."""


_EXTENSIONS = {".stl", ".obj"}


def _discover(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(q for q in sorted(p.rglob("*")) if q.suffix.lower() in _EXTENSIONS)
        elif p.suffix.lower() in _EXTENSIONS:
            files.append(p)
    # Dedupe while preserving order, using resolved paths
    return list(dict.fromkeys(f.resolve() for f in files))


@app.command()
def footprints(
    paths: list[Path] = typer.Argument(..., exists=True),  # noqa: B008
    footprints_dir: Path = typer.Option(  # noqa: B008
        None, help="cache dir (default: config.toml or ./footprints)"
    ),
    force: bool = typer.Option(False, help="regenerate even if a current doc exists"),
):
    """Generate footprint cache documents for STL/OBJ files."""
    out_dir = footprints_dir or load_config(None).footprints_dir
    files = _discover(paths)
    if not files:
        typer.echo("no STL/OBJ files found")
        return
    written = skipped = 0
    failures: list[tuple[Path, str]] = []
    for f in files:
        try:
            sha = file_sha256(f)
            if not force and has_current_doc(out_dir, sha):
                skipped += 1
                continue
            mask, origin, stats = extract_footprint(load_piece_mesh(f), CANONICAL_RES_MM)
            save_doc(out_dir, sha, mask, origin, stats, res_mm_per_px=CANONICAL_RES_MM)
            written += 1
            typer.echo(f"  {f.name}: ok ({stats['mask_px'][1]}x{stats['mask_px'][0]}px)")
        except Exception as e:  # per-file failures never halt the batch
            failures.append((f, str(e)))
            typer.echo(f"  {f.name}: FAILED ({e})")
    typer.echo(f"{written} written, {skipped} skipped, {len(failures)} failed -> {out_dir}")
    if failures:
        raise typer.Exit(code=1)


@app.command("pack")
def pack_command(
    paths: list[Path] = typer.Argument(..., exists=True),  # noqa: B008
    config: Path = typer.Option(None, help="config TOML (default ./config.toml)"),  # noqa: B008
    out: Path = typer.Option(None, help="output dir (default: config output_dir)"),  # noqa: B008
    footprints_dir: Path = typer.Option(None, help="cache dir (default: config)"),  # noqa: B008
    no_verify: bool = typer.Option(False, "--no-verify", help="skip merged-shadow self-check"),
    force: bool = typer.Option(False, help="regenerate footprint docs even if current"),
):
    """Pack STL/OBJ files onto build plates and export one merged STL per plate."""
    cfg = load_config(config)
    fp_dir = footprints_dir or cfg.footprints_dir
    out_dir = out or cfg.output_dir
    res = cfg.working_res_mm
    files = _discover(paths)
    if not files:
        typer.echo("no STL/OBJ files found")
        return

    # Stage 1+2: ensure cache docs, prepare masks, validate. Collect ALL errors.
    plate_shape = (round(cfg.plate_mm[1] / res), round(cfg.plate_mm[0] / res))
    plate_mask = np.zeros(plate_shape, np.uint8)
    m_px = math.ceil(cfg.edge_margin_mm / res) if cfg.edge_margin_mm > 0 else 0
    if m_px:
        plate_mask[:m_px, :] = plate_mask[-m_px:, :] = 1
        plate_mask[:, :m_px] = plate_mask[:, -m_px:] = 1
    angles = [i * 360.0 / cfg.rotations for i in range(cfg.rotations)]

    errors: list[tuple[Path, str]] = []
    piece_files: list[Path] = []
    masks: list[np.ndarray] = []
    origins: list[tuple[float, float]] = []
    areas: list[int] = []
    for f in files:
        try:
            sha = file_sha256(f)
            if force or not has_current_doc(fp_dir, sha):
                mask, origin, stats = extract_footprint(load_piece_mesh(f), CANONICAL_RES_MM)
                save_doc(fp_dir, sha, mask, origin, stats, res_mm_per_px=CANONICAL_RES_MM)
            doc = load_doc(fp_dir, sha)
            if doc.z_height_mm > cfg.build_height_mm:
                errors.append(
                    (
                        f,
                        f"height {doc.z_height_mm:.1f}mm exceeds build height "
                        f"{cfg.build_height_mm:.1f}mm",
                    )
                )
                continue
            mask, origin = prepare_mask(doc, cfg.spacing_mm, res)
            fits = any(
                rm.shape[0] <= plate_shape[0]
                and rm.shape[1] <= plate_shape[1]
                and legal_placement_map(plate_mask, rm).any()
                for rm in (rotate_mask(mask, a)[0] for a in angles)
            )
            if not fits:
                errors.append((f, "does not fit an empty plate at any rotation"))
                continue
            undilated, _ = prepare_mask(doc, 0.0, res)
            piece_files.append(f)
            masks.append(mask)
            origins.append(origin)
            areas.append(int(undilated.sum()))
        except Exception as e:
            errors.append((f, str(e)))
    if errors:
        typer.echo(f"{len(errors)} piece(s) cannot be packed:")
        for f, msg in errors:
            typer.echo(f"  {f.name}: {msg}")
        raise typer.Exit(code=1)

    # Stage 3: pack (placements come back sorted by piece index).
    placements = pack(masks, plate_shape, rotations=cfg.rotations, plate_mask=plate_mask)

    # Stage 4: exact transforms + export.
    transforms = []
    for pl in placements:
        _, aff = rotate_mask(masks[pl.piece], pl.angle)
        transforms.append(
            placement_transform(origins[pl.piece], aff, pl.row, pl.col, res, cfg.plate_mm)
        )
    plate_files = export_plates(piece_files, placements, transforms, out_dir)

    # Stage 5: self-check (subset assertion per plate).
    if not no_verify:
        occs = [np.zeros(plate_shape, np.uint8) for _ in plate_files]
        for pl in placements:
            rm, _ = rotate_mask(masks[pl.piece], pl.angle)
            occs[pl.plate][pl.row : pl.row + rm.shape[0], pl.col : pl.col + rm.shape[1]] |= rm
        failed = 0
        for path, occ in zip(plate_files, occs, strict=True):
            n = verify_plate(load_piece_mesh(path), occ, res, cfg.plate_mm, cfg.spacing_mm)
            status = "ok" if n == 0 else f"FAILED ({n} px outside prediction)"
            typer.echo(f"  verify {path.name}: {status}")
            failed += n > 0
        if failed:
            typer.echo(f"self-check failed on {failed} plate(s); output kept for inspection")
            raise typer.Exit(code=1)

    # Stage 6: report.
    usable = plate_shape[0] * plate_shape[1] - int(plate_mask.sum())
    lines = []
    for idx, path in enumerate(plate_files):
        on_plate = [pl for pl in placements if pl.plate == idx]
        pct = sum(areas[pl.piece] for pl in on_plate) / usable
        lines.append(f"{path.name}: {len(on_plate)} pieces, {pct:.1%} occupied")
        for pl in on_plate:
            t4 = transforms[pl.piece]
            rm, _ = rotate_mask(masks[pl.piece], pl.angle)
            rr, cc = np.nonzero(rm)
            cx = (pl.col + cc.mean()) * res - cfg.plate_mm[0] / 2
            cy = (pl.row + rr.mean()) * res - cfg.plate_mm[1] / 2
            rot = math.degrees(math.atan2(t4[1, 0], t4[0, 0])) % 360
            lines.append(
                f"  {piece_files[pl.piece].name}  x={cx:+.1f}mm  y={cy:+.1f}mm  rot={rot:.1f}deg"
            )
    lines.append(f"{len(piece_files)} pieces -> {len(plate_files)} plate(s)")
    report = "\n".join(lines)
    typer.echo(report)
    (Path(out_dir) / "report.txt").write_text(report + "\n", encoding="utf-8")

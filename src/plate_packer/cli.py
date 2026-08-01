"""plate-packer CLI (typer). First subcommand: footprints (generate cache docs)."""

import tomllib
from pathlib import Path

import trimesh
import typer

from plate_packer.footprint import extract_footprint
from plate_packer.footprint_io import (
    CANONICAL_RES_MM,
    file_sha256,
    has_current_doc,
    save_doc,
)

app = typer.Typer(no_args_is_help=True)

_EXTENSIONS = {".stl", ".obj"}


def _default_footprints_dir() -> Path:
    cfg = Path("config.toml")
    if cfg.exists():
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        value = data.get("paths", {}).get("footprints_dir")
        if value:
            return Path(value)
    return Path("footprints")


def _discover(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(q for q in sorted(p.rglob("*")) if q.suffix.lower() in _EXTENSIONS)
        elif p.suffix.lower() in _EXTENSIONS:
            files.append(p)
    return files


@app.command()
def footprints(
    paths: list[Path] = typer.Argument(..., exists=True),  # noqa: B008
    footprints_dir: Path = typer.Option(  # noqa: B008
        None, help="cache dir (default: config.toml or ./footprints)"
    ),
    force: bool = typer.Option(False, help="regenerate even if a current doc exists"),
):
    """Generate footprint cache documents for STL/OBJ files."""
    out_dir = footprints_dir or _default_footprints_dir()
    written = skipped = 0
    failures: list[tuple[Path, str]] = []
    for f in _discover(paths):
        try:
            sha = file_sha256(f)
            if not force and has_current_doc(out_dir, sha):
                skipped += 1
                continue
            mesh = trimesh.load_mesh(f, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            mask, origin, stats = extract_footprint(mesh, CANONICAL_RES_MM)
            save_doc(out_dir, sha, mask, origin, stats)
            written += 1
            typer.echo(f"  {f.name}: ok ({stats['mask_px'][1]}x{stats['mask_px'][0]}px)")
        except Exception as e:  # per-file failures never halt the batch
            failures.append((f, str(e)))
            typer.echo(f"  {f.name}: FAILED ({e})")
    typer.echo(f"{written} written, {skipped} skipped, {len(failures)} failed -> {out_dir}")
    if failures:
        raise typer.Exit(code=1)

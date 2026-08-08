"""Footprint cache documents: the stl_curator interface contract (ADR-009).

This is the ONLY module that knows the contract: content-addressed paths
(<dir>/<sha[:2]>/<sha>.json), the versioned JSON schema, and atomic writes.
Docs hold intrinsic data only — masks are undilated, at canonical resolution.
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

CANONICAL_RES_MM = 0.05
# Float tolerance for "working res is an integer multiple of canonical res" checks.
RES_RATIO_TOL = 1e-6
SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA = {1, 2}
_GENERATOR = f"plate-packer {version('plate-packer')}"


@dataclass(frozen=True)
class FootprintDoc:
    sha: str
    res_mm_per_px: float
    origin_mm: tuple[float, float]
    z_height_mm: float
    triangles: int
    dropped_nonfinite: int
    masks: list  # list[np.ndarray]; masks[0] is always full_shadow
    body_mask: object = None  # np.ndarray | None (model_body), None if absent
    cut_z_mm: float | None = None
    detector_version: int | None = None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def doc_path(footprints_dir: Path, sha: str) -> Path:
    return Path(footprints_dir) / sha[:2] / f"{sha}.json"


def _png_b64(mask) -> str:
    ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def save_doc(
    footprints_dir,
    sha,
    mask,
    origin_mm,
    stats,
    res_mm_per_px=CANONICAL_RES_MM,
    body_mask=None,
    cut_z_mm=None,
    detector_version=None,
) -> Path:
    footprints = [{"kind": "full_shadow", "z_band_mm": [0.0, None], "mask_png_b64": _png_b64(mask)}]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "stl_sha256": sha,
        "res_mm_per_px": res_mm_per_px,
        "origin_mm": [float(origin_mm[0]), float(origin_mm[1])],
        "z_height_mm": stats["z_height_mm"],
        "triangles": stats["triangles"],
        "dropped_nonfinite": stats["dropped_nonfinite"],
        "footprints": footprints,
    }
    if body_mask is not None:
        footprints.append(
            {
                "kind": "model_body",
                "z_band_mm": [float(cut_z_mm) if cut_z_mm is not None else 0.0, None],
                "mask_png_b64": _png_b64(body_mask),
            }
        )
        doc["cut_z_mm"] = float(cut_z_mm) if cut_z_mm is not None else 0.0
        doc["detector_version"] = detector_version
    path = doc_path(footprints_dir, sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)  # readers never see a torn doc
    return path


def _decode_mask(fp, sha) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(fp["mask_png_b64"]), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"corrupt mask PNG in doc {sha}")
    return (img > 0).astype(np.uint8)


def load_doc(footprints_dir, sha) -> FootprintDoc:
    raw = json.loads(doc_path(footprints_dir, sha).read_text(encoding="utf-8"))
    if raw.get("schema_version") not in _SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    full_masks, body_mask, cut_z_mm = [], None, None
    for fp in raw["footprints"]:
        if fp.get("kind") == "model_body":
            body_mask = _decode_mask(fp, raw["stl_sha256"])
            cut_z_mm = fp["z_band_mm"][0]
        else:
            full_masks.append(_decode_mask(fp, raw["stl_sha256"]))
    return FootprintDoc(
        sha=raw["stl_sha256"],
        res_mm_per_px=raw["res_mm_per_px"],
        origin_mm=tuple(raw["origin_mm"]),
        z_height_mm=raw["z_height_mm"],
        triangles=raw["triangles"],
        dropped_nonfinite=raw["dropped_nonfinite"],
        masks=full_masks,
        body_mask=body_mask,
        cut_z_mm=cut_z_mm,
        detector_version=raw.get("detector_version"),
    )


def has_current_doc(footprints_dir, sha) -> bool:
    p = doc_path(footprints_dir, sha)
    if not p.exists():
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # ValueError covers both json.JSONDecodeError and UnicodeDecodeError
        return False
    return raw.get("schema_version") in _SUPPORTED_SCHEMA

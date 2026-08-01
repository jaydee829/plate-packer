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
from pathlib import Path

import cv2
import numpy as np

CANONICAL_RES_MM = 0.05
SCHEMA_VERSION = 1
_GENERATOR = "plate-packer 0.1.0"


@dataclass(frozen=True)
class FootprintDoc:
    sha: str
    res_mm_per_px: float
    origin_mm: tuple[float, float]
    z_height_mm: float
    triangles: int
    dropped_nonfinite: int
    masks: list  # list[np.ndarray], uint8 {0,1}; v1 always length 1


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def doc_path(footprints_dir: Path, sha: str) -> Path:
    return Path(footprints_dir) / sha[:2] / f"{sha}.json"


def save_doc(footprints_dir, sha, mask, origin_mm, stats) -> Path:
    ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "stl_sha256": sha,
        "res_mm_per_px": CANONICAL_RES_MM,
        "origin_mm": [float(origin_mm[0]), float(origin_mm[1])],
        "z_height_mm": stats["z_height_mm"],
        "triangles": stats["triangles"],
        "dropped_nonfinite": stats["dropped_nonfinite"],
        "footprints": [
            {
                "kind": "full_shadow",
                "z_band_mm": [0.0, None],
                "mask_png_b64": base64.b64encode(png.tobytes()).decode(),
            }
        ],
    }
    path = doc_path(footprints_dir, sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)  # readers never see a torn doc
    return path


def load_doc(footprints_dir, sha) -> FootprintDoc:
    raw = json.loads(doc_path(footprints_dir, sha).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    masks = []
    for fp in raw["footprints"]:
        buf = np.frombuffer(base64.b64decode(fp["mask_png_b64"]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        masks.append((img > 0).astype(np.uint8))
    return FootprintDoc(
        sha=raw["stl_sha256"],
        res_mm_per_px=raw["res_mm_per_px"],
        origin_mm=tuple(raw["origin_mm"]),
        z_height_mm=raw["z_height_mm"],
        triangles=raw["triangles"],
        dropped_nonfinite=raw["dropped_nonfinite"],
        masks=masks,
    )


def has_current_doc(footprints_dir, sha) -> bool:
    p = doc_path(footprints_dir, sha)
    if not p.exists():
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return raw.get("schema_version") == SCHEMA_VERSION

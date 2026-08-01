"""Contract-path, round-trip, and curator-opacity tests for footprint docs."""

import base64
import json

import numpy as np
import pytest

from plate_packer.footprint_io import (
    CANONICAL_RES_MM,
    SCHEMA_VERSION,
    doc_path,
    file_sha256,
    has_current_doc,
    load_doc,
    save_doc,
)

SHA_A = "a" * 64
STATS = {
    "triangles": 12,
    "dropped_nonfinite": 0,
    "z_height_mm": 5.0,
    "footprint_mm": (20.0, 10.0),
    "mask_px": (101, 201),
    "coverage": 0.99,
    "raster_s": 0.01,
}


def checker_mask():
    mask = np.zeros((8, 6), np.uint8)
    mask[::2, ::2] = 1
    return mask


@pytest.mark.parametrize(
    ("sha", "expected_parts"),
    [
        pytest.param("ab" + "0" * 62, ("ab", "ab" + "0" * 62 + ".json"), id="ab-prefix"),
        pytest.param("7f" + "e" * 62, ("7f", "7f" + "e" * 62 + ".json"), id="7f-prefix"),
    ],
)
def test_doc_path_follows_contract(tmp_path, sha, expected_parts):
    p = doc_path(tmp_path, sha)
    assert p == tmp_path / expected_parts[0] / expected_parts[1]


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "x.stl"
    f.write_bytes(b"solid x" * 1000)
    assert file_sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_round_trip_preserves_mask_and_metadata(tmp_path):
    mask = checker_mask()
    save_doc(tmp_path, SHA_A, mask, (-10.0, -5.0), STATS)
    doc = load_doc(tmp_path, SHA_A)
    assert doc.sha == SHA_A
    assert doc.res_mm_per_px == CANONICAL_RES_MM
    assert doc.origin_mm == (-10.0, -5.0)
    assert doc.z_height_mm == 5.0
    assert doc.triangles == 12
    assert doc.dropped_nonfinite == 0
    assert len(doc.masks) == 1
    assert (doc.masks[0] == mask).all()


def test_doc_is_plain_json_per_curator_contract(tmp_path):
    """The curator must be able to treat the doc as opaque JSON."""
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    raw = json.loads(doc_path(tmp_path, SHA_A).read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["stl_sha256"] == SHA_A
    assert raw["res_mm_per_px"] == CANONICAL_RES_MM
    assert raw["footprints"][0]["kind"] == "full_shadow"
    assert raw["footprints"][0]["z_band_mm"] == [0.0, None]
    base64.b64decode(raw["footprints"][0]["mask_png_b64"], validate=True)


def test_no_tmp_file_left_behind(tmp_path):
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        pytest.param("none", False, id="absent"),
        pytest.param("current", True, id="current-version"),
        pytest.param("stale", False, id="stale-version-treated-as-absent"),
        pytest.param("garbage-bytes", False, id="garbage-bytes-treated-as-absent"),
    ],
)
def test_has_current_doc(tmp_path, setup, expected):
    if setup in ("current", "stale", "garbage-bytes"):
        if setup != "garbage-bytes":
            save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
        else:
            # Write raw invalid UTF-8 bytes to simulate corrupt doc
            p = doc_path(tmp_path, SHA_A)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\xff\xfe garbage \xff")
    if setup == "stale":
        p = doc_path(tmp_path, SHA_A)
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["schema_version"] = 0
        p.write_text(json.dumps(raw), encoding="utf-8")
    assert has_current_doc(tmp_path, SHA_A) is expected


def test_load_doc_stale_version_raises(tmp_path):
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    p = doc_path(tmp_path, SHA_A)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_doc(tmp_path, SHA_A)


def test_load_doc_corrupt_png_raises(tmp_path):
    """load_doc should raise ValueError on undecodable PNG data."""
    save_doc(tmp_path, SHA_A, checker_mask(), (0.0, 0.0), STATS)
    p = doc_path(tmp_path, SHA_A)
    raw = json.loads(p.read_text(encoding="utf-8"))
    # Corrupt the mask_png_b64 by setting it to valid base64 of garbage bytes
    raw["footprints"][0]["mask_png_b64"] = base64.b64encode(b"not a png").decode()
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt mask PNG"):
        load_doc(tmp_path, SHA_A)

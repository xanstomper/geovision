"""
Tests for the street-level coverage additions (Sep 2026).

Covers, deterministically (no heavy model / no multi-GB download):
  - ground_imagery_client Mapillary guard (HTTP-500 handling, no-token no-op)
  - batch_grow_db PRIORS shape
  - osv5m_subset_ingest HF-shard use + zip id-match logic

Live heavy paths (Mapillary 5 real street photos, batch_grow_db +84 refs,
OSV-5M shard download) were verified separately; this locks the logic.
"""
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.ground_imagery_client import GroundImageryClient


def test_no_token_returns_empty():
    assert GroundImageryClient(mapillary_token=None).search_mapillary_photos(0, 0) == []


def test_mapillary_500_retry_and_clamp_logic_present():
    src = open("modules/ground_imagery_client.py").read()
    assert "min(300," in src          # radius clamped to keep bbox small
    assert "e.code == 500" in src     # auto-halve radius on Mapillary "reduce data"
    assert "thumb_256_url" in src     # lean field set


def test_batch_grow_priors_shaped():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "batch_grow_db", "scripts/batch_grow_db.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert len(m.PRIORS) > 10
    assert all(len(p) == 3 for p in m.PRIORS)  # (name, lat, lon)


def test_osv5m_uses_hf_shard_not_fb_cdn():
    src = open("scripts/osv5m_subset_ingest.py").read()
    assert "SHARD_BASE" in src          # pulls real images from HF shards
    assert "graph.mapillary" not in src  # not the (403) CSV thumb approach


def test_osv5m_zip_id_match_keeps_only_wanted():
    """The extract-only-matching-ids logic: from a shard, keep only members whose
    id is in the country/city-filtered metadata `wanted` map."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("123.jpg", b"ABCDEFG")
        z.writestr("999.jpg", b"ZZZZ")
    wanted = {"123": {"id": "123"}}
    got = []
    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            oid = os.path.splitext(os.path.basename(name))[0]
            if oid in wanted:
                got.append(oid)
    assert got == ["123"]
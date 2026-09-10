"""
Tests for the bounded/clue-driven agent path (no internal API keys).

Verifies that a model with computer vision can drive GeoVision's clue tools
robustly: resolve_vision_clues returns promptly under a tiny budget (never
blocks the caller), degrades to partial offline grounding, and the MCP tool
registers. Also asserts GeoVision ships zero internal API-key literals.
"""
import os
import re
import time
import importlib

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _scan_no_real_keys():
    """Return any real-key literals found in source files (sk-/AIza/ghp_...)."""
    key_re = re.compile(
        r"(sk-[A-Za-z0-9]{12,}|AIza[0-9A-Za-z_-]{25,}|ghp_[A-Za-z0-9]{20,}"
        r"|ya29\.[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{25,})")
    hits = []
    for dp, _, fs in os.walk(ROOT):
        if any(x in dp for x in ("node_modules", ".git", "__pycache__", os.sep + "data" + os.sep)):
            continue
        for f in fs:
            if not f.endswith((".py", ".js", ".json", ".yaml", ".yml", ".toml", ".sh")):
                continue
            p = os.path.join(dp, f)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            hits += [f"{p}:{m.group(0)[:20]}..." for m in key_re.finditer(txt)]
    return hits


def test_no_internal_api_keys():
    assert _scan_no_real_keys() == []


def test_resolve_vision_clues_is_bounded():
    """A slow public API must not block the caller: tiny budget returns promptly."""
    import mcp_geovision_server as mcp
    t0 = time.time()
    out = mcp.tool_resolve_vision_clues({"city_hint": "Kyoto", "budget_seconds": 2.0})
    dt = time.time() - t0
    assert dt <= 15, f"resolve_vision_clues blocked caller for {dt:.1f}s"
    assert out.get("status") in ("partial", "limited", "success")


def test_resolve_vision_clues_registered():
    import mcp_geovision_server as mcp
    names = [t["name"] for t in mcp.TOOLS]
    assert "resolve_vision_clues" in names


def test_cv_model_clue_loop_returns_grounding():
    """A CV model sees an image, extracts clues (OCR text, chain store, side),
    feeds them to resolve_vision_clues, and gets a structured result — with
    zero keys, no hang. Conservative budget gives partial/success, not failure.
    """
    import mcp_geovision_server as mcp
    clues = {
        "city_hint": "Paris",
        "detected_text": ["Louvre museum"],
        "chain_stores": ["McDonald's"],
        "driving_side": "right",
        "budget_seconds": 12.0,
    }
    t0 = time.time()
    out = mcp.tool_resolve_vision_clues(clues)
    dt = time.time() - t0
    assert dt <= 20, f"clue loop blocked caller for {dt:.1f}s"
    assert out.get("status") in ("success", "partial", "limited")
    # structured output the model can act on: either a status + note, or coords
    assert isinstance(out, dict)
    assert "status" in out


def test_verification_exposes_reference_image_urls_to_cv_model():
    """The verification output a vision model receives must include the REAL
    ground-truth reference image URLs, so the model can look at them itself."""
    from modules.geo_harness import GeoVisionHarness
    cand = [{"latitude": 48.8584, "longitude": 2.2945, "confidence": 0.5,
             "source": "t", "city": "Paris", "country": "FR"}]
    out = GeoVisionHarness().visual_verify_candidates(
        "data/eval/wikipedia_landmarks_v1/images/eiffel_tower.jpg", cand, top_k=1)
    vs = out.get("verifications", [])
    assert len(vs) == 1
    v = vs[0]
    # reference_urls or reference_photos[].thumbnail_url must be real, fetchable image URLs
    urls = v.get("reference_urls") or [p.get("thumbnail_url") for p in v.get("reference_photos", [])]
    assert len(urls) > 0
    assert all(u and u.startswith("http") for u in urls)
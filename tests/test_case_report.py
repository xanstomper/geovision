"""
Tests for the GeoVision case-report generation (Sep 2026).

Verifies the report renderer deterministically:
  - markdown + html render from a representative investigation record
  - honest confidence: low/unknown confidence flagged; bands correct
  - full evidence chain, constraints, candidates, reference imagery sections present
  - renders an empty/limited record without raising
  - writes to a file path

The renderer is model-agnostic (renders whatever dict the harness returns).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.case_report import (render_markdown, render_html, render_case,
                                 _confidence_band)

DEMO = {
    "status": "success", "case_id": "OP-1", "image": "x.jpg",
    "best_estimate": {"latitude": 48.8584, "longitude": 2.2945,
                      "confidence": 0.88, "city": "Paris", "country": "FR",
                      "place_name": "Eiffel Tower"},
    "uncertainty": {"granularity": "city", "radius_km": 8, "note": "top-3 spread small"},
    "reasoning_chain": ["COARSE: urban tower", "RETRIEVAL: regional match (0.92)"],
    "constraints_applied": [{"name": "hemisphere", "value": "north", "note": "excludes south"}],
    "candidates": [{"latitude": 48.8584, "longitude": 2.2945, "confidence": 0.88,
                    "city": "Paris"}],
    "reference_urls_by_candidate": {"Paris": [{"url": "https://e.com/x.jpg", "name": "ref"}]},
    "stages": {"regional_retrieval": {"status": "success", "summary": "restricted to FR"}},
    "agent_hint": "cross-check with vision",
}


def test_confidence_bands():
    assert _confidence_band(0.9) == "high"
    assert _confidence_band(0.7) == "medium-high"
    assert _confidence_band(0.5) == "medium"
    assert _confidence_band(0.3) == "low"
    assert _confidence_band(0.05) == "very low"
    assert _confidence_band(None) == "unknown"
    assert _confidence_band(0) == "unknown"


def test_markdown_has_evidence_sections():
    md = render_markdown(DEMO)
    assert "## Summary" in md
    assert "48.85840" in md            # best estimate coords
    assert "Eiffel Tower" in md
    assert "## Reasoning chain (evidence)" in md
    assert "COARSE: urban tower" in md
    assert "## Constraints that survived elimination" in md
    assert "Hemisphere" in md
    assert "## Ranked candidate locations" in md
    assert "## Sources & methodology" in md


def test_markdown_honest_confidence_flag():
    md = render_markdown(DEMO)
    assert "uncalibrated" in md.lower()          # prior flagged, not fake-probability
    low = dict(DEMO)
    low["best_estimate"] = dict(DEMO["best_estimate"], confidence=0.05)
    assert "very low" in render_markdown(low)


def test_html_render_no_raise_and_escapes():
    h = render_html(DEMO)
    assert h.startswith("<!DOCTYPE html>")
    assert "Eiffel Tower" in h
    evil = dict(DEMO, reasoning_chain=["<script>alert(1)</script>"])
    assert "<script>alert" not in render_html(evil)   # escaped


def test_render_case_writes_files():
    with tempfile.TemporaryDirectory() as d:
        md = render_case(DEMO, f"{d}/c.md")
        html = render_case(DEMO, f"{d}/c.html")
        assert md.exists() and md.stat().st_size > 100
        assert html.exists() and html.stat().st_size > 100
        assert md.read_text().startswith("#")


def test_empty_record_renders_without_raise():
    md = render_markdown({"status": "limited", "image": "x"})
    assert "No confident estimate" in md
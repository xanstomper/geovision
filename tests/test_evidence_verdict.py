"""
Tests for the OceanIR-style evidence verdict (build_evidence_verdict).

Verifies the structured supports[]/contradictions[]/precision_tier/
verification_status/scope_consistent shape derived from an investigation record —
OceanIR's 'documented chain, not a pin' result model.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.geo_harness import GeoVisionHarness


def _rec(conf=0.85, hint="Rome", stages=("success", {"e": 1}, "success")):
    return {
        "best_estimate": {"latitude": 48.85, "longitude": 2.29, "confidence": conf,
                          "city": "Paris", "country": "FR", "place_name": "Eiffel Tower"},
        "reasoning_chain": ["COARSE: urban tower", "RETRIEVAL: Paris region matched"],
        "constraints_applied": [{"name": "road_line_color", "value": "yellow"},
                                {"name": "hemisphere", "value": "north"}],
        "notes": ["Eliminated (VLM): Tokyo"],
        "candidates": [{"latitude": 48.85, "longitude": 2.29, "confidence": conf},
                       {"latitude": 41.89, "longitude": 12.51, "confidence": 0.2}],
        "stages": {"deterministic_scan": {"status": stages[0]},
                   "regional_retrieval": {"estimates": stages[1]},
                   "visual_verify_candidates": {"status": stages[2]}},
        "location_hint": hint,
    }


def test_verdict_shape_and_precision_tier():
    v = GeoVisionHarness().build_evidence_verdict(_rec(0.85))
    for k in ("confidence", "precision_tier", "verification_status",
              "scope_consistent", "supports", "contradictions"):
        assert k in v
    assert v["precision_tier"] == "site_level"
    assert isinstance(v["supports"], list) and v["supports"]
    assert isinstance(v["contradictions"], list)


def test_verdict_precision_tiers():
    g = GeoVisionHarness()
    assert g.build_evidence_verdict(_rec(0.95))["precision_tier"] == "exact"
    assert g.build_evidence_verdict(_rec(0.80))["precision_tier"] == "site_level"
    assert g.build_evidence_verdict(_rec(0.60))["precision_tier"] == "neighborhood_level"
    assert g.build_evidence_verdict(_rec(0.40))["precision_tier"] == "city_level"
    assert g.build_evidence_verdict(_rec(0.10))["precision_tier"] == "region_level"


def test_verdict_corroborated_status():
    g = GeoVisionHarness()
    # 3 healthy stages -> corroborated
    assert g.build_evidence_verdict(_rec(0.8))["verification_status"] == "corroborated"
    # only 1 healthy stage -> partially
    assert g.build_evidence_verdict(
        _rec(0.8, stages=("success", {}, "failed")))["verification_status"] == "partially_corroborated"
    # none -> unverified
    assert g.build_evidence_verdict(
        _rec(0.4, stages=("failed", {}, "failed")))["verification_status"] == "unverified"


def test_verdict_scope_consistent():
    g = GeoVisionHarness()
    # hint 'Rome' not in Paris answer -> scope_consistent False (scene overrides hint)
    assert g.build_evidence_verdict(_rec(hint="Rome"))["scope_consistent"] is False
    # hint 'Paris' in answer -> consistent
    assert g.build_evidence_verdict(_rec(hint="Paris"))["scope_consistent"] is True
    # no hint -> consistent
    assert g.build_evidence_verdict(_rec(hint=None))["scope_consistent"] is True
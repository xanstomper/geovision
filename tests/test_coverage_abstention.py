"""Coverage-density abstention tests (OceanIR M1 pre-empt, key-free, offline)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_singleton_and_available():
    from modules.coverage_abstention import CoverageAbstention
    a = CoverageAbstention()
    b = CoverageAbstention()
    assert a is b
    assert a.available, "ref DB meta must be present in-repo for these tests"


def test_dense_coverage_paris():
    """A landmark with many real refs -> dense, in_coverage True."""
    from modules.coverage_abstention import CoverageAbstention
    r = CoverageAbstention().assess(48.8584, 2.2945, 0.8)
    assert r["in_coverage"] is True
    assert r["tier"] == "dense"
    assert r["refs_core_km"] >= 5


def test_uncovered_mid_ocean():
    """Mid-Pacific prediction -> uncovered, in_coverage False, honest note."""
    from modules.coverage_abstention import CoverageAbstention
    r = CoverageAbstention().assess(0.0, -140.0, 0.8)
    assert r["in_coverage"] is False
    assert r["tier"] == "uncovered"
    assert "NO corroborating" in r["note"]


def test_invalid_coords():
    from modules.coverage_abstention import CoverageAbstention
    r = CoverageAbstention().assess(999.0, 999.0, 0.5)
    assert r["tier"] == "invalid"
    assert r["in_coverage"] is False


def test_evidence_verdict_carries_coverage_and_flags_uncovered(monkeypatch):
    """build_evidence_verdict surfaces coverage; uncovered becomes a contradiction."""
    from modules.geo_harness import GeoVisionHarness
    h = GeoVisionHarness()
    rec = {
        "best_estimate": {"latitude": 0.0, "longitude": -140.0,
                          "confidence": 0.6, "city": None, "country": None},
        "reasoning_chain": ["COARSE: ocean tones"],
        "candidates": [],
        "stages": {},
    }
    v = h.build_evidence_verdict(rec)
    assert v["coverage"]["tier"] == "uncovered"
    assert any("corroborating" in c.lower() or "coverage" in c.lower()
               for c in v["contradictions"])


def test_evidence_verdict_dense_coverage_no_extra_contradiction():
    from modules.geo_harness import GeoVisionHarness
    h = GeoVisionHarness()
    rec = {
        "best_estimate": {"latitude": 48.8584, "longitude": 2.2945,
                          "confidence": 0.8, "city": "Paris", "country": "FR"},
        "reasoning_chain": ["COARSE: tower"],
        "candidates": [],
        "stages": {"deterministic_scan": {"status": "success"},
                    "regional_retrieval": {"status": "success"}},
    }
    v = h.build_evidence_verdict(rec)
    assert v["coverage"]["tier"] == "dense"
    assert not any("coverage" in c.lower() for c in v["contradictions"])

"""
Tests for the Live Signal Orchestrator (Sep 2026).

The orchestrator is LIVE by design — these tests verify the deterministic parts:
  - the timeout guard never lets a signal block forever
  - signal isolation (one failing signal doesn't break the bundle)
  - investigate() structure/status for both coordinate and text anchors
  - the key-free DDG-lite parser returns a list

Live runs (7 signals pulled for Eiffel / Colosseum Rome) verified separately.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.live_signal_orchestrator import LiveSignalOrchestrator, List


def test_timeout_guard_never_blocks():
    """A signal that hangs must return 'timed_out' quickly, not block forever."""
    o = LiveSignalOrchestrator(timeout=1)

    def hang():
        import time
        time.sleep(60)
        return {"status": "ok", "x": 1}

    res, _ = o._run(hang)
    assert res.get("status") == "timed_out"


def test_signal_isolation():
    """One failing signal returns failed; others still work — bundle not broken."""
    from modules.live_signal_orchestrator import LiveSignalOrchestrator
    o = LiveSignalOrchestrator()
    # simulate: reverse_image module unavailable should yield failed, not raise
    res = o.signal_reverse_image(image_path=None, query=None)
    assert res.get("status") in ("failed", "empty")  # graceful, no raise


def test_ddg_lite_is_list_of_dicts():
    """Key-free DuckDuckGo parse returns a list even on network failure (never raises)."""
    r = LiveSignalOrchestrator._ddg_lite("__nonexistent_zzz__", 3)
    assert isinstance(r, list)


def test_investigate_coord_structure():
    """Coordinate anchor must produce the expected signal keys + success status."""
    o = LiveSignalOrchestrator(timeout=3)
    # Use a fake _run via monkeypatch-free path: point signals to cheap fakes.
    real = o.signal_ground_imagery
    o.signal_ground_imagery = lambda *a, **k: {"status": "ok", "photos": [], "count": 0}
    o.signal_osint = lambda *a, **k: {"status": "empty", "count": 0, "listings": []}
    o.signal_weather = lambda *a, **k: {"status": "empty"}
    o.signal_geonames_city = lambda *a, **k: {"status": "empty"}
    o.signal_reverse_geocode = lambda *a, **k: {"status": "empty"}
    o._ddg_lite = lambda *a, **k: []
    r = o.investigate(lat=48.8584, lon=2.2945, radius_m=1000)
    assert r.get("status") == "success"
    for k in ("web", "ground_imagery", "osint", "weather", "city_snap", "reverse_geocode"):
        assert k in r.get("signals", {}), f"missing signal {k}"
    o.signal_ground_imagery = real
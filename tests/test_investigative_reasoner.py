"""
Tests for the Investigative Reasoner (Gemini-style exact-address loop).
=======================================================================

Covered (all offline unless marked live):
  1. Address harvesting from listing URLs (Zillow/Realtor/Redfin/Homes/Trulia)
     and from plain result text.
  2. US street-numbering deduction rules:
       R0_direct  — found address IS the image clue number
       R1_number_match — clue number parity/adjacency -> deduced target
       R2_adjacent_step — no clue number -> N±2 same-side lots
  3. Street canonicalization across listing-site spelling variants.
  4. Ensemble family registration (source='investigative' = own family,
     one vote per family, high weight).
  5. MCP + CLI wiring (registry integrity covered in test_mcp_registry.py).
  6. Key-free honesty: no VLM key -> Stage A skips cleanly, and the loop
     still runs end-to-end from agent-supplied clues.
  7. Live research path (marked, hits DDG + Nominatim for real).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.investigative_reasoner import (  # noqa: E402
    InvestigativeReasoner,
    _canon_street,
    _harvest_addresses,
    _street_core,
    CONF_DEDUCED_NUMBER_MATCH,
    CONF_DEDUCED_PARITY_ONLY,
)


# ---------------------------------------------------------------------- #
# address harvesting                                                      #
# ---------------------------------------------------------------------- #
def test_harvest_zillow_url():
    url = ("https://www.zillow.com/homedetails/147-W-Stratford-Pl-Madison-Heights-"
           "VA-24572/12345678_zpid/")
    assert _harvest_addresses(url, "") == ["147 W Stratford Pl"]


def test_harvest_realtor_redfin_homes_trulia_urls():
    urls = [
        "https://www.realtor.com/realestateandhomes-detail/147-W-Stratford-Pl_Madison-Heights_VA_24572_M59052-42361",
        "https://www.redfin.com/VA/Madison-Heights/147-W-Stratford-Pl-24572/home/194767974",
        "https://www.homes.com/property/147-W-Stratford-Pl-madison-heights-va/6j7enb582wgpe/",
        "https://www.trulia.com/home/147-w-stratford-pl-madison-heights-va-24572-123456",
    ]
    for u in urls:
        addrs = _harvest_addresses(u, "")
        assert addrs and addrs[0].startswith("147 "), u


def test_harvest_from_text_snippet():
    text = ("Property records show the ranch at 3171 Earley Farm Road, "
            "Madison Heights, VA 24572 was listed in 2024.")
    assert _harvest_addresses("", text) == ["3171 Earley Farm Road"]


def test_harvest_rejects_no_number():
    assert _harvest_addresses("https://www.zillow.com/homedetails/profile/", "") == []


# ---------------------------------------------------------------------- #
# street canonicalization                                                 #
# ---------------------------------------------------------------------- #
def test_canon_street_matches_variants():
    assert _canon_street("W Stratford Place") == _canon_street("W Stratford Pl")
    assert _canon_street("Earley Farm Road") == _canon_street("Earley Farm Rd")
    assert _street_core("West Stratford Place") == _street_core("Stratford Pl")


# ---------------------------------------------------------------------- #
# deduction rules                                                         #
# ---------------------------------------------------------------------- #
CLUES_151 = {
    "visible_numbers": ["151"],
    "high_leverage_features": ["rooftop solar panels on the neighbor"],
    "region_guess": "Madison Heights, VA",
}


def test_r1_number_match_deduces_target():
    r = InvestigativeReasoner()
    findings = [{"address": "147 W Stratford Place", "url": "x", "title": "t",
                 "query": "q"}]
    deduced = r.apply_numbering_deduction(CLUES_151, findings)
    rules = {d["rule"] for d in deduced}
    assert "R1_number_match" in rules
    target = [d for d in deduced if d["address"] == "151 W Stratford Place"]
    assert target, "should deduce 151 on the same street"
    assert "odd" in target[0]["reason"]  # parity explanation present


def test_r0_direct_when_found_is_clue_number():
    r = InvestigativeReasoner()
    clues = {"visible_numbers": ["3171"], "region_guess": "Madison Heights, VA"}
    findings = [{"address": "3171 Earley Farm Rd", "url": "x", "title": "t",
                 "query": "q"}]
    deduced = r.apply_numbering_deduction(clues, findings)
    assert deduced and deduced[0]["rule"] == "R0_direct"
    assert deduced[0]["address"] == "3171 Earley Farm Rd"


def test_r2_adjacent_step_without_clue_number():
    r = InvestigativeReasoner()
    clues = {"visible_numbers": [], "region_guess": "Madison Heights, VA"}
    findings = [{"address": "147 W Stratford Pl", "url": "x", "title": "t",
                 "query": "q"}]
    deduced = r.apply_numbering_deduction(clues, findings)
    addrs = {d["address"] for d in deduced}
    assert addrs == {"145 W Stratford Pl", "149 W Stratford Pl"}
    assert all(d["rule"] == "R2_adjacent_step" for d in deduced)


def test_no_false_match_across_parity():
    """Clue 152 (even) must NOT deduce from a found odd neighbor 147."""
    r = InvestigativeReasoner()
    clues = {"visible_numbers": ["152"], "region_guess": "X"}
    findings = [{"address": "147 W Stratford Pl", "url": "x", "title": "t",
                 "query": "q"}]
    deduced = r.apply_numbering_deduction(clues, findings)
    assert not [d for d in deduced if d["rule"] == "R1_number_match"]


# ---------------------------------------------------------------------- #
# end-to-end loop with mocked web + geocode (no network)                  #
# ---------------------------------------------------------------------- #
class _FakeOrch:
    def __init__(self, timeout=18):
        pass

    def signal_web_search(self, query, n=5):
        return {"status": "ok", "query": query, "results": [{
            "title": "147 W Stratford Pl, Madison Heights, VA 24572",
            "url": ("https://www.zillow.com/homedetails/147-W-Stratford-Pl-"
                    "Madison-Heights-VA-24572/12345678_zpid/"),
            "snippet": "Solar-equipped ranch home",
        }], "count": 1}


def _fake_geocode_factory():
    def _fake_geocode(addr):
        if "stratford" in addr.lower():
            n = addr.split()[0]
            if n.isdigit():  # house-level
                return {"latitude": 37.4564, "longitude": -79.0995,
                        "display_name": addr}
            return {"latitude": 37.4560, "longitude": -79.0990,  # street centroid
                    "display_name": addr}
        return None
    return _fake_geocode


def test_full_loop_web_fallback_mocks(monkeypatch):
    """No street_names in clues -> OSM-direct skips -> web fallback runs."""
    import modules.property_locator as pl
    monkeypatch.setattr(pl, "overpass_query", lambda q, timeout=40: [])
    r = InvestigativeReasoner()
    monkeypatch.setattr(r, "_searxng_rotate_search",
                        lambda q, n=6: ([], "fenced"))
    monkeypatch.setattr("modules.live_signal_orchestrator.LiveSignalOrchestrator",
                        _FakeOrch)
    monkeypatch.setattr("modules.listing_finder.geocode_address",
                        _fake_geocode_factory())

    clues = {"visible_numbers": ["151"],
             "high_leverage_features": ["rooftop solar panels on the neighbor"],
             "region_guess": "Madison Heights, VA"}
    rec = r.investigate(clues=clues, region_hint="Madison Heights, VA")
    assert rec["status"] == "success"
    assert rec["candidates"], "geocoded candidate must exist"
    top = rec["candidates"][0]
    assert top["source"] == "investigative"
    assert top["address"].startswith("151 ")
    assert top["confidence"] <= 0.85  # honest cap
    chain = " ".join(rec["reasoning_chain"])
    assert "DEDUCTION" in chain and "VERIFIED" in chain


def test_full_loop_osm_direct_mocks(monkeypatch):
    """street_names + numbers -> OSM-direct house geocode is the primary path
    and the web fallback never runs."""
    r = InvestigativeReasoner()
    web_called = []

    class _Spy(_FakeOrch):
        def signal_web_search(self, query, n=5):
            web_called.append(query)
            return super().signal_web_search(query, n)

    monkeypatch.setattr("modules.live_signal_orchestrator.LiveSignalOrchestrator",
                        _Spy)
    monkeypatch.setattr("modules.listing_finder.geocode_address",
                        _fake_geocode_factory())

    clues = {"visible_numbers": ["151"], "street_names": ["Stratford Pl"],
             "region_guess": "Madison Heights, VA"}
    rec = r.investigate(clues=clues, region_hint="Madison Heights, VA")
    assert rec["status"] == "success"
    assert not web_called, "OSM-direct hit must skip the web fallback"
    # R0_direct: the OSM parcel IS the clue number
    assert any(c["address"].startswith("151 ") for c in rec["candidates"])
    chain = " ".join(rec["reasoning_chain"])
    assert "RESEARCH[OSM-direct]" in chain


def test_skips_cleanly_without_clues_or_vlm(monkeypatch):
    """No VLM key + no agent clues -> honest skip, no crash, no candidates."""
    monkeypatch.delenv("GEOVISION_VLM_API_KEY", raising=False)
    r = InvestigativeReasoner()
    rec = r.investigate(image_path="/nonexistent.jpg")
    assert rec["status"] == "limited"
    assert rec["candidates"] == []
    assert "clues" in rec.get("note", "").lower() or rec.get("clues")


def test_no_region_means_no_research(monkeypatch):
    r = InvestigativeReasoner()
    rec = r.investigate(clues={"visible_numbers": ["5"]})
    assert rec["status"] == "limited"
    assert rec["candidates"] == []


# ---------------------------------------------------------------------- #
# ensemble family registration                                            #
# ---------------------------------------------------------------------- #
def test_investigative_is_own_ensemble_family():
    from modules.ensemble_fusion import FAMILY_NAMES, FAMILY_WEIGHT
    assert FAMILY_NAMES.get("investigative") == "investigative"
    assert FAMILY_WEIGHT.get("investigative", 0) >= 0.9


# ---------------------------------------------------------------------- #
# MCP tool wiring                                                         #
# ---------------------------------------------------------------------- #
def test_mcp_tool_registered():
    import mcp_geovision_server as mcp
    assert "investigative_locate" in mcp.TOOL_IMPLS
    names = [t["name"] for t in mcp.TOOLS]
    assert "investigative_locate" in names
    assert names.count("investigative_locate") == 1


def test_mcp_tool_requires_input():
    import mcp_geovision_server as mcp
    out = mcp.TOOL_IMPLS["investigative_locate"]({})
    assert "error" in out


# ---------------------------------------------------------------------- #
# Stage E — scene verification (mocked network)                           #
# ---------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code=200, content=b"\xff\xd8\xffJFIFfake"):
        self.status_code = status_code
        self.content = content


def test_verify_scene_urls_always_emitted(monkeypatch):
    """E3 URLs computed from the top candidate; E2 tile fetch content-checked;
    E1 Overpass empty -> honest no_tagged_parcels."""
    import requests as _rq
    monkeypatch.setattr(_rq, "get",
                        lambda *a, **k: _FakeResponse(200, b"\xff\xd8\xffxyz"))
    import modules.property_locator as pl
    monkeypatch.setattr(pl, "overpass_query", lambda q, timeout=40: [])

    from modules.investigative_reasoner import InvestigativeReasoner as R
    scene = R().verify_scene([{"latitude": 37.45643, "longitude": -79.09972,
                               "address": "151 Stratford Place"}])
    urls = scene["verification_urls"]
    assert "google.com/maps/@37.456430" in urls["street_view"]
    assert "mlat=37.45643" in urls["osm"]
    assert ",20z/data=!3m1!1e3" in urls["satellite"]
    # ESRI tile math computed here, content-validated JPEG
    assert scene["satellite"]["status"] == "fetched"
    assert scene["satellite"]["tile"][0] > 0 and scene["satellite"]["tile"][1] > 0
    assert scene["neighbors_status"] == "no_tagged_parcels"


def test_verify_scene_bad_tile_reported_honestly(monkeypatch):
    import requests as _rq
    import modules.property_locator as pl
    monkeypatch.setattr(_rq, "get",
                        lambda *a, **k: _FakeResponse(404, b"nope"))
    monkeypatch.setattr(pl, "overpass_query", lambda q, timeout=40: [])
    from modules.investigative_reasoner import InvestigativeReasoner as R
    scene = R().verify_scene([{"latitude": 0.0, "longitude": 0.0,
                               "address": "1 Test St"}])
    assert scene["satellite"]["status"] == "failed"
    assert "404" in scene["satellite"]["note"]


def test_verify_scene_empty_candidates():
    from modules.investigative_reasoner import InvestigativeReasoner as R
    scene = R().verify_scene([])
    assert scene["status"] == "skipped"
    assert scene["neighbors"] == [] and scene["verification_urls"] is None


def test_scene_parity_confirmation_bumps_confidence(monkeypatch):
    """E1: an Overpass neighbor parcel carrying the clue number confirms the
    block -> +0.05 confidence (capped at 0.85), SCENE CONFIRMED in chain."""
    r = InvestigativeReasoner()
    monkeypatch.setattr("modules.listing_finder.geocode_address",
                        _fake_geocode_factory())

    def _stub_scene(cands, radius_m=90):
        return {"status": "partial",
                "satellite": {"status": "fetched", "zoom": 20},
                "neighbors": [{"address": "149 Stratford Place",
                               "latitude": 37.4564, "longitude": -79.0997,
                               "via": "overpass_neighbor",
                               "url": "", "title": "", "query": ""}],
                "neighbors_status": "success",
                "verification_urls": {"street_view": "sv", "osm": "o",
                                      "satellite": "s"}}

    monkeypatch.setattr(r, "verify_scene", _stub_scene)
    rec = r.investigate(
        clues={"visible_numbers": ["149"], "street_names": ["Stratford Place"],
               "region_guess": "Madison Heights, VA"},
        region_hint="Madison Heights, VA")
    assert any("SCENE CONFIRMED" in s for s in rec["reasoning_chain"])
    # 0.82 deduced + 0.05 scene-confirmed -> capped at the honest 0.85 ceiling
    assert rec["candidates"][0]["confidence"] == 0.85


def test_web_fallback_reports_fenced_honestly(monkeypatch):
    """All SearXNG fenced + DDG empty -> status 'fenced', not fake success."""
    import requests as _rq

    def _boom(*a, **k):
        raise _rq.RequestException("fenced")

    monkeypatch.setattr(_rq, "get", _boom)

    class _EmptyOrch(_FakeOrch):
        def signal_web_search(self, query, n=5):
            return {"status": "empty", "query": query, "results": [],
                    "count": 0}

    monkeypatch.setattr("modules.live_signal_orchestrator."
                        "LiveSignalOrchestrator", _EmptyOrch)
    web = InvestigativeReasoner().research_property_records(
        {"visible_numbers": ["10"], "high_leverage_features": ["blue door"],
         "region_guess": "Nowheresville, VS"}, "Nowheresville, VS")
    assert web["status"] in ("fenced", "empty")
    assert web["findings"] == []


# ---------------------------------------------------------------------- #
# live research (real DDG + Nominatim) — the Madison Heights case         #
# ---------------------------------------------------------------------- #
@pytest.mark.slow
def test_live_research_madison_heights():
    """The exact Gemini-solved case: clues from the image -> OSM-direct house
    geocode -> verified coordinates. Real network (Nominatim)."""
    r = InvestigativeReasoner()
    rec = r.investigate(
        clues={
            "visible_numbers": ["151"],
            "street_names": ["Stratford Place"],
            "high_leverage_features": [
                "rooftop solar panels on the neighboring single-story home",
                "newer planned subdivision with rolling terrain",
            ],
            "property_style": "two-story vinyl-sided home",
            "setting": "planned residential subdivision",
            "region_guess": "Madison Heights, VA",
        },
        region_hint="Madison Heights, VA",
    )
    chain = " ".join(rec.get("reasoning_chain", []))
    assert "RESEARCH[OSM-direct]" in chain
    # Deterministic assertion: OSM has these parcels, so the loop MUST produce
    # the 151 Stratford Place candidate with coordinates.
    assert rec["status"] == "success", rec.get("note")
    target = [c for c in rec["candidates"]
              if c["address"].startswith("151 Stratford")]
    assert target, [c["address"] for c in rec["candidates"]]
    t = target[0]
    assert abs(t["latitude"] - 37.4564) < 0.002
    assert abs(t["longitude"] - (-79.0997)) < 0.002

"""Stage 3c retrieval-augmented re-ranker — offline unit tests.

Verifies the StreetCLIP retrieval evidence actually MOVES the candidate
ranking: proportional boost, hard-negative demotion, discriminative margin,
and GPS-anchor injection from the best live ground photo. No network, no
model loads — pure dict math.
"""
import pytest

from modules.geo_harness import GeoVisionHarness


def _cand(lat, lon, conf, source="geoclip"):
    return {"latitude": lat, "longitude": lon, "confidence": conf, "source": source}


def _verif(lat, lon, sim, photo_lat=None, photo_lon=None, photo_sim=None):
    url = f"https://example.com/{lat}_{lon}.jpg"
    matches = []
    photos = []
    if photo_lat is not None:
        matches.append({"thumbnail_url": url, "similarity": photo_sim})
        photos.append({"thumbnail_url": url, "latitude": photo_lat,
                       "longitude": photo_lon, "title": "ref"})
    return {
        "candidate": {"latitude": lat, "longitude": lon, "confidence": 0.3,
                      "source": "geoclip"},
        "visual_similarity": sim,
        "reference_photos": photos,
        "reference_matches": matches,
    }


@pytest.fixture
def h():
    return GeoVisionHarness()


def test_boost_proportional_and_capped(h):
    cands = [_cand(48.85, 2.29, 0.50)]
    vv = {"verifications": [_verif(48.85, 2.29, sim=0.80)]}
    out = h.retrieval_rerank(cands, vv)
    assert out["status"] == "success"
    # (0.80-0.20)/0.60 * 0.25 = 0.25
    assert cands[0]["confidence"] == pytest.approx(0.75, abs=1e-6)
    assert out["boosts"] and out["boosts"][0]["boost"] == pytest.approx(0.25, abs=1e-6)


def test_hard_negative_demotion(h):
    cands = [_cand(41.89, 12.49, 0.40)]
    vv = {"verifications": [_verif(41.89, 12.49, sim=0.05)]}
    out = h.retrieval_rerank(cands, vv)
    assert cands[0]["confidence"] == pytest.approx(0.34, abs=1e-6)
    assert out["demotions"]


def test_discriminative_margin_bonuses_best_only(h):
    cands = [_cand(48.85, 2.29, 0.50), _cand(51.50, -0.12, 0.50)]
    vv = {"verifications": [_verif(48.85, 2.29, sim=0.50),
                            _verif(51.50, -0.12, sim=0.30)]}
    out = h.retrieval_rerank(cands, vv)
    assert out["discriminative_margin"] == pytest.approx(0.20, abs=1e-6)
    # best gets margin bonus on top of proportional boost
    b0 = cands[0]["confidence"] - 0.50
    b1 = cands[1]["confidence"] - 0.50
    assert b0 > b1
    assert b0 == pytest.approx((0.50 - 0.20) / 0.60 * 0.25 + 0.06, abs=1e-6)


def test_gps_anchor_injection_from_best_photo(h):
    cands = [_cand(48.85, 2.29, 0.50)]
    vv = {"verifications": [_verif(48.85, 2.29, sim=0.55,
                                   photo_lat=48.8584, photo_lon=2.2945,
                                   photo_sim=0.55)]}
    out = h.retrieval_rerank(cands, vv)
    inj = out["injected"]
    assert inj is not None
    assert inj["latitude"] == pytest.approx(48.8584)
    assert inj["longitude"] == pytest.approx(2.2945)
    assert inj["source"] == "live_ground_retrieval"
    # 0.30 + 0.55*0.5 = 0.575, under the 0.75 cap
    assert inj["confidence"] == pytest.approx(0.575, abs=1e-6)
    assert inj["confidence"] <= 0.75


def test_injection_blocked_below_strong_match(h):
    cands = [_cand(48.85, 2.29, 0.50)]
    vv = {"verifications": [_verif(48.85, 2.29, sim=0.30,
                                   photo_lat=48.8584, photo_lon=2.2945,
                                   photo_sim=0.30)]}
    out = h.retrieval_rerank(cands, vv)
    assert out["injected"] is None


def test_empty_inputs_skip(h):
    out = h.retrieval_rerank([], {"verifications": []})
    assert out["status"] == "skipped"
    out = h.retrieval_rerank([_cand(0, 0, 0.1)], {"verifications": []})
    assert out["status"] == "skipped"


def test_wired_into_investigate_record(h):
    """The harness investigate() must run stage 3c on a real record path.

    Cheap probe: the method must exist and the investigate source must call
    it between visual verification and finalize (source-level wire check —
    the full E2E is covered by live canvas runs).
    """
    import inspect
    src = inspect.getsource(GeoVisionHarness.investigate)
    assert "retrieval_rerank" in src
    order = [src.index("visual_verify_candidates("),
             src.index("self.retrieval_rerank("),
             src.index("self._finalize(")]
    assert order == sorted(order), "3c must run after 3b and before finalize"

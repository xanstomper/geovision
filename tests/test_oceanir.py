"""
Tests for the OceanIR-style 'evidence workspace' backend (/api/oceanir).

Covers the modern-harness analyze endpoint contract (ranked alternatives,
reference imagery, evidence chain, honest 'unsure' flag, canvas link), with the
harness mocked so no heavy model runs in the suite. Tests use Flask's test client.
"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from web.app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _harness_result():
    return {
        "status": "success",
        "best_estimate": {"latitude": 48.8584, "longitude": 2.2945,
                          "confidence": 0.85, "city": "Paris", "country": "FR",
                          "place_name": "Eiffel Tower"},
        "candidates": [
            {"latitude": 48.8584, "longitude": 2.2945, "confidence": 0.85,
             "city": "Paris", "source": "geoclip"},
            {"latitude": 41.8935, "longitude": 12.5120, "confidence": 0.2,
             "city": "Rome", "source": "heuristic"},
        ],
        "reasoning_chain": ["COARSE: urban tower", "RETRIEVAL: regional match"],
        "reference_urls_by_candidate": {"Paris": [{"url": "https://e.com/x.jpg",
                                                     "name": "Eiffel ref"}]},
    }


def test_oceanir_returns_evidence_workspace(client, monkeypatch):
    import modules.geo_harness as gh
    monkeypatch.setattr(gh.GeoVisionHarness, "investigate",
                        lambda *a, **k: _harness_result())
    # canvas disabled for the unit test to avoid file writes / model-free
    data = {"canvas": "0"}
    data.update({"images": (io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 32),
                            "query.jpg")})
    r = client.post("/api/oceanir",
                    data=data,
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["status"] == "success"
    assert j["best"]["confidence"] == 0.85
    assert j["best"]["place"] == "Paris"
    assert j["unsure"] is False                      # conf 0.85 >= 0.3
    assert len(j["candidates"]) == 2                 # ranked alternatives
    assert len(j["reasoning"]) == 2                  # evidence chain
    assert "Paris" in j["reference_urls"]            # reference imagery


def test_oceanir_unsure_flag_for_low_confidence(client, monkeypatch):
    import modules.geo_harness as gh
    low = _harness_result()
    low["best_estimate"]["confidence"] = 0.1
    monkeypatch.setattr(gh.GeoVisionHarness, "investigate",
                        lambda *a, **k: low)
    r = client.post("/api/oceanir",
                    data={"canvas": "0",
                          "images": (io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 32),
                                     "q.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["unsure"] is True           # 'tells you when it is not sure'


def test_oceanir_page_serves(client):
    """The OceanIR-style evidence workspace page renders."""
    r = client.get("/ocean")
    assert r.status_code == 200
    assert "Evidence Workspace" in r.get_data(as_text=True)
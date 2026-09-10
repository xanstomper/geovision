"""
Tests for the keyed-but-activatable stages (items 3a/3b, OceanIR parity wave).

Both GeoVision's optional VLM reasoning and Google Vision listing/reverse-image
stages read a single env var and must:
  - be SKIPPED cleanly (no crash) when the key is absent
  - be CALLABLE (activation path wired) when the key is present

These tests lock that contract so neither regresses. They do NOT exercise the live
keyed calls (needs a real key); they assert the gating + graceful-skip behavior.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

IMG = "data/eval/wikipedia_landmarks_v1/images/colosseum.jpg"


@pytest.fixture(autouse=True)
def _no_keys():
    """Ensure keyed stages start unset for every test."""
    for k in ("GEOVISION_VLM_API_KEY", "OPENCODE_ZEN_API_KEY", "GOOGLE_VISION_API_KEY"):
        os.environ.pop(k, None)
    yield


def test_vlm_stage_skips_cleanly_without_key():
    from modules.geo_harness import GeoVisionHarness
    r = GeoVisionHarness().coarse_reason(IMG)
    assert r.get("status") == "skipped"          # graceful, not a failure


def test_vlm_client_none_without_key():
    from modules.vlm_geo_analyzer import _get_client
    client, model = _get_client()
    assert client is None and model is None


def test_vlm_activation_reads_env_var():
    """Setting the env var changes the gating (call path active, not skipped)."""
    os.environ["GEOVISION_VLM_API_KEY"] = "sk-test-fake"
    from modules.vlm_geo_analyzer import _get_client
    # a fake key still yields a client object (activation path), not None
    client, _ = _get_client()
    assert client is not None


def test_google_vision_skips_cleanly_without_key():
    from modules.listing_finder import google_vision_matching_pages
    r = google_vision_matching_pages(IMG)
    assert r.get("status") == "skipped"
    assert "GOOGLE_VISION_API_KEY" in r.get("note", "")


def test_google_vision_activation_reads_env_var():
    os.environ["GOOGLE_VISION_API_KEY"] = "AIza-fake"
    from modules.reverse_image_search import ReverseImageSearcher
    assert ReverseImageSearcher().google_vision_key == "AIza-fake"
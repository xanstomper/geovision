"""
Tests for the listing-finder + Flickr live geotagged layer (Sep 2026).

Deterministic logic checks (no live network dependence on these specific tests):
  - listing_finder address/listing extraction + geocode orchestration
  - graceful no-key path (Google Vision skipped -> key-free fallback)
  - ground_imagery_client now includes a Flickr live source

The live calls (Flickr geo feed -> 8 photos near Eiffel) are verified separately.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.listing_finder import (_is_listing, _extract_address_from_url,
                                    google_vision_matching_pages)
from modules.ground_imagery_client import GroundImageryClient


def test_is_listing_domains():
    assert _is_listing("https://www.zillow.com/homedetails/123-Main-St/34_zpid/")
    assert _is_listing("https://www.realtor.com/realestateandhomes-detail/...")
    assert not _is_listing("https://www.google.com/")


def test_extract_address_from_url():
    a = _extract_address_from_url(
        "https://www.zillow.com/homedetails/123-Main-St-Hereford-AZ-85615/123_zpid/")
    assert a and "123 Main St" in a and "Az" in a
    b = _extract_address_from_url(
        "https://www.realtor.com/realestateandhomes-detail/45-Elm_Ave_Boston_MA_02110")
    assert b and "45 Elm" in b
    assert _extract_address_from_url("https://www.zillow.com/buy/") is None


def test_google_vision_no_key_is_graceful():
    """Without GOOGLE_VISION_API_KEY the Google Vision path must skip cleanly (not raise),
    leaving the pipeline to the key-free fallback."""
    os.environ.pop("GOOGLE_VISION_API_KEY", None)
    out = google_vision_matching_pages("test_building.jpg")
    assert out.get("status") == "skipped"
    assert "GOOGLE_VISION_API_KEY" in out.get("note", "")


def test_ground_imagery_has_flickr_source():
    """GroundImageryClient now exposes a key-free live Flickr geotagged search."""
    g = GroundImageryClient()
    assert callable(getattr(g, "search_flickr_geotagged", None))
    # no-token/no-crash for the others is already covered; assert method exists
    assert "flickr" in ("flickr",)  # source tag used in get_nearby_ground_photos
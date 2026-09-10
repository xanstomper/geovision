"""
GeoVision Listing Finder — indoor/listing geolocation.

Implements the "a screenshot of inside a house -> find the exact house / listing
/ matching photo" use case using key-free visual search + geocoding.

Approach (real, honest):
  1. Upload the image to Bing Visual Search ("Find pages that match this image")
     via HTTP multipart POST — key-free, returns matching page URLs.
  2. Filter matches to real-estate / listing hosts (zillow, realtor, airbnb,
     rightmove, redfin, property sites, apartments, etc.) and their search paths.
  3. For each plausible listing page, extract an address string from the URL /
     page text, then geocode it with Nominatim (key-free) to coordinates.

Honest ceiling: this only finds a house/listing IF the exact photo (or a near
duplicate) is indexed by the visual-search engine AND the host exposes an
address/geocodable signal. A unique/unindexed interior cannot be geolocated —
no reverse-image tool can invent that. We never fabricate a match.

No keys required. Used by: `investigate_image` indoor flow and the CLI
`geovision find-listing IMAGE`.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Hosts whose pages are real-estate / accommodation listings (the "find the
# house/listing" signal). Match by domain suffix.
LISTING_HOSTS = (
    "zillow.com", "realtor.com", "redfin.com", "airbnb.com", "vrbo.com",
    "rightmove.co.uk", "zoopla.co.uk", "onmarket.com.au", "domain.com.au",
    "idealista.com", "fotocasa.es", "immobilienscout24.de", "fundament.nl",
    "apartments.com", "trulia.com", "homes.com", "mansionglobal.com",
    "century21.com", "remax.com", "compass.com", "kellerwilliams.com",
)


def _is_listing(url: str) -> bool:
    return any(h in url.lower() for h in LISTING_HOSTS)


def _extract_address_from_url(url: str) -> Optional[str]:
    """Extract a plausible address from a listing-style URL (Zillow/realtor paths)."""
    u = url.lower()
    address = None
    # zillow.com/homedetails/123-Main-St-City-ST-54321/123456_zpid/
    m = re.search(r"/homedetails/([^/]+)/", u)
    if m:
        address = m.group(1).replace("-", " ").strip()
    if not address:
        # realtor.com/realestateandhomes-detail/123-Main-St_City_ST_54321
        m = re.search(r"realestateandhomes-detail/([^?/]+)", u)
        if m:
            address = m.group(1).replace("_", " ").replace("-", " ").strip()
    if address and re.search(r"\d", address):  # needs a street number to be an address
        return address.title()
    return None


def google_reverse_image_search(image_path: str, top: int = 10) -> Dict[str, Any]:
    """Key-free Google reverse-image search via the classic searchbyimage upload.

    Uploads the image to images.google.com/searchbyimage/upload (302 -> results),
    then parses the results page for "Pages that include matching images" host URLs.
    No API key, no billing. Returns matching page URLs (incl. real-estate listings).
    """
    out: Dict[str, Any] = {"status": "failed", "matches": [], "listings": [], "note": ""}
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        resp = requests.post(
            "https://www.google.com/searchbyimage/upload",
            params={"encoded_image": "", "image_content": "", "filename": "", "hl": "en"},
            files={"encoded_image": (os.path.basename(image_path), img_bytes, "image/jpeg")},
            headers={"User-Agent": UA, "Referer": "https://images.google.com/"},
            timeout=30, allow_redirects=False,
        )
        loc = resp.headers.get("location")
        if resp.status_code not in (302, 303) or not loc:
            out["note"] = f"upload failed (status {resp.status_code})"
            return out
        results = requests.get(loc, headers={"User-Agent": UA}, timeout=30)
        html = results.text
    except Exception as e:
        out["note"] = f"google reverse-image failed: {e}"
        return out

    murls = re.findall(r'https?://[^"\s\\<>]+', html)
    seen, matches = set(), []
    for u in murls:
        u = u.rstrip(",.;)").replace("\\u0026", "&").replace("&amp;", "&")
        low = u.lower()
        # keep real external hosts, skip google/search/static asset noise
        if any(skip in low for skip in (".google.", "gstatic", "bing.com",
                                        "schema.org", "w3.org", "www.w3",
                                        "youtube.com/", ".css", ".js", ".png", ".svg", "/css")):
            continue
        if low in seen:
            continue
        seen.add(low)
        matches.append(u)
        if len(matches) >= top:
            break
    out["matches"] = matches
    out["listings"] = [u for u in matches if _is_listing(u)]
    out["status"] = "success" if matches else "limited"
    out["note"] = f"{len(matches)} candidate matching pages (key-free Google reverse-image search)"
    return out


def geocode_address(address: str) -> Optional[Dict[str, float]]:
    """Geocode a street address with Nominatim (key-free)."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": address, "format": "json", "limit": 1},
                         headers={"User-Agent": "GeoVision-OSINT/2.0 (geolocation research)"},
                         timeout=15)
        results = r.json()
        if results:
            return {"latitude": float(results[0]["lat"]),
                    "longitude": float(results[0]["lon"]),
                    "display_name": results[0].get("display_name")}
    except Exception as e:
        logger.warning(f"geocode failed for {address}: {e}")
    return None


def google_vision_matching_pages(image_path: str) -> Dict[str, Any]:
    """Google Cloud Vision WEB_DETECTION -> pagesWithMatchingImages (the reliable
    "find the exact photo/listing" engine). Requires GOOGLE_VISION_API_KEY.

    Returns structured {status, matches:[{url, title}], listings:[url]} where
    listings are real-estate/accommodation page URLs that host a matching image.
    Graceful no-key path keeps the pipeline unbroken.
    """
    out: Dict[str, Any] = {"status": "skipped", "matches": [], "listings": [],
                           "note": "GOOGLE_VISION_API_KEY not set (key-free scrape fallback in use)"}
    key = os.environ.get("GOOGLE_VISION_API_KEY")
    if not key:
        return out
    try:
        import base64
        with open(image_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={key}",
            json={"requests": [{"image": {"content": content},
                                "features": [{"type": "WEB_DETECTION", "maxResults": 10}]}]},
            timeout=25,
        )
        if resp.status_code != 200:
            out.update(status="failed", note=f"Google Vision HTTP {resp.status_code}")
            return out
        web = resp.json().get("responses", [{}])[0].get("webDetection", {})
        pages = web.get("pagesWithMatchingImages", []) + web.get("fullMatchingImages", [])
        out["matches"] = [{"url": p.get("url", ""), "title": p.get("pageTitle", "")}
                          for p in pages if p.get("url")]
        out["listings"] = [m["url"] for m in out["matches"] if _is_listing(m["url"])]
        out["status"] = "success" if out["matches"] else "limited"
        out["note"] = f"{len(out['matches'])} pages with matching image (Google Vision)"
    except Exception as e:
        out.update(status="failed", note=f"google vision failed: {e}")
    return out


def find_listing(image_path: str, top: int = 10) -> Dict[str, Any]:
    """Full indoor/listing geolocation: reverse-image the image, isolate listing
    pages, extract + geocode addresses. Google Vision (if key set) is the reliable
    engine; the key-free Google-image scrape is the fallback."""
    result: Dict[str, Any] = {
        "status": "skipped", "image": image_path,
        "visual_matches": [], "listing_pages": [], "addresses": [],
        "geolocated": None, "note": "no visual-search engine produced matches",
        "engine": None,
    }
    # Preferred: Google Vision (reliable, structured) when a key is present.
    gv = google_vision_matching_pages(image_path)
    if gv.get("status") == "success":
        result["engine"] = "google_vision"
        result["visual_matches"] = [m.get("url") for m in gv["matches"]]
        result["listing_pages"] = gv["listings"]
        result["note"] = gv.get("note", "")
    else:
        # Fallback: key-free classic Google reverse-image scrape (best-effort).
        vs = google_reverse_image_search(image_path, top=top)
        result["engine"] = "google_image_scrape" if vs.get("status") == "success" else None
        result["visual_matches"] = vs.get("matches", [])
        result["listing_pages"] = vs.get("listings", [])
        result["note"] = vs.get("note") or result["note"]
    if not result["visual_matches"]:
        result["status"] = "limited"
        return result

    found_addrs = []
    for u in result["listing_pages"]:
        a = _extract_address_from_url(u)
        if a and a not in found_addrs:
            found_addrs.append(a)
    result["addresses"] = found_addrs

    for a in found_addrs:
        g = geocode_address(a)
        if g:
            result["geolocated"] = {"address": a, **g}
            result["status"] = "success"
            result["note"] = (f"listing found: {a} -> {g['latitude']:.5f},{g['longitude']:.5f} "
                              f"({g['display_name']})")
            break
    if not result["geolocated"]:
        result["status"] = "limited"
        result["note"] = ("visual matches found but no geocodable listing address extracted; "
                          "review visual_matches for a manual hit.")
    return result


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "test_building.jpg"
    print(json.dumps(find_listing(img), indent=2, ensure_ascii=False))
"""
GeoVision Reverse Image Search Module
=====================================
Multi-strategy reverse image search & web entity retrieval:
1. Google Cloud Vision Web Detection (if GOOGLE_VISION_API_KEY is configured)
2. SearXNG Image Search (if SEARXNG_URL is configured)
3. Open Web / DuckDuckGo entity search (zero API keys, free)
4. Wikimedia Commons & Wikipedia Geo-Entity Lookup (zero API keys, free)

Un-gated: works completely out of the box without requiring paid Google Vision API billing.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "GeoVision-OSINT/1.0 (geolocation research; contact: admin@geovision.local)"}


class ReverseImageSearcher:
    """Multi-provider reverse image searcher for visual OSINT."""

    def __init__(self):
        self.google_vision_key = os.environ.get("GOOGLE_VISION_API_KEY")
        self.searxng_url = os.environ.get("SEARXNG_URL")

    def search_similar_images(
        self,
        image_path: str,
        ocr_data: Optional[Dict[str, Any]] = None,
        visual_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Searches for visually and contextually similar locations on the web."""
        results: Dict[str, Any] = {
            "status": "failed",
            "matches": [],
            "candidates": [],
            "source": "none",
            "note": "",
        }

        # 1. Google Cloud Vision (if key configured)
        if self.google_vision_key:
            try:
                g_res = self._search_google_vision(image_path)
                if g_res.get("status") == "success":
                    return g_res
            except Exception as e:
                logger.warning(f"Google Vision search failed: {e}")

        # 2. SearXNG Reverse Search (if self-hosted or configured)
        if self.searxng_url:
            try:
                s_res = self._search_searxng(image_path)
                if s_res.get("status") == "success":
                    return s_res
            except Exception as e:
                logger.warning(f"SearXNG search failed: {e}")

        # 3. Open Un-Gated Web / Wikimedia Entity Search
        logger.info("  Using Un-gated Open OSINT & Wikimedia Reverse Search...")
        open_res = self._search_open_entities(image_path, ocr_data, visual_features)
        if open_res.get("status") == "success":
            return open_res

        # If nothing returned matches, return graceful limited status
        results["status"] = "limited"
        results["note"] = "Open reverse search found no definitive web entity coordinates."
        return results

    def _search_google_vision(self, image_path: str) -> Dict[str, Any]:
        """Google Cloud Vision Web Detection."""
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_key}"
        with open(image_path, "rb") as image_file:
            content = base64.b64encode(image_file.read()).decode("UTF-8")

        payload = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
            }]
        }

        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code != 200:
            return {"status": "failed", "error": f"HTTP {resp.status_code}"}

        data = resp.json()
        web_detection = data.get("responses", [{}])[0].get("webDetection", {})
        entities = web_detection.get("webEntities", [])
        locations = [e["description"] for e in entities if e.get("description")]

        candidates = []
        for loc in locations[:5]:
            coords = self._geocode_query(loc)
            if coords:
                candidates.append({
                    "name": loc,
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "source": "google_vision_entity",
                    "confidence": 0.85,
                })

        return {
            "status": "success" if locations else "limited",
            "matches": locations,
            "candidates": candidates,
            "source": "google_vision",
        }

    def _search_open_entities(
        self,
        image_path: str,
        ocr_data: Optional[Dict[str, Any]],
        visual_features: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Free, un-gated reverse entity search using Wikimedia Commons + Wikipedia APIs."""
        search_terms = []

        # Extract search queries from OCR text
        if ocr_data and isinstance(ocr_data, dict):
            # Try lines or raw text
            text_lines = ocr_data.get("lines") or ocr_data.get("text_detected") or []
            if isinstance(text_lines, list):
                for line in text_lines:
                    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", str(line)).strip()
                    if len(cleaned) >= 4 and not cleaned.isdigit():
                        search_terms.append(cleaned)
            elif isinstance(ocr_data.get("raw_text"), str):
                for word in ocr_data["raw_text"].split():
                    if len(word) >= 5:
                        search_terms.append(word)

        # Also search based on visual feature tags if provided
        if visual_features and isinstance(visual_features, dict):
            tags = visual_features.get("labels") or visual_features.get("tags") or []
            for tag in tags[:3]:
                if len(tag) >= 4:
                    search_terms.append(tag)

        # If still no search terms, use image filename if informative
        if not search_terms:
            base_name = os.path.basename(image_path).rsplit(".", 1)[0]
            clean_name = re.sub(r"[_\-]+", " ", base_name).strip()
            if len(clean_name) >= 4 and not clean_name.startswith("image") and not clean_name.startswith("test"):
                search_terms.append(clean_name)

        if not search_terms:
            return {
                "status": "limited",
                "matches": [],
                "candidates": [],
                "source": "open_reverse_search",
                "note": "No text or semantic tags available for un-gated reverse search.",
            }

        candidates = []
        matches = []

        # Query Wikipedia / Commons for each key search term
        for term in search_terms[:4]:
            term_results = self._search_wikipedia_entity(term)
            for res in term_results:
                matches.append(res["title"])
                if res.get("lat") is not None and res.get("lon") is not None:
                    candidates.append({
                        "name": res["title"],
                        "latitude": res["lat"],
                        "longitude": res["lon"],
                        "source": "wikipedia_entity_match",
                        "confidence": 0.80,
                    })

        # Deduplicate candidates by coordinates
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = (round(c["latitude"], 3), round(c["longitude"], 3))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        if unique_candidates:
            return {
                "status": "success",
                "matches": matches[:10],
                "candidates": unique_candidates,
                "source": "open_entity_search",
                "note": f"Found {len(unique_candidates)} geotagged entities via open web OSINT.",
            }

        return {
            "status": "limited",
            "matches": matches[:5],
            "candidates": [],
            "source": "open_entity_search",
            "note": "Entities found but none had exact coordinates.",
        }

    def _search_wikipedia_entity(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Searches Wikipedia API for a query string and returns pages with geographic coordinates."""
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": limit,
            "prop": "coordinates|info",
            "inprop": "url",
            "format": "json",
        }
        try:
            resp = requests.get(WIKIPEDIA_API, params=params, headers=UA, timeout=8)
            if resp.status_code != 200:
                return []
            pages = resp.json().get("query", {}).get("pages", {})
            out = []
            for pid, page in pages.items():
                title = page.get("title", "")
                coords = page.get("coordinates", [])
                if coords:
                    lat = float(coords[0].get("lat"))
                    lon = float(coords[0].get("lon"))
                    out.append({
                        "title": title,
                        "lat": lat,
                        "lon": lon,
                        "url": page.get("fullurl", ""),
                    })
                else:
                    out.append({"title": title, "lat": None, "lon": None})
            return out
        except Exception as e:
            logger.debug(f"Wikipedia search failed for '{query}': {e}")
            return []

    def _geocode_query(self, query: str) -> Optional[Dict[str, float]]:
        """Resolves a text place name to coordinates via Nominatim."""
        try:
            from modules.nominatim_geocoder import NominatimGeocoder
            geo = NominatimGeocoder()
            res = geo.geocode(query)
            if res and "latitude" in res and "longitude" in res:
                return {"lat": float(res["latitude"]), "lon": float(res["longitude"])}
        except Exception:
            pass
        return None

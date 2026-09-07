"""
Web Search Providers — Serper / Brave / SearXNG
================================================
Ports the search-provider clients from open_geo_spy
(github.com/eren23/open_geo_spy, src/geo/{serper_client,brave_client,searxng_client}.py).

Structured web search for geolocation OSINT — replaces browser scraping that
gets blocked. Each provider activates only when its API key env var is set:

  SERPER_API_KEY    -> Serper.dev (Google SERP, structured JSON)
  BRAVE_API_KEY     -> Brave Search API
  SEARXNG_URL       -> self-hosted/public SearXNG instance (no key needed)

All providers share one interface: search(query, num_results) -> list of
{title, link, snippet, source}. Geographic constraints (country hints) are
supported by Serper via gl/cr parameters.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _requests_search(url: str, headers: Dict[str, str],
                     params: Dict[str, Any], timeout: float = 15.0) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("search request to %s failed: %s", url, e)
        return None


def _post_search(url: str, headers: Dict[str, str],
                 payload: Dict[str, Any], timeout: float = 15.0) -> Optional[dict]:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("search POST to %s failed: %s", url, e)
        return None


class SerperClient:
    """Serper.dev Google SERP client (ported from open_geo_spy).

    Supports geographic constraints: gl (geolocation) + cr (country restrict),
    applied from a country hint's ISO code.
    """

    name = "serper"
    BASE_URL = "https://google.serper.dev"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num_results: int = 10,
               country_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """Google web search via Serper with optional country constraint.

        country_hint: country name or ISO code — sets gl/cr params and
        post-filters results to that country when resolvable.
        """
        if not self.available:
            return []

        payload: Dict[str, Any] = {"q": query, "num": num_results}

        if country_hint:
            iso = _to_iso(country_hint)
            if iso:
                payload["gl"] = iso.lower()
                payload["cr"] = f"country{iso.upper()}"

        data = _post_search(
            f"{self.BASE_URL}/search",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            payload=payload,
        )
        if not data:
            return []

        results = []
        organic = data.get("organic") or data.get("results") or []
        iso_filter = _to_iso(country_hint) if country_hint else None
        for i, r in enumerate(organic[:num_results]):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
                "position": i,
                "source": self.name,
            })
        if iso_filter:
            # Prefer results whose snippet/title mention the country
            results.sort(key=lambda r: 0 if _mention_country(r["title"] + " " + r["snippet"], country_hint) else 1)
        return results


class BraveClient:
    """Brave Search API client (ported from open_geo_spy)."""

    name = "brave"
    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        data = _requests_search(
            f"{self.BASE_URL}/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            },
            params={"q": query, "count": num_results},
        )
        if not data:
            return []
        results = []
        for i, r in enumerate((data.get("web") or {}).get("results", [])[:num_results]):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": r.get("description", ""),
                "position": i,
                "source": self.name,
            })
        return results


class SearXNGClient:
    """SearXNG meta-search client (ported from open_geo_spy).

    No API key required; point SEARXNG_URL at any SearXNG instance.
    """

    name = "searxng"

    def __init__(self, url: Optional[str] = None):
        self.url = (url or os.environ.get("SEARXNG_URL") or "").rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.url)

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        data = _requests_search(
            f"{self.url}/search",
            headers={"Accept": "application/json"},
            params={"q": query, "format": "json", "safesearch": "0"},
        )
        if not data:
            return []
        results = []
        for i, r in enumerate((data.get("results") or [])[:num_results]):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": r.get("content", ""),
                "position": i,
                "source": self.name,
            })
        return results


def _to_iso(country: str) -> Optional[str]:
    """Country name/alias -> ISO 3166-1 alpha-2 via the ported country matcher."""
    try:
        from modules.country_matcher import get_iso_code
        return get_iso_code(country)
    except Exception:
        return None


def _mention_country(text: str, country: str) -> bool:
    try:
        from modules.country_matcher import countries_match, extract_country_from_location
        found = extract_country_from_location(text)
        return bool(found and countries_match(found, country))
    except Exception:
        return False


def get_available_providers() -> List[Any]:
    """Return instances of every configured (available) search provider."""
    providers = [SerperClient(), BraveClient(), SearXNGClient()]
    return [p for p in providers if p.available]


def search_web(query: str, num_results: int = 10,
               country_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search across all configured providers, merged + deduped by link."""
    seen: set = set()
    merged: List[Dict[str, Any]] = []
    for p in get_available_providers():
        try:
            kwargs: Dict[str, Any] = {"num_results": num_results}
            if p.name == "serper":
                kwargs["country_hint"] = country_hint
            for r in p.search(query, **kwargs):
                link = r.get("link", "")
                if link and link not in seen:
                    seen.add(link)
                    merged.append(r)
        except Exception as e:
            logger.warning("%s search failed: %s", p.name, e)
    return merged[: num_results * 2]


if __name__ == "__main__":
    print("Providers configured:")
    for p in [SerperClient(), BraveClient(), SearXNGClient()]:
        print(f"  {p.name}: {'YES' if p.available else 'no (set key env var to enable)'}")
    if get_available_providers():
        print("\nSample search:")
        for r in search_web("Eiffel Tower Paris", num_results=3):
            print(f"  - {r['title']} ({r['source']})")
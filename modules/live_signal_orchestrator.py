"""
GeoVision Live Signal Orchestrator — live pull-and-problem-solve for any model.

The core ask: "a model can pull live data references and compare and problem
solve." This module gives a model (any CV/VLM agent) ONE node that fans out to
MANY independent LIVE signal sources for a coordinate or a text query, and returns
a synthesized evidence bundle the model can cross-examine.

It orchestrates live (not pre-configured, not pre-downloaded):
  - web search          (DuckDuckGo key-free, SearXNG/Brave/Serper if configured)
  - reverse-image lookup  (open web entities / Wikimedia; Google Vision if key)
  - weather corroboration (Open-Meteo, key-free)
  - OSM/Overpass          (businesses, hotels, landmarks, POIs — live)
  - ground imagery        (Wikimedia + Mapillary + Flickr — live)
  - GeoNames city snap    (offline index)
  - Nominatim geocode     (address <-> coords, live)

Each signal is isolated: one source failing never breaks the bundle. Output is a
flat dict of {source: signal} the caller can reason over. No keys required for the
core sources (web/weather/overpass/imagery/geonames/nominatim).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LiveSignalOrchestrator:
    """Fan out to live sources and assemble independent evidence signals."""

    def __init__(self, timeout: int = 20, concurrency: bool = True):
        self.timeout = timeout
        self.concurrency = concurrency

    # ------------------------------------------------------------ #
    # Individual signals (each returns a small dict, never raises)  #
    # ------------------------------------------------------------ #
    def signal_web_search(self, query: str, n: int = 5) -> Dict[str, Any]:
        # Key-free first: SearXNG if configured, else DuckDuckGo lite HTML.
        try:
            from modules.web_search_providers import SearXNGClient, BraveClient, SerperClient
            out: List[Dict[str, Any]] = []
            try:
                s = SearXNGClient()
                if s.available():
                    out = s.search(query, n) or []
            except Exception:
                out = []
            if not out:
                out = self._ddg_lite(query, n)
            if not out:
                for cls in (BraveClient, SerperClient):
                    try:
                        c = cls()
                        if c.available():
                            out = c.search(query, n) or []
                            break
                    except Exception:
                        continue
            return {"status": "ok" if out else "empty", "query": query,
                    "results": out[:n], "count": len(out)}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    @staticmethod
    def _ddg_lite(query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Key-free DuckDuckGo HTML lite search (no API key)."""
        try:
            import requests
            import re
            r = requests.post("https://html.duckduckgo.com/html/",
                              data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                               r.text, re.S)[:n]
            import html as _h
            return [{"title": _h.unescape(re.sub(r"<[^>]+>", "", t)).strip(),
                     "url": u} for u, t in links]
        except Exception:
            return []

    def signal_reverse_image(self, image_path: Optional[str] = None,
                             query: Optional[str] = None) -> Dict[str, Any]:
        try:
            from modules.reverse_image_search import ReverseImageSearcher
            r = ReverseImageSearcher()
            if image_path:
                return r.search_similar_images(image_path)
            if query:
                return {"status": "ok", "entities": r._search_wikipedia_entity(query)}
            return {"status": "empty"}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_weather(self, lat: float, lon: float,
                       date: Optional[str] = None) -> Dict[str, Any]:
        try:
            from modules.weather_corroborator import WeatherCorroborator
            w = WeatherCorroborator().fetch_historical_weather(lat, lon, date or "")
            return {"status": "ok" if w else "empty", "weather": w}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_osint(self, lat: float, lon: float,
                     radius_m: int = 2500,
                     cats: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            from modules.property_locator import PropertyLocator
            r = PropertyLocator().list_nearby(
                lat, lon, radius_m=radius_m,
                categories=["lodging", "commercial", "dining", "office", "landuse",
                            "hotel", "rental"] if not cats else cats)
            return {"status": r.get("status"), "count": r.get("count"),
                    "listings": r.get("listings", [])}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_ground_imagery(self, lat: float, lon: float,
                              radius_m: int = 1200, limit: int = 8) -> Dict[str, Any]:
        try:
            from modules.ground_imagery_client import GroundImageryClient
            r = GroundImageryClient().get_nearby_ground_photos(lat, lon, radius_m=radius_m, limit=limit)
            return {"status": "ok", "photos": r.get("ground_photos", []),
                    "count": len(r.get("ground_photos", []))}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_geonames_city(self, lat: float, lon: float) -> Dict[str, Any]:
        try:
            from modules.geonames_city_snap import get_city_index
            idx = get_city_index()
            city = idx.snap(lat, lon) if hasattr(idx, "snap") else None
            return {"status": "ok" if city else "empty", "city": city}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        try:
            from modules.nominatim_geocoder import NominatimGeocoder
            r = NominatimGeocoder().reverse_geocode(lat, lon)
            return {"status": "ok" if r else "empty", "place": r}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    def signal_forward_geocode(self, address: str) -> Dict[str, Any]:
        try:
            from modules.nominatim_geocoder import NominatimGeocoder
            r = NominatimGeocoder().forward_geocode(address)
            return {"status": "ok" if r else "empty", "coords": r}
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}

    # ------------------------------------------------------------ #
    # Orchestration                                                  #
    # ------------------------------------------------------------ #
    def _run(self, fn: Callable[[], Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Exception]]:
        """Run a signal under a hard wall-clock timeout (daemon thread). Never blocks
        the caller forever even if a live endpoint hangs. `concurrency` only affects
        the orchestrator's fan-out, not this guard."""
        try:
            box: Dict[str, Any] = {}
            exc: List[Optional[Exception]] = [None]

            def _wrap():
                try:
                    box["r"] = fn()
                except Exception as e:  # pragma: no cover - defensive
                    exc[0] = e
            t = threading.Thread(target=_wrap, daemon=True)
            t.start()
            t.join(self.timeout)
            if t.is_alive():
                return {"status": "timed_out"}, None
            if box.get("r") is not None:
                return box["r"], None
            return {"status": "errored", "note": (str(exc[0]) if exc[0] else "unknown")}, exc[0]
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120]}, e

    def investigate(self, lat: Optional[float] = None, lon: Optional[float] = None,
                    query: Optional[str] = None, address: Optional[str] = None,
                    image_path: Optional[str] = None,
                    radius_m: int = 2500, date: Optional[str] = None) -> Dict[str, Any]:
        """Fan out to all live signals for the given anchor (coords or text query).

        Returns a dict of {source: signal}. A model cross-examines these to solve.
        """
        out: Dict[str, Any] = {"status": "failed", "signals": {}, "note": ""}

        # Geo anchor: if only text given, geocode it first.
        if lat is None or lon is None:
            if address:
                fc = self.signal_forward_geocode(address)
                out["signals"]["address"] = fc
                if fc.get("coords"):
                    lat = float(fc["coords"].get("lat"))
                    lon = float(fc["coords"].get("lon"))
            if (lat is None or lon is None) and query:
                gq = self.signal_forward_geocode(query)
                out["signals"]["query_geocode"] = gq
                if gq.get("coords"):
                    lat = float(gq["coords"].get("lat"))
                    lon = float(gq["coords"].get("lon"))

        if lat is None or lon is None:
            # text-only: web + reverse-image signals
            out["signals"]["web"] = self.signal_web_search(query or address or "", 6)
            out["signals"]["reverse_image"] = self.signal_reverse_image(image_path=image_path, query=query)
            out["status"] = "success" if out["signals"] else "failed"
            return out

        jobs: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
            ("web", lambda: self.signal_web_search(query or f"{lat:.4f},{lon:.4f}", 5)),
            ("ground_imagery", lambda: self.signal_ground_imagery(lat, lon, radius_m)),
            ("osint", lambda: self.signal_osint(lat, lon, radius_m)),
            ("weather", lambda: self.signal_weather(lat, lon, date)),
            ("city_snap", lambda: self.signal_geonames_city(lat, lon)),
            ("reverse_geocode", lambda: self.signal_reverse_geocode(lat, lon)),
        ]
        if image_path:
            jobs.append(("reverse_image", lambda: self.signal_reverse_image(image_path=image_path)))

        for name, fn in jobs:
            res, _ = self._run(fn)
            out["signals"][name] = res

        out["status"] = "success"
        out["note"] = (f"{len(out['signals'])} live signals pulled for "
                       f"({float(lat):.5f},{float(lon):.5f}); cross-examine {list(out['signals'])}")
        return out


if __name__ == "__main__":
    import sys
    o = LiveSignalOrchestrator()
    if len(sys.argv) >= 3 and sys.argv[1] == "--coord":
        print(json.dumps(o.investigate(float(sys.argv[2]), float(sys.argv[3])),
                         indent=2, default=str))
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "Eiffel Tower, Paris"
        print(json.dumps(o.investigate(query=q), indent=2, default=str))
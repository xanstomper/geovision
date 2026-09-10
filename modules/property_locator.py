"""
GeoVision PropertyLocator — track Airbnb/hotel/rental/commercial/property listings.

Enumerates and monitors real, geotagged hospitality + commercial + property
listings at any coordinate (or address), from OpenStreetMap via Overpass (free,
licensed, key-free). This is the "track airbnbs, hotels, rentals, and other
listings for things, businesses, and properties" capability.

Sources — honest boundaries:
  * OSM/Overpass is the reliable, ToS-clean, live source: hotels, guest houses,
    hostels, apartments, offices, shops, restaurants, banks, landuse/parcels are
    all in OSM with names + GPS. Works today, no key.
  * Airbnb/Booking/Zillow/Realtor have NO public listing API and forbid scraping —
    we do NOT scrape them. Instead, for an Airbnb/hotel you already know by name+
    vicinity, we can (a) locate its OSM footprint, (b) corroborate with ground
    photos, satellite, and reverse image, and (c) track it.

Use:
  PropertyLocator().listings(lat, lon, radius_m, categories)   -> nearby listings
  PropertyLocator().from_address(address)
  PropertyLocator().track(name_or_point)                        -> re-query + diff
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# category -> Overpass tag filter (key-free query)
CATEGORIES = {
    "lodging": r'nwr["tourism"~"hotel|guest_house|hostel|motel|apartment|chalet|camp_site"]',
    "hotel": r'nwr["tourism"="hotel"]',
    "rental": r'nwr["tourism"~"apartment|guest_house|chalet|vacant"]',
    "commercial": r'nwr["shop"]',
    "dining": r'nwr["amenity"~"restaurant|cafe|bar|fast_food"]',
    "office": r'nwr["office"]',
    "business": r'nwr["office"]',
    "landuse": r'way["landuse"~"residential|commercial|retail|industrial"]',
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_name(t: Dict[str, str]) -> str:
    for k in ("name", "brand", "tourism", "office", "shop", "amenity"):
        if t.get(k):
            return t[k]
    return "unnamed"


def overpass_query(query: str, timeout: int = 40) -> List[Dict[str, Any]]:
    """Run a real Overpass query with multi-endpoint fallback + hard timeout."""
    headers = {"User-Agent": "GeoVision-OSINT/2.0 (geolocation research)",
               "Accept": "application/json"}
    for endpoint in OVERPASS_ENDPOINTS:
        result = {}
        def _run():
            try:
                r = requests.post(endpoint, data={"data": query},
                                  headers=headers, timeout=timeout)
                if r.status_code == 200:
                    result["elements"] = r.json().get("elements", [])
            except Exception as e:
                result["error"] = str(e)[:120]
        import threading
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout + 10)
        if t.is_alive():
            logger.warning("Overpass %s timed out", endpoint)
            continue
        if "elements" in result:
            return result["elements"]
    logger.warning("all Overpass endpoints failed for query")
    return []


class PropertyLocator:
    def __init__(self, overpass=None):
        self._overpass = overpass

    # ---------------------------------------------------------------- #
    def _run_overpass(self, lat, lon, radius_m, category_filter):
        q = (
            f'[out:json][timeout:35];'
            f'({category_filter}(around:{int(radius_m)},{lat},{lon}););'
            f'out center 60;'
        )
        if self._overpass:
            return self._overpass(q) or []
        return overpass_query(q)

    def list_nearby(self, lat: float, lon: float, radius_m: int = 2000,
                    categories: Optional[List[str]] = None,
                    limit: int = 40) -> Dict[str, Any]:
        """Return real geotagged listings near (lat,lon) for the chosen categories."""
        cats = categories or ["lodging", "commercial", "dining", "office"]
        out: Dict[str, Any] = {
            "status": "success", "latitude": lat, "longitude": lon,
            "radius_m": radius_m, "categories": cats,
            "count": 0, "listings": [], "queried_at": _now(),
        }
        for cat in cats:
            filt = CATEGORIES.get(cat)
            if not filt:
                continue
            for e in self._run_overpass(lat, lon, radius_m, filt):
                tags = e.get("tags", {})
                c = e.get("center") or {}
                elat = c.get("lat", e.get("lat"))
                elon = c.get("lon", e.get("lon"))
                if elat is None or elon is None:
                    continue
                listing = {
                    "name": _clean_name(tags),
                    "type": cat,
                    "osm_type": e.get("type", ""),
                    "osm_id": e.get("id"),
                    "address": tags.get("addr:housenumber", "") + " " + tags.get("addr:street", ""),
                    "latitude": float(elat), "longitude": float(elon),
                    "tags": {k: tags[k] for k in ("phone", "website", "operator", "brand",
                                                   "wikidata", "opening_hours", "stars") if k in tags},
                    "source": "osm/overpass",
                }
                out["listings"].append(listing)
            time.sleep(0.3)  # be polite to Overpass
        # dedup by (name,lat,lon rounded)
        seen = set()
        uniq = []
        for l in out["listings"]:
            key = (l["name"].lower(), round(l["latitude"], 4), round(l["longitude"], 4))
            if key not in seen:
                seen.add(key)
                uniq.append(l)
        uniq.sort(key=lambda l: l["name"].lower())
        out["listings"] = uniq[:limit]
        out["count"] = len(uniq[:limit])
        out["status"] = "success" if out["count"] else "limited"
        return out

    def from_address(self, address: str, radius_m: int = 2000,
                     categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Geocode an address (Nominatim, key-free), then list nearby listings."""
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q": address, "format": "json", "limit": 1},
                             headers={"User-Agent": "GeoVision-OSINT/2.0 (geolocation)"},
                             timeout=15)
            js = r.json()
        except Exception as e:
            return {"status": "failed", "note": f"geocode failed: {e}"}
        if not js:
            return {"status": "failed", "note": f"address not found: {address}"}
        lat = float(js[0]["lat"]); lon = float(js[0]["lon"])
        result = self.list_nearby(lat, lon, radius_m, categories)
        result["address"] = js[0].get("display_name", address)
        return result

    # ---------------------------------------------------------------- #
    # Tracking (change monitoring)                                      #
    # ---------------------------------------------------------------- #
    @staticmethod
    def _naming_key(listing: Dict[str, Any]) -> tuple:
        return (listing.get("name", "").lower().strip(),
                round(float(listing.get("latitude", 0)), 4),
                round(float(listing.get("longitude", 0)), 4))

    def track(self, lat: float, lon: float, radius_m: int = 1500,
              categories: Optional[List[str]] = None,
              previous: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Monitor a point: query fresh listings and diff against `previous` to
        report newly-appeared, still-present, and vanished listings.

        Pass `previous` = the earlier run's `listings` to get a change report.
        Persist `result['listings']` and reuse it next time to keep tracking.
        """
        result = self.list_nearby(lat, lon, radius_m, categories)
        if previous is None:
            result["change"] = None
            result["change_note"] = "no previous snapshot given; save result['listings'] to track"
            return result
        prev_keys = {self._naming_key(p) for p in previous if isinstance(p, dict)}
        cur_keys = {self._naming_key(c) for c in result["listings"]}
        new = [c for c in result["listings"] if self._naming_key(c) not in prev_keys]
        gone = [p for p in previous if isinstance(p, dict) and self._naming_key(p) not in cur_keys]
        result["change"] = {
            "new_this_run": new,
            "gone_since_last": gone,
            "new_count": len(new), "gone_count": len(gone),
        }
        result["change_note"] = (f"{len(new)} new, {len(gone)} gone since last snapshot")
        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        print(json.dumps(PropertyLocator().list_nearby(float(sys.argv[1]), float(sys.argv[2])),
                         indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(PropertyLocator().from_address("Eiffel Tower, Paris"),
                         indent=2, ensure_ascii=False, default=str))
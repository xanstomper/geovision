"""
GeoNames Client — online GeoNames API for richer city/country data
==================================================================
Provides both online (GeoNames web service) and offline (CityIndex) resolution.
The online API supports features like country names, admin1 names, alternate names,
and more detailed administrative lookups. Falls back gracefully when no GeoNames
username is configured.

Usage:
    client = GeoNamesClient()
    # Online lookup
    data = client.reverse_geocode(48.8584, 2.2945)
    # Offline snap
    snap = client.offline_snap(48.8584, 2.2945)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Online GeoNames API (optional — requires USERNAME)
# ---------------------------------------------------------------------------

class GeoNamesClient:
    """Thin client for the GeoNames web services."""

    BASE_URL = "http://api.geonames.org"

    def __init__(self, username: Optional[str] = None):
        self.username = username or os.environ.get("GEONAMES_USERNAME")
        self._online_enabled = bool(self.username)
        if not self._online_enabled:
            logger.debug("GeoNames username not configured — online lookups disabled")

    def _get(self, service: str, params: Dict[str, str]) -> Optional[Dict]:
        """Execute a GET request to the GeoNames web service."""
        import requests
        params.setdefault("username", self.username)
        url = f"{self.BASE_URL}/{service}"
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("GeoNames %s failed: %s", service, e)
            return None

    def reverse_geocode(self, lat: float, lon: float, max_rows: int = 1) -> Optional[Dict]:
        """Reverse geocode coordinates using GeoNames findNearbyNameJSON."""
        if not self._online_enabled:
            return None
        params = {"lat": str(lat), "lng": str(lon), "maxRows": str(max_rows)}
        data = self._get("findNearbyNameJSON", params)
        if data and "geonames" in data:
            return {
                "name": data["geonames"][0].get("name"),
                "country_code": data["geonames"][0].get("countryCode"),
                "country_name": data["geonames"][0].get("countryName"),
                "admin1": data["geonames"][0].get("adminName1"),
                "admin2": data["geonames"][0].get("adminName2"),
                "latitude": float(data["geonames"][0].get("lat", lat)),
                "longitude": float(data["geonames"][0].get("lng", lon)),
            }
        return None

    def search_by_name(
        self,
        name: str,
        country: Optional[str] = None,
        feature_class: Optional[str] = None,
        max_rows: int = 10,
    ) -> List[Dict]:
        """Search GeoNames by place name."""
        if not self._online_enabled:
            return []
        params = {"name": name, "maxRows": str(max_rows)}
        if country:
            params["country"] = country
        if feature_class:
            params["featureClass"] = feature_class
        data = self._get("searchJSON", params)
        results = []
        if data and "geonames" in data:
            for g in data["geonames"]:
                results.append({
                    "name": g.get("name"),
                    "asciiname": g.get("asciiname"),
                    "country_code": g.get("countryCode"),
                    "country_name": g.get("countryName"),
                    "admin1": g.get("adminName1"),
                    "feature_class": g.get("fclass"),
                    "feature_code": g.get("fcode"),
                    "latitude": float(g.get("lat", 0)),
                    "longitude": float(g.get("lng", 0)),
                    "population": int(g.get("population", 0)),
                })
        return results

    def country_info(self, country_code: str) -> Optional[Dict]:
        """Get country metadata from GeoNames."""
        if not self._online_enabled:
            return None
        data = self._get("countryInfoJSON", {"country": country_code})
        if data and "geonames" in data:
            c = data["geonames"][0]
            return {
                "country_code": c.get("countryCode"),
                "country_name": c.get("countryName"),
                "continent": c.get("continent"),
                "capital": c.get("capital"),
                "area_km2": float(c.get("areaInKmSq", 0)),
                "population": int(c.get("population", 0)),
                "languages": c.get("languages"),
                "currency": c.get("currencyCode"),
                "neighbors": c.get("neighbors", "").split(",") if c.get("neighbors") else [],
            }
        return None

    # -----------------------------------------------------------------------
    # Offline fallback via the embedded GeoNames dataset
    # -----------------------------------------------------------------------

    def offline_snap(self, lat: float, lon: float, max_km: float = 75.0) -> Optional[Dict]:
        """Offline nearest-city lookup using the embedded GeoNames dataset."""
        from .geonames_city_snap import get_city_index
        idx = get_city_index()
        return idx.snap(lat, lon, max_km=max_km)

    def offline_nearest(self, lat: float, lon: float, k: int = 3) -> List[Dict]:
        """Offline nearest-city list using the embedded GeoNames dataset."""
        from .geonames_city_snap import get_city_index
        idx = get_city_index()
        return idx.nearest(lat, lon, k=k)

    @property
    def is_online(self) -> bool:
        return self._online_enabled


if __name__ == "__main__":
    client = GeoNamesClient()
    # Test offline
    snap = client.offline_snap(48.8584, 2.2945)
    print(f"Offline snap: {snap}")
    nearest = client.offline_nearest(48.8584, 2.2945, k=3)
    print(f"Offline nearest: {nearest}")
    if client.is_online:
        rev = client.reverse_geocode(48.8584, 2.2945)
        print(f"Online reverse: {rev}")
        results = client.search_by_name("London", country="GB")
        print(f"Online search London,GB: {len(results)} results")

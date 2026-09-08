"""
Ground Imagery Client
=====================
Retrieves real ground-level photographs and street imagery near candidate GPS coordinates.
Uses Wikimedia Commons Geosearch API (un-gated, free, zero API key) and Mapillary v4 API
(if MAPILLARY_ACCESS_TOKEN is configured).

Enables side-by-side ground truth visual verification to corroborate geolocation estimates.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request
import json
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "GeoVision/2.0 (geospatial-osint@geovision.local)"


class GroundImageryClient:
    """Fetches real ground-truth photos and street imagery near any coordinate."""

    def __init__(self, mapillary_token: Optional[str] = None):
        self.mapillary_token = mapillary_token or os.environ.get("MAPILLARY_ACCESS_TOKEN")

    def search_wikimedia_ground_photos(
        self,
        lat: float,
        lon: float,
        radius_m: int = 1000,
        limit: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Query Wikimedia Commons Geosearch API for photos taken near the coordinates.
        Returns a list of geotagged images with titles, distance, coordinates, and thumbnail URLs.
        """
        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": min(10000, max(10, radius_m)),
            "gslimit": limit,
            "format": "json",
        }
        url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            items = data.get("query", {}).get("geosearch", [])
            results = []
            for item in items:
                title = item.get("title", "")
                pageid = item.get("pageid")
                dist = item.get("dist", 0.0)
                img_lat = item.get("lat")
                img_lon = item.get("lon")
                
                # Format direct thumbnail URL via Special:FilePath
                clean_title = title.replace(" ", "_")
                thumb_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(clean_title)}?width=640"
                page_url = f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(clean_title)}"

                results.append({
                    "source": "wikimedia_commons",
                    "title": title,
                    "page_id": pageid,
                    "distance_m": round(float(dist), 1),
                    "latitude": img_lat,
                    "longitude": img_lon,
                    "thumbnail_url": thumb_url,
                    "page_url": page_url,
                })
            return results
        except Exception as e:
            logger.warning(f"Wikimedia geosearch failed for ({lat}, {lon}): {e}")
            return []

    def search_mapillary_photos(
        self,
        lat: float,
        lon: float,
        radius_m: int = 500,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query Mapillary v4 API for street-level photos near coordinates if token is available."""
        if not self.mapillary_token:
            return []

        import math
        lat_delta = radius_m / 111000.0
        lon_delta = radius_m / (111000.0 * max(math.cos(math.radians(lat)), 0.01))
        bbox = f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}"

        url = (
            f"https://graph.mapillary.com/images?"
            f"fields=id,geometry,thumb_1024_url,thumb_256_url,compass_angle&"
            f"bbox={bbox}&limit={limit}"
        )
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"OAuth {self.mapillary_token}"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for feat in data.get("data", []):
                geom = feat.get("geometry", {}).get("coordinates", [])
                if len(geom) >= 2:
                    results.append({
                        "source": "mapillary",
                        "image_id": feat.get("id"),
                        "latitude": geom[1],
                        "longitude": geom[0],
                        "compass_angle": feat.get("compass_angle"),
                        "thumbnail_url": feat.get("thumb_1024_url") or feat.get("thumb_256_url"),
                    })
            return results
        except Exception as e:
            logger.warning(f"Mapillary search failed for ({lat}, {lon}): {e}")
            return []

    def get_nearby_ground_photos(
        self,
        lat: float,
        lon: float,
        radius_m: int = 1000,
        limit: int = 8
    ) -> Dict[str, Any]:
        """Fetch ground truth photos from all available services (Wikimedia + Mapillary)."""
        wiki_photos = self.search_wikimedia_ground_photos(lat, lon, radius_m=radius_m, limit=limit)
        mapillary_photos = self.search_mapillary_photos(lat, lon, radius_m=radius_m, limit=limit)

        combined = wiki_photos + mapillary_photos
        combined.sort(key=lambda x: x.get("distance_m", 99999))

        return {
            "latitude": lat,
            "longitude": lon,
            "search_radius_m": radius_m,
            "total_found": len(combined),
            "ground_photos": combined[:limit],
        }

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
import urllib.error
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
            "gslimit": str(limit),
            "gsnamespace": "6",  # files only — not arbitrary geotagged wiki pages
            "format": "json",
        }
        url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            items = data.get("query", {}).get("geosearch", [])
            # Resolve real, working thumbnail URLs via the Commons imageinfo API.
            # (Special:FilePath on raw/non-file titles 404s — imageinfo returns the
            # actual thumb endpoint that always serves image bytes.)
            thumb_urls = self._resolve_thumb_urls([it.get("title", "") for it in items])
            results = []
            for item, thumb_url in zip(items, thumb_urls):
                title = item.get("title", "")
                pageid = item.get("pageid")
                dist = item.get("dist", 0.0)
                img_lat = item.get("lat")
                img_lon = item.get("lon")

                clean_title = title.replace(" ", "_")
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

    def _resolve_thumb_urls(self, titles: List[str]) -> List[Optional[str]]:
        """Resolve real working thumbnail URLs for Commons file titles via imageinfo.

        Returns one URL per title (None if unresolvable). imageinfo returns the
        actual thumb endpoint; Special:FilePath 404s on some titles so we prefer
        imageinfo and fall back to Special:FilePath only as a last resort.
        """
        if not titles:
            return []
        resolved = [None] * len(titles)
        try:
            q = {
                "action": "query", "titles": "|".join(titles),
                "prop": "imageinfo", "iiprop": "url", "iiurlwidth": "640",
                "format": "json", "formatversion": "2",
            }
            url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(q)}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            norm = {p.get("title"): p for p in data.get("query", {}).get("pages", [])}
            for i, t in enumerate(titles):
                pg = norm.get(t) or norm.get(t.replace("_", " "))
                ii = (pg.get("imageinfo") or [{}])[0] if pg else {}
                resolved[i] = ii.get("thumburl")
        except Exception as e:
            logger.warning("imageinfo thumb resolution failed: %s", e)
        # Final fallback: Special:Redirect/file on the thumbnail width (works for most)
        for i, t in enumerate(titles):
            if resolved[i]:
                continue
            clean = t.replace(" ", "_").replace("File:", "")
            resolved[i] = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{urllib.parse.quote(clean)}?width=640"
        return resolved

    def search_mapillary_photos(
        self,
        lat: float,
        lon: float,
        radius_m: int = 250,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query Mapillary v4 API for street-level photos near coordinates if token is available.

        Mapillary's v4 API returns HTTP 500 ("reduce the amount of data") when given
        a large bbox or too many fields in dense urban areas. Keep the bbox small
        (<=300m) and the field set lean, and halve the radius on 500 automatically
        so it degrades gracefully instead of erroring out.
        """
        if not self.mapillary_token:
            return []
        import math

        radius_m = max(60, min(300, int(radius_m)))
        fields = "id,geometry,thumb_256_url,compass_angle"
        for attempt_radius in (radius_m, max(60, radius_m // 2)):
            lat_delta = attempt_radius / 111000.0
            lon_delta = attempt_radius / (111000.0 * max(math.cos(math.radians(lat)), 0.01))
            bbox = f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}"
            url = (f"https://graph.mapillary.com/images?fields={fields}"
                   f"&bbox={bbox}&limit={int(limit)}")
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"OAuth {self.mapillary_token}"})
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break  # succeeded on this radius
            except urllib.error.HTTPError as e:
                if e.code == 500:  # "reduce the amount of data" — try a tighter bbox once
                    continue
                logger.warning(f"Mapillary HTTP {e.code} at ({lat},{lon}): {e}")
                return []
            except Exception as e:
                logger.warning(f"Mapillary search failed for ({lat}, {lon}): {e}")
                return []
        else:
            return []  # even halved radius failed with 500

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
                    "thumbnail_url": feat.get("thumb_256_url") or feat.get("thumb_1024_url"),
                })
        return results

    def search_flickr_geotagged(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1.0,
        limit: int = 10,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Live geotagged photos near a coordinate via Flickr's PUBLIC geo feed.

        Key-free and un-gated: Flickr's `services/feeds/geo` public endpoint returns
        real geotagged photos (incl. interiors/living spaces people share) without an
        API key. `tags` (e.g. 'interior', 'kitchen') narrows to a content type.
        """
        params = {
            "format": "json", "nojsoncallback": "1",
            "lat": lat, "lon": lon,
            "radius": max(0.1, min(10.0, radius_km)),
        }
        if tags:
            params["tags"] = tags
        url = "https://api.flickr.com/services/feeds/geo?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=12.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            results = []
            for item in data.get("items", [])[:limit]:
                # Flickr public feed gives roughly-geotagged photos; use the photo page
                # link as the reference, and the thumbnail as a viewable image.
                thumb = (item.get("media") or {}).get("m") or ""
                results.append({
                    "source": "flickr",
                    "title": item.get("title", ""),
                    "photo_id": (item.get("link") or "").rsplit("/", 1)[-1],
                    "latitude": lat,
                    "longitude": lon,
                    "thumbnail_url": thumb,
                    "page_url": item.get("link"),
                    "distance_m": 0.0,
                })
            return results
        except Exception as e:
            logger.warning(f"Flickr geo feed failed for ({lat}, {lon}): {e}")
            return []

    def get_nearby_ground_photos(
        self,
        lat: float,
        lon: float,
        radius_m: int = 1000,
        limit: int = 8
    ) -> Dict[str, Any]:
        """Fetch ground truth photos from all available services (Wikimedia + Mapillary + Flickr)."""
        wiki_photos = self.search_wikimedia_ground_photos(lat, lon, radius_m=radius_m, limit=limit)
        mapillary_photos = self.search_mapillary_photos(lat, lon, radius_m=radius_m, limit=limit)
        flickr_photos = self.search_flickr_geotagged(
            lat, lon, radius_km=min(1.5, radius_m / 1000.0), limit=limit)

        combined = wiki_photos + mapillary_photos + flickr_photos
        combined.sort(key=lambda x: x.get("distance_m", 99999))

        return {
            "latitude": lat,
            "longitude": lon,
            "search_radius_m": radius_m,
            "total_found": len(combined),
            "ground_photos": combined[:limit],
        }

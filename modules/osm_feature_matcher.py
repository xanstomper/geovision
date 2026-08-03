import requests
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Multiple Overpass endpoints for fallback
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


class OSMFeatureMatcher:
    def __init__(self):
        self.headers = {"User-Agent": "GeoVision/1.0"}

    def _query(self, query: str) -> dict:
        """Try multiple Overpass endpoints with fallback."""
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                response = requests.post(endpoint, data=query, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"Overpass rate-limited at {endpoint}. Trying next...")
                    time.sleep(2)
                    continue
                elif response.status_code >= 500:
                    logger.warning(f"Overpass {response.status_code} at {endpoint}. Trying next...")
                    continue
            except requests.Timeout:
                logger.warning(f"Overpass timeout at {endpoint}. Trying next...")
                continue
            except Exception as e:
                logger.error(f"Overpass error at {endpoint}: {e}")
                continue
        logger.error("All Overpass endpoints failed")
        return {}

    def find_pois(self, lat: float, lon: float, radius_m: int = 500) -> dict:
        query = f"""
        [out:json];
        (
          node["amenity"](around:{radius_m},{lat},{lon});
          node["shop"](around:{radius_m},{lat},{lon});
        );
        out center 50;
        """
        data = self._query(query)
        elements = data.get("elements", [])
        
        pois = []
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", "Unknown")
            poi_type = tags.get("amenity") or tags.get("shop")
            pois.append({"name": name, "type": poi_type, "lat": el.get("lat"), "lon": el.get("lon")})
            
        return {
            "count": len(pois),
            "pois": pois
        }

    def get_road_characteristics(self, lat: float, lon: float) -> dict:
        query = f"""
        [out:json];
        way["highway"](around:200,{lat},{lon});
        out tags;
        """
        data = self._query(query)
        elements = data.get("elements", [])
        
        highways = []
        surfaces = []
        for el in elements:
            tags = el.get("tags", {})
            if "highway" in tags:
                highways.append(tags["highway"])
            if "surface" in tags:
                surfaces.append(tags["surface"])
                
        return {
            "highway_types": list(set(highways)),
            "surface_types": list(set(surfaces))
        }

    def get_building_density(self, lat: float, lon: float, radius_m: int = 500) -> dict:
        query = f"""
        [out:json];
        way["building"](around:{radius_m},{lat},{lon});
        out count;
        """
        data = self._query(query)
        elements = data.get("elements", [])
        
        count = 0
        if elements:
            count = int(elements[0].get("tags", {}).get("total", 0) or elements[0].get("count", {}).get("ways", 0))
            if count == 0 and "count" in elements[0]:
                count = elements[0]["count"].get("ways", len(elements))
                
        return {
            "building_count": count,
            "density_score": min(count / 100.0, 1.0)
        }

import requests
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class OSMFeatureMatcher:
    def __init__(self):
        self.url = "https://overpass-api.de/api/interpreter"
        self.headers = {"User-Agent": "GeoVision/1.0"}

    def _query(self, query: str) -> dict:
        try:
            response = requests.post(self.url, data=query, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Overpass API query failed: {e}")
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

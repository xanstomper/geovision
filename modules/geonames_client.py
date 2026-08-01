import os
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class GeoNamesClient:
    def __init__(self, username: Optional[str] = None):
        self.username = username or os.environ.get("GEONAMES_USERNAME", "geovision")
        self.base_url = "http://api.geonames.org"

    def search(self, query: str) -> List[dict]:
        url = f"{self.base_url}/searchJSON"
        params = {
            "q": query,
            "maxRows": 10,
            "username": self.username
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("geonames", [])
        except Exception as e:
            logger.error(f"GeoNames search failed: {e}")
            return []

    def reverse_geocode(self, lat: float, lon: float) -> dict:
        url = f"{self.base_url}/findNearbyPlaceNameJSON"
        params = {
            "lat": lat,
            "lng": lon,
            "username": self.username
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            geonames = data.get("geonames", [])
            if geonames:
                return geonames[0]
            return {}
        except Exception as e:
            logger.error(f"GeoNames reverse geocode failed: {e}")
            return {}

    def find_nearby(self, lat: float, lon: float, feature_class: Optional[str] = None) -> List[dict]:
        url = f"{self.base_url}/findNearbyJSON"
        params = {
            "lat": lat,
            "lng": lon,
            "username": self.username
        }
        if feature_class:
            params["featureClass"] = feature_class
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("geonames", [])
        except Exception as e:
            logger.error(f"GeoNames find nearby failed: {e}")
            return []

import time
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NominatimGeocoder:
    """Free reverse/forward geocoding using OpenStreetMap Nominatim API."""
    
    BASE_URL = "https://nominatim.openstreetmap.org"
    USER_AGENT = "GeoVision/1.0 (contact@example.com)"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        
    def _respect_rate_limit(self):
        time.sleep(1.0)
        
    def reverse_geocode(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        self._respect_rate_limit()
        url = f"{self.BASE_URL}/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2"
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Reverse geocode failed: {e}")
            return None
            
    def forward_geocode(self, query: str) -> Optional[Dict[str, Any]]:
        self._respect_rate_limit()
        url = f"{self.BASE_URL}/search"
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                return data[0]
            return None
        except requests.RequestException as e:
            logger.error(f"Forward geocode failed: {e}")
            return None
            
    def validate_estimate(self, lat: float, lon: float) -> Dict[str, Any]:
        result = self.reverse_geocode(lat, lon)
        
        if result and "error" not in result:
            address = result.get("address", {})
            return {
                "is_on_land": True,
                "country": address.get("country"),
                "state": address.get("state"),
                "city": address.get("city") or address.get("town") or address.get("village")
            }
        
        # If error or no result, it might be in the ocean
        return {
            "is_on_land": False,
            "country": None,
            "state": None,
            "city": None
        }

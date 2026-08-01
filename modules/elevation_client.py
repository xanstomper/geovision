import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ElevationClient:
    def __init__(self):
        self.url = "https://api.open-elevation.com/api/v1/lookup"
        self.headers = {
            "User-Agent": "GeoVision/1.0"
        }

    def get_elevation(self, lat: float, lon: float) -> dict:
        params = {
            "locations": f"{lat},{lon}"
        }
        try:
            response = requests.get(self.url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                return {"elevation": results[0].get("elevation"), "lat": lat, "lon": lon}
            return {"elevation": None, "lat": lat, "lon": lon}
        except Exception as e:
            logger.error(f"Elevation lookup failed: {e}")
            try:
                om_url = "https://api.open-meteo.com/v1/elevation"
                om_params = {"latitude": lat, "longitude": lon}
                om_response = requests.get(om_url, params=om_params, headers=self.headers, timeout=10)
                om_response.raise_for_status()
                om_data = om_response.json()
                if "elevation" in om_data and om_data["elevation"]:
                    return {"elevation": om_data["elevation"][0], "lat": lat, "lon": lon}
            except Exception as e2:
                logger.error(f"Open-Meteo fallback failed: {e2}")
            return {"elevation": None, "lat": lat, "lon": lon}

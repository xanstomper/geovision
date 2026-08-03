import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WeatherCorroborator:
    """
    Advanced OSINT Weather Corroboration.
    Cross-references the EXIF timestamp of an image with historical weather APIs (Open-Meteo) 
    to eliminate candidate locations that had fundamentally different weather conditions 
    (e.g., eliminating a city that was raining if the photo shows clear sunny skies).
    """
    def __init__(self):
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}

    def fetch_historical_weather(self, lat: float, lon: float, date_str: str) -> Optional[Dict[str, Any]]:
        """Fetches historical daily weather for a specific lat/lon and date (YYYY-MM-DD)."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "weather_code,precipitation_sum,snowfall_sum,temperature_2m_max",
            "timezone": "auto"
        }
        try:
            resp = requests.get(self.archive_url, params=params, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "daily" in data:
                    daily = data["daily"]
                    return {
                        "weather_code": daily.get("weather_code", [None])[0],
                        "precipitation_mm": daily.get("precipitation_sum", [0])[0],
                        "snowfall_cm": daily.get("snowfall_sum", [0])[0],
                        "max_temp_c": daily.get("temperature_2m_max", [None])[0]
                    }
            else:
                logger.warning(f"Open-Meteo returned {resp.status_code}")
        except Exception as e:
            logger.error(f"Weather API request failed: {e}")
        return None

    def interpret_wmo_code(self, code: int) -> str:
        """Converts WMO weather codes to human-readable strings for VLM comparison."""
        if code is None:
            return "unknown"
        if code <= 3:
            return "clear/cloudy"
        if 45 <= code <= 48:
            return "foggy"
        if 51 <= code <= 67 or 80 <= code <= 82:
            return "rain"
        if 71 <= code <= 77 or 85 <= code <= 86:
            return "snow"
        if 95 <= code <= 99:
            return "thunderstorm"
        return "mixed"

    def verify_candidates(self, candidates: List[Dict[str, Any]], exif_data: Dict[str, Any], vlm_weather: str, date_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Adjusts the confidence of candidate locations based on historical weather.
        """
        date_str = None
        if date_override:
            # Assume date_override is in YYYY-MM-DD format
            date_str = date_override
        elif exif_data and exif_data.get("EXIF DateTimeOriginal"):
            try:
                date_full = str(exif_data["EXIF DateTimeOriginal"]).strip()
                dt = datetime.strptime(date_full, "%Y:%m:%d %H:%M:%S")
                date_str = dt.strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"  [-] Could not parse EXIF date for weather check: {e}")
        
        if not date_str:
            logger.info("  [-] No EXIF date or override date found. Skipping weather corroboration.")
            return candidates

        vlm_weather_low = vlm_weather.lower() if vlm_weather else ""
        logger.info(f"  [*] Cross-referencing historical weather for {date_str} (Observed: {vlm_weather_low})")

        for cand in candidates:
            lat = cand.get("latitude")
            lon = cand.get("longitude")
            if not lat or not lon:
                continue
                
            historical = self.fetch_historical_weather(lat, lon, date_str)
            if not historical:
                continue
                
            actual_weather_code = historical["weather_code"]
            actual_desc = self.interpret_wmo_code(actual_weather_code)
            
            cand["historical_weather"] = {
                "date": date_str,
                "wmo_code": actual_weather_code,
                "description": actual_desc,
                "precip_mm": historical.get("precipitation_mm"),
                "temp_c": historical.get("max_temp_c")
            }
            
            # Simple conflict detection
            conflict = False
            if "rain" in vlm_weather_low and actual_desc not in ["rain", "thunderstorm", "mixed"]:
                conflict = True
            elif "snow" in vlm_weather_low and actual_desc != "snow" and historical.get("snowfall_cm", 0) == 0:
                conflict = True
            elif "sunny" in vlm_weather_low or "clear" in vlm_weather_low:
                if actual_desc in ["rain", "snow", "thunderstorm"]:
                    conflict = True
            
            if conflict:
                logger.warning(f"  [!] WEATHER CONFLICT at {lat:.2f}, {lon:.2f}: Photo looks '{vlm_weather_low}', but historical data says '{actual_desc}'. Penalizing confidence.")
                cand["confidence"] = cand.get("confidence", 0.5) * 0.5  # Slash confidence by half
                cand["evidence"] = cand.get("evidence", {})
                cand["evidence"]["weather_conflict"] = f"Photo shows {vlm_weather_low}, historical data was {actual_desc}"
            else:
                logger.info(f"  [+] Weather verified at {lat:.2f}, {lon:.2f} ({actual_desc})")
                cand["evidence"] = cand.get("evidence", {})
                cand["evidence"]["weather_verified"] = actual_desc

        return candidates

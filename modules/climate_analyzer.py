import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ClimateAnalyzer:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
    def analyze_location(self, lat: float, lon: float) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto",
            "past_days": 30
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            if not daily:
                return {"error": "No daily data"}
                
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precips = daily.get("precipitation_sum", [])
            
            valid_max = [t for t in max_temps if t is not None]
            valid_min = [t for t in min_temps if t is not None]
            valid_precip = [p for p in precips if p is not None]
            
            avg_temp = None
            if valid_max and valid_min:
                avg_temp = (sum(valid_max)/len(valid_max) + sum(valid_min)/len(valid_min)) / 2.0
                
            avg_precip = None
            if valid_precip:
                avg_precip = sum(valid_precip) / len(valid_precip)
                
            climate_zone = "Unknown"
            if avg_temp is not None:
                if avg_temp > 25:
                    climate_zone = "Tropical/Arid"
                elif avg_temp > 15:
                    climate_zone = "Temperate"
                elif avg_temp > 5:
                    climate_zone = "Continental"
                else:
                    climate_zone = "Polar/Alpine"
                    
            return {
                "climate_zone": climate_zone,
                "avg_temp": avg_temp,
                "avg_precip": avg_precip,
                "confidence": 0.7 if avg_temp is not None else 0.0
            }
        except Exception as e:
            logger.error(f"Climate analysis failed: {e}")
            return {"error": str(e), "confidence": 0.0}

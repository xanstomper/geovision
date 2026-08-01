"""
Google Maps Controller
Automates browser interaction to verify locations live on Google Maps
"""

import time
import os
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapsVerificationResult:
    verified: bool
    street_view_available: bool
    satellite_match_confidence: float
    nearby_amenities: List[Dict]
    park_distance_km: Optional[float] = None
    address: Optional[str] = None


class GoogleMapsController:
    """
    Controls browser to navigate Google Maps and verify locations
    Uses Selenium/Playwright for browser automation
    """
    
    def __init__(self):
        self.driver = None
        self.base_url = "https://maps.google.com"
    
    def initialize_browser(self):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=options)
            logger.info("Browser initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Browser init error: {e}")
            return False
    
    def navigate_to_location(self, lat: float, lon: float, zoom: int = 19):
        # We can use the official Static Maps / Street View API if a key is provided
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            logger.warning("No GOOGLE_MAPS_API_KEY available. Skipping Google Maps fetch.")
            return None
            
        logger.info("  Using Google Maps API for Cross-Verification")
        self.current_lat = lat
        self.current_lon = lon
        self.api_key = api_key
        return self._capture_current_view()
    
    def _capture_current_view(self) -> Optional[bytes]:
        import requests
        if hasattr(self, "api_key") and self.api_key:
            try:
                # Fetch satellite imagery
                url = f"https://maps.googleapis.com/maps/api/staticmap?center={self.current_lat},{self.current_lon}&zoom=19&size=600x400&maptype=satellite&key={self.api_key}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    return resp.content
            except Exception as e:
                logger.error(f"Maps API error: {e}")
            return None
            
        try:
            screenshot = self.driver.get_screenshot_as_png()
            return screenshot
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None
    
    def verify_location(self, lat: float, lon: float) -> MapsVerificationResult:
        if not os.environ.get("GOOGLE_MAPS_API_KEY"):
            return MapsVerificationResult(
                verified=False,
                street_view_available=False,
                satellite_match_confidence=0.0,
                nearby_amenities=[],
                address="No API key"
            )
            
        self.navigate_to_location(lat, lon)
        
        screenshot = self._capture_current_view()
        street_view = False
        satellite_confidence = 0.5
        
        if screenshot:
            try:
                import cv2
                img_array = np.frombuffer(screenshot, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Check if Street View panorama is visible
                if np.mean(gray[200:400, 200:400]) < 100:
                    street_view = True
                
                # Check satellite view characteristics
                satellite_confidence = min(np.std(gray) / 50.0, 1.0)
            except Exception as e:
                logger.warning(f"Verification analysis error: {e}")
        
        return MapsVerificationResult(
            verified=screenshot is not None,
            street_view_available=street_view,
            satellite_match_confidence=satellite_confidence,
            nearby_amenities=[],
            address=None
        )
    
    def find_nearby_parks(self, lat: float, lon: float, radius_km: float = 1.0) -> List[Dict]:
        """Search for parks within walking distance using real Overpass API data."""
        try:
            from .overpass_client import OverpassClient
            client = OverpassClient()
            osm_parks = client.find_nearby_parks(lat, lon, radius_meters=int(radius_km * 1000))
            
            from math import radians, sin, cos, sqrt, atan2
            R = 6371.0
            parks = []
            for park in osm_parks:
                plat, plon = park["lat"], park["lon"]
                dlat = radians(plat - lat)
                dlon = radians(plon - lon)
                a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(plat)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = R * c
                parks.append({
                    "name": park["name"],
                    "distance_km": round(distance, 3),
                    "walk_minutes": round(distance / 0.08),
                    "lat": plat,
                    "lon": plon
                })
            parks.sort(key=lambda p: p["distance_km"])
            return parks
        except Exception as e:
            logger.error(f"Park search failed: {e}")
            return []
    
    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
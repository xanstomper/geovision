"""
Satellite Imagery Matcher
Real implementation using Google Static Maps API and OpenCV to compare ground features to satellite colors/density.
"""

import numpy as np
import requests
import cv2
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import math
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class SatelliteMatch:
    latitude: float
    longitude: float
    confidence: float
    match_type: str
    metadata: Dict = field(default_factory=dict)


class SatelliteMatcher:
    """
    Validates visual features against real satellite/aerial imagery using Google Maps API.
    """
    
    def __init__(self, google_maps_api_key=None):
        self.api_key = google_maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        self.cache_dir = Path("/home/jewboy420/geovision/data/satellite_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def match_features_to_satellite(self, features, center_lat, center_lon, radius_km=5.0):
        matches = []
        if center_lat is None or center_lon is None:
            return matches

        logger.info(f"  [SatelliteMatcher] Validating coordinates {center_lat}, {center_lon}...")
        
        # We sample a few points around the center to find the best match
        samples = [
            (center_lat, center_lon),
            (center_lat + 0.005, center_lon),
            (center_lat - 0.005, center_lon),
            (center_lat, center_lon + 0.005),
            (center_lat, center_lon - 0.005)
        ]
        
        for lat, lon in samples:
            sat_img = self.fetch_satellite_tile(lat, lon, zoom=18)
            if sat_img is not None:
                # Calculate satellite image mean color
                sat_mean_color = cv2.mean(sat_img)[:3]
                
                # Check vegetation density based on green pixels
                hsv = cv2.cvtColor(sat_img, cv2.COLOR_BGR2HSV)
                # Green mask
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                green_ratio = np.count_nonzero(mask) / (sat_img.shape[0] * sat_img.shape[1])
                
                # Compare to ground features
                confidence = 0.5 # Base confidence for having a tile
                
                # Adjust confidence based on expected vegetation
                expected_density = getattr(features, 'vegetation_density', 0.0)
                if expected_density > 0:
                    diff = abs(green_ratio - expected_density)
                    confidence += (0.2 * (1 - diff))
                
                # Determine match type based on satellite dominant color
                match_type = "urban_built_up"
                if green_ratio > 0.4:
                    match_type = "rural_or_park"
                elif green_ratio > 0.15:
                    match_type = "suburban"
                
                matches.append(SatelliteMatch(
                    latitude=lat,
                    longitude=lon,
                    confidence=min(0.95, confidence),
                    match_type=f"satellite_validated:{match_type}",
                    metadata={"green_ratio": round(green_ratio, 3)}
                ))
            else:
                logger.warning(f"  [SatelliteMatcher] Failed to fetch tile for {lat}, {lon}")
                
        # Sort by confidence
        matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        return matches[:3]
    
    def fetch_satellite_tile(self, lat, lon, zoom=18):
        cache_file = self.cache_dir / f"sat_{lat}_{lon}_{zoom}.jpg"
        if cache_file.exists():
            return cv2.imread(str(cache_file))

        headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}

        if self.api_key:
            try:
                url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={self.api_key}"
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imwrite(str(cache_file), img)
                        return img
                logger.warning(f"Google Maps API request returned status code {response.status_code}. Falling back to OpenStreetMap.")
            except Exception as e:
                logger.warning(f"Google Maps API fetch error: {e}. Falling back to OpenStreetMap.")

        # Fallback to OpenStreetMap tile if Google API key is missing or failed
        try:
            url = f"https://tile.openstreetmap.org/{zoom}/{self._lon2tile(lon, zoom)}/{self._lat2tile(lat, zoom)}.png"
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite(str(cache_file), img)
                    return img
            else:
                logger.error(f"OSM tile fetch returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"OSM tile fetch error: {e}")

        return None

    def _lon2tile(self, lon, zoom):
        return int((lon + 180.0) / 360.0 * (2.0 ** zoom))

    def _lat2tile(self, lat, zoom):
        lat_rad = math.radians(lat)
        return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2.0 ** zoom))
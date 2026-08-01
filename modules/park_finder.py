"""
Park Finder Module
Locates parks near given coordinates using spatial analysis and OpenStreetMap (Overpass API).
"""

from typing import List, Dict, Optional
from math import radians, sin, cos, sqrt, atan2
import logging

from .overpass_client import OverpassClient

logger = logging.getLogger(__name__)

def find_nearby_parks(lat: float, lon: float, max_distance_km: float = 2.0) -> List[Dict]:
    """Find parks within walking distance (8-10 min = ~1km) using public OSM data."""
    parks = []
    
    # Radius in meters for Overpass query
    radius_meters = int(max_distance_km * 1000)
    
    logger.info(f"    Querying OpenStreetMap (Overpass) for parks within {radius_meters}m...")
    client = OverpassClient()
    osm_parks = client.find_nearby_parks(lat, lon, radius_meters=radius_meters)
    
    R = 6371.0
    
    for park in osm_parks:
        plat = park["lat"]
        plon = park["lon"]
        
        # Calculate precise distance
        dlat = radians(plat - lat)
        dlon = radians(plon - lon)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(plat)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_km = R * c
        
        if distance_km <= max_distance_km:
            walk_time = round(distance_km / 0.08)  # 80m/min
            
            parks.append({
                "name": park["name"],
                "distance_km": round(distance_km, 3),
                "walk_minutes": walk_time,
                "lat": plat,
                "lon": plon,
                "within_8_10_min": 8 <= walk_time <= 10
            })
    
    parks.sort(key=lambda p: p["distance_km"])
    return parks
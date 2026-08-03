import requests
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class OverpassClient:
    """
    A client to interact with OpenStreetMap's Overpass API.
    Provides access to global public infrastructure, buildings, parks, and amenities.
    """
    def __init__(self):
        self.endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        self.headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}

    def _query_overpass(self, query: str) -> Optional[dict]:
        import time
        for endpoint in self.endpoints:
            try:
                resp = requests.post(endpoint, data={'data': query}, headers=self.headers, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    logger.warning(f"  [!] Overpass rate-limited at {endpoint}. Trying next...")
                    time.sleep(2)
                    continue
                elif resp.status_code >= 500:
                    logger.warning(f"  [!] Overpass {resp.status_code} at {endpoint}. Trying next...")
                    continue
                else:
                    logger.error(f"  [!] Overpass API error at {endpoint}: {resp.status_code}")
            except requests.Timeout:
                logger.warning(f"  [!] Overpass timeout at {endpoint}. Trying next...")
                continue
            except Exception as e:
                logger.error(f"  [!] Overpass request failed at {endpoint}: {e}")
                continue
        return None

    def find_nearby_parks(self, lat: float, lon: float, radius_meters: int = 1500) -> List[Dict]:
        """Find parks near a specific coordinate using OSM."""
        query = f"""
        [out:json][timeout:25];
        (
          node["leisure"="park"](around:{radius_meters},{lat},{lon});
          way["leisure"="park"](around:{radius_meters},{lat},{lon});
          relation["leisure"="park"](around:{radius_meters},{lat},{lon});
          way["boundary"="national_park"](around:{radius_meters},{lat},{lon});
        );
        out center;
        """
        try:
            data = self._query_overpass(query)
            if data:
                parks = []
                for element in data.get('elements', []):
                    tags = element.get('tags', {})
                    name = tags.get('name', 'Unnamed Park')
                    
                    # Elements of type 'way' or 'relation' with 'out center' have 'center' lat/lon
                    center = element.get('center', {})
                    plat = center.get('lat')
                    plon = center.get('lon')
                    
                    if not plat or not plon:
                        continue
                    
                    parks.append({
                        "name": name,
                        "lat": plat,
                        "lon": plon,
                        "tags": tags
                    })
                return parks
        except Exception as e:
            logger.error(f"Overpass request failed: {e}")
        return []

    def search_amenities_by_name(self, name_query: str, lat: float, lon: float, radius_meters: int = 50000) -> List[Dict]:
        """Search for amenities (like 'Tim Hortons') near a location."""
        # Case insensitive regex match for name
        safe_name = name_query.replace('"', '\\"')
        query = f"""
        [out:json][timeout:25];
        (
          node["name"~"{safe_name}", i](around:{radius_meters},{lat},{lon});
          way["name"~"{safe_name}", i](around:{radius_meters},{lat},{lon});
        );
        out center;
        """
        try:
            data = self._query_overpass(query)
            if data:
                results = []
                for element in data.get('elements', []):
                    tags = element.get('tags', {})
                    r_name = tags.get('name', 'Unknown')
                    
                    plat = element.get('lat') or element.get('center', {}).get('lat')
                    plon = element.get('lon') or element.get('center', {}).get('lon')
                    
                    if not plat or not plon:
                        continue
                        
                    results.append({
                        "name": r_name,
                        "lat": plat,
                        "lon": plon,
                        "type": tags.get('amenity', tags.get('shop', 'unknown'))
                    })
                return results
        except Exception as e:
            logger.error(f"Overpass request failed: {e}")
        return []

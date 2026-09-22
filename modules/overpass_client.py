import requests
import json
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class OverpassClient:
    """
    A client to interact with OpenStreetMap's Overpass API.
    Provides access to global public infrastructure, buildings, parks, and amenities.
    """
    def __init__(self, timeout: int = 30):
        self.endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        self.headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}
        self.timeout = timeout

    def _query_overpass(self, query: str) -> Optional[dict]:
        import time
        for endpoint in self.endpoints:
            try:
                resp = requests.post(endpoint, data={'data': query}, headers=self.headers, timeout=self.timeout)
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

    def query_nearby(self, lat: float, lon: float, radius: int = 1500) -> Dict[str, Any]:
        """Query diverse nearby infrastructure features: amenities, tourism, historic, and public transit."""
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          node["amenity"](around:{radius},{lat},{lon});
          node["tourism"](around:{radius},{lat},{lon});
          node["historic"](around:{radius},{lat},{lon});
          node["highway"="bus_stop"](around:{radius},{lat},{lon});
          way["building"](around:{radius},{lat},{lon});
        );
        out center 60;
        """
        features = []
        try:
            data = self._query_overpass(query)
            if data and "elements" in data:
                for el in data.get("elements", []):
                    tags = el.get("tags", {})
                    name = tags.get("name") or tags.get("description")
                    category = (
                        tags.get("amenity")
                        or tags.get("tourism")
                        or tags.get("historic")
                        or tags.get("building")
                        or tags.get("highway")
                        or "feature"
                    )
                    plat = el.get("lat") or el.get("center", {}).get("lat")
                    plon = el.get("lon") or el.get("center", {}).get("lon")
                    if plat is not None and plon is not None:
                        features.append({
                            "name": name,
                            "type": category,
                            "lat": plat,
                            "lon": plon,
                            "tags": tags,
                        })
        except Exception as e:
            logger.error(f"Overpass query_nearby failed: {e}")
        return {"features": features, "count": len(features)}

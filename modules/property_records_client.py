import requests
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PropertyRecordsClient:
    """
    Client for accessing open property, zoning, and land-use records.
    Crucial for US and Canada OSINT geolocation datasets (mimicking GeoSpy).
    """
    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    def find_land_use_records(self, lat: float, lon: float, radius_m: int = 1000) -> List[Dict]:
        """
        Retrieves detailed zoning and property boundaries (commercial, residential, retail) 
        from OpenStreetMap cadastral/land-use data.
        """
        query = f"""
        [out:json][timeout:25];
        (
          way["landuse"](around:{radius_m},{lat},{lon});
          way["building"](around:{radius_m},{lat},{lon});
          way["boundary"="administrative"](around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        OVERPASS_ENDPOINTS = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        
        headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}
        import time
        
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                response = requests.post(endpoint, data={'data': query}, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    records = []
                    for el in data.get('elements', []):
                        if el['type'] == 'way' and 'tags' in el:
                            records.append(el['tags'])
                    return records
                elif response.status_code == 429:
                    logger.warning(f"  [!] Overpass rate-limited at {endpoint}. Trying next...")
                    time.sleep(2)
                    continue
                elif response.status_code >= 500:
                    logger.warning(f"  [!] Overpass {response.status_code} at {endpoint}. Trying next...")
                    continue
                else:
                    logger.warning(f"  [!] Overpass returned {response.status_code} at {endpoint}")
            except requests.Timeout:
                logger.warning(f"  [!] Overpass timeout at {endpoint}. Trying next...")
                continue
            except Exception as e:
                logger.error(f"Failed to query property records at {endpoint}: {e}")
                continue
            
        logger.error("  [!] All Overpass endpoints failed for property records")
        return []



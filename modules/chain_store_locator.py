"""
Chain Store Locator — Real Overpass API chain store geolocation.
When OCR detects a known chain store name, queries OSM for all real locations.
"""
import requests
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Multiple Overpass endpoints for fallback
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


class ChainStoreLocator:
    """Finds exact locations of chain stores using the Overpass API directly."""

    def __init__(self):
        self.headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}

    def _query_overpass(self, query: str, timeout: int = 30) -> Optional[dict]:
        """Try multiple Overpass endpoints with retry logic."""
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(endpoint, data={'data': query},
                                     headers=self.headers, timeout=timeout)
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
                    logger.warning(f"  [!] Overpass returned {resp.status_code} at {endpoint}")
            except requests.Timeout:
                logger.warning(f"  [!] Overpass timeout at {endpoint}. Trying next...")
                continue
            except Exception as e:
                logger.error(f"  [!] Overpass error at {endpoint}: {e}")
                continue
        return None

    def locate_chain_stores(self, store_names: List[str],
                            region_hint_lat: float = None,
                            region_hint_lon: float = None) -> List[Dict]:
        """
        Query OSM Overpass for real locations of chain stores.
        If region_hint is provided, searches within 200km radius (state-level coverage).
        """
        results = []
        if not store_names:
            return results

        # 200km radius for state-level coverage instead of the original 50km
        radius = 200000

        for store in store_names:
            safe_store = store.replace('"', '\\"')

            if region_hint_lat is not None and region_hint_lon is not None:
                query = f"""
                [out:json][timeout:30];
                (
                  node["name"~"{safe_store}",i](around:{radius},{region_hint_lat},{region_hint_lon});
                  way["name"~"{safe_store}",i](around:{radius},{region_hint_lat},{region_hint_lon});
                );
                out center;
                """
            else:
                # Without region hint, skip — global queries are too slow
                logger.info(f"  [-] No region hint for chain store '{store}', skipping Overpass query")
                continue

            data = self._query_overpass(query)
            if data is not None:
                elements = data.get('elements', [])
                for el in elements:
                    lat = el.get('lat')
                    lon = el.get('lon')

                    if not lat and 'center' in el:
                        lat = el['center'].get('lat')
                        lon = el['center'].get('lon')

                    name = el.get('tags', {}).get('name', 'Unknown')
                    if lat and lon:
                        results.append({
                            "store_name": store,
                            "osm_name": name,
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "confidence": 0.95
                        })
                logger.info(f"  [+] Found {len(elements)} '{store}' locations via Overpass")
            else:
                logger.warning(f"  [!] All Overpass endpoints failed for '{store}'")

        return results

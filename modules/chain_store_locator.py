"""
Chain Store Locator — Real Overpass API chain store geolocation.
When OCR detects a known chain store name, queries OSM for all real locations.
"""
import requests
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChainStoreLocator:
    """Finds exact locations of chain stores using the Overpass API directly."""

    def __init__(self):
        self.endpoint = "https://overpass-api.de/api/interpreter"
        self.headers = {'User-Agent': 'GeoVision OSINT Tool/2.0 (contact: admin@geovision.local)'}

    def locate_chain_stores(self, store_names: List[str],
                            region_hint_lat: float = None,
                            region_hint_lon: float = None) -> List[Dict]:
        """
        Query OSM Overpass for real locations of chain stores.
        If region_hint is provided, searches within 50km radius.
        """
        results = []
        if not store_names:
            return results

        radius = 50000  # 50km search radius

        for store in store_names:
            safe_store = store.replace('"', '\\"')

            if region_hint_lat is not None and region_hint_lon is not None and region_hint_lat != 0:
                query = f"""
                [out:json][timeout:25];
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

            try:
                resp = requests.post(self.endpoint, data={'data': query},
                                     headers=self.headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
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
                elif resp.status_code == 429:
                    logger.warning(f"  [!] Overpass rate-limited for '{store}'. Skipping.")
                else:
                    logger.warning(f"  [!] Overpass returned {resp.status_code} for '{store}'")
            except Exception as e:
                logger.error(f"  [!] Error querying Overpass API for {store}: {e}")

        return results

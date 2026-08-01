import logging
from typing import List, Dict

try:
    from modules.overpass_client import OverpassClient
except ImportError:
    # Fallback or simple mock client if not exists
    class OverpassClient:
        def query(self, query_str): return {"elements": []}

logger = logging.getLogger(__name__)

class ChainStoreLocator:
    """Finds exact locations of chain stores using Overpass API."""
    
    def __init__(self):
        self.client = OverpassClient()
        
    def locate_chain_stores(self, store_names: List[str], region_hint_lat: float = None, region_hint_lon: float = None) -> List[Dict]:
        results = []
        if not store_names:
            return results
            
        radius = 50000  # 50km
        for store in store_names:
            safe_store = store.replace('"', '\\"')
            
            if region_hint_lat is not None and region_hint_lon is not None:
                # Query within radius
                query = f"""
                [out:json][timeout:25];
                (
                  node["name"~"(?i){safe_store}"](around:{radius},{region_hint_lat},{region_hint_lon});
                  way["name"~"(?i){safe_store}"](around:{radius},{region_hint_lat},{region_hint_lon});
                );
                out center;
                """
            else:
                # Global query (can be very slow, limiting to 50 results)
                query = f"""
                [out:json][timeout:25];
                (
                  node["name"~"(?i){safe_store}"];
                  way["name"~"(?i){safe_store}"];
                );
                out center 50;
                """
                
            try:
                response = self.client.query(query)
                elements = response.get('elements', [])
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
            except Exception as e:
                logger.error(f"Error querying Overpass API for {store}: {e}")
                
        return results

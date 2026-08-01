import requests
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class WikimediaClient:
    """
    Client for interacting with Wikimedia APIs.
    Used to access massive public imaging records and databases.
    """
    def __init__(self):
        self.commons_endpoint = "https://commons.wikimedia.org/w/api.php"
        self.wiki_endpoint = "https://en.wikipedia.org/w/api.php"

    def search_images_by_location(self, lat: float, lon: float, radius_meters: int = 1000) -> List[Dict]:
        """Search Wikimedia Commons for images taken near a specific coordinate."""
        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": radius_meters,
            "gsnamespace": "6", # File namespace
            "gslimit": "10",
            "format": "json"
        }
        headers = {'User-Agent': 'GeoVision OSINT Tool/1.0 (contact: admin@geovision.local)'}
        try:
            resp = requests.get(self.commons_endpoint, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("geosearch", [])
                results = []
                for p in pages:
                    results.append({
                        "title": p.get("title"),
                        "lat": p.get("lat"),
                        "lon": p.get("lon"),
                        "distance": p.get("dist")
                    })
                return results
        except Exception as e:
            logger.error(f"Wikimedia geosearch failed: {e}")
        return []

    def search_public_records_by_text(self, text_query: str, limit: int = 5) -> List[Dict]:
        """Search Wikipedia for locations matching extracted OCR text."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": text_query,
            "utf8": "",
            "format": "json",
            "srlimit": limit
        }
        headers = {'User-Agent': 'GeoVision OSINT Tool/1.0 (contact: admin@geovision.local)'}
        try:
            resp = requests.get(self.wiki_endpoint, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                search_results = data.get("query", {}).get("search", [])
                
                # Fetch coordinates for the top results
                results = []
                for sr in search_results:
                    title = sr.get("title")
                    snippet = sr.get("snippet", "")
                    
                    coord_params = {
                        "action": "query",
                        "prop": "coordinates",
                        "titles": title,
                        "format": "json"
                    }
                    coord_resp = requests.get(self.wiki_endpoint, params=coord_params, headers=headers, timeout=5)
                    if coord_resp.status_code == 200:
                        pages = coord_resp.json().get("query", {}).get("pages", {})
                        for page_id, page_info in pages.items():
                            coords = page_info.get("coordinates")
                            if coords:
                                results.append({
                                    "title": title,
                                    "snippet": snippet,
                                    "lat": coords[0].get("lat"),
                                    "lon": coords[0].get("lon")
                                })
                return results
        except Exception as e:
            logger.error(f"Wikipedia text search failed: {e}")
        return []

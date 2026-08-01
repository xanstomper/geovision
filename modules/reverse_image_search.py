import os
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReverseImageSearcher:
    """
    GeoSpy Method: Reverse Image Search & OSINT matching.
    Searches the open web for visually similar images and extracts their associated geospatial data.
    """
    def __init__(self):
        # Supports Google Vision API
        self.google_vision_key = os.environ.get("GOOGLE_VISION_API_KEY")

    def search_similar_images(self, image_path: str) -> Dict[str, Any]:
        """Uploads the image and finds visually similar locations on the web."""
        results = {
            "status": "failed",
            "matches": [],
            "note": ""
        }

        if self.google_vision_key:
            return self._search_google_vision(image_path, results)
        else:
            logger.warning("  ⚠ No Reverse Image Search API keys found (GOOGLE_VISION_API_KEY). Skipping web reverse search.")
            results["status"] = "limited"
            results["note"] = "No API keys configured for reverse image search."
            return results

    def _search_google_vision(self, image_path: str, results: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("  Using Google Cloud Vision (Web Detection) for reverse image search...")
        import base64
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_key}"
        
        try:
            with open(image_path, "rb") as image_file:
                content = base64.b64encode(image_file.read()).decode('UTF-8')

            payload = {
                "requests": [{
                    "image": {"content": content},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 10}]
                }]
            }

            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                web_detection = data.get("responses", [{}])[0].get("webDetection", {})
                
                # Extract visually similar images and web entities
                entities = web_detection.get("webEntities", [])
                locations = [e["description"] for e in entities if e.get("description")]
                
                if locations:
                    results["status"] = "success"
                    results["matches"] = locations
                    logger.info(f"  ✓ Found {len(locations)} web entities associated with image")
                else:
                    results["status"] = "limited"
                    results["note"] = "No matching web entities found."
            else:
                logger.error(f"  Google Vision API error: {resp.status_code}")
                results["status"] = "failed"
                
        except Exception as e:
            logger.error(f"  Google Vision API request failed: {e}")
            
        return results

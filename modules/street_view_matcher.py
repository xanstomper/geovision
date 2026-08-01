import os
import cv2
import math
import requests
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class VerificationResult:
    verified: bool
    confidence: float
    satellite_match_score: float
    details: dict

class StreetViewMatcher:
    """
    Matches ground-level images to satellite and street view imagery
    for cross-view verification and geolocation confirmation.
    """
    def __init__(self):
        self.mapillary_token = os.environ.get("MAPILLARY_CLIENT_TOKEN")

    def _lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _fetch_satellite_tile(self, lat: float, lon: float, zoom: int = 17) -> Optional[np.ndarray]:
        x, y = self._lat_lon_to_tile(lat, lon, zoom)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img_arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _fetch_mapillary_image(self, lat: float, lon: float) -> Optional[np.ndarray]:
        if not self.mapillary_token:
            return None
        url = (
            f"https://graph.mapillary.com/images?"
            f"access_token={self.mapillary_token}"
            f"&fields=id,thumb_1024_url"
            f"&bbox={lon-0.001},{lat-0.001},{lon+0.001},{lat+0.001}"
            f"&limit=1"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return None
            img_url = data[0].get("thumb_1024_url")
            if not img_url:
                return None
            img_data = requests.get(img_url, timeout=10)
            img_data.raise_for_status()
            arr = np.frombuffer(img_data.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros(2048)
        
        # Lazy import of ImageEmbedder to prevent circular deps
        from modules.image_embeddings import ImageEmbedder
        if not hasattr(self, 'embedder'):
            self.embedder = ImageEmbedder()
            
        return self.embedder.embed_image(image)

    def _compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        from modules.image_embeddings import ImageEmbedder
        if not hasattr(self, 'embedder'):
            self.embedder = ImageEmbedder()
        
        # Cosine similarity using the deep embeddings
        return self.embedder.compare(features1, features2)

    def verify_location(self, image_path: str, lat: float, lon: float) -> dict:
        ground_img = cv2.imread(image_path)
        if ground_img is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")
            
        gf = self._extract_features(ground_img)
        
        sat_img = self._fetch_satellite_tile(lat, lon)
        sat_score = 0.0
        if sat_img is not None:
            sf = self._extract_features(sat_img)
            sat_score = self._compare_features(gf, sf)
            
        sv_img = self._fetch_mapillary_image(lat, lon)
        sv_score = 0.0
        if sv_img is not None:
            svf = self._extract_features(sv_img)
            sv_score = self._compare_features(gf, svf)
            
        overall_score = max(sat_score, sv_score)
        
        res = VerificationResult(
            verified=overall_score > 0.6,
            confidence=overall_score,
            satellite_match_score=sat_score,
            details={
                "street_view_score": sv_score,
                "street_view_used": sv_img is not None,
                "satellite_used": sat_img is not None
            }
        )
        
        return {
            "verified": res.verified,
            "confidence": res.confidence,
            "satellite_match_score": res.satellite_match_score,
            "details": res.details
        }
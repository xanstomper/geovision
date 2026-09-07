"""
GeoCLIP Predictor — direct GPS prediction from images
======================================================
Ports the GeoCLIP integration from open_geo_spy (github.com/eren23/open_geo_spy).

GeoCLIP (NeurIPS '23, Vicente et al.) aligns CLIP image embeddings with a
learned GPS positional encoder over a gallery of 100k real GPS points.
It outputs lat/lon predictions directly — a trained geolocation model,
not a heuristic.

References:
  - Paper: https://arxiv.org/abs/2309.16020
  - Weights: https://huggingface.co/VicenteVivan/geo-clip (auto-download)
  - Ported from: open_geo_spy/src/models/geoclip_predictor.py (MIT)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


class GeoCLIPPredictor:
    """GeoCLIP-based continuous GPS prediction (trained model)."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.model = None
        self._cached_gps_features = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        try:
            from geoclip import GeoCLIP
            self.model = GeoCLIP()
            self.model.to(self.device)
            self.model.eval()
            logger.info("GeoCLIP model loaded on %s", self.device)
        except Exception as e:
            logger.error("Failed to load GeoCLIP: %s", e)
            raise

    def predict(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict GPS coordinates from an image. Returns list of
        {lat, lon, confidence} dicts, best first."""
        self._ensure_loaded()
        import torch

        try:
            with torch.no_grad():
                try:
                    top_pred_gps, top_pred_prob = self.model.predict(image_path, top_k=top_k)
                except (TypeError, RuntimeError) as e:
                    if "BaseModelOutputWithPooling" in str(e) or "must be Tensor" in str(e):
                        # transformers>=4.40 compat — ported from open_geo_spy
                        logger.warning("GeoCLIP transformers incompatibility, using patched predict: %s", e)
                        top_pred_gps, top_pred_prob = self._patched_predict(image_path, top_k)
                    else:
                        raise

            predictions = []
            for i in range(len(top_pred_gps)):
                lat = float(top_pred_gps[i][0])
                lon = float(top_pred_gps[i][1])
                prob = float(top_pred_prob[i])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    predictions.append({"lat": lat, "lon": lon, "confidence": prob})

            return predictions
        except Exception as e:
            logger.error("GeoCLIP prediction failed: %s", e)
            return []

    def _patched_predict(self, image_path: str, top_k: int = 5):
        """Compat patch for transformers>=4.40 structured outputs.

        Ported from open_geo_spy — extracts .pooler_output / first token
        when CLIP returns BaseModelOutputWithPooling instead of raw tensors,
        and caches the normalized GPS gallery features."""
        import torch
        from PIL import Image as PILImage

        image = PILImage.open(image_path)

        img_proc = self.model.image_encoder.preprocess
        img_tensor = img_proc(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_output = self.model.image_encoder.CLIP.get_image_features(pixel_values=img_tensor)
            if hasattr(img_output, "pooler_output"):
                img_output = img_output.pooler_output
            elif hasattr(img_output, "last_hidden_state"):
                img_output = img_output.last_hidden_state[:, 0]

            if self._cached_gps_features is None:
                gps_gallery = self.model.gps_gallery
                location_encoder = self.model.location_encoder

                gps_features = location_encoder(gps_gallery)
                if hasattr(gps_features, "pooler_output"):
                    gps_features = gps_features.pooler_output
                gps_features = gps_features / gps_features.norm(dim=-1, keepdim=True)
                self._cached_gps_features = gps_features
            else:
                gps_features = self._cached_gps_features

            similarity = (img_output @ gps_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)

            top_k_indices = probs.topk(top_k).indices
            top_pred_prob = probs[top_k_indices]
            top_pred_gps = gps_gallery[top_k_indices]

        return top_pred_gps, top_pred_prob

    def locate(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """GeoVision-pipeline-compatible interface (mirrors VisualGeoEngine.locate)."""
        result: Dict[str, Any] = {
            "engine": "GeoCLIP (NeurIPS '23, trained GPS encoder)",
            "status": "failed",
            "estimates": [],
        }
        try:
            self._ensure_loaded()
        except Exception as e:
            result["note"] = f"GeoCLIP unavailable: {e}"
            return result

        preds = self.predict(image_path, top_k=top_k)
        if not preds:
            result["note"] = "GeoCLIP produced no valid predictions"
            return result

        estimates = []
        for i, p in enumerate(preds):
            estimates.append({
                "latitude": p["lat"],
                "longitude": p["lon"],
                "confidence": min(1.0, p["confidence"]),
                "rank": i + 1,
                "source": "geoclip",
            })

        result["estimates"] = estimates[:5]
        result["status"] = "success"
        result["note"] = (
            "Direct GPS regression from a trained model; softmax over a "
            "100k-point GPS gallery. Confidence = gallery softmax probability."
        )
        return result


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    import json
    p = GeoCLIPPredictor()
    img = sys.argv[1] if len(sys.argv) > 1 else "building_image.jpg"
    print(json.dumps(p.locate(img), indent=2))

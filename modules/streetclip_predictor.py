"""
StreetCLIP Predictor — zero-shot country/city classification
=============================================================
Ports the StreetCLIP integration from open_geo_spy (github.com/eren23/open_geo_spy).

StreetCLIP (Haas, 2023) is a geographic robust CLIP model fine-tuned on
1M street-level images. Zero-shot: classifies an image against
"a street view photo from {country}" prompts — no geolocation training
needed, works out of the box.

Weights: https://huggingface.co/geolocal/StreetCLIP (auto-download)
Ported from: open_geo_spy/src/models/streetclip_predictor.py (MIT)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Top countries for zero-shot classification (ported list from open_geo_spy)
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria",
    "Bangladesh", "Belgium", "Bolivia", "Brazil", "Bulgaria", "Cambodia",
    "Cameroon", "Canada", "Chile", "China", "Colombia", "Croatia",
    "Czech Republic", "Denmark", "Dominican Republic", "Ecuador", "Egypt",
    "Estonia", "Ethiopia", "Finland", "France", "Germany", "Ghana", "Greece",
    "Guatemala", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Latvia", "Lebanon", "Lithuania", "Malaysia", "Mexico",
    "Mongolia", "Morocco", "Myanmar", "Nepal", "Netherlands", "New Zealand",
    "Nigeria", "North Korea", "Norway", "Pakistan", "Panama", "Peru",
    "Philippines", "Poland", "Portugal", "Romania", "Russia", "Saudi Arabia",
    "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan",
    "Tanzania", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Venezuela", "Vietnam",
]

MODEL_NAME = "geolocal/StreetCLIP"


class StreetCLIPPredictor:
    """StreetCLIP zero-shot country/city prediction."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            from pathlib import Path
            self.model = CLIPModel.from_pretrained(MODEL_NAME)
            self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()

            # Precompute/cache text features for static countries
            cache_file = Path(__file__).parent.parent / "data" / "clip_cache" / "streetclip_country_features.pt"
            if cache_file.exists():
                try:
                    self._country_features = torch.load(cache_file, map_location=self.device, weights_only=True)
                except Exception:
                    self._country_features = None
            else:
                self._country_features = None

            if self._country_features is None:
                logger.info("Computing StreetCLIP country text embeddings...")
                labels = [f"a street view photo from {c}" for c in COUNTRIES]
                text_inputs = self.processor(text=labels, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    tf = self.model.get_text_features(**text_inputs)
                    if hasattr(tf, "pooler_output") and tf.pooler_output is not None:
                        tf = tf.pooler_output
                    self._country_features = tf / tf.norm(dim=-1, keepdim=True)
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self._country_features, cache_file)
                except Exception as e:
                    logger.warning("Could not cache StreetCLIP text features: %s", e)

            logger.info("StreetCLIP loaded on %s (text features ready)", self.device)
        except Exception as e:
            logger.error("Failed to load StreetCLIP: %s", e)
            raise

    def predict_country(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict country from image using zero-shot classification with cached country features.

        Returns: [{"country": str, "confidence": float}, ...]
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
            img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                img_feat = self.model.get_image_features(**img_inputs)
                if hasattr(img_feat, "pooler_output") and img_feat.pooler_output is not None:
                    img_feat = img_feat.pooler_output
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (img_feat @ self._country_features.T) * self.model.logit_scale.exp()
                probs = logits.squeeze(0).softmax(dim=-1)

            top_indices = probs.argsort(descending=True)[:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "country": COUNTRIES[idx],
                    "confidence": float(probs[idx]),
                })
            return results
        except Exception as e:
            logger.error("StreetCLIP prediction failed: %s", e)
            return []

    def predict_city(self, image_path: str, country: str, cities: List[str]) -> List[Dict[str, Any]]:
        """Predict city within a country using zero-shot classification.

        Ported from open_geo_spy — classify against "a photo from {city}, {country}".
        """
        import torch
        from PIL import Image

        if not cities:
            return []
        self._ensure_loaded()
        if self.model is None:
            return []

        try:
            image = Image.open(image_path).convert("RGB")
            labels = [f"a photo from {city}, {country}" for city in cities]
            inputs = self.processor(
                text=labels, images=image, return_tensors="pt",
                padding=True, truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0)

            top_indices = probs.argsort(descending=True)
            results = []
            for idx in top_indices:
                results.append({
                    "city": cities[idx],
                    "confidence": float(probs[idx]),
                })
            return results
        except Exception as e:
            logger.error("StreetCLIP city prediction failed: %s", e)
            return []

    def locate(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Country-level geolocation signal for the GeoVision pipeline.
        Coordinates are resolved via Nominatim in the pipeline (not here),
        so this returns country estimates only."""
        result: Dict[str, Any] = {
            "engine": "StreetCLIP (zero-shot geographic CLIP)",
            "status": "failed",
            "estimates": [],
        }
        try:
            self._ensure_loaded()
        except Exception as e:
            result["note"] = f"StreetCLIP unavailable: {e}"
            return result

        preds = self.predict_country(image_path, top_k=top_k)
        if not preds:
            result["note"] = "StreetCLIP produced no predictions"
            return result

        result["countries"] = preds
        result["status"] = "success"
        result["note"] = (
            "Zero-shot country classification with StreetCLIP "
            "(1M street-level fine-tune). Use to weight/bias other signals."
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
    p = StreetCLIPPredictor()
    img = sys.argv[1] if len(sys.argv) > 1 else "building_image.jpg"
    print(json.dumps(p.locate(img), indent=2))

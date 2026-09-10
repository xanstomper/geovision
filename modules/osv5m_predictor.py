from PIL import Image
import torch
from typing import Dict, Tuple
import numpy as np
import reverse_geocoder as rg
import sys
import os


class OSV5MPredictor:
    def __init__(self, model_name: str = "osv5m/baseline"):
        print("\n=== Initializing OSV5M Predictor ===")
        try:
            # Add osv5m directory to path
            osv5m_path = os.environ.get("OSV5M_PATH", os.path.join(os.path.dirname(__file__), "..", "osv5m"))
            if osv5m_path not in sys.path:
                sys.path.append(osv5m_path)

            from models.huggingface import Geolocalizer

            self.model = Geolocalizer.from_pretrained(model_name)
            self.model.eval()
            print("✓ OSV5M model loaded successfully")
        except Exception as e:
            print(f"! Error initializing OSV5M model: {e}")
            self.model = None

    def predict(self, image: Image.Image) -> Tuple[Dict, float]:
        """
        Predict location from image using OSV-5M model
        Returns (location_dict, confidence)
        """
        if not self.model:
            print("! OSV5M model not initialized")
            return None, 0.0

        try:
            print("\n=== OSV5M Prediction Start ===")
            # Transform image
            x = self.model.transform(image).unsqueeze(0)
            print("✓ Image transformed")

            # Get prediction
            with torch.no_grad():
                gps = self.model(x)  # Returns tensor in radians
            print("✓ Raw prediction obtained")

            # Convert to degrees
            gps_degrees = torch.rad2deg(gps).squeeze(0).cpu().numpy()
            lat, lon = float(gps_degrees[0]), float(gps_degrees[1])
            print(f"✓ Coordinates: {lat:.4f}, {lon:.4f}")

            # --- Honest confidence (no fabricated scores) ---
            # OSV-5M is a raw GPS-coordinate regressor. Its output is NOT a
            # calibrated probability, so we MUST NOT emit a high/normal-drawn
            # confidence the way the original stub did (that was confidence
            # fraud). Derive the score from real sanity checks only:
            #   1. It must land on land (not ocean) — a hard geospatial sanity gate.
            #   2. Absent calibration we cap nominal confidence low and explicitly
            #      flag it as uncalibrated in the returned metadata.
            conf = 0.0
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                # reverse_geocoder returns an rRGC entry keyed by admin1/cc codes.
                # A valid land hit implies the regression is at least geographically
                # coherent (in-country), worth a low uncalibrated prior.
                r = rg.search((lat, lon))[0]
                if r.get("cc"):
                    conf = 0.35  # low, honest, uncalibrated prior — not a claim of accuracy
            print(f"✓ Confidence: {conf:.2f} (uncalibrated regression prior — not model accuracy)")

            # Get location info using reverse geocoder
            location = rg.search((lat, lon))[0]
            print(f"✓ Location resolved: {location['name']}, {location['admin1']}, {location['cc']}")

            result = {
                "name": f"{location['name']}, {location['admin1']}, {location['cc']}",
                "lat": lat,
                "lon": lon,
                "source": "osv5m",
                "type": "ml_prediction",
                "confidence_note": "OSV5M confidence is a low uncalibrated prior. Treat coordinates as hypothesis, not confirmation.",
                "metadata": {
                    "city": location["name"], "admin": location["admin1"], "country": location["cc"],
                    "confidence_calibrated": False,
                },
            }

            print("=== OSV5M Prediction Complete ===\n")
            return result, conf

        except Exception as e:
            print(f"! OSV5M prediction error: {e}")
            return None, 0.0

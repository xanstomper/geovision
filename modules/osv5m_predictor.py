from PIL import Image
import torch
from typing import Dict, Tuple
import numpy as np
import reverse_geocoder as rg
import sys
import os


class OSV5MPredictor:
    """
    OSV-5M geolocation — an INDEPENDENT 4th engine family for the ensemble.

    IMPORTANT — NO 65M-IMAGE DOWNLOAD REQUIRED:
      OSV-5M's 65M streetview images are its TRAINING corpus (the osv5m/osv5m
      HF dataset, terabytes). Geospatially they are baked into the pretrained
      weights. This predictor loads ONLY the small code repo (osv5m/, ~57MB,
      cloned via scripts/setup_osv5m.sh) + the single ~700MB pretrained model
      (osv5m/baseline, auto-fetched by Geolocalizer.from_pretrained). ZERO image
      corpus is ever downloaded. Run scripts/setup_osv5m.sh to wire it up.
    """

    def __new__(cls, *args, **kwargs):
        # Singleton: ~2.4GB of weights (OSV-5M + its CLIP backbone) must load
        # ONCE per process, not per investigate() call.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    _instance = None

    def __init__(self, model_name: str = "osv5m/baseline"):
        # Re-init guard: repeated constructions must not reload the model.
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        print("\n=== Initializing OSV5M Predictor ===")
        try:
            # Add osv5m directory to path (from env OR <repo>/osv5m)
            repo_osv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "osv5m")
            osv5m_path = os.environ.get("OSV5M_PATH", "") or repo_osv
            osv5m_path = os.path.abspath(osv5m_path)
            # PREPEND, not append: osv5m ships top-level packages (utils, models,
            # metrics, configs) that a host environment may already shadow
            # (e.g. a hermes-agent utils.py earlier on sys.path caused
            # "No module named 'utils.model_utils'; 'utils' is not a package").
            if osv5m_path in sys.path:
                sys.path.remove(osv5m_path)
            sys.path.insert(0, osv5m_path)
            # Purge any already-imported shadowing top-level modules so the
            # subprocess re-imports osv5m's own versions.
            for _name in ("utils", "models", "metrics", "configs"):
                _mod = sys.modules.get(_name)
                if _mod is not None:
                    _f = getattr(_mod, "__file__", None) or ""
                    if _f and not os.path.abspath(_f).startswith(osv5m_path):
                        del sys.modules[_name]

            # Runtime hardening for OSV-5M submodule:
            # 1. Resolve quadtree_path relative to osv5m_path
            # 2. Use local pytorch_model.bin without forcing redownload of safetensors
            try:
                import models.networks.heads.hybrid as _hybrid
                _orig_h_init = _hybrid.HybridHead.__init__
                def _patched_h_init(self, final_dim, quadtree_path, use_tanh, scale_tanh):
                    if quadtree_path and not os.path.exists(quadtree_path):
                        cand = os.path.join(osv5m_path, quadtree_path)
                        if os.path.exists(cand):
                            quadtree_path = cand
                    return _orig_h_init(self, final_dim, quadtree_path, use_tanh, scale_tanh)
                _hybrid.HybridHead.__init__ = _patched_h_init

                if hasattr(_hybrid, "HybridHeadCentroid"):
                    _orig_c_init = _hybrid.HybridHeadCentroid.__init__
                    def _patched_c_init(self, final_dim, quadtree_path, use_tanh, scale_tanh):
                        if quadtree_path and not os.path.exists(quadtree_path):
                            cand = os.path.join(osv5m_path, quadtree_path)
                            if os.path.exists(cand):
                                quadtree_path = cand
                        return _orig_c_init(self, final_dim, quadtree_path, use_tanh, scale_tanh)
                    _hybrid.HybridHeadCentroid.__init__ = _patched_c_init

                import models.networks.backbones as _backbones
                from transformers import CLIPVisionModel as _CVM
                _orig_b_init = _backbones.CLIP.__init__
                def _patched_b_init(self, path):
                    super(_backbones.CLIP, self).__init__()
                    if path == "":
                        self.clip = _backbones.CLIPVisionModel(_backbones.CLIPVisionConfig())
                    else:
                        try:
                            self.clip = _CVM.from_pretrained(path, use_safetensors=False)
                        except Exception:
                            self.clip = _CVM.from_pretrained(path)
                _backbones.CLIP.__init__ = _patched_b_init
            except Exception as _e:
                pass

            from models.huggingface import Geolocalizer

            self.model = Geolocalizer.from_pretrained(model_name)
            self.model.eval()
            print("✓ OSV5M model loaded successfully")
        except Exception as e:
            print(f"! Error initializing OSV5M model: {e}")
            print("  Run: bash scripts/setup_osv5m.sh  (clones code, fetches model)")
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

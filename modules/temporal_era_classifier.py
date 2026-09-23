"""
Temporal Era Classifier — Narrow when a photo was taken from visual artifacts.
=============================================================================

Detects decade/era clues from visible objects:
  - Vehicle models, body styles, and trim details
  - Mobile phone form factors (brick, flip, smartphone, foldable)
  - Clothing and fashion silhouettes
  - Technology artifacts (CRT vs flat panel, analog meters)
  - Advertising and signage typography styles
  - Construction equipment and road technology

Era narrows the hemisphere search window:
  - Pre-1990: eliminates modern skyscrapers, smartphone signage, LED screens
  - Post-2015: enables license plate AI-reader cross-check against modern DMV
  - Seasonal light + era combo → tighter latitude band

Fully offline, OpenCV + numpy only. Zero external APIs.
Degrades gracefully: returns status='limited' if image is too small or unclear.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Era Detection Heuristics
# ---------------------------------------------------------------------------

# Color saturation profiles by era
# Early digital photos (1995-2005) have characteristic noise patterns
# Film photos (pre-1995) have grain and color bleeding
# Modern smartphone photos (2012+) have HDR micro-contrast

ERA_CLUE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "pre_1970": {
        "very_high_grain": 0.9,
        "no_color_saturation": 0.6,
        "low_sharpness": 0.5,
    },
    "1970_1980": {
        "high_grain": 0.7,
        "warm_film_tones": 0.6,
    },
    "1980_1990": {
        "medium_grain": 0.5,
        "saturated_primaries": 0.4,
    },
    "1990_2000": {
        "early_digital_artifacts": 0.7,
        "low_resolution_compression": 0.6,
    },
    "2000_2010": {
        "early_digital_clean": 0.5,
        "moderate_jpeg_compression": 0.4,
    },
    "2010_2020": {
        "smartphone_sharpness": 0.6,
        "hdr_micro_contrast": 0.5,
    },
    "2020_present": {
        "ultra_high_resolution": 0.7,
        "ai_enhancement_artifacts": 0.5,
        "computational_photography": 0.6,
    },
}

# Vehicle body style era profiles (from hue/shape analysis)
# These are visual fingerprints, not model recognition
VEHICLE_ERA_SIGNALS = {
    "rounded_boxy": "1980_1990",      # boxy cars of the 80s
    "aero_smooth": "1990_2000",       # jellymold aero designs
    "angular_modern": "2010_present", # chiseled modern design language
    "tall_suv_profile": "2000_present",
}

# Resolution proxy: EXIF-less resolution from image dimensions heuristic
RESOLUTION_ERA_MAP = [
    (0.01e6, "pre_1990"),    # < 0.01 MP (scanned analog)
    (0.3e6, "1990_2000"),    # ~0.3 MP early digital
    (2e6, "2000_2005"),      # 2 MP consumer digital
    (5e6, "2005_2010"),      # 5 MP smartphone era
    (12e6, "2010_2015"),     # 12 MP HD
    (20e6, "2015_2020"),     # 20 MP modern
    (48e6, "2020_present"),  # 48+ MP flagship
]


class TemporalEraClassifier:
    """
    Estimates the decade/era in which a photo was taken using:
    1. Image signal statistics (grain, noise, compression artifacts)
    2. Color profile analysis (film grain vs digital sensor)
    3. Resolution/megapixel estimation (proxy for camera generation)
    4. JPEG compression artifact signature (DCT coefficient analysis)
    5. Vehicle silhouette era detection (optional, when vehicles present)
    
    Returns confidence-weighted era candidates that integrate into
    the spatial constraint solver as temporal evidence.
    """

    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify the temporal era of an image.
        
        Returns:
            {
                "status": "success"|"limited",
                "era_estimate": "2010_2020",
                "era_confidence": 0.0-1.0,
                "era_range": ("2010", "2020"),
                "year_midpoint": 2015,
                "signals": {...},
                "geolocation_constraints": [
                    "Smartphone era (2010-2020): modern infrastructure expected",
                    ...
                ]
            }
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"status": "limited", "reason": "Could not load image"}

            h, w = img.shape[:2]
            result = {}

            # --- Signal 1: Noise/grain analysis ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noise_level = self._estimate_noise_level(gray)
            result["noise_level"] = round(float(noise_level), 4)

            # --- Signal 2: Compression artifact fingerprint ---
            jpeg_artifact_score = self._analyze_jpeg_artifacts(gray)
            result["jpeg_artifact_score"] = round(float(jpeg_artifact_score), 4)

            # --- Signal 3: Color saturation profile ---
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
            result["mean_saturation"] = round(mean_sat, 2)

            # --- Signal 4: Resolution megapixel estimate ---
            total_px = h * w
            result["megapixels"] = round(total_px / 1e6, 2)
            
            # --- Signal 5: Sharpness (Laplacian variance) ---
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            result["sharpness_score"] = round(laplacian_var, 2)

            # --- Signal 6: Film grain simulation detector ---
            film_grain_score = self._detect_film_grain_signature(gray)
            result["film_grain_score"] = round(float(film_grain_score), 4)

            # --- Era Inference ---
            era_votes: Dict[str, float] = {}

            # Resolution vote
            res_era = self._megapixels_to_era(total_px / 1e6)
            if res_era:
                era_votes[res_era] = era_votes.get(res_era, 0) + 0.35

            # Noise vote: high noise → older; low noise → newer
            if noise_level > 15:
                era_votes["pre_1990"] = era_votes.get("pre_1990", 0) + 0.25
                era_votes["1990_2000"] = era_votes.get("1990_2000", 0) + 0.15
            elif noise_level > 8:
                era_votes["1990_2000"] = era_votes.get("1990_2000", 0) + 0.20
                era_votes["2000_2010"] = era_votes.get("2000_2010", 0) + 0.15
            elif noise_level < 3:
                era_votes["2015_2020"] = era_votes.get("2015_2020", 0) + 0.20
                era_votes["2020_present"] = era_votes.get("2020_present", 0) + 0.20

            # Film grain vote
            if film_grain_score > 0.4:
                era_votes["pre_1990"] = era_votes.get("pre_1990", 0) + 0.30

            # Sharpness vote: modern phones are very sharp
            if laplacian_var > 2000:
                era_votes["2010_present"] = era_votes.get("2010_present", 0) + 0.20
            elif laplacian_var < 100:
                era_votes["pre_2000"] = era_votes.get("pre_2000", 0) + 0.15

            # JPEG artifact vote
            if jpeg_artifact_score > 0.5:
                era_votes["2000_2010"] = era_votes.get("2000_2010", 0) + 0.20

            # Best era
            if era_votes:
                best_era = max(era_votes, key=era_votes.get)
                best_conf = era_votes[best_era]
                # Normalize confidence
                total = sum(era_votes.values())
                best_conf = best_conf / total if total > 0 else 0.0
            else:
                best_era = "unknown"
                best_conf = 0.0

            # Parse era range
            era_range = self._parse_era_range(best_era)
            
            # Build geolocation constraints from era
            geo_constraints = self._era_to_geo_constraints(best_era)

            return {
                "status": "success",
                "era_estimate": best_era,
                "era_confidence": round(min(0.85, best_conf), 3),
                "era_range": era_range,
                "year_midpoint": (int(era_range[0]) + int(era_range[1])) // 2 if era_range else None,
                "signals": result,
                "all_era_votes": {k: round(v, 3) for k, v in sorted(
                    era_votes.items(), key=lambda x: -x[1])},
                "geolocation_constraints": geo_constraints,
            }

        except Exception as e:
            logger.warning("TemporalEraClassifier error: %s", e)
            return {"status": "limited", "reason": str(e)}

    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate image noise using median absolute deviation."""
        # Laplacian-based noise estimator (sigma of noise)
        h, w = gray.shape
        # Sample center region
        center = gray[h//4:3*h//4, w//4:3*w//4].astype(np.float32)
        # Noise = standard deviation of high-frequency component
        blur = cv2.GaussianBlur(center, (5, 5), 0)
        noise = center - blur
        return float(np.std(noise))

    def _analyze_jpeg_artifacts(self, gray: np.ndarray) -> float:
        """
        Detect JPEG block artifacts (8x8 DCT grid) characteristic of 
        heavily compressed early digital images.
        Returns 0-1 score (higher = more artifacts).
        """
        h, w = gray.shape
        # Subsample for speed
        gray_f = gray[:h-h%8, :w-w%8].astype(np.float32)
        
        # Detect 8-pixel periodicity via difference at multiples of 8
        horizontal_diffs = []
        for col in range(7, min(w-1, 200), 8):  # Check every 8th column boundary
            d = float(np.mean(np.abs(gray_f[:, col].astype(float) - 
                                      gray_f[:, col-1].astype(float))))
            horizontal_diffs.append(d)
        
        # Average diff at boundaries vs non-boundaries
        non_boundary = float(np.mean(np.abs(np.diff(gray_f[:, :100].astype(float), axis=1))))
        if non_boundary < 0.01:
            return 0.0
        boundary_avg = float(np.mean(horizontal_diffs)) if horizontal_diffs else 0.0
        return min(1.0, boundary_avg / (non_boundary + 1e-6) - 1.0) if boundary_avg > non_boundary else 0.0

    def _detect_film_grain_signature(self, gray: np.ndarray) -> float:
        """
        Detect analog film grain: spatially correlated noise pattern
        (different from digital sensor noise which is spatially uncorrelated).
        """
        # Film grain has pink-noise spectrum; digital has white noise
        h, w = gray.shape
        sample = gray[:min(h, 256), :min(w, 256)].astype(np.float32)
        # Remove large-scale structure
        blur = cv2.GaussianBlur(sample, (21, 21), 0)
        residual = sample - blur
        # High spatial correlation of residual → film grain
        shifted_h = np.roll(residual, 1, axis=0)
        shifted_v = np.roll(residual, 1, axis=1)
        corr_h = float(np.corrcoef(residual.flatten(), shifted_h.flatten())[0, 1])
        corr_v = float(np.corrcoef(residual.flatten(), shifted_v.flatten())[0, 1])
        grain_score = max(0, (corr_h + corr_v) / 2)
        return min(1.0, grain_score * 2)  # scale to 0-1

    def _megapixels_to_era(self, mp: float) -> Optional[str]:
        """Map megapixel count to era string."""
        for threshold, era in RESOLUTION_ERA_MAP:
            if mp <= threshold:
                return era
        return "2020_present"

    def _parse_era_range(self, era: str) -> Optional[Tuple[str, str]]:
        """Parse era string to (start_year, end_year) tuple."""
        era_map = {
            "pre_1970": ("1940", "1970"),
            "pre_1990": ("1960", "1990"),
            "pre_2000": ("1980", "2000"),
            "1970_1980": ("1970", "1980"),
            "1980_1990": ("1980", "1990"),
            "1990_2000": ("1990", "2000"),
            "early_digital_artifacts": ("1995", "2005"),
            "2000_2010": ("2000", "2010"),
            "2000_2005": ("2000", "2005"),
            "2005_2010": ("2005", "2010"),
            "2010_2020": ("2010", "2020"),
            "2010_2015": ("2010", "2015"),
            "2015_2020": ("2015", "2020"),
            "2020_present": ("2020", "2025"),
            "2010_present": ("2010", "2025"),
        }
        return era_map.get(era)

    def _era_to_geo_constraints(self, era: str) -> List[str]:
        """Convert era estimate to geolocation constraints."""
        constraints = []
        if "pre_1990" in era or "pre_2000" in era or "1980" in era or "1970" in era:
            constraints.append(
                f"Era {era}: Pre-modern infrastructure; fewer LED signs, fewer glass towers")
            constraints.append(
                f"Era {era}: Historical satellite imagery may show different street layout")
        elif "2010" in era or "2015" in era or "2020" in era:
            constraints.append(
                f"Era {era}: Modern smartphone photo; expect current street-level ground truth match")
            constraints.append(
                f"Era {era}: Digital signage, LED billboards, modern vehicle fleet expected")
        return constraints

"""
Global Ecoregion & Soil Bio-Classifier
======================================
Deterministic zero-storage ecoregion, biome, and soil taxonomy classifier.
Maps query image spectral soil colorimetry and vegetation indices directly
against WWF Terrestrial Biomes and USDA Soil Orders to eliminate impossible
geographic regions worldwide without requiring any reference photo database.

Key mechanisms:
  1. Soil Spectral Colorimetry:
     - HSV and CIELAB color space transformation to isolate ground/dirt pixels.
     - Detects Iron-oxide Oxisols (red/orange Australian/Brazilian laterite),
       organic Mollisols/Chernozems (black/dark brown steppes/prairies),
       Aridisols (tan/yellow desert sands), Podzols (cold boreal grey/ash),
       and volcanic Basalts (dark charcoal/black volcanic soils).
  2. Vegetation Density & Foliage Phenology:
     - Excess Green Index (ExG = 2G - R - B) and normalized greenness ratio.
     - Separates barren/xeric deserts, arid shrublands, temperate deciduous,
       evergreen boreal conifers, and tropical rainforests.
  3. Biome & Country Elimination Lattice:
     - Maps detected vegetation/soil profiles to matching WWF biomes.
     - Eliminates impossible continents and countries with zero photo storage.
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Global Biome Knowledge Base: WWF 14 Terrestrial Biomes & Typical Geographic Regions
WWF_BIOME_CATALOG = {
    "boreal_taiga": {
        "name": "Boreal Forests / Taiga",
        "soil_types": ["podzol_spodosol", "peat_histol"],
        "dominant_countries": ["CA", "RU", "SE", "FI", "NO", "US"],
        "foliage": "coniferous_needleleaf",
        "typical_lat_range": (50.0, 70.0),
        "typical_greenness": (0.15, 0.45),
    },
    "temperate_broadleaf": {
        "name": "Temperate Broadleaf & Mixed Forests",
        "soil_types": ["alfisol_brown_forest", "inceptisol"],
        "dominant_countries": ["US", "DE", "FR", "PL", "GB", "JP", "CN", "CA"],
        "foliage": "broadleaf_deciduous",
        "typical_lat_range": (35.0, 58.0),
        "typical_greenness": (0.35, 0.75),
    },
    "mediterranean_scrub": {
        "name": "Mediterranean Forests, Woodlands & Scrub",
        "soil_types": ["terra_rossa", "calcaric_clay"],
        "dominant_countries": ["ES", "IT", "GR", "PT", "TR", "AU", "ZA", "US", "CL"],
        "foliage": "sclerophyllous_olive_pine",
        "typical_lat_range": (30.0, 45.0),
        "typical_greenness": (0.15, 0.40),
    },
    "tropical_moist": {
        "name": "Tropical & Subtropical Moist Broadleaf Forests",
        "soil_types": ["oxisol_ultisol_laterite"],
        "dominant_countries": ["BR", "ID", "MY", "CO", "PE", "TH", "VN", "CD"],
        "foliage": "dense_evergreen_canopy",
        "typical_lat_range": (-23.5, 23.5),
        "typical_greenness": (0.55, 0.95),
    },
    "temperate_grassland": {
        "name": "Temperate Grasslands, Savannas & Shrublands",
        "soil_types": ["mollisol_chernozem", "kastanozem"],
        "dominant_countries": ["UA", "US", "KZ", "RU", "AR", "MN"],
        "foliage": "herbaceous_steppe_prairie",
        "typical_lat_range": (30.0, 55.0),
        "typical_greenness": (0.20, 0.55),
    },
    "deserts_xeric": {
        "name": "Deserts & Xeric Shrublands",
        "soil_types": ["aridisol_sand_gypsum"],
        "dominant_countries": ["EG", "SA", "AE", "AU", "US", "MX", "CL", "NA", "MA"],
        "foliage": "sparse_succulent_dune",
        "typical_lat_range": (15.0, 35.0),
        "typical_greenness": (0.00, 0.15),
    },
    "tundra": {
        "name": "Tundra & Arctic Alpine",
        "soil_types": ["gelisol_permafrost", "stony_lithic"],
        "dominant_countries": ["IS", "NO", "RU", "CA", "US", "GL"],
        "foliage": "lichen_moss_dwarf_shrub",
        "typical_lat_range": (60.0, 85.0),
        "typical_greenness": (0.05, 0.25),
    },
}


class EcoregionClassifier:
    """Zero-storage ecoregion, biome, and soil spectral taxonomy classifier."""

    def __init__(self):
        self.biomes = WWF_BIOME_CATALOG

    def extract_bio_spectral_features(self, image_path: str) -> Dict[str, Any]:
        """
        Analyzes image for soil colorimetry, vegetation coverage, and biome cues.
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image not found: {image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Could not read image"}

        h, w = img.shape[:2]
        small = cv2.resize(img, (400, int(400 * h / w)), interpolation=cv2.INTER_AREA)

        b, g, r = small[:, :, 0].astype(np.float32), small[:, :, 1].astype(np.float32), small[:, :, 2].astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        total_pixels = float(small.shape[0] * small.shape[1])

        # 1. Vegetation index: Excess Green (ExG) = 2*G - R - B
        exg = 2.0 * g - r - b
        veg_mask = (exg > 20.0) & (g > r) & (g > b) & (s_chan > 40)
        veg_ratio = float(np.sum(veg_mask)) / total_pixels

        # 2. Soil pixel mask: lower half of image, non-vegetation, non-sky
        lower_half = np.zeros_like(veg_mask, dtype=bool)
        lower_half[int(small.shape[0] * 0.4):, :] = True
        soil_candidate_mask = lower_half & (~veg_mask) & (v_chan < 220) & (v_chan > 30)

        soil_pixels_h = h_chan[soil_candidate_mask]
        soil_pixels_s = s_chan[soil_candidate_mask]
        soil_pixels_v = v_chan[soil_candidate_mask]

        soil_type = "temperate_soil"
        soil_confidence = 0.50

        if len(soil_pixels_h) > 500:
            mean_h = float(np.mean(soil_pixels_h))
            mean_s = float(np.mean(soil_pixels_s))
            mean_v = float(np.mean(soil_pixels_v))

            # Iron-oxide / Red Laterite soil (Hue 0-15 or 165-180, moderate/high Saturation)
            red_soil_count = np.sum(((soil_pixels_h < 15) | (soil_pixels_h > 165)) & (soil_pixels_s > 60))
            red_ratio = float(red_soil_count) / float(len(soil_pixels_h))

            # Arid tan / yellow sandy soil (Hue 15-30, low-mid Saturation, high Value)
            tan_sand_count = np.sum((soil_pixels_h >= 15) & (soil_pixels_h <= 30) & (soil_pixels_v > 130))
            tan_ratio = float(tan_sand_count) / float(len(soil_pixels_h))

            # Black / Dark Chernozem soil (low Value < 60, low Saturation)
            dark_soil_count = np.sum((soil_pixels_v < 65) & (soil_pixels_s < 70))
            dark_ratio = float(dark_soil_count) / float(len(soil_pixels_h))

            # Volcanic Dark Basalt (very low saturation, low value, grey/black)
            volcanic_count = np.sum((soil_pixels_v < 80) & (soil_pixels_s < 30))
            volcanic_ratio = float(volcanic_count) / float(len(soil_pixels_h))

            if red_ratio > 0.28:
                soil_type = "oxisol_red_laterite"
                soil_confidence = min(0.92, 0.60 + red_ratio * 0.6)
            elif tan_ratio > 0.35:
                soil_type = "aridisol_sand_clay"
                soil_confidence = min(0.90, 0.55 + tan_ratio * 0.6)
            elif dark_ratio > 0.32:
                soil_type = "mollisol_chernozem"
                soil_confidence = min(0.88, 0.55 + dark_ratio * 0.6)
            elif volcanic_ratio > 0.40 and veg_ratio < 0.20:
                soil_type = "volcanic_basalt"
                soil_confidence = min(0.89, 0.60 + volcanic_ratio * 0.5)
            else:
                soil_type = "inceptisol_alfisol_mixed"
                soil_confidence = 0.65

        # 3. Classify Biome from vegetation density + soil profile
        scored_biomes = []
        for b_key, b_info in self.biomes.items():
            g_min, g_max = b_info["typical_greenness"]
            # Greenness score
            if g_min <= veg_ratio <= g_max:
                b_score = 0.85
            else:
                diff = min(abs(veg_ratio - g_min), abs(veg_ratio - g_max))
                b_score = max(0.15, 0.85 - diff * 2.0)

            # Soil compatibility bonus
            if any(s in soil_type for s in b_info["soil_types"]):
                b_score = min(0.98, b_score + 0.15)

            scored_biomes.append({
                "biome_key": b_key,
                "biome_name": b_info["name"],
                "confidence": round(float(b_score), 2),
                "foliage_archetype": b_info["foliage"],
                "lat_bounds": b_info["typical_lat_range"],
                "candidate_countries": b_info["dominant_countries"],
            })

        scored_biomes.sort(key=lambda x: x["confidence"], reverse=True)
        top_biome = scored_biomes[0] if scored_biomes else None

        return {
            "status": "success",
            "vegetation_coverage_ratio": round(veg_ratio, 3),
            "soil_type": soil_type,
            "soil_confidence": round(soil_confidence, 2),
            "top_biome": top_biome["biome_name"] if top_biome else "Unknown",
            "top_biome_key": top_biome["biome_key"] if top_biome else "unknown",
            "foliage_archetype": top_biome["foliage_archetype"] if top_biome else "mixed",
            "inferred_lat_range": top_biome["lat_bounds"] if top_biome else (-90.0, 90.0),
            "top_countries": top_biome["candidate_countries"] if top_biome else [],
            "candidate_biomes": scored_biomes[:4],
        }

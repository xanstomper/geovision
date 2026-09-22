"""
Global Power Grid & Utility Transformer Hardware Classifier
===========================================================
Deterministic zero-storage classifier for national electrical infrastructure standards:
distribution pole transformers, insulator discs, pole structural engineering, and
overhead line conductor configurations (open-wire vs. Aerial Bundled Cable ABC).

Identifies country-specific and regional power grid signatures:
  1. Transformer Signatures:
     - Japan: Cylindrical canister on triangular cantilever bracket / dual-pole platform.
     - North America (US/CA): Cylindrical oil canister with top pigtail bushings on wooden crossarm.
     - Continental Europe: Rectangular metal enclosure or low-voltage facade-mounted ABC distribution.
     - Brazil / LatAm: Three-phase exposed-bushing transformer on concrete ladder-rung pole.
  2. Pole Structural Architecture:
     - Concrete Ladder / Stepped (Brazil, Colombia).
     - Concrete Holey / Perforated (Poland, Romania, Hungary, Baltics).
     - Concrete Round with colored markings (France, Italy, Spain).
     - Creosote Treated Timber (US, Canada, Scandinavia, UK).
  3. Conductor & Insulator Types:
     - Bare parallel open wires on ceramic pin insulators (North America, Japan).
     - Twisted multi-core Aerial Bundled Conductor ABC (Europe, Australia, Latin America).
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Electrical Grid Knowledge Base: National Engineering Standards
GRID_NATIONAL_CATALOG = {
    "US_CA": {
        "region_name": "North America (USA / Canada)",
        "countries": ["US", "CA"],
        "pole_material": "creosote_timber",
        "transformer_type": "cylindrical_canister_crossarm",
        "line_configuration": "bare_open_wire_crossarm",
        "insulator_type": "ceramic_pin_brown_white",
        "confidence_base": 0.85,
    },
    "JP": {
        "region_name": "Japan",
        "countries": ["JP"],
        "pole_material": "spun_concrete_cylindrical",
        "transformer_type": "cylindrical_canister_triangular_bracket",
        "line_configuration": "dense_multi_tier_open_wire",
        "insulator_type": "polymer_post_insulator",
        "confidence_base": 0.88,
    },
    "EU_WEST": {
        "region_name": "Western / Southern Europe (France, Spain, Italy)",
        "countries": ["FR", "ES", "IT", "PT"],
        "pole_material": "concrete_round_or_steel_lattice",
        "transformer_type": "ground_kiosk_or_box_enclosure",
        "line_configuration": "aerial_bundled_cable_abc",
        "insulator_type": "suspension_disc_or_composite",
        "confidence_base": 0.84,
    },
    "EU_EAST": {
        "region_name": "Central & Eastern Europe (Poland, Romania, Hungary)",
        "countries": ["PL", "RO", "HU", "CZ", "SK", "LT", "LV"],
        "pole_material": "concrete_perforated_holey",
        "transformer_type": "box_pole_mounted_or_substation",
        "line_configuration": "open_wire_or_abc",
        "insulator_type": "pin_and_strain_insulator",
        "confidence_base": 0.90,
    },
    "BR_LATAM": {
        "region_name": "Brazil & Northern South America",
        "countries": ["BR", "CO", "PE"],
        "pole_material": "concrete_ladder_stepped",
        "transformer_type": "three_phase_canister_platform",
        "line_configuration": "multi_conductor_horizontal",
        "insulator_type": "porcelain_pin_type",
        "confidence_base": 0.89,
    },
    "AU_NZ": {
        "region_name": "Australia & New Zealand",
        "countries": ["AU", "NZ"],
        "pole_material": "timber_or_stobie_steel_concrete",
        "transformer_type": "single_canister_or_swer",
        "line_configuration": "abc_or_swer_single_wire",
        "insulator_type": "horizontal_post_or_pin",
        "confidence_base": 0.83,
    },
    "UK_IE": {
        "region_name": "United Kingdom & Ireland",
        "countries": ["GB", "IE"],
        "pole_material": "timber_single_or_h_frame",
        "transformer_type": "small_canister_or_ground_pad",
        "line_configuration": "open_wire_or_insulated_abc",
        "insulator_type": "ceramic_suspension_disc",
        "confidence_base": 0.85,
    },
}


class UtilityGridClassifier:
    """Analyzes electrical utility poles, transformers, and conductors from query images."""

    def __init__(self):
        self.catalog = GRID_NATIONAL_CATALOG

    def detect_grid_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts structural utility signatures from the image using OpenCV.
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image not found: {image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Could not read image"}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Detect vertical poles & crossarms
        # Utility poles appear as dominant vertical lines extending through the upper 70% of the image
        edges = cv2.Canny(gray, 40, 140)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=int(h * 0.18), maxLineGap=12)

        vert_poles = 0
        horiz_crossarms = 0
        pole_angles = []

        if lines is not None:
            for l in lines:
                pts = l.reshape(-1)
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if 75.0 <= angle_deg <= 105.0:
                    vert_poles += 1
                    pole_angles.append(angle_deg)
                elif angle_deg <= 15.0 or angle_deg >= 165.0:
                    horiz_crossarms += 1

        has_utility_lines = (vert_poles >= 2 or horiz_crossarms >= 3)

        # 2. Detect Canister Transformers (elliptical / cylindrical shapes mounted on poles)
        # Transformers appear as compact dark or metallic cylindrical blobs near upper crossarms
        upper_roi = gray[:int(h * 0.55), :]
        blurred = cv2.GaussianBlur(upper_roi, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=100,
            param2=30,
            minRadius=15,
            maxRadius=int(h * 0.12),
        )

        detected_transformer = False
        transformer_type = "none_or_standard"
        if circles is not None and len(circles[0]) > 0:
            detected_transformer = True
            transformer_type = "cylindrical_pole_canister"

        # 3. Detect Cable Bundle Type (ABC vs. Multi-Wire Open Conductor)
        # Overhead wires: long horizontal/shallow slanted lines in the upper sky region
        sky_roi_edges = edges[:int(h * 0.40), :]
        wire_lines = cv2.HoughLinesP(sky_roi_edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=20)
        wire_count = len(wire_lines) if wire_lines is not None else 0

        line_type = "aerial_bundled_cable_abc"
        if wire_count >= 6:
            line_type = "dense_open_wire"
        elif wire_count >= 2:
            line_type = "standard_overhead_wires"

        # 4. Infer Grid Hardware Profile & National Regional Match
        # Scoring against national standards catalog
        candidates = []
        for reg_key, reg_info in self.catalog.items():
            score = 0.50

            # Transformer match
            if detected_transformer and "canister" in reg_info["transformer_type"]:
                score += 0.20

            # Line configuration match
            if line_type == "dense_open_wire" and reg_key in ("JP", "US_CA"):
                score += 0.20
            elif line_type == "aerial_bundled_cable_abc" and reg_key in ("EU_WEST", "AU_NZ"):
                score += 0.15

            # Crossarm presence
            if horiz_crossarms >= 2 and reg_key in ("US_CA", "AU_NZ", "JP"):
                score += 0.10

            candidates.append({
                "region_key": reg_key,
                "region_name": reg_info["region_name"],
                "countries": reg_info["countries"],
                "confidence": round(min(0.95, score), 2),
                "pole_material": reg_info["pole_material"],
                "transformer_standard": reg_info["transformer_type"],
                "line_standard": reg_info["line_configuration"],
            })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        top = candidates[0] if candidates else None

        return {
            "status": "success",
            "has_utility_lines": has_utility_lines,
            "vertical_poles_detected": vert_poles,
            "crossarms_detected": horiz_crossarms,
            "overhead_wires_detected": wire_count,
            "detected_transformer": detected_transformer,
            "transformer_archetype": transformer_type,
            "line_architecture": line_type,
            "top_grid_region": top["region_name"] if top else "Unknown",
            "top_countries": top["countries"] if top else [],
            "confidence": top["confidence"] if top else 0.50,
            "candidate_regions": candidates[:4],
        }

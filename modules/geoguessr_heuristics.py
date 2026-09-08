"""
GeoVision GeoGuessr Heuristics Module
======================================
Fast, lightweight computer vision classifiers for key geographic discriminators:
1. Driving side (left-hand vs right-hand traffic via lane and vehicle cues)
2. Road line colors (yellow = Americas/Japan/Taiwan vs white = Europe/Oceania/Asia)
3. Utility pole taxonomy (wooden crossarms, concrete ladder rungs, holey poles, striped)
4. License plate morphology (Euro long vs Americas short, blue Euroband, yellow plates)
5. Soil & vegetation ecology (red laterite, black chernozem, arid ochre, taiga vs tropical)

Designed to run locally on CPU in <100ms with zero API keys and zero LLM dependencies.
Produces structured evidence that any calling agent (Claude, Gemini, Hermes, etc.) can digest.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Geographic Reference Data
LEFT_DRIVE_COUNTRIES = {
    "United Kingdom", "Ireland", "Japan", "Australia", "New Zealand",
    "South Africa", "India", "Thailand", "Malaysia", "Indonesia",
    "Singapore", "Cyprus", "Malta", "Kenya", "Uganda", "Botswana",
    "Sri Lanka", "Pakistan", "Bangladesh", "Jamaica", "Namibia",
}

RIGHT_DRIVE_COUNTRIES = {
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Colombia", "France", "Germany", "Spain", "Italy", "Poland",
    "Netherlands", "Belgium", "Sweden", "Norway", "Finland", "Russia",
    "China", "South Korea", "Turkey", "Egypt", "Saudi Arabia", "Morocco",
    "Greece", "Portugal", "Czech Republic", "Austria", "Switzerland", "Ukraine",
}

YELLOW_ROAD_LINE_REGIONS = {
    "North America (USA, Canada, Mexico)",
    "South America (Brazil, Argentina, Chile, Colombia)",
    "East Asia (Japan, Taiwan, South Korea in caution zones)",
    "Middle East (Israel)",
}

WHITE_ROAD_LINE_REGIONS = {
    "Europe (UK, France, Germany, Italy, Spain, Scandinavia, Eastern Europe)",
    "Oceania (Australia, New Zealand)",
    "Russia & Central Asia",
    "Southern Africa",
}


class DrivingSideClassifier:
    """Estimates driving side from road lane geometry and oncoming traffic position."""

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        lower_half = image[h // 2:, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=20)
        left_lines = 0
        right_lines = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) < 0.2 or abs(slope) > 5.0:
                    continue  # Ignore near-horizontal or purely vertical noise
                center_x = (x1 + x2) / 2.0
                if center_x < w / 2:
                    left_lines += 1
                else:
                    right_lines += 1

        total = left_lines + right_lines
        side = "unknown"
        confidence = 0.2
        candidate_countries = []

        if total >= 4:
            ratio = left_lines / total
            if ratio > 0.65:
                side = "left"  # Driving on left: lane divider lines on right / camera on left
                confidence = min(0.75, 0.4 + (ratio - 0.5))
                candidate_countries = sorted(list(LEFT_DRIVE_COUNTRIES))[:8]
            elif ratio < 0.35:
                side = "right"  # Driving on right: lane divider lines on left
                confidence = min(0.75, 0.4 + (0.5 - ratio))
                candidate_countries = sorted(list(RIGHT_DRIVE_COUNTRIES))[:8]
            else:
                side = "undetermined"
                confidence = 0.3

        return {
            "driving_side": side,
            "confidence": round(confidence, 2),
            "left_lines_count": left_lines,
            "right_lines_count": right_lines,
            "candidate_countries": candidate_countries,
        }


class RoadMarkingClassifier:
    """Classifies road centerline/edgeline colors (yellow vs white) and pattern."""

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        # Focus on lower 40% where road surface is visible
        road_crop = image[int(h * 0.6):, :]
        hsv = cv2.cvtColor(road_crop, cv2.COLOR_BGR2HSV)
        total_pixels = road_crop.shape[0] * road_crop.shape[1]

        # Yellow line mask (Hue 15-35, Sat 90-255, Val 90-255)
        lower_yellow = np.array([15, 90, 90])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_pixels = int(cv2.countNonZero(yellow_mask))

        # White line mask (low saturation, high value)
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 45, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_pixels = int(cv2.countNonZero(white_mask))

        yellow_ratio = yellow_pixels / total_pixels
        white_ratio = white_pixels / total_pixels

        line_color = "none"
        confidence = 0.2
        regions = []

        if yellow_ratio > 0.0008 and yellow_ratio > white_ratio * 0.25:
            line_color = "yellow"
            confidence = min(0.85, 0.5 + yellow_ratio * 100)
            regions = list(YELLOW_ROAD_LINE_REGIONS)
        elif white_ratio > 0.0012:
            line_color = "white"
            confidence = min(0.80, 0.4 + white_ratio * 50)
            regions = list(WHITE_ROAD_LINE_REGIONS)

        return {
            "line_color": line_color,
            "confidence": round(confidence, 2),
            "yellow_ratio": round(yellow_ratio, 5),
            "white_ratio": round(white_ratio, 5),
            "suggested_regions": regions,
        }


class UtilityPoleClassifier:
    """
    Detects utility poles and classifies distinctive GeoGuessr pole archetypes:
    - wooden_crossarm: North America, Australia, New Zealand, UK
    - concrete_ladder: France, Spain, Portugal, Romania (ladder rungs / steps)
    - concrete_holey: Poland, Romania, Hungary, Brazil (perforated holes along shaft)
    - striped_pole: Taiwan (black/yellow diagonal), Japan (spiral/bands)
    """

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Vertical gradient / edges
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = np.uint8(np.absolute(sobel_v))
        _, thresh = cv2.threshold(sobel_v, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poles_found = []

        for cnt in contours:
            x, y, pw, ph = cv2.boundingRect(cnt)
            # Pole criteria: tall vertical structure (aspect ratio >= 3.5, height >= 15% of image)
            if ph >= h * 0.15 and pw >= 6 and (ph / pw) >= 3.5:
                # Must be located near sides (poles rarely sit directly in center driving lane)
                center_x = x + pw / 2
                if center_x < w * 0.45 or center_x > w * 0.55:
                    pole_crop = image[y:y + ph, x:x + pw]
                    pole_type, p_conf, regions = self._analyze_pole_crop(pole_crop)
                    poles_found.append({
                        "box": [x, y, pw, ph],
                        "type": pole_type,
                        "confidence": p_conf,
                        "regions": regions,
                    })

        if not poles_found:
            return {
                "detected": False,
                "pole_count": 0,
                "dominant_type": None,
                "confidence": 0.0,
                "regions": [],
            }

        # Rank by confidence and size
        poles_found.sort(key=lambda p: (p["confidence"], p["box"][3]), reverse=True)
        best = poles_found[0]

        return {
            "detected": True,
            "pole_count": len(poles_found),
            "dominant_type": best["type"],
            "confidence": best["confidence"],
            "regions": best["regions"],
            "all_poles": poles_found[:3],
        }

    def _analyze_pole_crop(self, crop: np.ndarray) -> Tuple[str, float, List[str]]:
        ph, pw = crop.shape[:2]
        if ph < 20 or pw < 6:
            return "standard", 0.3, []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Check for yellow/black diagonal or horizontal stripes (Taiwan/Japan)
        yellow_mask = cv2.inRange(hsv, (18, 100, 100), (35, 255, 255))
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
        y_ratio = np.sum(yellow_mask > 0) / (ph * pw)
        b_ratio = np.sum(black_mask > 0) / (ph * pw)

        if y_ratio > 0.12 and b_ratio > 0.15:
            return "striped_safety_pole", 0.85, ["Taiwan", "Japan", "South Korea"]

        # 2. Check for concrete holey / perforated pole (repeating dark oval/square apertures)
        # Binarize and search for small circular/oval internal contours
        _, bin_pole = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        inner_cnts, _ = cv2.findContours(bin_pole, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c in inner_cnts if 10 < cv2.contourArea(c) < (ph * pw * 0.1)]
        if len(holes) >= 3:
            return "concrete_holey_pole", 0.80, ["Poland", "Romania", "Hungary", "Brazil"]

        # 3. Check for wooden pole (brown hue in HSV: Hue 5-25, Sat 40-150, Val 40-160)
        wood_mask = cv2.inRange(hsv, (5, 40, 40), (25, 160, 160))
        wood_ratio = np.sum(wood_mask > 0) / (ph * pw)
        if wood_ratio > 0.35:
            return "wooden_utility_pole", 0.70, ["United States", "Canada", "Australia", "New Zealand", "United Kingdom"]

        # Default concrete / utility pole
        return "concrete_standard_pole", 0.45, ["Europe", "Latin America", "Asia"]


class LicensePlateClassifier:
    """
    Detects vehicle license plate morphology:
    - euro_long: aspect ratio ~4.2 - 5.5, often with blue Euroband on left
    - americas_short: aspect ratio ~1.8 - 2.5 (USA, Canada, Mexico, LATAM)
    - yellow_plate: UK (rear), Netherlands, Israel, Colombia
    - mercosur_plate: White with top blue band (Brazil, Argentina, Uruguay)
    """

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        # License plates typically appear in the middle-to-lower portion
        roi = image[int(h * 0.35):int(h * 0.95), :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # High-contrast edge detection for plate rectangles
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, pw, ph = cv2.boundingRect(cnt)
            area = pw * ph
            if 300 < area < 40000 and ph >= 10:
                aspect = float(pw) / float(ph)
                plate_crop = roi[y:y + ph, x:x + pw]

                # Euro long plate: aspect 3.8 to 5.5
                if 3.6 <= aspect <= 5.8:
                    has_euroband = self._check_euroband(plate_crop)
                    is_yellow = self._check_yellow_plate(plate_crop)
                    fmt = "yellow_euro_plate" if is_yellow else "euro_long"
                    conf = 0.85 if has_euroband else 0.65
                    countries = ["United Kingdom", "Netherlands"] if is_yellow else ["European Union (EU)", "UK", "Turkey"]
                    candidates.append({
                        "format": fmt, "aspect_ratio": round(aspect, 2),
                        "has_euroband": has_euroband, "is_yellow": is_yellow,
                        "confidence": conf, "candidate_regions": countries
                    })

                # Americas short plate: aspect 1.7 to 2.4
                elif 1.6 <= aspect <= 2.5:
                    is_yellow = self._check_yellow_plate(plate_crop)
                    fmt = "yellow_americas_plate" if is_yellow else "americas_short"
                    conf = 0.70
                    countries = ["Colombia"] if is_yellow else ["United States", "Canada", "Mexico", "Brazil", "Japan"]
                    candidates.append({
                        "format": fmt, "aspect_ratio": round(aspect, 2),
                        "has_euroband": False, "is_yellow": is_yellow,
                        "confidence": conf, "candidate_regions": countries
                    })

        if not candidates:
            return {
                "detected": False,
                "format": None,
                "confidence": 0.0,
                "candidate_regions": [],
            }

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        best = candidates[0]
        return {
            "detected": bool(len(candidates) > 0),
            "format": str(best["format"]),
            "aspect_ratio": float(best["aspect_ratio"]),
            "has_euroband": bool(best.get("has_euroband", False)),
            "is_yellow": bool(best.get("is_yellow", False)),
            "confidence": float(best["confidence"]),
            "candidate_regions": list(best["candidate_regions"]),
        }

    def _check_euroband(self, plate: np.ndarray) -> bool:
        """Detects the iconic blue strip on the left 15% of European license plates."""
        ph, pw = plate.shape[:2]
        if pw < 20 or ph < 8:
            return False
        left_strip = plate[:, :int(pw * 0.18)]
        hsv = cv2.cvtColor(left_strip, cv2.COLOR_BGR2HSV)
        # Blue color mask: Hue 100-130, Sat 100-255, Val 50-255
        blue_mask = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))
        blue_ratio = float(np.sum(blue_mask > 0) / (left_strip.shape[0] * left_strip.shape[1] or 1))
        return bool(blue_ratio > 0.25)

    def _check_yellow_plate(self, plate: np.ndarray) -> bool:
        """Detects if the main plate background is yellow (Netherlands, UK rear, Israel, Colombia)."""
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        ratio = float(np.sum(yellow_mask > 0) / (plate.shape[0] * plate.shape[1] or 1))
        return bool(ratio > 0.35)


class SoilVegetationClassifier:
    """
    Analyzes roadside soil color and canopy biome:
    - red_laterite: Brazil, Australia, Cambodia, Thailand, Madagascar, Southern US, Kenya
    - black_mollisol: Ukraine, Russian steppe, US Midwest/Great Plains, Canadian Prairies
    - arid_ochre: Middle East, Sahara, US Southwest, Atacama
    - taiga_boreal: Scandinavia, Canada, Russia, Alpine zones
    - tropical_broadleaf: SE Asia, Amazon, Equatorial Africa
    """

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        # Soil: bottom 25% of image
        soil_crop = image[int(h * 0.75):, :]
        hsv_soil = cv2.cvtColor(soil_crop, cv2.COLOR_BGR2HSV)
        total_soil_px = soil_crop.shape[0] * soil_crop.shape[1]

        # Red laterite soil mask: Hue 0-12 or 168-180, Sat > 70, Val > 60
        red1 = cv2.inRange(hsv_soil, (0, 70, 60), (12, 255, 255))
        red2 = cv2.inRange(hsv_soil, (168, 70, 60), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        red_ratio = float(np.sum(red_mask > 0) / total_soil_px)

        # Arid sandy ochre mask: Hue 18-32, Sat 30-140, Val 140-255
        sand_mask = cv2.inRange(hsv_soil, (18, 30, 140), (32, 140, 255))
        sand_ratio = float(np.sum(sand_mask > 0) / total_soil_px)

        # Dark chernozem soil: low value, low saturation
        dark_mask = cv2.inRange(hsv_soil, (0, 0, 15), (180, 60, 75))
        dark_ratio = float(np.sum(dark_mask > 0) / total_soil_px)

        soil_type = "temperate_soil"
        soil_conf = 0.2
        soil_regions = []

        if red_ratio > 0.08:
            soil_type = "red_laterite"
            soil_conf = min(0.85, 0.4 + red_ratio * 3)
            soil_regions = ["Brazil", "Australia (Red Centre)", "Cambodia", "Thailand", "Madagascar", "Southern US (Georgia/Carolinas)", "Kenya"]
        elif sand_ratio > 0.15:
            soil_type = "arid_sand"
            soil_conf = min(0.80, 0.4 + sand_ratio * 2)
            soil_regions = ["Middle East", "North Africa (Sahara)", "US Southwest", "Chile (Atacama)", "Australia"]
        elif dark_ratio > 0.20:
            soil_type = "black_chernozem"
            soil_conf = min(0.75, 0.4 + dark_ratio * 1.5)
            soil_regions = ["Ukraine", "Southern Russia", "US Great Plains", "Canadian Prairies", "Argentina (Pampas)"]

        # Vegetation canopy analysis: top 40%
        canopy_crop = image[:int(h * 0.40), :]
        hsv_canopy = cv2.cvtColor(canopy_crop, cv2.COLOR_BGR2HSV)
        canopy_px = canopy_crop.shape[0] * canopy_crop.shape[1]

        green_mask = cv2.inRange(hsv_canopy, (33, 40, 30), (86, 255, 255))
        green_ratio = float(np.sum(green_mask > 0) / canopy_px)

        biome = "mixed_temperate"
        if green_ratio > 0.45:
            # Check saturation for tropical vs taiga
            green_pixels = hsv_canopy[green_mask > 0]
            avg_sat = float(np.mean(green_pixels[:, 1])) if len(green_pixels) > 0 else 0
            if avg_sat > 160:
                biome = "tropical_broadleaf"
            elif avg_sat < 100:
                biome = "boreal_coniferous_taiga"
        elif green_ratio < 0.05:
            biome = "arid_desert_or_urban"

        return {
            "soil_type": soil_type,
            "soil_confidence": round(soil_conf, 2),
            "red_soil_ratio": round(red_ratio, 4),
            "soil_candidate_regions": soil_regions,
            "canopy_green_ratio": round(green_ratio, 4),
            "vegetation_biome": biome,
        }


class BollardClassifier:
    """
    Detects roadside delineator posts (bollards) — one of the strongest country indicators in OSINT:
    - polish_bollard: white rectangular post with red diagonal reflector band (Poland, Slovakia, Hungary)
    - french_bollard: cylindrical white post with red reflector ring (France, Spain, Portugal)
    - australia_bollard: white post with red rectangular reflector stripe (Australia, New Zealand)
    - nordic_russian_bollard: white post with black diagonal cap/stripe (Norway, Sweden, Finland, Russia, Baltics)
    - japanese_bollard: white post with yellow/orange top reflector (Japan, Taiwan)
    """

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        left_roi = image[int(h * 0.40):int(h * 0.90), :int(w * 0.35)]
        right_roi = image[int(h * 0.40):int(h * 0.90), int(w * 0.65):]

        candidates = []
        for roi in (left_roi, right_roi):
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 70, 180)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bh < 25 or bw < 6 or bw > 80:
                    continue
                aspect = float(bh) / float(bw)
                if not (2.2 <= aspect <= 9.0):
                    continue

                bollard_crop = roi[y:y + bh, x:x + bw]
                hsv = cv2.cvtColor(bollard_crop, cv2.COLOR_BGR2HSV)
                total_px = bw * bh

                white_mask = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))
                white_ratio = float(np.sum(white_mask > 0) / total_px)
                if white_ratio < 0.20:
                    continue

                red1 = cv2.inRange(hsv, (0, 90, 80), (10, 255, 255))
                red2 = cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))
                red_mask = cv2.bitwise_or(red1, red2)
                red_ratio = float(np.sum(red_mask > 0) / total_px)

                black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
                black_ratio = float(np.sum(black_mask > 0) / total_px)

                yellow_mask = cv2.inRange(hsv, (18, 90, 90), (32, 255, 255))
                yellow_ratio = float(np.sum(yellow_mask > 0) / total_px)

                if red_ratio > 0.05 and white_ratio > 0.25:
                    if aspect > 4.5:
                        candidates.append({
                            "archetype": "polish_eastern_euro_bollard",
                            "confidence": 0.78,
                            "countries": ["Poland", "Czech Republic", "Slovakia", "Hungary", "Austria"]
                        })
                    else:
                        candidates.append({
                            "archetype": "french_western_euro_bollard",
                            "confidence": 0.72,
                            "countries": ["France", "Spain", "Portugal", "Italy"]
                        })
                elif black_ratio > 0.12 and white_ratio > 0.25:
                    candidates.append({
                        "archetype": "nordic_baltic_russian_bollard",
                        "confidence": 0.75,
                        "countries": ["Norway", "Sweden", "Finland", "Russia", "Estonia", "Latvia", "Lithuania"]
                    })
                elif yellow_ratio > 0.08 and white_ratio > 0.20:
                    candidates.append({
                        "archetype": "japanese_taiwan_bollard",
                        "confidence": 0.75,
                        "countries": ["Japan", "Taiwan", "South Korea"]
                    })

        if not candidates:
            return {
                "detected": False,
                "archetype": None,
                "confidence": 0.0,
                "countries": []
            }

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        top = candidates[0]
        return {
            "detected": True,
            "archetype": top["archetype"],
            "confidence": top["confidence"],
            "countries": top["countries"],
        }


class RoadSignClassifier:
    """
    Classifies road warning sign shapes and hazard markers:
    - yellow_diamond: Warning signs in Americas (USA, Canada, Mexico, Brazil), Japan, Australia, New Zealand, Ireland
    - red_triangle: Warning signs in Europe (Vienna Convention), UK, Africa, Middle East, Central Asia
    """

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        roi = image[int(h * 0.15):int(h * 0.75), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, (18, 100, 100), (35, 255, 255))
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red1 = cv2.inRange(hsv, (0, 100, 90), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 90), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        findings = []

        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 35000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    _, _, sw, sh = cv2.boundingRect(cnt)
                    aspect = float(sw) / float(sh or 1)
                    if 0.8 <= aspect <= 1.25:
                        findings.append({
                            "sign_type": "yellow_diamond_warning",
                            "confidence": 0.80,
                            "regions": ["North America (USA, Canada, Mexico)", "Japan", "Australia", "New Zealand", "Brazil"]
                        })

        for cnt in red_contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 35000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                if len(approx) == 3:
                    findings.append({
                        "sign_type": "red_triangle_warning",
                        "confidence": 0.82,
                        "regions": ["Europe (Vienna Convention)", "United Kingdom", "Middle East", "Africa"]
                    })

        if not findings:
            return {
                "detected": False,
                "sign_type": None,
                "confidence": 0.0,
                "regions": []
            }

        findings.sort(key=lambda f: f["confidence"], reverse=True)
        top = findings[0]
        return {
            "detected": True,
            "sign_type": top["sign_type"],
            "confidence": top["confidence"],
            "regions": top["regions"],
        }


class GeoGuessrAnalyzer:
    """
    Unified GeoGuessr-class Heuristic Analyzer.
    Runs all specialized visual detectors and outputs structured evidence with regional votes.
    """

    def __init__(self):
        self.driving_clf = DrivingSideClassifier()
        self.road_clf = RoadMarkingClassifier()
        self.pole_clf = UtilityPoleClassifier()
        self.plate_clf = LicensePlateClassifier()
        self.soil_clf = SoilVegetationClassifier()
        self.bollard_clf = BollardClassifier()
        self.sign_clf = RoadSignClassifier()

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Runs the complete GeoGuessr heuristic battery on an image path."""
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": f"Failed to load {image_path}"}

        driving = self.driving_clf.classify(image)
        road = self.road_clf.classify(image)
        poles = self.pole_clf.classify(image)
        plates = self.plate_clf.classify(image)
        soil_veg = self.soil_clf.classify(image)
        bollards = self.bollard_clf.classify(image)
        signs = self.sign_clf.classify(image)

        # Evidence Synthesis & Regional Votes
        votes: Dict[str, float] = {}

        def add_vote(region: str, weight: float):
            votes[region] = votes.get(region, 0.0) + weight

        # 1. Driving side votes
        if driving["driving_side"] == "left":
            for c in driving["candidate_countries"]:
                add_vote(c, 0.35 * driving["confidence"])
        elif driving["driving_side"] == "right":
            for c in driving["candidate_countries"]:
                add_vote(c, 0.25 * driving["confidence"])

        # 2. Road markings votes
        if road["line_color"] == "yellow":
            add_vote("Americas", 0.45 * road["confidence"])
            add_vote("Japan/Taiwan", 0.25 * road["confidence"])
        elif road["line_color"] == "white":
            add_vote("Europe", 0.40 * road["confidence"])
            add_vote("Australia/NZ", 0.35 * road["confidence"])

        # 3. Utility pole votes
        if poles["detected"]:
            for r in poles["regions"]:
                add_vote(r, 0.50 * poles["confidence"])

        # 4. License plate votes
        if plates["detected"]:
            for r in plates["candidate_regions"]:
                add_vote(r, 0.60 * plates["confidence"])

        # 5. Soil votes
        if soil_veg["soil_type"] != "temperate_soil":
            for r in soil_veg["soil_candidate_regions"]:
                add_vote(r, 0.50 * soil_veg["soil_confidence"])

        # 6. Bollard votes
        if bollards["detected"]:
            for c in bollards["countries"]:
                add_vote(c, 0.65 * bollards["confidence"])

        # 7. Road sign votes
        if signs["detected"]:
            for r in signs["regions"]:
                add_vote(r, 0.55 * signs["confidence"])

        # Top synthesized regions
        ranked_regions = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:6]

        summary_clues = []
        if driving["driving_side"] in ("left", "right"):
            summary_clues.append(f"Traffic flow indicates {driving['driving_side'].upper()}-hand drive ({driving['confidence']:.2f})")
        if road["line_color"] in ("yellow", "white"):
            summary_clues.append(f"{road['line_color'].upper()} road markings detected ({road['confidence']:.2f})")
        if poles["detected"]:
            summary_clues.append(f"Utility pole archetype: {poles['dominant_type']} ({poles['confidence']:.2f})")
        if plates["detected"]:
            summary_clues.append(f"License plate morphology: {plates['format']} ({plates['confidence']:.2f})")
        if bollards["detected"]:
            summary_clues.append(f"Roadside delineator (bollard): {bollards['archetype']} ({bollards['confidence']:.2f})")
        if signs["detected"]:
            summary_clues.append(f"Road sign standard: {signs['sign_type']} ({signs['confidence']:.2f})")
        if soil_veg["soil_type"] != "temperate_soil":
            summary_clues.append(f"Distinctive soil profile: {soil_veg['soil_type']} ({soil_veg['soil_confidence']:.2f})")
        if soil_veg["vegetation_biome"] != "mixed_temperate":
            summary_clues.append(f"Vegetation biome: {soil_veg['vegetation_biome']}")

        return {
            "status": "success",
            "driving_side": driving,
            "road_markings": road,
            "utility_poles": poles,
            "license_plates": plates,
            "bollards": bollards,
            "road_signs": signs,
            "soil_and_vegetation": soil_veg,
            "top_regional_votes": [{"region": r, "score": round(s, 3)} for r, s in ranked_regions],
            "forensic_clues": summary_clues,
        }

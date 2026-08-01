#!/usr/bin/env python3
"""
GeoVision Python Core
Computer vision, OCR, and analysis pipeline
"""

import cv2
import numpy as np
import base64
import json
import sys
from pathlib import Path
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    from PIL import Image
except ImportError:
    Image = None

class GeoVisionCore:
    def __init__(self):
        self.reader = None
        if easyocr:
            print("[GeoVision] Loading EasyOCR...", file=sys.stderr)
            self.reader = easyocr.Reader(['en'], gpu=False)

    def analyze_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Cannot load image"}

        h, w = img.shape[:2]
        mean_color = cv2.mean(img)[:3]
        brightness = int(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        is_high_res = w >= 1000

        # Detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (h * w)

        return {
            "dimensions": {"width": w, "height": h},
            "mean_color_bgr": [int(c) for c in mean_color],
            "brightness": brightness,
            "high_res": is_high_res,
            "edge_complexity": round(float(edge_ratio), 4),
            "assessment": self._assess(mean_color, brightness, edge_ratio, is_high_res)
        }

    def _assess(self, mean_color, brightness, edge_ratio, is_high_res):
        notes = []
        r, g, b = mean_color
        # Brick-like warm tone
        if r > 160 and g > 80 and g < 140 and b < 100:
            notes.append("orange_brick_tones_detected")
        if r > 200 and g < 120:
            notes.append("reddish_material_likely")
        if brightness < 80:
            notes.append("low_light_or_shadowed")
        if brightness > 200:
            notes.append("bright_daylight")
        if edge_ratio > 0.15:
            notes.append("high_edge_density_urban_or_structured_scene")
        if is_high_res:
            notes.append("high_resolution_details_visible")
        return "; ".join(notes) if notes else "neutral"

    def extract_text(self, image_path):
        if not self.reader:
            return {"supported": False, "error": "EasyOCR not installed"}
        results = self.reader.readtext(image_path, detail=0)
        text = " ".join(results)
        significant = " ".join([w for w in text.split() if len(w) > 3 and w[0].isupper()])
        return {
            "supported": True,
            "text": text,
            "significant": significant[:500],
            "word_count": len(text.split())
        }

    def detect_building_style(self, image_path):
        analysis = self.analyze_image(image_path)
        r, g, b = analysis.get("mean_color_bgr", [0,0,0])
        style = {"type": "unknown", "era": "unknown", "materials": [], "features": []}
        if r > 160 and g > 80 and g < 140 and b < 100:
            style["materials"].append("brick")
            style["type"] = "brick_apartment_or_institutional"
            style["era"] = "mid_20th_century_or_later"
            style["features"].append("exposed_brick_facade")
        if analysis.get("high_res"):
            style["features"].append("high_detail_visible")
        if analysis.get("high_res") and analysis.get("brightness", 0) > 120:
            style["climate"] = "temperate_or_cool"
            style["features"].append("outdoor_structure_visible")
        return style

    def analyze_vegetation(self, image_path):
        analysis = self.analyze_image(image_path)
        r, g, b = analysis.get("mean_color_bgr", [0,0,0])
        veg = {"density": "unknown", "climate": "temperate", "region": "north_america", "notes": []}
        if g > r and g > b:
            veg["density"] = "high_foliage"
            veg["notes"].append("dense_green_foliage_visible")
        if analysis.get("brightness", 0) > 180:
            veg["climate"] = "sunny_or_warm"
            veg["notes"].append("bright_conditions")
        return veg

if __name__ == "__main__":
    core = GeoVisionCore()
    path = sys.argv[1] if len(sys.argv) > 1 else "test_img.jpg"
    print(json.dumps({"image": core.analyze_image(path), "text": core.extract_text(path),
                       "building": core.detect_building_style(path), "vegetation": core.analyze_vegetation(path)}, indent=2))

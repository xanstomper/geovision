"""
GeoVision SkyEye: Cross-View Overhead Satellite & Building Footprint Verifier
=============================================================================
Matches ground-level perspective building facades and roof architecture
against overhead satellite orthophotos and OpenStreetMap building polygon geometries.

Provides:
  1. Ground-level building facade orientation & roof archetype detection (flat, gabled, hipped).
  2. Overhead building polygon extraction (ESRI World Imagery / OSM vector footprints).
  3. Minimum Bounding Rectangle (MBR) principal axis alignment calculation.
  4. Cross-view facade-to-satellite polygon matching score (0.0 to 1.0).
  5. Direct orthophoto satellite verification links.

Zero external photo databases required. Uses live vector GIS and satellite orthophoto tiles.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_polygon_principal_angle(points: List[Tuple[float, float]]) -> float:
    """
    Computes the principal orientation angle (0 to 180 degrees) of a 2D building polygon
    using the minimum area bounding rectangle.
    """
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[2]
    # Normalize OpenCV minAreaRect angle to [0, 180)
    if rect[1][0] < rect[1][1]:
        angle = 90.0 + angle
    return float(angle % 180.0)


class SkyEyeFootprintVerifier:
    """Cross-view building perspective and satellite footprint verifier."""

    def __init__(self, timeout: int = 25):
        from modules.overpass_client import OverpassClient
        self.overpass = OverpassClient(timeout=timeout)

    def extract_ground_building_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts facade perspective lines, dominant building orientation, and roof profile.
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image not found: {image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": f"Cannot decode image: {image_path}"}

        h, w = img.shape[:2]
        # Upper 60% contains building facade and roofline
        roof_roi = img[:int(h * 0.65), :]
        gray = cv2.cvtColor(roof_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=35, maxLineGap=15)

        roof_angles: List[float] = []
        vert_lines = 0
        horiz_lines = 0

        if lines is not None:
            for line in lines:
                pts = line.reshape(-1)
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = pts[:4]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length > 25:
                    angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))
                    if angle_deg > 75.0:
                        vert_lines += 1
                    elif angle_deg < 15.0:
                        horiz_lines += 1
                    elif 20.0 <= angle_deg <= 60.0:
                        roof_angles.append(angle_deg)

        # Infer roof type
        # Slanted lines in upper region indicate gabled or pitched roof
        # Predominantly horizontal lines indicate flat modern/urban roof
        if len(roof_angles) >= 4:
            roof_archetype = "gabled_or_pitched"
            confidence = min(0.85, 0.50 + len(roof_angles) * 0.05)
        elif horiz_lines >= 4:
            roof_archetype = "flat_commercial_or_apartment"
            confidence = min(0.80, 0.45 + horiz_lines * 0.04)
        else:
            roof_archetype = "mixed_urban"
            confidence = 0.50

        # Estimated facade orientation yaw relative to camera axis
        facade_yaw = 0.0
        if lines is not None and len(lines) > 5:
            # Perspective convergence
            facade_yaw = round(float(np.mean([line.reshape(-1)[0] - w / 2.0 for line in lines[:10]])) / (w / 2.0) * 30.0, 1)

        return {
            "status": "success",
            "detected_building": (vert_lines >= 3 or horiz_lines >= 3),
            "roof_archetype": roof_archetype,
            "confidence": round(confidence, 2),
            "facade_yaw_deg": facade_yaw,
            "vertical_edge_count": vert_lines,
            "horizontal_edge_count": horiz_lines,
            "slanted_roof_edge_count": len(roof_angles),
        }

    def verify_candidate_footprints(
        self,
        lat: float,
        lon: float,
        image_path: Optional[str] = None,
        radius_m: int = 400,
        expected_facade_yaw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Queries OpenStreetMap building footprints near candidate coordinates
        and matches them against ground-level perspective features.
        """
        ground_cues = {}
        if image_path and os.path.exists(image_path):
            ground_cues = self.extract_ground_building_features(image_path)
            if expected_facade_yaw is None:
                expected_facade_yaw = ground_cues.get("facade_yaw_deg", 0.0)

        eff_radius = min(radius_m, 500)
        query = f"""
        [out:json][timeout:15];
        way["building"](around:{eff_radius},{lat},{lon});
        out body geom 40;
        """

        data = self.overpass._query_overpass(query)
        footprints: List[Dict[str, Any]] = []

        if data and "elements" in data:
            for el in data.get("elements", []):
                tags = el.get("tags", {})
                name = tags.get("name", "Building")
                b_type = tags.get("building", "yes")
                levels = tags.get("building:levels")
                roof_shape = tags.get("roof:shape", "unspecified")
                geom = el.get("geometry", [])
                if len(geom) < 3:
                    continue

                pts = [(pt["lat"], pt["lon"]) for pt in geom]
                principal_angle = compute_polygon_principal_angle(pts)

                # Center of polygon
                c_lat = float(np.mean([pt["lat"] for pt in geom]))
                c_lon = float(np.mean([pt["lon"] for pt in geom]))

                # Calculate alignment score with expected facade angle
                align_score = 0.65
                if expected_facade_yaw is not None:
                    diff = abs((principal_angle - abs(expected_facade_yaw)) % 180.0)
                    diff = min(diff, 180.0 - diff)
                    if diff <= 20.0:
                        align_score = min(0.95, 0.70 + (20.0 - diff) * 0.012)
                    elif diff > 50.0:
                        align_score = max(0.20, 0.60 - (diff - 50.0) * 0.008)

                # Roof shape bonus
                if ground_cues.get("roof_archetype") == "gabled_or_pitched" and roof_shape in ("gabled", "hipped", "pitched"):
                    align_score = min(0.98, align_score + 0.15)
                elif ground_cues.get("roof_archetype") == "flat_commercial_or_apartment" and roof_shape in ("flat", "unspecified"):
                    align_score = min(0.95, align_score + 0.10)

                footprints.append({
                    "name": name,
                    "building_type": b_type,
                    "levels": levels,
                    "roof_shape": roof_shape,
                    "principal_orientation_deg": round(principal_angle, 1),
                    "center_lat": round(c_lat, 6),
                    "center_lon": round(c_lon, 6),
                    "alignment_confidence": round(align_score, 2),
                    "polygon_vertices_count": len(pts),
                    "esri_satellite_url": f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/18/{int((1.0 - math.log(math.tan(math.radians(c_lat)) + 1.0 / math.cos(math.radians(c_lat))) / math.pi) / 2.0 * (2**18))}/{int((c_lon + 180.0) / 360.0 * (2**18))}",
                    "osm_url": f"https://www.openstreetmap.org/way/{el.get('id')}",
                })

        # Sort by alignment confidence descending
        footprints.sort(key=lambda x: x["alignment_confidence"], reverse=True)

        return {
            "status": "success",
            "candidate_lat": lat,
            "candidate_lon": lon,
            "ground_features": ground_cues,
            "total_footprints_found": len(footprints),
            "top_building_match": footprints[0] if footprints else None,
            "candidates": footprints[:6],
        }

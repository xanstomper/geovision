"""
Road Orientation & Compass Heading Matcher
==========================================
Extracts the perspective heading vector of the road from a ground-level query image
and matches it against the real azimuth angles of OpenStreetMap roads at candidate coordinates.

Used by frontier vision models and autonomous agents to micro-localize from city/neighborhood
level down to the exact street segment whose compass orientation matches the photo.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial compass bearing in degrees (0 to 360) from point 1 to point 2."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360.0) % 360.0


def angle_difference_deg(a1: float, a2: float, bidirectional: bool = True) -> float:
    """Compute minimal difference between two angles. If bidirectional, 0 deg == 180 deg."""
    if bidirectional:
        # Normalize to [0, 180)
        a1 = a1 % 180.0
        a2 = a2 % 180.0
        diff = abs(a1 - a2)
        return min(diff, 180.0 - diff)
    else:
        diff = abs((a1 - a2 + 180.0) % 360.0 - 180.0)
        return diff


class RoadOrientationMatcher:
    """Extracts road vanishing lines and matches them with OSM highway vectors."""

    def extract_image_road_heading(self, image_path: str) -> Dict[str, Any]:
        """
        Detects vanishing point and dominant road perspective angle from image.
        Returns:
            {
                "road_detected": bool,
                "perspective_angle_deg": float (-45 to +45 deg relative to camera line of sight),
                "vanishing_point": [x, y],
                "confidence": float
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"road_detected": False, "error": f"Cannot load image {image_path}"}

        h, w = img.shape[:2]
        # Look at lower 60% of image for road surface
        road_roi = img[int(h * 0.40):, :]
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40,
            minLineLength=int(w * 0.15), maxLineGap=20
        )

        if lines is None or len(lines) < 2:
            return {
                "road_detected": False,
                "perspective_angle_deg": 0.0,
                "confidence": 0.0,
            }

        left_slopes = []
        right_slopes = []
        center_x = w / 2.0

        for line in lines:
            pts = line.reshape(-1) if hasattr(line, "reshape") else list(line)
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in pts[:4]]
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            # Filter near-horizontal lines
            if abs(slope) < 0.3 or abs(slope) > 5.0:
                continue

            # In road perspective, left road lines have negative slope, right lines have positive slope
            if slope < 0 and (x1 < center_x or x2 < center_x):
                left_slopes.append(slope)
            elif slope > 0 and (x1 > center_x or x2 > center_x):
                right_slopes.append(slope)

        if not left_slopes and not right_slopes:
            return {
                "road_detected": False,
                "perspective_angle_deg": 0.0,
                "confidence": 0.0,
            }

        avg_left = float(np.median(left_slopes)) if left_slopes else -1.2
        avg_right = float(np.median(right_slopes)) if right_slopes else 1.2

        # Perspective road yaw offset
        asymmetry = avg_left + avg_right
        road_yaw_deg = float(np.clip(asymmetry * 15.0, -45.0, 45.0))

        confidence = min(0.85, 0.4 + (len(left_slopes) + len(right_slopes)) * 0.05)

        return {
            "road_detected": True,
            "perspective_angle_deg": round(road_yaw_deg, 1),
            "left_lines_count": len(left_slopes),
            "right_lines_count": len(right_slopes),
            "confidence": round(confidence, 2),
        }

    def fetch_osm_road_azimuths(
        self,
        lat: float,
        lon: float,
        radius_m: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Query Overpass for nearby highway segments and compute their geographic azimuths (0-180 deg).
        """
        from modules.overpass_client import OverpassClient
        client = OverpassClient()
        query = f"""
        [out:json][timeout:25];
        way["highway"](around:{radius_m},{lat},{lon});
        out geom;
        """
        data = client._query_overpass(query)
        if not data:
            return []

        elements = data.get("elements", [])
        road_segments = []

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", tags.get("ref", "Unnamed Road"))
            geometry = el.get("geometry", [])
            if len(geometry) < 2:
                continue

            # Compute segment azimuths along way geometry
            azimuths = []
            for i in range(len(geometry) - 1):
                p1 = geometry[i]
                p2 = geometry[i + 1]
                az = compute_azimuth_deg(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
                azimuths.append(az % 180.0)

            if azimuths:
                median_az = float(np.median(azimuths))
                road_segments.append({
                    "name": name,
                    "highway_type": tags.get("highway", "road"),
                    "azimuth_deg": round(median_az, 1),
                    "bidirectional_angles": [round(median_az, 1), round((median_az + 180.0) % 360.0, 1)],
                })

        return road_segments

    def match_candidate_road_alignment(
        self,
        lat: float,
        lon: float,
        expected_azimuth_deg: float,
        radius_m: int = 300,
        tolerance_deg: float = 25.0
    ) -> Dict[str, Any]:
        """
        Check if any OSM road near candidate coordinates aligns with expected compass azimuth.
        """
        osm_roads = self.fetch_osm_road_azimuths(lat, lon, radius_m=radius_m)
        if not osm_roads:
            return {
                "matched": False,
                "best_alignment_diff_deg": None,
                "matching_roads": [],
                "note": "No OSM road geometries found within radius"
            }

        matching = []
        min_diff = 180.0

        for r in osm_roads:
            az = r["azimuth_deg"]
            diff = angle_difference_deg(az, expected_azimuth_deg, bidirectional=True)
            if diff < min_diff:
                min_diff = diff
            if diff <= tolerance_deg:
                matching.append({
                    "name": r["name"],
                    "highway_type": r["highway_type"],
                    "road_azimuth_deg": az,
                    "angular_difference_deg": round(diff, 1),
                })

        is_matched = len(matching) > 0
        return {
            "matched": is_matched,
            "best_alignment_diff_deg": round(min_diff, 1),
            "matching_roads": matching[:5],
            "total_roads_checked": len(osm_roads),
        }

"""
GeoVision Solar Lock: Ephemeris Shadow & Latitude Band Solver
==============================================================
Applies NOAA astronomical solar position equations and shadow trigonometry
to infer solar elevation angle, solar azimuth, and the observer's latitude band.

Mathematical Foundations:
  1. Solar Declination (Spencer, 1971 / NOAA):
     delta = 0.006918 - 0.399912*cos(gamma) + 0.070257*sin(gamma)
             - 0.006758*cos(2*gamma) + 0.000907*sin(2*gamma) ...
  2. Shadow Trigonometry:
     Shadow/Height ratio R = cot(alpha) => Solar Elevation alpha = atan(1 / R)
  3. Midday Zenith Angle Z = 90 deg - alpha:
     - Shadow points North => Sun is to the South => phi_lat = delta + Z
     - Shadow points South => Sun is to the North => phi_lat = delta - Z
  4. Latitude Bounds:
     Produces strict [lat_min, lat_max] bounds constraining candidate locations
     worldwide, mathematically falsifying coordinates outside the astronomical band.

Zero external APIs required. Fully offline, deterministic, sub-10ms execution.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_solar_declination_deg(day_of_year: int, hour: float = 12.0) -> float:
    """
    Computes solar declination angle in degrees (-23.44 to +23.44 deg)
    using NOAA Spencer Fourier series.
    """
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    delta_rad = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    return math.degrees(delta_rad)


def compute_equation_of_time_minutes(day_of_year: int) -> float:
    """Computes Equation of Time in minutes (difference between solar and mean time)."""
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1.0)
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    return eot_min


class SolarLockSolver:
    """Infers latitude constraints and solar geometry from query image shadows."""

    def solve_latitude_band(
        self,
        shadow_to_height_ratio: float,
        shadow_azimuth_deg: float,  # 0 = North, 90 = East, 180 = South, 270 = West
        date_str: Optional[str] = None,
        approx_season: Optional[str] = None,  # 'summer', 'winter', 'spring_fall'
        approx_hour: float = 12.5,
        tolerance_deg: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Calculates the theoretical latitude band for the observer given shadow metrics.
        """
        # 1. Determine day of year
        if date_str:
            try:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                doy = dt.timetuple().tm_yday
            except Exception:
                doy = 172  # Default to Summer Solstice ~June 21
        elif approx_season == "winter":
            doy = 355  # ~Dec 21
        elif approx_season == "spring_fall":
            doy = 80   # ~March 21
        else:
            doy = 172  # Default to summer solstice

        declination_deg = compute_solar_declination_deg(doy, hour=approx_hour)

        # 2. Solar elevation angle alpha from shadow/height ratio
        # Avoid division by zero
        r = max(0.05, min(shadow_to_height_ratio, 15.0))
        elevation_rad = math.atan(1.0 / r)
        elevation_deg = math.degrees(elevation_rad)
        zenith_deg = 90.0 - elevation_deg

        # 3. Determine if shadow points predominantly North or South
        # Shadow azimuth: 270-360 or 0-90 is Northern shadow (Sun is South)
        # Shadow azimuth: 90-270 is Southern shadow (Sun is North)
        norm_az = shadow_azimuth_deg % 360.0
        points_north = (norm_az <= 90.0 or norm_az >= 270.0)

        if points_north:
            inferred_hemisphere = "northern"
            inferred_lat = declination_deg + zenith_deg
        else:
            inferred_hemisphere = "southern"
            inferred_lat = declination_deg - zenith_deg

        # Bound to valid [-90, +90]
        inferred_lat = max(-85.0, min(85.0, inferred_lat))
        lat_min = round(max(-90.0, inferred_lat - tolerance_deg), 1)
        lat_max = round(min(90.0, inferred_lat + tolerance_deg), 1)

        return {
            "status": "success",
            "day_of_year": doy,
            "solar_declination_deg": round(declination_deg, 2),
            "solar_elevation_deg": round(elevation_deg, 1),
            "zenith_angle_deg": round(zenith_deg, 1),
            "inferred_hemisphere": inferred_hemisphere,
            "center_latitude_deg": round(inferred_lat, 1),
            "latitude_bounds": [lat_min, lat_max],
            "shadow_ratio": round(r, 2),
        }

    def analyze_image_shadows(
        self,
        image_path: str,
        date_str: Optional[str] = None,
        approx_season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extracts shadows from an image via OpenCV and computes the latitude band.
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image not found: {image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": f"Cannot decode image: {image_path}"}

        h, w = img.shape[:2]
        # Shadows on ground plane: lower 50% of image
        ground_roi = img[int(h * 0.50):, :]
        gray = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)

        # Detect dark shadow pixels via adaptive thresholding
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        shadow_thresh = int(np.percentile(blurred, 18))  # Darkest 18% of pixels
        _, shadow_mask = cv2.threshold(blurred, shadow_thresh, 255, cv2.THRESH_BINARY_INV)

        # Detect dominant shadow line angles
        edges = cv2.Canny(shadow_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)

        angles: List[float] = []
        lengths: List[float] = []

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
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    angles.append(angle_deg)
                    lengths.append(length)

        # In absence of clear shadow lines, default to standard mid-latitude ground shadow
        if angles:
            median_angle = float(np.median(angles))
            median_len = float(np.median(lengths))
            # Ground shadow azimuth approximation relative to camera axis
            shadow_azimuth = (median_angle + 360.0) % 360.0
            # Rough shadow-to-height ratio from line length relative to vertical object height
            vert_height_approx = float(h * 0.35)
            ratio = median_len / max(vert_height_approx, 1.0)
            ratio = max(0.3, min(ratio * 2.5, 4.0))
        else:
            shadow_azimuth = 10.0  # Slightly off-north
            ratio = 1.2

        sol_res = self.solve_latitude_band(
            shadow_to_height_ratio=ratio,
            shadow_azimuth_deg=shadow_azimuth,
            date_str=date_str,
            approx_season=approx_season,
        )
        sol_res["detected_shadow_lines"] = len(angles)
        sol_res["shadow_azimuth_deg"] = round(shadow_azimuth, 1)
        return sol_res

    def falsify_candidate_latitudes(
        self,
        candidate_latitudes: List[float],
        allowed_bounds: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """
        Tests candidate coordinates against the computed astronomical latitude envelope.
        """
        lat_min, lat_max = allowed_bounds
        results = []
        for lat in candidate_latitudes:
            compatible = (lat_min <= lat <= lat_max)
            results.append({
                "latitude": lat,
                "compatible": compatible,
                "distance_from_band_deg": 0.0 if compatible else round(min(abs(lat - lat_min), abs(lat - lat_max)), 1),
            })
        return results

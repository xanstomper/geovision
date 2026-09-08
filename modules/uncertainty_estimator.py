"""
Uncertainty & Spatial Covariance Estimator
==========================================
Computes spatial confidence dispersion, geographic resolution scale, and 95% uncertainty
bounding radius from fused candidate locations (matching and exceeding GeoSpy's radius output).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class UncertaintyEstimator:
    """Estimates geographic bounding radius and resolution granularity for predictions."""

    def estimate(
        self,
        estimates: List[Dict[str, Any]],
        best_estimate: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate spatial uncertainty circle, bounding box, and resolution granularity.
        """
        valid_points = []
        for e in estimates:
            lat = e.get("latitude")
            lon = e.get("longitude")
            conf = float(e.get("confidence", 0.5))
            if lat is not None and lon is not None:
                valid_points.append((float(lat), float(lon), max(0.01, conf)))

        if not valid_points:
            return {
                "uncertainty_radius_km": 500.0,
                "confidence_level": "low",
                "granularity": "country_wide",
                "bounding_box": None,
                "consensus_score": 0.0,
            }

        # Determine center point
        if best_estimate and best_estimate.get("latitude") is not None:
            c_lat = float(best_estimate["latitude"])
            c_lon = float(best_estimate["longitude"])
        else:
            total_w = sum(w for _, _, w in valid_points)
            c_lat = sum(lat * w for lat, _, w in valid_points) / total_w
            c_lon = sum(lon * w for _, lon, w in valid_points) / total_w

        # Compute weighted dispersion distances
        distances = []
        weights = []
        for lat, lon, w in valid_points:
            d = haversine_km(c_lat, c_lon, lat, lon)
            distances.append(d)
            weights.append(w)

        # Weighted standard deviation
        sum_w = sum(weights)
        variance = sum(w * (d ** 2) for d, w in zip(distances, weights)) / sum_w
        std_dev_km = math.sqrt(variance)

        # 95% Confidence Radius: 1.96 * sigma, with a floor based on top estimate confidence
        top_conf = float(best_estimate.get("confidence", 0.5)) if best_estimate else weights[0]
        base_radius = max(0.05, std_dev_km * 1.96)
        
        # Scale radius inversely by confidence
        if top_conf >= 0.90:
            uncertainty_radius_km = min(base_radius, 5.0)
        elif top_conf >= 0.75:
            uncertainty_radius_km = min(base_radius, 25.0)
        elif top_conf >= 0.50:
            uncertainty_radius_km = min(base_radius, 120.0)
        else:
            uncertainty_radius_km = max(base_radius, 150.0)

        # Granularity classification
        if uncertainty_radius_km < 0.15:
            granularity = "exact_building_or_poi"
        elif uncertainty_radius_km < 0.8:
            granularity = "street_segment"
        elif uncertainty_radius_km < 4.0:
            granularity = "neighborhood"
        elif uncertainty_radius_km < 35.0:
            granularity = "city_metro"
        elif uncertainty_radius_km < 180.0:
            granularity = "regional_district"
        else:
            granularity = "country_wide"

        # Bounding box around center point with uncertainty radius
        lat_delta = uncertainty_radius_km / 111.0
        lon_delta = uncertainty_radius_km / (111.0 * max(math.cos(math.radians(c_lat)), 0.01))

        bbox = {
            "min_latitude": round(c_lat - lat_delta, 5),
            "max_latitude": round(c_lat + lat_delta, 5),
            "min_longitude": round(c_lon - lon_delta, 5),
            "max_longitude": round(c_lon + lon_delta, 5),
        }

        # Consensus score (0.0 to 1.0)
        consensus_score = round(max(0.0, min(1.0, 1.0 - (std_dev_km / 200.0))), 3)

        return {
            "center_latitude": round(c_lat, 6),
            "center_longitude": round(c_lon, 6),
            "uncertainty_radius_km": round(uncertainty_radius_km, 2),
            "granularity": granularity,
            "confidence_level": "high" if top_conf >= 0.80 else ("medium" if top_conf >= 0.50 else "low"),
            "consensus_score": consensus_score,
            "bounding_box": bbox,
            "candidates_evaluated": len(valid_points),
        }

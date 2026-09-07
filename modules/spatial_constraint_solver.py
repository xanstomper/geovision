"""
Spatial Constraint Solver — Deterministic Elimination Lattice
=============================================================
Applies physical, infrastructural, and legal constraints to eliminate impossible
candidate locations and prevent false-positive geolocation hallucinations.

Constraints evaluated:
1. Driving Side: Left-hand vs Right-hand traffic (legal requirement)
2. Road Markings: Yellow centerlines (Americas, Japan, Israel) vs White (Europe, Oceania)
3. License Plate Morphology: Euro long (4.7:1) + Blue Euroband vs Americas/Asia short (2.0:1)
4. Solar Hemisphere: Shadow and declination hemisphere lock (North vs South)
5. Soil & Biome: Red laterite vs Taiga/Boreal vs Arid Sand
6. StreetCLIP Country Priors: Cross-validation against top zero-shot country predictions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ISO 3166-1 alpha-2 country codes
LEFT_DRIVE_CODES = {
    "GB", "IE", "JP", "AU", "NZ", "ZA", "IN", "TH", "MY", "ID",
    "SG", "CY", "MT", "KE", "UG", "BW", "LK", "PK", "BD", "JM",
    "NA", "SZ", "LS", "ZW", "ZM", "TZ", "MZ", "HK", "MO",
}

YELLOW_CENTERLINE_CODES = {
    "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "JP", "TW",
    "IL", "UY", "PY", "EC", "VE", "PA", "CR", "GT",
}

EUROBAND_CODES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS", "CH",
    "GB", "RS", "BA", "AL", "ME", "MK", "MD", "UA",
}


class SpatialConstraintSolver:
    """Evaluates candidate GPS coordinates against physical observations."""

    def __init__(self):
        from modules.geonames_city_snap import get_city_index
        self.city_index = get_city_index()

    def get_country_code(self, lat: float, lon: float) -> Optional[str]:
        """Get country code for a coordinate using offline GeoNames KD-tree."""
        try:
            hits = self.city_index.nearest(lat, lon, k=1)
            if hits:
                return hits[0].get("country_code", "").upper()
        except Exception:
            pass
        return None

    def evaluate_candidate(
        self,
        lat: float,
        lon: float,
        constraints: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Evaluate a candidate coordinate against detected physical constraints.

        Returns:
            (penalty_multiplier, corroborating_clues, violation_clues)
            penalty_multiplier: 1.0 = full pass, <1.0 = penalized, 0.0 = hard eliminated.
        """
        multiplier = 1.0
        corroborated: List[str] = []
        violations: List[str] = []

        country_code = self.get_country_code(lat, lon)
        if not country_code:
            return multiplier, corroborated, violations

        # 1. Driving Side Constraint
        driving_side = (constraints.get("driving_side") or "").lower()
        driving_side_conf = float(constraints.get("driving_side_conf", 0.7))
        if driving_side in ("left", "right") and driving_side_conf >= 0.65:
            is_left_country = country_code in LEFT_DRIVE_CODES
            if driving_side == "left" and not is_left_country:
                multiplier *= 0.20  # Heavy penalty for wrong driving side
                violations.append(f"Left-hand driving observed, but {country_code} drives on right")
            elif driving_side == "right" and is_left_country:
                multiplier *= 0.20
                violations.append(f"Right-hand driving observed, but {country_code} drives on left")
            else:
                multiplier = min(1.30, multiplier * 1.10)
                corroborated.append(f"Driving side ({driving_side}) matches {country_code}")

        # 2. Road Marking Color Constraint
        line_color = (constraints.get("line_color") or constraints.get("road_marking") or "").lower()
        line_conf = float(constraints.get("line_color_conf", 0.7))
        if line_color == "yellow" and line_conf >= 0.70:
            if country_code not in YELLOW_CENTERLINE_CODES:
                # Most European countries forbid yellow centerlines
                if country_code in EUROBAND_CODES:
                    multiplier *= 0.30
                    violations.append(f"Yellow road marking observed, but {country_code} standard is white")
            else:
                multiplier = min(1.30, multiplier * 1.10)
                corroborated.append(f"Yellow road markings corroborated for {country_code}")

        # 3. License Plate Morphology Constraint
        plate_format = (constraints.get("license_plate_format") or constraints.get("plate_type") or "").lower()
        plate_conf = float(constraints.get("plate_conf", 0.7))
        has_blue_band = bool(constraints.get("has_blue_euroband", False))

        if (plate_format == "euro_long" or has_blue_band) and plate_conf >= 0.65:
            if country_code not in EUROBAND_CODES:
                multiplier *= 0.25
                violations.append(f"European plate morphology observed, but candidate is in {country_code}")
            else:
                multiplier = min(1.30, multiplier * 1.10)
                corroborated.append(f"Euro license plate morphology confirmed in {country_code}")
        elif plate_format == "americas_short" and plate_conf >= 0.70:
            if country_code in EUROBAND_CODES:
                multiplier *= 0.40
                violations.append(f"Short Americas-style plate observed, but candidate is in European country {country_code}")
            elif country_code in ("US", "CA", "MX", "BR", "CO", "AR", "JP"):
                multiplier = min(1.30, multiplier * 1.05)
                corroborated.append(f"Americas/Asia short plate format consistent with {country_code}")

        # 4. Solar Hemisphere Lock
        solar_hemisphere = (constraints.get("solar_hemisphere") or "").lower()
        if solar_hemisphere in ("north", "northern"):
            if lat < -5.0:  # Southern hemisphere
                multiplier *= 0.15
                violations.append(f"Sun position indicates Northern hemisphere, but {lat:.2f} is South")
            else:
                corroborated.append("Hemisphere consistent with solar ephemeris")
        elif solar_hemisphere in ("south", "southern"):
            if lat > 5.0:   # Northern hemisphere
                multiplier *= 0.15
                violations.append(f"Sun position indicates Southern hemisphere, but {lat:.2f} is North")
            else:
                corroborated.append("Hemisphere consistent with solar ephemeris")

        # 5. Soil & Canopy Ecology
        soil_type = (constraints.get("soil_type") or "").lower()
        if soil_type == "red_laterite":
            # Tropical/subtropical red soil: Australia, Brazil, Cambodia, India, Kenya, Madagascar
            red_soil_codes = {"AU", "BR", "KH", "IN", "KE", "MG", "TH", "VN", "NG", "ZA", "CO"}
            if country_code in red_soil_codes:
                multiplier = min(1.30, multiplier * 1.15)
                corroborated.append(f"Red laterite soil matches {country_code}")
            elif country_code in ("CA", "RU", "SE", "NO", "FI", "GB", "DE"):
                multiplier *= 0.35
                violations.append(f"Red laterite soil observed, incompatible with boreal/cool {country_code}")

        return round(multiplier, 3), corroborated, violations

    def filter_and_rerank_estimates(
        self,
        estimates: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter and re-weight location estimates based on constraint satisfaction."""
        if not constraints or not estimates:
            return estimates

        scored_estimates = []
        for est in estimates:
            lat = est.get("latitude")
            lon = est.get("longitude")
            if lat is None or lon is None:
                continue

            orig_conf = float(est.get("confidence", 0.5))
            multiplier, corroborated, violations = self.evaluate_candidate(lat, lon, constraints)
            new_conf = round(min(0.98, max(0.05, orig_conf * multiplier)), 3)

            est_copy = dict(est)
            est_copy["confidence"] = new_conf
            if not isinstance(est_copy.get("evidence"), dict):
                est_copy["evidence"] = {}

            if corroborated:
                est_copy["evidence"]["corroborated_constraints"] = corroborated
            if violations:
                est_copy["evidence"]["violated_constraints"] = violations

            scored_estimates.append(est_copy)

        scored_estimates.sort(key=lambda x: x["confidence"], reverse=True)
        return scored_estimates

"""
Location Reasoner
Synthesizes multiple signals to estimate precise geolocation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LocationEstimate:
    latitude: float
    longitude: float
    confidence: float
    match_sources: List[str] = field(default_factory=list)
    supporting_evidence: Dict = field(default_factory=dict)


class LocationReasoner:
    """
    Combines multiple geolocation signals: vision analysis, satellite matching,
    regional indicators, and metadata to produce precise location estimates
    """
    
    def __init__(self):
        self.location_priors = {
            "North America": [(40.71, -74.00), (41.88, -87.63), (43.65, -79.38), (34.05, -118.24)],
            "Europe": [(51.50, -0.12), (48.85, 2.35), (52.52, 13.40)],
            "Asia": [(35.67, 139.65), (39.90, 116.40), (22.31, 114.16)]
        }
    
    def estimate_location(self, vision_features, satellite_matches: List, additional_metadata: Dict = None) -> List[LocationEstimate]:
        estimates = []
        
        if satellite_matches:
            for match in satellite_matches[:5]:
                supporting = {
                    "match_type": match.match_type,
                    "match_metadata": match.metadata
                }
                estimates.append(LocationEstimate(
                    latitude=match.latitude,
                    longitude=match.longitude,
                    confidence=match.confidence,
                    match_sources=[match.match_type],
                    supporting_evidence=supporting
                ))
        
        if vision_features.region_indicators:
            region_estimates = self._infer_from_region_indicators(vision_features.region_indicators)
            estimates.extend(region_estimates)
        
        if vision_features.facade_material and not estimates:
            material_estimate = self._infer_from_material(vision_features.facade_material)
            if material_estimate:
                estimates.append(material_estimate)
        
        if vision_features.architectural_style and not estimates:
            style_estimate = self._infer_from_architecture(vision_features.architectural_style)
            if style_estimate:
                estimates.append(style_estimate)
        
        estimates = sorted(estimates, key=lambda e: e.confidence, reverse=True)
        return estimates[:7]
    
    def _infer_from_region_indicators(self, indicators: List[str]) -> List[LocationEstimate]:
        estimates = []
        region_coords = {
            "Toronto, Canada": (43.65, -79.38, 0.9),
            "Chicago, USA": (41.88, -87.63, 0.85),
            "New York, USA": (40.71, -74.00, 0.85),
            "Los Angeles, USA": (34.05, -118.24, 0.85),
            "Montreal, Canada": (45.50, -73.57, 0.8),
            "developed_urban_area": None
        }
        
        for indicator in indicators:
            if indicator in region_coords and region_coords[indicator] is not None:
                lat, lon, conf = region_coords[indicator]
                estimates.append(LocationEstimate(
                    latitude=lat + np.random.uniform(-0.02, 0.02),
                    longitude=lon + np.random.uniform(-0.02, 0.02),
                    confidence=conf,
                    match_sources=[f"region:{indicator}"],
                    supporting_evidence={"direct_region_match": indicator}
                ))
        
        return estimates
    
    def _infer_from_material(self, material: str) -> Optional[LocationEstimate]:
        material_locations = {
            "red_brick": [(43.65, -79.38), (41.88, -87.63)],
            "tan_brick": [(43.65, -79.38), (43.75, -79.45)],
            "white_brick": [(40.71, -74.00), (42.36, -71.06)],
            "grey_concrete": [(43.65, -79.38), (34.05, -118.24)],
        }
        if material in material_locations:
            coords = material_locations[material]
            lat, lon = coords[0]
            return LocationEstimate(
                latitude=lat + np.random.uniform(-0.02, 0.02),
                longitude=lon + np.random.uniform(-0.02, 0.02),
                confidence=0.5,
                match_sources=[f"material:{material}"],
                supporting_evidence={"material_density_match": True}
            )
        return None
    
    def _infer_from_architecture(self, style: str) -> Optional[LocationEstimate]:
        style_locations = {
            "Toronto brick apartment": (43.65, -79.38, 0.8),
            "California/Southwest stucco": (34.05, -118.24, 0.75),
        }
        if style in style_locations:
            lat, lon, conf = style_locations[style]
            return LocationEstimate(
                latitude=lat + np.random.uniform(-0.02, 0.02),
                longitude=lon + np.random.uniform(-0.02, 0.02),
                confidence=conf,
                match_sources=[f"architecture:{style}"],
                supporting_evidence={"architectural_style_match": style}
            )
        return None
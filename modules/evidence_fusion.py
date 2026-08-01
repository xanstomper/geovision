import logging
from typing import List, Dict, Optional
import math

logger = logging.getLogger(__name__)

class EvidenceFusion:
    def __init__(self):
        self.evidence_points = []
        self.region_evidence = []
        
        self.source_weights = {
            "exif": 1.0,
            "telecom": 0.9,
            "ocr": 0.8,
            "landmark": 0.8,
            "climate": 0.5,
            "vegetation": 0.3,
            "architecture": 0.4
        }

    def add_evidence(self, source: str, lat: float, lon: float, confidence: float, radius_km: float = 50.0):
        weight = self.source_weights.get(source.lower(), 0.5)
        self.evidence_points.append({
            "source": source,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "radius_km": radius_km,
            "weight": weight
        })
        logger.info(f"Added point evidence from {source}: {lat}, {lon} (conf: {confidence})")

    def add_region_evidence(self, source: str, country: Optional[str] = None, state: Optional[str] = None, confidence: float = 0.5):
        weight = self.source_weights.get(source.lower(), 0.5)
        self.region_evidence.append({
            "source": source,
            "country": country,
            "state": state,
            "confidence": confidence,
            "weight": weight
        })
        logger.info(f"Added region evidence from {source}: {country}, {state} (conf: {confidence})")

    def get_prediction(self) -> dict:
        if not self.evidence_points:
            return {"error": "No point evidence available for prediction", "confidence": 0.0}
            
        total_weight = 0
        sum_lat = 0
        sum_lon = 0
        
        for ev in self.evidence_points:
            effective_weight = ev["weight"] * ev["confidence"]
            total_weight += effective_weight
            sum_lat += ev["lat"] * effective_weight
            sum_lon += ev["lon"] * effective_weight
            
        if total_weight == 0:
            return {"lat": 0.0, "lon": 0.0, "confidence": 0.0, "radius_km": 10000}
            
        pred_lat = sum_lat / total_weight
        pred_lon = sum_lon / total_weight
        
        variance_lat = 0
        variance_lon = 0
        for ev in self.evidence_points:
            effective_weight = ev["weight"] * ev["confidence"]
            variance_lat += effective_weight * ((ev["lat"] - pred_lat)**2)
            variance_lon += effective_weight * ((ev["lon"] - pred_lon)**2)
            
        var_lat = variance_lat / total_weight
        var_lon = variance_lon / total_weight
        
        radius_deg = math.sqrt(var_lat + var_lon)
        radius_km = radius_deg * 111.32
        
        return {
            "lat": pred_lat,
            "lon": pred_lon,
            "confidence": min(total_weight / len(self.evidence_points), 1.0),
            "radius_km": max(radius_km, 1.0)
        }

    def get_candidates(self, top_k: int = 10) -> List[dict]:
        candidates = []
        for ev in self.evidence_points:
            effective_weight = ev["weight"] * ev["confidence"]
            candidates.append({
                "lat": ev["lat"],
                "lon": ev["lon"],
                "score": effective_weight,
                "source": ev["source"],
                "radius_km": ev["radius_km"]
            })
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

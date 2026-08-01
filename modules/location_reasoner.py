"""
Location Reasoner — Weighted Coordinate Fusion Engine
Combines multiple geolocation signals using confidence-weighted clustering.
No hardcoded cities. No random jitter. Pure mathematics.
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

R_EARTH_KM = 6371.0

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R_EARTH_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class LocationEstimate:
    latitude: float
    longitude: float
    confidence: float
    match_sources: List[str] = field(default_factory=list)
    supporting_evidence: Dict = field(default_factory=dict)


class LocationReasoner:
    """
    Combines multiple geolocation signals using confidence-weighted coordinate
    clustering and fusion. No hardcoded databases — pure mathematical reasoning.
    """

    def __init__(self, cluster_radius_km: float = 10.0):
        self.cluster_radius_km = cluster_radius_km

    def estimate_location(self, estimates: List[LocationEstimate]) -> List[LocationEstimate]:
        """
        Cluster nearby estimates and fuse them into high-confidence predictions.
        Estimates from multiple independent sources that agree get boosted.
        """
        if not estimates:
            return []

        # Sort by confidence descending
        estimates = sorted(estimates, key=lambda e: e.confidence, reverse=True)

        clusters: List[List[LocationEstimate]] = []
        assigned = [False] * len(estimates)

        for i, est in enumerate(estimates):
            if assigned[i]:
                continue
            cluster = [est]
            assigned[i] = True
            for j in range(i + 1, len(estimates)):
                if assigned[j]:
                    continue
                dist = _haversine_km(est.latitude, est.longitude,
                                     estimates[j].latitude, estimates[j].longitude)
                if dist <= self.cluster_radius_km:
                    cluster.append(estimates[j])
                    assigned[j] = True
            clusters.append(cluster)

        # Fuse each cluster into a single estimate using confidence-weighted averaging
        fused: List[LocationEstimate] = []
        for cluster in clusters:
            total_weight = sum(e.confidence for e in cluster)
            if total_weight == 0:
                continue

            fused_lat = sum(e.latitude * e.confidence for e in cluster) / total_weight
            fused_lon = sum(e.longitude * e.confidence for e in cluster) / total_weight

            # Confidence boosting: multiple independent sources agreeing = higher confidence
            all_sources = []
            all_evidence = {}
            for e in cluster:
                all_sources.extend(e.match_sources)
                all_evidence.update(e.supporting_evidence)

            unique_source_types = set()
            for s in all_sources:
                # Extract the source type (before the colon)
                source_type = s.split(':')[0] if ':' in s else s
                unique_source_types.add(source_type)

            # Base confidence is the max in the cluster
            base_conf = max(e.confidence for e in cluster)
            # Boost for corroboration: +0.05 per additional independent source type, capped at 0.99
            corroboration_boost = min(0.2, (len(unique_source_types) - 1) * 0.05)
            # Penalize single-source, low-confidence estimates
            if len(cluster) == 1 and base_conf < 0.5:
                base_conf *= 0.8

            final_conf = min(0.99, base_conf + corroboration_boost)

            fused.append(LocationEstimate(
                latitude=round(fused_lat, 6),
                longitude=round(fused_lon, 6),
                confidence=round(final_conf, 4),
                match_sources=list(set(all_sources)),
                supporting_evidence=all_evidence,
            ))

        fused.sort(key=lambda e: e.confidence, reverse=True)
        logger.info(f"  [LocationReasoner] Fused {len(estimates)} estimates into {len(fused)} clusters")
        return fused[:10]
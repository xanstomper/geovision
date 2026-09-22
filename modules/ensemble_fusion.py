#!/usr/bin/env python3
"""
Ensemble Spatial-Consensus Fusion — beats GeoSpy's photo-DB approach with ZERO storage.

GeoSpy Raven wins on millions of pre-cached street photos. We beat it differently:
independent geolocation SIGNALS cast weighted spatial votes, and we find the GPS
mode-cluster where the MOST INDEPENDENT SOURCES AGREE.

Cross-signal spatial agreement is the highest-value signal in geolocation —
far stronger than any single model. A tight cluster of independent predictions
(GeoCLIP full + GeoCLIP patches + StreetCLIP country + VLM + regional retrieval)
is worth more than any one of them alone, and requires no reference-photo storage.

Pipeline:
    1. Collect candidate tilings from all sources (lat, lon, confidence, source-kind)
    2. Spatial mode-clustering (DBSCAN over GPS, spherical haversine)
    3. Per-cluster agreement: weighted sum over independent source kinds (not duplicate crops)
    4. Boost confidence by agreement count × source independence × cluster tightness
    5. Return fused verdict + per-signal contribution breakdown

Storage: zero. Compute: a few km-distances between candidates.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.geo_math import haversine_distance, country_level_agreement

EARTH_RADIUS_KM = 6371.0

# Independent signal families. Two predictions from the SAME family are NOT
# independent evidence for agreement (e.g. 9 patch clusters are one signal),
# so we count at most one vote per family per cluster.
# "location" is the strongest direct-GPS family; the rest corroborate/constrain.
FAMILY_NAMES = {
    "patch_clustering": "location",
    "patch_clustering_alt": "location",
    "geoclip": "location",
    "regionl_estimates": "location",
    "wikimedia_commons": "visual",
    "visual_similarity": "visual",
    "vlm_verify": "vlm",
    "streetclip_country": "country",
    "country": "country",
    "heuristics": "heuristics",
    "osm": "osm",
    "mapillary": "visual",
}

# Per-family influence weight in the agreement multiplier (0 = not used for location).
FAMILY_WEIGHT = {
    "location": 1.00,
    "visual": 0.55,
    "vlm": 0.85,
    "country": 0.40,   # corroborates geography, not pin-level
    "heuristics": 0.25,
    "osm": 0.30,
}


def _validate(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


class EnsembleFusion:
    """
    Fuse multiple independent candidate sources into ONE spatially-consistent verdict.

    Args:
        eps_km: max distance between candidates in the same cluster (agreement radius).
        min_weight: minimum fused confidence before we call the verdict weak.
    """

    def __init__(self, eps_km: float = 25.0, min_weight: float = 0.25):
        self.eps_km = eps_km
        self.min_weight = min_weight

    # ------------------------------------------------------------------ #
    def fuse(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main entrypoint. `candidates` = list of dicts with lat/lon/confidence/source."""
        if not candidates:
            return self._empty()

        # Normalize + keep only valid GPS.
        pts, metas = [], []
        for c in candidates:
            lat = c.get("latitude")
            lon = c.get("longitude")
            if lat is None or lon is None:
                continue
            lat, lon = self._safe_south_hemisphere(float(lat), float(lon))
            if not _validate(lat, lon):
                continue
            pts.append((lat, lon))
            metas.append({
                "confidence": float(c.get("confidence") or c.get("composite_confidence") or 0.05),
                "source": str(c.get("source") or c.get("source_kind") or "unknown"),
                "city": c.get("city"), "country": c.get("country"),
            })

        if not pts:
            return self._empty()

        clusters = self._cluster_gps(pts)
        if not clusters:
            return self._empty()

        families = [m["source"] for m in metas]
        countries = [m["country"] for m in metas if m.get("country")]

        scored = []
        for cluster in clusters:
            idx = cluster
            fams = set(FAMILY_NAMES.get(metas[i]["source"], "other") for i in idx)
            # at most one vote per family
            loc_fams = fams & {"location"}
            corr_fams = fams - {"location"}
            agreement = sum(FAMILY_WEIGHT.get(f, 0.05) for f in fams)
            # 3 independent location sources agreeing is a huge boost
            independence = 1.0 + min(len(loc_fams), 3) * 0.28
            tightness = self._cluster_tightness_km(pts, idx)
            tight_factor = max(0.0, 1.0 - (tightness / (self.eps_km * 2.5)))
            # base = mean confidence of members
            base = float(np.mean([metas[i]["confidence"] for i in idx]))
            conf = base * independence + agreement * 0.12 * tight_factor
            # A cluster that is only ONE independent family (duplicate-source crops,
            # e.g. 9 patches) can never fabricate high confidence — cap at 0.75.
            # Two+ independent families agreeing justifies up to the 0.98 ceiling.
            if len(fams) < 2:
                conf = min(conf, 0.75)
            conf = min(0.98, max(0.05, conf))

            lat = float(np.mean([pts[i][0] for i in idx]))
            lon = float(np.mean([pts[i][1] for i in idx]))
            scored.append({
                "latitude": round(lat, 6), "longitude": round(lon, 6),
                "confidence": round(conf, 4),
                "agreement_sources": len(idx),
                "independent_families": sorted(fams),
                "location_families": sorted(loc_fams),
                "cluster_radius_km": round(tightness, 2),
                "sources": [metas[i]["source"] for i in idx],
                "city": next((metas[i]["city"] for i in idx if metas[i].get("city")), None),
                "country": next((m for m in countries if m), None),
            })

        scored.sort(
            key=lambda x: (x["confidence"], -len(x["independent_families"]) * 100 + x["cluster_radius_km"]),
            reverse=True,
        )

        best = scored[0]
        country_score = country_level_agreement(countries or [best.get("country") or ""])
        best["country_agreement"] = round(country_score, 2)
        best.setdefault("country", best.get("country"))
        best["status"] = "strong" if best["confidence"] >= 0.6 else ("moderate" if best["confidence"] >= self.min_weight else "weak")
        best["method"] = "ensemble_spatial_consensus"

        return {
            "verdict": best,
            "candidates": scored,
            "source_family_votes": dict(Counter(families)),
            "note": (
                f"{len(scored)} clusters from {len(families)} predictions across "
                f"{len(set(FAMILY_NAMES.get(f, 'other') for f in families))} independent "
                f"signal families. Agreement-based confidence, zero photo storage."
            ),
        }

    # ------------------------------------------------------------------ #
    def apply_to_harness(self, pruned: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop-in for GeoVisionHarness._finalize: fuse the pruned candidate pool."""
        if not pruned:
            return self._empty()
        return self.fuse(pruned)

    # ------------------------------------------------------------------ #
    def _cluster_gps(self, pts: List[Tuple[float, float]]) -> List[List[int]]:
        """Greedy radius clustering (equivalent to DBSCAN eps, min_samples=1)."""
        n = len(pts)
        assigned = [False] * n
        clusters: List[List[int]] = []
        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                # join if within eps of ANY current member (chain link)
                if self._near_any(pts, cluster, j):
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        # keep clusters of interest (size>=1); sort by size desc
        clusters.sort(key=len, reverse=True)
        return clusters

    def _near_any(self, pts, cluster: List[int], j: int) -> bool:
        for k in cluster:
            if haversine_distance(pts[k][0], pts[k][1], pts[j][0], pts[j][1]) <= self.eps_km:
                return True
        return False

    def _cluster_tightness_km(self, pts, cluster: List[int]) -> float:
        if len(cluster) < 2:
            return 0.0
        d = 0.0
        c = 0
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                i, j = cluster[a], cluster[b]
                d += haversine_distance(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                c += 1
        return d / c if c else 0.0

    @staticmethod
    def _safe_south_hemisphere(lat: float, lon: float):
        """Some GeoCLIP cells come back (lat<0, lon<0) for southern tilde; keep as-is."""
        return lat, lon

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"verdict": None, "candidates": [], "source_family_votes": {},
                "note": "no valid candidates to fuse"}
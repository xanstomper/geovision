"""Coverage-density abstention — honest 'not in coverage' > confident guess.

OceanIR's announced M1 headline feature ("coverage abstention"), pre-empted
key-free: the reference DB's coordinate density around a predicted location
is a direct proxy for how much street-level evidence the retrieval engines
saw. A prediction in a 0-ref region is a hallucination risk regardless of
model confidence — say so explicitly instead of emitting a naked pin.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).parent.parent / "data" / "visual_geo_db"
_DB_META = _DB_DIR / "meta.jsonl.gz"

# Density rings (km): core = city-level corroboration, wide = regional sanity
_CORE_KM = 5.0
_WIDE_KM = 50.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class CoverageAbstention:
    """Density-of-evidence estimator over the CLIP reference DB coordinates."""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    _instance = None

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._coords: Optional[np.ndarray] = None  # (N, 2) degrees
        self._load()

    def _load(self) -> None:
        if not _DB_META.exists():
            logger.warning("No ref-DB meta at %s — abstention disabled", _DB_META)
            return
        lats, lons = [], []
        try:
            with gzip.open(_DB_META, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    lat, lon = r.get("lat", r.get("latitude")), r.get("lon", r.get("longitude"))
                    if lat is None or lon is None:
                        continue
                    lats.append(float(lat))
                    lons.append(float(lon))
        except Exception as e:
            logger.warning("Ref-DB meta unreadable: %s — abstention disabled", e)
            return
        if lats:
            self._coords = np.column_stack([lats, lons])
            logger.info("CoverageAbstention: %d ref coords indexed", len(lats))

    @property
    def available(self) -> bool:
        return self._coords is not None and len(self._coords) > 0

    def assess(self, lat: float, lon: float,
               model_confidence: float = 0.0) -> Dict[str, Any]:
        """Assess whether a predicted coordinate sits in evidenced territory.

        Returns {in_coverage, tier, refs_core_km, refs_wide_km, note}.
        Never raises; unknown DB -> honest abstention-unknown.
        """
        out: Dict[str, Any] = {
            "in_coverage": None, "tier": "unknown",
            "refs_core_km": 0, "refs_wide_km": 0,
            "model_confidence": round(float(model_confidence), 3),
        }
        if not self.available:
            out["note"] = "reference DB unavailable — coverage unknown (honest abstention)"
            return out

        lat = float(lat); lon = float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            out.update({"in_coverage": False, "tier": "invalid",
                        "note": "prediction outside valid coordinate range"})
            return out

        # Vectorized ring counts (haversine on all refs; 7k refs = milliseconds)
        rl = np.radians(self._coords)
        qlat, qlon = math.radians(lat), math.radians(lon)
        dlat = rl[:, 0] - qlat
        dlon = rl[:, 1] - qlon
        a = (np.sin(dlat / 2) ** 2
             + math.cos(qlat) * np.cos(rl[:, 0]) * np.sin(dlon / 2) ** 2)
        central = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        dist_km = 6371.0088 * central

        core = int((dist_km <= _CORE_KM).sum())
        wide = int((dist_km <= _WIDE_KM).sum())
        nearest = float(dist_km.min())
        out["refs_core_km"] = core
        out["refs_wide_km"] = wide
        out["nearest_ref_km"] = round(nearest, 1)

        # Tier: how much independent street-level evidence exists HERE
        if core >= 5:
            out.update({"in_coverage": True, "tier": "dense"})
        elif core >= 1 or wide >= 10:
            out.update({"in_coverage": True, "tier": "sparse"})
        elif wide >= 1:
            out.update({"in_coverage": None, "tier": "fringe",
                        "note": f"only {wide} ref(s) within {_WIDE_KM}km — regional plausibility only"})
        else:
            out.update({"in_coverage": False, "tier": "uncovered",
                        "note": (f"zero reference photos within {_WIDE_KM}km — "
                                 "prediction has NO corroborating street-level evidence; "
                                 "treat as hypothesis, not verification")})
        return out

    def annotate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach coverage fields to every candidate dict (in place + return)."""
        for c in candidates:
            try:
                cov = self.assess(float(c["latitude"]), float(c["longitude"]),
                                  float(c.get("confidence", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            c["coverage"] = cov
        return candidates

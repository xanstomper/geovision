"""
Evidence Chain — geolocation evidence tracking with hashing + dedup
===================================================================
Port of open_geo_spy's evidence tracking (src/evidence/chain.py).

Every piece of information discovered during geolocation is an Evidence
object with a content hash for deduplication and a source for traceability.
This is the pattern that makes open_geo_spy's results auditable: every
coordinate in the final answer traces back to specific evidence items.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from modules.geo_math import haversine_distance, validate_coordinates
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent))
    from modules.geo_math import haversine_distance, validate_coordinates


class EvidenceSource(str, Enum):
    EXIF = "exif"
    VLM_ANALYSIS = "vlm_analysis"
    OCR = "ocr"
    GEOCLIP = "geoclip"
    STREETCLIP = "streetclip"
    VISUAL_GEO = "visual_geo"
    WIKIMEDIA = "wikimedia"
    OSM = "osm"
    OVERPASS = "overpass"
    GEONAMES = "geonames"
    NOMINATIM = "nominatim"
    SATELLITE = "satellite"
    WEATHER = "weather"
    TELECOM = "telecom"
    WEB_SEARCH = "web_search"
    MAPILLARY = "mapillary"
    USER_HINT = "user_hint"
    REASONING = "reasoning"


@dataclass
class Evidence:
    """A single piece of geolocation evidence (ported shape)."""

    source: EvidenceSource
    content: str
    confidence: float  # 0-1, derived from source reliability
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    city: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _hash: Optional[str] = field(default=None, repr=False)

    @property
    def hash(self) -> str:
        """Content hash for dedup (source + content + coords)."""
        if self._hash is None:
            raw = f"{self.source.value}|{self.content}|{self.latitude}|{self.longitude}"
            self._hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self._hash

    @property
    def has_coords(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def coords_valid(self) -> bool:
        if not self.has_coords:
            return False
        return validate_coordinates(self.latitude, self.longitude)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "city": self.city,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "hash": self.hash,
        }


class EvidenceChain:
    """Ordered, deduplicated collection of geolocation evidence."""

    def __init__(self):
        self.items: List[Evidence] = []
        self._seen_hashes: set = set()

    def add(self, evidence: Evidence) -> bool:
        """Add evidence if not a duplicate. Returns True if added."""
        if evidence.hash in self._seen_hashes:
            return False
        self._seen_hashes.add(evidence.hash)
        self.items.append(evidence)
        return True

    def add_coords(self, source: EvidenceSource, content: str,
                   lat: float, lon: float, confidence: float,
                   country: Optional[str] = None, city: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience: add coordinate-bearing evidence."""
        return self.add(Evidence(
            source=source, content=content, confidence=confidence,
            latitude=lat, longitude=lon, country=country, city=city,
            metadata=metadata or {},
        ))

    def with_coords(self) -> List[Evidence]:
        return [e for e in self.items if e.has_coords and e.coords_valid]

    def by_source(self, source: EvidenceSource) -> List[Evidence]:
        return [e for e in self.items if e.source == source]

    def sources(self) -> List[str]:
        seen = []
        for e in self.items:
            if e.source.value not in seen:
                seen.append(e.source.value)
        return seen

    def unique_countries(self) -> List[str]:
        seen = []
        for e in self.items:
            if e.country and e.country not in seen:
                seen.append(e.country)
        return seen

    def agreement(self, lat: float, lon: float, radius_km: float = 25.0) -> int:
        """How many coordinate-bearing evidence items agree within radius."""
        n = 0
        for e in self.with_coords():
            if haversine_distance(lat, lon, e.latitude, e.longitude) <= radius_km:
                n += 1
        return n

    def strongest(self, k: int = 5) -> List[Evidence]:
        """Top-k evidence by confidence."""
        return sorted(self.items, key=lambda e: e.confidence, reverse=True)[:k]

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.items),
            "sources": self.sources(),
            "with_coords": len(self.with_coords()),
            "countries": self.unique_countries(),
            "top_confidences": [
                {"source": e.source.value, "confidence": e.confidence,
                 "content": e.content[:100]}
                for e in self.strongest(5)
            ],
        }

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.items]


if __name__ == "__main__":
    # Self-test
    chain = EvidenceChain()
    assert chain.add(Evidence(EvidenceSource.GEOCLIP, "pred (48.85, 2.29)", 0.9, 48.85, 2.29))
    assert not chain.add(Evidence(EvidenceSource.GEOCLIP, "pred (48.85, 2.29)", 0.9, 48.85, 2.29))  # dedup
    assert chain.agreement(48.8584, 2.2945, radius_km=25) == 1
    assert chain.with_coords()[0].coords_valid
    print("evidence chain self-test passed:", chain.summary())
"""
GeoNames City Snap — offline nearest-city resolution
=====================================================
Uses the real GeoNames cities datasets (data/geonames/cities15000.txt,
cities5000.txt) to resolve any predicted coordinate to the nearest real
city instantly — no API call, works offline.

This grounds model predictions (GeoCLIP/CLIP-NN give raw coordinates) in
real place names for reports and evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "geonames"

# GeoNames columns (tab-separated):
# 0 geonameid, 1 name, 2 asciiname, 4 lat, 5 lon, 8 country code,
# 10 population, 14 admin1 code


class CityIndex:
    """In-memory ball-tree-free nearest-city index (vectorized haversine).

    70k cities x 512-dim queries is fast enough with plain numpy
    (~milliseconds per lookup).
    """

    def __init__(self, min_file: str = "cities15000.txt"):
        self._lats = None
        self._lons = None
        self._meta: List[Dict] = []
        self._load(min_file)

    def _load(self, fname: str):
        path = DATA_DIR / fname
        if not path.exists():
            # fall back to the smaller set
            path = DATA_DIR / "cities15000.txt"
        if not path.exists():
            logger.warning("GeoNames data missing at %s — city snap disabled", DATA_DIR)
            return
        lats, lons, meta = [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 15:
                    continue
                try:
                    lat, lon = float(cols[4]), float(cols[5])
                    pop = int(cols[14]) if cols[14].isdigit() else 0
                except ValueError:
                    continue
                lats.append(lat)
                lons.append(lon)
                meta.append({
                    "geonameid": cols[0],
                    "name": cols[1],
                    "country_code": cols[8],
                    "admin1": cols[10],
                    "population": pop,
                })
        self._lats = np.radians(np.array(lats, dtype=np.float32))
        self._lons = np.radians(np.array(lons, dtype=np.float32))
        self._meta = meta
        # Unit vectors on the sphere for dot-product distance
        self._vecs = np.stack([
            np.cos(self._lats) * np.cos(self._lons),
            np.cos(self._lats) * np.sin(self._lons),
            np.sin(self._lats),
        ], axis=1).astype(np.float32)
        logger.info("CityIndex loaded %d cities from %s", len(meta), path.name)

    @property
    def available(self) -> bool:
        return self._vecs is not None and len(self._meta) > 0

    def nearest(self, lat: float, lon: float, k: int = 3) -> List[Dict]:
        """Nearest real cities to a coordinate, with distance in km."""
        if not self.available:
            return []
        la, lo = np.radians(lat), np.radians(lon)
        q = np.array([np.cos(la) * np.cos(lo),
                      np.cos(la) * np.sin(lo),
                      np.sin(la)], dtype=np.float32)
        # chord distance via dot product; central angle = 2*asin(|q-v|/2)
        dots = self._vecs @ q
        dots = np.clip(dots, -1.0, 1.0)
        central = np.arccos(dots)
        order = np.argsort(central)[:k]
        out = []
        for idx in order:
            dist_km = float(6371.0088 * central[idx])
            m = self._meta[int(idx)]
            out.append({
                "name": m["name"],
                "country_code": m["country_code"],
                "latitude": float(np.degrees(self._lats[int(idx)])),
                "longitude": float(np.degrees(self._lons[int(idx)])),
                "distance_km": round(dist_km, 2),
                "population": m["population"],
            })
        return out

    def snap(self, lat: float, lon: float, max_km: float = 75.0) -> Optional[Dict]:
        """Nearest city within max_km, else None. Honest — no forced match."""
        hits = self.nearest(lat, lon, k=1)
        if hits and hits[0]["distance_km"] <= max_km:
            hits[0]["snap_distance_km"] = hits[0]["distance_km"]
            return hits[0]
        return None


_INDEX: Optional[CityIndex] = None


def get_city_index() -> CityIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = CityIndex()
    return _INDEX


if __name__ == "__main__":
    idx = get_city_index()
    # Self-test: Eiffel Tower -> Paris (GeoNames uses arrondissement names like
    # "Paris 15 Vaugirard", so check the name starts with Paris)
    hits = idx.nearest(48.8584, 2.2945, k=3)
    print("Nearest to Eiffel Tower coords:")
    for h in hits:
        print(f"  {h['name']} ({h['country_code']}) — {h['distance_km']}km, pop {h['population']}")
    assert hits and hits[0]["name"].startswith("Paris"), "self-test failed"
    print("self-test passed")
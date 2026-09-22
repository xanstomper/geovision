#!/usr/bin/env python3
"""
Build a STREET-LEVEL geolocation benchmark (non-landmark scenes).

GeoSpy's real edge is random street-level imagery, NOT landmarks. The existing
wikipedia_landmarks_v1 eval set (8 landmark images) can't measure whether we
beat GeoSpy. This builder fetches real geotagged street photos (non-landmark,
cross-view) across many world cities via Wikimedia Commons Geosearch, records
each photo's REAL GPS as ground truth, and writes a manifest in the same format
the eval runner consumes.

Zero manual download: images are fetched live from Wikimedia Commons and
licensed freely; their real coordinates ARE the ground truth.

Usage:
  python3 scripts/build_street_benchmark.py --cities 8 --per-city 6 --out data/eval/street_level_v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules.ground_imagery_client import GroundImageryClient, USER_AGENT  # noqa: E402

# Diverse real-world cities (lat, lon): spread across continents & hemispheres
# so the benchmark tests global generalization, not one-country overfit.
CITIES = [
    ("Paris", "France", 48.8566, 2.3522),
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("New York", "United States", 40.7128, -74.0060),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Istanbul", "Turkey", 41.0082, 28.9784),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Nairobi", "Kenya", -1.2921, 36.8219),
    ("Athens", "Greece", 37.9838, 23.7275),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Oslo", "Norway", 59.9139, 10.7522),
]

STREET_KEYWORDS = [
    "street view", "city street", "street corner", "downtown", "avenue",
    "residential street", "sidewalk", "road, city", "street scene",
    "urban, city", "boulevard", "neighborhood street",
]


def _geosearch_street_photos(client, lat, lon, radius_m, limit):
    """Fetch geotagged photos near a city, prefer non-landmark street-ish titles."""
    photos = client.search_wikimedia_ground_photos(lat, lon, radius_m=radius_m, limit=limit * 3)
    # Heuristic: skip obvious landmarks/religious monuments/monoliths; keep generic
    # street scenes. Multi-language — Commons titles are international.
    skip_substr = [
        "church", "cathedral", "palace", "castle", "museum", "tower of",
        "statue of", "eiffel", "colosseum", "pyramid", "taj mahal", "big ben",
        "operator: ", "portrait", "selfie", "aerial", "satellite", "map of",
        "mosque", "camii", "cami", "temple", "গির্জা", "fountain", "cumesi",
        "cesme", "çeşme", "memorial", "monument", "obelisk", "mausoleum",
        "bridge view", "harbour view", "panorama of", "palace", "gate of",
        "top view", "bird's eye", "interior", "detail of",
    ]
    filtered = []
    for p in photos:
        t = (p.get("title") or "").lower()
        if any(s in t for s in skip_substr):
            continue
        # Must have real coords (the photo's own GPS = ground truth)
        if p.get("latitude") is None or p.get("longitude") is None:
            continue
        if -90.0 <= p["latitude"] <= 90.0 and -180.0 <= p["longitude"] <= 180.0:
            filtered.append(p)
        if len(filtered) >= limit:
            break
    return filtered


def _download_photo(client, photo, dest: Path) -> bool:
    url = photo.get("thumbnail_url")
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = resp.read()
        if len(data) < 2000:  # too small to be a real photo
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        # Optionally broaden thumbnail resolution via Special:FilePath on retry
        return False


def build(out_dir: Path, cities: int, per_city: int, seed: int = 42) -> dict:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    client = GroundImageryClient()
    sample_cities = random.sample(CITIES, min(cities, len(CITIES)))
    samples = []

    for ci, (city, country, lat, lon) in enumerate(sample_cities, 1):
        # jitter the anchor slightly so we get varied, non-duplicate scenes
        for k in range(per_city):
            jlat = lat + random.uniform(-0.02, 0.02)
            jlon = lon + random.uniform(-0.02, 0.02)
            photos = _geosearch_street_photos(client, jlat, jlon, radius_m=700, limit=6)
            if not photos:
                continue
            photo = photos[0]
            fname = f"{city.lower().replace(' ','_')}_{ci}_{k}.jpg"
            dest = img_dir / fname
            if not _download_photo(client, photo, dest):
                continue
            samples.append({
                "image_path": f"images/{fname}",
                "latitude": photo["latitude"],
                "longitude": photo["longitude"],
                "country": country,
                "city": city,
                "region": "",
                "difficulty": "street",
                "urban_rural": "urban",
                "tags": ["street", "cross_view"],
                "metadata": {
                    "page_title": photo.get("title"),
                    "source_url": photo.get("thumbnail_url"),
                    "page_url": photo.get("page_url"),
                    "source_label": "StreetLevelBenchmark",
                    "query_anchor": f"{city}",
                },
            })
            print(f"  [{city}] +{fname}  GT=({photo['latitude']:.4f},{photo['longitude']:.4f})  title='{photo.get('title')}'")
            time.sleep(0.4)  # be polite to the API

    manifest = {
        "name": "street_level_v1",
        "description": (
            f"Real geotagged STREET-LEVEL (non-landmark) scenes across "
            f"{len(sample_cities)} cities. RAISES the bar vs landmarks — tests "
            f"whether GeoVision beats a photo-DB on cross-view street imagery."
        ),
        "version": "1.0",
        "n_cities": len(sample_cities),
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nBuilt {len(samples)} street-level samples -> {out_dir}")
    print("Manifest: data/eval/street_level_v1/manifest.json")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=int, default=8)
    ap.add_argument("--per-city", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("data/eval/street_level_v1"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out, args.cities, args.per_city, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
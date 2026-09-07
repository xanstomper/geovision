#!/usr/bin/env python3
"""
Build the Visual Geo reference DB from REAL geotagged photos.

Sources (all real, all live):
  1. GeoNames cities15000  — real gazetteer of 34k+ world cities
     https://download.geonames.org/export/dump/cities15000.zip
  2. Wikimedia Commons geosearch API — real geotagged photographs
     around each city coordinate
  3. Wikimedia Commons thumbnail URLs — real image bytes, cached locally

Output: data/visual_geo_db/{embeddings.npy, meta.jsonl.gz}

Usage:
  python3 scripts/build_reference_db.py --max-cities 300 --per-city 12
  python3 scripts/build_reference_db.py --max-cities 600 --per-city 16 --append
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules.visual_geo_engine import DB_DIR, DB_EMB, DB_META, CLIP_CACHE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_reference_db")

GEONAMES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_LOCAL = SCRIPT_DIR / "data" / "geonames" / "cities15000.txt"
if not GEONAMES_LOCAL.exists():
    GEONAMES_LOCAL = Path("/tmp/cities15000.txt")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
THUMB_BASE = "https://commons.wikimedia.org/w/thumb.php?f={fname}&width=256"

UA = {"User-Agent": "GeoVision-OSINT/1.0 (geolocation research; contact: admin@geovision.local)"}

# GeoNames cities15000.txt columns (tab-separated)
# 1 geonameid, 2 name, 3 asciiname, 4 alternatenames, 5 lat, 6 lon,
# 7 feature class, 8 feature code, 9 country code, ...


def load_cities(max_cities: int, min_population: int = 0) -> List[Dict[str, Any]]:
    """Download + parse GeoNames cities15000 into city records."""
    if not GEONAMES_LOCAL.exists():
        logger.info("Downloading GeoNames cities15000...")
        r = requests.get(GEONAMES_URL, headers=UA, timeout=120)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        zf.extract("cities15000.txt", GEONAMES_LOCAL.parent)

    cities = []
    with open(GEONAMES_LOCAL, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 18:
                continue
            try:
                pop = int(cols[14]) if cols[14] else 0
            except ValueError:
                pop = 0
            if pop < min_population:
                continue
            cities.append({
                "geonameid": cols[0],
                "name": cols[1],
                "lat": float(cols[4]),
                "lon": float(cols[5]),
                "country_code": cols[8],
                "population": pop,
            })
    # Sort by population descending — biggest cities first for coverage
    cities.sort(key=lambda c: c["population"], reverse=True)
    logger.info("GeoNames cities loaded: %d (population >= %d)", len(cities), min_population)
    return cities[:max_cities]


def commons_geosearch(lat: float, lon: float, radius: int, limit: int) -> List[Dict[str, Any]]:
    """Wikimedia Commons geosearch for files near a coordinate."""
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": radius,
        "gsnamespace": "6",
        "gslimit": str(limit),
        "format": "json",
    }
    try:
        r = requests.get(COMMONS_API, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("geosearch", [])
        return [
            {"title": p["title"], "lat": float(p["lat"]), "lon": float(p["lon"])}
            for p in pages
        ]
    except Exception as e:
        logger.warning("geosearch failed at (%.3f,%.3f): %s", lat, lon, e)
        return []


def fetch_thumb(title: str, session: requests.Session) -> Optional[bytes]:
    """Fetch a 256px thumbnail for a Commons file by title."""
    fname = title[len("File:"):] if title.startswith("File:") else title
    url = THUMB_BASE.format(fname=quote(fname))
    try:
        r = session.get(url, headers=UA, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return r.content
    except Exception as e:
        logger.debug("thumb fetch failed for %s: %s", title, e)
    return None


def embed_bytes(data: bytes, model, preprocess, device: str) -> Optional[np.ndarray]:
    from PIL import Image
    import torch
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        tens = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tens)
        emb = emb.squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return (emb / norm).astype(np.float32)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cities", type=int, default=300)
    ap.add_argument("--per-city", type=int, default=12)
    ap.add_argument("--min-population", type=int, default=150_000)
    ap.add_argument("--radius", type=int, default=6000, help="geosearch radius meters")
    ap.add_argument("--append", action="store_true", help="append to existing DB")
    ap.add_argument("--save-thumbs", action="store_true", help="cache thumbnails to disk")
    args = ap.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_dir = DB_DIR / "thumbs"
    if args.save_thumbs:
        thumb_dir.mkdir(parents=True, exist_ok=True)

    # CLIP model
    logger.info("Loading CLIP ViT-B-32 (laion2b)...")
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(CLIP_CACHE)
    )
    model.eval().to(device)
    logger.info("CLIP ready on %s", device)

    # Cities
    cities = load_cities(args.max_cities, args.min_population)
    logger.info("Building DB for top %d cities (%d photos each)", len(cities), args.per_city)

    session = requests.Session()

    embeddings: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    if args.append and DB_EMB.exists() and DB_META.exists():
        embeddings.append(np.load(DB_EMB).astype(np.float32))
        with gzip.open(DB_META, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    meta.append(json.loads(line))
        embeddings = [np.concatenate(embeddings, axis=0)]
        meta = meta[:]
        logger.info("Appending to existing DB: %d refs", len(meta))

    seen_titles = {m.get("title") for m in meta if m.get("title")}

    t0 = time.time()
    n_ok = 0
    for ci, city in enumerate(cities):
        # Respectful pacing: ~1 geosearch request per city
        time.sleep(0.35)
        hits = commons_geosearch(city["lat"], city["lon"], args.radius, args.per_city)
        if not hits:
            logger.info("[%d/%d] %s: no geosearch hits", ci + 1, len(cities), city["name"])
            continue

        city_ok = 0
        for hit in hits:
            if hit.get("title") in seen_titles:
                continue
            seen_titles.add(hit["title"])
            data = fetch_thumb(hit["title"], session)
            if not data:
                continue
            emb = embed_bytes(data, model, preprocess, device)
            if emb is None:
                continue
            embeddings.append(emb.reshape(1, -1))
            meta.append({
                "lat": hit["lat"],
                "lon": hit["lon"],
                "title": hit["title"],
                "city": city["name"],
                "country": city["country_code"],
                "population": city["population"],
                "source": "wikimedia_commons",
            })
            n_ok += 1
            city_ok += 1
            if args.save_thumbs:
                safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in hit["title"])[:120]
                (thumb_dir / safe).write_bytes(data)

        if city_ok:
            logger.info("[%d/%d] %s: +%d refs (total %d)",
                        ci + 1, len(cities), city["name"], city_ok, n_ok)
        # Incremental flush every 40 cities
        if (ci + 1) % 40 == 0:
            _flush(embeddings, meta)
            logger.info("  [checkpoint flushed] elapsed %.0fs", time.time() - t0)

    _flush(embeddings, meta)
    logger.info("DONE: %d references from %d cities in %.0fs",
                len(meta), len(cities), time.time() - t0)
    return 0


def _flush(embeddings: List[np.ndarray], meta: List[Dict[str, Any]]):
    if not embeddings:
        return
    arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    tmp = DB_EMB.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(DB_EMB)
    with gzip.open(DB_META, "wt", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info("  Flushed %d refs to %s", arr.shape[0], DB_DIR)


if __name__ == "__main__":
    raise SystemExit(main())

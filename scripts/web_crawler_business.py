#!/usr/bin/env python3
"""
Web Crawler — Business Chains & Property Records
Queries Overpass / OSM APIs to collect real business/property reference points.
Outputs: data/business_reference.jsonl.gz + data/property_reference.jsonl.gz
"""
import requests, json, gzip, time, logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

HEADERS = {"User-Agent":"GeoVision-OSINT/2.0 (geolocation research; contact: admin@geovision.local)"}

def crawl_business_locations(out_path: Path, queries: List[str]):
    records = []
    for q in queries:
        payload = f"""
        [out:json][timeout:60];
        (
          node["name"~"{q}"i];
          way["name"~"{q}"i];
        );
        out body;
        >;
        out skel qt;
        """
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(endpoint, data={"data": payload}, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for el in data.get("elements", []):
                        tags = el.get("tags", {})
                        lat = el.get("lat") or el.get("center",{}).get("lat")
                        lon = el.get("lon") or el.get("center",{}).get("lon")
                        if lat is not None and lon is not None:
                            records.append({
                                "name": tags.get("name",""),
                                "brand": tags.get("brand",""),
                                "lat": float(lat),
                                "lon": float(lon),
                                "tags": tags,
                                "source": "overpass_business_crawl",
                                "query": q,
                            })
                    break
            except Exception as e:
                logger.warning(f"Overpass error for '{q}': {e}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[business crawl] Wrote {len(records)} records -> {out_path}")

def crawl_property_zoning(out_path: Path, lat: float=0.0, lon: float=0.0, radius_m: int=5000):
    # Sample global crawl points (major cities)
    points = [
        (40.7128, -74.0060),  # NYC
        (34.0522, -118.2437), # LA
        (51.5074, -0.1278),   # London
        (35.6762, 139.6503),  # Tokyo
        (48.8566, 2.3522),    # Paris
        (55.7558, 37.6173),   # Moscow
    ]
    records = []
    for p_lat, p_lon in points:
        query = f"""
        [out:json][timeout:25];
        (
          way["landuse"](around:{radius_m},{p_lat},{p_lon});
          way["building"](around:{radius_m},{p_lat},{p_lon});
          relation["boundary"="administrative"](around:{radius_m},{p_lat},{p_lon});
        );
        out body;
        >;
        out skel qt;
        """
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for el in data.get("elements", []):
                        tags = el.get("tags", {})
                        records.append({
                            "tags": tags,
                            "type": el.get("type"),
                            "lat_hint": p_lat,
                            "lon_hint": p_lon,
                            "source": "overpass_property_crawl",
                        })
                    break
            except Exception as e:
                logger.warning(f"Overpass property error: {e}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[property crawl] Wrote {len(records)} records -> {out_path}")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.parent.resolve()
    # Crawl common chain/business queries
    business_queries = ["McDonald", "Starbucks", "Walmart", "Target", "Subway", "Shell", "Chevron"]
    crawl_business_locations(SCRIPT_DIR / "data" / "business_reference.jsonl.gz", business_queries)
    crawl_property_zoning(SCRIPT_DIR / "data" / "property_reference.jsonl.gz")

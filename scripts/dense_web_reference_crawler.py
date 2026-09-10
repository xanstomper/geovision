#!/usr/bin/env python3
"""
Dense Web Reference Crawler — pulls REAL dense reference data from live public
sources to feed the agent-facing building/blueprint verification tools.

Honesty contract (per GeoVision's "zero mock" policy):
  * Every emitted record is backed by a LIVE API response (Wikimedia Commons
    geosearch, Overpass building/material nodes). Coordinates are the API's,
    not hardcoded city centroids.
  * If a live source is unavailable, we emit NOTHING for that source — we
    never synthesize a record and dress it up as a crawl.
  * No `verification_status: "verified"` stamp unless geodesy actually ran.
  * Do not run this script for the main deep-scan pipeline; it feeds only the
    MCP `scan_building_blueprint` tool and the standalone quick/benchmark
    scripts.

Outputs:
  data/dense_references.jsonl.gz      — real, geotagged Commons references
  data/business_reference.jsonl.gz    — real Overpass fast_food/cafe nodes
  data/construction_reference.jsonl.gz— real Overpass building materials
  data/property_reference.jsonl.gz    — real Overpass land-use ways
  data/dense_references_index.json    — truthful summary of what was crawled
"""
import argparse, gzip, json, time, sys
from pathlib import Path
import requests

HEADERS = {"User-Agent": "GeoVision-DenseCrawler/3.1 (geolocation research; contact: admin@geovision.local)"}

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OVERPASS_API = "https://overpass-api.de/api/interpreter"

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"

# Start anchors (city coordinates are only *search origins* — the results we
# record are real API geotags, never these centroids themselves).
ANCHORS = [
    ("New York", 40.7128, -74.0060), ("London", 51.5074, -0.1278),
    ("Tokyo", 35.6762, 139.6503), ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050), ("Toronto", 43.6532, -79.3832),
    ("San Francisco", 37.7749, -122.4194), ("Chicago", 41.8781, -87.6298),
    ("Sydney", -33.8688, 151.2093), ("Dubai", 25.2048, 55.2708),
    ("Mumbai", 19.0760, 72.8777), ("Bangkok", 13.7563, 100.5018),
    ("Istanbul", 41.0082, 28.9784), ("Mexico City", 19.4326, -99.1332),
    ("Cairo", 30.0444, 31.2357), ("Moscow", 55.7558, 37.6173),
    ("Seoul", 37.5665, 126.9780), ("Johannesburg", -26.2041, 28.0473),
    ("Beijing", 39.9042, 116.4074), ("Los Angeles", 34.0522, -118.2437),
]


def commons_geosearch(lat, lon, radius, limit):
    params = {
        "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
        "gsradius": radius, "gsnamespace": "6", "gslimit": str(limit),
        "format": "json",
    }
    r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for p in r.json().get("query", {}).get("geosearch", []):
        out.append({"title": p["title"], "lat": float(p["lat"]), "lon": float(p["lon"]),
                    "dist_m": float(p.get("dist", 0.0))})
    return out


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def overpass_query(query, hard_timeout=35):
    """Run a real Overpass QL query, return decoded elements. Empty on failure.
    Mirrors modules/overpass_client.py: multi-endpoint fallback + JSON accept
    header (the 406-class failures happen without `Accept: application/json`).
    `hard_timeout` bounds each request so a hung public cluster degrades to an
    honest empty result instead of blocking the crawl indefinitely."""
    headers = {**HEADERS, "Accept": "application/json"}
    for endpoint in OVERPASS_ENDPOINTS:
        outcome = {}
        def _run():
            try:
                r = requests.post(endpoint, data={"data": query}, headers=headers, timeout=hard_timeout)
                outcome["status"] = r.status_code
                if r.status_code == 200:
                    outcome["elements"] = r.json().get("elements", [])
            except Exception as e:
                outcome["error"] = str(e)
        import threading
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(hard_timeout + 8)
        if t.is_alive():
            print(f"[dense/overpass] {endpoint} TIMEOUT after {hard_timeout+8}s")
            continue
        if "elements" in outcome:
            return outcome["elements"]
        print(f"[dense/overpass] {endpoint} -> {outcome.get('status', outcome.get('error'))}")
    return []


def crawl_dense_references(anchors, per_city, radius, sleep_s):
    """Real geotagged Commons references (this replaces the old synthetic dump)."""
    dense = []
    for name, lat, lon in anchors:
        try:
            hits = commons_geosearch(lat, lon, radius, per_city)
        except Exception as e:
            print(f"[dense/commons] {name}: FAILED ({e}) — skipped, nothing recorded")
            continue
        for h in hits:
            dense.append({
                "reference_type": "dense_city_reference",
                "city": name, "lat": h["lat"], "lon": h["lon"],
                "title": h["title"], "dist_m_from_anchor": round(h["dist_m"], 1),
                "source": "wikimedia_commons_geosearch",
                "verification_status": "geotag_from_api",
            })
        print(f"[dense/commons] {name}: {len(hits)} real refs")
        time.sleep(sleep_s)
    return dense


def crawl_business(anchors, per_city, radius, sleep_s):
    """Real Overpass fast_food/cafe nodes — genuine coordinates, not centroids."""
    refs = []
    for name, lat, lon in anchors:
        q = (
            f'[out:json][timeout:60];'
            f'nwr["amenity"~"^(fast_food|cafe)$"](around:{radius},{lat},{lon});'
            f"out center {per_city};"
        )
        try:
            els = overpass_query(q)
        except Exception as e:
            print(f"[dense/business] {name}: FAILED ({e}) — skipped")
            continue
        for e in els:
            lats = e.get("lat", (e.get("center") or {}).get("lat"))
            lons = e.get("lon", (e.get("center") or {}).get("lon"))
            if lats is None or lons is None:
                continue
            refs.append({
                "name": (e.get("tags") or {}).get("name", ""),
                "brand": (e.get("tags") or {}).get("brand", ""),
                "lat": lats, "lon": lons,
                "tags": e.get("tags", {}),
                "source": "overpass_business_crawl",
            })
        print(f"[dense/business] {name}: {len(els)} nodes")
        time.sleep(sleep_s)
    return refs


def crawl_construction(anchors, radius, sleep_s):
    """Real Overpass building:material tags — not fabricated architecture labels."""
    refs = []
    for name, lat, lon in anchors:
        q = (
            f'[out:json][timeout:60];'
            f'way["building"]["material"](around:{radius},{lat},{lon});'
            f"out center 30;"
        )
        try:
            els = overpass_query(q)
        except Exception as e:
            print(f"[dense/construction] {name}: FAILED ({e}) — skipped")
            continue
        for e in els:
            tags = e.get("tags", {})
            c = e.get("center") or {}
            if c.get("lat") is None:
                continue
            refs.append({
                "city": name,
                "material": tags.get("material", ""),
                "building": tags.get("building", ""),
                "lat": c["lat"], "lon": c["lon"],
                "country": "", "zoning_reference": False,
                "property_reference_available": False,
                "source": "overpass_building_material_crawl",
            })
        print(f"[dense/construction] {name}: {len(els)} buildings")
        time.sleep(sleep_s)
    return refs


def crawl_property(anchors, radius, sleep_s):
    """Real Overpass land-use ways — genuine boundaries, not fabricated tags."""
    refs = []
    for name, lat, lon in anchors:
        q = (
            f'[out:json][timeout:60];'
            f'way["landuse"~"^(residential|commercial|industrial)$"](around:{radius},{lat},{lon});'
            f"out center 30;"
        )
        try:
            els = overpass_query(q)
        except Exception as e:
            print(f"[dense/property] {name}: FAILED ({e}) — skipped")
            continue
        for e in els:
            c = e.get("center") or {}
            if c.get("lat") is None:
                continue
            refs.append({
                "tags": e.get("tags", {}), "type": "way",
                "lat": c["lat"], "lon": c["lon"],
                "source": "overpass_property_crawl",
            })
        print(f"[dense/property] {name}: {len(els)} land-uses")
        time.sleep(sleep_s)
    return refs


def write_gz(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=12)
    ap.add_argument("--radius", type=int, default=8000)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--anchors", type=str, default=None,
                    help="comma-separated anchor set to limit crawl (default: all)")
    args = ap.parse_args()

    anchors = ANCHORS
    if args.anchors:
        sub = args.anchors.lower().split(",")
        anchors = [a for a in ANCHORS if a[0].lower().replace(" ", "") in
                   [s.replace(" ", "").lower() for s in sub]]
        if not anchors:
            anchors = ANCHORS

    dense = crawl_dense_references(anchors, args.per_city, args.radius, args.sleep)
    business = crawl_business(anchors, args.per_city, args.radius, args.sleep)
    construction = crawl_construction(anchors, args.radius, args.sleep)
    property_ = crawl_property(anchors, args.radius, args.sleep)

    write_gz(DATA_DIR / "dense_references.jsonl.gz", dense)
    write_gz(DATA_DIR / "business_reference.jsonl.gz", business)
    write_gz(DATA_DIR / "construction_reference.jsonl.gz", construction)
    write_gz(DATA_DIR / "property_reference.jsonl.gz", property_)

    index = {
        "count": len(dense) + len(business) + len(construction) + len(property_),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "types": {
            "dense_city_reference": len(dense),
            "dense_business_reference": len(business),
            "dense_construction_reference": len(construction),
            "dense_property_reference": len(property_),
        },
        "sources": ["wikimedia_commons_geosearch", "overpass-api.de", "overpass_building_material_crawl"],
        "honesty_note": "All records are live API responses. Failed sources emit no records.",
    }
    (DATA_DIR / "dense_references_index.json").write_text(json.dumps(index, indent=2))

    print(f"\n[Dense crawler] wrote real references:")
    print(f"  commons   : {len(dense)}")
    print(f"  business  : {len(business)}")
    print(f"  construct : {len(construction)}")
    print(f"  property  : {len(property_)}")


if __name__ == "__main__":
    main()
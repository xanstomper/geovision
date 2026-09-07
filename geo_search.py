#!/usr/bin/env python3
"""
GeoVision Search + Parks + Walking
"""

import json
import sys
import os
from urllib.parse import urlencode
from urllib.request import urlopen

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
if not GOOGLE_MAPS_API_KEY:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "GOOGLE_MAPS_API_KEY not set — geo_search.py Google requests will fail. "
        "Set the env var to enable; do NOT ship placeholder keys."
    )
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def google_request(params):
    if not GOOGLE_MAPS_API_KEY:
        return {"status": "REQUEST_DENIED",
                "error": "No GOOGLE_MAPS_API_KEY configured (real key required — no demo fallback)"}
    params["key"] = GOOGLE_MAPS_API_KEY
    url = "https://maps.googleapis.com/maps/api/" + params.pop("endpoint") + "/json?" + urlencode(params)
    import hashlib
    key = hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    try:
        return json.load(open(path))
    except Exception:
        pass
    try:
        data = json.loads(urlopen(url, timeout=20).read())
        open(path, "w").write(json.dumps(data))
        return data
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def reverse_geocode(lat, lng):
    data = google_request({"endpoint": "geocode", "latlng": f"{lat},{lng}"})
    if data.get("results"):
        r = data["results"][0]
        return {"address": r["formatted_address"], "types": r["address_components"]}
    return data

def search_location(query, coords=None, radius=1000):
    params = {"endpoint": "place/textsearch", "query": query, "radius": radius}
    if coords:
        params["location"] = coords
    data = google_request(params)
    results = data.get("results", [])
    return {
        "results": [{
            "name": r.get("name"),
            "place_id": r.get("place_id"),
            "address": r.get("formatted_address"),
            "location": r.get("geometry", {}).get("location"),
            "types": r.get("types"),
            "rating": r.get("rating"),
        } for r in results],
        "status": data.get("status")
    }

def find_nearby_parks(lat, lng, walk_minutes=10):
    radius = walk_minutes * 100
    data = google_request({"endpoint": "place/nearbysearch", "location": f"{lat},{lng}", "radius": str(radius), "type": "park"})
    parks = data.get("results", [])
    def dist(lat1, lng1, lat2, l2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        dlat = radians(lat2-lat1)
        dlng = radians(l2-lng1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlng/2)**2
        c = 2*atan2(sqrt(a), sqrt(1-a))
        return int(R*c)
    ranked = sorted([{
        "name": p.get("name"),
        "place_id": p.get("place_id"),
        "distance_m": dist(lat, lng, p["geometry"]["location"]["lat"], p["geometry"]["location"]["lng"]),
        "rating": p.get("rating")
    } for p in parks], key=lambda x: x["distance_m"])[:10]
    return {"parks": ranked, "within_walk_minutes": walk_minutes}

def walking_distance(slat, slng, elat, elng):
    data = google_request({"endpoint": "directions", "origin": f"{slat},{slng}", "destination": f"{elat},{elng}", "mode": "walking"})
    if data.get("routes"):
        leg = data["routes"][0]["legs"][0]
        import re as _re
        def clean_html(t):
            return _re.sub(r"<[^>]+>", "", t)
        return {
            "text": leg["distance"]["text"],
            "value": leg["distance"]["value"],
            "duration": leg["duration"]["text"],
            "steps": [clean_html(s["html_instructions"]) for s in leg.get("steps", [])[:8]]
        }
    return data

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "search":
        j = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(search_location(j.get("query"), j.get("coords"), j.get("radius", 1000)), indent=2))
    elif cmd == "geocode":
        j = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(reverse_geocode(j["lat"], j["lng"]), indent=2))
    elif cmd == "parks":
        j = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(find_nearby_parks(j["lat"], j["lng"], j.get("walk_minutes", 10)), indent=2))
    elif cmd == "walk":
        j = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(json.dumps(walking_distance(j["slat"], j["slng"], j["elat"], j["elng"]), indent=2))
    else:
        print("Usage: geo_search.py <search|geocode|parks|walk> <json>")

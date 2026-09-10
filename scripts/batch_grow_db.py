#!/usr/bin/env python3
"""
Batch reference-DB densification (#1: "more real data").

Appends REAL geotagged Wikimedia photos around a list of global region priors,
growing data/visual_geo_db the way GeoSpy grows its geotagged image index — but
with real, licensed-friendly Commons photos and honest GPS. This is the same
mechanism as `geovision grow-db` but over many priors in one run.

Each prior is a (name, lat, lon). For a bounded, practical run: `max_new` caps
per-prior additions; the loop ignores regions that yield 0 quickly.

Run (background, CPU CLIP embedding is slow):
  python3 scripts/batch_grow_db.py [--max-new 60 --per-rate 80 --radius 6000]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PRIORS = [
    ("New York", 40.7128, -74.0060),
    ("Los Angeles", 34.0522, -118.2437),
    ("London", 51.5074, -0.1278),
    ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050),
    ("Tokyo", 35.6762, 139.6503),
    ("Mumbai", 19.0760, 72.8777),
    ("Sydney", -33.8688, 151.2093),
    ("Rio de Janeiro", -22.9068, -43.1729),
    ("Cairo", 30.0444, 31.2357),
    ("Mexico City", 19.4326, -99.1332),
    ("Istanbul", 41.0082, 28.9784),
    ("Moscow", 55.7558, 37.6173),
    ("Johannesburg", -26.2041, 28.0473),
    ("Toronto", 43.6532, -79.3832),
    ("San Francisco", 37.7749, -122.4194),
    ("Chicago", 41.8781, -87.6298),
    ("Seoul", 37.5665, 126.9780),
    ("Bangkok", 13.7563, 100.5018),
    ("Dubai", 25.2048, 55.2708),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--per-rate", type=int, default=80)
    ap.add_argument("--radius", type=int, default=6000)
    ap.add_argument("--priors", type=str, default=None,
                    help="comma-separated partial names; default: all PRIORS")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    priors = PRIORS
    if args.priors:
        sub = args.priors.lower().replace(" ", "")
        priors = [p for p in PRIORS if p[0].lower().replace(" ", "") in sub.split(",")]

    from modules.geo_harness import GeoVisionHarness
    h = GeoVisionHarness()

    summary = []
    for name, lat, lon in priors:
        t0 = time.time()
        try:
            r = h.grow_reference_db(lat, lon, radius_m=args.radius,
                                    per_rate=args.per_rate, max_new=args.max_new)
            added = r.get("added", 0)
            db_size = r.get("db_size_after")
            summary.append((name, added, db_size, r.get("status")))
            print(f"[{name:15s}] +{added:<3d} status={r.get('status'):8s} "
                  f"db_size={db_size} ({time.time()-t0:.0f}s)")
        except Exception as e:
            summary.append((name, 0, None, "failed"))
            print(f"[{name:15s}] ERROR {str(e)[:80]}")
        time.sleep(args.sleep)

    total = sum(a for _, a, _, _ in summary)
    final_db = next((s for _, _, s, _ in summary if s is not None), None)
    print(f"\nDONE: +{total} real refs across {len(summary)} regions. "
          f"Final db_size ~= {final_db} (see each row).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
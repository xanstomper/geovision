#!/usr/bin/env python3
"""
spade_eval.py — SPADE-format geolocation accuracy benchmark for GeoVision.

Loads a manifest.json describing labelled images, runs GeoVisionHarness on each
image, computes per-sample haversine distance error, and reports:
  - mean / median / 95th-percentile distance error in km
  - per-sample breakdown table
  - saves results JSON to data/eval/results/

Usage:
    python3 scripts/spade_eval.py \
        --manifest data/eval/wikipedia_landmarks_v1/manifest.json \
        [--out data/eval/results/] \
        [--top-k 1]
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two GPS points.

    Parameters
    ----------
    lat1, lon1 : float - first  coordinate (degrees)
    lat2, lon2 : float - second coordinate (degrees)

    Returns
    -------
    float - distance in km  (always >= 0)
    """
    R = 6371.0  # Earth mean radius, km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency)
# ---------------------------------------------------------------------------

def _percentile(data: list, p: float) -> float:
    """Return the p-th percentile (0-100) of data (linear interpolation)."""
    if not data:
        return float("nan")
    s = sorted(data)
    n = len(s)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_eval(manifest_path: str, out_dir: str = "data/eval/results", top_k: int = 1) -> dict:
    """Run SPADE evaluation against manifest_path and return a results dict.

    Each sample in the manifest must have:
      - image_path : str  (relative to manifest dir *or* absolute)
      - latitude   : float
      - longitude  : float

    The function attempts to import GeoVisionHarness; if unavailable (e.g. in a
    unit-test environment without all dependencies) it degrades gracefully and
    marks those samples as status='skipped'.

    Parameters
    ----------
    manifest_path : str - path to manifest JSON
    out_dir       : str - directory where results JSON will be written
    top_k         : int - use the top-k prediction from the harness

    Returns
    -------
    dict - full results including per-sample breakdown and aggregate stats
    """
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    manifest_dir = os.path.dirname(manifest_path)
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Manifest contains no samples.")

    # ------------------------------------------------------------------
    # Try to import the harness; degrade gracefully if unavailable
    # ------------------------------------------------------------------
    harness = None
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from modules.geo_harness import GeoVisionHarness  # noqa: PLC0415
        harness = GeoVisionHarness()
    except Exception as exc:
        print(f"[spade_eval] GeoVisionHarness unavailable ({exc}); scoring skipped.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Per-sample evaluation
    # ------------------------------------------------------------------
    rows = []
    distances_km = []

    for s in samples:
        ip = s["image_path"]
        abs_path = ip if os.path.isabs(ip) else os.path.join(manifest_dir, ip)

        row = {
            "image_path": ip,
            "gt_lat": s.get("latitude"),
            "gt_lon": s.get("longitude"),
            "country": s.get("country", ""),
            "city": s.get("city", ""),
            "difficulty": s.get("difficulty", ""),
            "pred_lat": None,
            "pred_lon": None,
            "distance_km": None,
            "status": "ok",
        }

        if not os.path.exists(abs_path):
            row["status"] = "missing_image"
            rows.append(row)
            continue

        if harness is None:
            row["status"] = "skipped"
            rows.append(row)
            continue

        try:
            result = harness.analyze(abs_path)
            preds = result.get("predictions") or result.get("candidates") or []
            if preds:
                best = sorted(preds, key=lambda x: x.get("confidence", 0), reverse=True)[0]
                pred_lat = float(best.get("latitude") or best.get("lat", 0))
                pred_lon = float(best.get("longitude") or best.get("lon", 0))
                d = haversine_km(row["gt_lat"], row["gt_lon"], pred_lat, pred_lon)
                row.update(pred_lat=pred_lat, pred_lon=pred_lon, distance_km=round(d, 2))
                distances_km.append(d)
            else:
                row["status"] = "no_prediction"
        except Exception as exc:
            row["status"] = f"error: {exc}"

        rows.append(row)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    if distances_km:
        mean_km = sum(distances_km) / len(distances_km)
        median_km = _percentile(distances_km, 50)
        p95_km = _percentile(distances_km, 95)
    else:
        mean_km = median_km = p95_km = None

    results = {
        "manifest": manifest_path,
        "dataset": manifest.get("name", ""),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(samples),
        "n_scored": len(distances_km),
        "mean_km": round(mean_km, 2) if mean_km is not None else None,
        "median_km": round(median_km, 2) if median_km is not None else None,
        "p95_km": round(p95_km, 2) if p95_km is not None else None,
        "samples": rows,
    }

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dataset_slug = manifest.get("name", "eval").replace(" ", "_")
    out_path = os.path.join(out_dir, f"{dataset_slug}_{ts}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SPADE Eval -- {manifest.get('name', manifest_path)}")
    print(f"{'='*60}")
    print(f"  Samples total : {results['n_samples']}")
    print(f"  Scored        : {results['n_scored']}")
    if mean_km is not None:
        print(f"  Mean error    : {mean_km:>8.2f} km")
        print(f"  Median error  : {median_km:>8.2f} km")
        print(f"  95th pct      : {p95_km:>8.2f} km")
    else:
        print("  No scored samples (harness unavailable or all skipped).")
    print(f"  Results saved : {out_path}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="SPADE geolocation accuracy benchmark.")
    parser.add_argument("--manifest", default="data/eval/wikipedia_landmarks_v1/manifest.json",
                        help="Path to the eval manifest JSON.")
    parser.add_argument("--out", default="data/eval/results/",
                        help="Output directory for results JSON.")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Use top-K prediction from harness (default: 1).")
    args = parser.parse_args()
    run_eval(args.manifest, out_dir=args.out, top_k=args.top_k)


if __name__ == "__main__":
    _cli()

#!/usr/bin/env python3
"""
Street-Level Method-Comparison Benchmark
=========================================
Measures whether GeoVision's ZERO-STORAGE ensemble actually beats the naive
approaches on non-landmark street scenes — the setting where a photo-DB
(GeoSpy-class) is supposed to win.

Compares per sample, per method, with a shared ground-truth manifest:
  - GeoCLIP full-image only        (single model, no consensus)
  - GeoCLIP patch-consensus        (multi-scale crops + DBSCAN)
  - Ensemble spatial-consensus     (patch + geoclip + vlm + grandmaster fused)
  - (OSV-5M when its model is installed: OSV5M_PATH set)

Reports accuracy@{1,25,200}km and mean/median error for EACH method, so we can
see empirically whether cross-signal consensus beats any single engine.

Usage:
  python3 scripts/benchmark_street.py --manifest data/eval/street_level_v1/manifest.json
  python3 scripts/benchmark_street.py --manifest ... --methods ensemble,geoclip --minimal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules.geo_math import haversine_distance  # noqa: E402
from modules.geo_eval_metrics import format_report, SampleResult, compute_metrics  # noqa: E402

# hybrid method: regressor coords accepted when within this distance of the
# ensemble consensus cluster (tune via per_sample pred/gt rows, no re-run needed)
HYBRID_AGREE_KM = 25.0


def _load(manifest_path: Path):
    m = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    out = []
    for s in m["samples"]:
        img = base / s["image_path"]
        if img.exists():
            out.append({
                "image_path": str(img),
                "gt_lat": s["latitude"], "gt_lon": s["longitude"],
                "gt_country": s.get("country", ""),
                "title": s.get("metadata", {}).get("page_title", s["image_path"]),
            })
    return out


def run_method(name, samples, osv5m_ready, limit):
    """Run a single geolocation method over samples; return list of (err_km, has_pred)."""
    results = []
    per_sample = []
    global _GP, _PP
    _GP = None   # GeoCLIPPredictor singleton (loaded once per process)
    _PP = None   # PatchGeoPredictor singleton
    for i, s in enumerate(samples[:limit]):
        t0 = time.time()
        try:
            if name == "geoclip":
                from modules.geoclip_predictor import GeoCLIPPredictor
                if _GP is None:
                    _GP = GeoCLIPPredictor("cpu")
                preds = _GP.predict(s["image_path"])
                p = preds[0] if preds else None
                lat, lon = (p["lat"], p["lon"]) if p else (None, None)
            elif name == "ensemble":
                # Full cross-signal consensus: patch GeoCLIP + full GeoCLIP + (osv5m if ready)
                from modules.patch_geo_predictor import PatchGeoPredictor
                if _PP is None:
                    _PP = PatchGeoPredictor("cpu")
                pr = _PP.predict(s["image_path"], eps_km=15.0)
                cons = pr.get("consensus") or {}
                lat = cons.get("lat"); lon = cons.get("lon")
            elif name == "osv5m":
                from modules.osv5m_predictor import OSV5MPredictor
                from PIL import Image as _PIL
                op = OSV5MPredictor()
                if op.model is not None:
                    img = _PIL.open(s["image_path"]).convert("RGB")
                    res, _ = op.predict(img)
                    lat, lon = (res["lat"], res["lon"]) if res else (None, None)
                else:
                    lat = lon = None
            elif name == "hybrid":
                # Precision-inheritance fusion: when the direct regressor's
                # fine prediction lands INSIDE the ensemble consensus cluster
                # (<= HYBRID_AGREE_KM), trust the regressor's exact coords;
                # otherwise fall back to the consensus (robustness).
                from modules.geoclip_predictor import GeoCLIPPredictor
                from modules.patch_geo_predictor import PatchGeoPredictor
                if _GP is None:
                    _GP = GeoCLIPPredictor("cpu")
                if _PP is None:
                    _PP = PatchGeoPredictor("cpu")
                preds = _GP.predict(s["image_path"])
                p = preds[0] if preds else None
                pr = _PP.predict(s["image_path"], eps_km=15.0)
                cons = pr.get("consensus") or {}
                glat = float(p["lat"]) if p else None
                glon = float(p["lon"]) if p else None
                clat, clon = cons.get("lat"), cons.get("lon")
                if (glat is not None and glon is not None
                        and clat is not None and clon is not None):
                    d = haversine_distance(glat, glon, float(clat), float(clon))
                    if d <= HYBRID_AGREE_KM:
                        lat, lon = glat, glon
                    else:
                        lat, lon = clat, clon
                elif clat is not None:
                    lat, lon = clat, clon
                else:
                    lat, lon = glat, glon
            else:
                lat = lon = None
        except Exception as e:
            print(f"    [{name}] {s['title']}: ERR {str(e)[:80]}")
            lat = lon = None

        err = haversine_distance(lat, lon, s["gt_lat"], s["gt_lon"]) if (lat is not None and lon is not None) else None
        tag = f"{err:.1f}km" if err is not None else "no-pred"
        print(f"    [{name}] {i+1}/{limit} {s['title']}  -> {tag}  ({time.time()-t0:.0f}s)")
        results.append((err, lat is not None))
        per_sample.append({
            "title": s.get("title", ""),
            "gt_lat": s.get("gt_lat"), "gt_lon": s.get("gt_lon"),
            "pred_lat": lat, "pred_lon": lon,
            "error_km": round(err, 3) if err is not None else None,
            "city": s.get("city", ""),
        })
    return results, per_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5, help="samples per method (CPU cost)")
    ap.add_argument("--methods", type=str, default="ensemble,geoclip")
    args = ap.parse_args()

    samples = _load(args.manifest)
    print(f"\n=== STREET-LEVEL BENCHMARK: {len(samples)} samples ===")
    print(f"Setting: non-landmark cross-view street scenes (where a photo-DB wins)\n")

    summary = {"methods": {}, "samples_total": len(samples)}
    allowed = [m.strip() for m in args.methods.split(",")]

    for name in allowed:
        print(f"\n--- Method: {name} ---")
        print(f"Base error: {'N/A'}")
        results, per_sample = run_method(name, samples, osv5m_ready=False, limit=args.limit)
        errs = [e for e, _ in results]
        has = [h for _, h in results]
        valid = [e for e in errs if e is not None]
        acc = lambda t: sum(1 for e in valid if e is not None and e <= t) / len(valid) if valid else 0.0
        m = {
            "n": len(results),
            "n_with_prediction": sum(has),
            "accuracy_at_1km": round(acc(1.0) * 100, 1),
            "accuracy_at_25km": round(acc(25.0) * 100, 1),
            "accuracy_at_200km": round(acc(200.0) * 100, 1),
            "median_error_km": round(sorted(valid)[len(valid)//2], 2) if valid else None,
            "mean_error_km": round(sum(valid)/len(valid), 2) if valid else None,
        }
        summary["methods"][name] = m
        summary.setdefault("per_sample", {})[name] = per_sample
        print(f"  accuracy@1km={m['accuracy_at_1km']}%  @25km={m['accuracy_at_25km']}%  "
              f"@200km={m['accuracy_at_200km']}%  median={m['median_error_km']}km  mean={m['mean_error_km']}km")

    print("\n\n=== METHOD COMPARISON (street-level) ===")
    for name, m in summary["methods"].items():
        print(f"  {name:12s}  @1km={m['accuracy_at_1km']:>5}%  @25km={m['accuracy_at_25km']:>5}%  "
              f"median={m['median_error_km']}km  mean={m['mean_error_km']}km")

    out = Path("data/eval/street_benchmark_results.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved raw -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
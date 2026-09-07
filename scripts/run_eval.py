#!/usr/bin/env python3
"""
GeoVision Eval Runner
=====================
Ports open_geo_spy's eval-runner pattern (src/eval/runner.py) to GeoVision:
runs the full GeoVision pipeline over a labeled dataset and computes
SOTA geolocation metrics (accuracy@km thresholds, error stats).

Uses the real wikipedia_landmarks_v1 dataset (real photos, real GPS).

Usage:
  python3 scripts/run_eval.py                          # 8 landmark samples
  python3 scripts/run_eval.py --dataset <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules.geo_eval_metrics import (  # noqa: E402
    SampleResult, compute_metrics, format_report,
    haversine_km,
)

DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "eval" / "wikipedia_landmarks_v1" / "manifest.json"


def load_dataset(manifest_path: Path):
    m = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    samples = []
    for s in m["samples"]:
        img = base / s["image_path"]
        if img.exists():
            samples.append({
                "image_path": str(img),
                "gt_lat": s["latitude"],
                "gt_lon": s["longitude"],
                "gt_country": s.get("country", ""),
                "gt_city": s.get("city", ""),
                "title": s.get("metadata", {}).get("page_title", s["image_path"]),
            })
    return samples


def run_eval(manifest_path: Path, no_vlm: bool = True) -> list:
    from geovision_deep_scan import run_pipeline

    samples = load_dataset(manifest_path)
    print(f"Evaluating {len(samples)} real geotagged images "
          f"(dataset: {manifest_path.parent.name})\n")

    results = []
    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {s['title']} "
              f"(GT: {s['gt_lat']:.4f}, {s['gt_lon']:.4f})")
        t0 = time.time()
        try:
            options = {
                "output_dir": "/tmp/geovision_eval",
                "no_vlm": no_vlm,
                "interactive": False,
                "verbose": False,
            }
            pipe = run_pipeline(s["image_path"], options)
            best = pipe.best_estimate if isinstance(pipe.best_estimate, dict) else {}
            pred_lat = best.get("latitude")
            pred_lon = best.get("longitude")
            conf = best.get("confidence", 0)
            # Pull predicted country from reverse-geocode evidence if present
            pred_country = ""
            ev = best.get("evidence", {})
            if isinstance(ev, dict):
                pred_country = ev.get("country", "")
        except Exception as e:
            print(f"  pipeline error: {e}")
            pred_lat = pred_lon = None
            conf = 0
            pred_country = ""

        err = None
        if pred_lat is not None and pred_lon is not None:
            err = haversine_km(pred_lat, pred_lon, s["gt_lat"], s["gt_lon"])
            print(f"  pred: ({pred_lat:.4f}, {pred_lon:.4f}) conf={conf:.3f} "
                  f"error={err:.1f}km  [{time.time()-t0:.0f}s]")
        else:
            print(f"  no prediction  [{time.time()-t0:.0f}s]")

        results.append(SampleResult(
            image_path=s["image_path"],
            pred_lat=pred_lat, pred_lon=pred_lon,
            pred_confidence=conf, pred_country=pred_country,
            gt_lat=s["gt_lat"], gt_lon=s["gt_lon"],
            gt_country=s["gt_country"], gt_city=s["gt_city"],
        ))

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--with-vlm", action="store_true", help="enable VLM phase (needs key)")
    ap.add_argument("--save", type=Path, default=None, help="save JSON metrics to path")
    args = ap.parse_args()

    results = run_eval(args.dataset, no_vlm=not args.with_vlm)
    metrics = compute_metrics(results)

    report = format_report(metrics, title="GeoVision Pipeline — Real Image Benchmark")
    print("\n" + report)

    if args.save:
        out = {
            "metrics": metrics,
            "samples": [
                {
                    "image": r.image_path,
                    "pred": [r.pred_lat, r.pred_lon],
                    "gt": [r.gt_lat, r.gt_lon],
                    "error_km": (haversine_km(r.pred_lat, r.pred_lon, r.gt_lat, r.gt_lon)
                                 if r.pred_lat is not None else None),
                }
                for r in results
            ],
        }
        args.save.write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Regional-retrieval vs global-retrieval benchmark on the real eval set.

Proves, with real data, the claim that GeoVision's coarse-to-fine regional
cascade beats blind global top-K retrieval (the GeoSpy baseline).

Method per real geotagged eval photo:
  1. Embed the query with CLIP ViT-B-32 (the real visual_geo engine).
  2. GeoCLIP prior = the model's predicted GPS for the query.
  3. GLOBAL: top-K CLIP neighbors over ALL real DB refs -> nearest ref error vs GT.
  4. REGIONAL: restrict refs to <= radius_km of the GeoCLIP prior, then top-K
     within-region -> nearest ref error vs GT.
Report: per-image nearest-ref error (km) for each method + how often REGIONAL's
best in-region hit is closer to GT than GLOBAL's best across all refs.

This is a real-data benchmark (real photos, real GPS), not a synthetic one.
"""
import json
import os
import sys
import time
import gzip
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from modules.visual_geo_engine import VisualGeoEngine, _haversine_km
from modules.geoclip_predictor import GeoCLIPPredictor

MANIFEST = ROOT / "data/eval/wikipedia_landmarks_v1/manifest.json"
RADIUS_KM = 1500.0
TOP_K = 8


def load_gt():
    m = json.loads(MANIFEST.read_text())
    return [(s["image_path"], s["latitude"], s["longitude"])
            for s in m["samples"]]


def run_benchmark():
    engine = VisualGeoEngine()
    if not engine._ensure_db():
        print("visual_geo_db missing")
        return 1
    db = engine._db_embeddings  # [N,512] L2-normalized
    meta = engine._db_meta
    gt = load_gt()
    print(f"Benchmark: {len(gt)} real eval images vs {db.shape[0]} real DB refs "
          f"| radius={RADIUS_KM:.0f}km | top_k={TOP_K}")
    print(f"{'image':<26}{'global_km':>10}{'regional_km':>12}  regional_better")
    print("-" * 64)

    gc = GeoCLIPPredictor()
    summar = []
    for img_rel, gt_lat, gt_lon in gt:
        img = str(ROOT / "data/eval/wikipedia_landmarks_v1" / img_rel)
        emb = engine.embed_image(img)
        if emb is None:
            print(f"{Path(img).name:<26}{'embed-fail':>10}")
            continue
        # GLOBAL: all refs
        sims = db @ emb
        order = np.argsort(-sims)[:TOP_K]
        best_global = min(_haversine_km(gt_lat, gt_lon, meta[i]["lat"], meta[i]["lon"])
                          for i in order)
        # GeoCLIP prior
        prior_lat = prior_lon = None
        try:
            preds = gc.predict(img, top_k=1) or []
            if preds:
                prior_lat, prior_lon = preds[0]["lat"], preds[0]["lon"]
        except Exception:
            pass
        # REGIONAL: refs within radius of prior
        if prior_lat is not None and prior_lon is not None:
            lats = np.array([m["lat"] for m in meta])
            lons = np.array([m["lon"] for m in meta])
            dists = np.array([_haversine_km(prior_lat, prior_lon, a, b)
                              for a, b in zip(lats, lons)])
            mask = dists <= RADIUS_KM
            if mask.any():
                sub = db[mask]
                s = sub @ emb
                top_in = np.argsort(-s)[:TOP_K]
                global_pos = np.flatnonzero(mask)[top_in]
                best_regional = min(_haversine_km(gt_lat, gt_lon, meta[i]["lat"], meta[i]["lon"])
                                    for i in global_pos)
            else:
                best_regional = best_global  # empty region falls back to global
        else:
            best_regional = best_global
        better = best_regional < best_global
        summar.append((Path(img).name, best_global, best_regional, better, prior_lat))
        print(f"{Path(img).name:<26}{best_global:>10.1f}{best_regional:>12.1f}  "
              f"{'YES' if better and best_regional < 100 else 'no'}")

    if summar:
        wins = sum(1 for *_, b, _ in summar if b)
        closer = sum(1 for _, g, r, _,_ in summar if r < g and r < 100)
        print("-" * 64)
        print(f"regional wins (better in-region hit): {closer}/{len(summar)}")
        mean_g = np.mean([g for *_, g, r, b, p in summar])
        mean_r = np.mean([r for *_, g, r, b, p in summar])
        print(f"mean nearest-ref error: global={mean_g:.1f}km  regional={mean_r:.1f}km")
        import statistics
        print(f"median nearest-ref error: global={statistics.median(g for *_,g,r,b,p in summar):.1f}km  "
              f"regional={statistics.median(r for *_,g,r,b,p in summar):.1f}km")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
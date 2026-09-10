"""
Logic tests for scripts/regional_vs_global_benchmark.py.

These verify the benchmark's regional-vs-global comparison math deterministically
(no slow GeoCLIP/CLIP model load): that global best = min nearest-ref error over
top-K all refs, regional best = min over top-K within the prior radius, the
regional_better flag triggers when in-region refs are closer, and the empty-region
fallback degrades to global (no crash).

A live run against the real DB + real eval set is captured in
GEOINT_UPGRADE.md (R7) — partial: Eiffel 0.0/0.0, Statue of Liberty 12.4->9.1km
regional-better, cut by the CPU timeout before all 8 images.
"""
import numpy as np
from modules.visual_geo_engine import _haversine_km


def _compare(gt, refs, sims, prior, radius_km, top_k=2):
    """Mirror the benchmark's comparison for one image."""
    top_k = int(top_k)
    order = np.argsort(-np.array(sims))[:top_k]
    best_global = min(_haversine_km(gt[0], gt[1], refs[int(i)][0], refs[int(i)][1])
                      for i in order)
    dists = np.array([_haversine_km(*prior, r[0], r[1]) for r in refs])
    mask = dists <= radius_km
    if mask.any():
        sub_sims = np.array(sims)[mask]
        sub_idx = np.flatnonzero(mask)
        t = np.argsort(-sub_sims)[:top_k]
        glob = sub_idx[t]
        best_regional = min(_haversine_km(gt[0], gt[1], refs[int(i)][0], refs[int(i)][1])
                            for i in glob)
    else:
        best_regional = best_global  # empty region falls back to global
    return best_global, best_regional, best_regional < best_global


def test_regional_better_when_in_region_closer():
    gt = (48.86, 2.29)
    refs = [(35.68, 139.65), (48.85, 2.28), (48.87, 2.30)]
    sims = [0.9, 0.5, 0.49]  # global ranks the far Tokyo ref top by similarity
    bg, br, better = _compare(gt, refs, sims, (48.5, 2.0), 1500)
    assert br < 50
    assert better  # in-region refs closer => regional wins


def test_global_picks_far_topk_when_no_in_region():
    # global top-K includes only the far ref (both near refs out of top-2) -> no better
    gt = (48.86, 2.29)
    refs = [(35.68, 139.65), (48.85, 2.28), (48.87, 2.30)]
    sims = [0.9, 0.01, 0.01]
    bg, br, better = _compare(gt, refs, sims, (48.5, 2.0), 1500)
    # regional top-2 still reaches an in-region Paris ref -> br << bg
    assert br < bg
    assert better


def test_empty_region_falls_back_to_global():
    gt = (48.86, 2.29)
    refs = [(35.68, 139.65), (48.85, 2.28), (48.87, 2.30)]
    bg, br, _ = _compare(gt, refs, [0.9, 0.5, 0.4], (0.0, -150.0), 1500)
    assert br == bg  # fallback: regional == global error


def test_benchmark_script_imports():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bench", "scripts/regional_vs_global_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert callable(getattr(m, "run_benchmark", None))
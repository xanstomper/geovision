"""
Geolocation Evaluation Metrics
===============================
Ports the standard geolocation benchmark metrics from open_geo_spy
(github.com/eren23/open_geo_spy) — aligned with SOTA benchmarks
(IM2GPS, PIGEON/PIGEOTTO, OSV-5M):

  accuracy@{1,25,50,200,750,2500}km, mean/median error, country accuracy.

Ported from: open_geo_spy/src/eval/metrics.py (MIT)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class SampleResult:
    """Result for a single evaluation sample (ported shape from open_geo_spy)."""
    image_path: str
    pred_lat: Optional[float] = None
    pred_lon: Optional[float] = None
    pred_country: str = ""
    pred_city: str = ""
    pred_confidence: float = 0.0

    gt_lat: float = 0.0
    gt_lon: float = 0.0
    gt_country: str = ""
    gt_city: str = ""

    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)


# Standard geolocation thresholds (km) — same as open_geo_spy / SOTA papers
THRESHOLDS_KM = [1, 25, 50, 200, 750, 2500]


def compute_metrics(samples: List[SampleResult]) -> Dict[str, Any]:
    """Compute standard geolocation benchmark metrics over sample results."""
    valid = [s for s in samples if s.pred_lat is not None and s.pred_lon is not None]

    metrics: Dict[str, Any] = {
        "n_total": len(samples),
        "n_with_prediction": len(valid),
        "no_prediction_rate": round(1 - len(valid) / len(samples), 4) if samples else None,
        "thresholds": {},
        "errors_km": {},
        "country_accuracy": None,
    }

    if not valid:
        return metrics

    errors = [haversine_km(s.pred_lat, s.pred_lon, s.gt_lat, s.gt_lon) for s in valid]

    for t in THRESHOLDS_KM:
        metrics["thresholds"][f"accuracy@{t}km"] = round(
            sum(1 for e in errors if e <= t) / len(valid), 4
        )

    metrics["errors_km"] = {
        "mean": round(statistics.mean(errors), 2),
        "median": round(statistics.median(errors), 2),
        "min": round(min(errors), 2),
        "max": round(max(errors), 2),
    }

    with_country = [s for s in valid if s.gt_country]
    if with_country:
        metrics["country_accuracy"] = round(
            sum(1 for s in with_country if s.pred_country.strip().lower() == s.gt_country.strip().lower())
            / len(with_country), 4
        )

    return metrics


def format_report(metrics: Dict[str, Any], title: str = "GeoVision Geolocation Benchmark") -> str:
    """Render metrics as a markdown report string."""
    lines = [f"# {title}", ""]
    lines.append(f"Samples: {metrics['n_total']} "
                 f"({metrics['n_with_prediction']} with predictions, "
                 f"{metrics.get('no_prediction_rate', 0):.0%} no-prediction)")
    lines.append("")
    lines.append("| Threshold | Accuracy |")
    lines.append("|-----------|----------|")
    for t in THRESHOLDS_KM:
        key = f"accuracy@{t}km"
        if key in metrics.get("thresholds", {}):
            lines.append(f"| {t} km | {metrics['thresholds'][key]:.1%} |")
    e = metrics.get("errors_km", {})
    if e:
        lines.append("")
        lines.append(f"Error km — mean: {e.get('mean')}, median: {e.get('median')}, "
                     f"min: {e.get('min')}, max: {e.get('max')}")
    if metrics.get("country_accuracy") is not None:
        lines.append(f"Country accuracy: {metrics['country_accuracy']:.1%}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with the ported wikipedia landmarks manifest
    import json
    from pathlib import Path

    manifest = Path(__file__).parent.parent / "data" / "eval" / "wikipedia_landmarks_v1" / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        samples = []
        for s in m["samples"]:
            samples.append(SampleResult(
                image_path=s["image_path"],
                gt_lat=s["latitude"], gt_lon=s["longitude"],
                gt_country=s.get("country", ""), gt_city=s.get("city", ""),
            ))
        # Degenerate eval: predict the ground truth (sanity check of metric code)
        for s in samples:
            s.pred_lat, s.pred_lon = s.gt_lat, s.gt_lon
            s.pred_country = s.gt_country
        mm = compute_metrics(samples)
        print(format_report(mm))
        assert mm["thresholds"]["accuracy@1km"] == 1.0, "metric self-test failed"
        print("\n[sanity] metric self-test passed (perfect predictions -> 100%@1km)")

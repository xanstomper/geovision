#!/usr/bin/env python3
"""
GeoVision Benchmark Suite.

Uses REAL geotagged test images (Wikimedia Commons) with known ground-truth
coordinates — no dummy/synthetic images. Measures both pipeline latency AND
geolocation accuracy (distance error in km from ground truth).
"""
import os
import sys
import time
import json
import shutil
import math
from pathlib import Path
from functools import wraps
from PIL import Image

# Add parent directory to path to import geovision_deep_scan
sys.path.insert(0, str(Path(__file__).parent.parent))

import geovision_deep_scan

# Real test set: (title, ground-truth lat, lon) — real geotagged Commons files
# (titles verified via Commons search API)
REAL_TEST_IMAGES = [
    # Toronto City Hall — real photo, known coords
    ("File:Toronto city hall August 2017 01.jpg", 43.6532, -79.3832),
    # Eiffel Tower — real photo, known coords
    ("File:Tour Eiffel Wikimedia Commons.jpg", 48.8584, 2.2945),
    # Golden Gate Bridge — real photo, known coords
    ("File:SF Golden Gate Bridge splash CA.jpg", 37.8199, -122.4783),
]

# Dictionary to store benchmark results
benchmark_results = {}
current_image = None
original_phases = {}

THUMB_URL = "https://commons.wikimedia.org/w/thumb.php?f={fname}&width=512"
UA = {"User-Agent": "GeoVision-Benchmark/1.0 (benchmark suite)"}


def fetch_real_test_images(out_dir="benchmark_images"):
    """Download REAL geotagged photographs from Wikimedia Commons as the test set."""
    import requests
    from urllib.parse import quote
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for title, lat, lon in REAL_TEST_IMAGES:
        fname = title[len("File:"):]
        local = Path(out_dir) / (fname.replace("/", "_") + ".bench.jpg")
        if not local.exists():
            url = THUMB_URL.format(fname=quote(fname))
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            local.write_bytes(r.content)
        paths.append((str(local), title, lat, lon))
    return paths


def time_phase(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        phase_name = func.__name__
        if current_image:
            if current_image not in benchmark_results:
                benchmark_results[current_image] = {}
            benchmark_results[current_image][phase_name] = duration

        return result
    return wrapper


def setup_benchmark():
    for name in dir(geovision_deep_scan):
        if name.startswith("phase") and callable(getattr(geovision_deep_scan, name)):
            func = getattr(geovision_deep_scan, name)
            original_phases[name] = func
            setattr(geovision_deep_scan, name, time_phase(func))


def teardown_benchmark():
    for name, func in original_phases.items():
        setattr(geovision_deep_scan, name, func)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def run_benchmark():
    global current_image

    test_images = fetch_real_test_images()
    setup_benchmark()

    total_times = {}
    accuracy_results = {}
    try:
        for img, title, gt_lat, gt_lon in test_images:
            current_image = os.path.basename(img)
            benchmark_results[current_image] = {}

            print(f"Running benchmark on {current_image}...")
            start_total = time.time()

            options = {
                "output_dir": "./benchmark_reports",
                "near_park": False,
                "region": None,
                "interactive": False,
                "verbose": False
            }

            result = geovision_deep_scan.run_pipeline(img, options)
            total_times[current_image] = time.time() - start_total

            # Accuracy: distance from best estimate to ground truth
            best = result.best_estimate if isinstance(result.best_estimate, dict) else {}
            if best.get("latitude") is not None and best.get("longitude") is not None:
                err = haversine_km(best["latitude"], best["longitude"], gt_lat, gt_lon)
                accuracy_results[current_image] = {
                    "ground_truth": {"lat": gt_lat, "lon": gt_lon, "title": title},
                    "estimate": {"lat": best["latitude"], "lon": best["longitude"]},
                    "error_km": round(err, 2),
                    "within_25km": err <= 25,
                    "within_200km": err <= 200,
                    "within_750km": err <= 750,
                }
            else:
                accuracy_results[current_image] = {
                    "ground_truth": {"lat": gt_lat, "lon": gt_lon, "title": title},
                    "error_km": None,
                    "note": "pipeline produced no location estimate",
                }

    finally:
        teardown_benchmark()

    return total_times, accuracy_results


def generate_report(total_times, accuracy_results):
    print("\n" + "=" * 50)
    print("GEOVISION BENCHMARK REPORT (real images, real ground truth)")
    print("=" * 50)

    report_data = {
        "summary": {},
        "per_image": benchmark_results,
        "accuracy": accuracy_results,
        "total_times": total_times
    }

    all_phases = set()
    for img, phases in benchmark_results.items():
        all_phases.update(phases.keys())

    averages = {}
    for phase in sorted(all_phases):
        times = [benchmark_results[img].get(phase, 0) for img in benchmark_results]
        if times:
            averages[phase] = sum(times) / len(times)

    avg_total = sum(total_times.values()) / len(total_times) if total_times else 0

    # Accuracy summary at GeoSpy-style thresholds
    valid = [a for a in accuracy_results.values() if a.get("error_km") is not None]
    summary_acc = {
        "n": len(accuracy_results),
        "n_with_estimate": len(valid),
        "within_25km": sum(1 for a in valid if a["within_25km"]),
        "within_200km": sum(1 for a in valid if a["within_200km"]),
        "within_750km": sum(1 for a in valid if a["within_750km"]),
        "median_error_km": sorted(a["error_km"] for a in valid)[len(valid) // 2] if valid else None,
    }

    report_data["summary"] = {
        "average_phase_times": averages,
        "average_total_time": avg_total,
        "accuracy": summary_acc,
    }

    # Print Markdown Table
    print("\n### Average Phase Latency\n")
    print("| Phase | Average Time (s) |")
    print("|-------|------------------|")
    for phase, avg_t in sorted(averages.items()):
        print(f"| {phase} | {avg_t:.4f} |")
    print(f"| **Total Pipeline** | **{avg_total:.4f}** |")

    print("\n### Geolocation Accuracy (real ground truth)\n")
    print("| Image | Error (km) | ≤25km | ≤200km | ≤750km |")
    print("|-------|-----------:|:-----:|:------:|:------:|")
    for name, a in accuracy_results.items():
        if a.get("error_km") is not None:
            print(f"| {name} | {a['error_km']:.1f} | {'✓' if a['within_25km'] else '✗'} | "
                  f"{'✓' if a['within_200km'] else '✗'} | {'✓' if a['within_750km'] else '✗'} |")
        else:
            print(f"| {name} | n/a | - | - | - |")

    # Save JSON report
    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nDetailed JSON report saved to {report_path}")


if __name__ == "__main__":
    total_times, accuracy_results = run_benchmark()
    generate_report(total_times, accuracy_results)

    # Cleanup
    if os.path.exists("benchmark_images"):
        shutil.rmtree("benchmark_images")
    if os.path.exists("benchmark_reports"):
        shutil.rmtree("benchmark_reports")

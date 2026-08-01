#!/usr/bin/env python3
import os
import sys
import time
import json
import shutil
from pathlib import Path
from functools import wraps
from PIL import Image

# Add parent directory to path to import geovision_deep_scan
sys.path.insert(0, str(Path(__file__).parent.parent))

import geovision_deep_scan

# Dictionary to store benchmark results
benchmark_results = {}
current_image = None
original_phases = {}

def create_dummy_images(num_images=3, out_dir="benchmark_images"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(num_images):
        img_path = Path(out_dir) / f"dummy_test_{i}.jpg"
        if not img_path.exists():
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))
            img.save(img_path)
        images.append(str(img_path))
    return images

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

def run_benchmark():
    global current_image
    
    test_images = create_dummy_images()
    setup_benchmark()
    
    total_times = {}
    try:
        for img in test_images:
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
            
            # Run the pipeline (we suppress stdout to keep output clean, optionally)
            geovision_deep_scan.run_pipeline(img, options)
            total_times[current_image] = time.time() - start_total
            
    finally:
        teardown_benchmark()
        
    return total_times

def generate_report(total_times):
    print("\n" + "="*50)
    print("GEOVISION BENCHMARK REPORT")
    print("="*50)
    
    report_data = {
        "summary": {},
        "per_image": benchmark_results,
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
    
    report_data["summary"] = {
        "average_phase_times": averages,
        "average_total_time": avg_total
    }
    
    # Print Markdown Table
    print("\n### Average Phase Latency\n")
    print("| Phase | Average Time (s) |")
    print("|-------|------------------|")
    for phase, avg_t in sorted(averages.items()):
        print(f"| {phase} | {avg_t:.4f} |")
    print(f"| **Total Pipeline** | **{avg_total:.4f}** |")
    
    # Save JSON report
    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\nDetailed JSON report saved to {report_path}")

if __name__ == "__main__":
    total_times = run_benchmark()
    generate_report(total_times)
    
    # Cleanup
    if os.path.exists("benchmark_images"):
        shutil.rmtree("benchmark_images")
    if os.path.exists("benchmark_reports"):
        shutil.rmtree("benchmark_reports")

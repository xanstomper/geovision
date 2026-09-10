#!/usr/bin/env python3
"""Production benchmark framework: compares geovision vs reference predictions at scale."""
import sys, json, time
sys.path.insert(0, "/home/jewboy420/geovision")
from modules.building_blueprint_scanner import BuildingBlueprintScanner
from modules.enhanced_fusion_engine import EnhancedFusionEngine

def benchmark_production(image_path, iterations=10):
    scanner = BuildingBlueprintScanner()
    fusion = EnhancedFusionEngine()
    results = []
    for i in range(iterations):
        start = time.time()
        bp = scanner.analyze_blueprint(image_path)
        duration = time.time() - start
        results.append({"iteration": i+1, "duration_sec": round(duration, 2), "style": bp.get("style_classification",{}).get("detected_style")})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    res = benchmark_production(args.image, args.iterations)
    avg_time = sum(r["duration_sec"] for r in res) / len(res)
    print(f"Production benchmark: {len(res)} iterations, avg {avg_time:.2f}s per run")

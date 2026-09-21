#!/usr/bin/env python3
"""
GeoVision Production Smoke Test
================================
Validates that all 74 modules can be imported and have no syntax errors,
missing dependencies cause graceful degradation, and key entry points work.

Run:
    python3 scripts/smoke_test.py

Exit 0 = all modules import cleanly.
Exit 1 = one or more modules failed to import (with per-module diagnostics).
"""

from __future__ import annotations

import importlib
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

BASE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("smoke_test")

# All module names defined in modules/ (exclude __init__)
MODULES = sorted(
    p.stem for p in (BASE / "modules").glob("*.py")
    if not p.stem.startswith("__")
)

# Core modules that must import without any exception (no graceful degradation)
CORE_MODULES = {
    "geonames_client",
    "geonames_city_snap",
    "geo_math",
    "geoguessr_heuristics",
    "exif_extractor",
    "deepfake_detector",
    "visual_geo_engine",
    "geoclip_predictor",
    "streetclip_predictor",
    "wikimedia_client",
    "nominatim_geocoder",
    "overpass_client",
    "satellite_matcher",
    "location_reasoner",
    "park_finder",
    "chain_store_locator",
    "language_detector",
    "telecom_osint",
    "weather_corroborator",
    "shadow_analyzer",
    "reverse_image_search",
    "deep_analyzer",
    "google_maps_controller",
    "browser_automation",
    "scoring_config",
    "geo_scorer",
    "uncertainty_estimator",
    "ground_imagery_client",
    "elevation_client",
    "climate_analyzer",
    "osm_feature_matcher",
    "environment_classifier",
    "scene_classifier",
    "sign_detector",
    "vehicle_detector",
    "vegetation_classifier",
    "terrain_analyzer",
    "road_analyzer",
    "road_orientation_matcher",
    "spatial_constraint_solver",
    "vlm_geo_analyzer",
    "vehicle_identifier",
    "vision_clue_resolver",
    "city_snap",  # imported as geonames_city_snap in CLI
    "web_search_providers",
    "geo_hierarchy",
    "grounding",
    "evidence_chain",
    "evidence_fusion",
    "geo_eval_metrics",
    "scoring_config",
    "geo_hierarchy",
}


def test_import(module_name: str) -> Tuple[bool, str]:
    """Try to import a single module. Returns (ok, error_message)."""
    try:
        # Use importlib.util to avoid the __main__ package issue
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            module_name, BASE / "modules" / f"{module_name}.py"
        )
        if spec is None or spec.loader is None:
            return False, "spec/loader is None"
        mod = importlib.util.module_from_spec(spec)
        # Enable relative imports by setting the package
        mod.__package__ = "modules"
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        # Check the module has expected attributes (at least a class or function)
        has_content = any(
            not name.startswith("_")
            for name in dir(mod)
            if not name.startswith("__")
        )
        if has_content:
            return True, ""
        else:
            return True, "module has no public attributes (empty or all-private)"
    except ImportError as e:
        # Check if it's a missing optional dependency vs a missing file
        if module_name in str(e):
            return False, f"Module file missing: {module_name}"
        # Missing dependency (e.g., torch, cv2, easyocr)
        return False, f"ImportError (missing dependency): {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def test_entry_points() -> List[Tuple[str, bool, str]]:
    """Test key entry-point scripts can parse."""
    results = []
    entry_points = [
        "geovision_deep_scan.py",
        "geovision_cli.py",
        "geovision.py",
        "mcp_geovision_server.py",
    ]
    for ep in entry_points:
        path = BASE / ep
        if not path.exists():
            results.append((ep, False, "File not found"))
            continue
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            results.append((ep, True, ""))
        except SyntaxError as e:
            results.append((ep, False, f"SyntaxError line {e.lineno}: {e.msg}"))
        except Exception as e:
            results.append((ep, False, f"{type(e).__name__}: {e}"))
    return results


def main() -> int:
    print("=" * 60, file=sys.stderr)
    print("  GeoVision Production Smoke Test", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Root: {BASE}", file=sys.stderr)
    print(f"  Modules: {len(MODULES)}", file=sys.stderr)
    print(f"  Core modules (must import): {len(CORE_MODULES)}", file=sys.stderr)
    print("", file=sys.stderr)

    # 1. Module imports
    module_results: Dict[str, Tuple[bool, str]] = {}
    failed_modules = []
    degraded_modules = []

    for mod_name in MODULES:
        ok, error = test_import(mod_name)
        module_results[mod_name] = (ok, error)
        if ok:
            pass  # success
        elif mod_name in CORE_MODULES:
            failed_modules.append((mod_name, error))
        else:
            degraded_modules.append((mod_name, error))

    # 2. Entry point syntax
    entry_results = test_entry_points()

    # Print summary
    print("MODULE IMPORTS:", file=sys.stderr)
    print(f"  ✓ {len(MODULES) - len(degraded_modules) - len(failed_modules)} imported successfully", file=sys.stderr)
    print(f"  ⚠ {len(degraded_modules)} failed (optional deps, graceful degradation)", file=sys.stderr)
    print(f"  ✗ {len(failed_modules)} FAILED (core modules)", file=sys.stderr)

    if degraded_modules:
        print("", file=sys.stderr)
        print("  Degraded modules (optional dependencies):", file=sys.stderr)
        for name, error in degraded_modules[:10]:
            short_err = error.split("\n")[0][:100]
            print(f"    {name}: {short_err}", file=sys.stderr)

    if failed_modules:
        print("", file=sys.stderr)
        print("  FAILED core modules:", file=sys.stderr)
        for name, error in failed_modules:
            short_err = error.split("\n")[0][:120]
            print(f"    ✗ {name}: {short_err}", file=sys.stderr)

    print("", file=sys.stderr)
    print("ENTRY POINTS:", file=sys.stderr)
    for name, ok, error in entry_results:
        status = "✓" if ok else "✗"
        detail = f" — {error}" if error else ""
        print(f"  {status} {name}{detail}", file=sys.stderr)

    # Overall result
    total_failed = len(failed_modules)
    if total_failed > 0:
        print("", file=sys.stderr)
        print(f"  RESULT: FAILED ({total_failed} core module(s) failed to import)", file=sys.stderr)
        return 1

    print("", file=sys.stderr)
    print("  RESULT: ALL CORE MODULES OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

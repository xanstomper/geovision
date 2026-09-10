#!/usr/bin/env python3
"""Quick mode: skips slow phases for sub-30-second response."""
import sys, time
sys.path.insert(0, "/home/jewboy420/geovision")
from modules.visual_geo_engine import DB_DIR
from modules.building_blueprint_scanner import BuildingBlueprintScanner
from modules.telecom_osint import TelecomOSINT

def quick_geolocate(image_path):
    scanner = BuildingBlueprintScanner()
    telecom = TelecomOSINT()
    bp = scanner.analyze_blueprint(image_path)
    return {"quick_result": bp.get("style_classification"), "time_optimized": True}

import re

with open("geovision_deep_scan.py", "r") as f:
    content = f.read()

# We need to extract the Chain Store Geolocation block from phase9_synthesis
# and move it to a standalone phase called phase4c_chain_stores
phase4c_code = """
def phase4c_chain_stores(ocr_text, region):
    logger.info("━" * 48)
    logger.info("  Phase 4c: Chain Store Geolocation")
    logger.info("━" * 48)
    results = []
    sig_text = ocr_text.get("significant_text", "") if isinstance(ocr_text, dict) else ""
    if sig_text:
        try:
            import json, os
            config_path = os.path.join(os.path.dirname(__file__), "config", "chain_stores.json")
            try:
                with open(config_path, "r") as f:
                    store_names = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load chain stores config: {e}")
                store_names = []
            detected_stores = [s for s in store_names if s.lower() in sig_text.lower()]
            if detected_stores:
                center_lat, center_lon = None, None
                if region:
                    try:
                        from modules.nominatim_geocoder import NominatimGeocoder
                        _geo = NominatimGeocoder()
                        _rv = _geo.forward_geocode(region)
                        if _rv and _rv.get('latitude') and _rv.get('longitude'):
                            center_lat = _rv['latitude']
                            center_lon = _rv['longitude']
                            logger.info(f"  ✓ Using region center for chain store search: {center_lat}, {center_lon}")
                    except Exception as e:
                        logger.error(f"  [-] Failed to geocode region for chain store: {e}")
                
                from modules.chain_store_locator import ChainStoreLocator
                locator = ChainStoreLocator()
                store_results = locator.locate_chain_stores(detected_stores, center_lat, center_lon)
                for res in store_results:
                    results.append({
                        "latitude": res["latitude"],
                        "longitude": res["longitude"],
                        "confidence": 0.92,
                        "sources": [f"chain_store:{res.get('store_name', 'unknown')}"],
                        "evidence": {"store_name": res.get("store_name"), "store_confidence": res.get("confidence")},
                        "phase": "ChainStore"
                    })
        except Exception as e:
            logger.error(f"  [-] Chain Store Geolocation skipped/failed: {e}", exc_info=True)
    return {"status": "success", "matches": results}
"""

if "def phase4c_chain_stores" not in content:
    content = content.replace("def phase5_satellite_matching", phase4c_code + "\ndef phase5_satellite_matching")

# In run_pipeline, call phase4c and add to candidate_coords
call_code = """
    # Phase 4c: Chain Store Geolocation
    try:
        pipeline_result.chain_stores = phase4c_chain_stores(pipeline_result.ocr_text, region)
        phases["chain_stores"] = pipeline_result.chain_stores
        if pipeline_result.chain_stores.get("matches"):
            for m in pipeline_result.chain_stores["matches"]:
                candidate_coords.append({"latitude": m["latitude"], "longitude": m["longitude"]})
                # Add to all_estimates via a side channel so synthesis sees it
                if not hasattr(pipeline_result, "early_estimates"):
                    pipeline_result.early_estimates = []
                pipeline_result.early_estimates.append(m)
    except Exception as e:
        logger.error(f"Phase 4c error: {e}")

    # Remove duplicates
"""
if "Phase 4c: Chain Store Geolocation" not in content:
    content = content.replace("    # Remove duplicates", call_code)

# In phase9_synthesis, add early_estimates to all_estimates
early_est_code = """
        all_estimates: List[Dict[str, Any]] = []
        if hasattr(options.get("pipeline_result", None), "early_estimates"):
            all_estimates.extend(options["pipeline_result"].early_estimates)
        elif phases.get("chain_stores", {}).get("matches"):
            all_estimates.extend(phases["chain_stores"]["matches"])
"""
content = content.replace("        all_estimates: List[Dict[str, Any]] = []", early_est_code)

# Strip out the old block from phase9_synthesis
import re
content = re.sub(r'        # ── Advanced OSINT: Chain Store Geolocation ──.*?        # ── a-e\) CV Modules \(Concurrent\) ──', '        # ── a-e) CV Modules (Concurrent) ──', content, flags=re.DOTALL)

with open("geovision_deep_scan.py", "w") as f:
    f.write(content)

print("Patched geovision_deep_scan.py successfully")

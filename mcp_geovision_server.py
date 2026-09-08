#!/usr/bin/env python3
"""
GeoVision MCP Server — agent-invocable geolocation tool
=========================================================
Standards-based MCP server (JSON-RPC over stdio) exposing the FULL
GeoVision pipeline as tools any AI agent can call.

The killer tool: `geolocate_image` — give it an image path, get back
coordinates, confidence, evidence chain, and reasoning. An agent with
computer vision receives the image (or a path to it), calls this tool,
and returns the located result to the user. No human interaction needed.

Register with any MCP client (Claude Desktop, Cursor, Hermes, custom):
  {"mcpServers": {"geovision": {"command": "python3", "args": ["/path/to/geovision/mcp_geovision_server.py"]}}}

Also exposes lean sub-tools for agents that want staged reasoning:
  geolocate_quick  — models only (GeoCLIP + StreetCLIP + CLIP-NN), fast
  ocr_extract      — text extraction
  reverse_geocode  — coords -> address
  verify_location  — visual verification of a candidate coordinate
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# GeoVision root = this file's directory
BASE = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE))

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("geovision-mcp")

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_geolocate_image(args: Dict[str, Any]) -> Dict[str, Any]:
    """Full GeoVision deep scan: every phase, fused estimates, evidence."""
    image_path = args.get("image_path") or args.get("imagePath")
    if not image_path:
        return {"error": "image_path required"}
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        return {"error": f"image not found: {p}"}

    from geovision_deep_scan import run_pipeline

    options = {
        "output_dir": str(BASE / "reports" / "mcp"),
        "near_park": bool(args.get("near_park", False)),
        "region": args.get("region"),
        "interactive": False,
        "verbose": False,
        "no_vlm": not _vlm_configured(),
    }
    result = run_pipeline(str(p), options)

    out: Dict[str, Any] = {
        "success": result.success,
        "phases_completed": result.phases_completed,
        "phases_failed": result.phases_failed,
        "location_estimates": [
            {
                "latitude": e.get("latitude"),
                "longitude": e.get("longitude"),
                "confidence": e.get("confidence"),
                "sources": e.get("sources", []),
                "phase": e.get("phase", ""),
                "evidence": e.get("evidence", {}),
            }
            for e in (result.location_estimates or [])[:5]
        ],
        "best_estimate": result.best_estimate if isinstance(result.best_estimate, dict) else {},
        "errors": result.errors,
        "report_json": None,
        "report_html": None,
    }
    # Report paths (pipeline writes timestamped files)
    try:
        reports = sorted((BASE / "reports" / "mcp").glob("geovision_scan_*.json"))
        if reports:
            out["report_json"] = str(reports[-1])
            out["report_html"] = str(reports[-1].with_suffix(".html"))
    except Exception:
        pass
    return out


def tool_geolocate_quick(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fast path: GeoCLIP direct GPS regression + instant GeoNames city snap.
    Single model, no web calls, no 94-prompt zero-shot — genuinely quick
    (~2-4 min cold on CPU, seconds warm)."""
    image_path = args.get("image_path") or args.get("imagePath")
    if not image_path:
        return {"error": "image_path required"}
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        return {"error": f"image not found: {p}"}

    out: Dict[str, Any] = {"image": str(p)}
    try:
        from modules.geoclip_predictor import GeoCLIPPredictor
        gc = GeoCLIPPredictor().locate(str(p), top_k=3)
        out["geoclip"] = gc
    except Exception as e:
        out["geoclip"] = {"status": "failed", "error": str(e)}

    # Fast GeoGuessr CV heuristics (<100ms)
    try:
        from modules.geoguessr_heuristics import GeoGuessrAnalyzer
        gh = GeoGuessrAnalyzer().analyze(str(p))
        out["geoguessr_heuristics"] = gh
        out["forensic_clues"] = gh.get("forensic_clues", [])
    except Exception as e:
        out["geoguessr_heuristics"] = {"status": "failed", "error": str(e)}

    # Cached StreetCLIP country classification (~1s)
    try:
        from modules.streetclip_predictor import StreetCLIPPredictor
        sc = StreetCLIPPredictor().locate(str(p), top_k=3)
        out["streetclip"] = sc
    except Exception as e:
        out["streetclip"] = {"status": "failed", "error": str(e)}

    estimates = []
    gc_ests = out.get("geoclip", {}).get("estimates", [])
    if gc_ests:
        top = gc_ests[0]
        est = {
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "confidence": top["confidence"],
            "source": "geoclip",
        }
        # Instant grounding: nearest real city via GeoNames (no API call)
        try:
            from modules.geonames_city_snap import get_city_index
            city = get_city_index().snap(top["latitude"], top["longitude"])
            if city:
                est["nearest_city"] = city["name"]
                est["city_country"] = city["country_code"]
                est["city_distance_km"] = city["distance_km"]
        except Exception:
            pass
        estimates.append(est)
    out["estimates"] = estimates
    return out


def tool_geoguessr_heuristics(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fast computer vision classifiers for driving side, road markings,
    utility poles, license plates, and soil/vegetation biome (<100ms, offline)."""
    image_path = args.get("image_path") or args.get("imagePath")
    if not image_path:
        return {"error": "image_path required"}
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        return {"error": f"image not found: {p}"}
    from modules.geoguessr_heuristics import GeoGuessrAnalyzer
    return GeoGuessrAnalyzer().analyze(str(p))


def tool_reverse_image_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated reverse image search: searches Wikimedia Commons, Wikipedia,
    and open web OSINT for matching landmarks and geographic entities."""
    image_path = args.get("image_path") or args.get("imagePath")
    if not image_path:
        return {"error": "image_path required"}
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        return {"error": f"image not found: {p}"}
    from modules.reverse_image_search import ReverseImageSearcher
    return ReverseImageSearcher().search_similar_images(str(p))


def tool_ocr_extract(args: Dict[str, Any]) -> Dict[str, Any]:
    image_path = args.get("image_path") or args.get("imagePath")
    if not image_path:
        return {"error": "image_path required"}
    from geovision_deep_scan import phase2_ocr
    return phase2_ocr(str(Path(image_path).expanduser().resolve()))


def tool_reverse_geocode(args: Dict[str, Any]) -> Dict[str, Any]:
    lat, lon = args.get("lat"), args.get("lon")
    if lat is None or lon is None:
        return {"error": "lat and lon required"}
    from modules.nominatim_geocoder import NominatimGeocoder
    g = NominatimGeocoder()
    return g.reverse_geocode(float(lat), float(lon)) or {"error": "no result"}


def tool_verify_location(args: Dict[str, Any]) -> Dict[str, Any]:
    """Visually verify a candidate coordinate against the query image using
    real reference photos + StreetCLIP similarity."""
    image_path = args.get("image_path") or args.get("imagePath")
    lat, lon = args.get("lat"), args.get("lon")
    if not image_path or lat is None or lon is None:
        return {"error": "image_path, lat, lon required"}
    from modules.candidate_visual_verification import verify_candidates
    return verify_candidates(
        str(Path(image_path).expanduser().resolve()),
        [{"latitude": float(lat), "longitude": float(lon)}],
    )


def tool_search_web(args: Dict[str, Any]) -> Dict[str, Any]:
    """Web search OSINT (Serper/Brave/SearXNG — active if keys configured)."""
    query = args.get("query")
    if not query:
        return {"error": "query required"}
    from modules.web_search_providers import search_web, get_available_providers
    providers = [p.name for p in get_available_providers()]
    if not providers:
        return {"error": "no search providers configured (set SERPER_API_KEY, BRAVE_API_KEY, or SEARXNG_URL)"}
    return {"providers": providers, "results": search_web(query, num_results=int(args.get("limit", 10)))}


def tool_resolve_vision_clues(args: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge for cloud vision models: accepts visual observations (city/country hints,
    street names, OCR text, chain stores, amenities, driving side, park proximity)
    and executes deterministic GIS grounding, OSM searches, city snap, and satellite verification."""
    from modules.vision_clue_resolver import VisionClueResolver
    return VisionClueResolver().resolve(args)


def tool_city_snap(args: Dict[str, Any]) -> Dict[str, Any]:
    """Instant offline nearest-city lookup via the 70k GeoNames dataset."""
    lat, lon = args.get("lat"), args.get("lon")
    if lat is None or lon is None:
        return {"error": "lat and lon required"}
    max_km = float(args.get("max_distance_km", 75.0))
    k = int(args.get("k", 3))
    from modules.geonames_city_snap import get_city_index
    idx = get_city_index()
    return {
        "snapped_city": idx.snap(float(lat), float(lon), max_km=max_km),
        "nearest_cities": idx.nearest(float(lat), float(lon), k=k),
    }


def tool_forward_geocode(args: Dict[str, Any]) -> Dict[str, Any]:
    """Forward geocode a place name, street, or address to lat/lon (OSM Nominatim)."""
    query = args.get("query")
    if not query:
        return {"error": "query required"}
    from modules.nominatim_geocoder import NominatimGeocoder
    hit = NominatimGeocoder().forward_geocode(query)
    return hit or {"error": f"No geocoding results found for '{query}'"}


def tool_osm_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """Query OpenStreetMap Overpass for POIs, parks, road types, or building density."""
    lat, lon = args.get("lat"), args.get("lon")
    if lat is None or lon is None:
        return {"error": "lat and lon required"}
    lat, lon = float(lat), float(lon)
    radius_m = int(args.get("radius_m", 500))
    q_type = args.get("query_type", "pois").lower()

    if q_type == "parks":
        from modules.overpass_client import OverpassClient
        return {"parks": OverpassClient().find_nearby_parks(lat, lon, radius_meters=radius_m)}
    elif q_type == "roads":
        from modules.osm_feature_matcher import OSMFeatureMatcher
        return OSMFeatureMatcher().get_road_characteristics(lat, lon)
    elif q_type == "density":
        from modules.osm_feature_matcher import OSMFeatureMatcher
        return OSMFeatureMatcher().get_building_density(lat, lon, radius_m=radius_m)
    elif q_type == "amenity_search":
        name = args.get("name_query", "")
        if not name:
            return {"error": "name_query required for amenity_search"}
        from modules.overpass_client import OverpassClient
        return {"matches": OverpassClient().search_amenities_by_name(name, lat, lon, radius_meters=radius_m)}
    else:
        from modules.osm_feature_matcher import OSMFeatureMatcher
        return OSMFeatureMatcher().find_pois(lat, lon, radius_m=radius_m)


def tool_satellite_landcover(args: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect satellite tile (ESRI/ArcGIS World Imagery) at coordinates for vegetation
    density and landcover type (urban, suburban, rural)."""
    lat, lon = args.get("lat"), args.get("lon")
    if lat is None or lon is None:
        return {"error": "lat and lon required"}
    lat, lon = float(lat), float(lon)
    zoom = int(args.get("zoom", 18))
    from modules.satellite_matcher import SatelliteMatcher
    sm = SatelliteMatcher()
    tile = sm.fetch_satellite_tile(lat, lon, zoom=zoom)
    if tile is None:
        return {"error": f"Failed to fetch satellite tile for {lat}, {lon}"}
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = float(np.count_nonzero(mask) / (tile.shape[0] * tile.shape[1]))
    landcover = "urban_built_up"
    if green_ratio > 0.40:
        landcover = "rural_or_park"
    elif green_ratio > 0.15:
        landcover = "suburban"
    cache_file = sm.cache_dir / f"sat_{lat}_{lon}_{zoom}.jpg"
    return {
        "latitude": lat,
        "longitude": lon,
        "zoom": zoom,
        "classification": landcover,
        "green_ratio": round(green_ratio, 3),
        "tile_path": str(cache_file) if cache_file.exists() else None,
    }


def tool_sun_shadow_estimate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate solar declination, day of year, and estimated latitude band
    from a datetime string and optional shadow angle."""
    dt_str = args.get("datetime_str") or args.get("date") or args.get("datetime")
    if not dt_str:
        return {"error": "datetime_str required (e.g. '2024:05:15 14:30:00' or '2024-05-15 14:30:00')"}
    dt_str = str(dt_str).replace("-", ":")
    shadow_angle = args.get("shadow_angle_deg")
    if shadow_angle is not None:
        shadow_angle = float(shadow_angle)
    from modules.shadow_analyzer import ShadowAnalyzer
    sa = ShadowAnalyzer()
    res = sa.estimate_latitude_from_sun({"DateTimeOriginal": dt_str}, shadow_angle_deg=shadow_angle)
    return res or {"error": "Failed to parse datetime for solar analysis"}


def tool_weather_corroborate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch historical weather conditions from Open-Meteo archive for lat/lon and date (YYYY-MM-DD)."""
    lat, lon = args.get("lat"), args.get("lon")
    date_str = args.get("date")
    if lat is None or lon is None or not date_str:
        return {"error": "lat, lon, and date (YYYY-MM-DD) required"}
    from modules.weather_corroborator import WeatherCorroborator
    wc = WeatherCorroborator()
    data = wc.fetch_historical_weather(float(lat), float(lon), str(date_str))
    if not data:
        return {"error": f"No historical weather data available for {lat}, {lon} on {date_str}"}
    data["weather_description"] = wc.interpret_wmo_code(data.get("weather_code"))
    return data


def tool_elevation_lookup(args: Dict[str, Any]) -> Dict[str, Any]:
    """Query ground elevation in meters at specific coordinates."""
    lat, lon = args.get("lat"), args.get("lon")
    if lat is None or lon is None:
        return {"error": "lat and lon required"}
    from modules.elevation_client import ElevationClient
    return ElevationClient().get_elevation(float(lat), float(lon))


def tool_road_heading_match(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract road perspective heading from an image and/or match candidate coordinates against OSM highway azimuths."""
    from modules.road_orientation_matcher import RoadOrientationMatcher
    matcher = RoadOrientationMatcher()
    out = {}
    image_path = args.get("image_path")
    if image_path:
        out["image_road_perspective"] = matcher.extract_image_road_heading(str(Path(image_path).expanduser().resolve()))
    lat, lon = args.get("lat"), args.get("lon")
    heading = args.get("expected_heading_deg") or args.get("heading")
    if lat is not None and lon is not None:
        if heading is not None:
            out["candidate_alignment"] = matcher.match_candidate_road_alignment(
                float(lat), float(lon), expected_azimuth_deg=float(heading), radius_m=int(args.get("radius_m", 300))
            )
        else:
            out["osm_roads"] = matcher.fetch_osm_road_azimuths(float(lat), float(lon), radius_m=int(args.get("radius_m", 300)))
    return out


def _vlm_configured() -> bool:
    return bool(os.environ.get("GEOVISION_VLM_API_KEY")
                or os.environ.get("OPENCODE_ZEN_API_KEY"))


# ---------------------------------------------------------------------------
# MCP protocol (JSON-RPC 2.0 over stdio, line-delimited)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "geolocate_image",
        "description": "FULL geolocation deep-scan of an image: EXIF, OCR, CLIP/GeoCLIP/StreetCLIP models, "
                       "OSINT databases, satellite matching, evidence fusion. Returns ranked coordinates with "
                       "confidence and evidence. Use when a user asks to geolocate/find where a photo was taken.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
                "region": {"type": "string", "description": "Optional region hint to narrow search"},
                "near_park": {"type": "boolean", "description": "Enable park proximity analysis"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "geolocate_quick",
        "description": "FAST model-only geolocation (GeoCLIP + StreetCLIP + CLIP nearest-neighbor). "
                       "No web calls; seconds per image after first load. Good for quick triage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "ocr_extract",
        "description": "Extract visible text from an image (signs, storefronts, plates).",
        "inputSchema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "reverse_geocode",
        "description": "Convert lat/lon to a human-readable address (OSM Nominatim).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "verify_location",
        "description": "Visually verify a candidate lat/lon against a query image using real reference "
                       "photos and StreetCLIP similarity scoring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["image_path", "lat", "lon"],
        },
    },
    {
        "name": "search_web",
        "description": "Web search for OSINT (activates with SERPER_API_KEY / BRAVE_API_KEY / SEARXNG_URL).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "geoguessr_heuristics",
        "description": "Fast computer vision classifiers for GeoGuessr cues: driving side (left vs right), "
                       "road line color (yellow vs white), utility pole archetypes (wooden crossarms, "
                       "concrete ladder, holey poles), license plate format, and soil/canopy ecology.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "reverse_image_search",
        "description": "Un-gated reverse image search and geotagged web entity matching across Wikimedia Commons, "
                       "Wikipedia, and open OSINT endpoints (zero API keys required).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "resolve_vision_clues",
        "description": "Bridge for external AI vision models: resolves visual observations (city/country hints, "
                       "street names, OCR text, store chains, amenities, driving side, park proximity) into "
                       "grounded GPS coordinates using Nominatim, Overpass, GeoNames, and satellite verification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city_hint": {"type": "string", "description": "Observed or deduced city name"},
                "country_hint": {"type": "string", "description": "Observed or deduced country name or code"},
                "state_or_province": {"type": "string", "description": "Observed state, province, or region"},
                "street_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Street names or intersection clues spotted in the image"
                },
                "detected_text": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "OCR text or signs spotted in the image"
                },
                "chain_stores": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified brand or chain store names (e.g. Tim Hortons, Lawson, Target)"
                },
                "amenities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Amenity types (e.g. cafe, pharmacy, school, bank)"
                },
                "near_park": {"type": "boolean", "description": "True if a park or public green space is visible"},
                "driving_side": {"type": "string", "enum": ["left", "right"], "description": "Observed driving side"},
                "road_heading_deg": {"type": "number", "description": "Observed or calculated road compass azimuth in degrees (0-360)"},
                "radius_meters": {"type": "number", "description": "Search radius around city centroid (default 35000m)"}
            },
        },
    },
    {
        "name": "city_snap",
        "description": "Instant offline nearest-city lookup via the 70k GeoNames dataset. Snaps any lat/lon to "
                       "the nearest city with distance, country code, and population (no web calls).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "max_distance_km": {"type": "number", "default": 75.0, "description": "Maximum snap radius in km"},
                "k": {"type": "number", "default": 3, "description": "Number of nearest cities to return"}
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "forward_geocode",
        "description": "Convert an address, street name, landmark, or city name into exact coordinates and bounding box (OSM Nominatim).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Place name, street, or address to geocode"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "osm_query",
        "description": "Query OpenStreetMap infrastructure near coordinates: POIs/shops, nearby parks, road characteristics, "
                       "building density, or specific amenity search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "radius_m": {"type": "number", "default": 500, "description": "Search radius in meters"},
                "query_type": {
                    "type": "string",
                    "enum": ["pois", "parks", "roads", "density", "amenity_search"],
                    "default": "pois",
                    "description": "Type of OSM query to run"
                },
                "name_query": {"type": "string", "description": "Amenity name to search for if query_type is amenity_search"}
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "satellite_landcover",
        "description": "Inspect high-resolution satellite imagery (ArcGIS/ESRI World Imagery) at coordinates for vegetation "
                       "green ratio and landcover classification (urban, suburban, rural).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "zoom": {"type": "number", "default": 18, "description": "Tile zoom level (15-19)"}
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "sun_shadow_estimate",
        "description": "Estimate solar declination, day of year, and latitude constraints from an image date/time string and optional shadow angle.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "datetime_str": {"type": "string", "description": "Date/time string (e.g. '2024:05:15 14:30:00' or '2024-05-15')"},
                "shadow_angle_deg": {"type": "number", "description": "Estimated shadow angle in degrees if visible"}
            },
            "required": ["datetime_str"],
        },
    },
    {
        "name": "weather_corroborate",
        "description": "Retrieve historical weather conditions from Open-Meteo archive for lat/lon on a specific date (YYYY-MM-DD) "
                       "to corroborate or eliminate candidate locations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["lat", "lon", "date"],
        },
    },
    {
        "name": "elevation_lookup",
        "description": "Look up ground elevation in meters above sea level for any coordinates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"}
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "road_heading_match",
        "description": "Extract road perspective angle/heading from an image and/or match candidate coordinates against OpenStreetMap highway compass azimuths to pinpoint exact street segments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Optional image path to extract perspective road yaw/heading"},
                "lat": {"type": "number", "description": "Candidate latitude to check OSM highway azimuths"},
                "lon": {"type": "number", "description": "Candidate longitude to check OSM highway azimuths"},
                "expected_heading_deg": {"type": "number", "description": "Expected or observed road compass azimuth (0-360 or 0-180)"},
                "radius_m": {"type": "integer", "description": "Search radius around coordinates in meters (default: 300)"}
            },
        },
    },
]

TOOL_IMPLS = {
    "geolocate_image": tool_geolocate_image,
    "geolocate_quick": tool_geolocate_quick,
    "ocr_extract": tool_ocr_extract,
    "reverse_geocode": tool_reverse_geocode,
    "verify_location": tool_verify_location,
    "search_web": tool_search_web,
    "geoguessr_heuristics": tool_geoguessr_heuristics,
    "reverse_image_search": tool_reverse_image_search,
    "resolve_vision_clues": tool_resolve_vision_clues,
    "city_snap": tool_city_snap,
    "forward_geocode": tool_forward_geocode,
    "osm_query": tool_osm_query,
    "satellite_landcover": tool_satellite_landcover,
    "sun_shadow_estimate": tool_sun_shadow_estimate,
    "weather_corroborate": tool_weather_corroborate,
    "elevation_lookup": tool_elevation_lookup,
    "road_heading_match": tool_road_heading_match,
}


def rpc_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return rpc_result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "geovision", "version": "2.0.0"},
        })
    if method == "notifications/initialized":
        return {}  # notification — no response
    if method == "ping":
        return rpc_result(req_id, {})
    if method == "tools/list":
        return rpc_result(req_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return rpc_error(req_id, -32602, f"unknown tool: {name}")
        try:
            out = impl(args)
            return rpc_result(req_id, {
                "content": [{"type": "text", "text": json.dumps(out, default=str)}],
                "isError": bool(out.get("error")),
            })
        except Exception as e:
            logger.error("tool %s failed: %s\n%s", name, e, traceback.format_exc())
            return rpc_result(req_id, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })
    if method == "resources/list":
        return rpc_result(req_id, {"resources": []})
    if method == "prompts/list":
        return rpc_result(req_id, {"prompts": []})
    return rpc_error(req_id, -32601, f"method not found: {method}")


def main():
    logger.info("GeoVision MCP server starting (stdio)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp:  # notifications get no response
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
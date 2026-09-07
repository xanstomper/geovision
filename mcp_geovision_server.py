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
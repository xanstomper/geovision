#!/usr/bin/env python3
"""
GeoVision Python MCP Server
Standards-based MCP server using JSON-RPC over stdio
"""

import json
import sys
import subprocess
from pathlib import Path

TOOLS = {
    "analyze_image": {"description": "Analyze image for geo clues", "params": ["imagePath"]},
    "extract_text": {"description": "OCR extract text", "params": ["imagePath"]},
    "detect_building_style": {"description": "Detect building style and era", "params": ["imagePath"]},
    "analyze_vegetation": {"description": "Analyze vegetation/region", "params": ["imagePath"]},
    "search_location": {"description": "Vector search places", "params": ["query", "coords", "radius"]},
    "reverse_geocode": {"description": "Lat/lng to address", "params": ["lat", "lng"]},
    "find_nearby_parks": {"description": "Find parks within walking distance", "params": ["lat", "lng", "walkMinutes"]},
    "calculate_walking_distance": {"description": "Walking dist/time between points", "params": ["startLat", "startLng", "endLat", "endLng"]},
    "open_google_maps": {"description": "Open Google Maps with coords/query", "params": ["query", "lat", "lng", "zoom"]},
    "get_current_tab": {"description": "Get current browser tab", "params": []},
    "navigate_map": {"description": "Navigate map to viewport", "params": ["lat", "lng", "zoom"]},
    "take_screenshot": {"description": "Take screenshot", "params": ["fullScreen", "outputPath"]},
    "move_mouse": {"description": "Move mouse", "params": ["x", "y"]},
    "click": {"description": "Click at position", "params": ["x", "y", "button"]},
    "type_text": {"description": "Type text", "params": ["text"]},
    "press_key": {"description": "Press keyboard keys", "params": ["keys"]},
    "geoguess_analyze": {"description": "Full geoguess pipeline", "params": ["imagePath", "description", "nearPark", "region"]},
}

BASE = Path(__file__).parent
PYTHON_TOOLS = {
    "analyze_image": f"cd {BASE} && python3 geovision.py",
    "extract_text": f"cd {BASE} && python3 geovision.py",
    "detect_building_style": f"cd {BASE} && python3 geovision.py",
    "analyze_vegetation": f"cd {BASE} && python3 geovision.py",
    "search_location": f"cd {BASE} && python3 geo_search.py",
    "reverse_geocode": f"cd {BASE} && python3 geo_search.py",
    "find_nearby_parks": f"cd {BASE} && python3 geo_search.py",
    "calculate_walking_distance": f"cd {BASE} && python3 geo_search.py",
}

def run_python_tool(tool, params):
    cmds = {
        "analyze_image": ["analyze_image", json.dumps({"path": params.get("imagePath")})],
        "extract_text": ["ocr", json.dumps({"path": params.get("imagePath")})],
        "search_location": ["search", json.dumps({"query": params.get("query"), "coords": params.get("coords"), "radius": params.get("radius", 1000)})],
        "reverse_geocode": ["geocode", json.dumps({"lat": params.get("lat"), "lng": params.get("lng")})],
        "find_nearby_parks": ["parks", json.dumps({"lat": params.get("lat"), "lng": params.get("lng"), "walk_minutes": params.get("walkMinutes", 10)})],
        "calculate_walking_distance": ["walk", json.dumps({"slat": params.get("startLat"), "slng": params.get("startLng"), "elat": params.get("endLat"), "elng": params.get("endLng")})],
    }
    args = cmds.get(tool)
    if not args:
        return {"error": f"No Python handler for {tool}"}
    try:
        result = subprocess.run(["python3"] + args, capture_output=True, text=True, timeout=30)
        return {"stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}
    except Exception as e:
        return {"error": str(e)}


def handle_request(request):
    method = request.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "geovision-python-mcp", "version": "1.0.0"},
            }
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": info["description"],
                        "inputSchema": {"type": "object", "properties": {k: {"type": "string"} for k in info["params"]}}
                    }
                    for name, info in TOOLS.items()
                ]
            }
        }
    if method == "tools/call":
        tool = request.get("params", {}).get("name")
        args = request.get("params", {}).get("arguments", {})
        py = run_python_tool(tool, args)
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"content": [{"type": "text", "text": json.dumps(py, indent=2)}]}
        }
    return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32601, "message": "Method not found"}}


def main():
    line = sys.stdin.readline()
    while line:
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
        line = sys.stdin.readline()

if __name__ == "__main__":
    main()

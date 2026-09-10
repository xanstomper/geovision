"""
Registry-integrity tests for the MCP server + CLI wiring (Sep 2026).

Permanently locks the wiring that ad-hoc runners kept re-verifying:
  - MCP TOOLS == TOOL_IMPLS (1:1), every tool registered, expected count
  - key listing/property/lookup tool names present
  - CLI subcommands register (investigate, grow-db, find-listing, track-properties)

These are cheap, offline, and catch wiring regressions on commit.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mcp_geovision_server as mcp

EXPECTED_TOOLS = {
    "geolocate_image", "geolocate_quick", "ocr_extract", "reverse_geocode",
    "verify_location", "search_web", "geoguessr_heuristics", "reverse_image_search",
    "resolve_vision_clues", "city_snap", "forward_geocode", "osm_query",
    "satellite_landcover", "sun_shadow_estimate", "weather_corroborate",
    "elevation_lookup", "road_heading_match", "environment_classify",
    "nearby_ground_imagery", "uncertainty_bounds", "investigate_image",
    "grow_reference_db", "query_listings",
}

CLI_COMMANDS = ["investigate", "grow-db", "find-listing", "track-properties"]


def test_mcp_tools_implements_are_one_to_one():
    names = [t["name"] for t in mcp.TOOLS]
    assert len(names) == len(mcp.TOOL_IMPLS)
    assert set(names) == set(mcp.TOOL_IMPLS)


def test_expected_tools_present():
    names = set(t["name"] for t in mcp.TOOLS)
    missing = EXPECTED_TOOLS - names
    assert not missing, f"missing tools: {missing}"


def test_cli_subcommands_register():
    """CLI subcommands appear in `geovision_cli.py --help` output (offline, no run)."""
    h = subprocess.run(
        [sys.executable, "geovision_cli.py", "--help"],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
        timeout=30)
    out = h.stdout.lower() if h.stdout else h.stderr.lower()
    for cmd in CLI_COMMANDS:
        assert cmd in out, f"CLI command {cmd!r} not registered"
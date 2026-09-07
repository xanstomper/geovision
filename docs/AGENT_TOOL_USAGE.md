# GeoVision as an Agent Tool (MCP)

GeoVision exposes its full geolocation pipeline as MCP tools, so ANY AI agent
(Claude Desktop, Cursor, Hermes, custom agents) with computer vision can
geolocate images by invoking GeoVision — the agent sees the image, reasons
about it, calls the tool, and returns located coordinates with evidence.

## Quick Start (any MCP client)

Register the server:

```json
{
  "mcpServers": {
    "geovision": {
      "command": "python3",
      "args": ["/home/jewboy420/geovision/mcp_geovision_server.py"]
    }
  }
}
```

Then the user says: **"use geovision and geolocate this image"** — the agent
calls `geolocate_image` with the image path and returns coordinates,
confidence, and the full evidence chain.

## Tools

| Tool | Speed | What it does |
|------|-------|--------------|
| `geolocate_image` | 2–10 min (full pipeline: EXIF, OCR, models, OSINT, satellite, fusion) | Complete deep-scan with ranked estimates + evidence |
| `geolocate_quick` | ~2–5 min cold, seconds warm (models only) | GeoCLIP direct GPS + StreetCLIP country + CLIP nearest-neighbor vs reference DB |
| `ocr_extract` | seconds | EasyOCR text extraction (signs, storefronts, plates) |
| `reverse_geocode` | ~1s | lat/lon → address (OSM Nominatim) |
| `verify_location` | ~10–30s | Visual verification of a candidate coordinate against the query image using real reference photos + StreetCLIP similarity |
| `search_web` | ~1s | OSINT web search (needs SERPER_API_KEY / BRAVE_API_KEY / SEARXNG_URL) |

## Model/API requirements (all optional — graceful degradation)

GeoVision needs NO API key for its core (models + OSM/Wikimedia/Overpass are
free). Optional keys unlock more:

| Env var | Unlocks |
|---------|---------|
| `GEOVISION_VLM_API_KEY` (+ `GEOVISION_VLM_BASE_URL`, `GEOVISION_VLM_MODEL`) | VLM reasoning phase (structured feature extraction + strict evidence-hierarchy reasoning). Any OpenAI-compatible vision endpoint. |
| `SERPER_API_KEY` / `BRAVE_API_KEY` / `SEARXNG_URL` | Web-search OSINT |
| `MAPILLARY_ACCESS_TOKEN` | Street-level imagery verification |
| `GOOGLE_MAPS_API_KEY` | Google Maps geocoding/Street View cross-check |

Model weights (CLIP ViT-B-32, GeoCLIP, StreetCLIP) auto-download on first
use (~1.2GB total) and cache locally. First call is slow (model load);
a persistent MCP server process keeps them warm.

## Architecture

```
User: "geolocate this image"
  └─ Agent (any vision model) ── sees image, saves path
       └─ MCP call: geolocate_image {image_path}
            └─ GeoVision pipeline (9+ phases, all real):
                 EXIF → OCR → ResNet50 → CLIP-NN vs 1.7k+ real geotagged
                 Commons photos → GeoCLIP GPS regression → StreetCLIP
                 country → Wikimedia/Overpass OSINT → satellite tiles →
                 evidence fusion (haversine clustering + confidence
                 calibration) → ranked estimates + evidence chain
       └─ Agent reads result: coords + confidence + evidence
  └─ Answer to user with map link + reasoning
```

Every estimate carries its sources (geoclip / clip_nn / ocr_osint / exif /
satellite / vlm) — full traceability, no invented numbers.

## Accuracy expectations (honest)

- Landmarks & distinctive scenes: GeoCLIP is near-exact (Eiffel Tower photo
  → 48.8584,2.2946 vs truth 48.8584,2.2945)
- Generic street scenes: city/region level accuracy; improves as the
  reference DB grows (`scripts/build_reference_db.py --append`)
- The strict confidence calibration (ported from open_geo_spy) caps
  confidence at 0.5 when there's no city-specific evidence — GeoVision
  tells the truth rather than guessing confidently

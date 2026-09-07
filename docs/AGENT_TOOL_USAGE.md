# GeoVision Agent Integration Guide (MCP & CLI)

GeoVision is an open-source, GeoSpy-class geospatial OSINT and computer vision engine designed specifically to be called by **any AI agent or cloud model with computer vision** (Claude 3.5 Sonnet, GPT-4o, Gemini Pro/Flash, Hermes, Cline, Cursor, Antigravity).

GeoVision requires **no internal LLM or API keys** to run. When an AI vision model inspects an image, the model performs visual perception (reading signs, identifying architecture, spotting store chains), while GeoVision executes deterministic geospatial grounding:
- Coordinate prediction via GeoCLIP ViT regression
- Zero-shot country classification via StreetCLIP (1.5s with precomputed text cache)
- Instant offline nearest-city snapping across 70k GeoNames cities
- Fast OpenCV GeoGuessr classifiers (<100ms: driving side, road marks, utility poles, license plates, soil/canopy)
- Overpass OpenStreetMap queries (POIs, parks, roads, building density)
- High-res satellite landcover verification (ArcGIS/ESRI World Imagery)
- Historical weather corroboration (Open-Meteo archive)
- Un-gated reverse image search (Wikimedia Commons / Wikipedia Geo APIs)

---

## 1. Quick Start: Register MCP Server

Register `mcp_geovision_server.py` with any MCP-compatible agent (Claude Desktop, Cursor, Hermes, Continue, Roo Code):

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

The MCP server communicates via JSON-RPC 2.0 over standard I/O and exposes **16 modular tools**.

---

## 2. MCP Tools Reference

### A. Core Geolocation
| Tool | Speed | Description |
|------|-------|-------------|
| `geolocate_image` | 1–3 min | **Full deep-scan pipeline**: EXIF, OCR, GeoCLIP, StreetCLIP, 6.7k reference DB matching, OSM Overpass, satellite tiles, evidence fusion with ranked estimates. |
| `geolocate_quick` | ~1.5s warm | **Fast model triage**: GeoCLIP direct GPS + StreetCLIP country + instant GeoNames city snap + GeoGuessr heuristics. Zero web calls. |
| `resolve_vision_clues` | ~2–5s | **Cloud Vision Model Bridge**: Give it your visual observations (`city_hint`, `street_names`, `chain_stores`, `amenities`, `near_park`, `driving_side`), and GeoVision resolves grounded GPS coordinates with GIS proof. |

### B. Computer Vision & Heuristics
| Tool | Speed | Description |
|------|-------|-------------|
| `geoguessr_heuristics` | <100ms | OpenCV classifiers: driving side (left vs right), road markings (yellow vs white), utility poles (wooden, holey concrete, ladder), license plate aspect ratio, soil/canopy biome. |
| `ocr_extract` | ~1–2s | EasyOCR text extraction for road signs, shop fronts, vehicle plates. |
| `reverse_image_search` | ~2–4s | Un-gated reverse image search across Wikimedia Commons and Wikipedia Geo APIs for matching landmarks and geotagged web entities (zero API keys). |
| `verify_location` | ~5–10s | Visually verify candidate coordinates against reference photos using StreetCLIP cosine similarity. |

### C. Geospatial & Environmental Grounding
| Tool | Speed | Description |
|------|-------|-------------|
| `city_snap` | <10ms | Instant offline nearest-city lookup via 70k GeoNames dataset (`lat`, `lon`, `max_distance_km`). |
| `forward_geocode` | ~500ms | Convert place name, street, or address to exact lat/lon and bounding box (OSM Nominatim). |
| `reverse_geocode` | ~500ms | Convert lat/lon to human-readable street address. |
| `osm_query` | ~1–2s | Query OpenStreetMap infrastructure: POIs, nearby parks (`query_type="parks"`), road types, building density, or named amenity search. |
| `satellite_landcover` | ~1s | Fetch ESRI/ArcGIS World Imagery tile at coordinates; returns green vegetation ratio and landcover classification (`urban_built_up`, `suburban`, `rural_or_park`). |
| `weather_corroborate` | ~500ms | Check historical weather from Open-Meteo archive for `lat`, `lon`, and `date` (`YYYY-MM-DD`). |
| `sun_shadow_estimate` | <10ms | Compute solar declination, day of year, and estimated latitude band from timestamp and shadow angle. |
| `elevation_lookup` | ~500ms | Query elevation above sea level in meters. |
| `search_web` | ~1s | OSINT web search (activates if `SERPER_API_KEY`, `BRAVE_API_KEY`, or `SEARXNG_URL` is set). |

---

## 3. The Vision Model Bridge: `resolve_vision_clues`

When an AI model with vision (e.g. Claude 3.5 Sonnet or GPT-4o) receives a photo from a user, the model can inspect the image and call `resolve_vision_clues`:

```json
{
  "name": "resolve_vision_clues",
  "arguments": {
    "city_hint": "Toronto",
    "country_hint": "Canada",
    "street_names": ["Yonge St"],
    "chain_stores": ["Tim Hortons"],
    "near_park": true,
    "driving_side": "right"
  }
}
```

**GeoVision returns:**
```json
{
  "status": "success",
  "best_estimate": {
    "latitude": 43.782123,
    "longitude": -79.416194,
    "confidence": 0.92,
    "display_name": "Yonge Street, North York, Toronto, Ontario, Canada",
    "nearest_city": {"name": "Willowdale West", "country_code": "CA", "distance_km": 0.0},
    "satellite": {"classification": "urban_built_up", "green_ratio": 0.059, "verified": true},
    "nearby_parks": [{"name": "Hendon Park", "distance_m": 343}],
    "evidence": [
      "Street / intersection match for 'Yonge St'",
      "Corroborated: 35 park(s) within 1.5km (nearest: Hendon Park)",
      "Satellite landcover: urban_built_up (green ratio: 5.9%)"
    ],
    "google_maps_url": "https://www.google.com/maps?q=43.782123,-79.416194"
  }
}
```

---

## 4. CLI Subcommands (for Bash-calling Agents)

Coding agents with bash tools (Cline, Claude Code, Cursor, Antigravity, OpenCode) can invoke subcommands directly from the terminal. All subcommands support `-j` / `--json-only` for silent, pure JSON output pipeable directly to `jq`:

```bash
# 1. Fast triage (<2s)
python3 ~/geovision/geovision_cli.py quick /path/to/image.jpg -j | jq .

# 2. GeoGuessr CV heuristics (<100ms)
python3 ~/geovision/geovision_cli.py heuristics /path/to/image.jpg -j

# 3. Vision clue resolution (GIS grounding)
python3 ~/geovision/geovision_cli.py resolve --city "Toronto" --street "Yonge St" --near-park -j

# 4. Instant offline GeoNames city snap
python3 ~/geovision/geovision_cli.py snap --lat 48.8584 --lon 2.2945 -j

# 5. Forward and reverse geocoding
python3 ~/geovision/geovision_cli.py geocode "Eiffel Tower" -j
python3 ~/geovision/geovision_cli.py reverse-geocode --lat 48.8584 --lon 2.2945 -j

# 6. OpenStreetMap Overpass queries
python3 ~/geovision/geovision_cli.py osm --lat 48.8584 --lon 2.2945 --type parks --radius 1000 -j

# 7. Satellite landcover check
python3 ~/geovision/geovision_cli.py satellite --lat 48.8584 --lon 2.2945 -j

# 8. Historical weather check
python3 ~/geovision/geovision_cli.py weather --lat 48.8584 --lon 2.2945 --date 2024-05-01 -j

# 9. OCR text extraction
python3 ~/geovision/geovision_cli.py ocr /path/to/image.jpg -j

# 10. Full deep scan
python3 ~/geovision/geovision_cli.py scan /path/to/image.jpg --no-vlm -j
```

---

## 5. Confidence Calibration & Honesty Principles

GeoVision enforces strict confidence calibration (derived from `open_geo_spy` architecture):
- **0.85 – 0.98**: Grounded by specific named street entities, verified chain store addresses, or tight GeoCLIP spatial consensus ($\le 25\text{km}$ spread).
- **0.50 – 0.70**: Distinctive regional architecture, confirmed city-level match, or broad cluster agreement.
- **0.20 – 0.45**: Generic street scene, unconfirmed city centroid, or scattered model predictions.
- **Null OSM Matches**: Never assigned high confidence simply because "buildings exist nearby"; unverified generic infrastructure is labeled `generic_osm_nearby` and capped at 0.30 confidence.

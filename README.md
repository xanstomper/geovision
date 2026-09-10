<div align="center">

<img src="https://img.shields.io/badge/GeoVision-Intelligence%20Platform-00d4ff?style=for-the-badge&logo=satellite&logoColor=white" alt="GeoVision"/>

# GeoVision

**Multi-phase OSINT geolocation pipeline. Drop in an image, get a pinned coordinate.**

*Fuses computer vision, trained geolocation models, real-world OSINT APIs, and optional VLM reasoning into a single forensic-grade investigation stack.*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-412991?style=flat-square&logo=openai&logoColor=white)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Flask](https://img.shields.io/badge/Web%20UI-Flask%209999-000000?style=flat-square&logo=flask)](http://localhost:9999)
[![Status](https://img.shields.io/badge/Status-Active-00ff88?style=flat-square)]()

</div>

---

## What is GeoVision?

GeoVision is a self-hosted, open-source geolocation engine that identifies where a photo was taken using a 15-phase investigation pipeline — no GPS required. It works the same way a skilled OSINT analyst would: layering visual cues, text, infrastructure patterns, trained models, and external data sources until a location is either confirmed or eliminated.

It runs entirely on your machine. No data leaves. No third-party AI service sees your images unless you configure a VLM endpoint.

The pipeline fuses:
- **Computer vision** — driving side, road markings, license plate format, bollards, soil color, vegetation, architecture
- **OCR + language detection** — Unicode script family, area codes, ccTLDs, currency symbols, business signs
- **CLIP-based geolocation models** — direct GPS regression from image embeddings
- **Real-time OSINT APIs** — OpenStreetMap Overpass, Nominatim, Wikipedia, Open-Meteo historical weather
- **Deepfake/authenticity detection** — 6-technique forensic analysis before geolocation runs
- **Vehicle identification** — YOLO detection + VLM make/model identification, driving-side constraint
- **Case management** — multi-scan investigation tracking with notes, maps, and export

---

## Quick Start

```bash
git clone https://github.com/xanstomper/geovision.git
cd geovision
pip install -r requirements.txt

# Optional: set a VLM key for AI-assisted reasoning
echo "OPENCODE_ZEN_API_KEY=sk-your-key-here" >> .env

# Launch the web UI
python3 web/app.py
# → http://localhost:9999
```

For CLI use:
```bash
python3 geovision_cli.py /path/to/image.jpg
python3 geovision_cli.py /path/to/image.jpg --region "London" --no-vlm
python3 geovision_cli.py /path/to/image.jpg --verbose --near-park
```

---

## Pipeline Phases

GeoVision runs phases in a fixed sequence. Each phase injects candidates and constraints into a shared evidence pool that the Synthesis phase fuses into a ranked coordinate list.

| Phase | Name | What it does |
|---|---|---|
| **0** | EXIF Extraction | GPS, timestamps, device fingerprint from image metadata |
| **0b** | Authenticity Analysis | ELA, DCT frequency, metadata, noise, face consistency, compression ghost |
| **1b** | Shadow / Solar Analysis | Solar hemisphere estimation from shadow geometry and EXIF timestamp |
| **1** | Visual Features | Driving side, road markings, license plate format, bollards, architecture, soil |
| **2** | OCR | EasyOCR → Tesseract → PaddleOCR chain, significant-text filtering |
| **2b** | Vehicle Identification | YOLO detection, color, plate OCR, VLM make/model, driving-side constraint |
| **3** | Deep Features | ResNet50 embedding (or fast numpy LBP fallback) |
| **3b** | Reverse Image Search | Wikimedia / Wikipedia open entity matching |
| **3c** | Visual Geo Engine | CLIP ViT-B-32 → cosine nearest-neighbor on geotagged reference DB |
| **3d** | Trained Models | GeoCLIP direct GPS regression + StreetCLIP country classification |
| **4** | OSINT Matching | Wikipedia + Wikimedia coordinate lookups |
| **4b** | Property Records | OSM Overpass land-use / zoning validation |
| **4c** | Chain Store Locator | Brand name → Overpass → exact physical addresses |
| **5** | Satellite Matching | Terrain and feature correlation via OSM tiles |
| **6** | Park Proximity | Haversine-based park proximity scoring |
| **7** | Cross-View Verification | Google Maps URL generation for manual confirmation |
| **8b** | VLM Reasoning | Two-stage vision-language model: feature extraction → location prediction |
| **9** | Synthesis | Cluster fusion, spatial constraint solver, uncertainty estimation, final ranking |

Phases run in parallel where possible. Any phase that fails is skipped cleanly — the pipeline always produces output.

---

## Feature Breakdown

### Visual Geolocation (Core)

The primary signal when EXIF and OCR are empty is a CLIP ViT-B-32 nearest-neighbor search against a reference database of real geotagged photos pulled from Wikimedia Commons. Similarity scores are calibrated against spatial agreement of top-k neighbors — a weak CLIP match scores near 0, not a fake high number.

```bash
# Build the reference database
python3 scripts/build_reference_db.py --max-cities 300 --per-city 12 --min-population 300000

# Grow it later (appends, doesn't rebuild)
python3 scripts/build_reference_db.py --max-cities 600 --per-city 16 --append
```

### GeoGuessr-style Heuristics

Pure OpenCV analysis that runs in under 100ms with no internet connection:

- **Driving side** — lane detection + perspective geometry
- **Road line color** — yellow centerlines = North America; white = Europe/Asia
- **License plate morphology** — Euro narrow band vs US short plate vs Asian wide
- **Bollard detection** — Dutch, UK, French, generic archetypes
- **Road sign shapes** — octagon, inverted triangle, diamond detection
- **Soil & vegetation color** — red laterite, black chernozem, pale desert

### Trained Geolocation Models

- **GeoCLIP** (NeurIPS 2023) — aligns CLIP with a GPS positional encoder trained over a 100k-point world gallery. Outputs lat/lon coordinates with a calibrated confidence derived from top-k spatial spread, not raw gallery softmax.
- **StreetCLIP** — zero-shot country classification using a CLIP model fine-tuned on 1M street-level photos. Country hints feed into synthesis as soft priors.

### OSINT Modules

| Module | Data source | What it adds |
|---|---|---|
| `telecom_osint.py` | NANP + ITU databases | Area code → region with ~95% certainty |
| `language_detector.py` | Unicode script tables | Script family, ccTLD, currency symbol → country/region |
| `weather_corroborator.py` | Open-Meteo historical archive | Cross-references EXIF date vs historical weather at each candidate |
| `chain_store_locator.py` | OSM Overpass | Detected brand → exact store coordinates |
| `property_records_client.py` | OSM Overpass | Land-use / zoning confirmation |
| `nominatim_geocoder.py` | OSM Nominatim | Ocean-drop validation + full address resolution |
| `wikimedia_client.py` | Wikimedia / Wikipedia API | Landmark matching from OCR text |
| `shadow_analyzer.py` | Astronomical ephemeris | Solar hemisphere from shadow geometry |
| `road_orientation_matcher.py` | OSM highway graph | Road heading alignment at each candidate coordinate |

### Image Authenticity Detection

Runs before geolocation. Six techniques, no additional dependencies:

- **ELA (Error Level Analysis)** — detects splice boundaries and AI-uniform compression
- **DCT Frequency Analysis** — periodic grid artifacts characteristic of diffusion models
- **Metadata Inspection** — AI software tags (`Stable Diffusion`, `Midjourney`, `DALL-E`, `Adobe Firefly`, etc.), missing camera fields, AI-typical dimensions
- **PRNU Noise Analysis** — uniform noise pattern indicates AI generation
- **Face Consistency** — over-symmetry and edge softness from GAN artifacts
- **Compression Ghost** — absence of prior JPEG history from AI renders

Output: `LIKELY_GENUINE` / `SUSPICIOUS` / `LIKELY_MANIPULATED` / `LIKELY_AI_GENERATED` + per-technique breakdown shown in the web UI.

### Vehicle Identification

- YOLO v8n detection with automatic model download (~6MB)
- Per-vehicle color (HSV), type classification, license plate detection + OCR
- Optional VLM make/model/year identification using the configured API key
- Driving-side inference from vehicle position in frame — feeds into synthesis as a geolocation constraint

### Case Management

Track multiple images as part of a single investigation:

- Create named cases with description and tags
- Link scan results to cases with notes and timestamps
- Interactive Leaflet map showing all scan locations per case
- Export full case as JSON
- Web UI at `/cases`

---

## Web Interface

```bash
python3 web/app.py
```

Open `http://localhost:9999`. Features:
- Drag-and-drop multi-image upload with live thumbnail preview
- Real-time phase progress tracker (no raw log spam)
- Full-width dark Leaflet map with numbered candidate pins
- Best estimate card with resolved address, confidence %, Google Maps and Street View links
- Evidence matrix chips (country, road alignment verified, etc.)
- All candidates ranked with confidence bars and source tags
- Nearby ground photos from Wikimedia/Mapillary
- Image authenticity card with per-technique bars
- Vehicle analysis card with color swatches, type badges, plate text
- Cases nav link for investigation tracking

---

## Configuration

GeoVision works without any API keys. VLM reasoning (Phase 8b) and vehicle make/model identification are skipped cleanly if no key is set.

```bash
# .env or ~/.hermes/.env
OPENCODE_ZEN_API_KEY=sk-your-key-here  # enables VLM reasoning
GEOVISION_VLM_MODEL=opencode/gpt-5.4-nano  # default model
GEOVISION_VLM_BASE_URL=https://opencode.ai/zen/v1  # default endpoint
```

Any OpenAI-compatible endpoint works — Ollama, LM Studio, OpenRouter, etc.

```bash
# Example: use a local Ollama model
GEOVISION_VLM_BASE_URL=http://localhost:11434/v1
GEOVISION_VLM_API_KEY=ollama
GEOVISION_VLM_MODEL=llava:13b
```

---

## CLI Reference

```
usage: geovision_cli.py [OPTIONS] IMAGE

Arguments:
  IMAGE               Path to the image file to analyze

Options:
  --output-dir DIR    Output directory for reports (default: ./reports)
  --region TEXT       Narrow search to a specific region or city
  --near-park         Enable park proximity filtering
  --no-vlm            Skip VLM reasoning phase (fully offline mode)
  --interactive       Open Google Maps in browser after scan
  --verbose           Debug-level logging
  --json-only         Output raw JSON to stdout (for scripting)
  --help              Show this message and exit
```

---

## Output

Every scan writes two files to the output directory:

**`geovision_scan_<timestamp>.json`** — full forensic payload:
```json
{
  "best_estimate": {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "confidence": 0.87,
    "sources": ["geoclip:rank1", "visual_geo:clip_nn"],
    "evidence": { "country": "France", "city": "Paris", ... }
  },
  "location_estimates": [...],
  "uncertainty": { "uncertainty_radius_km": 12.4, "granularity": "city" },
  "deepfake_analysis": { "verdict": "LIKELY_GENUINE", "authenticity_score": 0.83 },
  "vehicle_analysis": { "vehicle_count": 2, "driving_side_evidence": "right", ... },
  ...
}
```

**`geovision_scan_<timestamp>.html`** — standalone interactive HTML report with map, ranked estimates table, phase status badges, OCR text, and park proximity.

---

## Requirements

```
Python 3.10+
opencv-python >= 4.8
torch >= 2.0, torchvision >= 0.15
ultralytics >= 8.0          # YOLO vehicle detection
open-clip-torch >= 2.24     # CLIP ViT-B-32
geoclip >= 0.1              # GeoCLIP GPS regression
transformers >= 4.30        # StreetCLIP
easyocr >= 1.7              # Primary OCR engine
flask >= 2.3                # Web UI
folium >= 0.14              # Interactive HTML maps
```

Full list in [`requirements.txt`](requirements.txt).

---

## Accuracy Notes

GeoVision is a fusion pipeline, not a magic box. Accuracy depends heavily on what the image contains:

- **Landmark or sign with identifiable text** → city-level or street-level accuracy
- **Distinctive infrastructure** (chain store, unique architecture) → city/district accuracy
- **Generic outdoor scene** → country or region-level, sometimes less
- **Featureless interior** → low confidence, wide uncertainty radius

The `uncertainty` field in the JSON output contains `uncertainty_radius_km` and a `granularity` label (`street` / `neighborhood` / `city` / `region` / `country`) computed from the spread and sources of the top estimates. Trust that more than the raw confidence number.

---

## Project Structure

```
geovision/
├── geovision_deep_scan.py     # Main pipeline (run_pipeline entry point)
├── geovision_cli.py           # Typer CLI wrapper
├── web/
│   ├── app.py                 # Flask server (port 9999)
│   └── templates/
│       ├── index.html         # Main scan UI
│       ├── cases.html         # Case list
│       └── case_detail.html   # Case detail + map
├── modules/
│   ├── visual_geo_engine.py   # CLIP nearest-neighbor geolocation
│   ├── deepfake_detector.py   # Image authenticity analysis
│   ├── vehicle_identifier.py  # YOLO + VLM vehicle ID
│   ├── case_manager.py        # SQLite case management
│   ├── vlm_geo_analyzer.py    # VLM two-stage reasoning
│   ├── geoguessr_heuristics.py# Driving side, road markings, plates
│   ├── chain_store_locator.py # Brand → OSM coordinates
│   ├── telecom_osint.py       # Area code → region
│   ├── language_detector.py   # Script/ccTLD/currency detection
│   ├── weather_corroborator.py# Historical weather cross-reference
│   └── ...                    # 50+ additional modules
├── core/
│   └── vision_engine.py       # OpenCV visual feature extraction
├── scripts/
│   └── build_reference_db.py  # Reference DB builder
├── data/
│   ├── visual_geo_db/         # CLIP reference embeddings
│   └── cases.db               # Case management database
└── requirements.txt
```

---

## Topics

`geolocation` `osint` `computer-vision` `image-forensics` `geospatial` `deepfake-detection` `clip` `yolo` `vehicle-detection` `flask` `pytorch` `openstreetmap` `intelligence` `investigative-tools` `image-analysis` `vlm` `geoclip` `streetclip` `self-hosted` `privacy`

---

<div align="center">

Built for investigators, researchers, and anyone who needs to know *where*.

</div>

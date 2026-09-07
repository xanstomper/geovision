# GeoVision AI - The Ultimate OSINT Geolocation Engine

GeoVision is an ultra-advanced, Open-Source Intelligence (OSINT) geolocation pipeline designed to match and exceed the capabilities of tools like GeoSpy. It transforms standard computer vision and AI into a logic-driven forensic investigator. 

Rather than relying purely on subjective AI guessing, GeoVision fuses **Vision-Language Models (VLMs)** with hard mathematical OSINT—cross-referencing historical weather records, telecommunication area codes, language scripts, road markings, satellite imagery, and municipal zoning data to pinpoint exact geographic coordinates from a single image.

All data sources and tools are 100% real. There is zero mock data, zero hardcoded fallback logic, and zero API hallucination.

## 🧠 Visual Geo Engine (GeoSpy-class core)

The core geolocation signal — the thing GeoSpy actually does — is **image → CLIP embedding → nearest-neighbor against a database of real geotagged photos**:

- **`modules/visual_geo_engine.py`** — embeds the query image with CLIP ViT-B-32 (LAION-2B, 151M params) and cosine-matches it against the reference DB. Produces location estimates with honest confidence derived from neighbor similarity + spatial agreement (no invented numbers — a weak match scores near 0).
- **`scripts/build_reference_db.py`** — builds the reference DB from **real, existing, free datasets**:
  - [GeoNames cities15000](https://download.geonames.org/export/dump/cities15000.zip) — the real 34k-city world gazetteer
  - [Wikimedia Commons geosearch API](https://commons.wikimedia.org/w/api.php) — real geotagged photographs at each city
  - Reference DB lands in `data/visual_geo_db/` (embeddings + per-photo metadata)

```bash
# Build the reference DB (real photos, real coordinates)
python3 scripts/build_reference_db.py --max-cities 300 --per-city 12 --min-population 300000
# Grow it later — appends without rebuilding
python3 scripts/build_reference_db.py --max-cities 600 --per-city 16 --append
```

This is what makes the pipeline produce location estimates **even when EXIF and OCR give nothing** — previously the pipeline collapsed to zero candidates on metadata-free images.

## 🔥 Advanced OSINT Arsenal (18-Module Pipeline)

GeoVision is equipped with cutting-edge investigative modules that typical AI tools lack:

- 🌦️ **Historical Weather Corroboration (`weather_corroborator.py`)**: Extracts the exact date/time from image EXIF data and queries the Open-Meteo Historical Archive API for all AI-candidate locations. If the AI guesses a town in Florida but the photo shows rain while historical records confirm clear skies, GeoVision slashes the confidence of that location.
- 📞 **Telecom Area Code Extraction (`telecom_osint.py`)**: Intercepts raw Optical Character Recognition (OCR) text *before* AI analysis. Uses Regex to hunt for North American Numbering Plan (NANP) area codes and international dialing codes on storefronts or trucks. If an area code is detected, GeoVision instantly bypasses AI guesswork and snaps to the correct region with 95%+ mathematical certainty.
- 🛣️ **Road Marking & Side Analysis (`road_analyzer.py`)**: Uses real OpenCV HSV color thresholding and Hough transformations to detect yellow vs white center lines (North America vs Europe/Asia) and identify the driving side of the road.
- 📝 **Language & Script Detection (`language_detector.py`)**: Detects Unicode script families (Latin, Cyrillic, CJK, Arabic, Thai), country-code top-level domains (ccTLDs like `.co.uk`, `.ca`), and currency symbols ($, €, £, ¥) from raw OCR text to instantly lock down regions.
- 🏪 **Chain Store Geolocation (`chain_store_locator.py`)**: When OCR detects major global brands (e.g., Walmart, Starbucks, McDonald's), GeoVision queries the Overpass API for all precise physical locations of that chain within the candidate radius.
- 🌍 **Reverse Geocode Validation (`nominatim_geocoder.py`)**: Uses OpenStreetMap's Nominatim API to mathematically validate that predicted coordinates land on solid ground (eliminating ocean hallucinations) and resolves them to full physical addresses.
- 🏢 **Cadastral & Land Use Searching (`property_records_client.py`)**: Queries the OpenStreetMap Overpass API for exact zoning data (commercial, residential, industrial) and property boundaries to confirm a location matches the expected biome and infrastructure.
- 🛰️ **Satellite & Street View Matching (`satellite_matcher.py` / `street_view_matcher.py`)**: Compares the ground-level image geometry against actual map tiles (Google Maps / OpenStreetMap) to verify building footprints and street layouts.
- 🌓 **Solar & Shadow Ephemeris (`shadow_analyzer.py`)**: Estimates solar declination and latitude bands based on EXIF timestamps and visual shadow features.

## 🛠️ The 9-Phase Geolocation Engine

1. **Visual Feature Extraction**: OpenCV analysis for color, edges, and architecture.
2. **Intelligent OCR**: EasyOCR/Tesseract with strict noise filtering to prevent hallucinated text matching.
3. **Deep Features**: ResNet50 mathematical fingerprinting for the image.
3c. **Visual Geo Engine**: CLIP ViT-B-32 embedding + cosine nearest-neighbor against the real geotagged reference DB — produces estimates with zero EXIF/OCR input (the GeoSpy core).
4. **OSINT Database Matching**: Queries Wikipedia and OpenStreetMap APIs for exact text and infrastructure matches (protected by robust User-Agent headers).
5. **Satellite Matching**: Correlates features with regional satellite imagery via OSM/Google tiles.
6. **Park Proximity Analysis**: Uses haversine mathematics to find nearby parks based on precise walking times.
7. **Cross-Verification**: Uses browser automation/API calls to check Google Maps/Street View.
8. **OpenCode VLM Integration**: Consults advanced Vision-Language Models (DeepSeek/Stepfun) natively via the Hermes architecture for semantic reasoning.
9. **Synthesis**: Compiles OSINT data, executes Weather, Telecom, Road, and Language overrides, fuses the coordinates via a weighted haversine mathematical cluster, and ranks candidate coordinates into a comprehensive JSON and interactive HTML report.

## 🚀 Setup & Execution

GeoVision is designed to run locally, prioritizing privacy and raw computational power.

```bash
git clone https://github.com/xanstomper/geovision.git
cd geovision
pip install -r requirements.txt
```

GeoVision is natively built for Hermes + OpenCode Zen architectures. Make sure your keys are set in `~/.hermes/.env` or in the root `.env` file:
```
OPENCODE_ZEN_API_KEY=sk-your_key_here
```

### Web Interface (Terminal Mode)

Launch the ultra-fast live-streaming web UI:
```bash
python3 web/app.py
```
Navigate to `http://localhost:9999` in your browser. The web UI actively executes the 9-phase deep scan in a subprocess and streams the backend terminal logs natively to your browser before triangulating the map.

### CLI Quick Start

You can also run scans directly from the terminal:
```bash
# Standard Deep Scan Forensic Analysis
python3 geovision_cli.py /path/to/image.jpg

# Advanced Scan with specific region targeting
python3 geovision_cli.py /path/to/image.jpg --near-park --region "Virginia"
```

### Outputs

Every scan automatically generates:
1. An **interactive HTML map report** showing candidate clusters, exact coordinates, and confidence levels.
2. A **JSON forensic payload** containing all extracted text, deep features, OSM node data, and coordinate estimates for easy integration into larger intelligence platforms.

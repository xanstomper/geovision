# GeoVision AI - The Ultimate OSINT Geolocation Engine

GeoVision is an ultra-advanced, Open-Source Intelligence (OSINT) geolocation pipeline designed to match and exceed the capabilities of tools like GeoSpy. It transforms standard computer vision and AI into a logic-driven forensic investigator. 

Rather than relying purely on subjective AI guessing, GeoVision fuses **Vision-Language Models (VLMs)** with hard mathematical OSINT—cross-referencing historical weather records, telecommunication area codes, satellite imagery, and municipal zoning data to pinpoint exact geographic coordinates from a single image.

## 🔥 Advanced OSINT Arsenal

GeoVision is equipped with cutting-edge investigative modules that typical AI tools lack:

- 🌦️ **Historical Weather Corroboration (`weather_corroborator.py`)**: 
  Extracts the exact date/time from image EXIF data and queries the Open-Meteo Historical Archive API for all AI-candidate locations. If the AI guesses a town in Florida but the photo shows rain while historical records confirm clear skies, GeoVision slashes the confidence of that location.
- 📞 **Telecom Area Code Extraction (`telecom_osint.py`)**: 
  Intercepts raw Optical Character Recognition (OCR) text *before* AI analysis. Uses Regex to hunt for North American Numbering Plan (NANP) area codes and international dialing codes on storefronts or trucks. If an area code is detected, GeoVision instantly bypasses AI guesswork and snaps to the correct region with 95%+ mathematical certainty.
- 🏢 **Cadastral & Land Use Searching (`property_records_client.py`)**:
  Queries the OpenStreetMap Overpass API for exact zoning data (commercial, residential, industrial) and property boundaries to confirm a location matches the expected biome and infrastructure.
- 🌍 **Satellite & Street View Matching (`street_view_matcher.py`)**:
  Compares the ground-level image geometry against actual map tiles and satellite imagery to verify building footprints and street layouts.
- 🌓 **Solar & Shadow Ephemeris (`shadow_analyzer.py`)**:
  Estimates solar declination and latitude bands based on EXIF timestamps and visual shadow features.

## 🛠️ The 9-Phase Geolocation Pipeline

1. **Visual Feature Extraction**: OpenCV analysis for color, edges, and architecture.
2. **Intelligent OCR**: EasyOCR/Tesseract with strict noise filtering to prevent hallucinated text matching.
3. **Deep Features**: ResNet50 mathematical fingerprinting for the image.
4. **OSINT Database Matching**: Queries Wikipedia and OpenStreetMap Overpass APIs for exact text and infrastructure matches (protected by robust User-Agent headers).
5. **Satellite Matching**: Correlates features with regional satellite imagery.
6. **Park Proximity Analysis**: Finds nearby parks based on precise walking times.
7. **Cross-Verification**: Uses browser automation to check Google Maps.
8. **OpenCode VLM Integration**: Consults advanced Vision-Language Models (DeepSeek/Stepfun) natively via the Hermes architecture for semantic reasoning.
9. **Synthesis**: Compiles OSINT data, executes Weather and Telecom overrides, and ranks candidate coordinates into a comprehensive JSON and interactive HTML report.

## 🚀 Setup & Execution

GeoVision is designed to run locally, prioritizing privacy and raw computational power.

```bash
git clone https://github.com/xanstomper/geovision.git
cd geovision
pip install -r requirements.txt
pip install rich typer

# GeoVision is natively built for Hermes + OpenCode Zen architectures
export OPENCODE_ZEN_API_KEY=your_key_here
```

### CLI Quick Start

The fastest way to deploy the engine is via the terminal dashboard:

```bash
# Standard Deep Scan Forensic Analysis
python3 geovision_cli.py /path/to/image.jpg

# Advanced Scan with specific region targeting
python3 geovision_cli.py /path/to/image.jpg --near-park --region "Virginia"

# Output raw JSON for pipeline integration
python3 geovision_cli.py /path/to/image.jpg --json-only
```

### Outputs

Every scan automatically generates:
1. An interactive **HTML Report** with Folium maps, candidate plots, and extracted features.
2. A raw **JSON dump** of the pipeline result.
Both are saved to `./reports/` by default.

---
*Built for absolute precision. Say goodbye to AI hallucinations.*

# GeoVision Architecture

## Overview
GeoVision is a state-of-the-art, purely programmatic OSINT geolocation pipeline. It is designed to extract, fuse, and reason about location data from images without relying on external Vision-Language Models (VLMs) or third-party AI provider APIs. By employing deterministic computer vision techniques, targeted machine learning classifiers (ResNet), and robust OSINT API clients, GeoVision provides verifiable and probabilistic geolocation estimates.

## Core Design Principles
1. **No VLM Dependency:** The system avoids black-box reasoning. All visual feature extraction is done using explicit algorithms (OpenCV) and fine-tuned local models (PyTorch/torchvision), ensuring the pipeline is reproducible and privacy-respecting.
2. **Probabilistic Evidence Fusion:** Instead of returning a single, opaque guess, GeoVision aggregates clues from various extractors into a probabilistic model. Each piece of evidence is weighted by its source reliability and detection confidence to compute an expected latitude/longitude center and a variance-based uncertainty radius.
3. **Modular Pipeline:** The system is heavily modularized, featuring numerous specialized modules for detecting specific types of features (e.g., vegetation, scenes, shadows) and interacting with OSINT APIs (e.g., OpenStreetMap, Wikimedia, property records).

## Pipeline Architecture
The GeoVision pipeline, executed primarily via `geovision_deep_scan.py`, runs through multiple phases:

*   **Phase 0: Metadata Extraction** 
    Extracts raw EXIF data and timestamps to form foundational evidence (e.g., direct GPS, if available).
*   **Phase 1: Visual Feature Extraction**
    Utilizes `VisionEngine` and `GeoVisionCore` (OpenCV) to extract low-level features such as edge density, color histograms, sky/vegetation ratios, and structural lines.
    *   *Phase 1b: Shadow Analysis* - Estimates latitude bands using sun declination and time of day.
*   **Phase 2: OCR Text Extraction**
    Extracts text using local OCR engines (EasyOCR, Tesseract, PaddleOCR) to detect street signs, storefronts, and license plates.
*   **Phase 3: Deep Feature & Semantic Extraction**
    Employs ResNet50 for high-dimensional feature vectors. Integrates with the new `SceneClassifier` and `VegetationClassifier` to understand the environmental context.
*   **Phase 4: Real OSINT Data Matching**
    Cross-references extracted text and features against open records (Wikimedia, OSM via `OSMFeatureMatcher`, telecom databases, property records).
*   **Phase 5: Evidence Fusion**
    The `EvidenceFusion` module aggregates all generated evidence points (EXIF, OCR, Telecom, Climate, Vegetation, Architecture). It computes a weighted spatial average and variance to yield a final candidate location and confidence radius.
*   **Phase 6 & 7: Cross-Verification & Reporting**
    Validates candidates using satellite/street-view matching and generates a comprehensive HTML/JSON report.

## The New Modules
Recent updates have introduced several key modules that significantly expand the pipeline's capabilities:

1.  **SceneClassifier:** Uses PyTorch and a pre-trained ResNet50 model to classify images into distinct environmental categories (e.g., Urban, Rural, Coastal, Mountain).
2.  **VegetationClassifier:** Analyzes visible light (VARI) and HSV color spaces using OpenCV to estimate vegetation density and guess climate conditions (e.g., tropical, arid, temperate).
3.  **OSMFeatureMatcher:** Interacts directly with the Overpass API to query OpenStreetMap for nearby points of interest (POIs), road characteristics, and building density.
4.  **EvidenceFusion:** The core probabilistic engine that ingests weighted clues and calculates the final geolocation estimate using statistical variance.
5.  **TelecomOSINT / Property Records / Elevation (etc.):** A suite of newly integrated OSINT modules designed to query public infrastructure and geographic databases based on visual clues.

## Probabilistic Evidence Fusion
The `EvidenceFusion` module is the brain of the geolocation estimator.
*   **Input:** Evidence points are added with `(lat, lon, confidence, radius_km)`.
*   **Weighting:** Each source has an inherent weight (e.g., `exif=1.0`, `ocr=0.8`, `vegetation=0.3`). The effective weight of a clue is `source_weight * confidence`.
*   **Prediction:** The final coordinates are calculated as the weighted average of all evidence points. The uncertainty (radius) is derived from the spatial variance of the points relative to the predicted center. This provides a mathematically sound search radius rather than an arbitrary guess.

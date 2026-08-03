#!/usr/bin/env python3
"""
GeoVision Deep Scan — GeoSpy-class Full Geolocation Pipeline
=============================================================

Flagship entry point that runs the complete multi-phase geolocation pipeline:

  Phase 1 — Visual Feature Extraction   (OpenCV + VisionEngine)
  Phase 2 — OCR Text Extraction          (EasyOCR with fallback)
  Phase 3 — Deep Feature Extraction      (ResNet / color histogram fallback)
  Phase 4 — Geolocation DB Matching      (GeolocationDatabase)
  Phase 5 — Satellite Imagery Matching   (SatelliteMatcher)
  Phase 6 — Park Proximity Analysis      (ParkFinder + GeolocationDB)
  Phase 7 — Cross-View Verification      (BrowserAutomation / fallback)
  Phase 8 — Synthesis Report Generation  (HTML + JSON + interactive map)

Usage
-----
  # CLI
  python geovision_deep_scan.py /path/to/image.jpg
  python geovision_deep_scan.py /path/to/image.jpg --output-dir ./reports --near-park
  python geovision_deep_scan.py /path/to/image.jpg --region Toronto
  python geovision_deep_scan.py /path/to/image.jpg --interactive --verbose

  # Programmatic
  from geovision_deep_scan import run_pipeline
  result = run_pipeline("/path/to/image.jpg", options={"region": "Toronto"})
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

try:
    from rich.logging import RichHandler
    from rich.console import Console
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)]
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger("GeoVisionDeepScan")


# ---------------------------------------------------------------------------
# Third-party imports with graceful degradation
# ---------------------------------------------------------------------------

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None
    logger.warning("OpenCV / numpy not available")

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = nn = models = transforms = None

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    folium = None
    FOLIUM_AVAILABLE = False

try:
    from modules import (
        SatelliteMatcher, SatelliteMatch, LocationReasoner, LocationEstimate,
        DeepAnalyzer, GoogleMapsController, MapsVerificationResult,
        find_nearby_parks as park_finder_search, BrowserAutomation,
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"GeoVision modules not fully available: {e}")
    SatelliteMatcher = SatelliteMatch = LocationReasoner = LocationEstimate = None
    DeepAnalyzer = GoogleMapsController = MapsVerificationResult = None
    park_finder_search = BrowserAutomation = None

# Geolocation DB completely removed in favor of real APIs
GEODB_AVAILABLE = False
GeolocationDatabase = GeolocationDB = None

try:
    from core.vision_engine import VisionEngine, VisionFeatures
    VISION_ENGINE_AVAILABLE = True
except ImportError:
    VISION_ENGINE_AVAILABLE = False
    VisionEngine = VisionFeatures = None

try:
    import exifread
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete result of the geolocation pipeline."""
    image_path: str
    timestamp: str
    success: bool
    phases_completed: List[str] = field(default_factory=list)
    phases_failed: List[str] = field(default_factory=list)

    # Phase results
    exif_data: Dict[str, Any] = field(default_factory=dict)
    visual_features: Dict[str, Any] = field(default_factory=dict)
    shadow_analysis: Dict[str, Any] = field(default_factory=dict)
    ocr_text: Dict[str, Any] = field(default_factory=dict)
    deep_features: Dict[str, Any] = field(default_factory=dict)
    reverse_image_search: Dict[str, Any] = field(default_factory=dict)
    db_matches: List[Dict[str, Any]] = field(default_factory=list)
    property_records: Dict[str, Any] = field(default_factory=dict)
    satellite_matches: List[Dict[str, Any]] = field(default_factory=list)
    park_proximity: Dict[str, Any] = field(default_factory=dict)
    cross_verification: Dict[str, Any] = field(default_factory=dict)
    synthesis_report: Dict[str, Any] = field(default_factory=dict)

    # Consolidated estimates
    location_estimates: List[Dict[str, Any]] = field(default_factory=list)
    best_estimate: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase0_exif_extraction(image_path: str) -> Dict[str, Any]:
    """
    Phase 0 — EXIF Metadata Extraction
    Extracts embedded metadata and GPS coordinates.
    """
    phase_name = "phase0_exif_extraction"
    logger.info("━" * 48)
    logger.info("  Phase 0: EXIF Extraction (GeoSpy Method)")
    logger.info("━" * 48)

    result = {"phase": phase_name, "status": "failed", "data": {}, "gps": None}
    
    try:
        from modules.exif_extractor import extract_exif_data, get_gps_from_exif
        exif_data = extract_exif_data(image_path)
        if exif_data:
            result["data"] = exif_data
            result["status"] = "success"
            gps = get_gps_from_exif(exif_data)
            if gps:
                result["gps"] = gps
                logger.info(f"  ✓ High Confidence GPS Hit: {gps['latitude']:.4f}, {gps['longitude']:.4f}")
        else:
            result["status"] = "limited"
    except Exception as e:
        logger.warning(f"  EXIF Extraction Failed: {e}")
        result["error"] = str(e)
    
    return result

def phase1b_shadow_analysis(exif_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1b — Shadow & Solar Analysis (Advanced OSINT).
    Attempts to estimate latitude bands based on sun declination from EXIF timestamp.
    """
    phase_name = "phase1b_shadow_analysis"
    logger.info("━" * 48)
    logger.info("  Phase 1b: Shadow & Solar Analysis")
    logger.info("━" * 48)

    try:
        from modules.shadow_analyzer import ShadowAnalyzer
        analyzer = ShadowAnalyzer()
        solar_data = analyzer.estimate_latitude_from_sun(exif_data)
        
        if solar_data:
            logger.info(f"  [+] Solar Analysis Successful. Declination: {solar_data.get('solar_declination'):.2f}")
            return {
                "phase": phase_name,
                "status": "success",
                "solar_declination": solar_data.get('solar_declination'),
                "estimated_time": solar_data.get('estimated_time')
            }
        else:
            logger.info("  [-] No EXIF DateTime found, skipping shadow analysis.")
            return {"phase": phase_name, "status": "skipped", "reason": "No EXIF DateTime"}
    except Exception as e:
        logger.error(f"  [!] Shadow analysis failed: {e}")
        return {"phase": phase_name, "status": "failed", "error": str(e)}

def phase1_visual_features(image_path: str) -> Dict[str, Any]:
    """
    Phase 1 — Visual Feature Extraction using OpenCV + VisionEngine.
    """
    phase_name = "phase1_visual_features"
    logger.info("━" * 48)
    logger.info("  Phase 1: Visual Feature Extraction")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "features": {}, "raw_analysis": {},
    }

    try:
        # Strategy 1: VisionEngine
        if VISION_ENGINE_AVAILABLE and VisionEngine is not None:
            logger.info("  Using VisionEngine...")
            engine = VisionEngine()
            features = engine.analyze_image(image_path)
            if features is not None:
                fd = features.to_dict() if hasattr(features, 'to_dict') else {}
                result["features"] = fd
                result["status"] = "success"
                result["source"] = "VisionEngine"
                logger.info("  ✓ VisionEngine complete")
                return result

        # Strategy 2: GeoVisionCore
        try:
            from geovision import GeoVisionCore
            logger.info("  Using GeoVisionCore...")
            core = GeoVisionCore()
            analysis = core.analyze_image(image_path)
            building = core.detect_building_style(image_path)
            veg = core.analyze_vegetation(image_path)
            result["features"] = {
                "image_analysis": analysis,
                "building_style": building,
                "vegetation": veg,
            }
            result["status"] = "success"
            result["source"] = "GeoVisionCore"
            return result
        except Exception:
            logger.warning("  GeoVisionCore unavailable, falling back to raw OpenCV")

        # Strategy 3: Raw OpenCV
        if cv2 is None or np is None:
            result["error"] = "OpenCV/numpy unavailable"
            return result

        img = cv2.imread(image_path)
        if img is None:
            result["error"] = "Cannot load image"
            return result

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Dominant colors via k-means
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_colors = [f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}" for c in centers]

        mean_color = cv2.mean(img)[:3]
        brightness = int(np.mean(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges) / (h * w))

        # Sky
        top_third = img[:h // 3, :]
        hsv_top = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
        sky_mask = cv2.inRange(hsv_top, (80, 20, 100), (130, 255, 255))
        sky_ratio = float(np.sum(sky_mask > 0) / (top_third.shape[0] * top_third.shape[1] or 1))

        # Vegetation
        green_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        veg_ratio = float(np.sum(green_mask > 0) / (h * w or 1))

        # Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        num_lines = len(lines) if lines is not None else 0

        result["features"] = {
            "dimensions": {"width": w, "height": h},
            "dominant_colors": dominant_colors,
            "mean_color_bgr": [int(c) for c in mean_color],
            "brightness": brightness,
            "edge_ratio": round(edge_ratio, 4),
            "sky_ratio": round(sky_ratio, 4),
            "vegetation_ratio": round(veg_ratio, 4),
            "num_lines": num_lines,
            "is_high_res": w >= 1000,
        }
        result["status"] = "success"
        result["source"] = "OpenCV"
        logger.info("  ✓ OpenCV feature extraction complete")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 1 error: {e}")

    return result
def phase2_ocr(image_path: str) -> Dict[str, Any]:
    """
    Phase 2 — OCR text extraction using EasyOCR with fallback.
    Identifies signs, license plates, building numbers, and text clues.
    """
    phase_name = "phase2_ocr"
    logger.info("━" * 48)
    logger.info("  Phase 2: OCR Text Extraction")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "text": "", "significant_text": "",
        "word_count": 0, "detections": [],
    }

    try:
        # Strategy 1: GeoVisionCore EasyOCR
        try:
            from geovision import GeoVisionCore
            core = GeoVisionCore()
            if core.reader is not None:
                logger.info("  Using GeoVisionCore EasyOCR...")
                ocr_result = core.extract_text(image_path)
                if ocr_result.get("supported", False):
                    result["text"] = ocr_result.get("text", "")
                    result["significant_text"] = ocr_result.get("significant", "")
                    result["word_count"] = ocr_result.get("word_count", 0)
                    result["status"] = "success"
                    result["source"] = "GeoVisionCore_EasyOCR"
                    logger.info(f"  ✓ Extracted {result['word_count']} words")
                    return result
        except Exception as e:
            logger.warning(f"  GeoVisionCore OCR failed: {e}")

        # Strategy 2: Direct EasyOCR
        if easyocr:
            logger.info("  Using EasyOCR directly...")
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            raw_results = reader.readtext(image_path, detail=1)
            
            lines = []
            if raw_results:
                # Sort by y-center
                sorted_res = sorted(raw_results, key=lambda r: (r[0][0][1] + r[0][2][1])/2)
                current_line = [sorted_res[0]]
                for i in range(1, len(sorted_res)):
                    res = sorted_res[i]
                    prev = current_line[-1]
                    y_c = (res[0][0][1] + res[0][2][1])/2
                    prev_yc = (prev[0][0][1] + prev[0][2][1])/2
                    height = prev[0][2][1] - prev[0][0][1]
                    if abs(y_c - prev_yc) < height * 0.5:
                        current_line.append(res)
                    else:
                        lines.append(current_line)
                        current_line = [res]
                lines.append(current_line)

            texts = []
            significant = []
            for line in lines:
                # Sort line by x-center
                line = sorted(line, key=lambda r: (r[0][0][0] + r[0][1][0])/2)
                # Use a stricter confidence threshold (0.5 for general text, 0.7 for significant) to reduce noise
                line_str = " ".join([r[1] for r in line if r[2] >= 0.5])
                if line_str:
                    texts.append(line_str)
                    # Only add to significant if the confidence is high enough to warrant an OSINT search
                    sig_str = " ".join([r[1] for r in line if r[2] >= 0.7])
                    if sig_str and any(len(w) > 3 and w[0].isupper() for w in sig_str.split()):
                        significant.append(sig_str)
                        
            result["text"] = " | ".join(texts)
            result["significant_text"] = " | ".join(significant)[:500]
            result["word_count"] = sum(len(t.split()) for t in texts)
            result["detections"] = [
                {"text": t, "confidence": round(float(c), 3)}
                for _, t, c in raw_results if c >= 0.5
            ]
            result["status"] = "success"
            result["source"] = "EasyOCR"
            logger.info(f"  ✓ EasyOCR extracted {result['word_count']} words in {len(texts)} lines")
            return result

        # Strategy 3: pytesseract
        try:
            import pytesseract
            from PIL import Image as PILImage
            logger.info("  Using pytesseract...")
            pil_img = PILImage.open(image_path)
            text = pytesseract.image_to_string(pil_img)
            significant = " ".join(
                [w for w in text.split() if len(w) > 3 and w[0].isupper()]
            )
            if text.strip():
                result["text"] = text.strip()
                result["significant_text"] = significant[:500]
                result["word_count"] = len(text.split())
                result["status"] = "success"
                result["source"] = "Tesseract"
                logger.info(f"  ✓ Tesseract extracted {result['word_count']} words")
                return result
        except ImportError:
            logger.warning("  pytesseract not available")

        # Strategy 4: PaddleOCR
        try:
            from paddleocr import PaddleOCR
            logger.info("  Using PaddleOCR...")
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            raw = ocr.ocr(image_path, cls=True)
            texts = []
            if raw and raw[0]:
                for line in raw[0]:
                    txt = line[1][0]
                    texts.append(txt)
            result["text"] = " ".join(texts)
            result["significant_text"] = " ".join(
                [t for t in texts if len(t) > 3 and t[0].isupper()]
            )[:500]
            result["word_count"] = len(texts)
            result["status"] = "success"
            result["source"] = "PaddleOCR"
            logger.info(f"  ✓ PaddleOCR extracted {result['word_count']} words")
            return result
        except ImportError:
            logger.warning("  PaddleOCR not available")

        result["status"] = "limited"
        result["error"] = "No OCR engine available"
        logger.warning("  ⚠ No OCR engine available")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 2 error: {e}")

    return result
def phase3_deep_features(image_path: str) -> Dict[str, Any]:
    """
    Phase 3 — Deep feature extraction using ResNet or fallback to color histograms.
    Produces a feature vector for similarity comparison.
    """
    phase_name = "phase3_deep_features"
    logger.info("━" * 48)
    logger.info("  Phase 3: Deep Feature Extraction")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "feature_vector": [], "feature_dim": 0, "method": "none",
    }

    try:
        # Strategy 1: DeepAnalyzer (ResNet50)
        if MODULES_AVAILABLE and DeepAnalyzer is not None:
            logger.info("  Using DeepAnalyzer (ResNet50)...")
            analyzer = DeepAnalyzer()
            features = analyzer.extract_features(image_path)
            if features is not None and np.any(features):
                fn = float(np.linalg.norm(features))
                result["feature_vector"] = features.tolist()[:64]
                result["feature_dim"] = len(features)
                result["feature_norm"] = round(fn, 4)
                result["method"] = "ResNet50"
                result["status"] = "success"
                logger.info(f"  ✓ ResNet50: {len(features)} dims, norm={fn:.4f}")
                return result
            logger.warning("  DeepAnalyzer returned empty features")

        # Strategy 2: Direct torchvision ResNet
        if TORCH_AVAILABLE and models is not None:
            logger.info("  Loading ResNet50 directly...")
            try:
                resnet = models.resnet50(pretrained=True)
                model = nn.Sequential(*list(resnet.children())[:-1])
                model.eval()
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
                if Image:
                    pil_img = Image.open(image_path).convert("RGB")
                    tensor = preprocess(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        feats = model(tensor).flatten().numpy()
                    fn = float(np.linalg.norm(feats))
                    result["feature_vector"] = feats.tolist()[:64]
                    result["feature_dim"] = len(feats)
                    result["feature_norm"] = round(fn, 4)
                    result["method"] = "ResNet50_direct"
                    result["status"] = "success"
                    logger.info(f"  ✓ ResNet50 direct: {len(feats)} dims, norm={fn:.4f}")
                    return result
            except Exception as e:
                logger.warning(f"  ResNet50 direct failed: {e}")

        # Strategy 3: Color histogram + LBP (fallback)
        if cv2 is not None and np is not None:
            logger.info("  Using color histogram + LBP features (fallback)...")
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Color histograms
                hr = cv2.calcHist([img], [2], None, [64], [0, 256]).flatten()
                hg = cv2.calcHist([img], [1], None, [64], [0, 256]).flatten()
                hb = cv2.calcHist([img], [0], None, [64], [0, 256]).flatten()
                hist = np.concatenate([hr, hg, hb])
                hist = hist / (hist.sum() + 1e-8)

                # Edge histogram
                edges = cv2.Canny(gray, 50, 150)
                eh = cv2.calcHist([edges], [0], None, [32], [0, 256]).flatten()
                eh = eh / (eh.sum() + 1e-8)

                # Simplified LBP texture
                lbp = np.zeros(256, dtype=np.float32)
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        c = gray[i, j]
                        code = 0
                        code |= (gray[i-1, j-1] > c) << 7
                        code |= (gray[i-1, j] > c) << 6
                        code |= (gray[i-1, j+1] > c) << 5
                        code |= (gray[i, j+1] > c) << 4
                        code |= (gray[i+1, j+1] > c) << 3
                        code |= (gray[i+1, j] > c) << 2
                        code |= (gray[i+1, j-1] > c) << 1
                        code |= (gray[i, j-1] > c)
                        lbp[code] += 1
                lbp = lbp / (lbp.sum() + 1e-8)

                full = np.concatenate([hist, eh, lbp])
                result["feature_vector"] = full[:64].tolist()
                result["feature_dim"] = len(full)
                result["feature_norm"] = round(float(np.linalg.norm(full)), 4)
                result["method"] = "color_histogram_LBP"
                result["status"] = "success"
                logger.info(f"  ✓ Histogram features: {len(full)} dims")
                return result

        result["error"] = "No deep feature extraction method available"
        result["status"] = "failed"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 3 error: {e}")

    return result

def phase3b_reverse_image_search(image_path: str) -> Dict[str, Any]:
    """
    Phase 3b — Reverse Image Search (GeoSpy Method).
    """
    phase_name = "phase3b_reverse_image_search"
    logger.info("━" * 48)
    logger.info("  Phase 3b: Reverse Image Search OSINT")
    logger.info("━" * 48)

    result = {"phase": phase_name, "status": "failed", "matches": []}
    try:
        from modules.reverse_image_search import ReverseImageSearcher
        searcher = ReverseImageSearcher()
        search_results = searcher.search_similar_images(image_path)
        result.update(search_results)
    except Exception as e:
        logger.warning(f"  Reverse Image Search Failed: {e}")
        result["error"] = str(e)
    return result

def phase4_db_matching(visual_features: Dict[str, Any],
                       ocr_result: Dict[str, Any],
                       reverse_result: Dict[str, Any],
                       region: Optional[str] = None) -> Dict[str, Any]:
    """
    Phase 4 — Real OSINT searching using Wikimedia and Overpass API.
    Replaces the mocked database with actual internet queries.
    """
    phase_name = "phase4_db_matching"
    logger.info("━" * 48)
    logger.info("  Phase 4: Real OSINT Data Matching")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "matches": [], "top_cities": [], "matched_features": {},
    }

    try:
        from modules.wikimedia_client import WikimediaClient
        wiki = WikimediaClient()
        
        matches = []
        
        # 1. Search Wikipedia for OCR text
        if ocr_result.get("text"):
            text = ocr_result.get("significant_text", ocr_result.get("text"))
            if len(text) > 4:
                # Load known chain store names to filter them out of Wikipedia results.
                # Wikipedia returns corporate HQ coordinates for brand names (e.g. "Walmart" 
                # returns Bentonville, AR), which poisons the location estimates.
                # Chain stores are handled separately by ChainStoreLocator in Phase 9.
                known_chains = set()
                try:
                    config_path = os.path.join(os.path.dirname(__file__), "config", "chain_stores.json")
                    with open(config_path, "r") as f:
                        known_chains = set(s.lower() for s in json.load(f))
                except Exception:
                    pass
                
                # Skip Wikipedia search if the text is just a chain store name
                text_lower = text.strip().lower()
                is_chain_only = text_lower in known_chains
                
                if not is_chain_only:
                    logger.info(f"  Searching Wikipedia for OCR text: {text[:50]}...")
                    wiki_results = wiki.search_public_records_by_text(text, limit=3)
                    for wr in wiki_results:
                        if wr.get("lat") and wr.get("lon"):
                            # Double-check: skip if Wikipedia title is a known chain store
                            title_lower = wr.get("title", "").lower()
                            if title_lower in known_chains:
                                logger.info(f"  [-] Skipping Wikipedia result '{wr.get('title')}' — chain store HQ coordinates")
                                continue
                            matches.append({
                                "city": wr.get("title"),
                                "latitude": wr.get("lat"),
                                "longitude": wr.get("lon"),
                                "confidence": 0.8,
                                "matched_feature": f"OCR Text: {text[:20]}"
                            })
                else:
                    logger.info(f"  [-] OCR text '{text}' is a known chain store — skipping Wikipedia (handled by ChainStoreLocator)")

        if matches:
            result["matches"] = matches
            result["top_cities"] = list(set([m["city"] for m in matches]))
            result["status"] = "success"
            logger.info(f"  ✓ Found {len(matches)} potential locations via OSINT search")
        else:
            result["status"] = "limited"
            result["note"] = "No OSINT location matches found from visual features or text."
            logger.info("  ⚠ No OSINT matches found")
            
    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 4 error: {e}")

    return result
def phase4b_property_records(db_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phase 4b — OSINT Property & Cadastral Records Query (US/Canada).
    Queries open records (OSM/Overpass) for detailed property data near DB matches.
    """
    phase_name = "phase4b_property_records"
    logger.info("━" * 48)
    logger.info("  Phase 4b: Property & Land-Use Records (US/Canada/Global)")
    logger.info("━" * 48)
    
    if not db_matches:
        logger.info("  [-] No prior matches to query property records for.")
        return {"phase": phase_name, "status": "skipped", "records": []}

    try:
        from modules.property_records_client import PropertyRecordsClient
        client = PropertyRecordsClient()
        
        # Only query top 2 estimates to save API limits
        property_data = []
        for match in db_matches[:2]:
            lat = match.get("latitude")
            lon = match.get("longitude")
            if lat and lon:
                logger.info(f"  [*] Querying land-use records for {lat:.4f}, {lon:.4f}")
                records = client.find_land_use_records(lat, lon, radius_m=500)
                if records:
                    logger.info(f"  [+] Found {len(records)} property/boundary records.")
                    property_data.append({
                        "source_match": match.get("matched_feature"),
                        "latitude": lat,
                        "longitude": lon,
                        "records": records[:5]  # Keep top 5 to avoid bloating
                    })
        return {"phase": phase_name, "status": "success", "records": property_data}
    except Exception as e:
        logger.error(f"  [!] Property records query failed: {e}")
        return {"phase": phase_name, "status": "failed", "error": str(e), "records": []}

def phase5_satellite_matching(visual_features: Dict[str, Any],
                               candidate_coords: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phase 5 — Satellite imagery matching using SatelliteMatcher.
    Matches architectural style, materials, and vegetation patterns to regions.
    """
    phase_name = "phase5_satellite_matching"
    logger.info("━" * 48)
    logger.info("  Phase 5: Satellite Imagery Matching")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "matches": [], "top_regions": [],
    }

    if not MODULES_AVAILABLE or SatelliteMatcher is None:
        result["error"] = "SatelliteMatcher not available"
        result["status"] = "failed"
        logger.warning("  ⚠ SatelliteMatcher not available")
        return result

    try:
        # Build feature object compatible with SatelliteMatcher
        class FeatureObj:
            def __init__(self, data):
                self.architectural_style = data.get("architectural_style")
                self.facade_material = data.get("facade_material")
                self.region_indicators = data.get("region_indicators", [])
                self.tree_types = data.get("tree_types", [])
                self.vegetation_density = data.get("vegetation_density", 0.0)

        top_candidates = candidate_coords[:10] if candidate_coords else []
        if not top_candidates:
            logger.warning("  [-] No coordinates available for satellite matching")
            return {"status": "limited", "note": "No coordinates provided"}

        feature_obj = FeatureObj(visual_features)
        matcher = SatelliteMatcher()
        all_satellite_matches = []

        for cand in top_candidates:
            lat = cand["latitude"]
            lon = cand["longitude"]
            logger.info(f"  [*] Querying satellite imagery around {lat:.4f}, {lon:.4f}...")
            
            matches = matcher.match_features_to_satellite(
                feature_obj, lat, lon, radius_km=5.0,
            )
            if matches:
                all_satellite_matches.extend(matches)

        if all_satellite_matches:
            result["matches"] = [
                {
                    "latitude": round(m.latitude, 6),
                    "longitude": round(m.longitude, 6),
                    "confidence": round(m.confidence, 4),
                    "match_type": m.match_type,
                    "metadata": m.metadata,
                }
                for m in all_satellite_matches
            ]
            result["top_regions"] = list(set(m["match_type"] for m in result["matches"]))
            result["status"] = "success"
            logger.info(f"  ✓ {len(all_satellite_matches)} satellite matches found across candidates")
        else:
            result["status"] = "limited"
            result["note"] = "No satellite matches found for features."
            logger.warning("  ⚠ No satellite matches")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 5 error: {e}")

    return result
def phase6_park_proximity(visual_features: Dict[str, Any],
                           satellite_matches: Dict[str, Any],
                           near_park: bool = False,
                           candidate_coords: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phase 6 — Park proximity analysis using park_finder + GeolocationDatabase.
    """
    phase_name = "phase6_park_proximity"
    logger.info("━" * 48)
    logger.info("  Phase 6: Park Proximity Analysis")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "parks": [], "nearest_park": None, "walk_analysis": {},
    }

    local_coords: List[Tuple[float, float]] = []
    if satellite_matches.get("matches"):
        for m in satellite_matches["matches"][:10]:
            local_coords.append((m["latitude"], m["longitude"]))
    elif candidate_coords:
        for c in candidate_coords[:10]:
            local_coords.append((c["latitude"], c["longitude"]))

    try:
        all_parks = []
        for lat, lon in local_coords:


            if MODULES_AVAILABLE and park_finder_search is not None:
                try:
                    for p in park_finder_search(lat, lon, max_distance_km=2.0):
                        all_parks.append({
                            "name": p["name"], "city": p.get("city", "unknown"),
                            "latitude": p.get("lat", lat),
                            "longitude": p.get("lon", lon),
                            "distance_km": p["distance_km"],
                            "walk_minutes": p.get("walk_minutes", round(p["distance_km"] / 0.08)),
                            "within_8_10_min": p.get("within_8_10_min", False),
                            "source": "ParkFinder",
                        })
                except Exception as e:
                    logger.warning(f"  ParkFinder error: {e}")

        seen = set()
        unique = []
        for p in all_parks:
            if p["name"] not in seen:
                seen.add(p["name"])
                unique.append(p)
        unique.sort(key=lambda x: x["distance_km"])
        result["parks"] = unique[:10]

        if unique:
            result["nearest_park"] = unique[0]
            result["status"] = "success"
            result["walk_analysis"] = {
                "nearest_park_distance_km": unique[0]["distance_km"],
                "nearest_park_walk_minutes": unique[0].get("walk_minutes", 0),
                "walkable_parks_count": sum(1 for p in unique if p.get("walk_minutes", 99) <= 15),
                "parks_within_8_10_min": [p["name"] for p in unique if p.get("within_8_10_min")],
                "near_park_flag": near_park,
            }
            if near_park:
                result["near_park_filter"] = {
                    "active": True,
                    "parks_within_8_10_min": [p["name"] for p in unique if p.get("within_8_10_min")],
                    "count": sum(1 for p in unique if p.get("within_8_10_min")),
                }
            logger.info(f"  ✓ {len(unique)} parks, nearest: {unique[0]['name']}")
        else:
            result["status"] = "limited"
            result["note"] = "No parks found"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 6 error: {e}")

    return result
def phase7_cross_verification(location_estimates: List[Dict[str, Any]],
                               interactive: bool = False) -> Dict[str, Any]:
    """
    Phase 7 — Cross-view verification using browser automation or fallback.
    """
    phase_name = "phase7_cross_verification"
    logger.info("━" * 48)
    logger.info("  Phase 7: Cross-View Verification")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "verifications": [], "browser_used": False, "interactive": interactive,
    }

    if not location_estimates:
        result["error"] = "No location estimates to verify"
        result["status"] = "limited"
        return result

    try:
        # Strategy 1: BrowserAutomation
        if MODULES_AVAILABLE and BrowserAutomation is not None:
            logger.info("  Attempting browser automation...")
            browser = BrowserAutomation(backend="mcp")
            connected = browser.connect()
            if not connected:
                for backend in ["pyautogui", "selenium"]:
                    logger.info(f"  Trying backend: {backend}")
                    browser = BrowserAutomation(backend=backend)
                    if browser.connect():
                        connected = True
                        break
            if connected:
                result["browser_used"] = True
                for est in location_estimates[:3]:
                    lat = est.get("latitude", 0)
                    lon = est.get("longitude", 0)
                    try:
                        browser.navigate(f"https://www.google.com/maps/@{lat},{lon},19z", wait=1.5)
                        result["verifications"].append({
                            "latitude": lat, "longitude": lon,
                            "maps_url": f"https://www.google.com/maps/@{lat},{lon},19z",
                            "verified": True, "method": "browser_navigation",
                        })
                    except Exception as e:
                        logger.warning(f"  Browser navigation error: {e}")
                browser.close()
                result["status"] = "success" if result["verifications"] else "limited"
                return result

        # Strategy 2: GoogleMapsController
        if MODULES_AVAILABLE and GoogleMapsController is not None:
            logger.info("  Using GoogleMapsController (Selenium)...")
            controller = GoogleMapsController()
            for est in location_estimates[:3]:
                lat = est.get("latitude", 0)
                lon = est.get("longitude", 0)
                try:
                    ver = controller.verify_location(lat, lon)
                    result["verifications"].append({
                        "latitude": lat, "longitude": lon,
                        "verified": ver.verified,
                        "street_view_available": ver.street_view_available,
                        "satellite_confidence": ver.satellite_match_confidence,
                        "method": "GoogleMapsController",
                    })
                except Exception as e:
                    logger.warning(f"  Controller error: {e}")
            controller.close()
            result["browser_used"] = True
            result["status"] = "success" if result["verifications"] else "limited"
            return result

        # Strategy 3: Generate URLs for manual verification
        logger.info("  Generating Google Maps URLs for manual verification...")
        for est in location_estimates[:5]:
            lat = est.get("latitude", 0)
            lon = est.get("longitude", 0)
            result["verifications"].append({
                "latitude": lat, "longitude": lon,
                "maps_url": f"https://www.google.com/maps/@{lat},{lon},19z",
                "streetview_url": f"https://www.google.com/maps/@{lat},{lon},3a,75y,90h,90t",
                "verified": False, "manual": True, "method": "url_generation",
            })
        result["status"] = "limited"
        result["note"] = "URLs generated for manual verification"

        if interactive:
            try:
                import webbrowser
                if result["verifications"]:
                    webbrowser.open(result["verifications"][0]["maps_url"])
                    result["interactive"] = True
                    result["browser_used"] = True
                    logger.info("  Browser opened for interactive verification")
            except Exception as e:
                logger.warning(f"  Could not open browser: {e}")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 7 error: {e}")

    return result


def phase9_synthesis(phases: Dict[str, Dict[str, Any]],
                      options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 9 — Synthesis and report generation.
    Combines all phases into a unified report with ranked location estimates.
    """
    phase_name = "phase9_synthesis"
    logger.info("━" * 48)
    logger.info("  Phase 8: Synthesis & Report Generation")
    logger.info("━" * 48)

    result: Dict[str, Any] = {
        "phase": phase_name, "status": "failed",
        "location_estimates": [], "best_estimate": {},
        "confidence_summary": {}, "evidence_summary": {},
        "execution_summary": {},
    }

    try:
        all_estimates: List[Dict[str, Any]] = []

        # From EXIF (Highest Confidence)
        exif = phases.get("exif_data", {})
        if exif.get("gps"):
            gps = exif["gps"]
            all_estimates.append({
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "confidence": 1.0,  # Absolute certainty
                "sources": ["exif_metadata:gps"],
                "evidence": {"exif_data": "GPS Coordinates embedded in file"},
                "phase": "EXIF",
            })

        # From satellite
        sat = phases.get("satellite_matches", {})
        if sat.get("matches"):
            for m in sat["matches"][:5]:
                all_estimates.append({
                    "latitude": m["latitude"], "longitude": m["longitude"],
                    "confidence": m["confidence"] * 0.8,
                    "sources": [f"satellite:{m['match_type']}"],
                    "evidence": {"match_type": m["match_type"]},
                    "phase": "Satellite",
                })

        # From DB matching
        db = phases.get("db_matches", {})
        
        if db.get("public_records"):
            for pr in db["public_records"][:5]:
                if pr.get("lat") and pr.get("lon"):
                    all_estimates.append({
                        "latitude": pr["lat"],
                        "longitude": pr["lon"],
                        "confidence": 0.85,
                        "sources": [f"public_record:{pr.get('title')}"],
                        "evidence": {"title": pr.get("title")},
                        "phase": "PublicRecords",
                    })
                    
        # From property records (OSM/Land Use)
        prop = phases.get("property_records", {})
        if prop.get("status") == "success" and prop.get("records"):
            for pr in prop["records"]:
                all_estimates.append({
                    "latitude": pr["latitude"],
                    "longitude": pr["longitude"],
                    "confidence": 0.88,
                    "sources": [f"property_records:{pr.get('source_match', 'osm')}"],
                    "evidence": {"records_count": len(pr.get("records", []))},
                    "phase": "PropertyRecords",
                })


        # From park proximity
        park = phases.get("park_proximity", {})
        if park.get("nearest_park"):
            np_ = park["nearest_park"]
            all_estimates.append({
                "latitude": np_.get("latitude", np_.get("lat", 0)),
                "longitude": np_.get("longitude", np_.get("lon", 0)),
                "confidence": 0.4,
                "sources": ["park_proximity"],
                "evidence": {"park": np_["name"], "distance_km": np_.get("distance_km", 0)},
                "phase": "Park",
            })

        # ── Advanced OSINT: Telecom Area Code Extraction ──
        ocr = phases.get("ocr_text", {})
        if ocr.get("text"):
            try:
                from modules.telecom_osint import TelecomOSINT
                telecom = TelecomOSINT()
                findings = telecom.analyze_ocr_for_telecom_data(ocr["text"])
                for f in findings:
                    all_estimates.append({
                        "latitude": f["latitude"],
                        "longitude": f["longitude"],
                        "confidence": f["confidence"],
                        "sources": [f"telecom:{f['code']}"],
                        "evidence": {"telecom_match": f"Matched phone/area code {f['code']} to {f['region']}"},
                        "phase": "Telecom"
                    })
            except Exception as e:
                logger.error(f"  [-] Telecom OSINT skipped/failed: {e}", exc_info=True)

        # ── Advanced OSINT: Language & Script Detection ──
        ocr = phases.get("ocr_text", {})
        if ocr.get("text"):
            try:
                from modules.language_detector import LanguageDetector
                detector = LanguageDetector()
                findings = detector.analyze_text(ocr["text"])
                for f in findings:
                    all_estimates.append({
                        "latitude": f["latitude"],
                        "longitude": f["longitude"],
                        "confidence": f.get("confidence", 0.5),
                        "sources": [f"language:{f.get('region', 'unknown')}"],
                        "evidence": {"language_match": f"Detected language/script for region {f.get('region')}"},
                        "phase": "Language"
                    })
            except Exception as e:
                logger.error(f"  [-] Language Detection skipped/failed: {e}", exc_info=True)

        # ── Advanced OSINT: Road Feature Analysis ──
        image_path_opt = options.get("image_path", "")
        if image_path_opt:
            try:
                from modules.road_analyzer import RoadAnalyzer
                analyzer = RoadAnalyzer()
                road_data = analyzer.analyze_road_features(image_path_opt)
                phases["road_analysis"] = road_data
                if road_data.get("region_indicators"):
                    center_lat, center_lon = 0.0, 0.0
                    if all_estimates:
                        best_est = sorted(all_estimates, key=lambda x: x["confidence"], reverse=True)[0]
                        center_lat = best_est["latitude"]
                        center_lon = best_est["longitude"]
                    
                    for ind in road_data["region_indicators"]:
                        all_estimates.append({
                            "latitude": center_lat,
                            "longitude": center_lon,
                            "confidence": road_data.get("confidence", 0.5) * 0.8,
                            "sources": [f"road:{ind}"],
                            "evidence": {"road_features": f"Driving side: {road_data.get('driving_side')}, Line color: {road_data.get('line_color')}, Indicator: {ind}"},
                            "phase": "Road"
                        })
            except Exception as e:
                logger.error(f"  [-] Road Analysis skipped/failed: {e}", exc_info=True)

        # ── Advanced OSINT: Chain Store Geolocation ──
        sig_text = phases.get("ocr_text", {}).get("significant_text", "")
        if sig_text:
            try:
                config_path = os.path.join(os.path.dirname(__file__), "config", "chain_stores.json")
                try:
                    with open(config_path, "r") as f:
                        store_names = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load chain stores config: {e}")
                    store_names = []
                detected_stores = [s for s in store_names if s.lower() in sig_text.lower()]
                if detected_stores:
                    center_lat, center_lon = None, None
                    if all_estimates:
                        best_est = sorted(all_estimates, key=lambda x: x["confidence"], reverse=True)[0]
                        center_lat = best_est["latitude"]
                        center_lon = best_est["longitude"]
                    
                    # If no prior estimates, try to get region center from the --region flag
                    if center_lat is None and options.get('region'):
                        try:
                            from modules.nominatim_geocoder import NominatimGeocoder
                            _geo = NominatimGeocoder()
                            _rv = _geo.forward_geocode(options['region'])
                            if _rv and _rv.get('latitude') and _rv.get('longitude'):
                                center_lat = _rv['latitude']
                                center_lon = _rv['longitude']
                                logger.info(f"  ✓ Using region center for chain store search: {center_lat}, {center_lon}")
                        except Exception as e:
                            logger.error(f"  [-] Failed to geocode region for chain store: {e}")
                    
                    from modules.chain_store_locator import ChainStoreLocator
                    locator = ChainStoreLocator()
                    store_results = locator.locate_chain_stores(detected_stores, center_lat, center_lon)
                    for res in store_results:
                        all_estimates.append({
                            "latitude": res["latitude"],
                            "longitude": res["longitude"],
                            "confidence": 0.92,
                            "sources": [f"chain_store:{res.get('store_name', 'unknown')}"],
                            "evidence": {"store_name": res.get("store_name"), "store_confidence": res.get("confidence")},
                            "phase": "ChainStore"
                        })
            except Exception as e:
                logger.error(f"  [-] Chain Store Geolocation skipped/failed: {e}", exc_info=True)

        # ── a-e) CV Modules (Concurrent) ──
        import concurrent.futures

        def _run_sc():
            from modules.scene_classifier import SceneClassifier
            return SceneClassifier().classify(options.get('image_path', ''))
            
        def _run_sd():
            from modules.sign_detector import SignDetector
            return SignDetector().detect_signs(options.get('image_path', ''))
            
        def _run_vd():
            from modules.vehicle_detector import VehicleDetector
            return VehicleDetector().detect_vehicles(options.get('image_path', ''))
            
        def _run_vc():
            from modules.vegetation_classifier import VegetationClassifier
            return VegetationClassifier().classify(options.get('image_path', ''))
            
        def _run_ta():
            from modules.terrain_analyzer import TerrainAnalyzer
            return TerrainAnalyzer().analyze(options.get('image_path', ''))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            fut_sc = executor.submit(_run_sc)
            fut_sd = executor.submit(_run_sd)
            fut_vd = executor.submit(_run_vd)
            fut_vc = executor.submit(_run_vc)
            fut_ta = executor.submit(_run_ta)
            
            try:
                scene = fut_sc.result()
                if scene.get('status') == 'success':
                    phases['scene_classification'] = scene
                    if scene.get('region_hints'):
                        from modules.nominatim_geocoder import NominatimGeocoder
                        geocoder = NominatimGeocoder()
                        for rh in scene['region_hints']:
                            try:
                                val = geocoder.forward_geocode(rh)
                                if val and val.get('latitude') and val.get('longitude'):
                                    all_estimates.append({
                                        "latitude": val["latitude"],
                                        "longitude": val["longitude"],
                                        "confidence": 0.1,  # Lowered from 0.6 so continent centers don't win
                                        "sources": [f"scene_region:{rh}"],
                                        "evidence": {"scene_type": scene.get("scene_type"), "matched_region": rh},
                                        "phase": "SceneClassification"
                                    })
                                    logger.info(f"  ✓ Added region candidate '{rh}' from scene classifier")
                            except Exception as e:
                                logger.error(f"  [-] Failed to geocode region hint '{rh}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f'  [-] Scene classification skipped/failed: {e}', exc_info=True)
                
            try:
                sign_data = fut_sd.result()
                if sign_data.get('status') == 'success' and sign_data.get('detected_signs'):
                    # Combine all detected sign text
                    all_sign_text = " ".join(
                        s.get('text', '') for s in sign_data['detected_signs'] if s.get('text')
                    ).strip()
                    if all_sign_text:
                        phases['sign_detection'] = sign_data
                        phases['sign_detection']['combined_text'] = all_sign_text
                        logger.info(f"  ✓ Sign text detected: {all_sign_text[:60]}")
                        from modules.wikimedia_client import WikimediaClient
                        wiki = WikimediaClient()
                        wiki_results = wiki.search_public_records_by_text(all_sign_text, limit=3)
                        for wr in wiki_results:
                            if wr.get("lat") and wr.get("lon"):
                                all_estimates.append({
                                    "latitude": wr.get("lat"),
                                    "longitude": wr.get("lon"),
                                    "confidence": 0.8,
                                    "sources": [f"sign_detection:{wr.get('title')}"],
                                    "evidence": {"sign_text": all_sign_text, "matched_title": wr.get('title')},
                                    "phase": "SignDetection"
                                })
            except Exception as e:
                logger.error(f'  [-] Sign detection skipped/failed: {e}', exc_info=True)
                
            try:
                vehicle_data = fut_vd.result()
                if vehicle_data.get('status') == 'success' and vehicle_data.get('driving_side_estimate'):
                    phases['vehicle_detection'] = vehicle_data
                    logger.info(f"  ✓ Vehicle driving side: {vehicle_data.get('driving_side_estimate')}")
            except Exception as e:
                logger.error(f'  [-] Vehicle detection skipped/failed: {e}', exc_info=True)
                
            try:
                veg_data = fut_vc.result()
                if veg_data.get('status') == 'success':
                    phases['vegetation_classification'] = veg_data
            except Exception as e:
                logger.error(f'  [-] Vegetation classification skipped/failed: {e}', exc_info=True)
                
            try:
                terrain_data = fut_ta.result()
                if terrain_data.get('status') == 'success':
                    phases['terrain_analysis'] = terrain_data
            except Exception as e:
                logger.error(f'  [-] Terrain analysis skipped/failed: {e}', exc_info=True)

        # ── f) Candidate Coordinates Validation ──
        if all_estimates:
            try:
                from modules.elevation_client import ElevationClient
                from modules.climate_analyzer import ClimateAnalyzer
                from modules.osm_feature_matcher import OSMFeatureMatcher
                
                ec = ElevationClient()
                ca = ClimateAnalyzer()
                osm = OSMFeatureMatcher()
                
                clim_est = phases.get('vegetation_classification', {}).get('climate_estimate')
                
                def _val_candidate(est):
                    lat, lon = est['latitude'], est['longitude']
                    elev, clim, poi = None, None, None
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as inner_exec:
                        f_elev = inner_exec.submit(ec.get_elevation, lat, lon)
                        f_clim = inner_exec.submit(ca.analyze_location, lat, lon)
                        f_poi = inner_exec.submit(osm.find_pois, lat, lon)
                        
                        try: elev = f_elev.result()
                        except Exception as e: logger.error(f'Elevation validation failed: {e}', exc_info=True)
                        
                        try: clim = f_clim.result()
                        except Exception as e: logger.error(f'Climate validation failed: {e}', exc_info=True)
                        
                        try: poi = f_poi.result()
                        except Exception as e: logger.error(f'POI validation failed: {e}', exc_info=True)
                        
                    if not isinstance(est.get('evidence'), dict):
                        est['evidence'] = {}
                    est['evidence']['elevation'] = elev
                    est['evidence']['climate_consistency'] = clim
                    est['evidence']['poi_density'] = poi
                    return est
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as val_exec:
                    futures = [val_exec.submit(_val_candidate, est) for est in all_estimates[:3]]
                    for f in concurrent.futures.as_completed(futures):
                        try: f.result()
                        except Exception as e: logger.error(f'Candidate validation failed: {e}', exc_info=True)
                        
            except Exception as e:
                logger.error(f'  [-] Candidate validation skipped/failed: {e}', exc_info=True)

        # ── Advanced OSINT: Weather Corroboration ──
        # Cross-reference the final estimates with historical weather if EXIF date is available
        try:
            from modules.weather_corroborator import WeatherCorroborator
            weather_checker = WeatherCorroborator()
            
            # Extract weather heuristics from scene classification if available
            extracted_weather = ""
            scene = phases.get("scene_classification", {})
            if scene.get("status") == "success":
                top_classes = [c.get("class", "").lower() for c in scene.get("top_5", [])]
                for tc in top_classes:
                    if any(w in tc for w in ["snow", "ski", "alp", "ice", "glacier"]):
                        extracted_weather = "snow"
                        break
                    elif any(w in tc for w in ["umbrella", "rain", "storm"]):
                        extracted_weather = "rain"
                        break
                    elif any(w in tc for w in ["sunglass", "sunscreen", "beach", "desert", "sand"]):
                        extracted_weather = "sunny"
                        break
                        
            all_estimates = weather_checker.verify_candidates(all_estimates, exif, extracted_weather, date_override=options.get('date'))
        except Exception as e:
            logger.error(f"  [-] Weather corroboration skipped/failed: {e}", exc_info=True)

        # ── Advanced OSINT: Reverse Geocode Validation ──
        try:
            from modules.nominatim_geocoder import NominatimGeocoder
            geocoder = NominatimGeocoder()
            
            sorted_ests = sorted(all_estimates, key=lambda x: x["confidence"], reverse=True)
            top_3 = sorted_ests[:3]
            
            def _validate_geocode(est):
                try:
                    return geocoder.validate_estimate(est["latitude"], est["longitude"])
                except Exception as e:
                    logger.error(f'Nominatim failed: {e}', exc_info=True)
                    return {"is_land": True}
                    
            validated_estimates = []
            import concurrent.futures
            
            # Since top_3 might have elements also in all_estimates, we validate them.
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as geo_exec:
                # Keep a map of est -> validation result
                # but est is a dict, so use its id or just a list of futures
                futures = {geo_exec.submit(_validate_geocode, est): est for est in top_3}
                val_results = {}
                for f in concurrent.futures.as_completed(futures):
                    est = futures[f]
                    try:
                        val = f.result()
                        if val.get("is_land", True):
                            if val.get("country") or val.get("state"):
                                if not isinstance(est.get("evidence"), dict):
                                    est["evidence"] = {}
                                if val.get("country"):
                                    est["evidence"]["country"] = val["country"]
                                if val.get("state"):
                                    est["evidence"]["state"] = val["state"]
                            val_results[id(est)] = True
                        else:
                            val_results[id(est)] = False
                    except Exception as e:
                        logger.error(f"Geocode processing failed: {e}", exc_info=True); val_results[id(est)] = True
                        
            for est in all_estimates:
                if est in top_3:
                    if val_results.get(id(est), True):
                        validated_estimates.append(est)
                else:
                    validated_estimates.append(est)
                    
            all_estimates = validated_estimates
        except Exception as e:
            logger.error(f"  [-] Reverse Geocode Validation skipped/failed: {e}", exc_info=True)

        # Merge nearby estimates
        merged = []
        fusion_success = False
        try:
            from modules.location_reasoner import LocationReasoner, LocationEstimate
            reasoner = LocationReasoner(cluster_radius_km=15.0)
            
            estimates_objects = []
            for est in all_estimates:
                estimates_objects.append(LocationEstimate(
                    latitude=est.get("latitude", 0.0),
                    longitude=est.get("longitude", 0.0),
                    confidence=est.get("confidence", 0.5),
                    match_sources=est.get("sources", []),
                    supporting_evidence=est.get("evidence", {})
                ))
            
            fused_objects = reasoner.estimate_location(estimates_objects)
            if fused_objects:
                for fo in fused_objects:
                    merged.append({
                        "latitude": fo.latitude,
                        "longitude": fo.longitude,
                        "confidence": fo.confidence,
                        "sources": fo.match_sources,
                        "evidence": fo.supporting_evidence,
                        "phase": "Synthesis"
                    })
                fusion_success = True
        except Exception as e:
            logger.error(f'  [-] LocationReasoner fusion skipped/failed: {e}', exc_info=True)

        if not fusion_success:
            for est in all_estimates:
                lat, lon = est["latitude"], est["longitude"]
                found = False
                for existing in merged:
                    if abs(existing["latitude"] - lat) < 0.05 and abs(existing["longitude"] - lon) < 0.05:
                        existing["confidence"] = max(existing["confidence"], est["confidence"])
                        for s in est.get("sources", []):
                            if s not in existing["sources"]:
                                existing["sources"].append(s)
                        if isinstance(est.get("evidence"), dict):
                            existing["evidence"].update(est["evidence"])
                        found = True
                        break
                if not found:
                    merged.append({
                        "latitude": lat, "longitude": lon,
                        "confidence": est["confidence"],
                        "sources": est.get("sources", []),
                        "evidence": est.get("evidence", {}),
                        "phase": est.get("phase", "unknown"),
                    })

        merged.sort(key=lambda x: x["confidence"], reverse=True)
        result["location_estimates"] = merged[:7]
        if merged:
            result["best_estimate"] = merged[0]

        if merged:
            confs = [m["confidence"] for m in merged]
            result["confidence_summary"] = {
                "highest": round(max(confs), 4),
                "lowest": round(min(confs), 4),
                "average": round(sum(confs) / len(confs), 4),
                "estimates_count": len(merged),
            }

        all_sources = set()
        for m in merged:
            all_sources.update(m["sources"])
        result["evidence_summary"] = {
            "unique_sources": list(all_sources),
            "source_count": len(all_sources),
            "phases_contributing": list(set(m["phase"] for m in merged)),
        }

        completed = [k for k, v in phases.items() if v.get("status") == "success"]
        failed = [k for k, v in phases.items() if v.get("status") == "failed"]
        limited = [k for k, v in phases.items() if v.get("status") == "limited"]
        skipped = [k for k, v in phases.items() if v.get("status") == "skipped"]

        result["execution_summary"] = {
            "phases_completed": completed,
            "phases_failed": failed,
            "phases_limited": limited,
            "phases_skipped": skipped,
            "total_phases": len(phases),
            "success_rate": f"{len(completed)}/{len(phases)}",
        }

        result["status"] = "success"
        logger.info(f"  ✓ Synthesis complete — {len(merged)} estimates, "
                     f"best conf: {result['confidence_summary'].get('highest', 0):.4f}")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"  ✗ Phase 8 error: {e}")

    return result
# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------


def generate_html_report(pipeline_result: PipelineResult, output_path: str) -> str:
    """Generate interactive HTML report with map, features, and estimates."""
    logger.info("  Generating HTML report...")

    result = pipeline_result.to_dict()
    best = result.get("best_estimate", {}) or {}
    estimates = result.get("location_estimates", []) or []
    phases_completed = result.get("phases_completed", []) or []
    phases_failed = result.get("phases_failed", []) or []

    map_center_lat = best.get("latitude", 0) if best else 0
    map_center_lon = best.get("longitude", 0) if best else 0

    # Build map
    map_html = ""
    if FOLIUM_AVAILABLE and folium:
        try:
            m = folium.Map(location=[map_center_lat, map_center_lon],
                           zoom_start=11, control_scale=True)
            if best:
                lat = best.get('latitude')
                lon = best.get('longitude')
                if lat is not None and lon is not None:
                    folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>Best Estimate</b><br>Conf: {best.get('confidence', 0.0):.3f}",
                    icon=folium.Icon(color="red", icon="star")).add_to(m)
            colors = ["blue", "green", "purple", "orange", "darkred", "lightred", "beige"]
            for i, est in enumerate(estimates[:7]):
                lat, lon = est.get("latitude"), est.get("longitude")
                if lat is None or lon is None: continue
                if lat == 0 and lon == 0: continue
                c = colors[i % len(colors)]
                src = "<br>".join(est.get("sources", []))
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8 + int(est.get("confidence", 0.5) * 10),
                    popup=f"<b>#{i+1}</b><br>Conf: {est.get('confidence', 0.0):.3f}<br>{src}",
                    color=c, fill=True, fill_opacity=0.6).add_to(m)
            pk = result.get("park_proximity", {})
            if pk and pk.get("parks"):
                for p in pk["parks"][:5]:
                    lat = p.get("latitude", p.get("lat", 0))
                    lon = p.get("longitude", p.get("lon", 0))
                    if lat is not None and lon is not None:
                        folium.Marker(
                            location=[lat, lon],
                            popup=f"<b>🌳 {p['name']}</b><br>{p.get('distance_km', 0.0):.3f} km",
                            icon=folium.Icon(color="green", icon="leaf")).add_to(m)
            map_html = m._repr_html_()
        except Exception as e:
            logger.error(f"Folium error: {e}", exc_info=True)

    if not map_html:
        map_html = (f'<iframe width="100%" height="500" '
                    f'src="https://www.openstreetmap.org/export/embed.html?bbox='
                    f'{map_center_lon - 0.1},{map_center_lat - 0.1},'
                    f'{map_center_lon + 0.1},{map_center_lat + 0.1}'
                    f'&layer=mapnik" style="border:1px solid #ccc"></iframe>')

    visual = result.get("visual_features", {}).get("features", {})
    visual_json = json.dumps(visual, indent=2, default=str)[:2000]
    ocr_text = result.get("ocr_text", {}).get("text", "No text extracted")
    ocr_sig = result.get("ocr_text", {}).get("significant_text", "")

    # Phase badges
    phase_names = [
        "phase1b_shadow_analysis", "phase1_visual_features", "phase2_ocr", "phase3_deep_features",
        "phase3b_reverse_image_search", "phase4_db_matching", "phase4b_property_records",
        "phase5_satellite_matching", "phase6_park_proximity",
        "phase7_cross_verification", "phase9_synthesis",
    ]
    phase_labels = [
        "Shadow Analysis", "Visual Features", "OCR Extraction", "Deep Features",
        "Reverse Image Search", "DB Matching", "Property Records",
        "Satellite Matching", "Park Proximity",
        "Cross-Verification", "Synthesis",
    ]
    phase_badges = ""
    for pn, pl in zip(phase_names, phase_labels):
        if pn in phases_completed:
            badge = '<span class="badge badge-success">&#10003;</span>'
        elif pn in phases_failed:
            badge = '<span class="badge badge-failed">&#10007;</span>'
        else:
            badge = '<span class="badge badge-limited">&mdash;</span>'
        phase_badges += f"<li>{badge} {pl}</li>"

    # Estimates table
    est_rows = ""
    for i, est in enumerate(estimates[:7]):
        lat, lon, conf = est.get("latitude", 0.0), est.get("longitude", 0.0), est.get("confidence", 0.0)
        src = ", ".join(est.get("sources", []))
        ph = est.get("phase", "?")
        link = f"https://www.google.com/maps/@{lat},{lon},17z"
        est_rows += f"""<tr><td>#{i+1}</td><td>{lat:.4f}</td><td>{lon:.4f}</td>
        <td>{conf:.3f}</td><td>{ph}</td><td><small>{src}</small></td>
        <td><a href="{link}" target="_blank">&#118279;</a></td></tr>"""



    # Parks table
    pk_rows = ""
    park_data = result.get("park_proximity") or {}
    for p in (park_data.get("parks", []) or [])[:5]:
        pk_rows += f"""<tr><td>{p.get('name', '?')}</td>
        <td>{p.get('distance_km', 0.0):.3f} km</td>
        <td>{p.get('walk_minutes', 0)} min</td>
        <td>{'&#9989;' if p.get('within_8_10_min') else '&#10060;'}</td></tr>"""

    # Errors section
    errors_html = ""
    for e in result.get("errors", []):
        errors_html += f'<div class="error-box">{e}</div>'
    if errors_html:
        errors_html = f'<div class="section"><h2>&#9888;&#65039; Errors</h2>{errors_html}</div>'

    # Build full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GeoVision Deep Scan Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#0d1117; color:#c9d1d9; line-height:1.6; }}
  .container {{ max-width:1100px; margin:0 auto; padding:20px; }}
  h1 {{ color:#58a6ff; font-size:28px; margin-bottom:5px; }}
  h2 {{ color:#58a6ff; font-size:20px; margin:25px 0 15px;
        border-bottom:1px solid #30363d; padding-bottom:8px; }}
  .header {{ text-align:center; padding:30px 0; }}
  .header .subtitle {{ color:#8b949e; font-size:14px; }}
  .header .timestamp {{ color:#484f58; font-size:11px; }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
                gap:15px; margin:20px 0; }}
  .stat-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
               padding:20px; text-align:center; }}
  .stat-card .value {{ font-size:32px; font-weight:bold; color:#58a6ff; }}
  .stat-card .label {{ font-size:11px; color:#8b949e; text-transform:uppercase; }}
  .best-estimate {{ background:linear-gradient(135deg,#1a2332,#161b22);
                    border:1px solid #58a6ff; border-radius:11px; padding:25px; text-align:center; }}
  .best-estimate .coords {{ font-size:24px; color:#f0f6fc; }}
  .best-estimate .conf {{ font-size:18px; color:#58a6ff; }}
  .map-container {{ background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden; }}
  table {{ width:100%; border-collapse:collapse; margin:15px 0; }}
  th,td {{ border:1px solid #30363d; padding:10px 11px; text-align:left; font-size:14px; }}
  th {{ background:#1a2332; color:#58a6ff; font-weight:600; }}
  tr:nth-child(even) {{ background:#161b22; }}
  tr:hover {{ background:#1c2333; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
  .badge-success {{ background:#1a3a2a; color:#3fb950; }}
  .badge-failed {{ background:#3a1a1a; color:#f85149; }}
  .badge-limited {{ background:#3a2a1a; color:#d29922; }}
  .code-block {{ background:#0d1117; border:1px solid #30363d; border-radius:6px;
                padding:16px; font-family:'SF Mono',monospace; font-size:13px; overflow-x:auto; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin:20px 0; }}
  .error-box {{ background:#3a1a1a; border:1px solid #f85149; border-radius:6px; padding:11px; }}
  .footer {{ text-align:center; padding:30px 0; color:#484f58; font-size:11px; }}
  a {{ color:#58a6ff; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  ul {{ list-style:none; }}
  li {{ padding:6px 0; font-size:14px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>&#118752; GeoVision Deep Scan</h1>
    <div class="subtitle">GeoSpy-Class Geolocation Pipeline Report</div>
    <div class="timestamp">Generated: {result.get('timestamp', 'unknown')}</div>
  </div>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{len(estimates)}</div><div class="label">Location Estimates</div></div>
    <div class="stat-card"><div class="value">{len(phases_completed)}/11</div><div class="label">Phases Completed</div></div>
    <div class="stat-card"><div class="value">{best.get('confidence', 0.0):.3f}</div><div class="label">Best Confidence</div></div>
    <div class="stat-card"><div class="value">{len(park_data.get('parks', []))}</div><div class="label">Parks Found</div></div>
  </div>
  <div class="best-estimate">
    <h2>&#118205; Best Estimate</h2>
    <div class="coords">{best.get('latitude', 0.0):.6f}, {best.get('longitude', 0.0):.6f}</div>
    <div class="conf">Confidence: {best.get('confidence', 0.0):.3f}</div>
    <div style="margin-top:10px;font-size:13px;color:#8b949e">
      Sources: {', '.join(best.get('sources', ['N/A']))}
    </div>
    <div style="margin-top:15px">
      <a href="https://www.google.com/maps/@{best.get('latitude', 0)},{best.get('longitude', 0)},18z"
         target="_blank" style="background:#58a6ff;color:#0d1117;padding:8px 20px;
         border-radius:6px;text-decoration:none;font-weight:600">&#117758; Google Maps</a>
      <a href="https://www.google.com/maps/@{best.get('latitude', 0)},{best.get('longitude', 0)},3a,75y,90h,90t"
         target="_blank" style="background:#30363d;color:#c9d1d9;padding:8px 20px;
         border-radius:6px;text-decoration:none;font-weight:600;margin-left:10px">&#118247; Street View</a>
    </div>
  </div>
  <div class="map-container">{map_html}</div>
  <div class="section">
    <h2>&#118205; Ranked Location Estimates</h2>
    <table><thead><tr><th>Rank</th><th>Latitude</th><th>Longitude</th><th>Confidence</th><th>Phase</th><th>Sources</th><th>Maps</th></tr></thead>
    <tbody>{est_rows if est_rows else '<tr><td colspan="7">No estimates generated</td></tr>'}</tbody></table>
  </div>
  <div class="section">
    <h2>&#118304; Pipeline Phase Status</h2>
    <ul>{phase_badges}</ul>
  </div>
  </div>
  <div class="section">
    <h2>&#118202; Visual Features</h2>
    <div class="code-block">{visual_json}</div>
  </div>
  <div class="section">
    <h2>&#118260; OCR Text</h2>
    <p><strong>Text:</strong> {ocr_text[:500]}</p>
    {f'<p><strong>Significant:</strong> {ocr_sig[:300]}</p>' if ocr_sig else ''}
  </div>
  <div class="section">
    <h2>&#117796; Nearby Parks</h2>
    {f'<table><thead><tr><th>Park</th><th>Distance</th><th>Walk</th><th>8-10 min</th></tr></thead><tbody>{pk_rows}</tbody></table>' if pk_rows else '<p>No parks found in proximity analysis</p>'}
  </div>
  <div class="section">
    <h2>&#119504; Deep Features</h2>
    <p><strong>Method:</strong> {(result.get('deep_features') or {}).get('method', 'N/A')}</p>
    <p><strong>Norm:</strong> {(result.get('deep_features') or {}).get('feature_norm', 'N/A')}</p>
    <p><strong>Dim:</strong> {(result.get('deep_features') or {}).get('feature_dim', 'N/A')}</p>
  </div>
  {errors_html}
  <div class="section">
    <h2>&#118193; Image</h2>
    <p><strong>Path:</strong> {result.get('image_path', 'N/A')}</p>
    <p><strong>Success:</strong> {'&#9989; Yes' if result.get('success') else '&#10060; No'}</p>
  </div>
  <div class="footer">GeoVision Deep Scan v2.0 &bull; Generated by GeoVision Geolocation System</div>
</div>
</body>
</html>"""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    logger.info(f"  &#10003; HTML report: {output_file}")
    return str(output_file.resolve())


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(image_path: str, options: Optional[Dict[str, Any]] = None) -> PipelineResult:
    """Run the full GeoVision deep scan pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    options : dict, optional
        Pipeline options:
            - output_dir  : Directory for output files.
            - near_park   : Enable park proximity filtering.
            - region      : Narrow search to specific region.

            - interactive : Open browser for verification.
            - verbose     : Enable verbose logging.

    Returns
    -------
    PipelineResult
    """
    options = options or {}
    output_dir = Path(options.get("output_dir", DEFAULT_OUTPUT_DIR))
    near_park = options.get("near_park", False)
    region = options.get("region", None)
    interactive = options.get("interactive", False)
    verbose = options.get("verbose", False)

    if verbose:
        logging.getLogger("GeoVisionDeepScan").setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path_resolved = str(Path(image_path).resolve())
    options["image_path"] = image_path_resolved

    pipeline_result = PipelineResult(
        image_path=image_path_resolved,
        timestamp=timestamp,
        success=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║     🛰️  GeoVision Deep Scan — GeoSpy Pipeline v2.0       ║")
    logger.info("╚" + "═" * 60 + "╝")
    logger.info(f"  Image: {image_path_resolved}")
    logger.info(f"  Output: {output_dir}")
    logger.debug(f"  Options: near_park={near_park}, region={region}, "
                 f"interactive={interactive}")
    logger.info("")

    # Validate image
    if not os.path.isfile(image_path_resolved):
        pipeline_result.errors.append(f"Image not found: {image_path_resolved}")
        return pipeline_result

    file_size_mb = os.path.getsize(image_path_resolved) / (1024 * 1024)
    if file_size_mb > 50:
        pipeline_result.errors.append(f"Image too large: {file_size_mb:.1f} MB (max 50)")
        return pipeline_result

    phases: Dict[str, Dict[str, Any]] = {}

    # Phase 0: EXIF Metadata
    try:
        pipeline_result.exif_data = phase0_exif_extraction(image_path_resolved)
        phases["exif_data"] = pipeline_result.exif_data
        if pipeline_result.exif_data.get("status") == "success":
            pipeline_result.phases_completed.append("phase0_exif_extraction")
        else:
            pipeline_result.phases_failed.append("phase0_exif_extraction")
    except Exception as e:
        pipeline_result.phases_failed.append("phase0_exif_extraction")
        pipeline_result.errors.append(f"Phase 0 error: {e}")
        phases["exif_data"] = {"status": "failed", "error": str(e)}

    # Phase 1b: Shadow Analysis
    try:
        pipeline_result.shadow_analysis = phase1b_shadow_analysis(pipeline_result.exif_data)
        phases["shadow_analysis"] = pipeline_result.shadow_analysis
        if pipeline_result.shadow_analysis.get("status") == "success":
            pipeline_result.phases_completed.append("phase1b_shadow_analysis")
        else:
            pipeline_result.phases_failed.append("phase1b_shadow_analysis")
    except Exception as e:
        pipeline_result.phases_failed.append("phase1b_shadow_analysis")
        pipeline_result.errors.append(f"Phase 1b error: {e}")
        phases["shadow_analysis"] = {"status": "failed", "error": str(e)}

    # Phase 1: Visual Features
    try:
        pipeline_result.visual_features = phase1_visual_features(image_path_resolved)
        phases["visual_features"] = pipeline_result.visual_features
        if pipeline_result.visual_features.get("status") == "success":
            pipeline_result.phases_completed.append("phase1_visual_features")
        else:
            pipeline_result.phases_failed.append("phase1_visual_features")
    except Exception as e:
        pipeline_result.phases_failed.append("phase1_visual_features")
        pipeline_result.errors.append(f"Phase 1 error: {e}")
        phases["visual_features"] = {"status": "failed", "error": str(e)}

    # Phase 2: OCR
    try:
        pipeline_result.ocr_text = phase2_ocr(image_path_resolved)
        phases["ocr_text"] = pipeline_result.ocr_text
        if pipeline_result.ocr_text.get("status") in ("success", "limited"):
            pipeline_result.phases_completed.append("phase2_ocr")
        else:
            pipeline_result.phases_failed.append("phase2_ocr")
    except Exception as e:
        pipeline_result.phases_failed.append("phase2_ocr")
        pipeline_result.errors.append(f"Phase 2 error: {e}")
        phases["ocr_text"] = {"status": "failed", "error": str(e)}

    # Phase 3: Deep Features
    try:
        pipeline_result.deep_features = phase3_deep_features(image_path_resolved)
        phases["deep_features"] = pipeline_result.deep_features
        if pipeline_result.deep_features.get("status") == "success":
            pipeline_result.phases_completed.append("phase3_deep_features")
        else:
            pipeline_result.phases_failed.append("phase3_deep_features")
    except Exception as e:
        pipeline_result.phases_failed.append("phase3_deep_features")
        pipeline_result.errors.append(f"Phase 3 error: {e}")
        phases["deep_features"] = {"status": "failed", "error": str(e)}

    # Phase 3b: Reverse Image Search
    try:
        pipeline_result.reverse_image_search = phase3b_reverse_image_search(image_path_resolved)
        phases["reverse_image_search"] = pipeline_result.reverse_image_search
        if pipeline_result.reverse_image_search.get("status") == "success":
            pipeline_result.phases_completed.append("phase3b_reverse_image_search")
        else:
            pipeline_result.phases_failed.append("phase3b_reverse_image_search")
    except Exception as e:
        pipeline_result.phases_failed.append("phase3b_reverse_image_search")
        pipeline_result.errors.append(f"Phase 3b error: {e}")
        phases["reverse_image_search"] = {"status": "failed", "error": str(e)}

    # Phase 4: DB Matching
    try:
        vis_for_db = pipeline_result.visual_features.get("features", {})
        if not vis_for_db:
            vis_for_db = pipeline_result.visual_features
        pipeline_result.db_matches = phase4_db_matching(vis_for_db, pipeline_result.ocr_text, pipeline_result.reverse_image_search, region)
        phases["db_matches"] = pipeline_result.db_matches
        if pipeline_result.db_matches.get("status") in ("success", "limited"):
            pipeline_result.phases_completed.append("phase4_db_matching")
        else:
            pipeline_result.phases_failed.append("phase4_db_matching")
    except Exception as e:
        pipeline_result.phases_failed.append("phase4_db_matching")
        pipeline_result.errors.append(f"Phase 4 error: {e}")
        phases["db_matches"] = {"status": "failed", "error": str(e)}


    # Gather candidate coordinates from all prior sources and region flag
    candidate_coords = []
    
    # 1. EXIF GPS
    if pipeline_result.exif_data.get("gps"):
        gps = pipeline_result.exif_data["gps"]
        if "latitude" in gps and "longitude" in gps:
            candidate_coords.append({"latitude": gps["latitude"], "longitude": gps["longitude"]})
            
    # 2. Region flag
    if region:
        try:
            from modules.nominatim_geocoder import NominatimGeocoder
            geocoder = NominatimGeocoder()
            val = geocoder.forward_geocode(region)
            if val and val.get("latitude") and val.get("longitude"):
                candidate_coords.append({"latitude": val["latitude"], "longitude": val["longitude"]})
                logger.info(f"  ✓ Region '{region}' geocoded to {val['latitude']}, {val['longitude']}")
        except Exception as e:
            logger.error(f"  [-] Failed to geocode region '{region}': {e}", exc_info=True)
            
    # 3. Phase 4 DB Matches
    if pipeline_result.db_matches.get("matches"):
        for m in pipeline_result.db_matches["matches"]:
            if "latitude" in m and "longitude" in m:
                candidate_coords.append({"latitude": m["latitude"], "longitude": m["longitude"]})
                
    # Remove duplicates
    seen_coords = set()
    unique_candidates = []
    for c in candidate_coords:
        key = (round(c["latitude"], 4), round(c["longitude"], 4))
        if key not in seen_coords:
            seen_coords.add(key)
            unique_candidates.append(c)
    candidate_coords = unique_candidates

    # Phase 4b: Property Records
    try:
        pipeline_result.property_records = phase4b_property_records(candidate_coords)
        phases["property_records"] = pipeline_result.property_records
        if pipeline_result.property_records.get("status") == "success":
            pipeline_result.phases_completed.append("phase4b_property_records")
        else:
            pipeline_result.phases_failed.append("phase4b_property_records")
    except Exception as e:
        pipeline_result.phases_failed.append("phase4b_property_records")
        pipeline_result.errors.append(f"Phase 4b error: {e}")
        phases["property_records"] = {"status": "failed", "error": str(e)}

    # Phase 5: Satellite Matching
    try:
        vis_for_sat = pipeline_result.visual_features.get("features", {})
        if not vis_for_sat:
            vis_for_sat = pipeline_result.visual_features
        pipeline_result.satellite_matches = phase5_satellite_matching(vis_for_sat, candidate_coords)
        phases["satellite_matches"] = pipeline_result.satellite_matches
        if pipeline_result.satellite_matches.get("status") in ("success", "limited"):
            pipeline_result.phases_completed.append("phase5_satellite_matching")
        else:
            pipeline_result.phases_failed.append("phase5_satellite_matching")
    except Exception as e:
        pipeline_result.phases_failed.append("phase5_satellite_matching")
        pipeline_result.errors.append(f"Phase 5 error: {e}")
        phases["satellite_matches"] = {"status": "failed", "error": str(e)}

    # Phase 6: Park Proximity
    try:
        vis_for_pk = pipeline_result.visual_features.get("features", {})
        if not vis_for_pk:
            vis_for_pk = pipeline_result.visual_features
        pipeline_result.park_proximity = phase6_park_proximity(vis_for_pk, pipeline_result.satellite_matches, near_park, candidate_coords)
        phases["park_proximity"] = pipeline_result.park_proximity
        if pipeline_result.park_proximity.get("status") in ("success", "limited"):
            pipeline_result.phases_completed.append("phase6_park_proximity")
        else:
            pipeline_result.phases_failed.append("phase6_park_proximity")
    except Exception as e:
        pipeline_result.phases_failed.append("phase6_park_proximity")
        pipeline_result.errors.append(f"Phase 6 error: {e}")
        phases["park_proximity"] = {"status": "failed", "error": str(e)}

    # Phase 7: Cross-View Verification
    prelim_estimates = []
    if pipeline_result.satellite_matches.get("matches"):
        for m in pipeline_result.satellite_matches["matches"][:3]:
            prelim_estimates.append({
                "latitude": m["latitude"], "longitude": m["longitude"],
                "confidence": m.get("confidence", 0.5),
            })
    
    if not prelim_estimates and candidate_coords:
        for c in candidate_coords[:3]:
            prelim_estimates.append({
                "latitude": c["latitude"], "longitude": c["longitude"],
                "confidence": 0.5,
            })


    try:
        pipeline_result.cross_verification = phase7_cross_verification(prelim_estimates, interactive)
        phases["cross_verification"] = pipeline_result.cross_verification
        if pipeline_result.cross_verification.get("status") in ("success", "limited"):
            pipeline_result.phases_completed.append("phase7_cross_verification")
        else:
            pipeline_result.phases_failed.append("phase7_cross_verification")
    except Exception as e:
        pipeline_result.phases_failed.append("phase7_cross_verification")
        pipeline_result.errors.append(f"Phase 7 error: {e}")
        phases["cross_verification"] = {"status": "failed", "error": str(e)}

    # Phase 9: Synthesis
    try:
        pipeline_result.synthesis_report = phase9_synthesis(phases, options)
        phases["synthesis_report"] = pipeline_result.synthesis_report
        if pipeline_result.synthesis_report.get("status") == "success":
            pipeline_result.best_estimate = pipeline_result.synthesis_report.get("best_estimate", {})
            pipeline_result.location_estimates = pipeline_result.synthesis_report.get("location_estimates", [])
            pipeline_result.phases_completed.append("phase9_synthesis")
        else:
            pipeline_result.phases_failed.append("phase9_synthesis")
    except Exception as e:
        pipeline_result.phases_failed.append("phase9_synthesis")
        pipeline_result.errors.append(f"Phase 9 error: {e}")
        phases["synthesis_report"] = {"status": "failed", "error": str(e)}

    # Overall success
    pipeline_result.success = len(pipeline_result.phases_completed) >= 5

    # Save JSON
    json_path = output_dir / f"geovision_scan_{timestamp}.json"
    try:
        json_path.write_text(pipeline_result.to_json(), encoding="utf-8")
        logger.info(f"  ✓ JSON: {json_path}")
    except Exception as e:
        pipeline_result.errors.append(f"JSON save error: {e}")

    # Generate HTML report
    html_path = output_dir / f"geovision_scan_{timestamp}.html"
    try:
        generate_html_report(pipeline_result, str(html_path))
        logger.info(f"  ✓ HTML: {html_path}")
    except Exception as e:
        pipeline_result.errors.append(f"HTML report error: {e}")
        logger.error(f"  ✗ HTML report error: {e}")

    # Summary
    logger.info("")
    logger.info("╔" + "═" * 60 + "╗")
    logger.info("║                Pipeline Complete                        ║")
    logger.info("╚" + "═" * 60 + "╝")
    logger.info(f"  Phases completed: {len(pipeline_result.phases_completed)}/8")
    logger.info(f"  Phases failed:    {len(pipeline_result.phases_failed)}")
    logger.info(f"  Location estimates: {len(pipeline_result.location_estimates)}")
    if pipeline_result.best_estimate:
        be = pipeline_result.best_estimate
        logger.info(f"  Best estimate: ({be.get('latitude', '?'):.4f}, "
                     f"{be.get('longitude', '?'):.4f}) "
                     f"confidence={be.get('confidence', 0):.3f}")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  HTML: {html_path}")
    logger.info("")

    return pipeline_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GeoVision Deep Scan — GeoSpy-class Geolocation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s image.jpg
  %(prog)s image.jpg --output-dir ./reports --near-park
  %(prog)s image.jpg --region Toronto
  %(prog)s image.jpg --interactive --verbose
        """,
    )
    parser.add_argument("image", type=str, help="Path to the input image file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--near-park", action="store_true", default=False,
                        help="Enable park proximity filtering")
    parser.add_argument("--region", type=str, default=None,
                        help="Narrow search to a specific region")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Open browser for interactive verification")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose (debug) logging")
    parser.add_argument("--json-only", action="store_true", default=False,
                        help="Output only JSON to stdout (no HTML report)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1

    options = {
        "output_dir": args.output_dir,
        "near_park": args.near_park,
        "region": args.region,
        "interactive": args.interactive,
        "verbose": args.verbose,
    }

    if args.json_only:
        # Silence all loggers to ensure clean JSON output on stdout
        logging.getLogger().setLevel(logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)
        
    result = run_pipeline(args.image, options)

    if args.json_only:
        print(result.to_json())
    else:
        be = result.best_estimate
        if be:
            print(f"Best estimate: ({be.get('latitude', '?'):.4f}, "
                  f"{be.get('longitude', '?'):.4f}) "
                  f"confidence={be.get('confidence', 0):.3f}", file=sys.stderr)
        print(f"Phases: {len(result.phases_completed)}/8 completed, "
              f"{len(result.phases_failed)} failed", file=sys.stderr)
        print(f"Output: {args.output_dir}", file=sys.stderr)

    return 0 if result.success else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())

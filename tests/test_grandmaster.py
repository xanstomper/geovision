"""
Tests for Grandmaster Forensics & Zero-Storage Geolocation Engine
=================================================================
Validates:
  - GrandmasterForensicsEngine instantiation and offline execution
  - Visual forensics extraction (<100ms OpenCV classifiers)
  - Hard negative-evidence falsification lattice (driving side, solar hemisphere)
  - Spherical DBSCAN spatial clustering on GPS coordinates
  - Multi-scale hierarchical crop extraction (global, center, horizon, road, grid)
  - JIT micro-GIS topological verification resilience
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from modules.grandmaster_forensics import GrandmasterForensicsEngine
from modules.patch_geo_predictor import (
    _dbscan_gps,
    _haversine_km,
    extract_multiscale_crops,
    PatchGeoPredictor,
)

TEST_IMG = os.path.join(ROOT, "test_building.jpg")
if not os.path.exists(TEST_IMG):
    # The old hand-dropped fixture is gone (disk cleanup); fall back to the
    # real committed eval image so the tests still exercise a real photo.
    TEST_IMG = os.path.join(
        ROOT, "data", "eval", "wikipedia_landmarks_v1", "images", "building_test.jpg"
    )


def test_grandmaster_engine_instantiation():
    engine = GrandmasterForensicsEngine(device="cpu")
    assert engine.device == "cpu"
    assert engine._heuristics_analyzer is None  # lazy loaded


def test_extract_visual_forensics():
    engine = GrandmasterForensicsEngine(device="cpu")
    fb = engine.extract_visual_forensics(TEST_IMG)

    assert "driving_side" in fb
    assert "road_markings" in fb
    assert "utility_pole" in fb
    assert "license_plate" in fb
    assert "soil_and_biome" in fb
    assert "solar_shadow" in fb

    # Verify score ranges
    assert 0.0 <= fb["driving_side"].get("confidence", 0.0) <= 1.0
    assert 0.0 <= fb["road_markings"].get("confidence", 0.0) <= 1.0
    assert 0.0 <= fb["solar_shadow"].get("confidence", 0.0) <= 1.0


def test_falsification_lattice_driving_side():
    engine = GrandmasterForensicsEngine(device="cpu")

    # Mock forensics: Left-hand driving detected with 80% confidence
    mock_forensics = {
        "driving_side": {"driving_side": "left", "confidence": 0.80},
        "license_plate": {},
        "road_markings": {},
        "soil_and_biome": {},
        "solar_shadow": {},
    }

    # Candidate 1: London (Left-hand, GB)
    # Candidate 2: Paris (Right-hand, FR)
    # Candidate 3: New York (Right-hand, US)
    candidates = [
        {"latitude": 51.5074, "longitude": -0.1278, "source": "test_gb"},
        {"latitude": 48.8566, "longitude": 2.3522, "source": "test_fr"},
        {"latitude": 40.7128, "longitude": -74.0060, "source": "test_us"},
    ]

    surviving, eliminated = engine.apply_falsification_lattice(candidates, mock_forensics)

    # London must survive; Paris and NY must be eliminated
    surviving_countries = [c.get("country") for c in surviving]
    eliminated_countries = [e.get("country") for e in eliminated]

    assert "GB" in surviving_countries
    assert "FR" in eliminated_countries
    assert "US" in eliminated_countries
    assert len(eliminated) == 2

    # Check elimination reason text
    for e in eliminated:
        assert any("driving contradicts" in r.lower() for r in e.get("reasons", []))


def test_falsification_lattice_solar_hemisphere():
    engine = GrandmasterForensicsEngine(device="cpu")

    # Mock forensics: Sun shadow indicates Southern Hemisphere with 75% confidence
    mock_forensics = {
        "driving_side": {},
        "license_plate": {},
        "road_markings": {},
        "soil_and_biome": {},
        "solar_shadow": {"inferred_hemisphere": "southern", "confidence": 0.75},
    }

    # Candidate 1: Sydney (-33.86° S -> valid)
    # Candidate 2: Berlin (+52.52° N -> invalid, northern temperate)
    candidates = [
        {"latitude": -33.8688, "longitude": 151.2093, "source": "sydney"},
        {"latitude": 52.5200, "longitude": 13.4050, "source": "berlin"},
    ]

    surviving, eliminated = engine.apply_falsification_lattice(candidates, mock_forensics)

    surv_lats = [c["latitude"] for c in surviving]
    elim_lats = [e["lat"] for e in eliminated]

    assert -33.8688 in surv_lats
    assert 52.5200 in elim_lats
    assert any("southern hemisphere" in r.lower() for r in eliminated[0]["reasons"])


def test_spherical_dbscan_clustering():
    # 3 points near Paris (within ~2 km)
    paris_pts = [(48.8584, 2.2945), (48.8606, 2.3376), (48.8530, 2.3499)]
    # 2 points near Tokyo (~9700 km away)
    tokyo_pts = [(35.6762, 139.6503), (35.6895, 139.6917)]

    pts = paris_pts + tokyo_pts
    clusters = _dbscan_gps(pts, eps_km=15.0, min_samples=2)

    # Should find two distinct clusters (0 and 1)
    assert len(clusters) == 2
    c0_indices = clusters[0]
    c1_indices = clusters[1]

    # Cluster sizes must be 3 and 2
    sizes = sorted([len(c0_indices), len(c1_indices)])
    assert sizes == [2, 3]


def test_multiscale_crop_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        crops = extract_multiscale_crops(TEST_IMG, tmpdir, grid_size=3)

        # 1 full + 1 center + 1 horizon + 1 road + 9 grid = 13 crops
        assert len(crops) == 13

        names = [c["name"] for c in crops]
        assert "full_image" in names
        assert "center_zoom" in names
        assert "horizon_canopy" in names
        assert "road_infrastructure" in names
        assert "grid_0_0" in names
        assert "grid_2_2" in names

        # Verify all crop files actually exist and are readable
        for c in crops:
            assert os.path.exists(c["path"])
            assert os.path.getsize(c["path"]) > 100
            assert 0.0 < c["weight"] <= 1.0


def test_jit_micro_gis_verification():
    engine = GrandmasterForensicsEngine(device="cpu")
    # Coordinates for Eiffel Tower, Paris
    res = engine.jit_micro_gis_verification(48.8584, 2.2945, radius_m=500)

    assert res.get("status") == "success"
    assert res.get("lat") == 48.8584
    assert res.get("lon") == 2.2945
    assert isinstance(res.get("osm_features"), list)
    assert isinstance(res.get("ground_photos"), list)

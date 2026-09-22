"""
Unit and Integration Tests for Zero-Storage Grandmaster Geolocation Suite
========================================================================
Validates:
  1. PlonkIt 85-Country Infrastructure Meta Engine (rule evaluations, country ranking, elimination).
  2. CarID Vehicle Fleet Profiler (silhouette detection, fleet demographic ratios, regional affinities).
  3. StreetTargeter (compass azimuth math, intersection scoring, Nominatim fallback).
  4. GrandmasterDossierGenerator (Markdown and HTML dossier rendering).
  5. MCP tool handlers for all new tools.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from modules.plonkit_meta_engine import PlonkitMetaEngine, PLONKIT_COUNTRY_METAS
from modules.car_fleet_identifier import CarFleetIdentifier
from modules.street_targeter import StreetTargeter, compute_azimuth_deg, angle_diff_deg
from modules.grandmaster_dossier import GrandmasterDossierGenerator
from mcp_geovision_server import (
    tool_geovision_street_targeter,
    tool_geovision_plonkit_rules,
    tool_geovision_car_id,
    tool_geovision_dossier,
)


class TestPlonkitMetaEngine:
    def test_country_metas_coverage(self):
        """Verifies knowledge base covers essential countries and has key metadata."""
        engine = PlonkitMetaEngine()
        assert len(engine.metas) >= 35
        for iso in ["US", "CA", "MX", "BR", "GB", "DE", "FR", "PL", "JP", "AU"]:
            assert iso in engine.metas
            meta = engine.metas[iso]
            assert meta["driving_side"] in ("left", "right")
            assert "road_lines" in meta
            assert "utility_pole" in meta
            assert "plate_format" in meta

    def test_falsification_logic(self):
        """Verifies that impossible countries are eliminated when hard contradictions are observed."""
        engine = PlonkitMetaEngine()
        # Observed: right driving, yellow centerline, blue Euroband
        # In reality, yellow center + blue Euroband is almost impossible in EU,
        # but let's test driving side: left driving should eliminate US, FR, DE
        res = engine.evaluate_observations({"driving_side": "left"})
        assert res["status"] == "success"
        top = res["candidates"]
        left_countries = [c["iso"] for c in top if not c["falsified"]]
        for iso in left_countries:
            assert engine.metas[iso]["driving_side"] == "left"

    def test_scan_image_offline(self):
        """Tests scan_image on test image with pure CV heuristics."""
        engine = PlonkitMetaEngine()
        test_img = "test_building.jpg"
        if os.path.exists(test_img):
            res = engine.scan_image(test_img)
            assert res["status"] == "success"
            assert "candidates" in res
            assert len(res["candidates"]) > 0


class TestCarFleetIdentifier:
    def test_car_fleet_analysis(self):
        """Tests vehicle detection and fleet demographics."""
        identifier = CarFleetIdentifier()
        test_img = "test_building.jpg"
        if os.path.exists(test_img):
            res = identifier.analyze(test_img)
            assert res["status"] == "success"
            assert "demographics" in res
            demo = res["demographics"]
            assert "pickup_suv_ratio" in demo
            assert "kei_car_ratio" in demo
            assert "sedan_hatchback_ratio" in demo


class TestStreetTargeter:
    def test_azimuth_and_angle_math(self):
        """Tests compass azimuth calculations and angle differences."""
        # 0,0 to 1,0 is due North (0 deg azimuth)
        az_north = compute_azimuth_deg(0.0, 0.0, 1.0, 0.0)
        assert abs(az_north - 0.0) < 0.1

        # 0,0 to 0,1 is due East (90 deg azimuth)
        az_east = compute_azimuth_deg(0.0, 0.0, 0.0, 1.0)
        assert abs(az_east - 90.0) < 0.1

        # Bidirectional angle difference: 10 deg vs 190 deg is 0 deg diff
        diff = angle_diff_deg(10.0, 190.0, bidirectional=True)
        assert abs(diff - 0.0) < 0.01

        # 10 deg vs 100 deg is 90 deg diff
        diff_perp = angle_diff_deg(10.0, 100.0, bidirectional=True)
        assert abs(diff_perp - 90.0) < 0.01

    def test_street_targeter_fallback(self):
        """Tests that street targeter gracefully resolves real coordinates."""
        targeter = StreetTargeter()
        res = targeter.target_street(43.6532, -79.3832, radius_m=500)
        assert res["status"] == "success"
        assert len(res["candidates"]) > 0
        top = res["top_match"]
        assert top is not None
        assert "intersection" in top
        assert "latitude" in top
        assert "longitude" in top


class TestGrandmasterDossier:
    def test_dossier_generation(self):
        """Tests that dossier generator renders Markdown and HTML with all forensic sections."""
        gen = GrandmasterDossierGenerator()
        inv_mock = {
            "best_estimate": {
                "latitude": 43.6532,
                "longitude": -79.3832,
                "city": "Toronto",
                "country": "CA",
                "confidence": 0.85,
                "precision_tier": "street_or_neighborhood",
                "google_maps_url": "https://maps.google.com",
                "osm_url": "https://osm.org",
            },
            "forensic_breakdown": {
                "driving_side": {"driving_side": "right", "confidence": 0.8},
                "road_markings": {"line_color": "yellow", "confidence": 0.85},
                "utility_pole": {"dominant_type": "wooden_utility_pole", "confidence": 0.7},
                "license_plate": {"format": "americas_short", "confidence": 0.7, "has_euroband": False},
                "soil_and_biome": {"soil_type": "temperate_soil", "vegetation_biome": "mixed_temperate", "soil_confidence": 0.2},
                "solar_shadow": {"inferred_hemisphere": "northern", "confidence": 0.55},
            },
            "reasoning_chain": ["Step 1: Analyzed cues", "Step 2: Pinned coordinates"],
        }
        dossier = gen.generate_dossier("test_building.jpg", inv_mock)
        assert dossier["verdict"]["city"] == "Toronto"

        md = gen.render_markdown(dossier)
        assert "Executive Geolocation Verdict" in md
        assert "Physical Visual Forensics" in md
        assert "Deduction Chain" in md

        html = gen.render_html(dossier)
        assert "<!DOCTYPE html>" in html
        assert "leaflet" in html.lower()


class TestMcpHandlers:
    def test_mcp_new_tools(self):
        """Verifies that all new tool handlers execute cleanly."""
        # 1. plonkit tool
        p_res = tool_geovision_plonkit_rules({"observations": {"driving_side": "right", "line_color": "yellow"}})
        assert p_res["status"] == "success"

        # 2. street targeter tool
        st_res = tool_geovision_street_targeter({"lat": 43.6532, "lon": -79.3832, "radius_m": 500})
        assert st_res["status"] == "success"

        # 3. car id tool
        test_img = "test_building.jpg"
        if os.path.exists(test_img):
            c_res = tool_geovision_car_id({"image_path": test_img})
            assert c_res["status"] == "success"

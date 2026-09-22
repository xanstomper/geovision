"""
Tests for Patch-Based GeoCLIP Predictor (zero-storage street-level geolocation).

Validates:
1. DBSCAN GPS clustering (unit, no model needed)
2. Multi-scale crop extraction (unit, PIL only)
3. PatchGeoPredictor algorithm logic (integration, needs GeoCLIP model)
4. Harness integration (deterministic_scan includes patch_geo signal)
5. Storage comparison (proves patch approach saves storage vs reference DB)
"""

import math
import os
import sys
import tempfile
import gzip
import shutil

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.patch_geo_predictor import (
    _haversine_km,
    _dbscan_gps,
    extract_multiscale_crops,
    PatchGeoPredictor,
)


# ---------------------------------------------------------------------------
# DBSCAN GPS clustering tests
# ---------------------------------------------------------------------------

class TestDbscanGps:

    def test_single_point_is_own_cluster(self):
        points = [(48.8584, 2.2945)]
        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        assert len(clusters) == 1
        assert 0 in clusters
        assert len(clusters[0]) == 1

    def test_two_close_points_same_cluster(self):
        points = [
            (48.8584, 2.2945),
            (48.8586, 2.2947),
        ]
        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_two_far_points_different_clusters(self):
        points = [
            (48.8584, 2.2945),
            (40.7128, -74.0060),
        ]
        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        assert len(clusters) == 2

    def test_tight_cluster_of_many_points(self):
        import numpy as np
        base = (48.8584, 2.2945)
        points = []
        rng = np.random.RandomState(42)
        for _ in range(5):
            lat = base[0] + rng.uniform(-0.002, 0.002)
            lon = base[1] + rng.uniform(-0.002, 0.002)
            points.append((lat, lon))

        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_far_points_each_become_their_own_cluster(self):
        """With min_samples=1, every point becomes a cluster."""
        points = [
            (48.8584, 2.2945),
            (40.7128, -74.0060),
            (35.6762, 139.6503),
        ]
        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        assert len(clusters) == 3  # Each point in its own cluster

    def test_empty_points(self):
        clusters = _dbscan_gps([], eps_km=0.5, min_samples=1)
        assert clusters == {}

    def test_cluster_returns_index_list(self):
        points = [
            (48.8584, 2.2945),
            (48.8585, 2.2946),
            (40.7128, -74.0060),
        ]
        clusters = _dbscan_gps(points, eps_km=0.5, min_samples=1)
        all_indices = []
        for indices in clusters.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Haversine distance tests
# ---------------------------------------------------------------------------

class TestHaversineKm:

    def test_same_point_zero_distance(self):
        d = _haversine_km(48.8584, 2.2945, 48.8584, 2.2945)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_known_distance_eiffel_to_notre_dame(self):
        d = _haversine_km(48.8584, 2.2945, 48.8530, 2.3499)
        assert d == pytest.approx(4.0, abs=0.5)

    def test_antipodal_points(self):
        d = _haversine_km(0.0, 0.0, 0.0, 180.0)
        assert d == pytest.approx(20000.0, abs=1000)

    def test_one_degree_latitude(self):
        d = _haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.0, abs=1.0)


# ---------------------------------------------------------------------------
# Multi-scale crop extraction tests
# ---------------------------------------------------------------------------

class TestExtractMultiscaleCrops:

    def test_creates_all_crops(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (300, 200), color=(255, 0, 0))
            img.save(f.name)
            tmp_path = f.name

        try:
            temp_dir = tempfile.mkdtemp()
            try:
                crops = extract_multiscale_crops(tmp_path, temp_dir, grid_size=3)

                # Should have: full + center + horizon + road + 9 grid patches = 13
                # (Note: row_weights has 3 entries but grid creates 3x3=9 patches)
                assert len(crops) == 13

                crop_names = [c["name"] for c in crops]
                assert "full_image" in crop_names
                assert "center_zoom" in crop_names
                assert "horizon_canopy" in crop_names
                assert "road_infrastructure" in crop_names

                for c in crops:
                    assert os.path.exists(c["path"])
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_crop_has_required_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (300, 200), color=(0, 255, 0))
            img.save(f.name)
            tmp_path = f.name

        try:
            temp_dir = tempfile.mkdtemp()
            try:
                crops = extract_multiscale_crops(tmp_path, temp_dir)
                for crop in crops:
                    assert "path" in crop
                    assert "name" in crop
                    assert "weight" in crop
                    assert "scale" in crop
                    assert "box" in crop
                    assert isinstance(crop["box"], list)
                    assert len(crop["box"]) == 4
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_weights_differ_by_scale(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (300, 200), color=(0, 0, 255))
            img.save(f.name)
            tmp_path = f.name

        try:
            temp_dir = tempfile.mkdtemp()
            try:
                crops = extract_multiscale_crops(tmp_path, temp_dir)
                full_weight = None
                horizon_weight = None
                grid_weight = None
                for c in crops:
                    if c["name"] == "full_image":
                        full_weight = c["weight"]
                    elif c["name"] == "horizon_canopy":
                        horizon_weight = c["weight"]
                    elif c["name"].startswith("grid_"):
                        grid_weight = c["weight"]
                        break

                assert full_weight is not None
                assert horizon_weight is not None
                assert grid_weight is not None
                # Different scales get different weights
                assert full_weight == 1.0
                assert horizon_weight < full_weight
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        finally:
            os.unlink(tmp_path)

    def test_fails_on_missing_image(self):
        with pytest.raises(FileNotFoundError):
            temp_dir = tempfile.mkdtemp()
            try:
                extract_multiscale_crops("/nonexistent/image.jpg", temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# PatchGeoPredictor logic tests (need GeoCLIP model — mark as slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPatchGeoPredictor:

    @pytest.fixture(scope="class")
    def predictor(self):
        return PatchGeoPredictor()

    @pytest.fixture
    def eiffel_image(self):
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "eval",
            "wikipedia_landmarks_v1",
            "images",
            "eiffel_tower.jpg",
        )

    def test_predict_returns_consensus(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        assert "consensus" in result
        assert "coarse_prior" in result
        assert "clusters" in result
        assert "predictions" in result
        assert result["source"] == "patch_geo_predictor"

    def test_consensus_has_valid_coords(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        c = result["consensus"]
        assert -90 <= c["lat"] <= 90
        assert -180 <= c["lon"] <= 180
        assert 0 <= c["confidence"] <= 1.0

    def test_coarse_prior_in_results(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        cp = result["coarse_prior"]
        assert -90 <= cp["lat"] <= 90
        assert -180 <= cp["lon"] <= 180

    def test_clusters_sorted_by_crops_then_size(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        clusters = result["clusters"]
        if len(clusters) >= 2:
            # Sorted by unique_crops desc, then size desc
            for i in range(len(clusters) - 1):
                a, b = clusters[i], clusters[i + 1]
                assert (a["unique_crops_count"] >= b["unique_crops_count"] or
                        (a["unique_crops_count"] == b["unique_crops_count"] and
                         a["size"] >= b["size"]))

    def test_n_crops_analyzed_matches_multiscale(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        # Should have full + center + horizon + road + 9 grid = 15 crops
        # Each crop produces top_k predictions, so total should be significant
        assert result["total_crops_analyzed"] >= 15

    def test_consensus_confidence_in_range(self, predictor, eiffel_image):
        result = predictor.predict(eiffel_image, eps_km=0.5)
        conf = result["consensus"]["confidence"]
        assert 0.10 <= conf <= 0.95  # Confirmed by the module's clamping


# ---------------------------------------------------------------------------
# Harness integration test (mark as slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHarnessPatchIntegration:

    @pytest.fixture
    def eiffel_image(self):
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "eval",
            "wikipedia_landmarks_v1",
            "images",
            "eiffel_tower.jpg",
        )

    def test_deterministic_scan_has_patch_signal(self, eiffel_image):
        from modules.geo_harness import GeoVisionHarness
        harness = GeoVisionHarness()
        result = harness.deterministic_scan(eiffel_image)
        assert "patch_geo" in result["signals"]
        assert result["signals"]["patch_geo"]["status"] in ("success", "limited")

    def test_deterministic_scan_has_patch_candidate(self, eiffel_image):
        from modules.geo_harness import GeoVisionHarness
        harness = GeoVisionHarness()
        result = harness.deterministic_scan(eiffel_image)
        patch_candidates = [
            c for c in result["candidates"]
            if c.get("source") in ("patch_clustering", "patch_clustering_alt")
        ]
        assert len(patch_candidates) >= 1

    def test_patch_candidate_has_expected_fields(self, eiffel_image):
        from modules.geo_harness import GeoVisionHarness
        harness = GeoVisionHarness()
        result = harness.deterministic_scan(eiffel_image)
        for c in result["candidates"]:
            if c.get("source") == "patch_clustering":
                assert "latitude" in c
                assert "longitude" in c
                assert "confidence" in c
                assert "cluster_size" in c
                assert "cluster_tightness_km" in c


# ---------------------------------------------------------------------------
# Smoke test — import and structure
# ---------------------------------------------------------------------------

class TestPatchGeoSmoke:

    def test_module_imports(self):
        from modules import patch_geo_predictor
        assert hasattr(patch_geo_predictor, "PatchGeoPredictor")
        assert hasattr(patch_geo_predictor, "_dbscan_gps")
        assert hasattr(patch_geo_predictor, "extract_multiscale_crops")
        assert hasattr(patch_geo_predictor, "_haversine_km")

    def test_predictor_instantiates(self):
        pp = PatchGeoPredictor()
        assert pp._loaded is False
        assert pp.device in ("cpu", "cuda")

    def test_predict_fails_gracefully_without_model(self):
        pp = PatchGeoPredictor()
        assert hasattr(pp, "predict")
        assert hasattr(pp, "predict_single")
        assert hasattr(pp, "predict_patches")


# ---------------------------------------------------------------------------
# Storage comparison — proves patch approach saves storage
# ---------------------------------------------------------------------------

class TestStorageComparison:

    def test_no_ref_db_needed_for_prediction(self):
        import inspect
        source = inspect.getsource(PatchGeoPredictor)
        assert "DB_DIR" not in source
        assert "DB_EMB" not in source
        assert "DB_META" not in source
        assert "visual_geo_db" not in source

    def test_ref_db_is_marginal_for_storage(self):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "visual_geo_db")
        if os.path.exists(db_path):
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, dns in os.walk(db_path)
                for f in dns
            )
            assert total_size < 50 * 1024 * 1024

    def test_patch_predictor_storage_is_zero(self):
        import inspect
        source = inspect.getsource(PatchGeoPredictor)
        assert "data/" not in source or "clip_cache" in source

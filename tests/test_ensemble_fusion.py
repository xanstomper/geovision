"""Tests for Ensemble Spatial-Consensus Fusion (beats GeoSpy photo-DB, zero storage)."""

import os
import sys
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ensemble_fusion import EnsembleFusion, _validate, FAMILY_NAMES


def _cand(lat, lon, conf, source, country=None):
    return {"latitude": lat, "longitude": lon, "confidence": conf,
            "source": source, "country": country}


class TestValidate:
    def test_valid(self):
        assert _validate(48.8, 2.3)
        assert _validate(-33.8, 151.2)

    def test_invalid(self):
        assert not _validate(91, 0)
        assert not _validate(0, 181)
        assert not _validate(-91, 0)


class TestFusionBasics:
    def test_empty(self):
        out = EnsembleFusion().fuse([])
        assert out["verdict"] is None
        assert out["candidates"] == []

    def test_no_valid_candidates(self):
        out = EnsembleFusion().fuse([{"latitude": None, "longitude": None}])
        assert out["verdict"] is None

    def test_single_candidate(self):
        out = EnsembleFusion().fuse([_cand(48.8584, 2.2945, 0.7, "geoclip")])
        v = out["verdict"]
        assert abs(v["latitude"] - 48.8584) < 1e-6
        assert abs(v["longitude"] - 2.2945) < 1e-6
        assert v["confidence"] > 0

    def test_dedupes_by_source_family(self):
        # 9 patch clusters all same family = 1 location vote, not 9
        cands = [_cand(48.858, 2.294, 0.8, "patch_clustering") for _ in range(9)]
        out = EnsembleFusion().fuse(cands)
        v = out["verdict"]
        # one independent family → lower independence multiplier
        assert v["independent_families"] == ["location"]
        assert v["confidence"] < 0.9  # not over-boosted by duplicate family


class TestCrossSignalAgreement:
    def test_independent_agreement_boosts(self):
        # 3 DIFFERENT signal families agreeing near Paris
        cands = [
            _cand(48.8584, 2.2945, 0.6, "geoclip"),
            _cand(48.8570, 2.2951, 0.6, "patch_clustering"),
            _cand(48.8600, 2.2920, 0.6, "vlm_verify", country="France"),
        ]
        out = EnsembleFusion().fuse(cands)
        v = out["verdict"]
        # 3 independent families (location, location, vlm)
        assert len(v["independent_families"]) >= 2
        assert v["confidence"] >= 0.6
        # centroid near Paris
        assert abs(v["latitude"] - 48.858) < 0.05
        assert abs(v["longitude"] - 2.294) < 0.05

    def test_disagreeing_sources_dont_pool(self):
        # far apart candidates → separate clusters, no false agreement boost
        cands = [
            _cand(48.8584, 2.2945, 0.9, "geoclip"),
            _cand(40.7128, -74.0060, 0.9, "geoclip"),
        ]
        out = EnsembleFusion(eps_km=25.0).fuse(cands)
        v = out["verdict"]
        assert len(out["candidates"]) == 2
        # each kept its own base confidence, no agreement inflation across continents
        assert v["confidence"] <= 0.98

    def test_country_agreement(self):
        cands = [
            _cand(48.8584, 2.2945, 0.5, "geoclip", "France"),
            _cand(48.8530, 2.3499, 0.5, "geoclip", "France"),
        ]
        out = EnsembleFusion().fuse(cands)
        assert out["verdict"]["country_agreement"] == 1.0


class TestApplyToHarness:
    def test_dropin(self):
        pruned = [
            _cand(48.8584, 2.2945, 0.6, "patch_clustering"),
            _cand(48.8570, 2.2951, 0.6, "geoclip"),
        ]
        out = EnsembleFusion().apply_to_harness(pruned)
        assert out["verdict"] is not None
        assert out["verdict"]["latitude"] is not None

    def test_empty_harness(self):
        out = EnsembleFusion().apply_to_harness([])
        assert out["verdict"] is None


class TestStorageFree:
    def test_no_reference_db(self):
        import inspect
        src = inspect.getsource(EnsembleFusion)
        assert "visual_geo_db" not in src
        assert "DB_EMB" not in src

    def test_source_families_mapping(self):
        assert FAMILY_NAMES["patch_clustering"] == "location"
        assert FAMILY_NAMES["geoclip"] == "location"
        assert FAMILY_NAMES["vlm_verify"] == "vlm"
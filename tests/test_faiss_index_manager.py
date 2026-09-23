import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.faiss_index_manager import FaissIndexManager, FAISS_AVAILABLE
from modules.visual_geo_engine import VisualGeoEngine, DB_FAISS


def _generate_test_embeddings(n=100, dim=64, seed=42):
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def test_faiss_availability():
    """Verify FAISS detection status."""
    assert FaissIndexManager.is_available() == FAISS_AVAILABLE


def test_ip_search_faiss_and_numpy_agreement():
    """FAISS and NumPy fallback must yield identical top-k nearest neighbors."""
    embeddings = _generate_test_embeddings(100, 64, seed=123)
    query = embeddings[15].copy()  # Query that should match index 15 exactly

    # 1. FAISS manager
    mgr_faiss = FaissIndexManager(metric="IP")
    mgr_faiss.build_index(embeddings)
    idx_f, dist_f = mgr_faiss.search(query, k=5)

    # 2. NumPy manager (forced fallback)
    mgr_np = FaissIndexManager(metric="IP")
    mgr_np.build_index(embeddings)
    # Simulate FAISS unavailable by removing index
    mgr_np.index = None
    idx_np, dist_np = mgr_np.search(query, k=5)

    assert idx_f[0] == 15
    assert idx_np[0] == 15
    assert np.allclose(dist_f[0], 1.0, atol=1e-5)
    assert np.allclose(dist_np[0], 1.0, atol=1e-5)

    np.testing.assert_array_equal(idx_f, idx_np)
    np.testing.assert_allclose(dist_f, dist_np, atol=1e-5)


def test_l2_search_faiss_and_numpy_agreement():
    """FAISS L2 and NumPy L2 search must match."""
    embeddings = _generate_test_embeddings(100, 64, seed=456)
    query = embeddings[22].copy()

    mgr_faiss = FaissIndexManager(metric="L2")
    mgr_faiss.build_index(embeddings)
    idx_f, dist_f = mgr_faiss.search(query, k=5)

    mgr_np = FaissIndexManager(metric="L2")
    mgr_np.build_index(embeddings)
    mgr_np.index = None
    idx_np, dist_np = mgr_np.search(query, k=5)

    assert idx_f[0] == 22
    assert idx_np[0] == 22
    assert np.allclose(dist_f[0], 0.0, atol=1e-5)
    assert np.allclose(dist_np[0], 0.0, atol=1e-5)

    np.testing.assert_array_equal(idx_f, idx_np)
    np.testing.assert_allclose(dist_f, dist_np, atol=1e-5)


def test_save_and_load_index():
    """Verify index can be serialized to disk and loaded back."""
    embeddings = _generate_test_embeddings(50, 32, seed=789)
    query = embeddings[5].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir) / "test_idx.bin"
        mgr = FaissIndexManager(index_path=idx_path, metric="IP")
        mgr.build_index(embeddings)
        assert mgr.save_index()
        assert idx_path.exists()

        # Load into a new manager
        mgr2 = FaissIndexManager(index_path=idx_path, metric="IP")
        assert mgr2.load_index()
        idx, dist = mgr2.search(query, k=3)
        assert idx[0] == 5
        assert np.isclose(dist[0], 1.0, atol=1e-5)


def test_sync_or_build_mtime_rebuild():
    """sync_or_build must rebuild index when embeddings file is modified."""
    embeddings = _generate_test_embeddings(40, 32, seed=101)

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_path = Path(tmpdir) / "embeddings.npy"
        idx_path = Path(tmpdir) / "index.bin"

        np.save(emb_path, embeddings)
        mgr = FaissIndexManager(index_path=idx_path, metric="IP")
        mgr.sync_or_build(embeddings=embeddings, embeddings_path=emb_path, index_path=idx_path)
        assert idx_path.exists()
        old_mtime = idx_path.stat().st_mtime

        # Sleep briefly and touch / rewrite embeddings to simulate an update
        time.sleep(0.05)
        new_embeddings = _generate_test_embeddings(50, 32, seed=202)
        np.save(emb_path, new_embeddings)

        # Make sure emb_path mtime is strictly greater than idx_path
        os.utime(emb_path, (time.time() + 1.0, time.time() + 1.0))

        # Re-sync
        mgr.sync_or_build(embeddings=new_embeddings, embeddings_path=emb_path, index_path=idx_path)
        assert mgr.index.ntotal == 50
        assert idx_path.stat().st_mtime >= old_mtime


def test_batch_query_shape():
    """search with 2D query array must return (B, k) shaped arrays."""
    embeddings = _generate_test_embeddings(60, 32, seed=303)
    queries = embeddings[:4]  # shape (4, 32)

    mgr = FaissIndexManager(metric="IP")
    mgr.build_index(embeddings)

    indices, distances = mgr.search(queries, k=5)
    assert indices.shape == (4, 5)
    assert distances.shape == (4, 5)
    assert np.all(indices[:, 0] == np.array([0, 1, 2, 3]))


def test_clamped_k_when_k_greater_than_db():
    """k larger than database size must be clamped without throwing errors."""
    embeddings = _generate_test_embeddings(10, 16, seed=404)
    query = embeddings[0]

    mgr = FaissIndexManager(metric="IP")
    mgr.build_index(embeddings)

    indices, distances = mgr.search(query, k=25)
    assert len(indices) == 10
    assert len(distances) == 10


def test_visual_geo_engine_faiss_integration():
    """Verify VisualGeoEngine utilizes FaissIndexManager and cached binary index."""
    engine = VisualGeoEngine()
    loaded = engine._ensure_db()
    assert loaded
    assert engine.db_size() > 0
    assert DB_FAISS.exists()

    # Search with a known reference vector
    ref_emb = engine._db_embeddings[0]
    order, sims = engine.search_neighbors(ref_emb, top_k=5)
    assert len(order) == 5
    assert len(sims) == 5
    assert order[0] == 0
    assert np.isclose(sims[0], 1.0, atol=1e-4)

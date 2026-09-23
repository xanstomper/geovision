# FaissIndexManager — FAISS-accelerated CLIP embedding search
# Drops in as replacement for numpy dot-product NN search
# Falls back transparently when faiss not installed

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Transparent FAISS detection
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, Exception) as e:
    faiss = None
    FAISS_AVAILABLE = False
    logger.debug("FAISS unavailable (%s); falling back to NumPy", e)


class FaissIndexManager:
    """
    FaissIndexManager — FAISS-accelerated CLIP embedding search
    Drops in as replacement for numpy dot-product NN search
    Falls back transparently when faiss not installed
    """

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        metric: str = "IP",
    ):
        """
        Args:
            index_path: Optional default path to cached binary FAISS index file.
            metric: "IP" (Inner Product / Cosine similarity for unit vectors) or "L2".
        """
        self.index_path: Optional[Path] = Path(index_path).resolve() if index_path else None
        self.metric: str = metric.upper()
        if self.metric not in ("IP", "L2"):
            raise ValueError(f"Unsupported metric: {metric}. Must be 'IP' or 'L2'.")

        self.index: Any = None
        self._embeddings: Optional[np.ndarray] = None
        self.use_faiss: bool = FAISS_AVAILABLE

    @staticmethod
    def is_available() -> bool:
        """Returns True if FAISS is installed and available in the current environment."""
        return FAISS_AVAILABLE

    def is_faiss_active(self) -> bool:
        """Returns True if an active FAISS index object is instantiated in memory."""
        return bool(FAISS_AVAILABLE and self.index is not None)

    def build_index(
        self,
        embeddings: np.ndarray,
        metric: Optional[str] = None,
    ) -> Any:
        """
        Build an index from an embeddings array [N, D].
        Uses FAISS Flat index if available, else prepares embeddings for NumPy fallback.

        Args:
            embeddings: float32 array [N, D] of embeddings (assumed L2-normalized for IP).
            metric: "IP" or "L2" (overrides default metric if provided).

        Returns:
            The FAISS index object if FAISS is available, or None if in NumPy fallback mode.
        """
        if metric is not None:
            self.metric = metric.upper()
            if self.metric not in ("IP", "L2"):
                raise ValueError(f"Unsupported metric: {metric}. Must be 'IP' or 'L2'.")

        # Retain C-contiguous float32 array for search / fallback
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if not FAISS_AVAILABLE:
            logger.info(
                "FAISS not installed; initialized NumPy search fallback (%d vectors)",
                len(self._embeddings),
            )
            self.index = None
            return None

        try:
            n_vecs, dim = self._embeddings.shape
            if self.metric == "IP":
                idx = faiss.IndexFlatIP(dim)
            else:
                idx = faiss.IndexFlatL2(dim)

            idx.add(self._embeddings)
            self.index = idx
            logger.info(
                "Built FAISS Flat%s index with %d vectors (dim=%d)",
                self.metric,
                n_vecs,
                dim,
            )
            return self.index
        except Exception as e:
            logger.warning("Failed building FAISS index: %s. Falling back to NumPy.", e)
            self.index = None
            return None

    def save_index(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save the current FAISS index to disk.

        Args:
            path: Destination file path. If None, uses self.index_path.

        Returns:
            True if successfully saved, False otherwise.
        """
        target = Path(path).resolve() if path else self.index_path
        if target is None:
            logger.warning("No index path specified for save_index.")
            return False

        if not FAISS_AVAILABLE or self.index is None:
            logger.debug("Cannot save FAISS index: FAISS unavailable or index not built.")
            return False

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(target))
            logger.info("Saved FAISS index to %s", target)
            return True
        except Exception as e:
            logger.error("Failed saving FAISS index to %s: %s", target, e)
            return False

    def load_index(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load a cached FAISS index from disk.

        Args:
            path: Source file path. If None, uses self.index_path.

        Returns:
            True if successfully loaded, False otherwise.
        """
        target = Path(path).resolve() if path else self.index_path
        if target is None or not target.exists():
            return False

        if not FAISS_AVAILABLE:
            logger.debug("FAISS unavailable; cannot load index from %s", target)
            return False

        try:
            self.index = faiss.read_index(str(target))
            logger.info("Loaded FAISS index from %s (%d vectors)", target, self.index.ntotal)
            return True
        except Exception as e:
            logger.warning("Failed loading FAISS index from %s: %s", target, e)
            self.index = None
            return False

    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Alias for save_index."""
        return self.save_index(path)

    def load(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Alias for load_index."""
        return self.load_index(path)

    def sync_or_build(
        self,
        embeddings: np.ndarray,
        embeddings_path: Optional[Union[str, Path]] = None,
        index_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Synchronize FAISS index with disk cache:
        - If cached index exists and is newer than embeddings_path (mtime check), load it.
        - If stale, missing, size mismatched, or corrupted, rebuild and cache to disk.
        - Falls back gracefully to NumPy if FAISS is unavailable or encounters error.

        Args:
            embeddings: float32 array [N, D].
            embeddings_path: Path to the embeddings file (to check modification time).
            index_path: Path to the cached binary FAISS index file.

        Returns:
            The active FAISS index, or None if in NumPy fallback mode.
        """
        if index_path:
            self.index_path = Path(index_path).resolve()

        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        n_vecs = self._embeddings.shape[0]

        if not FAISS_AVAILABLE:
            logger.info("FAISS not available; using NumPy fallback for %d embeddings", n_vecs)
            self.index = None
            return None

        # Check if cache exists and is fresh
        if self.index_path and self.index_path.exists():
            rebuild_needed = False
            if embeddings_path:
                emb_p = Path(embeddings_path).resolve()
                if emb_p.exists():
                    try:
                        emb_mtime = emb_p.stat().st_mtime
                        idx_mtime = self.index_path.stat().st_mtime
                        if idx_mtime < emb_mtime:
                            logger.info(
                                "FAISS index cache (%s) is older than embeddings (%s); rebuilding.",
                                self.index_path,
                                emb_p,
                            )
                            rebuild_needed = True
                    except OSError as e:
                        logger.warning("Failed comparing mtime: %s", e)
                        rebuild_needed = True

            if not rebuild_needed:
                if self.load_index(self.index_path):
                    if self.index is not None and getattr(self.index, "ntotal", None) == n_vecs:
                        return self.index
                    logger.info(
                        "Cached index size mismatch (ntotal=%s vs N=%d); rebuilding.",
                        getattr(self.index, "ntotal", None),
                        n_vecs,
                    )

        # Rebuild index and save to cache
        self.build_index(self._embeddings)
        if self.index is not None and self.index_path:
            self.save_index(self.index_path)

        return self.index

    def search(
        self,
        query_emb: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors for query embedding(s).

        Args:
            query_emb: 1D array [D] or 2D array [B, D] of query embeddings.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (indices, distances):
            - If query_emb is 1D:
                indices: [k] int64 array of neighbor indices
                distances: [k] float32 array of scores/distances
            - If query_emb is 2D:
                indices: [B, k] int64 array
                distances: [B, k] float32 array
            For metric="IP", distances are inner products (cosine similarities for unit vectors).
            For metric="L2", distances are squared Euclidean distances.
        """
        arr = np.asarray(query_emb, dtype=np.float32)
        is_1d = (arr.ndim == 1)
        if is_1d:
            q = np.ascontiguousarray(arr.reshape(1, -1), dtype=np.float32)
        else:
            q = np.ascontiguousarray(arr, dtype=np.float32)

        # Try FAISS search if active
        if FAISS_AVAILABLE and self.index is not None:
            try:
                ntotal = int(self.index.ntotal)
                if ntotal == 0:
                    empty_i = np.empty((0,), dtype=np.int64) if is_1d else np.empty((q.shape[0], 0), dtype=np.int64)
                    empty_d = np.empty((0,), dtype=np.float32) if is_1d else np.empty((q.shape[0], 0), dtype=np.float32)
                    return empty_i, empty_d

                k_clamped = min(k, ntotal)
                distances, indices = self.index.search(q, k_clamped)
                if is_1d:
                    return indices[0], distances[0]
                return indices, distances
            except Exception as e:
                logger.warning("FAISS search error (%s); falling back to NumPy", e)

        # NumPy fallback
        return self._search_numpy(q, k, is_1d=is_1d)

    def _search_numpy(
        self,
        q: np.ndarray,
        k: int,
        is_1d: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy implementation of nearest neighbor search."""
        if self._embeddings is None:
            raise RuntimeError("FaissIndexManager has no embeddings available for search.")

        n_vecs = self._embeddings.shape[0]
        if n_vecs == 0:
            empty_i = np.empty((0,), dtype=np.int64) if is_1d else np.empty((q.shape[0], 0), dtype=np.int64)
            empty_d = np.empty((0,), dtype=np.float32) if is_1d else np.empty((q.shape[0], 0), dtype=np.float32)
            return empty_i, empty_d

        k_clamped = min(k, n_vecs)

        if self.metric == "IP":
            # Inner product / Cosine similarity (larger is closer)
            sims = np.matmul(q, self._embeddings.T)  # [B, N]
            order = np.argsort(-sims, axis=1)[:, :k_clamped]
            scores = np.take_along_axis(sims, order, axis=1)
            if is_1d:
                return order[0].astype(np.int64), scores[0].astype(np.float32)
            return order.astype(np.int64), scores.astype(np.float32)
        else:
            # L2 squared distance (smaller is closer)
            q_sq = np.sum(q ** 2, axis=1, keepdims=True)  # [B, 1]
            emb_sq = np.sum(self._embeddings ** 2, axis=1, keepdims=True).T  # [1, N]
            dists = np.maximum(0.0, q_sq + emb_sq - 2.0 * np.matmul(q, self._embeddings.T))
            order = np.argsort(dists, axis=1)[:, :k_clamped]
            scores = np.take_along_axis(dists, order, axis=1)
            if is_1d:
                return order[0].astype(np.int64), scores[0].astype(np.float32)
            return order.astype(np.int64), scores.astype(np.float32)

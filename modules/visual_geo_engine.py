"""
Visual Geo Engine — the GeoSpy-class core
==========================================
Image -> CLIP embedding -> cosine nearest-neighbors against a reference DB
of REAL geotagged photos (Wikimedia Commons), yielding location estimates
with honest confidence derived from neighbor agreement.

No mocks. No synthetic data. Every reference image is a real photograph
with real coordinates from Wikimedia Commons.

Reference DB layout (data/visual_geo_db/):
    embeddings.npy   float32 [N, D]  L2-normalized CLIP embeddings
    meta.jsonl.gz    one JSON per line: {"lat","lon","title","city","country","source"}
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DB_DIR = SCRIPT_DIR / "data" / "visual_geo_db"
DB_EMB = DB_DIR / "embeddings.npy"
DB_META = DB_DIR / "meta.jsonl.gz"
DB_FAISS = DB_DIR / "faiss_index.bin"
CLIP_CACHE = SCRIPT_DIR / "data" / "clip_cache"

# Mean Earth radius in km
R_EARTH_KM = 6371.0088


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


class VisualGeoEngine:
    """CLIP + geotagged reference DB -> location estimates."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if _torch_cuda() else "cpu")
        self._model = None
        self._preprocess = None
        self._db_embeddings: Optional[np.ndarray] = None
        self._db_meta: List[Dict[str, Any]] = []
        self._index_manager = None

    # ------------------------------------------------------------------
    # CLIP
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return True
        try:
            import torch
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(CLIP_CACHE)
            )
            model.eval().to(self.device)
            self._model = model
            self._preprocess = preprocess
            logger.info("CLIP ViT-B-32 (laion2b) loaded on %s", self.device)
            return True
        except Exception as e:
            logger.warning("CLIP unavailable: %s", e)
            return False

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """CLIP embedding for an image file, L2-normalized float32 [512]."""
        if not self._ensure_model():
            return None
        import torch
        from PIL import Image
        try:
            img = Image.open(image_path).convert("RGB")
            tens = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._model.encode_image(tens)
            emb = emb.squeeze(0).float().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            return (emb / norm).astype(np.float32)
        except Exception as e:
            logger.error("embed_image failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Reference DB
    # ------------------------------------------------------------------
    def db_size(self) -> int:
        return 0 if self._db_embeddings is None else int(self._db_embeddings.shape[0])

    def _ensure_db(self) -> bool:
        if self._db_embeddings is not None and self._index_manager is not None:
            return True
        if not (DB_EMB.exists() and DB_META.exists()):
            logger.info("No visual geo DB at %s", DB_DIR)
            return False
        try:
            self._db_embeddings = np.load(DB_EMB).astype(np.float32)
            meta = []
            with gzip.open(DB_META, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        meta.append(json.loads(line))
            self._db_meta = meta
            if len(meta) != self._db_embeddings.shape[0]:
                logger.error(
                    "DB mismatch: %d embeddings vs %d meta rows",
                    self._db_embeddings.shape[0], len(meta),
                )
                self._db_embeddings = None
                return False

            try:
                from modules.faiss_index_manager import FaissIndexManager
                self._index_manager = FaissIndexManager(
                    index_path=DB_FAISS,
                    metric="IP",
                )
                self._index_manager.sync_or_build(
                    embeddings=self._db_embeddings,
                    embeddings_path=DB_EMB,
                    index_path=DB_FAISS,
                )
            except Exception as fe:
                logger.warning("FAISS initialization failed (%s); using numpy fallback", fe)
                self._index_manager = None

            backend = "FAISS (IP)" if (self._index_manager and self._index_manager.is_faiss_active()) else "NumPy fallback"
            logger.info("Loaded visual geo DB: %d references [%s]", len(meta), backend)
            return True
        except Exception as e:
            logger.error("Failed loading visual geo DB: %s", e)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_neighbors(self, query: np.ndarray, top_k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors for a query CLIP embedding [512].
        Returns (indices, similarities) using FAISS or NumPy fallback.
        """
        if not self._ensure_db():
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        if self._index_manager is not None:
            try:
                return self._index_manager.search(query, k=top_k)
            except Exception as e:
                logger.warning("Index manager search failed (%s); falling back to direct numpy", e)

        # Direct NumPy fallback (embeddings are L2-normalized)
        sims = self._db_embeddings @ query
        order = np.argsort(-sims)[:top_k]
        return order, sims[order]

    def locate(self, image_path: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Locate an image against the reference DB.
        Returns dict with status, estimates (lat/lon/confidence/city/country),
        and neighbor stats. Confidence reflects real neighbor agreement —
        no invented numbers.
        """
        backend = "FAISS" if (self._index_manager and self._index_manager.is_faiss_active()) else "NumPy"
        result: Dict[str, Any] = {
            "engine": f"CLIP ViT-B-32 (laion2b) + Wikimedia Commons reference DB [{backend}]",
            "backend": backend,
            "status": "failed",
            "db_size": 0,
            "estimates": [],
        }
        if not self._ensure_db():
            result["note"] = "Reference DB not built. Run scripts/build_reference_db.py"
            return result
        result["db_size"] = self.db_size()
        backend = "FAISS" if (self._index_manager and self._index_manager.is_faiss_active()) else "NumPy"
        result["backend"] = backend
        result["engine"] = f"CLIP ViT-B-32 (laion2b) + Wikimedia Commons reference DB [{backend}]"

        query = self.embed_image(image_path)
        if query is None:
            result["note"] = "Could not embed query image"
            return result

        # Nearest neighbor search (FAISS accelerated with automatic numpy fallback)
        order, sims = self.search_neighbors(query, top_k=top_k)

        neighbors = []
        for idx, sim in zip(order, sims):
            m = self._db_meta[int(idx)]
            neighbors.append({
                "lat": float(m["lat"]),
                "lon": float(m["lon"]),
                "similarity": float(sim),
                "title": m.get("title", ""),
                "city": m.get("city", ""),
                "country": m.get("country", ""),
            })

        result["neighbors"] = neighbors

        # Cluster neighbors: agreement within 25 km counts toward one location.
        clusters: List[Tuple[float, float, List[Dict]]] = []
        for n in neighbors:
            placed = False
            for c in clusters:
                if _haversine_km(n["lat"], n["lon"], c[0], c[1]) <= 25.0:
                    c[2].append(n)
                    placed = True
                    break
            if not placed:
                clusters.append((n["lat"], n["lon"], [n]))

        # Score = mean similarity * agreement factor (support/k)
        estimates = []
        for lat, lon, members in clusters:
            mean_sim = float(np.mean([m["similarity"] for m in members]))
            agreement = len(members) / float(top_k)
            # Honest confidence: strong similarity required to score high,
            # agreement boosts multi-view evidence.
            conf = max(0.0, min(1.0, (mean_sim - 0.6) / 0.35)) * (0.5 + 0.5 * agreement)
            top = max(members, key=lambda m: m["similarity"])
            estimates.append({
                "latitude": float(np.mean([m["lat"] for m in members])),
                "longitude": float(np.mean([m["lon"] for m in members])),
                "confidence": round(conf, 4),
                "support": len(members),
                "mean_similarity": round(mean_sim, 4),
                "city": top.get("city", ""),
                "country": top.get("country", ""),
                "source": "visual_geo_engine",
            })

        estimates.sort(key=lambda e: e["confidence"], reverse=True)
        result["estimates"] = estimates[:5]
        result["status"] = "success" if estimates else "limited"
        if estimates:
            result["note"] = (
                f"Top match similarity {neighbors[0]['similarity']:.3f}; "
                f"DB size {self.db_size()}. Confidence derives from neighbor "
                f"similarity and spatial agreement — small DB => treat as coarse."
            )
        return result


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    engine = VisualGeoEngine()
    img = sys.argv[1] if len(sys.argv) > 1 else str(SCRIPT_DIR / "building_image.jpg")
    out = engine.locate(img)
    print(json.dumps(out, indent=2))

"""
GeoVision Harness — model-agnostic, agent-driven geolocation investigation engine
===================================================================================

The "massive harness": a single orchestration layer that ANY vision-capable model
or agent (Gemini, GPT, Claude, local LLaVA, Hermes, etc.) can drive to run a full
multi-stage geolocation investigation and produce a complete case file.

Design
------
Two signal classes are fused:

  * DETERMINISTIC (always runs, no API key, no black box):
      - GeoCLIP direct GPS regression
      - CLIP ViT-B-32 reference-DB nearest-neighbor (real geotagged photos)
      - GeoGuessr CV heuristics (driving side, road marks, plates, bollards, signs)
      - Environment/biome classifier
      - Spatial constraint solver (negative-evidence elimination lattice)
      - Ground-truth photo retrieval (Wikimedia Commons / Mapillary) for top candidates
      - Best-effort OSINT: reverse-image search, weather, satellite, chain stores

  * MODEL-AGNOSTIC VLM (optional, OpenAI-compatible endpoint — any model):
      Stage 1 — coarse reasoning: describe scene, country/region/city, negative
                evidence, solar geometry, vegetation. Produces priors + constraints.
      Stage 3 — verification: show the query next to retrieved ground-photo
                references for each candidate; cross-check and explain.
      Configured via GEOVISION_VLM_* env vars (see modules/vlm_geo_analyzer.py).
      If no VLM is configured, the harness runs on deterministic signal alone.

Any stage that fails is skipped cleanly — the harness ALWAYS returns a structured
result, never a fabricated one. Confidence reflects real evidence agreement.

Usage (any model with vision you supply the image path):
    from modules.geo_harness import GeoVisionHarness
    h = GeoVisionHarness()
    case = h.investigate("photo.jpg")            # full pipeline + case record
    case = h.investigate("photo.jpg", save_case=True, case_name="OP-1234")

CLI:  geovision harness IMAGE [--case-name NAME] [--save-case]
MCP:  investigate_image tool (see mcp_geovision_server.py)
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent.resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GeoVisionHarness:
    """Orchestrates a full, staged geolocation investigation of a single image."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if _cuda() else "cpu")
        self._vlm = None

    # ------------------------------------------------------------------ #
    # Stage 1 — Coarse VLM reasoning (model-agnostic, optional)           #
    # ------------------------------------------------------------------ #
    def coarse_reason(self, image_path: str,
                      evidence_summary: Optional[str] = None,
                      location_hint: Optional[str] = None) -> Dict[str, Any]:
        """Feature extraction + strict reasoning via any OpenAI-compatible VLM.

        Returns dict with status, features, prediction. Degrades to
        status='skipped' when no VLM is configured (deterministic stages run anyway).
        """
        from modules.vlm_geo_analyzer import vlm_analyze
        start = time.time()
        res = vlm_analyze(image_path, evidence_summary=evidence_summary,
                          location_hint=location_hint)
        res["stage"] = "coarse_reason"
        res["duration_s"] = round(time.time() - start, 2)
        return res

    # ------------------------------------------------------------------ #
    # Stage 2 — Deterministic mass-scan + constrained retrieval           #
    # ------------------------------------------------------------------ #
    def deterministic_scan(self, image_path: str) -> Dict[str, Any]:
        """Run all non-VLM signal generators in isolation; each may fail cleanly.

        Returns {stage, candidates[], signals{...}, constraints[list[str]]}.
        constraints: negative-evidence statements (e.g. 'no left-hand traffic')
        extracted from deterministic signal for the constraint solver.
        """
        out: Dict[str, Any] = {"stage": "deterministic_scan", "status": "success",
                               "candidates": [], "signals": {}, "constraints": []}
        p = str(image_path)

        # --- GeoCLIP direct GPS regression (returns list[{lat,lon,confidence}]) ---
        try:
            from modules.geoclip_predictor import GeoCLIPPredictor
            g = GeoCLIPPredictor(self.device)
            preds = g.predict(p, top_k=3) or []
            for e in preds:
                lat, lon = e.get("lat"), e.get("lon")
                if lat is None or lon is None:
                    continue
                out["candidates"].append({
                    "latitude": float(lat), "longitude": float(lon),
                    "confidence": float(e.get("confidence", 0.01)),
                    "source": "geoclip",
                })
            out["signals"]["geoclip"] = {"status": "success" if preds else "limited",
                                         "n": len(preds)}
        except Exception as e:
            out["signals"]["geoclip"] = {"status": "failed", "error": str(e)[:200]}

        # --- CLIP reference-DB nearest neighbor (real geotagged photos) ---
        try:
            from modules.visual_geo_engine import VisualGeoEngine
            engine = VisualGeoEngine(self.device)
            res = engine.locate(p, top_k=6)
            for e in res.get("estimates", []):
                out["candidates"].append({
                    "latitude": float(e["latitude"]), "longitude": float(e["longitude"]),
                    "confidence": float(e.get("confidence", 0.0)),
                    "source": "visual_geo",
                    "support": e.get("support", 1),
                    "city": e.get("city", ""), "country": e.get("country", ""),
                })
            out["signals"]["visual_geo"] = {"status": res.get("status"),
                                            "db_size": res.get("db_size", 0)}
        except Exception as e:
            out["signals"]["visual_geo"] = {"status": "failed", "error": str(e)[:200]}

        # --- GeoGuessr CV heuristics (driving, road marks, plates, signs, soil) ---
        try:
            from modules.geoguessr_heuristics import GeoGuessrAnalyzer
            gh = GeoGuessrAnalyzer().analyze(p)
            out["signals"]["geoguessr"] = gh
            # negative-evidence constraints from heuristics
            ds = (gh.get("driving_side") or {}).get("driving_side", "unknown")
            if ds in ("left", "right"):
                out["constraints"].append(
                    f"driving_side={ds} (conf {gh['driving_side'].get('confidence',0)})")
            lc = (gh.get("road_markings") or {}).get("line_color", "")
            if lc:
                out["constraints"].append(f"road_line_color={lc}")
            soil = (gh.get("soil_and_vegetation") or {}).get("soil_type", "")
            if soil:
                out["constraints"].append(f"soil_type={soil}")
        except Exception as e:
            out["signals"]["geoguessr"] = {"status": "failed", "error": str(e)[:200]}

        # --- Environment / biome classifier ---
        try:
            from modules.environment_classifier import EnvironmentClassifier
            env = EnvironmentClassifier().classify(p) or {}
            out["signals"]["environment"] = env
            if env.get("environment_type") or env.get("region"):
                val = env.get("environment_type") or env.get("region")
                out["constraints"].append(f"environment={val}")
        except Exception as e:
            out["signals"]["environment"] = {"status": "failed", "error": str(e)[:200]}

        # --- OCR (real EasyOCR-with-fallback chain from the deep scanner) ---
        try:
            from geovision_deep_scan import phase2_ocr
            out["signals"]["ocr"] = phase2_ocr(p)
        except Exception:
            out["signals"]["ocr"] = {"status": "skipped"}

        # --- Consolidate candidates ---
        seen = set()
        uniq = []
        for c in out["candidates"]:
            key = (round(c["latitude"], 3), round(c["longitude"], 3))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        out["candidates"] = uniq
        return out

    def prune_with_constraints(self, candidates: List[Dict[str, Any]],
                               coarse: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Apply the spatial/negative-evidence constraint lattice to candidates.

        Builds a constraint dict from the coarse VLM prediction (country/hint,
        hemisphere) + deterministic constraints, then re-ranks candidates via
        SpatialConstraintSolver. Hard-incompatible candidates drop near zero.
        """
        try:
            from modules.spatial_constraint_solver import SpatialConstraintSolver
        except Exception as e:
            logger.warning("constraint solver unavailable: %s", e)
            return candidates
        if not candidates:
            return candidates

        constraints: Dict[str, Any] = {}
        for c in (self._last_scan.get("constraints") if hasattr(self, "_last_scan") else []):
            k, _, v = c.partition("=")
            if k and v:
                constraints[k] = v
        pred = (coarse or {}).get("prediction") or {}
        if pred.get("country"):
            constraints["country_hint"] = pred["country"]
        if pred.get("region"):
            constraints.setdefault("country_hint", pred["region"])
        hem = (constraints.get("solar_hemisphere")
               or (pred.get("hemisphere") if isinstance(pred, dict) else None))
        if hem:
            constraints["solar_hemisphere"] = hem

        solver = SpatialConstraintSolver()
        return solver.filter_and_rerank_estimates(candidates, constraints)

    # ------------------------------------------------------------------ #
    # Stage 3 — VLM verification against real reference photos            #
    # ------------------------------------------------------------------ #
    def verify_candidates(self, image_path: str,
                          candidates: List[Dict[str, Any]],
                          top_k: int = 3) -> Dict[str, Any]:
        """For each top candidate, fetch REAL ground truth photos and ask the
        (model-agnostic) VLM whether the query matches the reference.

        Returns {stage, verifications[]}. Each verification bundles the reference
        imagery alongside the VLM's cross-check + reasoning. Skips cleanly if no
        VLM is configured; reference photos are always fetched.
        """
        res: Dict[str, Any] = {"stage": "verify_candidates", "verifications": []}
        if not candidates:
            res["note"] = "no candidates to verify"
            return res

        from modules.ground_imagery_client import GroundImageryClient
        from modules.vlm_feature_prompts import encode_image_data_url
        import base64

        gic = GroundImageryClient()

        for cand in candidates[:top_k]:
            lat, lon = cand.get("latitude"), cand.get("longitude")
            if lat is None or lon is None:
                continue
            refs = gic.get_nearby_ground_photos(lat, lon, radius_m=2500, limit=4)
            photos = refs.get("ground_photos", [])
            v = {
                "candidate": {"latitude": lat, "longitude": lon,
                              "confidence": cand.get("confidence"),
                              "source": cand.get("source"),
                              "city": cand.get("city", ""), "country": cand.get("country", "")},
                "reference_photos": photos,
                "reference_urls": [ph.get("thumbnail_url") for ph in photos],
                "vlm_verdict": None,
            }
            if photos:
                v["reference_loaded"] = True
            # VLM cross-check — model-agnostic
            client, model = _get_vlm()
            if client and photos:
                verdict = self._call_verification(client, model, image_path, photos, lat, lon)
                v["vlm_verdict"] = verdict
                v["vlm_verified"] = bool(verdict)
            else:
                v["vlm_verdict"] = {"status": "skipped",
                                    "note": "no VLM configured (deterministic signal still valid)"}
            res["verifications"].append(v)
        res["status"] = "success"
        return res

    def _call_verification(self, client, model, image_path, photos, lat, lon) -> Optional[Dict]:
        """One model-agnostic verification call: query + references side-by-side."""
        from modules.vlm_feature_prompts import encode_image_data_url
        import requests, io
        prompt = (
            "You are a forensic geolocation verifier. The first image is the QUERY "
            "image that needs a location. The remaining images are GEOTAGGED reference "
            "photos taken near candidate coordinates (%s, %s). "
            "Compare them carefully: same architecture/terrain/signage/vegetation? "
            "Respond ONLY with JSON: {\"matches\": true|false, \"confidence\": 0.0-1.0, "
            "\"reasoning\": \"which shared visual cues confirm or refute the match\"}."
        ) % (round(float(lat), 4), round(float(lon), 4))
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}}]
        for ph in photos[:3]:
            url = ph.get("thumbnail_url")
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
        try:
            from modules.vlm_geo_analyzer import _call_vlm_json
            return _call_vlm_json(client, model, [{"role": "user", "content": content}])
        except Exception as e:
            return {"status": "failed", "error": str(e)[:200]}

    # ------------------------------------------------------------------ #
    # Regional retrieval — the coarse-to-fine upgrade (better than GeoSpy)
    # ------------------------------------------------------------------ #
    def regional_retrieval(self,
                           image_path: str,
                           prior_lat: Optional[float] = None,
                           prior_lon: Optional[float] = None,
                           radius_km: float = 1500.0,
                           top_k: int = 8) -> Dict[str, Any]:
        """Region-constrained CLIP nearest-neighbor retrieval.

        Method (improves on GeoSpy's global retrieval):
          take the coarse GPS prior (e.g. GeoCLIP's prediction), restrict the
          CLIP reference-DB search to REAL geotagged photos within `radius_km`
          of that prior, then re-rank within-region by cosine similarity ×
          spatial agreement. This drops cross-hemisphere/ocean false matches
          before ranking — ambiguity is pruned ahead of the nearest-neighbor.

        Returns {status, prior, radius_km, db_size, in_region, estimates[],
                 neighbors[], note}. Each reference is a real geotagged photo.
        """
        out: Dict[str, Any] = {
            "stage": "regional_retrieval", "status": "failed", "estimates": [],
            "prior": {"latitude": prior_lat, "longitude": prior_lon},
            "radius_km": radius_km, "db_size": 0, "in_region": 0,
            "note": "",
        }
        try:
            import numpy as np
            from modules.visual_geo_engine import VisualGeoEngine, _haversine_km
            engine = VisualGeoEngine(self.device)
            if not engine._ensure_db():
                out["note"] = "reference DB not built"
                return out
            emb = engine.embed_image(str(image_path))
            if emb is None:
                out["note"] = "query embedding failed"
                return out

            db_emb = engine._db_embeddings          # [N, 512] L2-normalized
            meta = engine._db_meta                  # [{lat, lon, title, ...}]
            out["db_size"] = db_emb.shape[0]

            if prior_lat is None or prior_lon is None:
                # No prior → fall back to global retrieval (still real)
                order = np.argsort(-(db_emb @ emb))[:top_k]
                in_region_n = db_emb.shape[0]
                idx = order
            else:
                # Region mask by haversine on real geotags
                lats = np.array([m["lat"] for m in meta], dtype=np.float64)
                lons = np.array([m["lon"] for m in meta], dtype=np.float64)
                dists = np.array([
                    _haversine_km(prior_lat, prior_lon, la, lo)
                    for la, lo in zip(lats, lons)], dtype=np.float64)
                mask = dists <= radius_km
                in_region_n = int(mask.sum())
                out["in_region"] = in_region_n
                if in_region_n == 0:
                    # empty region → globally expand to avoid a dead answer
                    order = np.argsort(-(db_emb @ emb))[:top_k]
                    idx = order
                else:
                    sub = db_emb[mask]
                    sims = sub @ emb
                    top_in_region = np.argsort(-sims)[:top_k]
                    global_pos = np.flatnonzero(mask)[top_in_region]
                    idx = global_pos

            sims_full = db_emb @ emb
            neighbors = []
            for i in idx:
                m = meta[int(i)]
                neighbors.append({
                    "lat": float(m["lat"]), "lon": float(m["lon"]),
                    "similarity": float(sims_full[int(i)]),
                    "title": m.get("title", ""), "city": m.get("city", ""),
                    "country": m.get("country", ""),
                    "distance_km_from_prior": round(
                        float(_haversine_km(prior_lat, prior_lon,
                                            m["lat"], m["lon"])) if prior_lat is not None else -1, 1),
                })
            out["neighbors"] = neighbors

            # Cluster neighbors within 75 km → estimates
            clusters = []
            for n in neighbors:
                placed = False
                for c in clusters:
                    if _haversine_km(n["lat"], n["lon"], c[0], c[1]) <= 75.0:
                        c[2].append(n); placed = True; break
                if not placed:
                    clusters.append((n["lat"], n["lon"], [n]))
            estimates = []
            for lat, lon, members in clusters:
                mean_sim = float(np.mean([x["similarity"] for x in members]))
                agreement = len(members) / top_k
                conf = max(0.0, min(1.0, (mean_sim - 0.55) / 0.40)) * (0.5 + 0.5 * agreement)
                top = max(members, key=lambda x: x["similarity"])
                estimates.append({
                    "latitude": float(np.mean([x["lat"] for x in members])),
                    "longitude": float(np.mean([x["lon"] for x in members])),
                    "confidence": round(conf, 4), "support": len(members),
                    "mean_similarity": round(mean_sim, 4),
                    "city": top.get("city", ""), "country": top.get("country", ""),
                    "source": "regional_retrieval",
                    "in_region": in_region_n,
                })
            estimates.sort(key=lambda e: e["confidence"], reverse=True)
            out["estimates"] = estimates[:5]
            out["status"] = "success" if estimates else "limited"
            out["note"] = (
                f"retrieval restricted to {in_region_n}/{db_emb.shape[0]} real refs "
                f"within {radius_km:.0f}km of prior (regional cascade)"
                if prior_lat is not None
                else "no prior → global retrieval fallback")
            return out
        except Exception as e:
            out["note"] = str(e)[:200]
            return out

    # ------------------------------------------------------------------ #
    # Real-data growth: grow the retrieval index around a GPS prior
    # ------------------------------------------------------------------ #
    def grow_reference_db(self,
                          prior_lat: float,
                          prior_lon: float,
                          radius_m: int = 5000,
                          per_rate: int = 40,
                          max_new: int = 200) -> Dict[str, Any]:
        """Grow data/visual_geo_db with REAL geotagged photos near a GPS prior.

        This is the "more real data" path that grows the retrieval index the way
        GeoSpy grows its geotagged image index — but with real, licensed-friendly
        Commons photos and honest GPS. Reuses the live Wikimedia geosearch +
        CLIP-embed append machinery from scripts/build_reference_db.py.

        Every appended reference is a real photograph with its real GPS geotag
        (source=wikimedia_commons). Dedupes by file title. Skips cleanly if CLIP
        is unavailable or no hits are found. Designed to be run iteratively
        (once per region of interest) to densify retrieval coverage.

        Returns {added, skipped, radius_m, prior, db_size_after, note}.
        """
        out: Dict[str, Any] = {
            "stage": "grow_reference_db", "status": "skipped",
            "added": 0, "skipped": 0, "prior": {"latitude": prior_lat, "longitude": prior_lon},
            "radius_m": radius_m, "db_size_after": None, "note": "",
        }
        try:
            import time, requests, io
            import numpy as np
            from pathlib import Path
            from modules.visual_geo_engine import DB_DIR, DB_EMB, DB_META, CLIP_CACHE
            from modules.visual_geo_engine import VisualGeoEngine
            import open_clip, torch
        except Exception as e:
            out["note"] = f"CLIP/DB deps unavailable: {e}"[:200]
            return out

        try:
            engine = VisualGeoEngine(self.device)
            if not engine._ensure_db():
                out["note"] = "visual_geo_db missing; run build_reference_db.py first"
                return out
            # load existing DB (embeddings + meta) to dedup + append
            existing_meta = engine._db_meta
            existing_titles = {m.get("title") for m in existing_meta if m.get("title")}
            emb_list = [engine._db_embeddings]
            meta_list = list(existing_meta)

            # CLIP embedder
            import open_clip, torch
            device = self.device
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(CLIP_CACHE))
            model.eval().to(device)

            from scripts.build_reference_db import commons_geosearch, fetch_thumb, embed_bytes
            session = requests.Session()
            hits = commons_geosearch(prior_lat, prior_lon, radius_m, per_rate)
            added = 0
            skipped = 0
            for h in hits:
                title = h.get("title")
                if not title or title in existing_titles:
                    continue
                data = fetch_thumb(title, session)
                if not data:
                    skipped += 1
                    continue
                emb = embed_bytes(data, model, preprocess, device)
                if emb is None:
                    skipped += 1
                    continue
                existing_titles.add(title)
                emb_list.append(emb.reshape(1, -1))
                meta_list.append({
                    "lat": h.get("lat"), "lon": h.get("lon"), "title": title,
                    "city": "", "country": "", "population": 0,
                    "source": "wikimedia_commons",
                })
                added += 1
                if added >= max_new:
                    break
                time.sleep(0.25)

            if added == 0:
                out["status"] = "limited"
                out["note"] = f"no new real refs in {radius_m}m of prior ({(out['skipped'])} fetch/embed failures)"
                return out

            # atomic append flush (matches build_reference_db._flush format)
            arr = np.concatenate([a for a in emb_list if a is not None],
                                 axis=0).astype(np.float32)
            tmp = DB_EMB.with_suffix(".tmp.npy")
            np.save(tmp, arr); tmp.replace(DB_EMB)
            with gzip.open(DB_META, "wt", encoding="utf-8") as f:
                for m in meta_list:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            out["added"] = added
            out["skipped"] = skipped
            out["db_size_after"] = int(arr.shape[0])
            out["status"] = "success"
            out["note"] = (f"appended {added} real geotagged Commons photos in "
                           f"{radius_m}m of ({prior_lat:.2f},{prior_lon:.2f}); "
                           f"DB now {int(arr.shape[0])} refs")
            return out
        except Exception as e:
            out["note"] = str(e)[:200]
            out["status"] = "failed"
            return out

    # ------------------------------------------------------------------ #
    # Offline visual-match verification (Stage-3, no VLM key needed)
    # ------------------------------------------------------------------ #
    def visual_verify_candidates(self,
                                 image_path: str,
                                 candidates: List[Dict[str, Any]],
                                 top_k: int = 3) -> Dict[str, Any]:
        """Cross-check each top candidate against its REAL ground-truth photos
        using a real StreetCLIP image-to-image cosine (offline, no API key).

        This is the Stage-3 cross-check for when no VLM is configured: embed the
        query and each Wikimedia/Mapillary reference photo with StreetCLIP, score
        the actual visual similarity, and re-rank candidates by real match. Falls
        back cleanly if StreetCLIP or reference fetch fails.
        """
        res: Dict[str, Any] = {"stage": "visual_verify_candidates",
                               "status": "skipped", "verifications": [], "note": ""}
        if not candidates:
            res["note"] = "no candidates"
            return res
        try:
            from modules.visual_similarity import VisualSimilarityScorer
            from modules.ground_imagery_client import GroundImageryClient
            import requests, io
            from PIL import Image
            import numpy as np
        except Exception as e:
            res["note"] = f"deps unavailable: {e}"[:200]
            return res

        try:
            scorer = VisualSimilarityScorer(self.device)
            scorer._ensure_loaded()
        except Exception as e:
            res["note"] = f"StreetCLIP unavailable: {e}"[:200]
            return res

        try:
            query_img = Image.open(image_path).convert("RGB")
            query_emb = scorer.encode_image(query_img)
        except Exception as e:
            res["note"] = f"query embed failed: {e}"[:200]
            return res

        gic = GroundImageryClient()
        verifications = []
        for cand in candidates[:top_k]:
            lat, lon = cand.get("latitude"), cand.get("longitude")
            if lat is None or lon is None:
                continue
            refs = gic.get_nearby_ground_photos(lat, lon, radius_m=2500, limit=4)
            photos = refs.get("ground_photos", [])
            best_sim = 0.0
            matches = []
            for ph in photos:
                url = ph.get("thumbnail_url")
                if not url:
                    continue
                try:
                    r = requests.get(url, timeout=15,
                                     headers={"User-Agent": "GeoVision/2.0 (geospatial-osint)"})
                    if r.status_code != 200 or not r.content:
                        continue
                    ref_img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    ref_emb = scorer.encode_image(ref_img)
                    sim = float(np.dot(query_emb, ref_emb))
                    if sim > best_sim:
                        best_sim = sim
                    matches.append({"thumbnail_url": url, "similarity": round(sim, 4),
                                    "title": ph.get("title", "")})
                except Exception:
                    continue
            verifications.append({
                "candidate": {"latitude": lat, "longitude": lon,
                              "confidence": cand.get("confidence"),
                              "source": cand.get("source"),
                              "city": cand.get("city", ""), "country": cand.get("country", "")},
                "reference_photos": photos,
                "reference_loaded": bool(photos),
                "visual_similarity": round(best_sim, 4),
                "reference_matches": matches[:4],
                # A real StreetCLIP cross-view match is a meaningful signal
                "visual_match": best_sim > 0.28,
            })
        res["verifications"] = verifications
        if verifications:
            res["status"] = "success"
            res["note"] = ("offline StreetCLIP query-vs-reference visual scoring "
                           f"over {len(verifications)} candidates (no VLM key required)")
        else:
            res["note"] = "no reference photos fetched for any candidate"
        return res

    # ------------------------------------------------------------------ #
    # Master entrypoint                                                   #
    # ------------------------------------------------------------------ #
    def investigate(self, image_path: str,
                    evidence_summary: Optional[str] = None,
                    location_hint: Optional[str] = None,
                    save_case: bool = False,
                    case_name: Optional[str] = None,
                    case_description: str = "",
                    case_tags: Optional[List[str]] = None,
                    top_k_verify: int = 3,
                    radius_km: float = 1500.0,
                    use_regional: bool = True,
                    with_listings: bool = False,
                    auto_report: bool = False) -> Dict[str, Any]:
        """Run the full multi-stage investigation and produce a structured case record.

        Stages:
          Stage 1 coarse_reason (VLM, optional)  -> priors + constraints
          Stage 2 deterministic_scan             -> candidates from all deterministic signal
          regional_retrieval (better-than-GeoSpy)-> region-constrained CLIP-NN
          constraint prune                        -> eliminate impossible candidates
          Stage 3 verify_candidates (VLM+photos) -> cross-check top candidates
           deep-dive OSINT (best-effort)          -> reverse-image / chain-store
          fusion+ranking+uncertainty              -> final answer + reasoning chain

        Always returns a dict; no failure raises. save_case=True persists to the
        SQLite CaseManager and links a full deep-scan report.
        """
        p = Path(image_path).expanduser().resolve()
        if not p.exists():
            return {"status": "failed", "error": f"image not found: {p}", "stage": "harness"}
        started = time.time()
        t0 = _now()

        record: Dict[str, Any] = {
            "harness": "GeoVisionHarness",
            "status": "limited",
            "image_path": str(p),
            "started_at": t0,
            "duration_s": None,
            "stages": {},
            "candidates": [],
            "best_estimate": None,
            "uncertainty": None,
            "constraints_applied": [],
            "reasoning_chain": [],
            "notes": [],
        }

        # Stage 1 — coarse VLM reasoning
        coarse = self.coarse_reason(str(p), evidence_summary=evidence_summary,
                                    location_hint=location_hint)
        record["stages"]["coarse_reason"] = coarse
        if (coarse.get("prediction") or {}).get("reasoning"):
            record["reasoning_chain"].append("COARSE: " + coarse["prediction"]["reasoning"])
        if (coarse.get("prediction") or {}).get("eliminated"):
            record["notes"].append("Eliminated (VLM): " +
                                   "; ".join(coarse["prediction"]["eliminated"]))

        # Stage 2 — deterministic scan
        scan = self.deterministic_scan(str(p))
        self._last_scan = scan
        record["stages"]["deterministic_scan"] = scan
        record["constraints_applied"] = scan.get("constraints", [])
        candidates = scan.get("candidates", [])

        # Constraint pruning (negative evidence)
        pruned = self.prune_with_constraints(candidates, coarse)
        pruned.sort(key=lambda c: c.get("confidence", 0), reverse=True)
        record["candidates"] = pruned[:12]

        # Regional retrieval — coarse-to-fine upgrade (better-than-GeoSpy).
        # Seed the region with the best deterministic prior (GeoCLIP top candidate
        # or VLM prediction), restrict CLIP-NN to real refs within radius_km.
        regional = {"status": "skipped", "note": "regional retrieval disabled"}
        if use_regional:
            # pick coarse prior: GeoCLIP estimate first, else VLM prediction
            prior_lat = prior_lon = None
            for c in candidates:
                if c.get("source") == "geoclip":
                    prior_lat, prior_lon = c["latitude"], c["longitude"]
                    break
            if prior_lat is None:
                pred = (coarse.get("prediction") or {})
                if pred.get("latitude") is not None and pred.get("longitude") is not None:
                    prior_lat, prior_lon = float(pred["latitude"]), float(pred["longitude"])
            if prior_lat is not None and prior_lon is not None:
                regional = self.regional_retrieval(str(p), prior_lat, prior_lon,
                                                   radius_km=radius_km, top_k=8)
                # merge region-constrained estimates into the candidate pool
                seen = {(round(x["latitude"], 2), round(x["longitude"], 2))
                        for x in pruned}
                for e in regional.get("estimates", []):
                    key = (round(e["latitude"], 2), round(e["longitude"], 2))
                    if key not in seen:
                        seen.add(key)
                        pruned.append(e)
                if regional.get("estimates"):
                    record["reasoning_chain"].append(
                        f"REGIONAL retrieval constrained to {regional.get('in_region',0)} real refs "
                        f"within {radius_km:.0f}km of coarse prior "
                        f"({prior_lat:.2f},{prior_lon:.2f}) re-ranked by CLIP")
        record["stages"]["regional_retrieval"] = regional

        # Deep-dive OSINT (best-effort, each isolated)
        deep = self._deep_dive(str(p), pruned[:5])
        record["deep_dive"] = deep

        # Stage 3 — VLM verification of top candidates with real reference photos
        verify = self.verify_candidates(str(p), pruned, top_k=top_k_verify)
        record["stages"]["verify_candidates"] = verify
        for v in verify.get("verifications", []):
            if v.get("vlm_verified"):
                record["reasoning_chain"].append(
                    f"VERIFY match at {v['candidate']['latitude']:.4f},{v['candidate']['longitude']:.4f}")

        # Stage 3b — offline StreetCLIP visual-match cross-check (real, no VLM key).
        # Runs even when no VLM is configured so Stage-3 evidence is never empty.
        vverify = self.visual_verify_candidates(str(p), pruned, top_k=top_k_verify)
        record["stages"]["visual_verify_candidates"] = vverify
        for v in vverify.get("verifications", []):
            if v.get("visual_match"):
                # boost that candidate's confidence — real cross-view signal
                for c in pruned:
                    if (abs(c.get("latitude", 1e9) - v["candidate"]["latitude"]) < 0.5) and \
                       (abs(c.get("longitude", 1e9) - v["candidate"]["longitude"]) < 0.5):
                        c["confidence"] = min(0.97, float(c.get("confidence", 0.0)) + 0.12)
                record["reasoning_chain"].append(
                    f"VISUAL match (StreetCLIP sim {v.get('visual_similarity')}) at "
                    f"{v['candidate']['latitude']:.4f},{v['candidate']['longitude']:.4f}")

        # Fusion & final ranking (regional retrieval evidence is fused in)
        best, uncertainty = self._finalize(pruned, coarse, verify)
        record["best_estimate"] = best
        record["uncertainty"] = uncertainty
        record["duration_s"] = round(time.time() - started, 2)
        record["status"] = "success" if best else "limited"
        if best and not record.get("reasoning_chain"):
            record["reasoning_chain"] = ["Prediction produced by deterministic multi-signal fusion"]

        # agent_hint: contract telling a vision-capable model how to continue.
        # Collect the REAL ground-truth reference photo URLs per candidate so the
        # model can open them and do its own visual cross-check (its own vision).
        ref_view = {}
        for stage in ("verify_candidates", "visual_verify_candidates"):
            for v in (record.get("stages", {}).get(stage, {}).get("verifications") or []):
                lat = v.get("candidate", {}).get("latitude")
                refs = v.get("reference_urls") or [p.get("thumbnail_url") for p in v.get("reference_photos", [])]
                if lat is not None and refs:
                    ref_view.setdefault(round(lat, 3), list(refs))
        hint = ("Use your own vision to cross-check the candidate ground-truth reference "
                "photos (reference_urls below) against the query image and confirm/refute "
                "the best_estimate. If ambiguous, extract more clues from the image "
                "(OCR text, architecture, chain stores) and call resolve_vision_clues "
                "with them, or re-run investigate_image with a location_hint. "
                "confidence < 0.30 = weak, treat as hypothesis not confirmation.")
        record["agent_hint"] = hint
        record["reference_urls_by_candidate"] = ref_view

        # Optional property/business listing snapshot (consolidated into the case).
        if with_listings and best:
            try:
                from modules.property_locator import PropertyLocator
                ls = PropertyLocator().list_nearby(best["latitude"], best["longitude"],
                                                   radius_m=min(3000, max(500, int(radius_km * 1000))),
                                                   categories=["lodging", "commercial", "dining", "office"],
                                                   limit=15)
                record["nearby_listings"] = {
                    "count": ls.get("count"), "radius_m": ls.get("radius_m"),
                    "listings": ls.get("listings", []), "status": ls.get("status"),
                }
            except Exception as e:
                record["nearby_listings"] = {"status": "failed", "note": str(e)[:120]}

        # Optional auto-write of a professional report (Markdown; HTML if saved case).
        report_path = None
        if auto_report and best:
            try:
                from modules.case_report import render_case
                ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                out_md = SCRIPT_DIR / "reports" / f"investigate_{ts}.md"
                render_case(record, str(out_md),
                            title=case_name or "GeoVision Case Report")
                report_path = str(out_md)
            except Exception as e:
                logger.warning("auto report write failed: %s", e)
                report_path = None
        if report_path:
            record["report_path"] = report_path

        # Optional persistence to case manager
        if save_case:
            record["case_id"] = self._persist_case(record, p, case_name,
                                                   case_description, case_tags)
        return record

    # ------------------------------------------------------------------ #
    # Deep-dive OSINT pass                                                #
    # ------------------------------------------------------------------ #
    def _deep_dive(self, image_path: str,
                   candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Best-effort OSINT per top candidate: reverse-image unmask, chain stores,
        property records. Each is isolated and skips cleanly on failure."""
        out: Dict[str, Any] = {}
        if not candidates:
            return out
        lat, lon = candidates[0]["latitude"], candidates[0]["longitude"]
        try:
            from modules.reverse_image_search import ReverseImageSearcher
            out["reverse_image"] = ReverseImageSearcher().search_similar_images(image_path)
        except Exception:
            pass
        try:
            from modules.chain_store_locator import ChainStoreLocator
            out["chain_stores"] = ChainStoreLocator()._query_overpass(
                f'[out:json][timeout:40];node["shop"](around:2500,{lat},{lon});out 8;')
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------ #
    # Finalize: rank + uncertainty                                        #
    # ------------------------------------------------------------------ #
    def _finalize(self, candidates, coarse, verify):
        if not candidates:
            return None, None
        best = dict(candidates[0])
        best["confidence"] = min(0.97, float(best.get("confidence", 0.0)))
        # If VLM verified one candidate, boost it
        for v in verify.get("verifications", []):
            if v.get("vlm_verified") and v.get("vlm_verdict", {}).get("matches"):
                if abs(v["candidate"]["latitude"] - best["latitude"]) < 0.5 and \
                   abs(v["candidate"]["longitude"] - best["longitude"]) < 0.5:
                    best["confidence"] = min(0.97, best["confidence"] + 0.15)
                    best["vlm_verification"] = True
        uncertainty = None
        try:
            from modules.uncertainty_estimator import UncertaintyEstimator
            u = UncertaintyEstimator().estimate(candidates, best_estimate=best)
            uncertainty = u if isinstance(u, dict) else {"granularity": "region"}
        except Exception:
            pass
        return best, uncertainty

    def _persist_case(self, record, image_path, case_name, description, tags):
        """Save the investigation to the SQLite CaseManager + write JSON report."""
        try:
            from modules.case_manager import CaseManager
            cm = CaseManager()
            name = case_name or f"Image {image_path.stem}"
            best = record.get("best_estimate") or {}
            case = cm.create_case(name, description, tags=tags or [])
            scan_id = f"harness-{int(time.time())}"
            json_path = None
            md_path = None
            html_path = None
            if best:
                out_dir = SCRIPT_DIR / "reports"
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / f"case_{case['id']}_{scan_id}.json"
                json_path.write_text(json.dumps(record, indent=2, default=str))
                # Professional, honest report for the case (Markdown + optional HTML)
                try:
                    from modules.case_report import render_case
                    title = f"GeoVision Case {case['id']} — {name}"
                    md_path = out_dir / f"case_{case['id']}_{scan_id}.md"
                    html_path = out_dir / f"case_{case['id']}_{scan_id}.html"
                    render_case(record, str(md_path), title=title)
                    render_case(record, str(html_path), title=title)
                except Exception as e:
                    logger.warning("case report render failed: %s", e)
                    md_path = html_path = None
            cm.add_scan_to_case(
                case["id"], scan_id,
                image_path=str(image_path),
                best_lat=best.get("latitude"), best_lon=best.get("longitude"),
                best_conf=best.get("confidence"),
                summary={"best_estimate": record.get("best_estimate"),
                         "candidates": len(record.get("candidates", [])),
                         "report_md_path": str(md_path) if md_path else None},
                scan_json_path=str(json_path) if json_path else None,
                scan_html_path=str(html_path) if html_path else None,
            )
            cm.add_note(case["id"],
                        f"GeoVisionHarness investigation: {record.get('reasoning_chain', ['n/a'])}")
            return case["id"]
        except Exception as e:
            logger.warning("case persistence failed: %s", e)
            return None


def _get_vlm():
    """Build the model-agnostic OpenAI-compatible client (shared with vlm_geo_analyzer)."""
    from modules.vlm_geo_analyzer import _get_client
    return _get_client()


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "building_image.jpg"
    h = GeoVisionHarness()
    print(json.dumps(h.investigate(path), indent=2, ensure_ascii=False, default=str))
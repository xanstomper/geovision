"""
Candidate Visual Verification
==============================
Ports open_geo_spy's CandidateVerificationAgent pattern
(src/agents/candidate_verification_agent.py) using GeoVision's ported
building blocks:

  candidate coords -> fetch real reference photos (Wikimedia Commons /
  Mapillary) -> StreetCLIP-embed them -> cosine similarity vs the query
  image -> re-rank candidates by real visual match.

Produces VISUAL_MATCH evidence that distinguishes same-category candidates
(e.g., two cities that both look plausible) using actual photographs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "GeoVision-OSINT/1.0 (geolocation research)"}


def fetch_reference_photos(lat: float, lon: float, radius_m: int = 2000,
                           limit: int = 3) -> List[Dict[str, Any]]:
    """Fetch real geotagged photo URLs near a coordinate from Wikimedia Commons."""
    import requests
    try:
        r = requests.get(COMMONS_API, params={
            "action": "query", "list": "geosearch",
            "gscoord": f"{lat}|{lon}", "gsradius": radius_m,
            "gsnamespace": "6", "gslimit": str(limit), "format": "json",
        }, headers=UA, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("geosearch", [])
        from urllib.parse import quote
        refs = []
        for p in pages:
            title = p["title"]
            fname = title[len("File:"):] if title.startswith("File:") else title
            refs.append({
                "url": f"https://commons.wikimedia.org/w/thumb.php?f={quote(fname)}&width=640",
                "candidate_name": f"({lat:.3f},{lon:.3f})",
                "title": title,
                "lat": p["lat"], "lon": p["lon"],
            })
        return refs
    except Exception as e:
        logger.warning("commons geosearch failed: %s", e)
        return []


def verify_candidates(query_image_path: str,
                      candidates: List[Dict[str, Any]],
                      refs_per_candidate: int = 2) -> Dict[str, Any]:
    """
    Visually verify location candidates against the query image.

    Args:
        query_image_path: path to the image being geolocated
        candidates: [{latitude, longitude, ...}] — typically the pipeline's
                    top location estimates
        refs_per_candidate: how many reference photos to fetch per candidate

    Returns:
        {status, rankings: [{latitude, longitude, similarity, n_refs, title}],
         best: {...}}
    """
    result: Dict[str, Any] = {
        "engine": "StreetCLIP visual similarity over real reference photos",
        "status": "failed",
        "rankings": [],
    }
    if not candidates:
        result["note"] = "no candidates to verify"
        return result

    try:
        from modules.visual_similarity import VisualSimilarityScorer
    except ImportError as e:
        result["note"] = f"visual similarity unavailable: {e}"
        return result

    scorer = VisualSimilarityScorer()

    rankings = []
    for cand in candidates[:5]:
        lat, lon = cand["latitude"], cand["longitude"]
        refs = fetch_reference_photos(lat, lon, limit=refs_per_candidate)
        if not refs:
            rankings.append({
                "latitude": lat, "longitude": lon,
                "similarity": None, "n_refs": 0,
                "note": "no reference photos found nearby",
            })
            continue

        # score_candidates is async; run it synchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        scored = loop.run_until_complete(
            scorer.score_candidates(query_image_path, refs)
        )
        if scored:
            best = scored[0]
            rankings.append({
                "latitude": lat, "longitude": lon,
                "similarity": round(best["similarity"], 4),
                "n_refs": len(scored),
                "title": best.get("title", ""),
                "ref_url": best.get("url", ""),
            })
        else:
            rankings.append({
                "latitude": lat, "longitude": lon,
                "similarity": None, "n_refs": 0,
                "note": "reference photos failed to download/encode",
            })

    scored_rankings = [r for r in rankings if r["similarity"] is not None]
    if scored_rankings:
        scored_rankings.sort(key=lambda r: r["similarity"], reverse=True)
        result["rankings"] = rankings
        result["best"] = scored_rankings[0]
        result["status"] = "success"
        result["note"] = (
            f"Verified {len(scored_rankings)}/{len(candidates)} candidates "
            f"against real photos; best similarity {scored_rankings[0]['similarity']}"
        )
    else:
        result["rankings"] = rankings
        result["status"] = "limited"
        result["note"] = "no reference photos could be scored"

    return result


if __name__ == "__main__":
    import sys
    import json
    img = sys.argv[1] if len(sys.argv) > 1 else "building_image.jpg"
    cands = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [
        {"latitude": 43.6532, "longitude": -79.3832},   # Toronto
        {"latitude": 40.7128, "longitude": -74.0060},   # NYC
    ]
    print(json.dumps(verify_candidates(img, cands), indent=2))
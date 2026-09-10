import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image

from modules.visual_similarity import VisualSimilarityScorer
from modules.ground_imagery_client import GroundImageryClient, USER_AGENT


EIFFEL = "data/eval/wikipedia_landmarks_v1/images/eiffel_tower.jpg"


def test_encode_image_returns_flat_l2_vector():
    """encode_image must return a flat 768-dim L2-normalized vector (fixes the
    3D / BaseModelOutputWithPooling bug that crashed cosine similarity)."""
    sc = VisualSimilarityScorer("cpu")
    sc._ensure_loaded()
    em = sc.encode_image(Image.open(EIFFEL).convert("RGB"))
    assert em.ndim == 1
    assert em.shape == (768,)
    assert abs(float(np.linalg.norm(em)) - 1.0) < 1e-3
    assert bool(em.any())  # real (non-zero) signal
    # same-image cosine must be ~1.0 (would crash on 3D tensors before the fix)
    d = float(np.dot(em, sc.encode_image(Image.open(EIFFEL).convert("RGB"))))
    assert d > 0.99


def test_ground_imagery_urls_fetch_as_images():
    """GroundImageryClient must return File: pages with real fetchable thumbnails
    (fixes the Special:FilePath 404 + missing gsnamespace=6 bug)."""
    import urllib.request
    gic = GroundImageryClient()
    refs = gic.get_nearby_ground_photos(48.8584, 2.2945, radius_m=2000, limit=4)
    photos = refs.get("ground_photos", [])
    assert len(photos) >= 1
    assert photos[0]["title"].startswith("File:")  # file namespace, not wiki pages
    ok = 0
    for ph in photos:
        req = urllib.request.Request(ph["thumbnail_url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status == 200 and r.headers.get("content-type", "").startswith("image"):
                    ok += 1
        except Exception:
            pass
    assert ok == len(photos)  # every reference URL must serve real image bytes


def test_offline_visual_verify_matches_real_eiffel():
    """visual_verify_candidates must detect a genuine cross-view match: the Eiffel
    query vs real Eiffel reference photos scores above the 0.28 visual_match gate."""
    from modules.geo_harness import GeoVisionHarness
    cands = [{"latitude": 48.8584, "longitude": 2.2945, "confidence": 0.5,
              "source": "t", "city": "Paris", "country": "FR"}]
    out = GeoVisionHarness().visual_verify_candidates(EIFFEL, cands, top_k=1)
    assert out.get("status") == "success"
    vs = out.get("verifications", [])
    assert len(vs) == 1
    v = vs[0]
    assert len(v.get("reference_photos", [])) > 0
    assert v.get("visual_similarity", 0.0) > 0.28
    assert v.get("visual_match") is True
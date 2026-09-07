"""
VLM Geo Analyzer — Vision-Language Model geolocation reasoning
================================================================
Ports open_geo_spy's VLM feature-extraction + reasoning pattern into GeoVision.

Uses any OpenAI-compatible vision endpoint (OpenCode Zen, Gemini via
OpenAI-compat, etc.) configured via env vars:
    GEOVISION_VLM_BASE_URL  (default: https://opencode.ai/zen/v1)
    GEOVISION_VLM_API_KEY   (default: $OPENCODE_ZEN_API_KEY)
    GEOVISION_VLM_MODEL     (default: opencode/gpt-5.4-nano)

Two calls (ported prompt patterns from open_geo_spy):
  1. Feature extraction — structured JSON of ALL geo clues
  2. Location reasoning — candidates + confidence from those features
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from modules.vlm_feature_prompts import (
    FEATURE_EXTRACTION_PROMPT,
    FEATURE_EXTRACTION_PROMPT_WITH_HINT,
    encode_image_data_url,
)

logger = logging.getLogger(__name__)

REASONING_PROMPT = """You are a geolocation analyst (GeoSpy-class). Based on the extracted visual features below, reason about where in the world this photo was taken.

Features:
{features}

Other evidence from the pipeline:
{evidence}

Respond with a JSON object:
{{
  "country": "most likely country (empty string if unknown)",
  "region": "state/province/region (empty if unknown)",
  "city": "most likely city (empty if unknown)",
  "latitude": <float or null>,
  "longitude": <float or null>,
  "confidence": <0.0-1.0, honest — low if features are ambiguous>,
  "reasoning": "2-4 sentences citing the specific features that drove the decision",
  "eliminated": ["regions/countries the features rule out and why"]
}}
"""


def _get_client():
    """Build an OpenAI-compatible client from env config. Returns (client, model) or (None, None)."""
    api_key = os.environ.get("GEOVISION_VLM_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
    if not api_key:
        return None, None
    base_url = os.environ.get("GEOVISION_VLM_BASE_URL", "https://opencode.ai/zen/v1")
    model = os.environ.get("GEOVISION_VLM_MODEL", "opencode/gpt-5.4-nano")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client, model
    except Exception as e:
        logger.warning("VLM client unavailable: %s", e)
        return None, None


def _call_vlm_json(client, model, messages) -> Optional[dict]:
    """One VLM call, returning parsed JSON (fences stripped) or None."""
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        text = resp.choices[0].message.content or ""
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1)
        return json.loads(text)
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        return None


def vlm_analyze(image_path: str,
                evidence_summary: Optional[str] = None,
                location_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Full VLM geolocation analysis:
      call 1 — extract structured features (open_geo_spy prompt)
      call 2 — reason to a location prediction

    Returns dict with keys: status, features, prediction, note.
    Degrades gracefully (status="skipped") when no VLM is configured.
    """
    result: Dict[str, Any] = {
        "status": "skipped",
        "features": None,
        "prediction": None,
        "note": "No VLM configured (set GEOVISION_VLM_API_KEY or OPENCODE_ZEN_API_KEY)",
    }

    client, model = _get_client()
    if client is None:
        return result

    # ---- Call 1: feature extraction ----
    prompt = (
        FEATURE_EXTRACTION_PROMPT_WITH_HINT.format(location_hint=location_hint)
        if location_hint else FEATURE_EXTRACTION_PROMPT
    )
    messages1 = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
        ],
    }]
    features = _call_vlm_json(client, model, messages1)
    if not features or "error" in (features or {}):
        result["status"] = "failed"
        result["note"] = f"feature extraction failed: {(features or {}).get('error', 'no JSON')}"
        return result
    result["features"] = features

    # ---- Call 2: reasoning to a prediction ----
    evidence_txt = evidence_summary or "none"
    messages2 = [{
        "role": "user",
        "text": None,
        "content": [
            {"type": "text", "text": REASONING_PROMPT.format(
                features=json.dumps(features, ensure_ascii=False)[:4000],
                evidence=evidence_txt[:2000],
            )},
        ],
    }]
    prediction = _call_vlm_json(client, model, messages2)
    if not prediction:
        result["status"] = "limited"
        result["note"] = "features extracted but reasoning call failed"
        return result

    result["prediction"] = prediction
    result["status"] = "success"
    result["note"] = (
        f"VLM: {prediction.get('country', '?')} / {prediction.get('city', '?')} "
        f"(conf {prediction.get('confidence', 0)})"
    )
    return result


def vlm_geo_estimates(vlm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a vlm_analyze result into pipeline location estimates."""
    pred = vlm_result.get("prediction") or {}
    lat, lon = pred.get("latitude"), pred.get("longitude")
    if lat is None or lon is None:
        # If the VLM only named a place, geocode it via Nominatim in the caller
        return []
    conf = float(pred.get("confidence", 0) or 0)
    return [{
        "latitude": float(lat),
        "longitude": float(lon),
        "confidence": max(0.0, min(1.0, conf)),
        "source": "vlm_reasoning",
        "country": pred.get("country", ""),
        "city": pred.get("city", ""),
        "reasoning": pred.get("reasoning", ""),
    }]


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "building_image.jpg"
    print(json.dumps(vlm_analyze(img), indent=2, ensure_ascii=False))
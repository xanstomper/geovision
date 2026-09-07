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

# ---------------------------------------------------------------------------
# Strict evidence-hierarchy reasoning prompt — ported verbatim from
# open_geo_spy src/agents/reasoning_agent.py. This is their hardest-won
# prompt: transit-operator uniqueness, evidence hierarchy, conflict
# resolution, and hard confidence-calibration rules that prevent the LLM
# from assigning high confidence without city-specific evidence.
# ---------------------------------------------------------------------------
STRICT_REASONING_PROMPT = """You are an expert geolocation analyst specializing in precise city and neighborhood identification from visual evidence. Your primary task is to determine the MOST SPECIFIC location possible — neighborhood beats city beats region beats country.

## Evidence Chain
{evidence}

## Environment Type
{env_type}

## Location Hint
{location_hint}

## ⛔ MANDATORY REASONING PROTOCOL — EVERY STEP IS REQUIRED

### Step 0: EVIDENCE EXTRACTION (DO NOT SKIP — COMPLETE THIS BEFORE CHOOSING ANY LOCATION)
Before you name ANY city, you MUST extract and list ALL visible text and identifiers from the image:
- **Transit text**: tram/bus stop names on signs, operator logos/names, route numbers, line designations, station name plates
- **OCR text**: business names, street signs, building names, advertisements — especially anything containing city names, district names, or postal codes
- **Vehicle markings**: fleet numbers, operator names on buses/trams, license plate region codes
- **Infrastructure details**: signal design, stop shelter type, track type, vehicle livery

⛔ If transit infrastructure (trams, buses, stops, stations) is visible, you MUST identify the operator and/or stop name BEFORE proceeding to Step 2. This is NON-NEGOTIABLE. A tram stop name uniquely and exclusively identifies its city.

### Step 1: Country Lock
Identify country from hard indicators: license plate format, language/script on signs, road markings, electrical outlet types, vehicle types. If uncertain, list the top 2 candidates with reasons.

### Step 2: City Discrimination (CRITICAL — MOST ERRORS OCCUR HERE)

**⛔ THE CARDINAL RULE: When transit infrastructure is visible, the transit operator and/or stop name ALONE determines the city. You are NOT permitted to guess a city from regional proximity when transit evidence is available. A named tram stop exists in exactly ONE city — it does not exist in any other city, no matter how close.**

**Transit identifiers are UNIQUE to their city:**
- Stop names are exclusive: a named stop exists in exactly one city
- Operator logos are exclusive to their transit authority's city
- Route numbers and line designations are city-specific
- Vehicle livery and stop design vary by transit authority

**Other city-level discriminators (use ONLY when transit identifiers are unreadable):**
- OCR text containing city or district names
- Phone area codes
- Local business chains operating in only one city
- Street name formats specific to a city

**✅ MANDATORY VERIFICATION CHECKLIST — complete before finalizing city:**
□ Did I extract a specific transit stop name? → If YES, city is determined. Stop here.
□ Did I extract a transit operator logo/name? → If YES, city is determined. Stop here.
□ Did I find OCR text containing a city or district name? → If YES, use that city.
□ Am I choosing a city based only on regional proximity, coordinate clusters, or general impression? → If YES, STOP. You lack city-level evidence. Set confidence ≤0.5.

### Step 3: Precision Downgrade Rule
If you cannot find city-specific evidence (transit names, OCR with city/district, unique landmarks):
- State explicitly which city-level evidence is MISSING
- Choose the city with the MOST specific corroborating evidence, NOT the most famous one
- LOWER your confidence to ≤0.5 — no city-specific evidence means no city-level confidence

## EVIDENCE HIERARCHY (highest to lowest reliability)
1. **Named transit infrastructure** (stop names, operator logos, route maps) — uniquely identifies city — ⛔ OVERRULES coordinates, hints, and all other evidence
2. **OCR with place names** (street signs with district names, business addresses with city) — city/district level
3. **Tight coordinate cluster** (<50km agreement across multiple models) — narrows region but does NOT identify city; cannot override transit evidence
4. **Visual match** (high similarity to reference photos) — strong for specific locations
5. **License plate region codes** — narrows to state/province only, NOT city
6. **Country-level agreement** — confirms country but NOT city
7. **User hint** — soft prior only; NEVER overrides transit or OCR evidence

## CONFLICT RESOLUTION
When evidence sources disagree:
- A named transit stop or operator logo ALWAYS beats a coordinate cluster or user hint
- Multiple models agreeing on coordinates does NOT override a transit operator identification — coordinates are approximate; transit names are exact
- State the conflict explicitly in your reasoning
- Follow the more specific, more reliable evidence
- Lower confidence when sources conflict

## CONFIDENCE CALIBRATION (STRICT — assigning high confidence without city-specific evidence is a critical error)
- 0.9–1.0: Named transit stop AND/OR OCR with city/district name AND tight coordinate cluster AND visual match
- 0.7–0.9: Transit operator or route identified AND coordinate agreement AND country-level corroboration
- 0.5–0.7: Country certain, city inferred from regional evidence but NO city-specific text/transit — ⛔ MAXIMUM confidence when you lack city-specific identifiers
- 0.3–0.5: Country certain, city guessed from general regional features with no specific evidence
- 0.0–0.3: Country uncertain or multiple countries possible

⛔ NEVER assign confidence ≥0.7 without a named transit stop, operator, or OCR text containing a city/district name. This is a hard rule — violations are errors.

Return your answer as JSON:
{{
  "name": "Most specific location name (neighborhood/district if possible, then city)",
  "country": "Country name",
  "region": "State/province",
  "city": "City name",
  "latitude": float,
  "longitude": float,
  "confidence": 0.0-1.0 (strictly per calibration scale above — no exceptions),
  "transit_evidence": "Named transit stop/operator identified from image, or 'NONE — no city-specific transit evidence found'",
  "ocr_evidence": "Key OCR text extracted from image, or 'NONE — no city-specific OCR evidence found'",
  "reasoning": "Step-by-step: (1) Country evidence, (2) Transit identifiers extracted — name each one and state which city it maps to, (3) OCR text extracted and what it indicates, (4) City discrimination — cite specific evidence for chosen city vs alternatives, (5) Conflict resolution if applicable, (6) Confidence justification per calibration scale",
  "evidence_used": ["list of key evidence pieces, each tagged as 'city-specific' or 'region-level' or 'country-level'"]
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

    # ---- Call 2: reasoning to a prediction (strict evidence-hierarchy
    # prompt from open_geo_spy when we have evidence text; simple otherwise)
    evidence_txt = evidence_summary or "none"
    if evidence_txt != "none":
        reasoning_prompt = STRICT_REASONING_PROMPT.format(
            evidence=evidence_txt[:6000],
            env_type=(features.get("environment_type", "UNKNOWN")
                      if isinstance(features, dict) else "UNKNOWN"),
            location_hint=location_hint or "none",
        )
    else:
        reasoning_prompt = REASONING_PROMPT.format(
            features=json.dumps(features, ensure_ascii=False)[:4000],
            evidence=evidence_txt[:2000],
        )
    messages2 = [{
        "role": "user",
        "content": [
            {"type": "text", "text": reasoning_prompt},
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
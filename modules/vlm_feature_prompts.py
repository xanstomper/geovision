"""
VLM Geolocation Feature-Extraction Prompts
==========================================
Ports the structured geolocation feature-extraction prompts from open_geo_spy
(github.com/eren23/open_geo_spy, src/extraction/features.py).

These prompts drive a VLM (any OpenAI-compatible vision model) to extract
ALL geolocation-relevant features in one pass as structured JSON —
the single highest-leverage prompt pattern in that repo.

Ported from: open_geo_spy/src/extraction/features.py (MIT)
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

FEATURE_EXTRACTION_PROMPT = """Analyze this image for geolocation clues. Extract ALL visual features that could help identify the location.

Return a JSON object with these fields:
{
  "landmarks": ["list of recognizable landmarks, monuments, statues"],
  "architecture_style": "dominant architectural style (e.g., 'Central European', 'Southeast Asian', 'Modern American')",
  "building_types": ["types of buildings: residential, commercial, industrial, religious, government"],
  "vegetation": {
    "type": "tropical/temperate/arid/boreal/none",
    "density": "dense/moderate/sparse/none",
    "notable_species": ["palm trees", "pine trees", etc.]
  },
  "terrain": ["flat", "hilly", "mountainous", "coastal", etc.],
  "water_bodies": ["ocean", "river", "lake", etc.],
  "infrastructure": {
    "road_type": "highway/urban_road/rural_road/path/none",
    "road_markings": "description of lane markings, colors",
    "traffic_side": "left/right/unclear",
    "power_lines": true/false,
    "rail": true/false
  },
  "vehicles": {
    "types": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "notable": "any distinctive vehicle features (brand, taxi color, bus style)"
  },
  "environment_type": "URBAN/SUBURBAN/RURAL/INDUSTRIAL/AIRPORT/COASTAL/FOREST/MOUNTAIN/DESERT/PARK/HIGHWAY",
  "weather_climate": "sunny/cloudy/rainy/snowy/foggy + hot/warm/cool/cold",
  "time_of_day": "morning/midday/afternoon/evening/night",
  "cultural_indicators": ["flags", "writing systems", "clothing styles", "food types"],
  "country_clues": ["specific clues pointing to a country or region"],
  "confidence_notes": "brief note on how diagnostic these features are"
}
"""

FEATURE_EXTRACTION_PROMPT_WITH_HINT = """Analyze this image for geolocation clues.

CONTEXT: The user suggests this image may be from: {location_hint}

Treat this as optional guidance: look for features that confirm OR contradict it. Contradictions are as important as confirmations.

Extract ALL visual features that could help identify the location. Return a JSON object with these fields:
{{
  "landmarks": ["list of recognizable landmarks, monuments, statues - prioritize those known in {location_hint}"],
  "architecture_style": "dominant architectural style typical of this region",
  "building_types": ["types of buildings: residential, commercial, industrial, religious, government"],
  "vegetation": {{
    "type": "tropical/temperate/arid/boreal/none",
    "density": "dense/moderate/sparse/none",
    "notable_species": ["species typical for {location_hint}"]
  }},
  "terrain": ["flat", "hilly", "mountainous", "coastal", etc.],
  "water_bodies": ["ocean", "river", "lake", etc.],
  "infrastructure": {{
    "road_type": "highway/urban_road/rural_road/path/none",
    "road_markings": "description of lane markings, colors - note if typical for {location_hint}",
    "traffic_side": "left/right/unclear - verify this matches {location_hint}",
    "power_lines": true/false,
    "rail": true/false
  }},
  "vehicles": {{
    "types": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "notable": "any distinctive vehicle features typical for {location_hint} (brand, taxi color, bus style)"
  }},
  "environment_type": "URBAN/SUBURBAN/RURAL/INDUSTRIAL/AIRPORT/COASTAL/FOREST/MOUNTAIN/DESERT/PARK/HIGHWAY",
  "weather_climate": "sunny/cloudy/rainy/snowy/foggy + hot/warm/cool/cold",
  "time_of_day": "morning/midday/afternoon/evening/night",
  "cultural_indicators": ["flags", "writing systems", "clothing styles", "food types typical for {location_hint}"],
  "country_clues": ["specific clues confirming or contradicting {location_hint}"],
  "hint_verification": {{
    "supports_hint": true/false,
    "contradictions": ["features that contradict {location_hint}"],
    "confidence_in_hint": 0.0-1.0
  }},
  "confidence_notes": "brief note on how well features align with {location_hint}"
}}
"""


def encode_image_data_url(image_path: str) -> str:
    """Encode a local image as a data: URL for OpenAI-compatible vision APIs."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    lower = image_path.lower()
    mime = "image/jpeg"
    if lower.endswith(".png"):
        mime = "image/png"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    return f"data:{mime};base64,{b64}"


def extract_visual_features(
    image_path: str,
    client: Any,
    model: Optional[str] = None,
    location_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured geolocation features from an image via a VLM.

    Uses the open_geo_spy prompt pattern: one pass, full structured JSON.
    Works with any OpenAI-compatible client (OpenAI, OpenCode Zen, etc.).

    Args:
        image_path: path to the image
        client: OpenAI-compatible client instance (sync)
        model: model name for vision; provider default if None
        location_hint: optional user hint — enables contradiction checking

    Returns:
        parsed feature dict (or {"error": ...} on failure)
    """
    import json

    prompt = (
        FEATURE_EXTRACTION_PROMPT_WITH_HINT.format(location_hint=location_hint)
        if location_hint
        else FEATURE_EXTRACTION_PROMPT
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
            ],
        }
    ]

    kwargs: Dict[str, Any] = {"messages": messages}
    if model:
        kwargs["model"] = model

    try:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        # Strip markdown fences if present
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "raw": text if 'text' in dir() else None}
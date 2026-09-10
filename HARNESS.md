# GeoVision Harness — Agent & Model Onboarding

The **GeoVision Harness** is a model-agnostic geolocation *investigation engine*.
Any AI agent or vision-capable LLM can drive it to automatically geolocate an
image, deep-dive every clue, and produce a complete, evidence-backed case file —
without knowing GeoVision's internals.

It turns GeoVision from "a tool you call for a coordinate" into "an investigator
you delegate a target to."

---

## How to drive it (all models, all interfaces)

You (the model) supply an **image path**. GeoVision returns a structured
case record. You do the visual reading; GeoVision does the deterministic GIS,
retrieval, cross-checking, and case-file persistence.

**Option A — MCP tool (recommended for agents):**
```
investigate_image {
  "image_path": "/abs/path/to/photo.jpg",
  "location_hint": "optional region/city",
  "evidence_summary": "optional clues you already spotted (OCR text, chain stores, etc.)",
  "top_k": 3,
  "save_case": true,
  "case_name": "OP-1234",
  "case_description": "Subject location investigation",
  "case_tags": ["ops","field"]
}
```

**Option B — CLI:**
```bash
python3 geovision_cli.py investigate /path/to/photo.jpg --save-case --case-name "OP-1234"
python3 geovision_cli.py investigate /path/to/photo.jpg -j        # raw JSON
```

**Option C — direct (Python):**
```python
from modules.geo_harness import GeoVisionHarness
case = GeoVisionHarness().investigate("photo.jpg", save_case=True, case_name="OP-1234")
```

All three return the same case-record shape (below).

---

## What runs under the hood (the 3-stage investigation)

```
Image
  │
  ├─ STAGE 1  VLM coarse reasoning  (optional, any OpenAI-compatible vision model)
  │           · scene description, country/region/city guess
  │           · negative evidence (what is ABSENT → what regions are ruled out)
  │           · solar/shadow geometry, vegetation → latitude band
  │           OUTPUT: priors + constraints
  │
  ├─ STAGE 2  Deterministic constrained retrieval  (no API, always runs)
  │           · GeoCLIP direct GPS regression
  │           · CLIP ViT-B-32 on real geotagged reference DB (6,714 photos)
  │           · GeoGuessr CV heuristics (driving side, road marks, plates, signs, soil)
  │           · environment/biome + OCR
  │           · SpatialConstraintSolver = negative-evidence ELIMINATION lattice
  │             (rejects candidates that violate observed physical/legal constraints)
  │           OUTPUT: ranked, constraint-pruned candidate list
  │
  ├─ DEEP-DIVE  best-effort OSINT  (each isolated, fails cleanly)
  │           · reverse-image unmask, chain-store / OSM lookups
  │
  ├─ STAGE 3  VLM verification  (optional)
  │           · for each top candidate, fetch REAL ground photos (Wikimedia/Mapillary)
  │           · show query + reference side-by-side; VLM cross-checks & explains
  │           OUTPUT: per-candidate match verdict + reasoning
  │
  └─ FUSION  → best estimate + uncertainty radius + reasoning chain
```

Each stage is isolated — a failed stage is skipped cleanly and the harness still
returns a structured result. **No fabricated output, ever.**

### The VLM is model-agnostic
Point it at ANY OpenAI-compatible vision endpoint — Gemini, GPT, Claude,
OpenRouter, or a local LLaVA/Ollama model:
```bash
export GEOVISION_VLM_BASE_URL=...   # e.g. your provider's /v1
export GEOVISION_VLM_API_KEY=...
export GEOVISION_VLM_MODEL=...       # e.g. gemini-2.5-pro, gpt-4o, llama-vision
```
No VLM configured? The deterministic Stage 2 (GeoCLIP / CLIP / heuristics /
constraints / ground imagery) still produces a full result.

---

## The case record returned

```jsonc
{
  "harness": "GeoVisionHarness",
  "status": "success | limited",
  "image_path": "...",
  "case_id": 7,                       // when save_case=true
  "best_estimate": {
    "latitude": .., "longitude": .., "confidence": 0.0-1.0,
    "source": "geoclip", "vlm_verification": true?
  },
  "uncertainty": { "uncertainty_radius_km": .., "granularity": "city" },
  "candidates": [ { "latitude", "longitude", "confidence", "source" } ],
  "constraints_applied": ["driving_side=right", "environment=urban"],
  "stages": {
    "coarse_reason": { "status": "skipped|success", "prediction": {...} },
    "deterministic_scan": { "signals": { geoclip, visual_geo, geoguessr, environment, ocr } },
    "verify_candidates": { "verifications": [ { "candidate", "reference_photos", "vlm_verdict" } ] }
  },
  "deep_dive": { "reverse_image": {...}, "chain_stores": {...} },
  "reasoning_chain": ["COARSE: ...", "VERIFY match at ..."],
  "duration_s": ..,
  "notes": ["Eliminated (VLM): ..."]
}
```

---

## How your agent should use it (recommended flow)

1. **Look** at the image yourself (you have vision).
2. Gather clues: sign text, chain stores, architecture, terrain, oddities.
3. Call `investigate_image` with those clues as `evidence_summary` + `save_case: true`.
4. Read the returned case record: candidates, constraints, verification verdicts.
5. **Augment** — if the top candidate is unsatisfying, pass more clues or a
   `location_hint` and re-run; or dig with the granular sub-tools
   (`osm_query`, `satellite_landcover`, `nearby_ground_imagery`,
   `reverse_image_search`, `resolve_vision_clues`).
6. **Annotate** the case via the CaseManager (notes) and export the case JSON.

The `reasoning_chain` is your chain-of-evidence: when you present the final
answer, cite which candidate won, which constraints confirmed it, and which
regions were eliminated and why.

---

## Which tool for what job

| Need | Tool |
|---|---|
| **LIVE detective canvas — see everything the agent does visually** | **`canvas_start` + `canvas_add` + `canvas_finish`** (MCP) — open viewer_url in a browser; auto-refreshes. Pass `canvas_session` into `investigate_image` to stream the whole investigation |
| Full investigation + case file | `investigate_image` (harness) |
| **Pull many live signals to cross-examine & problem-solve** | **`query_signals`** (MCP) / **CLI `signals`** — web + imagery + OSINT + weather + city + geo, live, key-free |
| **Generate a professional case report** | **`render_case_report`** (MCP) / **CLI `render-report`** — auto-written on `--save-case` |
| **List/track property & business listings (hotels, rentals, offices, shops)** | **`query_listings`** (MCP) / **CLI `track-properties`** |
| **Find exact house/listing from a photo (indoor)** | **`find-listing`** (CLI; needs free GOOGLE_VISION_API_KEY for reliability) |
| **Grow the real reference index (more real data)** | **`grow_reference_db`** (CLI `grow-db`), **`scripts/batch_grow_db.py`** (many regions) |
| **Add street-level layer (Mapillary)** | `nearby_ground_imagery` (uses local gitignored `MAPILLARY_ACCESS_TOKEN`) |
| **Add OSV-5M street-layer subset** | `scripts/osv5m_subset_ingest.py` (pulls HF shard; ~2.25GB, on-demand) |
| One coordinate fast | `geolocate_quick` |
| Deep scan, all platforms | `geolocate_image` |
| Pick between candidates with photos | `nearby_ground_imagery`, `verify` |
| Poke a candidate with GIS data | `osm_query`, `satellite_landcover`, `weather_corroborate` |
| Research a found place name | `forward_geocode`, `reverse_geocode`, `search_web` |

---

## Honesty & verification notes

- **GeoCLIP softmax probs ~0.01** — the *coordinates* are the signal, not the
  probability. Judge by top-3 spatial spread and multi-model agreement, not the
  raw score.
- **Confidence bands**: 0.85–0.98 = confirmed entity/address; 0.50–0.70 =
  regional architectural match; 0.20–0.45 = generic scene / unconfirmed centroid.
- **`limited` / `skipped`** stage statuses are legitimate outcomes, not failures.
- Reference DB (`data/visual_geo_db`) holds **real geotagged photos**; the
  harness never synthesizes a match.
- The whole pipeline is local — image data never leaves the machine unless you
  configure a VLM endpoint.

## Security & robustness (for agent authors)

- **No internal API keys.** GeoVision ships zero baked-in credentials. Real keys
  (VLM, Google, Mapillary) are read from env vars when present, and every tool
  **degrades gracefully** (status `skipped`/`partial`/`limited`) when a key is
  absent or a public API is down — the deterministic, free layer (GeoCLIP, CLIP
  reference DB, geoguessr heuristics, GeoNames city snap, Wikimedia geosearch,
  Overpass fallback, Open-Meteo, ArcGIS ESRI) always works with zero keys.
- **Tools are bounded.** Slow public APIs (Overpass/Nominatim) are wrapped so a
  tool call returns within a budget (default `budget_seconds`, ~20s; tunable).
  A CV model never blocks indefinitely; on timeout you get a `status: partial`
  result grounded by the fast offline GeoNames index. Pass `budget_seconds` to
  control the tradeoff between completeness and latency.
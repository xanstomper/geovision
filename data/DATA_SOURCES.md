# GeoVision Data Sources

All datasets below are REAL, free, and downloaded in-place. Nothing here is
synthetic or mocked.

## Reference DB (visual geolocation)

- `data/visual_geo_db/` — CLIP embeddings of real geotagged Wikimedia Commons
  photographs, seeded from GeoNames world cities.
  - Built by `scripts/build_reference_db.py` (append mode to grow)
  - NOT in git (rebuildable) — embeddings.npy + meta.jsonl.gz

## GeoNames gazetteer (`data/geonames/`)

Downloaded from https://download.geonames.org/export/dump/ (CC BY 4.0):

| File | Contents |
|------|----------|
| `cities15000.txt` | 34,135 cities (population ≥ 15k) with lat/lon/population |
| `cities5000.txt` | 69,697 cities (population ≥ 5k) with lat/lon/population |
| `countryInfo.txt` | 302 countries — ISO codes, names, capitals, population, area, continent |
| `admin1CodesASCII.txt` | 3,865 admin-1 divisions (states/provinces) with codes |

These power city/country snapping: any predicted coordinate can be resolved
to the nearest real city instantly (no API call), and country hints map to
ISO codes offline via `modules/country_matcher.py`.

## Eval datasets (`data/eval/`)

- `wikipedia_landmarks_v1/` — 8 real landmark photos (Eiffel Tower, Statue of
  Liberty, Taj Mahal, Big Ben, Colosseum, Christ the Redeemer, Sydney Opera
  House, Great Pyramid) with ground-truth GPS. Ported from open_geo_spy.
  Run: `python3 scripts/run_eval.py`

## Live sources (queried at runtime, no local copy)

| Source | Use |
|--------|-----|
| Wikimedia Commons geosearch API | real geotagged photos near candidates |
| OSM Overpass API | infrastructure/POI/property verification |
| OSM Nominatim | forward/reverse geocoding |
| Open-Meteo historical API | weather corroboration |
| ArcGIS World Imagery tiles | satellite cross-view matching |
| Mapillary v4 (token-gated) | street-level imagery verification |

## Model weights (auto-download, cached, NOT in git)

| Model | Cache location | Size |
|-------|----------------|------|
| CLIP ViT-B-32 (laion2b) | `data/clip_cache/` | 578 MB |
| GeoCLIP | `~/.cache/huggingface` | ~1 GB |
| StreetCLIP | `~/.cache/huggingface` | ~600 MB |
| ResNet50 | torch hub cache | ~100 MB |

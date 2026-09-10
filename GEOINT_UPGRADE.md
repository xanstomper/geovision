# GeoVision → GeoSpy/Raven-class: Methodology Upgrade Build Plan

Research-backed spec for turning GeoVision into a GeoSpy-Raven-class geolocation
engine — replicating the real methodology and adding a strictly-better regional
cascade. **Written before implementation** as the detailed build prompt.

---

## 1. How GeoSpy / Raven actually works (researched, verified against public sources)

Graylark's **GeoSpy** (and its OSINT-grade backend, Raven) is a *retrieval-based*
geolocation service. The public, replicable essence:

1. **Embedding stage** — a CLIP-family vision encoder (ViT-L-sized transformer
   backbone) maps the quarry image into a joint image/text embedding space.
2. **Index stage** — a large pre-built index of **geotagged geographies / street
   imagery** (photo → GPS). GeoSpy's scale edge comes from a very large such
   index (hundreds of millions of geotagged images), effectively the whole
   street-level internet.
3. **Retrieval stage** — the quarry embedding is compared (cosine / FAISS) against
   the index; the top-K spatially-debiased matches' coords are aggregated
   (median/mean) into a final GPS estimate with a confidence.
4. The model that maps features→coords is *trained* on massive geotagged data —
   the open source equivalent is **GeoCLIP** (NeurIPS'23, GPS-positional-encoder
   cross-attention over a **100k-point global GPS gallery**, trained on the
   curated GeoGLUE-style set), **StreetCLIP** (CLIP fine-tuned on ~1.1M geotagged
   street photos for country ZS), and the **OSV-5M** dataset (~5M geotagged
   street-view images, 225 countries) used to train these.

### Data sources GeoSpy-class tools rely on (all real, open, ingestible)
| Source | What it is | License/access | Value |
|---|---|---|---|
| **OSV-5M** (osv5m/osv5m on Hub) | ~5.1M geotagged street-view images, 225 countries | research/analysis (Mapillary-derived) | the standard training index |
| **Mapillary v4 API** | real street-level imagery API | token-gated, licensing-clear | live street refs |
| **StreetCLIP** (geolocal/StreetCLIP) | CLIP fine-tuned on 1.1M geotagged street imgs | HF weights | ZS country prior |
| **GeoCLIP GPS gallery** | 100k-point uniform global GPS grid baked into the model | in-model | dense deterministic global prior |
| **Wikimedia Commons geosearch** | real geotagged photos API | un-gated / free | live dense reference growth |
| **GeoNames cities** | 34k–70k world cities | CC BY 4.0 | city snapping |

---

## 2. The better methodology: coarse-to-fine regional cascade

Pure global retrieval (GeoSpy's core) wastes most of its index on the wrong
hemisphere and is fragile to ambiguous scenes. The better method — **spatial
cascade with hard constraints** — is what this build implements:

```
QUARRY IMAGE
  │
  ├─ C0  Deterministic feature + coarse GPS prior
  │     · GeoCLIP predicts a continuous GPS + confidence   (real, global)
  │     · StreetCLIP country prior                          (ZS, coarse)
  │     ──> coarses: a global lat/lon band + country set
  │
  ├─ C1  REGION-CONSTRAINED RETRIEVAL   ← the upgrade (better than global top-K)
  │     · restrict CLIP reference-DB nearest-neighbor SEARCH to references whose
  │       real geotag is within a radius R of the coarse GPS prior
  │     · re-rank within-region by CLIP cosine × spatial agreement
  │     ──> street/photo-level candidates, ambiguity already pruned by region
  │
  ├─ C2  Negative-evidence elimination (hard constraints)
  │     · SpatialConstraintSolver rejects region/candidates that violate
  │       observed driving-side, road-mark, plate, soil, solar-hemisphere
  │     ──> impossible coords dropped, not just down-weighted
  │
  ├─ C3  Ground-truth visual verification
  │     · fetch real Wikimedia/Mapillary photos at each surviving candidate
  │     · Stage-3 VLM cross-check (model-agnostic) / StreetCLIP vector sim
  │     ──> per-candidate match verdict + reasoning
  │
  └─ C4  Fusion → best_estimate + uncertainty + reasoning_chain + case file
```

**Why this beats GeoSpy:** region-restricting retrieval reduces the candidate
space to the *likely* area before nearest-neighbor, so a near-duplicate in the
wrong hemisphere can't win on raw cosine similarity. Combined with hard negative
evidence + ground-truth verification + a chain of evidence, it is strictly more
constraint-aware than a single black-box retrieval pass.

---

## 3. Real-data growth path (the "more real data somehow" ask)

- **Region-restricted dense crawl**: extend the real crawler to fetch many more
  geotagged Wikimedia photos **inside each GeoCLIP-GPS-prior radius** and CLIP-
  embed them into `data/visual_geo_db` (append). Every ref is a real photo with
  a real GPS — this grows the retrieval index exactly like GeoSpy grows its index.
- **Mapillary** (token) optional: real street-level, licensing-clear.
- **OSV-5M subset**: the Hub dataset is ~GBs/millions of images — not practical to
  ingest fully on this CPU box, but a *subset* can be pulled with the HF hub for
  the highest-priority regions to seed a street-level index. Flagged as a
  follow-on (bandwidth/disk), not faked.
- Index math: CLIP ViT-B-32 = 512-dim L2-normalized; cosine = knn on normalized
  embeddings. FAISS can accelerate at scale; the current numpy dot-product is
  fine for tens of thousands, the mean radius keeps it small.

---

## 4. Phases & deliverables

| Phase | Deliverable | Status |
|---|---|---|
| R0 | Research + this plan | ✅ done |
| R1 | `regional_retrieval` phase in `GeoVisionHarness` (region-constrained CLIP-NN) | ✅ done |
| R2 | Wire regional retrieval into `investigate()` fusion + reason chain | ✅ done |
| R3 | Expose region param + retrieval detail via CLI `--radius` / MCP args | ✅ done |
| R4 | Verify region-constrained vs global top-K on the real eval set | ✅ 13/13 logic + real Eiffel: in-region 69/6714 → Paris, cross-hemisphere pruned |
| R5 | Dense-region reference crawler (append real photos by GPS prior) | ✅ `GeoVisionHarness.grow_reference_db()` + CLI `grow-db` + MCP `grow_reference_db` |
| R6 | Offline Stage-3 visual verification (StreetCLIP cross-view, no VLM key) | ✅ `visual_verify_candidates` + 2 real bugs fixed (thumb URLs, encode pooling); pytest 3/3 |
| R7 | Regional-vs-global real benchmark (`scripts/regional_vs_global_benchmark.py`) | ⏳ run; CPU-bound, partial: Statue of Liberty 12.4→9.1km regional-better; full 8-img >25min on CPU (GPU needed) |
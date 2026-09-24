# Competitor Moats — Commercial & OSINT Geolocation Tool Landscape (2026-09-24)

Research for GeoVision (`~/geovision`): what each competitor actually offers (verified from
fetched pages only), what server-side data/index they ride on, pricing, the moat, and the
exploitable weakness GeoVision can target.

**Method / honesty note.** Every claim below marked ✅ comes from a page fetched this session
(curl, HTTP code recorded). Anything marked ⚠ is community knowledge or prior research
(`~/.hermes/skills/osint/geovision-geolocate/references/external-tool-methodologies.md`) and
was NOT verifiable from this box this session — treat accordingly. Blocked fetches are listed,
not guessed around.

| Fetch | Result |
|---|---|
| geospy.ai (+ /blog, /blog/lgm-vs-llm) | ✅ 200 |
| app.geospy.ai | ✅ 200 — JS shell only ("Raven by Graylark", 3KB) |
| oceanir.ai, /pricing, /blog×3, /solutions | ✅ 200 |
| docs.oceanir.ai (+5 subpages) | ✅ 200 |
| app.oceanir.ai | ✅ 200 |
| picarta.ai (+ /pricing, /api) | ✅ 200 |
| support.google.com/websearch/answer/1325808 (Lens/search-by-image) | ✅ 200 |
| yandex.com/images | ✅ 200 (JS app; meta text only) |
| yandex.com/support/images-search* | ❌ 404 |
| tineye.com, tineye.com/how-it-works | ❌ 403 Cloudflare JS challenge |
| en.wikipedia.org REST summaries (Lens, TinEye, GeoGuessr, Yandex) | ✅ 200 |
| geohints.com | ✅ 200 |
| plonkit.net | ❌ JS-walled (empty shell) |
| limnero.com | ✅ 200 — **cat-toy shop; name is a decoy, not a geo tool** |
| graydata.io | ❌ unreachable (000) |

---

## 1. GeoSpy (Graylark Technologies) — geospy.ai

**Positioning (✅ fetched):** "The Visual Intelligence Platform. Turn photos into intelligence.
No metadata required." "Helped solve over 10,000+ cases worldwide." Named capabilities on the
landing page: **Geo-Estimation**, **Precision Targeting**, **Property Targeting** — "take low
context images and determine locations using only the pixels."

**Business model (✅ fetched):** "Available for qualified government and enterprise
organizations… up to meter level accuracy… Contact us for licensing and pricing details." —
**No self-serve pricing exists on the site.** Sales-led, LE/gov-first. The app
(app.geospy.ai) is branded **Raven** and is a JS shell to curl.

**LGM narrative (✅ fetched, Jan 2026 blog "LGM vs LLM"):** GeoSpy claims to be a *Large
Geospatial Model* trained on "real images, scans/photos, 3D data (photogrammetric scans,
volumetric 3D data)" — a "spatial brain" answering "where am I, what am I looking at, how am I
oriented." Target verticals listed: law enforcement (fugitive caught in 20 minutes), insurance
fraud ("truthlayer" matching road texture, foliage, solar angles to the claimed location),
food-delivery proof-of-delivery ("porch DNA"), journalism/media verification.

**Server-side reliance:** proprietary training corpus (massive street imagery + 3D/scans) —
per their own copy the data is the barrier ("The data is heavy… building an LGM is like
reconstructing the library"). ⚠ Community knowledge: ~100GB-scale Street View-derived index.

**Moat:** (1) proprietary training data scale incl. 3D; (2) government/LE brand trust + case
studies; (3) meter-level claim on their coverage; (4) closed distribution — you can't even
self-evaluate it without contacting sales.

**Exploitable weakness:** opaque (no evidence chain on the marketing surface; ⚠ known
black-box in practice), no free/self-serve path on current site, no transparency or
reproducibility, and closed = can't be audited, self-hosted, or chained into an analyst's own
tooling. Long-tail non-covered regions fall back to generic inference — exactly where
GeoVision's evidence-first ensemble competes.

---

## 2. OceanIR (Oceanir, LLC) — oceanir.ai / app / docs

**Positioning (✅ fetched):** "**A pin is not a verification.** Oceanir shows where an image
was taken, what it matches in the real world, and the visual evidence behind the result. From
the pixels, not the metadata." "**Tells you when it is not sure.**" Landing metrics: **52s
median review time; 33,800+ images/clips processed.**

**Model (✅ docs):** current REST model **orca-1.5**; **Manta** unreleased; **M1** announced
(2026-09-02): one model, three modes — **Geo-Estimation** (works anywhere, image alone),
**Street-Match** ("resolved against a surveyed reference… recognition instead of inference"),
**Property-Verify** ("the proof, or I abstain"). M1's headline: **it can say a place is not in
coverage** — "a more useful answer than a confident guess." Access staged to named orgs, not
open.

**Depths (✅ pricing + docs):** **D1** free surface scan (no account); **D2** verified read
with reasoning = 2 credits; **D3** forensic bundle with multi-pass zoom = 10 credits.
API returns confidence 0–1 + `visual_clues` [{label, detail}] at D2+.

**Pricing (✅ fetched):** Starter **$10/mo** = 30 credits (~3 forensic runs), D3, supporting
**and contradicting** evidence per candidate, full history. Pro **$29/mo** = 100 credits
(~10 forensic runs) + **rejected candidates and why they were ruled out**, **PDF/CSV evidence
exports**, individual API access, Desktop access. Enterprise = custom, 5,000 credits/mo,
batch API 10k calls/hr, shared cases/team seats/review queues, SSO/SAML/DPA. Credit packs
$0.60–0.80/credit. Free-tier credits reset daily.

**Privacy policy as a feature (✅ docs):** every API coordinate is **rounded to a 100m grid by
default** — deterministic, irreversible; 4-decimal precision exists only as an enterprise
override with documented justification.

**Platform (✅ fetched):** "Oceanir Desktop" — a *Visual Intelligence Workspace*: ingest a
folder of images/videos locally, organize into cases, per-image precision tiers (Country Very
high / Region / City / Site High / **Precise location Low**), review notes, "the work is a
process, not a single query." **MCP support** (connect AI clients; OAuth agent tokens).
Safety Guidelines for Geolocation Models published.

**Server-side reliance:** their own reference coverage ("surveyed reference" for Street-Match
/ M1 recognition) + their model.

**Moat:** the *defensible-evidence workflow*, not the coordinates — abstention, contradiction
ledger, exports, cases, MCP distribution, honest precision tiers. Pricing sells "reads" not
seats. M1's coverage-honesty is their next lock.

**Exploitable weakness:** cloud-only (no self-host — privacy-sensitive desks can't use it),
**100m coordinate ceiling on self-serve**, and credit metering makes a deep D3 run cost
**$6–8** — heavy for bulk analysis. GeoVision inverts every one of these: self-hosted,
full precision, $0/query, same evidence-workspace UX.

---

## 3. Picarta (Picarta S.r.l., Italy) — picarta.ai

**Positioning (✅ fetched):** "Picarta finds where any photo was taken in the world using AI."
No EXIF/GPS needed — "works even on screenshots, edited photos." Homepage features:
**Search in a Country** (pick country on a map), **Search Worldwide**, **Focus on Specific
Area / Search Entire Country**. Result fields: **Latitude/Longitude, Time stamp, Camera
details, Google Maps link**, top-k map, download. "Upgrade your account to get the full
results details." Used by "OSINT investigators, journalists, law enforcement."

**Pricing (✅ fetched):** Free = **1 search**. Wallet **$15.90 one-time = 20 searches**.
Subscription **$59.9/mo (annual $49.9) = 100 searches/mo**. Enterprise = custom, production
API, **on-premise deployment**, SLA. **API is Enterprise-only** (email gated).

**API shape (✅ fetched, public docs):** Python pkg `picarta`; `/classify` POST with base64
image; params: `TOP_K` (≤10), `COUNTRY_CODE`, `ADMIN1`, `ADMIN2`, `center_latitude/longitude`
+ `radius` (**≤25km prior circle**), `context` (free-text hints ≤500 chars); admin1/admin2
enumeration endpoints (`/admin1/{cc}`, `/admin2/{cc}/{admin1}`).

**Moat:** the **region-constrained query API** (country/admin1/admin2/25km-circle priors +
free-text context) — the best parametrized geolocation API of the bunch; on-premise for
enterprises; dead-simple wallet pricing.

**Exploitable weakness:** flat predictions (coords + confidence) with **no evidence chain, no
contradictions, no verification** — a pin, exactly what OceanIR's tagline attacks. API gated
behind enterprise sales. Free tier nearly unusable (1 search).

---

## 4. Google Lens (search-by-image on Google)

**Verified behavior (✅ support page):** upload file / drag-drop / paste URL / right-click in
Chrome ("Search with Google Lens" sidebar); results include **AI Overviews, object results,
similar images, websites with the image or a similar image**, plus an "Add to your search"
text box after upload, region selection, and image removal. **It returns web matches, not
coordinates, confidence, or evidence** — geolocation is emergent, not a feature.

**Server-side reliance:** Google's full web index + Knowledge Graph + object recognition. The
moat IS the index; nobody key-free can replicate it.

**Moat:** breadth of the indexed web → unbeatable on famous places, products, landmarks,
repeat-posted images; free.

**Exploitable weakness (verified on this box, prior sessions ⚠ + confirmed bot-fencing):**
automation is hostile — `requests.post` to the Lens upload endpoint is dead; a JS-capable
stealth browser (Playwright) is required. And on the **long tail of non-famous, never-indexed
street scenes it produces "similar images" that may be thousands of km away** — no spatial
reasoning, no uncertainty, no abstention. This is precisely GeoVision's target gap.

---

## 5. Yandex Images

**Verified (✅):** yandex.com/images is live ("Search by images and photos — Yandex
Images"), a heavy JS app; support pages 404; automation captcha-fenced.

**⚠ Community knowledge (not verifiable from fetched text):** its reverse-image search is
renowned for surfacing the *original/source* page of travel & street photos via a huge
independent index with different crawl geography (RU web + ex-USSR + tourism boards), often
beating Google on obscure places. No coordinates/confidence — same emergent-geo caveat as
Lens.

**Moat:** independent massive image index (crawl geography Google doesn't duplicate).

**Exploitable weakness:** bot-fencing (captcha), RU-jurisdiction privacy concerns for
investigators, and no evidence/uncertainty layer.

---

## 6. TinEye (Idée Inc.)

**Verified (✅ Wikipedia; site itself 403 Cloudflare):** "the first image search engine to use
image identification technology rather than keywords, metadata or watermarks. Upon submitting
an image, TinEye creates a 'unique and compact digital signature or fingerprint' and matches
it with other indexed images… able to match even heavily edited versions… **will not usually
return similar images**."

**Moat:** exact-duplicate fingerprint matching over a long-lived crawl history — best-in-class
for "where has THIS exact image appeared / when did it first appear" (sorted by oldest).

**Exploitable weakness:** for geolocation it's nearly useless on first-seen images — no
similar-scene matching, no geo signals. Complementary (chronolocation/date-first-seen), not a
competitor for scene-level geolocation.

---

## 7. Knowledge corpora (not geolocators but the cue-intel moat)

- **GeoHints (✅ fetched):** human-curated per-country visual-cue atlas — bollards, camera
  gens, license plates, utility poles, road lines, signage, Google-car vehicles, house
  numbers, soil/scenery, etc. Free. Its moat is curation; GeoVision's
  `geoguessr_heuristics`/`plonkit_rules`/`utility_grid`/`camera_gen` already encode much of it
  as *classifiers* rather than reading material.
- **PlonkIt (❌ JS-walled):** GeoGuessr meta-learning guide; GeoVision already ships its
  taxonomy as `geovision_plonkit_rules`.
- **Bellingcat resources (✅ fetched):** maintains a Geolocation section; notably published
  "LLMs Vs. Geolocation: GPT-5 Performs Worse Than Other AI Models" — general LLMs underperform
  dedicated geolocation tooling; supports GeoVision's ensemble-over-single-VLM design.
- **graydata.io (❌ unreachable from this box):** could not verify — not assessed.

---

## Cross-cutting summary

| Tool | What you get | Index/data reliance | Pricing (verified) | Core moat | Exploitable weakness |
|---|---|---|---|---|---|
| GeoSpy/Raven | Ranked coords, "meter-level" claim | Proprietary mega-corpus + 3D | Sales-only (gov/LE) | Data scale + LE brand | Black box, no self-serve, no evidence UX |
| OceanIR (Orca/M1) | Ranked candidates + evidence + contradictions + abstention | Own surveyed reference coverage | Free D1; $10–29/mo; $0.60–0.80/credit; D3 = 10cr (~$6–8) | Defensible-evidence workflow, Desktop cases, MCP | 100m coord rounding, cloud-only, credit costs |
| Picarta | Top-k coords + confidence, region-constrained API | Own model | 1 free; $15.9/20; $49.9–59.9/mo; API=enterprise | Best parametrized API (admin1/2 + 25km circle + context) | No evidence chain; API gated |
| Google Lens | Web matches, similar images | Full Google web index | Free | The index itself | Bot-fenced; no geo reasoning; weak on long tail |
| Yandex Images | Similar/original-source matches | Independent RU-geography index | Free | Independent index | Captcha-fenced; no evidence layer |
| TinEye | Exact-duplicate matches, oldest-first | Own crawl fingerprint DB | Free tier ⚠ | Fingerprint + history | No similar-scene matching → weak for geo |

**GeoVision's already-won moat-busters:** key-free + self-hosted (vs OceanIR credits & GeoSpy
sales walls), full-precision coordinates (vs OceanIR's 100m rounding), free evidence chain +
detective canvas (vs OceanIR Pro features), open/reproducible (vs all black boxes), and honest
street-level benchmarking (vs marketing numbers).

---

## Top 5 differentiators GeoVision lacks — BUILDABLE key-free on CPU-only Linux, ranked by impact

1. **Stealth-browser Lens/Yandex driver (long-tail web retrieval).** The single biggest
   retrieval gap: Google Lens + Yandex hold indexes nothing key-free duplicates, and Lens is
   now hard-JS-gated. Buildable on this box: Playwright + bundled Chromium (already installed
   via `scripts/sv_sweep_playwright.py` precedent) + `modules/browser_stealth.py`, driving
   `google.com/search-by-image` upload and `yandex.com/images/search?rpt=imageview`, scraping
   matched-page titles/URLs, then geocoding matched page text via Nominatim/OSM and feeding
   results into the ensemble as a `web_match` family. Attacks exactly the long tail where
   Lens beats everything today. Impact: ★★★★★.

2. **Coverage-density abstention ("in/out of reference coverage") per candidate.** OceanIR's
   M1 headline feature — pre-empt it: for every candidate coordinate, compute reference-DB
   density (Wikimedia/Mapillary/Flickr refs within r km — it's just a histogram over the
   existing `visual_geo_db` + live-pull counts) and emit `coverage: in (N refs ≤ 25km) / out`
   with an explicit **abstention flag when a confident-looking answer has zero coverage**.
   Pure CPU, one function in `ensemble_fusion`/`geo_harness`, directly strengthens the honesty
   story OceanIR sells. Impact: ★★★★★.

3. **Region-prior workbench (Picarta-style admin1/admin2 + map-picker constraints).**
   GeoVision has `location_hint` + `regional_retrieval`, but no structured region constraint:
   add Nominatim admin1/admin2 boundary resolution + Overpass `area` filters + a
   country/region picker on the `/ocean` workspace that hard-constrains retrieval, street
   targeting, and investigative loops to the chosen admin area. Turns "probably Toscana" into
   a bounded verifiable hunt — how investigators actually work, and Picarta's best API idea.
   Impact: ★★★★☆.

4. **Case ingest + cross-image correlation (OceanIR Desktop parity).** Desktop's moat is the
   *process*: folder ingest, cases, evidence accumulating across images. GeoVision has cases,
   multi-upload, and an orphaned `MultiImageSessionCorrelator` (flagged in the skill notes as
   unwired) — finish wiring it: folder-drop ingest, cross-image candidate clustering (same
   scene/vehicle/plate across frames), per-case precision tiers, and case-level consensus.
   Impact: ★★★★☆.

5. **Rejected-candidate ledger + PDF/CSV evidence exports.** OceanIR charges Pro ($29/mo) for
   exactly this: "rejected candidates and why they were ruled out" + PDF/CSV exports. The
   ensemble already produces eliminations and the `evidence_verdict` contradictions; surface
   them as a first-class panel + CSV/PDF export in `case_report.py` (HTML exists; add CSV +
   print-PDF). Cheap build, direct under-cut of a paid feature, strong "court-ready" pitch.
   Impact: ★★★☆☆.

**Honorable mention:** a clean documented REST endpoint (`POST /api/v1/geolocate`, base64 or
URL, OpenAPI spec like OceanIR's) on the existing Flask app — the adoption channel that makes
"self-hosted, $0/query, full precision" legible against OceanIR's credit meter.

---

## BUILD STATUS (2026-09-24, GeoVision main session)

- ✅ **Gap #2 (coverage abstention) — SHIPPED** same day: `modules/coverage_abstention.py`,
  wired into `build_evidence_verdict` + `/ocean` UI badges (dense/thin/uncovered).
  Verified: Paris=dense (32 refs, 0.0km nearest), mid-Pacific=uncovered (4344km).
- ✅ **Gap #4 (cross-image correlation) — backend SHIPPED** (MultiImageSessionCorrelator
  wired into /api/oceanir multi-upload + UI correlation card); folder-drop ingest + case-level
  consensus remains.
- 🚧 **Hybrid precision inheritance** (targets geoclip @1km > ensemble @1km gap):
  wired into `_finalize` (≤25km agreement → inherit regressor coords); 24-sample
  hybrid benchmark running (~916s/sample — both pipelines per sample).
- ⏳ Gaps #1 (stealth Lens/Yandex driver), #3 (region-prior workbench),
  #5 (rejected-candidate ledger + exports) — not started.

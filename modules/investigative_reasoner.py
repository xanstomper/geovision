"""
Investigative Reasoner — the Gemini-style exact-address investigation loop.
==========================================================================

Why this exists: pure visual pattern-matching (GeoCLIP/StreetCLIP/patch
consensus) resolves country/city/region reliably but plateaus at "similar
vernacular suburb 500km away" for exact street addresses. A frontier
assistant (Gemini) with web-search grounding beat the visual pipeline on a
Madison Heights VA house by running an *investigative* loop instead:

  1. HIGH-LEVERAGE CLUE EXTRACTION — read the mailbox number (151), spot the
     neighbor's full rooftop solar array, note "newer planned subdivision".
  2. PROPERTY-RECORD RESEARCH — query property records / listings in the
     candidate town for those exact features -> find the NEIGHBOR at
     147 W Stratford Place.
  3. DEDUCTION — US street numbering: consecutive odd numbers share one
     side; adjacent lot to 147 -> 151 Stratford Pl.
  4. VERIFICATION — geocode + map/Street View cross-check of the deduced
     address.

This module implements exactly that loop, key-free, honestly:

  Stage A  extract_high_leverage_clues()  — one VLM call with an
           investigator prompt (visible numbers, neighbor features,
           setting). Skips cleanly when no VLM key; an agent with its own
           vision can pass `clues` directly and research still runs.
  Stage B  research_property_records()    — feature-rich web queries
           (city+state+features+listing-site anchors, the pattern proven in
           references/real-estate-web-search.md) via LiveSignalOrchestrator,
           then address harvesting from listing URLs/titles.
  Stage C  apply_numbering_deduction()    — parity/adjacency rules turn a
           found NEIGHBOR address into the TARGET address when the clue
           number matches (or ±2 stepping when it doesn't).
  Stage D  verify_addresses()             — Nominatim geocode of found +
           deduced addresses; only geocoded candidates enter the ensemble
           as a genuinely independent signal family ("investigative" —
           text/records evidence, zero overlap with visual engines).

Honesty rules (repo law):
  - No fabricated addresses. If research returns nothing, status=limited
    and the stage contributes zero candidates.
  - Confidence is uncalibrated and capped (0.85) — real confirmation needs
    Street View / ground truth.
  - Every deduced candidate carries its reasoning string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Wall-clock budgets (this box's Overpass/DDG endpoints degrade regularly —
# the harness must never block on them).
VLM_BUDGET_S = 90
RESEARCH_TIMEOUT_S = 18
MAX_QUERIES = 4
MAX_GEOCODES = 4

# Confidence policy (uncalibrated, honest caps)
CONF_FOUND_GEOCODED = 0.70      # address surfaced by a real listing/records hit
CONF_DEDUCED_NUMBER_MATCH = 0.82  # deduced AND matches an image clue number
CONF_DEDUCED_PARITY_ONLY = 0.62   # adjacent-parity stepping without number match
CONF_NO_GEOCODE = 0.40
CONF_CAP = 0.85

# US street-numbering deduction windows
PARITY_MATCH_WINDOW = 8   # |found - clue| within this -> same-side neighbor
PARITY_STEP_WINDOW = 2    # adjacent-lot stepping: N±2 shares the street side

INVESTIGATIVE_CLUE_PROMPT = """You are a real-world location investigator. Study this image and extract the HIGH-LEVERAGE clues that could resolve it to an EXACT street address — not a country or city.

Look specifically for:
1. visible_numbers: every legible number (mailboxes, house fronts, curb markings, address plaques, street signs, store numbers)
2. street_names: any readable street/road/subdivision/neighborhood name signs
3. high_leverage_features: unusual, SEARCHABLE features on THIS or NEIGHBORING properties (e.g. "full rooftop solar panel array on the neighbor", "bright blue front door", "attached RV garage", "stone gateposts") — things real-estate listings, county records or local news would mention
4. property_style: architecture of the main building (e.g. "two-story vinyl-sided colonial", "ranch with attached garage")
5. setting: the kind of place (e.g. "newer planned subdivision with rolling terrain", "rural farmland", "dense urban rowhouses")
6. business_names: any readable shop/company/sign names that carry location value
7. region_guess: your best guess of town/county/state (or country if not US)

Return ONLY JSON:
{"visible_numbers": [], "street_names": [], "high_leverage_features": [], "property_style": "", "setting": "", "business_names": [], "region_guess": ""}"""

# Listing URL → address extractors (extends listing_finder's pair with the
# Redfin/Homes/Trulia path shapes). Order matters: most specific first.
_LISTING_URL_PATTERNS = (
    re.compile(r"zillow\.com/homedetails/([0-9][^/]+)/", re.I),
    re.compile(r"realtor\.com/realestateandhomes-detail/([0-9][^?_/]+)", re.I),
    re.compile(r"redfin\.com/[A-Z]{2}/[^/]+/([0-9][^/]+)", re.I),
    re.compile(r"homes\.com/property/([0-9][^/]+)/", re.I),
    re.compile(r"trulia\.com/home/([0-9][^/]+)", re.I),
)

# Plain-text US street address (titles/snippets of records & listing pages)
_TEXT_ADDRESS_RE = re.compile(
    r"\b(\d{1,6})\s+((?:[A-Z][A-Za-z0-9.'\-]*\s){1,5}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Pl|Place|Ct|Court|"
    r"Way|Blvd|Boulevard|Ter|Terrace|Cir|Circle|Hwy|Highway|Pkwy|Parkway|"
    r"Trail|Trl|Path|Walk|Row))\b"
)

# Street-suffix canonicalization so "147 W Stratford Place" vs "W Stratford Pl"
# compare equal across listing sites.
_SUFFIX_CANON = {
    "street": "st", "ave": "ave", "avenue": "ave", "road": "rd", "rd": "rd",
    "drive": "dr", "lane": "ln", "place": "pl", "pl": "pl", "court": "ct",
    "circle": "cir", "boulevard": "blvd", "way": "way", "terrace": "ter",
    "parkway": "pkwy", "highway": "hwy", "trail": "trl", "st": "st",
}
_DIR_WORDS = {"n", "s", "e", "w", "north", "south", "east", "west",
              "ne", "nw", "se", "sw"}


def _canon_street(name: str) -> str:
    """'W Stratford Place' -> 'w stratford pl' (direction kept, suffix canon)."""
    toks = [t for t in re.split(r"[\s.]+", name.lower().strip()) if t]
    if not toks:
        return ""
    if toks[-1] in _SUFFIX_CANON:
        toks[-1] = _SUFFIX_CANON[toks[-1]]
    toks = [t for t in toks if t not in ("the",)]
    return " ".join(toks)


def _street_core(name: str) -> str:
    """Drop the leading direction so 'W Stratford' ~ 'Stratford' matches."""
    toks = _canon_street(name).split()
    if toks and toks[0] in _DIR_WORDS:
        toks = toks[1:]
    return " ".join(toks)


class InvestigativeReasoner:
    """Gemini-style exact-address loop: clues -> records research -> deduction
    -> geocode verification. Key-free; degrades honestly at every stage."""

    def __init__(self, timeout: int = RESEARCH_TIMEOUT_S):
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Stage A — high-leverage clue extraction (VLM)                       #
    # ------------------------------------------------------------------ #
    def extract_high_leverage_clues(self, image_path: str) -> Dict[str, Any]:
        """One VLM investigator call — STRICTLY OPT-IN: reads only
        GEOVISION_VLM_* env vars (the calling agent may point it at any
        OpenAI-compatible endpoint). GeoVision itself ships with NO model API
        keys. The primary path is agent-supplied clues: the calling agent uses
        its OWN vision and passes `clues=` — then this stage is never needed."""
        api_key = os.environ.get("GEOVISION_VLM_API_KEY")
        if not api_key:
            return {"status": "skipped",
                    "note": ("no VLM configured (GeoVision ships key-free) — "
                             "supply clues from your own vision instead")}
        try:
            from openai import OpenAI
            base_url = os.environ.get("GEOVISION_VLM_BASE_URL",
                                      "https://opencode.ai/zen/v1")
            model = os.environ.get("GEOVISION_VLM_MODEL", "gpt-5.4-nano")
            client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception as e:
            return {"status": "skipped", "note": f"vlm client unavailable: {str(e)[:120]}"}

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": INVESTIGATIVE_CLUE_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": _data_url(image_path)}},
            ],
        }]
        out: Dict[str, Any] = {}

        def _call():
            out.update(_call_vlm_json(client, model, messages) or {})

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(VLM_BUDGET_S)
        if t.is_alive():
            return {"status": "timeout",
                    "note": f"clue extraction exceeded {VLM_BUDGET_S}s budget"}
        if not out:
            return {"status": "failed", "note": "VLM returned no parseable JSON"}
        return {"status": "success", "clues": _norm_clues(out)}

    # ------------------------------------------------------------------ #
    # Stage B — property-record research (feature-rich web queries)       #
    # ------------------------------------------------------------------ #
    def build_queries(self, clues: Dict[str, Any],
                      region_hint: Optional[str]) -> List[str]:
        """Feature-rich query templates (town + features + listing anchors).
        The documented production pattern: single-dimension queries return
        nothing; the combination surfaces the listing."""
        if not region_hint:
            return []
        feats = [f for f in (clues.get("high_leverage_features") or []) if f]
        nums = [str(n) for n in (clues.get("visible_numbers") or []) if str(n).strip()]
        style = clues.get("property_style") or ""
        setting = clues.get("setting") or ""
        queries: List[str] = []
        anchors = "Zillow Redfin Realtor property records"
        if feats:
            queries.append(f"{region_hint} {feats[0]} home {anchors}")
        if nums:
            queries.append(f'{region_hint} "{nums[0]}"')
        if len(feats) > 1:
            queries.append(f"{region_hint} {feats[0]} {feats[1]} {anchors}")
        if style or setting:
            queries.append(f"{region_hint} {style or ''} {setting or ''} "
                           f"residential {anchors}".replace("  ", " "))
        # dedupe, cap
        seen, out = set(), []
        for q in queries:
            q = " ".join(q.split())[:200]
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[:MAX_QUERIES]

    def research_direct_addresses(self, clues: Dict[str, Any],
                                  region_hint: Optional[str]) -> Dict[str, Any]:
        """OSM house-precision research path — NO search engine involved.

        When the image yields a readable street name AND house number, query
        Nominatim directly: '{number} {street}, {region}'. OSM in the US often
        carries addr:housenumber geometry at parcel precision (verified live:
        '151 Stratford Place, Madison Heights VA' -> 37.45643,-79.09972).
        This bypasses the search-engine bot-fences entirely and is the
        primary research route; web search is the fallback for feature-only
        clues (no readable address).
        """
        streets = [s for s in (clues.get("street_names") or []) if s][:2]
        numbers = []
        for n in (clues.get("visible_numbers") or []):
            digits = re.sub(r"\D", "", str(n))
            if digits:
                numbers.append(digits)
        numbers = numbers[:2]
        if not streets or not numbers or not region_hint:
            return {"status": "skipped",
                    "note": "needs street_names + visible_numbers + region",
                    "findings": []}
        try:
            from modules.listing_finder import geocode_address
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120], "findings": []}

        def _geo(q: str) -> Optional[Dict[str, Any]]:
            try:
                return geocode_address(q)
            except Exception:
                return None

        findings: List[Dict[str, Any]] = []
        street_found = False
        for st in streets:
            # street-level sanity check first: does this street exist in region?
            sg = _geo(f"{st}, {region_hint}")
            if sg is None:
                continue
            street_found = True
            findings.append({
                "address": st,
                "via": "osm_street_geocode",
                "latitude": sg.get("latitude"),
                "longitude": sg.get("longitude"),
                "display_name": sg.get("display_name", ""),
                "url": "", "title": f"street confirmed in OSM: {st}", "query": "",
            })
            for num in numbers:
                g = _geo(f"{num} {st}, {region_hint}")
                if g and g.get("latitude") is not None:
                    findings.append({
                        "address": f"{num} {st}",
                        "via": "osm_house_geocode",
                        "latitude": g["latitude"],
                        "longitude": g["longitude"],
                        "display_name": g.get("display_name", ""),
                        "url": "", "title": "OSM house-precision parcel",
                        "query": f"{num} {st}, {region_hint}",
                    })
        if findings:
            house_hits = [f for f in findings if f["via"] == "osm_house_geocode"]
            status = "success" if house_hits else "street_only"
            note = (f"{len(house_hits)} house-precision OSM hit(s); "
                    f"street confirmed in OSM" if house_hits
                    else "street confirmed in OSM but no house-number parcel")
        elif street_found:
            status, note = "empty", "street lookup failed in region"
        else:
            status, note = "empty", f"no OSM match for {streets} in {region_hint}"
        return {"status": status, "findings": findings, "note": note}

    def research_property_records(self, clues: Dict[str, Any],
                                  region_hint: Optional[str]) -> Dict[str, Any]:
        """Run the queries through the key-free live web signal, then harvest
        candidate addresses from listing URLs / result text."""
        queries = self.build_queries(clues, region_hint)
        if not queries:
            return {"status": "skipped",
                    "note": "no region context or clues to research",
                    "queries": [], "findings": []}
        try:
            from modules.live_signal_orchestrator import LiveSignalOrchestrator
            orch = LiveSignalOrchestrator(timeout=self.timeout)
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120],
                    "queries": queries, "findings": []}

        findings: List[Dict[str, Any]] = []
        for q in queries:
            try:
                res = orch.signal_web_search(q, n=6)
            except Exception as e:
                res = {"status": "failed", "note": str(e)[:120], "results": []}
            for r in (res.get("results") or []):
                url = r.get("url") or r.get("link") or ""
                text = " ".join(str(r.get(k, "")) for k in ("title", "snippet"))
                for addr in _harvest_addresses(url, text):
                    findings.append({
                        "address": addr,
                        "url": url,
                        "title": str(r.get("title", ""))[:160],
                        "query": q,
                    })
        # dedupe by canonical address, keep first (highest-ranked) hit
        dedup: Dict[str, Dict[str, Any]] = {}
        for f in findings:
            key = _canon_street(" ".join(f["address"].split()[1:]))
            if key and key not in dedup:
                dedup[key] = f
        status = "success" if dedup else ("empty" if queries else "skipped")
        return {"status": status, "queries": queries,
                "findings": list(dedup.values())[:8],
                "note": f"{len(dedup)} unique candidate addresses from web research"}

    # ------------------------------------------------------------------ #
    # Stage C — US street-numbering + adjacency deduction                 #
    # ------------------------------------------------------------------ #
    def apply_numbering_deduction(self, clues: Dict[str, Any],
                                  findings: List[Dict[str, Any]]
                                  ) -> List[Dict[str, Any]]:
        """Turn researched NEIGHBOR addresses into deduced TARGET addresses.

        Rules (US convention, the ones Gemini used):
          R1 number-match: clue number M, found neighbor (N, Street) with
             parity(M)==parity(N) and |M-N| <= 8 -> address (M, Street) is the
             target: consecutive odd (or even) numbers share one street side.
          R2 adjacent-step: found (N, Street) alone -> N-2 and N+2 are the
             same-side adjacent lots (weaker; only when no clue number).
        """
        clue_nums = []
        for n in (clues.get("visible_numbers") or []):
            try:
                v = int(re.sub(r"\D", "", str(n)))
                if 1 <= v <= 999999:
                    clue_nums.append(v)
            except Exception:
                continue
        deduced: List[Dict[str, Any]] = []
        for f in findings:
            toks = f["address"].split()
            if len(toks) < 2 or not toks[0].isdigit():
                continue
            n_found, street = int(toks[0]), " ".join(toks[1:])
            match = None
            for m in clue_nums:
                if m % 2 == n_found % 2 and abs(m - n_found) <= PARITY_MATCH_WINDOW:
                    match = m
                    break
            if match is not None and match != n_found:
                deduced.append({
                    "address": f"{match} {street}",
                    "rule": "R1_number_match",
                    "reason": (f"image shows number {match}; researched neighbor at "
                               f"{n_found} {street} — US numbering places consecutive "
                               f"{'odd' if match % 2 else 'even'} numbers on the same "
                               f"street side, so {match} {street} is the adjacent parcel"),
                    "basis": f["address"],
                })
            elif clue_nums and match is not None and match == n_found:
                # the found address IS the clue number — it's the target itself
                deduced.append({
                    "address": f["address"], "rule": "R0_direct",
                    "reason": (f"listing/records hit {n_found} {street} matches the "
                               f"number visible in the image"),
                    "basis": f["address"],
                })
            elif not clue_nums:
                for step in (-PARITY_STEP_WINDOW, PARITY_STEP_WINDOW):
                    deduced.append({
                        "address": f"{n_found + step} {street}",
                        "rule": "R2_adjacent_step",
                        "reason": (f"records confirm {n_found} {street}; the "
                                   f"same-side adjacent lots are "
                                   f"{n_found + step} {street}"),
                        "basis": f["address"],
                    })
        return deduced

    # ------------------------------------------------------------------ #
    # Stage D — geocode verification of found + deduced addresses         #
    # ------------------------------------------------------------------ #
    def verify_addresses(self, findings: List[Dict[str, Any]],
                         deduced: List[Dict[str, Any]],
                         clue_nums: Optional[List[int]] = None
                         ) -> Dict[str, Any]:
        """Nominatim-geocode the strongest candidates. Only geocoded addresses
        become ensemble candidates (source='investigative')."""
        clue_nums = clue_nums or []
        try:
            from modules.listing_finder import geocode_address
        except Exception as e:
            return {"status": "failed", "note": str(e)[:120], "verified": []}

        # rank: R0/R1 deductions first, then found, then R2 — cap geocodes
        ranked: List[Tuple[str, Dict[str, Any], float]] = []
        for d in deduced:
            conf = (CONF_DEDUCED_NUMBER_MATCH
                    if any(str(cn) and f" {cn} " in f" {d['address']} "
                           for cn in clue_nums) or d["rule"] == "R0_direct"
                    else CONF_DEDUCED_PARITY_ONLY)
            ranked.append((d["address"], {"kind": "deduced", **d}, conf))
        for f in findings:
            ranked.append((f["address"], {"kind": "found", **f}, CONF_FOUND_GEOCODED))

        verified, tried = [], set()
        for addr, meta, conf in ranked:
            if len(verified) >= MAX_GEOCODES or addr.lower() in tried:
                continue
            tried.add(addr.lower())
            # Pre-geocoded (OSM direct path) — reuse, don't re-query Nominatim
            if meta.get("latitude") is not None:
                verified.append({
                    "address": addr,
                    "latitude": meta["latitude"],
                    "longitude": meta["longitude"],
                    "display_name": meta.get("display_name", ""),
                    "confidence": min(CONF_CAP, conf),
                    "kind": meta.get("kind"),
                    "rule": meta.get("rule") or meta.get("via"),
                    "reason": meta.get("reason", ""),
                })
                continue
            geo = None
            try:
                geo = geocode_address(addr)
            except Exception as e:
                logger.warning("geocode failed for %s: %s", addr, e)
            if geo and geo.get("latitude") is not None:
                verified.append({
                    "address": addr,
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "display_name": geo.get("display_name", ""),
                    "confidence": min(CONF_CAP, conf),
                    "kind": meta.get("kind"),
                    "rule": meta.get("rule"),
                    "reason": meta.get("reason", ""),
                })
        return {"status": "success" if verified else "empty",
                "verified": verified,
                "note": (f"{len(verified)}/{len(tried)} addresses geocoded "
                         f"(Nominatim)")}

    # ------------------------------------------------------------------ #
    # Orchestrator — the full loop                                        #
    # ------------------------------------------------------------------ #
    def investigate(self, image_path: Optional[str] = None,
                    clues: Optional[Dict[str, Any]] = None,
                    region_hint: Optional[str] = None,
                    coarse_city: Optional[str] = None) -> Dict[str, Any]:
        """Run the full investigative loop. `clues` (from the caller's own
        vision) or the VLM extraction feeds research; `region_hint` or the
        coarse VLM city names the town to research. Returns a record with
        reasoning_chain + ensemble-ready candidates."""
        rec: Dict[str, Any] = {
            "engine": "investigative_reasoner",
            "status": "limited",
            "region_hint": region_hint or coarse_city or None,
            "reasoning_chain": [],
            "candidates": [],
        }
        region = region_hint or coarse_city or None

        # Stage A
        if clues:
            clues = _norm_clues(clues)
            rec["clues"] = {"status": "agent_supplied", "clues": clues}
            rec["reasoning_chain"].append(
                f"CLUES (agent-supplied): numbers={clues.get('visible_numbers')} "
                f"features={clues.get('high_leverage_features')}")
        elif image_path:
            ca = self.extract_high_leverage_clues(image_path)
            rec["clues"] = ca
            if ca.get("status") == "success":
                clues = ca["clues"]
                rec["reasoning_chain"].append(
                    f"CLUES extracted: numbers={clues.get('visible_numbers')} "
                    f"features={clues.get('high_leverage_features')} "
                    f"style={clues.get('property_style')} setting={clues.get('setting')}")
            else:
                rec["reasoning_chain"].append(
                    f"CLUES unavailable: {ca.get('note', ca.get('status'))}")
        if clues is None:
            rec["note"] = ("no clues available (VLM skipped/failed and none "
                           "supplied) — pass clues via MCP to research without VLM")
            return rec

        if region is None and clues.get("region_guess"):
            region = clues["region_guess"]
            rec["region_hint"] = region
        if not region:
            rec["note"] = ("no region context — web property research needs a "
                           "town/city (location_hint or VLM region_guess)")
            return rec
        rec["reasoning_chain"].append(f"REGION: {region}")

        # Stage B1 — OSM house-precision research (deterministic, no search
        # engine, immune to bot-fences). Primary route when the image shows a
        # readable street name + house number.
        research = self.research_direct_addresses(clues, region)
        rec["research_direct"] = research
        if research.get("status") == "success":
            rec["reasoning_chain"].append(
                "RESEARCH[OSM-direct]: " + "; ".join(
                    f"{f['address']} ({f['via']})" for f in research["findings"]))
        else:
            rec["reasoning_chain"].append(
                f"RESEARCH[OSM-direct]: {research.get('note', research.get('status'))}")

        # Stage B2 — web property-record research (feature-rich queries via the
        # live web signal). Fallback when OSM-direct had no house hit; still
        # valuable for feature-only clues (no readable address in image).
        findings = [f for f in research.get("findings", [])
                    if f.get("via") == "osm_house_geocode"]
        if not findings:
            web = self.research_property_records(clues, region)
            rec["research_web"] = web
            if web.get("findings"):
                rec["reasoning_chain"].append(
                    "RESEARCH[web]: property-record/listing hits — " + "; ".join(
                        f["address"] for f in web["findings"][:4]))
            else:
                rec["reasoning_chain"].append(
                    f"RESEARCH[web]: no address-bearing hits "
                    f"({len(web.get('queries', []))} queries)")
            findings = web.get("findings", [])

        if not findings:
            rec["status"] = "limited"
            rec["note"] = ("research found no address-bearing records — honest "
                           "negative (no fabrication)")
            return rec

        # Stage C
        deduced = self.apply_numbering_deduction(clues, findings)
        rec["deductions"] = deduced
        for d in deduced[:3]:
            rec["reasoning_chain"].append(
                f"DEDUCTION[{d['rule']}]: {d['address']} — {d['reason']}")

        # Stage D
        clue_nums = []
        for n in (clues.get("visible_numbers") or []):
            try:
                clue_nums.append(int(re.sub(r"\D", "", str(n))))
            except Exception:
                pass
        verif = self.verify_addresses(findings, deduced, clue_nums)
        rec["verification"] = verif
        for v in verif.get("verified", []):
            rec["candidates"].append({
                "latitude": v["latitude"],
                "longitude": v["longitude"],
                "confidence": v["confidence"],
                "source": "investigative",
                "address": v["address"],
                "city": region,
                "reasoning": v["reason"] or f"records/listing hit: {v['address']}",
            })
            rec["reasoning_chain"].append(
                f"VERIFIED: {v['address']} -> "
                f"({v['latitude']:.5f},{v['longitude']:.5f}) conf={v['confidence']}")
        if rec["candidates"]:
            rec["status"] = "success"
            rec["candidates"].sort(key=lambda c: c["confidence"], reverse=True)
        else:
            rec["note"] = "addresses found but none geocoded — honest negative"
        return rec


# ---------------------------------------------------------------------- #
# helpers                                                                 #
# ---------------------------------------------------------------------- #
def _data_url(image_path: str) -> str:
    from modules.vlm_geo_analyzer import encode_image_data_url
    return encode_image_data_url(image_path)


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
        logger.warning("investigative VLM call failed: %s", e)
        return None


def _norm_clues(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a VLM/agent clue dict into the canonical shape (lists of str)."""
    def _lst(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if str(v).strip() else []

    return {
        "visible_numbers": _lst(raw.get("visible_numbers")),
        "street_names": _lst(raw.get("street_names")),
        "high_leverage_features": _lst(raw.get("high_leverage_features")),
        "property_style": str(raw.get("property_style") or "").strip(),
        "setting": str(raw.get("setting") or "").strip(),
        "business_names": _lst(raw.get("business_names")),
        "region_guess": str(raw.get("region_guess") or "").strip(),
    }


def _harvest_addresses(url: str, text: str) -> List[str]:
    """Pull candidate street addresses from a listing URL and result text."""
    out: List[str] = []
    for pat in _LISTING_URL_PATTERNS:
        m = pat.search(url or "")
        if m:
            slug = m.group(1).replace("_", " ").replace("-", " ")
            slug = re.sub(r"\b(zpid|M\d+|VA|NY|CA|TX|FL)\b.*$", "",
                          slug, flags=re.I).strip()
            m2 = _TEXT_ADDRESS_RE.search(slug.title())
            if m2:
                out.append(f"{m2.group(1)} {m2.group(2)}")
                break
    if not out:
        m3 = _TEXT_ADDRESS_RE.search(text or "")
        if m3:
            out.append(f"{m3.group(1)} {m3.group(2)}")
    return out

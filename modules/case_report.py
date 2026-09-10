"""
GeoVision Case Report — professional investigation reports.

Renders a full investigation case record (from GeoVisionHarness.investigate, or
any dict with the documented keys) into a self-contained, honest, shareable
report: Markdown (.md) and optionally standalone HTML (.html).

What "good at reports/cases" means here — the report:
  * is honest: confidence comes as a BAND (not a fake single number) with an
    explicit "uncalibrated" flag when no calibrated signal exists; no fabricated
    matches or businesses.
  * is a chain of EVIDENCE, not a claim: every prediction is backed by the
    reasoning_chain, constraints that survived elimination, ranked candidates,
    and the reference images a reviewer can re-open.
  * is reproducible: includes queried_at, model/sources used, and the input image.
  * works for ANY model/CV input: it renders whatever the harness returned; the
    harness is model-agnostic, and this renderer is input-agnostic.

Example:
    from modules.geo_harness import GeoVisionHarness
    rec = GeoVisionHarness().investigate("photo.jpg")
    from modules.case_report import render_case
    render_case(rec, "reports/OP-1234.md")          # markdown
    render_case(rec, "reports/OP-1234.html")        # standalone HTML
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Confidence is inherently fuzzy for a single-image estimate; we report a band
# plus an honesty flag rather than implying calibrated precision.
def _confidence_band(c: Any) -> str:
    try:
        x = float(c)
    except (TypeError, ValueError):
        return "unknown"
    if x <= 0:
        return "unknown"
    if x >= 0.85:
        return "high"
    if x >= 0.6:
        return "medium-high"
    if x >= 0.4:
        return "medium"
    if x >= 0.2:
        return "low"
    return "very low"


def _fmt_coord(v: Any) -> str:
    try:
        return f"{float(v):.5f}"
    except (TypeError, ValueError):
        return "n/a"


def _plain(s: Any) -> str:
    return str(s).replace("_", " ").replace("-", " ").title()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- #
# Markdown rendering                                                      #
# --------------------------------------------------------------------- #
def render_markdown(rec: Dict[str, Any], title: str = "GeoVision Case Report") -> str:
    best = rec.get("best_estimate") or {}
    unc = rec.get("uncertainty") or {}
    lines: list[str] = []
    a = lines.append

    a(f"# {title}")
    a("")
    a(f"- **Status**: `{rec.get('status', 'unknown')}`")
    a(f"- **Case ID**: {rec.get('case_id', '—')}")
    a(f"- **Image**: `{rec.get('image', 'n/a')}`")
    a(f"- **Queried**: {rec.get('queried_at', rec.get('timestamp', _now()))}")
    if rec.get("duration_s"):
        a(f"- **Duration**: {rec.get('duration_s')}s")
    a("")

    a("## Summary")
    a("")
    if best:
        a(f"**Best estimate**: `{_fmt_coord(best.get('latitude'))}, "
          f"{_fmt_coord(best.get('longitude'))}` "
          f"(confidence: **{_confidence_band(best.get('confidence'))}**"
          + ("" if best.get("confidence_calibrated") else " · *uncalibrated*")
          + ")")
        for k in ("city", "country", "landmark", "place_name", "type"):
            if best.get(k):
                a(f"- **{_plain(k)}**: {best[k]}")
    else:
        a("_No confident estimate was produced._ The signals did not converge.")
    if unc:
        a(f"- **Uncertainty**: {unc.get('granularity', 'unknown')} granularity"
          + (f" · approx. {unc.get('radius_km')} km" if unc.get("radius_km") else "")
          + (f" · {unc.get('note', '')}" if unc.get("note") else ""))
    a("")

    a("## Reasoning chain (evidence)")
    a("")
    chain = rec.get("reasoning_chain") or []
    if chain:
        for i, step in enumerate(chain, 1):
            a(f"{i}. {step}")
    else:
        a("_No reasoning chain recorded (deterministic-only path)._")
    a("")

    cons = rec.get("constraints_applied") or rec.get("constraints") or []
    if cons:
        a("## Constraints that survived elimination")
        a("")
        for c in cons if isinstance(cons, list) else cons.values():
            if isinstance(c, dict):
                a(f"- **{_plain(c.get('name', 'constraint'))}**: "
                  f"{c.get('value', '')} — {c.get('note', '')}")
            else:
                a(f"- {c}")
        a("")

    cands = rec.get("candidates") or []
    if cands:
        a("## Ranked candidate locations")
        a("")
        a("| # | lat | lon | confidence | notes |")
        a("|---|-----|-----|------------|-------|")
        for i, c in enumerate(cands[:12], 1):
            lat = _fmt_coord(c.get("latitude"))
            lon = _fmt_coord(c.get("longitude"))
            conf = _confidence_band(c.get("confidence"))
            notes = "; ".join(str(c.get(k)) for k in ("city", "country", "place_name")
                              if c.get(k))
            a(f"| {i} | {lat} | {lon} | {conf} | {notes} |")
        a("")

    # Reference images the reviewer can re-open
    refs = rec.get("reference_urls_by_candidate") or {}
    if refs:
        a("## Reference imagery (ground truth to re-check)")
        a("")
        for cand_label, items in refs.items():
            if not items:
                continue
            a(f"**{cand_label}**")
            for it in (items if isinstance(items, list) else [items]):
                url = it if isinstance(it, str) else it.get("url", it.get("thumbnail_url", ""))
                cap = "" if isinstance(it, str) else it.get("name", "") or it.get("title", "")
                a(f"- [{cap or url}]({url})")
            a("")

    stages = rec.get("stages") or {}
    if stages:
        a("## Stage details")
        a("")
        for name, s in stages.items():
            if not s or not isinstance(s, dict):
                continue
            st = s.get("status", s.get("state", ""))
            a(f"### {_plain(name)} ({st})")
            for key in ("prediction", "summary", "note", "best_estimate"):
                if s.get(key) and isinstance(s[key], str):
                    a(f"- {s[key]}")
            ver = s.get("verifications")
            if ver and isinstance(ver, list):
                vcount = sum(1 for v in ver if v.get("visual_match") or v.get("vlm_verified"))
                a(f"- verifications: {vcount}/{len(ver)} matched")
            a("")

    a("## Sources & methodology")
    a("")
    a("- **Sources**: deterministic GeoCLIP + CLIP reference DB + heuristics + "
      "constraints + offline visual verify; optional VLM.")
    a("- **Data**: real geotagged references (Wikimedia/Mapillary/Flickr) + free "
      "OSINT (OpenStreetMap/Overpass, GeoNames, Nominatim).")
    a("- **Confidence** is reported as a band and flagged *uncalibrated* unless a "
      "calibrated signal exists — it is an honest prior, not a measured probability.")
    if rec.get("agent_hint"):
        a(f"- **Next step**: {rec['agent_hint']}")
    a("")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# HTML rendering (standalone, no external assets)                        #
# --------------------------------------------------------------------- #
def render_html(rec: Dict[str, Any], title: str = "GeoVision Case Report") -> str:
    md = render_markdown(rec, title)
    # Minimal, dependency-free markdown->html for the headings/lists we emit.
    esc = html.escape
    body_lines = []
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("### "):
            body_lines.append(f"<h3>{esc(s[4:])}</h3>")
        elif s.startswith("## "):
            body_lines.append(f"<h2>{esc(s[3:])}</h2>")
        elif s.startswith("# "):
            body_lines.append(f"<h1>{esc(s[2:])}</h1>")
        elif s.startswith("- "):
            body_lines.append(f"<li>{esc(s[2:])}</li>")
        elif s.startswith("| ") and "---" not in s:
            body_lines.append(f"<div class='row'>{esc(s)}</div>")
        elif s == "---":
            continue
        elif not s.strip():
            body_lines.append("")
        else:
            body_lines.append(f"<p>{esc(s)}</p>")
    body = "\n".join(body_lines)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       max-width:820px;margin:24px auto;padding:0 20px;color:#1c1e21;
       line-height:1.55;background:#fff}}
  h1{{border-bottom:3px solid #2f6f4f;padding-bottom:8px}}
  h2{{color:#2f6f4f;margin-top:28px;border-bottom:1px solid #e0e0e0;padding-bottom:4px}}
  h3{{color:#444;margin-top:18px}}
  li{{margin:3px 0}}
  .row{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#666}}
  code{{background:#f4f4f4;padding:1px 5px;border-radius:4px}}
</style></head><body>
{body}
</body></html>"""


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #
def render_case(rec: Dict[str, Any], output: str,
                title: str = "GeoVision Case Report") -> Path:
    """Render an investigation record to Markdown or standalone HTML based on the
    output file extension (.md or .html). Returns the written Path."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        out.write_text(render_html(rec, title), encoding="utf-8")
    else:
        out.write_text(render_markdown(rec, title), encoding="utf-8")
    return out


if __name__ == "__main__":
    # Self-test: synthesize a representative record and render both formats.
    import tempfile
    demo = {
        "status": "success", "case_id": "OP-777", "image": "demo.jpg",
        "queried_at": _now(), "duration_s": 4.2,
        "best_estimate": {"latitude": 48.8584, "longitude": 2.2945,
                          "confidence": 0.83, "city": "Paris", "country": "FR",
                          "place_name": "Eiffel Tower"},
        "uncertainty": {"granularity": "city", "radius_km": 8, "note": "top-3 spread small"},
        "reasoning_chain": ["COARSE: urban tower, Paris region",
                            "RETRIEVAL: regional CLIP matched Eiffel refs (sim 0.92)",
                            "VISUAL: StreetCLIP cross-view confirmed"],
        "constraints_applied": [{"name": "hemisphere", "value": "north",
                                 "note": "day-light footprint excludes south"}],
        "candidates": [{"latitude": 48.8584, "longitude": 2.2945, "confidence": 0.83,
                        "city": "Paris"}, {"latitude": 48.86, "longitude": 2.30,
                        "confidence": 0.5, "city": "Paris"}],
        "reference_urls_by_candidate": {"Paris": [{"url": "https://example.com/eiffel.jpg",
                                                    "name": "Eiffel by night"}]},
        "stages": {"regional_retrieval": {"status": "success",
                                           "summary": "restricted to France refs"},
                   "visual_verify_candidates": {"status": "success",
                                                 "verifications": [{"visual_match": True}]}},
        "agent_hint": "Cross-check the reference photos with your own vision.",
    }
    with tempfile.TemporaryDirectory() as d:
        p = render_case(demo, f"{d}/case.md")
        h = render_case(demo, f"{d}/case.html")
        print("WROTE", p, p.stat().st_size, "bytes")
        print("WROTE", h, h.stat().st_size, "bytes")
        print("--- md preview ---")
        print(open(p).read()[:700])
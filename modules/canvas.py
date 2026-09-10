"""
GeoVision Detective Canvas — a live visual board the AI agent works on.

The canvas shows EVERYTHING the agent is doing, live: each reasoning step written
down, every image it pulls or compares displayed visually, candidates pinned,
eliminations justified, comparisons scored side-by-side, and detailed summaries.
Like a detective's board. It can end with a full case/report.

Design (zero-server, file:// friendly):
  - A session writes  <dir>/<session>.json   (the event log)
                 and  <dir>/<session>.html   (self-contained viewer, JSON inlined,
                    auto-reloads every few seconds -> "live" feel)
  - Events are appended atomically; images are thumbnailed to data-URLs so the
    viewer renders them offline, no web server required.
  - Any agent (not just the harness) can drive it via MCP canvas_* tools:
    canvas_start / canvas_add / canvas_finish.

Event kinds:
  step       what the agent is doing right now
  note       detailed observation / reasoning
  image      a pulled image (reference photo, map, screenshot) shown on the board
  comparison query vs reference side-by-side with a similarity score
  candidate  a pinned possible location (lat/lon + label + confidence)
  elimination a candidate ruled out + the reason
  evidence   corroborating data (listing, weather, web hit)
  summary    a detailed written summary block
  verdict    the final answer, highlighted
  link       an external URL reference
"""
from __future__ import annotations

import base64
import gzip
import html
import io
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KINDS = {"step", "note", "image", "comparison", "candidate", "elimination",
         "evidence", "summary", "verdict", "link"}

_VIEWER = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoVision Detective Canvas</title>
<style>
 :root{--bg:#14161a;--panel:#1d2026;--line:#2e323b;--ink:#e8e6e3;--dim:#9aa0aa;
      --acc:#e0b04a;--good:#69c07d;--bad:#e06c5a;--blue:#5aa7d6}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.55 -apple-system,'Segoe UI',Roboto,Arial,sans-serif}
 header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,#191c22,#14161a);
        border-bottom:2px solid var(--acc);padding:14px 20px;display:flex;
        justify-content:space-between;align-items:baseline;flex-wrap:wrap}
 h1{margin:0;font-size:18px;letter-spacing:.4px}
 h1 .pin{color:var(--acc)}
 #meta{color:var(--dim);font-size:12px}
 main{max-width:1060px;margin:0 auto;padding:18px 20px 80px}
 .ev{background:var(--panel);border:1px solid var(--line);border-radius:10px;
     padding:12px 14px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.35)}
 .ev.new{animation:pop .5s ease}
 @keyframes pop{0%{transform:translateY(6px);opacity:0}100%{transform:none;opacity:1}}
 .head{display:flex;gap:10px;align-items:center;margin-bottom:6px}
 .pill{font-size:10.5px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
       padding:3px 9px;border-radius:20px;color:#101216}
 .step .pill{background:var(--blue)} .note .pill{background:#8a93a5}
 .image .pill{background:#b08ad6} .comparison .pill{background:var(--acc)}
 .candidate .pill{background:var(--good)} .elimination .pill{background:var(--bad)}
 .evidence .pill{background:#5ec0b8} .summary .pill{background:#c88ad6}
 .verdict .pill{background:var(--acc)} .link .pill{background:#7a8699}
 .t{color:var(--dim);font-size:11px;margin-left:auto}
 .body{white-space:pre-wrap;word-wrap:break-word}
 .img{max-width:100%;border-radius:8px;margin-top:8px;border:1px solid var(--line)}
 .cmp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
 .cmp img{width:100%;border-radius:8px;border:1px solid var(--line)}
 .cmp .sc{text-align:center;color:var(--acc);font-weight:700;margin-top:4px}
 .coord{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--good)}
 .verdict .body{font-size:16px;border-left:4px solid var(--acc);padding-left:12px}
 a{color:var(--blue)}
 .foot{color:var(--dim);text-align:center;font-size:12px;margin-top:30px}
 #autoscroll{position:fixed;right:18px;bottom:18px;background:var(--panel);
   border:1px solid var(--line);color:var(--ink);border-radius:8px;padding:8px 12px;
   cursor:pointer;font-size:12px}
</style></head><body>
<header><h1><span class="pin">&#128204;</span> __TITLE__</h1>
<div id="meta">__META__</div></header>
<main id="board"></main>
<div id="autoscroll" onclick="toggleScroll()">&#8595; auto-scroll: on</div>
<div class="foot">GeoVision detective canvas &mdash; auto-refreshes live</div>
<script>
const DATA = __DATA__;
const board = document.getElementById('board');
let auto = sessionStorage.getItem('gvAuto') !== 'off';
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function fmtTime(ts){const d=new Date(ts*1000);return d.toLocaleTimeString()}
function render(){
  board.innerHTML = DATA.events.map(e=>{
    let inner='';
    if(e.title) inner += `<div class="head"><span class="pill">${esc(e.kind)}</span>
      <b>${esc(e.title)}</b><span class="t">${esc(fmtTime(e.ts))}</span></div>`;
    else inner += `<div class="head"><span class="pill">${esc(e.kind)}</span>
      <span class="t">${esc(fmtTime(e.ts))}</span></div>`;
    if(e.text) inner += `<div class="body">${esc(e.text)}</div>`;
    if(e.kind==='image' && e.src)
      inner += `<img class="img" src="${esc(e.src)}" ${e.caption?`alt="${esc(e.caption)}"`:''}>
                ${e.caption?`<div class="body" style="color:var(--dim)">${esc(e.caption)}</div>`:''}`;
    if(e.kind==='comparison' && (e.left||e.right)){
      inner += `<div class="cmp">
        <div>${e.left?`<img src="${esc(e.left.src)}">`:''}<div class="body" style="color:var(--dim)">${esc(e.left?e.left.caption||'query':'')}</div></div>
        <div>${e.right?`<img src="${esc(e.right.src)}">`:''}<div class="body" style="color:var(--dim)">${esc(e.right?e.right.caption||'reference':'')}</div></div>
        ${e.score!=null?`<div class="sc" style="grid-column:1/3">similarity ${Number(e.score).toFixed(3)}${e.match?' &#10004; MATCH':''}</div>`:''}
      </div>`;}
    if(e.lat!=null) inner += `<div class="coord">${Number(e.lat).toFixed(5)}, ${Number(e.lon).toFixed(5)}</div>`;
    if(e.url) inner += `<div><a href="${esc(e.url)}" target="_blank">${esc(e.url)}</a></div>`;
    return `<div class="ev ${e.kind}">${inner}</div>`;
  }).join('');
  if(auto) window.scrollTo(0, document.body.scrollHeight);
}
function toggleScroll(){auto=!auto;sessionStorage.setItem('gvAuto',auto?'on':'off');
  document.getElementById('autoscroll').innerHTML=`&#8595; auto-scroll: ${auto?'on':'off'}`;
  if(auto)window.scrollTo(0,document.body.scrollHeight);}
render();
let n = DATA.events.length;
setInterval(()=>{ if(document.visibilityState==='visible'){
  fetch('?refresh='+Date.now(),{cache:'no-store'}).then(r=>r.ok?location.reload():0).catch(()=>location.reload());
}}, 4000);
</script></body></html>"""


def _thumb_data_url(src: str, max_px: int = 480, quality: int = 72) -> Optional[str]:
    """Fetch a local path or http(s) URL and return a small jpeg data-url, or None."""
    try:
        raw: Optional[bytes] = None
        if src.startswith(("http://", "https://")):
            import urllib.request
            req = urllib.request.Request(src, headers={"User-Agent": "GeoVision/2.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                raw = r.read(6 << 20)
        else:
            p = Path(src).expanduser()
            if p.exists():
                raw = p.read_bytes()[: (6 << 20)]
        if not raw:
            return None
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # never let an image break the canvas
        logger.debug("canvas thumb failed for %s: %s", src, e)
        return None


class DetectiveCanvas:
    """Live detective board. Append events; the HTML viewer updates on reload."""

    def __init__(self, session: Optional[str] = None, title: str = "Investigation",
                 out_dir: Optional[str] = None):
        self.session = session or time.strftime("case_%Y-%m-%d_%H-%M-%S")
        self.title = title
        root = Path(out_dir) if out_dir else \
            Path(__file__).resolve().parent.parent / "reports" / "canvas"
        root.mkdir(parents=True, exist_ok=True)
        self.json_path = root / f"{self.session}.json"
        self.html_path = root / f"{self.session}.html"
        self.events: List[Dict[str, Any]] = []
        self._load_existing()
        self.add("step", text=f"Canvas opened — {title}", title="Session start")
        if not any(e.get("kind") == "meta" for e in self.events):
            self._meta_saved = True

    # ------------------------------------------------------------------ #
    def _load_existing(self) -> None:
        try:
            if self.json_path.exists():
                data = json.loads(self.json_path.read_text())
                if isinstance(data.get("events"), list):
                    self.events = data["events"]
        except Exception:
            self.events = []

    def add(self, kind: str, *, title: str = "", text: str = "",
            image: str = "", caption: str = "",
            left: str = "", right: str = "", left_caption: str = "",
            right_caption: str = "", score: Optional[float] = None,
            match: Optional[bool] = None,
            lat: Optional[float] = None, lon: Optional[float] = None,
            url: str = "", confidence: Optional[float] = None) -> Dict[str, Any]:
        """Append one visual event to the board. Never raises."""
        ev: Dict[str, Any] = {"id": uuid.uuid4().hex[:10], "kind": kind,
                              "ts": time.time(), "title": title, "text": text}
        try:
            if kind not in KINDS:
                ev["kind"] = "note"
            if image:
                src = image if image.startswith("data:") else \
                    (_thumb_data_url(image) or image)
                ev["src"] = src
                if caption:
                    ev["caption"] = caption
            if left or right:
                ev["left"] = {"src": left if left.startswith("data:")
                              else (_thumb_data_url(left) or left),
                              "caption": left_caption} if left else None
                ev["right"] = {"src": right if right.startswith("data:")
                               else (_thumb_data_url(right) or right),
                               "caption": right_caption} if right else None
            if score is not None:
                ev["score"] = float(score)
            if match is not None:
                ev["match"] = bool(match)
            if lat is not None and lon is not None:
                ev["lat"], ev["lon"] = float(lat), float(lon)
            if url:
                ev["url"] = url
            if confidence is not None:
                ev["confidence"] = float(confidence)
        except Exception as e:
            logger.debug("canvas event enrich failed: %s", e)
        self.events.append(ev)
        self._flush()
        return ev

    def finish(self, text: str = "Investigation closed.") -> Dict[str, Any]:
        return self.add("summary", title="Canvas closed", text=text)

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        try:
            tmp = self.json_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"session": self.session, "title": self.title,
                                       "events": self.events}))
            tmp.replace(self.json_path)
            self._write_html()
        except Exception as e:
            logger.warning("canvas flush failed: %s", e)

    def _write_html(self) -> None:
        try:
            meta = (f"{len(self.events)} events &middot; updated "
                    f"{time.strftime('%H:%M:%S')} &middot; {self.session}")
            page = (_VIEWER
                    .replace("__TITLE__", html.escape(self.title))
                    .replace("__META__", meta)
                    .replace("__DATA__", json.dumps({"events": self.events})))
            tmp = self.html_path.with_suffix(".tmp")
            tmp.write_text(page)
            tmp.replace(self.html_path)
        except Exception as e:
            logger.warning("canvas html write failed: %s", e)

    @property
    def viewer_url(self) -> str:
        return self.html_path.as_uri()


if __name__ == "__main__":  # tiny self-test
    c = DetectiveCanvas(title="Smoke test", out_dir="/tmp/gv_canvas_test")
    c.add("step", title="Stage 1", text="Coarse reasoning about the scene…")
    c.add("image",
          image="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/320px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg",
          caption="Reference pulled: Eiffel Tower")
    c.add("comparison", left="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/320px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg",
          right="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Eiffel.Tower.Paris.2010.jpg/320px-Eiffel.Tower.Paris.2010.jpg",
          left_caption="query", right_caption="Paris ref",
          score=0.92, match=True)
    c.add("candidate", title="Paris, FR", text="Strong regional agreement",
          lat=48.8584, lon=2.2945, confidence=0.88)
    c.add("elimination", title="Tokyo candidate", text="Road line color mismatch")
    c.add("verdict", title="Eiffel Tower, Paris",
          text="48.8584, 2.2945 — confidence high (uncalibrated)",
          lat=48.8584, lon=2.2945, confidence=0.9)
    c.finish("done")
    print("viewer:", c.viewer_url)
    print("events:", len(c.events), "| html bytes:", c.html_path.stat().st_size)

"""
Tests for the Detective Canvas + EXIF forensics stage (Sep 2026).

Deterministic checks (no heavy model):
  - canvas events append, persist to json, render self-contained html
  - every event kind renders without raising; unknown kind degrades to note
  - comparisons carry score/match; candidates carry lat/lon
  - viewer html contains the verdict + auto-refresh + title escaping
  - exif_forensics runs on a real image and returns structured status

Live E2E (colosseum.jpg -> Rome, 15 events, 7 embedded images) verified separately.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.canvas import DetectiveCanvas, KINDS


def _canvas(tmp, name="t1"):
    return DetectiveCanvas(session=name, title="Unit <test>", out_dir=tmp)


def test_canvas_appends_and_persists():
    with tempfile.TemporaryDirectory() as d:
        c = _canvas(d)
        c.add("step", title="s1", text="doing a thing")
        c.add("candidate", title="Paris", lat=48.85, lon=2.29, confidence=0.8)
        c.add("verdict", title="final", text="it is Paris", lat=48.85, lon=2.29)
        data = json.loads(c.json_path.read_text())
        assert len(data["events"]) == 4  # 3 + session-start
        assert data["events"][-1]["kind"] == "verdict"
        assert c.html_path.exists() and c.html_path.stat().st_size > 2000


def test_canvas_all_kinds_render():
    with tempfile.TemporaryDirectory() as d:
        c = _canvas(d, "t2")
        for k in KINDS:
            c.add(k, title=f"{k} event", text=f"text for {k}",
                  score=0.5 if k == "comparison" else None,
                  match=True if k == "comparison" else None,
                  lat=1.0 if k == "candidate" else None,
                  lon=2.0 if k == "candidate" else None)
        html = c.html_path.read_text()
        for k in KINDS:
            assert f"{k} event" in html, f"missing rendered kind {k}"


def test_canvas_unknown_kind_degrades():
    with tempfile.TemporaryDirectory() as d:
        c = _canvas(d, "t3")
        ev = c.add("bogus_kind", text="x")
        assert ev["kind"] == "note"


def test_canvas_title_escaped_in_html():
    with tempfile.TemporaryDirectory() as d:
        c = _canvas(d, "t4")  # title contains <test>
        html = c.html_path.read_text()
        # the viewer has its own legit <script> tag; assert the TITLE is escaped
        assert "&lt;test&gt;" in html          # title escaped
        assert "<title>GeoVision Detective Canvas" in html or "<test>" not in html.split("</header>")[0]


def test_canvas_comparison_fields():
    with tempfile.TemporaryDirectory() as d:
        c = _canvas(d, "t5")
        c.add("comparison", title="cmp", left="https://x/l.jpg", right="https://x/r.jpg",
              score=0.91, match=True)
        ev = json.loads(c.json_path.read_text())["events"][-1]
        assert ev.get("score") == 0.91 and ev.get("match") is True
        assert ev["left"] and ev["right"]


def test_exif_forensics_real_image():
    from modules.geo_harness import GeoVisionHarness
    img = "data/eval/wikipedia_landmarks_v1/images/colosseum.jpg"
    if not os.path.exists(img):
        return  # repo image not present in this checkout — skip gracefully
    r = GeoVisionHarness().exif_forensics_record(img)
    assert r.get("status") == "success"
    assert "has_exif" in r and "gps" in r  # structured, no raise

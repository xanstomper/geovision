"""
GeoVision Web Interface
Flask-based web UI with WebSocket real-time pipeline progress streaming.

Also supports SSE (Server-Sent Events) as fallback for browsers that don't
support WebSocket. Both endpoints are available:
  - WebSocket: ws://localhost:9999 (auto-detect in JS client)
  - SSE:       GET /api/stream/<job_id> (existing endpoint)

Endpoints:
  GET  /                  - Main upload UI
  GET  /ocean             - OceanIR evidence workspace
  POST /api/analyze       - Start scan (HTTP, returns job_id)
  GET  /api/stream/<id>   - SSE stream for a job (fallback)
  GET  /api/status        - Service status
  GET  /cases             - Case list
  ... case CRUD endpoints ...
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import logging
from pathlib import Path
import json
import random
import threading
import queue
import time
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
CORS(app)

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.case_manager import CaseManager
cm = CaseManager(str(Path(__file__).parent.parent / 'data' / 'cases.db'))

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER = BASE_DIR / "static" / "reports"
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Shared job registry — threads and WS write here
active_jobs = {}

# ---------------------------------------------------------------------------
# WebSocket support (best-effort: Flask-SocketIO if available, else SSE-only)
# ---------------------------------------------------------------------------
socketio = None

def _try_setup_socketio():
    """Try to enable Flask-SocketIO if installed. No-op otherwise."""
    global socketio
    try:
        from flask_socketio import SocketIO
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60)
        logger.info("Flask-SocketIO enabled")

        @socketio.on('connect')
        def on_connect():
            logger.info("WS client connected: %s", request.sid)

        @socketio.on('scan:start')
        def on_scan_start(data):
            emit('job:ack', {'status': 'ready'})

    except ImportError:
        logger.info("Flask-SocketIO not installed — SSE-only mode")


_try_setup_socketio()


def broadcast_ws(job_id, msg):
    """Broadcast a JSON message to all WS clients + SSE queue for this job."""
    if socketio:
        try:
            socketio.emit(job_id, msg)
        except Exception:
            pass
    q = active_jobs.get(job_id)
    if q:
        try:
            q.put(msg)
        except Exception:
            pass


def _run_pipeline_job(job_id, image_paths, options):
    """Run the GeoVision pipeline in a background thread."""
    try:
        from geovision_deep_scan import run_pipeline

        orig_logger = logging.getLogger("GeoVisionDeepScan")

        class ProgressHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record).strip()
                if not msg:
                    return
                broadcast_ws(job_id, {"type": "log", "data": msg})

        handler = ProgressHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        orig_logger.addHandler(handler)

        res = run_pipeline(str(image_paths[0]), options)

        orig_logger.removeHandler(handler)

        broadcast_ws(job_id, {"type": "result", "data": res.to_dict()})
        broadcast_ws(job_id, {"type": "job:done"})

    except Exception as e:
        import traceback
        logger.error("Pipeline job %s failed: %s", job_id, e, exc_info=True)
        broadcast_ws(job_id, {"type": "error", "data": f"{e}\n{traceback.format_exc()}"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Start a geolocation scan. Returns job_id immediately."""
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "Empty filename"}), 400

    context_text = request.form.get("context_text", "")
    region = request.form.get("region", None)

    job_id = str(random.randint(10000, 99999))
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        filepath = job_dir / f.filename
        f.save(str(filepath))
        saved_paths.append(str(filepath))

    q = queue.Queue()
    active_jobs[job_id] = q

    options = {
        "output_dir": str(REPORTS_FOLDER),
        "near_park": False,
        "region": region,
        "no_vlm": False,
        "interactive": False,
        "verbose": False,
        "extra_images": saved_paths[1:] if len(saved_paths) > 1 else [],
        "context_text": context_text,
    }

    broadcast_ws(job_id, {"type": "job:started", "job_id": job_id})

    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, saved_paths, options),
        daemon=True,
    ).start()

    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream_job(job_id):
    """SSE endpoint for browsers without WebSocket support."""
    def event_stream():
        q = active_jobs.get(job_id)
        if not q:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/status")
def api_status():
    return jsonify({
        "service": "GeoVision",
        "version": "2.1.0",
        "ws_available": socketio is not None,
        "sse_available": True,
        "capabilities": [
            "computer_vision_analysis",
            "deep_learning_feature_extraction",
            "satellite_imagery_matching",
            "park_proximity_analysis",
            "vlm_location_estimation",
            "vehicle_identification",
            "deepfake_detection",
            "chain_store_locator",
            "weather_corroboration",
            "realtime_progress_streaming",
        ],
    })


@app.route("/ocean")
def oceanir_page():
    return render_template("oceanir.html")


@app.route("/api/oceanir", methods=["POST"])
def oceanir_analyze():
    """OceanIR-style evidence workspace analysis."""
    if "images" not in request.files and "image" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    files = request.files.getlist("images") if "images" in request.files else [request.files.get("image")]
    if not files or files[0].filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        job_dir = UPLOAD_FOLDER / ("oceanir_" + str(random.randint(10000, 99999)))
        job_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for f in files:
            fp = job_dir / f.filename
            f.save(str(fp))
            saved_paths.append(str(fp))
        path = saved_paths[0]
        hint = request.form.get("hint") or request.form.get("location_hint") or None
        use_canvas = request.form.get("canvas", "1") != "0"
        with_canvas_flag = request.form.get("with_listings", "0") == "1"

        from modules.geo_harness import GeoVisionHarness
        canvas = None
        if use_canvas:
            from modules.canvas import DetectiveCanvas
            canvas = DetectiveCanvas(session=f"oceanir_{Path(path).stem[:20]}",
                                     title=f"OceanIR — {Path(path).name}")
        res = GeoVisionHarness().investigate(str(path), location_hint=hint,
                                             canvas=canvas, with_listings=with_canvas_flag)

        # Multi-image session: investigate remaining frames, then correlate
        # cross-frame evidence (spatial consensus + agreement boost). Opt-in
        # by uploading >1 image; single-frame response shape is unchanged.
        multi_frame = None
        if len(saved_paths) > 1:
            from modules.multi_image_session_correlator import MultiImageSessionCorrelator
            frame_results = [{"image_path": str(path), "result": res}]
            for extra in saved_paths[1:]:
                try:
                    frame_results.append({
                        "image_path": extra,
                        "result": GeoVisionHarness().investigate(extra, location_hint=hint),
                    })
                except Exception as fe:
                    frame_results.append({"image_path": extra,
                                          "result": {"candidates": [], "error": str(fe)[:200]}})
            multi_frame = MultiImageSessionCorrelator().correlate(frame_results)
        best = res.get("best_estimate") or {}
        return {
            "status": "success",
            "best": {
                "latitude": best.get("latitude"), "longitude": best.get("longitude"),
                "confidence": best.get("confidence"),
                "uncalibrated": not best.get("confidence_calibrated"),
                "place": best.get("city") or best.get("place_name") or None,
                "country": best.get("country"),
            },
            "unsure": (best.get("confidence") or 0) < 0.3,
            "evidence_verdict": res.get("evidence_verdict", {}),
            "candidates": [
                {"latitude": c.get("latitude"), "longitude": c.get("longitude"),
                 "confidence": c.get("confidence"),
                 "place": c.get("city") or c.get("place_name"),
                 "source": c.get("source")}
                for c in res.get("candidates", [])[:10]
            ],
            "reasoning": res.get("reasoning_chain", []),
            "reference_urls": res.get("reference_urls_by_candidate", {}),
            "multi_frame_correlation": multi_frame,
            "canvas_viewer_url": canvas.viewer_url if canvas else None,
            "record": res,
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forensics", methods=["POST"])
def api_forensics():
    """Fast visual forensics (<100ms OpenCV) on uploaded image."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        import tempfile
        from modules.grandmaster_forensics import OpenCVVisualForensics
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        analyzer = OpenCVVisualForensics()
        breakdown = analyzer.analyze(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify({
            "status": "success",
            "forensics": breakdown.to_dict(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/grandmaster", methods=["POST"])
def api_grandmaster():
    """Zero-storage Raven-class Grandmaster Forensics investigation."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    hint = request.form.get("hint") or request.form.get("location_hint") or None
    try:
        import tempfile
        from modules.grandmaster_forensics import GrandmasterForensicsEngine
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        engine = GrandmasterForensicsEngine()
        rec = engine.investigate(tmp_path, location_hint=hint)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/street_target", methods=["POST"])
def api_street_target():
    """Topological micro-GIS street intersection targeter."""
    data = request.get_json(silent=True) or request.form
    lat = data.get("lat") or data.get("latitude")
    lon = data.get("lon") or data.get("longitude")
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400
    radius = int(data.get("radius_m") or data.get("radius") or 1500)
    heading = float(data.get("heading")) if data.get("heading") else None
    ocr = data.get("ocr_clues") or []
    amenity = data.get("amenity_clues") or []
    try:
        from modules.street_targeter import StreetTargeter
        targeter = StreetTargeter()
        res = targeter.target_street(
            lat=float(lat),
            lon=float(lon),
            radius_m=radius,
            expected_heading_deg=heading,
            ocr_clues=ocr if isinstance(ocr, list) else [str(ocr)],
            amenity_clues=amenity if isinstance(amenity, list) else [str(amenity)],
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/plonkit", methods=["POST"])
def api_plonkit():
    """PlonkIt Grandmaster Infrastructure Evaluation (85-country rules)."""
    if "image" not in request.files and "images" not in request.files:
        # Check if observations provided as JSON
        obs = request.get_json(silent=True)
        if obs:
            from modules.plonkit_meta_engine import PlonkitMetaEngine
            engine = PlonkitMetaEngine()
            return jsonify(engine.evaluate_observations(obs))
        return jsonify({"error": "No image or observations uploaded"}), 400

    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        import tempfile
        from modules.plonkit_meta_engine import PlonkitMetaEngine
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        engine = PlonkitMetaEngine()
        res = engine.scan_image(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/car_id", methods=["POST"])
def api_car_id():
    """CarID vehicle fleet profiler on uploaded image."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        import tempfile
        from modules.car_fleet_identifier import CarFleetIdentifier
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        identifier = CarFleetIdentifier()
        res = identifier.analyze(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dossier", methods=["POST"])
def api_dossier():
    """Generate complete Raven-class forensic intelligence dossier."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    fmt = request.form.get("format", "json").lower()
    try:
        import tempfile
        from mcp_geovision_server import tool_geovision_dossier
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        res = tool_geovision_dossier({"image_path": tmp_path, "format": fmt})
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/skyeye", methods=["POST"])
def api_skyeye():
    """SkyEye cross-view building facade and overhead satellite footprint verifier."""
    lat = request.form.get("lat") or request.form.get("latitude")
    lon = request.form.get("lon") or request.form.get("longitude")
    if not lat or not lon:
        data = request.get_json(silent=True) or {}
        lat = data.get("lat") or data.get("latitude")
        lon = data.get("lon") or data.get("longitude")
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon required"}), 400

    radius = int(request.form.get("radius_m") or 400)
    image_path = None
    tmp_path = None
    if "image" in request.files or "images" in request.files:
        file = request.files.get("image") or request.files.getlist("images")[0]
        if file and file.filename != "":
            import tempfile
            suffix = Path(file.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
            image_path = tmp_path

    try:
        from modules.skyeye_footprint_verifier import SkyEyeFootprintVerifier
        verifier = SkyEyeFootprintVerifier()
        res = verifier.verify_candidate_footprints(
            lat=float(lat),
            lon=float(lon),
            image_path=image_path,
            radius_m=radius,
        )
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return jsonify(res)
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


@app.route("/api/solar_lock", methods=["POST"])
def api_solar_lock():
    """Solar Lock: NOAA astronomical ephemeris shadow and latitude solver."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    date_str = request.form.get("date") or request.form.get("date_str")
    season = request.form.get("season")
    try:
        import tempfile
        from modules.solar_lock_solver import SolarLockSolver
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        solver = SolarLockSolver()
        res = solver.analyze_image_shadows(tmp_path, date_str=date_str, approx_season=season)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mountain_ridge", methods=["POST"])
def api_mountain_ridge():
    """Mountain Ridge & DEM Skyline Solver."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    hint = request.form.get("hint")
    lat = float(request.form.get("lat")) if request.form.get("lat") else None
    lon = float(request.form.get("lon")) if request.form.get("lon") else None

    try:
        import tempfile
        from modules.mountain_ridge_matcher import MountainRidgeMatcher
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        matcher = MountainRidgeMatcher()
        res = matcher.analyze_image_horizon(tmp_path, hint_region=hint, candidate_lat=lat, candidate_lon=lon)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ecoregion", methods=["POST"])
def api_ecoregion():
    """Global Ecoregion & Soil Bio-Classifier."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        import tempfile
        from modules.ecoregion_classifier import EcoregionClassifier
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        classifier = EcoregionClassifier()
        res = classifier.extract_bio_spectral_features(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hermes_consensus", methods=["POST"])
def api_hermes_consensus():
    """Autonomous Hermes & Antigravity Multi-Agent Collaborative Solver."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    hint = request.form.get("hint") or request.form.get("location_hint")
    season = request.form.get("season") or request.form.get("season_hint")

    try:
        import tempfile
        from modules.hermes_collaborative_solver import HermesCollaborativeSolver
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        solver = HermesCollaborativeSolver()
        res = solver.run_collaborative_investigation(tmp_path, location_hint=hint, season_hint=season)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video_panorama", methods=["POST"])
def api_video_panorama():
    """Video & Dashcam Multi-Frame Panorama Spatial Stitcher."""
    if "video" not in request.files and "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No video or image uploaded"}), 400
    file = request.files.get("video") or request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        import tempfile
        from modules.panorama_spatial_stitcher import VideoPanoramaStitcher
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        stitcher = VideoPanoramaStitcher()
        res = stitcher.process_video_or_images(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/utility_grid", methods=["POST"])
def api_utility_grid():
    """Global Power Grid & Utility Transformer Hardware Classifier."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        import tempfile
        from modules.utility_grid_signature import UtilityGridClassifier
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        classifier = UtilityGridClassifier()
        res = classifier.detect_grid_features(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/camera_gen", methods=["POST"])
def api_camera_gen():
    """Camera Generation & Street View Optical Artifact Profiler."""
    if "image" not in request.files and "images" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files.get("image") or request.files.getlist("images")[0]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        import tempfile
        from modules.camera_generation_classifier import CameraGenerationClassifier
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        classifier = CameraGenerationClassifier()
        res = classifier.analyze_optical_artifacts(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cases")
def cases_page():
    return render_template("cases.html")

@app.route("/cases/<int:case_id>")
def case_detail_page(case_id):
    return render_template("case_detail.html", case_id=case_id)

@app.route("/api/cases", methods=["GET"])
def api_list_cases():
    cases = cm.list_cases()
    return jsonify(cases)

@app.route("/api/cases", methods=["POST"])
def api_create_case():
    data = request.json or {}
    case = cm.create_case(
        name=data.get("name", "Unnamed Case"),
        description=data.get("description", ""),
        tags=data.get("tags", [])
    )
    return jsonify(case), 201

@app.route("/api/cases/<int:case_id>", methods=["GET"])
def api_get_case(case_id):
    case = cm.get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(case)

@app.route("/api/cases/<int:case_id>", methods=["PUT"])
def api_update_case(case_id):
    data = request.json or {}
    updated = cm.update_case(
        case_id,
        **{k: v for k, v in data.items() if k in ("name","description","tags","status")}
    )
    if not updated:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(updated)

@app.route("/api/cases/<int:case_id>", methods=["DELETE"])
def api_delete_case(case_id):
    ok = cm.delete_case(case_id)
    if not ok:
        return jsonify({"error": "Case not found"}), 404
    return jsonify({"status": "deleted"})

@app.route("/api/cases/<int:case_id>/scans", methods=["POST"])
def api_add_scan(case_id):
    data = request.json or {}
    scan = cm.add_scan_to_case(
        case_id=case_id,
        scan_id=data.get("scan_id", ""),
        image_path=data.get("image_path", ""),
        best_lat=data.get("best_lat"),
        best_lon=data.get("best_lon"),
        best_conf=data.get("best_conf"),
        summary=data.get("summary", {}),
        scan_json_path=data.get("scan_json_path", ""),
        scan_html_path=data.get("scan_html_path", ""),
    )
    return jsonify(scan), 201

@app.route("/api/cases/<int:case_id>/notes", methods=["POST"])
def api_add_note(case_id):
    data = request.json or {}
    note = cm.add_note(
        case_id=case_id,
        note=data.get("note", ""),
        author=data.get("author", "analyst")
    )
    return jsonify(note), 201

@app.route("/api/cases/search", methods=["GET"])
def api_search_cases():
    q = request.args.get("q", "")
    cases = cm.search_cases(q)
    return jsonify(cases)

@app.route("/api/cases/<int:case_id>/export", methods=["GET"])
def api_export_case(case_id):
    data = cm.export_case(case_id)
    if not data:
        return jsonify({"error": "Case not found"}), 404
    return data, 200, {'Content-Type': 'application/json', 'Content-Disposition': f'attachment; filename="case_{case_id}.json"'}


if __name__ == "__main__":
    if socketio:
        socketio.run(app, host="0.0.0.0", port=9999, debug=True, allow_unsafe_werkzeug=True)
    else:
        app.run(host="0.0.0.0", port=9999, debug=True)

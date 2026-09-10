"""
GeoVision Web Interface
Flask-based web UI for uploading images and viewing results
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import json
import urllib.request
import base64
import random
import subprocess
import threading
import queue
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.case_manager import CaseManager
cm = CaseManager(str(Path(__file__).parent.parent / 'data' / 'cases.db'))

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static/uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER = BASE_DIR / "static/reports"
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Store active jobs for streaming
active_jobs = {}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    context_text = request.form.get("context_text", "")
    
    job_id = str(random.randint(10000, 99999))
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for f in files:
        filepath = job_dir / f.filename
        f.save(str(filepath))
        saved_paths.append(str(filepath))
        
    # Launch background thread to run pipeline and capture output
    q = queue.Queue()
    active_jobs[job_id] = q
    
    def run_job():
        try:
            main_image = saved_paths[0]
            extra_images = saved_paths[1:] if len(saved_paths) > 1 else []
            
            # Create a small runner script so we can capture stdout/stderr easily in a subprocess
            runner_script = f"""
import sys
import json
sys.path.insert(0, '{BASE_DIR.parent}')
from geovision_deep_scan import run_pipeline

options = {{
    "output_dir": '{str(REPORTS_FOLDER)}',
    "near_park": False,
    "region": None,
    "no_vlm": False,
    "interactive": False,
    "verbose": False,
    "extra_images": {json.dumps(extra_images)},
    "context_text": {json.dumps(context_text)}
}}

res = run_pipeline('{main_image}', options)
print("===GEOVISION_JSON_START===")
print(res.to_json())
print("===GEOVISION_JSON_END===")
"""
            # Using PYTHONUNBUFFERED=1 to ensure live streaming
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                ["python3", "-c", runner_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    q.put({"type": "log", "data": line.strip()})
                    
            process.stdout.close()
            process.wait()
            q.put({"type": "done"})
        except Exception as e:
            q.put({"type": "error", "data": str(e)})

    threading.Thread(target=run_job, daemon=True).start()
    
    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream_job(job_id):
    def event_stream():
        q = active_jobs.get(job_id)
        if not q:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
            
        json_buffer = []
        capturing_json = False
        
        while True:
            try:
                msg = q.get(timeout=30)
                if msg["type"] == "done":
                    # Emit final result
                    try:
                        final_json = json.loads("".join(json_buffer))
                        
                        # Add location_estimates for the frontend fallback loop
                        if "location_estimates" not in final_json:
                            final_json["location_estimates"] = []
                            
                        if "best_estimate" in final_json and final_json["best_estimate"]:
                            final_json["location_estimates"].insert(0, {
                                "latitude": final_json["best_estimate"].get("latitude"),
                                "longitude": final_json["best_estimate"].get("longitude"),
                                "confidence": final_json["best_estimate"].get("confidence"),
                                "sources": final_json["best_estimate"].get("sources", []),
                                "evidence": final_json["best_estimate"].get("evidence", {})
                            })
                            
                        yield f"data: {json.dumps({'type': 'result', 'data': final_json})}\n\n"
                    except Exception as e:
                        logger.error(f"Error parsing final JSON: {e}")
                        pass
                    break
                elif msg["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': msg['data']})}\n\n"
                    break
                elif msg["type"] == "log":
                    line = msg["data"]
                    if line == "===GEOVISION_JSON_START===":
                        capturing_json = True
                        continue
                    elif line == "===GEOVISION_JSON_END===":
                        capturing_json = False
                        continue
                        
                    if capturing_json:
                        json_buffer.append(line)
                    else:
                        # Send log line to frontend
                        yield f"data: {json.dumps({'type': 'log', 'data': line})}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
                
    return app.response_class(event_stream(), mimetype="text/event-stream")


@app.route("/api/status")
def api_status():
    return jsonify({
        "service": "GeoVision",
        "version": "2.0.0",
        "capabilities": [
            "computer_vision_analysis",
            "deep_learning_feature_extraction",
            "satellite_imagery_matching",
            "park_proximity_analysis",
            "vlm_location_estimation"
        ]
    })

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
    app.run(host="0.0.0.0", port=9999, debug=True)
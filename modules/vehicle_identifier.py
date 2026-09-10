"""
GeoVision Vehicle Identifier
==============================
Detects vehicles in images and identifies make/model/year.

Pipeline per image:
  1. YOLO v8n detection (auto-downloads ~6MB)
  2. Per-crop: color HSV, type from aspect ratio, plate region detection + OCR
  3. VLM make/model ID if API key available
  4. Driving-side context from vehicle position in frame
  5. Aggregate geolocation hints

All steps degrade gracefully.
"""
from __future__ import annotations
import json, logging, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# YOLO COCO class IDs for vehicles
_VEHICLE_CLASSES = {2:"car",3:"motorcycle",5:"bus",7:"truck",1:"bicycle"}

# HSV color ranges (lower, upper) in OpenCV HSV (H:0-180)
_COLOR_RANGES = [
    ("red",    (  0,100, 80),(10,255,255)),
    ("red",    (160,100, 80),(180,255,255)),
    ("orange", ( 11,100, 80),( 25,255,255)),
    ("yellow", ( 26,100, 80),( 34,255,255)),
    ("green",  ( 35, 50, 50),( 85,255,255)),
    ("cyan",   ( 86, 50, 50),(100,255,255)),
    ("blue",   (101, 50, 50),(130,255,255)),
    ("purple", (131, 50, 50),(160,255,255)),
    ("white",  (  0,  0,200),(180, 30,255)),
    ("black",  (  0,  0,  0),(180,255, 50)),
    ("silver", (  0,  0,170),(180, 25,220)),
    ("grey",   (  0,  0, 80),(180, 20,180)),
]

_PLATE_FORMATS = {
    "european": "European (520×110mm)",
    "american": "American (305×152mm)",
    "square":   "Square / Asian wide",
}


def _detect_color(crop_bgr: np.ndarray) -> str:
    """Return dominant vehicle color name from a BGR crop."""
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        counts: Dict[str, int] = {}
        for name, lo, hi in _COLOR_RANGES:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            cnt = int(mask.sum() // 255)
            counts[name] = counts.get(name, 0) + cnt
        if counts:
            return max(counts, key=counts.get)
    except Exception:
        pass
    return "unknown"


def _classify_type(bbox_w: int, bbox_h: int, cls_id: int, conf: float) -> str:
    """Estimate vehicle body type from YOLO class + bounding box aspect ratio."""
    cls_name = _VEHICLE_CLASSES.get(cls_id, "vehicle")
    if cls_name in ("motorcycle", "bicycle"):
        return cls_name
    if cls_name == "bus":
        return "bus"
    if cls_name == "truck":
        return "truck"
    # car: further refine by aspect ratio
    ratio = bbox_w / max(bbox_h, 1)
    if ratio > 2.2:
        return "sedan/coupe"
    elif ratio > 1.6:
        return "SUV/wagon"
    else:
        return "van/minivan"


def _detect_plate(crop_bgr: np.ndarray) -> Dict[str, Any]:
    """Look for a license plate region in the lower third of a vehicle crop."""
    out: Dict[str, Any] = {"detected": False, "format": None, "region": None, "text": None}
    try:
        h, w = crop_bgr.shape[:2]
        lower = crop_bgr[h * 2 // 3:, :]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        # High-contrast rectangular regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw < 30 or ph < 8:
                continue
            ar = pw / max(ph, 1)
            if not (1.5 < ar < 6.0):
                continue
            area = pw * ph
            score = float(area * min(ar, 5.0))
            if score > best_score:
                best_score = score
                best = (x, y + h * 2 // 3, pw, ph)
        if best:
            x, y, pw, ph = best
            ar = pw / max(ph, 1)
            if ar > 4.0:
                fmt = "european"
            elif ar > 1.8:
                fmt = "american"
            else:
                fmt = "square"
            out.update({"detected": True, "format": fmt, "region": [x, y, x + pw, y + ph]})
            # Try OCR on the plate crop
            try:
                import easyocr
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                plate_crop = crop_bgr[max(0, y - h * 2 // 3):y - h * 2 // 3 + ph, x:x + pw]
                if plate_crop.size > 0:
                    results = reader.readtext(plate_crop, detail=0, paragraph=True)
                    text = " ".join(results).strip()
                    if text:
                        out["text"] = text.upper()
            except Exception:
                pass
    except Exception as e:
        logger.debug("Plate detection: %s", e)
    return out


def _driving_context(bbox_x: int, img_w: int) -> Dict[str, Any]:
    """Infer traffic side from which side of the image the vehicle occupies."""
    center_x = bbox_x
    if center_x < img_w * 0.40:
        pos = "left_side"
        implies = "right_hand_traffic"
    elif center_x > img_w * 0.60:
        pos = "right_side"
        implies = "left_hand_traffic"
    else:
        pos = "center"
        implies = "unknown"
    return {"position": pos, "implies_traffic_side": implies}


def _vlm_identify(crop_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """Use VLM to identify vehicle make/model/year. Returns None if unavailable."""
    api_key = os.environ.get("GEOVISION_VLM_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
    if not api_key:
        return None
    try:
        import base64, io as _io
        from openai import OpenAI
        base_url = os.environ.get("GEOVISION_VLM_BASE_URL", "https://opencode.ai/zen/v1")
        model    = os.environ.get("GEOVISION_VLM_MODEL", "opencode/gpt-5.4-nano")
        # Encode crop
        _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = (
            "Identify this vehicle. Provide make (manufacturer), model, approximate year "
            "or year range, and body type (sedan/SUV/truck/van/motorcycle/bus/other). "
            "Be specific. If you cannot identify the exact model, give your best estimate. "
            'Respond ONLY with JSON: {"make":"...","model":"...","year_range":"...","body_type":"...","confidence":0.0-1.0}'
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
            ]}],
        )
        text = resp.choices[0].message.content or ""
        # Strip fences
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        logger.debug("VLM vehicle ID: %s", e)
    return None


class VehicleIdentifier:
    """Detect and identify vehicles in a photo."""

    def __init__(self):
        self._yolo = None
        self._yolo_tried = False

    def _get_yolo(self):
        if self._yolo_tried:
            return self._yolo
        self._yolo_tried = True
        try:
            import torch
            from ultralytics import YOLO
            cache = Path(__file__).parent.parent / "data" / "yolo_cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
            # NEVER trigger a network auto-download (YOLO("yolov8n.pt") downloads ~6MB
            # and can hang the whole pipeline when the network is blocked, which under
            # a signal-managing shell aborts to SystemExit). Only load a model that is
            # already present locally; otherwise degrade gracefully.
            model_path = cache / "yolov8n.pt"
            if not model_path.exists():
                logger.info("VehicleIdentifier: yolov8n.pt not in %s — skipping YOLO "
                            "(vehicle/plate signals degraded)", cache)
                return None
            self._yolo = YOLO(str(model_path))
            logger.info("VehicleIdentifier: YOLOv8n loaded from %s", model_path)
        except Exception as e:
            logger.warning("VehicleIdentifier: YOLO unavailable — %s", e)
        return self._yolo

    def analyze(self, image_path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": "failed", "vehicle_count": 0, "vehicles": [],
            "driving_side_evidence": "unknown", "dominant_vehicle_color": None,
            "geolocation_hints": [],
        }
        try:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                out["summary"] = "Cannot load image"
                return out
            img_h, img_w = img_bgr.shape[:2]

            detections: List[Dict[str, Any]] = []

            # --- YOLO detection ---
            yolo = self._get_yolo()
            if yolo is not None:
                try:
                    import concurrent.futures
                    def _run_yolo():
                        return yolo(img_bgr, verbose=False, conf=0.30)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_run_yolo)
                        results = fut.result(timeout=25)
                    for res in results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id not in _VEHICLE_CLASSES:
                                continue
                            conf = float(box.conf[0])
                            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                            detections.append({"bbox":[x1,y1,x2,y2],"cls_id":cls_id,"det_conf":conf})
                except Exception as e:
                    logger.warning("YOLO run failed: %s", e)

            # Fallback: simple motion/contour-based vehicle heuristic
            if not detections:
                detections = self._fallback_detect(img_bgr)

            vehicles = []
            color_votes: Dict[str,int] = {}
            side_votes: Dict[str,int] = {}

            for idx, det in enumerate(detections[:8]):
                x1,y1,x2,y2 = det["bbox"]
                # Expand by 10%
                pad_x = int((x2-x1)*0.10); pad_y = int((y2-y1)*0.10)
                x1c = max(0,x1-pad_x); y1c = max(0,y1-pad_y)
                x2c = min(img_w,x2+pad_x); y2c = min(img_h,y2+pad_y)
                crop = img_bgr[y1c:y2c, x1c:x2c]

                color = _detect_color(crop)
                color_votes[color] = color_votes.get(color,0)+1

                bw, bh = x2-x1, y2-y1
                vtype = _classify_type(bw, bh, det.get("cls_id",2), det.get("det_conf",0.5))

                plate = _detect_plate(crop)
                driving = _driving_context((x1+x2)//2, img_w)
                impl = driving["implies_traffic_side"]
                if impl != "unknown":
                    side_votes[impl] = side_votes.get(impl,0)+1

                # VLM (only for first 2 vehicles to limit API calls)
                vlm_data = None
                if idx < 2:
                    vlm_data = _vlm_identify(crop)

                v: Dict[str, Any] = {
                    "id": idx+1,
                    "bbox": [x1,y1,x2,y2],
                    "type": vtype,
                    "color": color,
                    "confidence": round(det.get("det_conf",0.5),3),
                    "make": vlm_data.get("make") if vlm_data else None,
                    "model": vlm_data.get("model") if vlm_data else None,
                    "year_range": vlm_data.get("year_range") if vlm_data else None,
                    "vlm_confidence": vlm_data.get("confidence") if vlm_data else None,
                    "license_plate": plate,
                    "driving_context": driving,
                }
                vehicles.append(v)

            out["vehicle_count"] = len(vehicles)
            out["vehicles"] = vehicles

            # Dominant color
            if color_votes:
                out["dominant_vehicle_color"] = max(color_votes, key=color_votes.get)

            # Driving side consensus
            if side_votes:
                top = max(side_votes, key=side_votes.get)
                out["driving_side_evidence"] = "right" if "right_hand" in top else "left"

            # Geolocation hints
            hints = []
            ds = out["driving_side_evidence"]
            if ds == "right":
                hints.append("Vehicle positions suggest right-hand traffic (USA/Canada/Europe/China/Brazil/etc.)")
            elif ds == "left":
                hints.append("Vehicle positions suggest left-hand traffic (UK/Australia/Japan/India/etc.)")

            for v in vehicles:
                pl = v.get("license_plate",{})
                if pl.get("detected"):
                    fmt = pl.get("format","")
                    if fmt == "european":
                        hints.append("European-format license plate detected")
                    elif fmt == "american":
                        hints.append("North American-format license plate detected")
                if v.get("make"):
                    hints.append(f"Vehicle identified: {v['make']} {v.get('model','')}")
                    break  # just first

            out["geolocation_hints"] = list(dict.fromkeys(hints))  # dedup
            out["status"] = "success"

        except Exception as e:
            logger.error("VehicleIdentifier: %s", e, exc_info=True)
            out["status"] = "failed"
            out["error"] = str(e)

        return out

    def _fallback_detect(self, img_bgr: np.ndarray) -> List[Dict[str,Any]]:
        """Simple contour-based vehicle region heuristic when YOLO is unavailable."""
        detections = []
        try:
            h, w = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(cnt)
                if area < (h*w*0.03):
                    continue
                x,y,cw,ch = cv2.boundingRect(cnt)
                ar = cw/max(ch,1)
                if 0.5 < ar < 5.0:
                    detections.append({"bbox":[x,y,x+cw,y+ch],"cls_id":2,"det_conf":0.4})
                if len(detections) >= 4:
                    break
        except Exception:
            pass
        return detections

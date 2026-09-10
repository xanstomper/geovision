"""
GeoVision Video Frame Sampler — inference from a video clip (OceanIR-style).

OceanIR accepts a video and samples frames across it. This module:
  1. opens a video with OpenCV
  2. samples N frames spread across the clip (evenly spaced, not burst)
  3. optionally de-dup similar frames (camera pans produce near-duplicates)
  4. runs each informative frame through the geolocation harness
  5. fuses per-frame best estimates into one verdict (mode/consensus of coords)

CPU-only, key-free. Local-frame sampling + harness calls; no subprocess shells.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def sample_frames(video_path: str, n: int = 5, min_gap_frames: Optional[int] = None) -> List[int]:
    """Return frame indices evenly sampled across the clip. `n` frames. The gap is
    proportional to clip length so short clips keep all n; only true near-burst
    frames (camera pan) are deduped."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 300
        n_eff = min(n, max(1, total))
        # even spread across the whole clip
        step = max(1, total // n_eff)
        idx = list(range(0, total, step))[:n_eff]
        if not idx:
            idx = [0]
        # proportional min gap: at most 10% of clip between kept frames
        gap = min_gap_frames if min_gap_frames is not None else max(1, int(total * 0.10))
        dedup = []
        last = -gap
        for i in sorted(idx):
            if i - last >= gap:
                dedup.append(i)
                last = i
        return dedup or [0]
    finally:
        cap.release()


def read_frames(video_path: str, indices: List[int]) -> Dict[int, Any]:
    """Read specific frames -> numpy BGR arrays (sized for analysis)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    out: Dict[int, Any] = {}
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                out[i] = frame
    finally:
        cap.release()
    return out


def save_frames(video_path: str, indices: List[int], out_dir: str,
                prefix: str = "frame") -> List[str]:
    """Write sampled frames to disk as .jpg, return their paths."""
    import cv2
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    frames = read_frames(video_path, indices)
    paths = []
    for i, frame in frames.items():
        p = os.path.join(out_dir, f"{prefix}_{i:05d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
    return paths


def geolocate_video(video_path: str, n_frames: int = 5,
                    harness: Optional[Any] = None,
                    location_hint: Optional[str] = None,
                    canvas: Optional[Any] = None) -> Dict[str, Any]:
    """Sample frames from a video and run the geolocation harness on each, then
    fuse into a consensus verdict (mean of frame estimates, majority city)."""
    import tempfile, collections, numpy as np
    idx = sample_frames(video_path, n=n_frames)
    if harness is None:
        from modules.geo_harness import GeoVisionHarness
        harness = GeoVisionHarness()

    if canvas is not None:
        canvas.add("step", title="Video geolocation",
                   text=f"sampled {len(idx)} frames across the clip; analyzing each")

    estimates: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        paths = save_frames(video_path, idx, td)
        for i, fp in zip(idx, paths):
            if canvas is not None:
                canvas.add("image", title=f"frame {i}", image=fp,
                           caption=f"sampled frame {i}")
            rec = harness.investigate(fp, location_hint=location_hint, canvas=canvas)
            est = rec.get("best_estimate")
            if est:
                estimates.append({"frame": i, **est, "record": rec})

    # Fuse: mean of lat/lon + majority city.
    if not estimates:
        return {"status": "limited", "frames": len(idx), "note": "no frame yielded an estimate"}
    mean_lat = float(np.mean([e.get("latitude", 0) for e in estimates]))
    mean_lon = float(np.mean([e.get("longitude", 0) for e in estimates]))
    cities = collections.Counter(str(e.get("city") or "") for e in estimates if e.get("city"))
    country = collections.Counter(str(e.get("country") or "") for e in estimates if e.get("country"))
    fused = {
        "status": "success",
        "frames_analyzed": len(idx),
        "frame_estimates": [{k: e.get(k) for k in ("frame", "latitude", "longitude", "confidence", "city", "country")}
                            for e in estimates],
        "fused_latitude": mean_lat,
        "fused_longitude": mean_lon,
        "majority_city": cities.most_common(1)[0][0] if cities else None,
        "majority_country": country.most_common(1)[0][0] if country else None,
        "confidence": float(np.mean([e.get("confidence", 0) for e in estimates])),
        "uncalibrated": True,
    }
    if canvas is not None:
        canvas.add("verdict", title="Video geolocation (fused)",
                   text=(f"{mean_lat:.5f}, {mean_lon:.5f}\n"
                         f"majority {fused['majority_city']}, {fused['majority_country']} "
                         f"across {len(estimates)} frames"),
                   lat=mean_lat, lon=mean_lon, confidence=fused["confidence"])
    return fused


if __name__ == "__main__":
    import sys
    print(geolocate_video(sys.argv[1], n_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 5))
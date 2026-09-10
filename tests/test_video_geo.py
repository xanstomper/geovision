"""
Tests for the video frame sampler (OceanIR-style video geolocation).

Frame sampling is CPU/OpenCV determinism — tests generate a small synthetic clip
and assert even spread + dedup. The full geolocate_video harness call is heavy, so
it's exercised separately; here we validate the sampling + frame IO.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from modules.video_geo import sample_frames, save_frames


@pytest.fixture()
def clip(tmp_path):
    import cv2
    p = str(tmp_path / "clip.mp4")
    N, w, h, fps = 50, 64, 64, 5
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(N):
        vw.write(np.full((h, w, 3), i, dtype=np.uint8))
    vw.release()
    return p


def test_sample_frames_even_spread(clip):
    idx = sample_frames(clip, n=5)
    assert len(idx) == 5
    assert idx[0] == 0 and idx[-1] == 40      # 50 frames, even spread step 10
    assert idx == sorted(idx)                 # ascending


def test_sample_more_than_frames(clip):
    """Asking for more frames than exist clamps to available."""
    idx = sample_frames(clip, n=100)
    assert len(idx) <= 50 and len(idx) >= 1


def test_save_frames_writes_jpg(clip, tmp_path):
    paths = save_frames(clip, [0, 20, 40], str(tmp_path / "f"),
                        prefix="frame")
    assert len(paths) == 3
    assert all(os.path.exists(x) and os.path.getsize(x) > 0 for x in paths)


def test_cannot_open_nonexistent():
    with pytest.raises(ValueError):
        sample_frames("/nonexistent/video.mp4")
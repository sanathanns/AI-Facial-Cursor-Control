"""Integration test for MediaPipe Face Landmarker Tasks API and monotonic timestamps."""
import os
import time
import numpy as np
import pytest
from src.config import AppConfig
from src.face_tracker import FaceTracker, FaceLandmarksResult


def test_tracker_model_path_resolution():
    cfg = AppConfig()
    assert os.path.isabs(cfg.model_path)
    assert os.path.exists(cfg.model_path)
    assert "face_landmarker.task" in cfg.model_path


def test_tracker_monotonic_timestamps():
    cfg = AppConfig()
    tracker = FaceTracker(cfg.model_path)

    ts1 = tracker._next_monotonic_timestamp()
    ts2 = tracker._next_monotonic_timestamp()
    ts3 = tracker._next_monotonic_timestamp()

    assert ts1 < ts2 < ts3
    tracker.close()


def test_tracker_process_synthetic_frame():
    cfg = AppConfig()
    tracker = FaceTracker(cfg.model_path)

    # Blank black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(frame)

    assert isinstance(res, FaceLandmarksResult)
    # On a completely black frame, no face should be detected
    assert res.face_detected is False
    assert res.nose_tip is None

    tracker.close()

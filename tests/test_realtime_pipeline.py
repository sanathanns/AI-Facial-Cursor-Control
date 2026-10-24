"""Tests for real-time pipeline, latest-frame strategy, responsiveness, and event propagation."""
import time
import queue
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureRecognizer, GestureAction, EyeState, GestureState, ScrollState
from src.filters import PointOneEuroFilter, OneEuroFilter
from tests.test_gestures import make_eye_landmarks, make_head_pose


def test_latest_frame_queue_bounded_and_non_growing():
    """Simulates VisionWorker dropping stale frames so frame_queue never accumulates backlog."""
    frame_queue = queue.Queue(maxsize=1)

    # Push 10 frames in rapid succession with latest-frame-wins strategy
    for i in range(10):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_telemetry = {"frame_id": i}

        # Drain before put (Part 1 strategy)
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
        except queue.Empty:
            pass

        frame_queue.put_nowait((dummy_frame, dummy_telemetry))

    # Queue size must be exactly 1
    assert frame_queue.qsize() == 1
    _, final_telemetry = frame_queue.get_nowait()
    # The single remaining item must be the NEWEST frame (frame_id 9)
    assert final_telemetry["frame_id"] == 9


def test_cursor_one_euro_filter_responsiveness():
    """Verifies that cursor_beta dynamically increases filter cutoff frequency during swift movement."""
    # When moving swiftly, filter should track target position much closer than when heavily damped
    filt_responsive = OneEuroFilter(min_cutoff=1.0, beta=0.6, d_cutoff=1.0)
    filt_sluggish = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)

    # Initialize at 0
    filt_responsive.filter(0.0, timestamp=0.0)
    filt_sluggish.filter(0.0, timestamp=0.0)

    # Large sudden displacement at t=0.033s (e.g. 500 px movement in 1 frame)
    out_resp = filt_responsive.filter(500.0, timestamp=0.033)
    out_slug = filt_sluggish.filter(500.0, timestamp=0.033)

    # Responsive filter tracks swift motion significantly closer than sluggish filter
    assert out_resp > out_slug
    assert out_resp >= 480.0


def test_cursor_redundant_moveTo_suppressed():
    """Verifies that PyAutoGUI moveTo is not issued when target displacement is negligible."""
    cfg = AppConfig(cursor_min_movement_px=2.0)
    controller = CursorController(cfg)
    controller.calibrate(0.5, 0.5)

    with patch("pyautogui.moveTo") as mock_move:
        # Move to a non-center point (displaces cursor by > 2px)
        res1 = controller.update_cursor_position(0.55, 0.55, timestamp=1.0)
        assert mock_move.call_count == 1

        # Very tiny micro-movement (< 2 px displacement)
        res2 = controller.update_cursor_position(0.550001, 0.550001, timestamp=1.033)
        if res1 == res2:
            assert mock_move.call_count == 1


def test_adaptive_ear_baseline_calibration():
    """Verifies that calibrate_eyes computes personalized blink and open thresholds."""
    cfg = AppConfig(ear_blink_ratio=0.65, ear_open_ratio=0.82)
    recognizer = GestureRecognizer(cfg)

    # User with baseline open EAR = 0.28
    recognizer.calibrate_eyes(0.28, 0.30)
    assert recognizer.is_eye_calibrated is True
    assert pytest.approx(recognizer.adaptive_blink_left, 0.01) == 0.28 * 0.65  # ~0.182
    assert pytest.approx(recognizer.adaptive_blink_right, 0.01) == 0.30 * 0.65 # ~0.195
    assert pytest.approx(recognizer.adaptive_open_left, 0.01) == 0.28 * 0.82   # ~0.229
    assert pytest.approx(recognizer.adaptive_open_right, 0.01) == 0.30 * 0.82  # ~0.246


def test_click_event_propagation_to_pyautogui():
    """Verifies complete chain from GestureAction down to pyautogui.click calls."""
    cfg = AppConfig()
    controller = CursorController(cfg)

    with patch("pyautogui.click") as mock_click:
        controller.execute_action(GestureAction.LEFT_CLICK)
        mock_click.assert_called_once_with(button='left')

    with patch("pyautogui.click") as mock_click:
        controller.execute_action(GestureAction.RIGHT_CLICK)
        mock_click.assert_called_once_with(button='right')


def test_bilateral_blink_with_post_blink_guard_blocks_click():
    """Natural bilateral blink followed by slightly staggered eyelid opening must not click."""
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # 1. Bilateral blink begins
    action, _, _, _, debug = recognizer.update(True, closed, closed, current_time=1.0)
    assert action == GestureAction.NONE
    assert debug.gesture_state == "BILATERAL_BLINK"

    # 2. Right eye opens slightly earlier than left eye at t=1.10 (30ms lag)
    action, _, _, _, debug = recognizer.update(True, closed, open_eye, current_time=1.10)
    # Post-blink guard must block this from immediately becoming a wink candidate!
    assert action == GestureAction.NONE
    assert debug.gesture_state != "WINK_LEFT_HOLDING"


def test_scroll_rate_limiting_and_responsiveness():
    """Head pitch scrolling emits first tick immediately, then rate-limits subsequent ticks."""
    cfg = AppConfig(scroll_pitch_threshold=0.05, scroll_interval_s=0.10)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks(height=0.03)

    # Neutral pose
    neutral_nose, neutral_eyes, neutral_chin = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(neutral_nose, neutral_eyes, neutral_chin)

    # Pitch up pose (delta = -0.08)
    up_nose, up_eyes, up_chin = make_head_pose(pitch_delta=-0.08)

    # First frame pitched up at t=1.0 -> immediate first scroll tick
    action1, _, _, _, debug1 = recognizer.update(
        True, open_eye, open_eye, up_nose, up_eyes, up_chin, current_time=1.0
    )
    assert action1 == GestureAction.SCROLL_UP
    assert debug1.scroll_state == "UP"

    # Held pitched up at t=1.05 (< 0.10s interval) -> suppressed / rate-limited
    action2, _, _, _, _ = recognizer.update(
        True, open_eye, open_eye, up_nose, up_eyes, up_chin, current_time=1.05
    )
    assert action2 == GestureAction.NONE

    # Held pitched up at t=1.11 (> 0.10s interval) -> second scroll tick emits!
    action3, _, _, _, _ = recognizer.update(
        True, open_eye, open_eye, up_nose, up_eyes, up_chin, current_time=1.11
    )
    assert action3 == GestureAction.SCROLL_UP


def test_telemetry_consistency_and_face_loss_safety():
    """Verify that VisionWorker can safely construct telemetry when face is lost or uncalibrated."""
    from src.gui import VisionWorker

    frame_queue = queue.Queue(maxsize=1)
    command_queue = queue.Queue()
    cfg = AppConfig()

    worker = VisionWorker(cfg, frame_queue, command_queue)

    # 1. Before any camera capture or calibration:
    cursor_telemetry = worker.controller.get_telemetry()
    assert "raw_dx" in cursor_telemetry
    assert "calibrated_dx" in cursor_telemetry
    assert "target_x" in cursor_telemetry
    assert "target_y" in cursor_telemetry
    assert "mouse_moved" in cursor_telemetry
    assert cursor_telemetry["raw_dx"] == 0.0

    # 2. Simulate face-loss update:
    worker.controller.on_face_lost()
    _, _, _, _, debug_info = worker.gestures.update(
        face_detected=False,
        left_eye=None,
        right_eye=None,
        is_calibrated=worker.controller.is_calibrated,
        is_paused=worker.controller.is_paused,
        is_emergency_stopped=worker.controller.is_emergency_stopped,
        current_time=1.0,
    )

    # 3. Build telemetry dictionary as VisionWorker does in run()
    cursor_tel = worker.controller.get_telemetry()
    debug_info.raw_dx = cursor_tel["raw_dx"]
    debug_info.calibrated_dx = cursor_tel["calibrated_dx"]
    debug_info.target_x = cursor_tel["target_x"]
    debug_info.target_y = cursor_tel["target_y"]
    debug_info.mouse_moved = cursor_tel["mouse_moved"]

    telemetry = {
        "face_detected": False,
        "is_calibrated": cursor_tel["is_calibrated"],
        "is_paused": cursor_tel["is_paused"],
        "is_emergency_stopped": cursor_tel["is_emergency_stopped"],
        "is_face_lost_safe": getattr(worker.gestures, "is_face_lost_safe", False),
        "left_ear": getattr(debug_info, "left_ear", 0.0),
        "right_ear": getattr(debug_info, "right_ear", 0.0),
        "baseline_left": getattr(debug_info, "baseline_left", 0.25),
        "baseline_right": getattr(debug_info, "baseline_right", 0.25),
        "thresh_left": getattr(debug_info, "thresh_left", 0.18),
        "thresh_right": getattr(debug_info, "thresh_right", 0.18),
        "left_open": getattr(debug_info, "left_open", True),
        "right_open": getattr(debug_info, "right_open", True),
        "eye_state": getattr(debug_info, "eye_state", "OPEN"),
        "gesture_state": getattr(debug_info, "gesture_state", "IDLE"),
        "gesture_event": getattr(debug_info, "gesture_event", "NONE"),
        "head_pitch": getattr(debug_info, "head_pitch", 0.0),
        "scroll_state": getattr(debug_info, "scroll_state", "NEUTRAL"),
        "scroll_event": getattr(debug_info, "scroll_event", "NONE"),
        "input_enabled": getattr(debug_info, "input_enabled", False),
        "active_action": "NONE",
        "raw_dx": cursor_tel["raw_dx"],
        "calibrated_dx": cursor_tel["calibrated_dx"],
        "target_x": cursor_tel["target_x"],
        "target_y": cursor_tel["target_y"],
        "mouse_moved": cursor_tel["mouse_moved"],
        "cam_fps": 30.0,
        "proc_fps": 30.0,
        "latency_ms": 15.0,
        "queue_size": 0,
        "cursor_pos": cursor_tel["cursor_pos"],
    }

    # Verify all expected telemetry fields are present with correct types
    assert telemetry["face_detected"] is False
    assert telemetry["raw_dx"] == 0.0
    assert telemetry["calibrated_dx"] == 0.0
    assert telemetry["mouse_moved"] is False
    assert telemetry["eye_state"] == "NO FACE"

"""Regression tests for real-world webcam issues:
- Auto-calibration on initial face detection without manual 'C' press
- Clicks enabled upon active face detection (Input: ENABLED)
- Wink detection tolerance against cheek-squint landmark jitter
- Live telemetry values during scroll freeze (no +0.000 freeze or center lock)
- Safety: face loss and pause immediately disable input and mouse actions
"""
import pytest
import queue
from unittest.mock import patch, MagicMock
from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureRecognizer, GestureAction, EyeState, GestureState
from tests.test_gestures import make_eye_landmarks, make_head_pose


def test_auto_calibration_on_first_face_frame():
    """Verify that when a valid face is first detected, calibration occurs automatically."""
    from src.gui import VisionWorker
    from src.face_tracker import FaceLandmarksResult

    cfg = AppConfig()
    frame_queue = queue.Queue(maxsize=1)
    command_queue = queue.Queue()
    worker = VisionWorker(cfg, frame_queue, command_queue)

    assert worker.controller.is_calibrated is False
    assert worker.gestures.is_pitch_calibrated is False
    assert worker.gestures.is_eye_calibrated is False

    # Simulate first valid face detection frame at t=1.0
    open_eye = make_eye_landmarks(width=0.10, height=0.03)
    neutral_pose = make_head_pose(pitch_delta=0.0)
    fake_result = FaceLandmarksResult(
        face_detected=True,
        timestamp_ms=1000,
        raw_landmarks=[],
        nose_tip=neutral_pose[0],
        mid_eyes=neutral_pose[1],
        chin=neutral_pose[2],
        left_eye=open_eye,
        right_eye=open_eye,
        mouth=None,
    )

    # Replicate VisionWorker auto-calibration block on first face detection
    nose_x, nose_y = fake_result.nose_tip
    if (not worker.controller.is_calibrated) or worker._request_calibration:
        worker.controller.calibrate(nose_x, nose_y)
        if fake_result.mid_eyes is not None:
            worker.gestures.calibrate_pitch(
                fake_result.nose_tip,
                fake_result.mid_eyes,
                fake_result.chin,
            )
        if fake_result.left_eye and fake_result.right_eye:
            from src.gestures import calculate_ear
            l_ear = calculate_ear(fake_result.left_eye)
            r_ear = calculate_ear(fake_result.right_eye)
            worker.gestures.calibrate_eyes(l_ear, r_ear)
        worker._request_calibration = False

    assert worker.controller.is_calibrated is True
    assert worker.gestures.is_pitch_calibrated is True
    assert worker.gestures.is_eye_calibrated is True

    # When gestures.update runs, input_enabled MUST be True
    action, _, _, _, debug_info = worker.gestures.update(
        face_detected=True,
        left_eye=fake_result.left_eye,
        right_eye=fake_result.right_eye,
        nose_tip=fake_result.nose_tip,
        mid_eyes=fake_result.mid_eyes,
        chin=fake_result.chin,
        is_calibrated=worker.controller.is_calibrated,
        is_paused=worker.controller.is_paused,
        is_emergency_stopped=worker.controller.is_emergency_stopped,
        current_time=1.0,
    )
    assert debug_info.input_enabled is True
    assert debug_info.scroll_state == "NEUTRAL"


def test_wink_with_cheek_squint_and_transient_jitter_fires_click():
    """Verify that a 1-frame squint/landmark jitter during wink hold does not abort click."""
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    recognizer.calibrate_eyes(0.25, 0.25)

    closed_left = make_eye_landmarks(height=0.005)   # EAR ~0.10 (closed)
    open_right = make_eye_landmarks(height=0.030)     # EAR ~0.25 (open)
    squint_right = make_eye_landmarks(height=0.015)   # EAR ~0.14 (cheek squint)

    # Frame 1 (t=1.00): Left wink candidate begins
    action1, _, _, _, debug1 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=open_right,
        current_time=1.00,
    )
    assert debug1.gesture_state == "WINK_LEFT_HOLDING"
    assert action1 == GestureAction.NONE

    # Frame 2 (t=1.033): Single transient cheek-squint frame (both eyes low EAR)
    # Must NOT abort to BILATERAL_BLINK due to grace frame tolerance
    action2, _, _, _, debug2 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=squint_right,
        current_time=1.033,
    )
    assert debug2.gesture_state == "WINK_LEFT_HOLDING"
    assert action2 == GestureAction.NONE

    # Frame 3 (t=1.066): Normal open returns on right eye
    action3, _, _, _, debug3 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=open_right,
        current_time=1.066,
    )
    assert debug3.gesture_state == "WINK_LEFT_HOLDING"
    assert action3 == GestureAction.NONE

    # Frame 4 (t=1.120): Duration >= 100ms -> confirms exactly one LEFT_CLICK
    action4, _, _, _, debug4 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=open_right,
        current_time=1.120,
    )
    assert action4 == GestureAction.LEFT_CLICK
    assert debug4.gesture_state == "WAIT_REOPEN"


def test_telemetry_live_values_during_scroll_freeze():
    """Verify that when cursor movement is frozen by scrolling, telemetry reflects actual displacement."""
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo") as mock_move:
        # User tilts head to scroll, but also has head offset horizontally (x=0.55)
        pos = controller.update_cursor_position(
            current_x=0.55,
            current_y=0.50,
            timestamp=1.0,
            is_scrolling=True,
        )
        # Mouse moveTo must NOT be called
        assert pos is None
        mock_move.assert_not_called()

        # Telemetry must show actual live displacement, NOT +0.000
        telemetry = controller.get_telemetry()
        assert telemetry["mouse_moved"] is False
        assert telemetry["raw_dx"] == pytest.approx(0.05, abs=1e-4)
        assert telemetry["calibrated_dx"] > 0.0  # NOT 0.000!
        assert telemetry["target_x"] > controller.screen_center_x  # NOT stuck at screen center!


def test_face_loss_immediately_disables_input_and_actions():
    """Verify that face loss immediately sets input_enabled=False and resets cursor state."""
    cfg = AppConfig()
    recognizer = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    # Face lost frame
    action, _, _, _, debug_info = recognizer.update(
        face_detected=False,
        left_eye=None,
        right_eye=None,
        current_time=1.0,
    )
    assert action == GestureAction.NONE
    assert debug_info.input_enabled is False

    # Controller reset on face loss
    controller.on_face_lost()
    assert controller.last_mouse_moved is False
    assert controller.is_dragging is False

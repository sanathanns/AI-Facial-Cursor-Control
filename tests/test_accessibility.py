"""Automated tests for accessibility-first cursor control, scrolling, and wink detection.

Focuses on:
1. Small comfortable head movements traversing the screen without neck strain.
2. Independent horizontal and vertical sensitivity controls.
3. Preserving neutral calibration across sensitivity changes.
4. Dedicated Scroll Mode freezing cursor and triggering on small pitch movements.
5. Eye asymmetry classification handling real-world winks with squinting.
6. Strict rejection of natural bilateral blinks.
7. Instant wink firing at min duration without hold disqualification.
"""
from unittest.mock import patch, MagicMock
import pytest
import math

from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureRecognizer, GestureAction, EyeState, GestureState, ScrollState
from tests.test_gestures import make_eye_landmarks, make_head_pose


def test_small_head_movement_traverses_screen():
    """Verify that a small comfortable 2-3 cm head movement (dx=0.035) moves cursor across screen."""
    cfg = AppConfig(
        sensitivity_x=2.5,
        sensitivity_y=2.5,
        comfort_range_x=0.050,
        comfort_range_y=0.045,
        dead_zone_radius=0.012,
    )
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        # Small head turn right: dx = 0.035 (comfortable movement)
        pos = controller.update_cursor_position(0.535, 0.50, timestamp=1.0)
        assert pos is not None
        # Effective dx = 0.035 - 0.012 = 0.023
        # norm_dx = 0.023 / 0.050 = 0.46
        # target_x = 960 + 0.46 * 2.5 * 960 = 2064 -> clamped to 1919
        assert pos[0] >= 1700, f"Expected substantial horizontal traversal, got {pos[0]}"
        assert 520 <= pos[1] <= 560


def test_independent_horizontal_and_vertical_sensitivity():
    """Verify changing sensitivity_x alters horizontal travel without affecting vertical travel."""
    cfg = AppConfig(
        sensitivity_x=1.5,
        sensitivity_y=3.0,
        comfort_range_x=0.050,
        comfort_range_y=0.045,
        dead_zone_radius=0.010,
    )
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos1 = controller.update_cursor_position(0.53, 0.53, timestamp=1.0)
        assert pos1 is not None

        # Increase horizontal sensitivity only
        controller.config.sensitivity_x = 3.0
        controller.update_config_params()
        pos2 = controller.update_cursor_position(0.53, 0.53, timestamp=1.033)
        assert pos2 is not None

        # Horizontal travel should be greater in pos2
        assert (pos2[0] - 960) > (pos1[0] - 960)
        # Vertical travel should remain consistent
        assert abs(pos2[1] - pos1[1]) < 5


def test_changing_sensitivity_preserves_neutral_calibration():
    """Verify modifying sensitivity or comfort range sliders preserves neutral anchor coordinates."""
    cfg = AppConfig(sensitivity_x=2.0, sensitivity_y=2.0)
    controller = CursorController(cfg)
    controller.calibrate(0.485, 0.515)
    assert controller.neutral_x == 0.485
    assert controller.neutral_y == 0.515
    assert controller.is_calibrated is True

    # User adjusts sliders in GUI
    controller.config.sensitivity_x = 4.5
    controller.config.comfort_range_x = 0.035
    controller.update_config_params()

    # Neutral calibration anchor must remain intact
    assert controller.neutral_x == 0.485
    assert controller.neutral_y == 0.515
    assert controller.is_calibrated is True


def test_automatic_scroll_freezes_cursor_and_scrolls_on_small_pitch():
    """Verify automatic scrolling freezes cursor and activates with minimal head pitch."""
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, scroll_interval_s=0.10)
    recognizer = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    open_eye = make_eye_landmarks()
    neutral_pose = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(*neutral_pose)

    with patch("pyautogui.moveTo") as mock_move:
        # While scrolling, cursor position must freeze and not move mouse
        pos = controller.update_cursor_position(0.55, 0.52, timestamp=1.0, is_scrolling=True)
        assert pos is None
        mock_move.assert_not_called()

    # Small comfortable pitch upward (-0.025 < -0.020)
    small_up_pose = make_head_pose(pitch_delta=-0.025)
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=small_up_pose[0],
        mid_eyes=small_up_pose[1],
        chin=small_up_pose[2],
        current_time=1.0,
    )
    assert action == GestureAction.SCROLL_UP
    assert debug.scroll_state == "UP"


def test_small_pitch_variation_within_threshold_prevents_accidental_scroll():
    """Natural small pitch fluctuations within threshold must NOT trigger scrolling."""
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.025)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()
    neutral_pose = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(*neutral_pose)

    # Slight pitch delta (+0.012 < +0.025) during normal pointer movement
    slight_pose = make_head_pose(pitch_delta=0.012)
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=slight_pose[0],
        mid_eyes=slight_pose[1],
        chin=slight_pose[2],
        current_time=1.0,
    )
    # Must NOT scroll when pitch is within comfort threshold
    assert action == GestureAction.NONE
    assert debug.scroll_state == "NEUTRAL"


def test_real_world_wink_with_squinting_triggers_click():
    """Verify that a wink with cheek squint (reducing open eye EAR) is reliably classified."""
    cfg = AppConfig(
        ear_blink_ratio=0.75,
        ear_open_ratio=0.82,
        ear_asymmetry_min_diff=0.030,
        wink_min_duration_s=0.10,
    )
    recognizer = GestureRecognizer(cfg)
    # User's resting baseline open EAR is 0.22
    recognizer.calibrate_eyes(0.22, 0.22)

    # Closed left eye EAR ~0.13, squinting right eye EAR ~0.18
    closed_left = make_eye_landmarks(height=0.005)   # EAR ~0.13
    squint_right = make_eye_landmarks(height=0.015)  # EAR ~0.18

    # Start wink candidate at t=1.00
    action1, _, _, _, debug1 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=squint_right,
        current_time=1.00,
    )
    assert debug1.eye_state in ("LEFT WINK", "LEFT CLOSED")
    assert debug1.gesture_state == "WINK_LEFT_HOLDING"
    assert action1 == GestureAction.NONE

    # After 120ms (t=1.12), click fires immediately
    action2, _, _, _, debug2 = recognizer.update(
        face_detected=True,
        left_eye=closed_left,
        right_eye=squint_right,
        current_time=1.12,
    )
    assert action2 == GestureAction.LEFT_CLICK
    assert debug2.gesture_state == "WAIT_REOPEN"


def test_bilateral_blink_strictly_ignored_as_click():
    """Verify natural symmetric blink (both eyes closing) never creates a click candidate."""
    cfg = AppConfig(ear_blink_ratio=0.75, wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    recognizer.calibrate_eyes(0.24, 0.24)

    closed = make_eye_landmarks(height=0.005)

    # Blink begins
    action1, _, _, _, debug1 = recognizer.update(True, closed, closed, current_time=1.00)
    assert action1 == GestureAction.NONE
    assert debug1.eye_state in ("BLINK", "BOTH CLOSED")
    assert debug1.gesture_state == "BILATERAL_BLINK"

    # Held closed
    action2, _, _, _, debug2 = recognizer.update(True, closed, closed, current_time=1.15)
    assert action2 == GestureAction.NONE
    assert debug2.gesture_state == "BILATERAL_BLINK"

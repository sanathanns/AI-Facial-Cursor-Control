"""Tests verifying cursor stability, jitter suppression, and scroll anti-oscillation fixes.

Covers:
1. Small pitch fluctuations around neutral do not trigger scrolling.
2. Sustained intentional head tilt activates scrolling appropriately.
3. Continuous tilt does not repeatedly retrigger or bounce scrolling.
4. Returning to neutral releases scrolling promptly and restores cursor movement.
5. Stationary head in neutral / deadzone produces zero mouse moves (AT_REST).
6. Deliberate small head movements outside deadzone remain responsive.
7. Neutral rearm prevents retriggering at hysteresis boundary.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureRecognizer, GestureAction, ScrollState
from tests.test_gestures import make_eye_landmarks, make_head_pose


@pytest.fixture(autouse=True)
def mock_pyautogui():
    with patch("pyautogui.moveTo") as m_move, patch("pyautogui.click") as m_click, patch("pyautogui.scroll") as m_scroll:
        yield m_move


def test_small_pitch_fluctuations_around_neutral_do_not_trigger_scrolling():
    """Small natural fluctuations (sensor noise, breathing) around neutral baseline
    must NOT trigger scrolling or freeze the cursor.
    """
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, dead_zone_radius=0.015)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Fluctuations within +/-0.012 (below the 0.020 activation threshold)
    noise_deltas = [0.004, -0.006, 0.011, -0.008, 0.009, -0.011, 0.002, 0.000]
    for i, delta in enumerate(noise_deltas):
        pose = make_head_pose(pitch_delta=delta)
        action, _, _, _, debug = gestures.update(
            face_detected=True,
            left_eye=open_eye,
            right_eye=open_eye,
            nose_tip=pose[0],
            mid_eyes=pose[1],
            chin=pose[2],
            is_calibrated=True,
            current_time=1.0 + i * 0.033,
        )
        assert debug.scroll_state == "NEUTRAL", f"Failed on delta {delta}: state={debug.scroll_state}"
        assert action == GestureAction.NONE

        # Verify cursor is allowed to move
        pos = controller.update_cursor_position(
            pose[0][0] + 0.03,  # intentional X movement
            pose[0][1],
            timestamp=1.0 + i * 0.033,
            is_scrolling=(debug.scroll_state != "NEUTRAL"),
        )
        assert pos is not None
        assert controller.last_mouse_suppressed is False


def test_sustained_intentional_head_tilt_activates_scrolling_appropriately():
    """Intentional head tilt beyond threshold activates scrolling and freezes cursor."""
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, scroll_interval_s=0.10)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    tilt_pose = make_head_pose(pitch_delta=+0.028)
    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=tilt_pose[0],
        mid_eyes=tilt_pose[1],
        chin=tilt_pose[2],
        is_calibrated=True,
        current_time=1.0,
    )
    assert debug.scroll_state == "DOWN"
    assert action == GestureAction.SCROLL_DOWN

    # Cursor must be suppressed while scrolling
    pos = controller.update_cursor_position(
        tilt_pose[0][0] + 0.05,
        tilt_pose[0][1],
        timestamp=1.0,
        is_scrolling=(debug.scroll_state != "NEUTRAL"),
    )
    assert pos is None
    assert controller.last_mouse_suppressed is True
    assert controller.last_mouse_suppress_reason == "SCROLLING"


def test_continuous_tilt_does_not_repeatedly_retrigger_or_bounce_scrolling():
    """Holding head in a tilted position maintains scroll state without bouncing to NEUTRAL."""
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, scroll_interval_s=0.10)
    gestures = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    gestures.calibrate_pitch(*neutral_pose)

    # Sustained downward tilt over 30 frames (approx 1 second)
    down_pose = make_head_pose(pitch_delta=+0.026)
    scroll_action_count = 0
    neutral_drop_count = 0

    for i in range(30):
        t = 1.0 + i * 0.033
        action, _, _, _, debug = gestures.update(
            face_detected=True,
            left_eye=open_eye,
            right_eye=open_eye,
            nose_tip=down_pose[0],
            mid_eyes=down_pose[1],
            chin=down_pose[2],
            is_calibrated=True,
            current_time=t,
        )
        if debug.scroll_state == "NEUTRAL":
            neutral_drop_count += 1
        assert debug.scroll_state == "DOWN", f"Frame {i}: dropped to {debug.scroll_state}!"
        if action == GestureAction.SCROLL_DOWN:
            scroll_action_count += 1

    assert neutral_drop_count == 0, "Scroll state bounced to NEUTRAL while head was still tilted!"
    # Over 1.0s with interval 0.10s, should have triggered ~10 scroll actions
    assert 8 <= scroll_action_count <= 11


def test_returning_to_neutral_releases_scrolling_promptly_and_restores_cursor():
    """When returning toward neutral, scroll state releases promptly and cursor movement resumes immediately."""
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, scroll_release_ratio=0.60)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # 1. Tilt DOWN
    down_pose = make_head_pose(pitch_delta=+0.030)
    gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        is_calibrated=True,
        current_time=1.0,
    )
    assert gestures.scroll_state == ScrollState.DOWN

    # 2. Return to neutral on the very next frame
    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=neutral_pose[0],
        mid_eyes=neutral_pose[1],
        chin=neutral_pose[2],
        is_calibrated=True,
        current_time=1.033,
    )
    assert debug.scroll_state == "NEUTRAL", "Scroll failed to release promptly on return to neutral!"
    assert action == GestureAction.NONE

    # 3. Cursor movement must resume immediately without lag
    pos = controller.update_cursor_position(
        neutral_pose[0][0] + 0.04,
        neutral_pose[0][1],
        timestamp=1.033,
        is_scrolling=(debug.scroll_state != "NEUTRAL"),
    )
    assert pos is not None
    assert controller.last_mouse_suppressed is False
    assert controller.last_mouse_moved is True


def test_stationary_head_in_neutral_deadzone_produces_zero_mouse_moves_at_rest(mock_pyautogui):
    """Stationary head at neutral with micro-tremor within deadzone produces 0 moveTo calls (AT_REST)."""
    cfg = AppConfig(
        dead_zone_radius=0.015,
        cursor_min_movement_px=2.0,
        enable_vertical_cursor=True,
    )
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    # First frame to establish baseline
    controller.update_cursor_position(0.50, 0.50, timestamp=1.0, is_scrolling=False)
    mock_pyautogui.reset_mock()

    # Small micro-fluctuations inside deadzone (<= 0.015)
    micro_jitters = [
        (0.501, 0.502),
        (0.499, 0.501),
        (0.502, 0.498),
        (0.500, 0.501),
        (0.498, 0.500),
    ]

    for i, (x, y) in enumerate(micro_jitters):
        t = 1.0 + (i + 1) * 0.033
        controller.update_cursor_position(x, y, timestamp=t, is_scrolling=False)
        assert controller.last_mouse_moved is False
        assert controller.last_mouse_suppress_reason == "AT_REST"

    # PyAutoGUI moveTo must not have been invoked
    mock_pyautogui.assert_not_called()


def test_deliberate_small_head_movements_outside_deadzone_remain_responsive(mock_pyautogui):
    """Deliberate head movement beyond deadzone ramps smoothly and moves cursor."""
    cfg = AppConfig(
        dead_zone_radius=0.015,
        cursor_min_movement_px=2.0,
        sensitivity_x=2.5,
        enable_vertical_cursor=True,
    )
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    # Establish baseline
    controller.update_cursor_position(0.50, 0.50, timestamp=1.0, is_scrolling=False)
    mock_pyautogui.reset_mock()

    # Move head deliberately to dx = +0.025 (well outside 0.015 deadzone)
    target_pos = controller.update_cursor_position(0.525, 0.50, timestamp=1.05, is_scrolling=False)

    assert target_pos is not None
    # Cursor should have moved to the right of center
    assert target_pos[0] > controller.screen_center_x
    assert controller.last_mouse_moved is True
    assert controller.last_mouse_suppressed is False
    mock_pyautogui.assert_called_once()


def test_neutral_rearm_prevents_retriggering_at_hysteresis_boundary():
    """If head pitch drops just below release threshold but does not visit neutral deadband,
    re-triggering is blocked to prevent hunting / rapid bouncing.
    """
    cfg = AppConfig(
        scroll_enabled=True,
        scroll_pitch_threshold=0.020,
        scroll_release_ratio=0.60,  # release at 0.012
    )
    gestures = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    gestures.calibrate_pitch(*neutral_pose)

    # 1. Trigger SCROLL_DOWN
    down_pose = make_head_pose(pitch_delta=+0.025)
    gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        is_calibrated=True,
        current_time=1.0,
    )
    assert gestures.scroll_state == ScrollState.DOWN

    # 2. Drop pitch slightly below release threshold (+0.011 < 0.012),
    # but still outside neutral deadband (+0.008)
    release_pose = make_head_pose(pitch_delta=+0.011)
    _, _, _, _, debug2 = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=release_pose[0],
        mid_eyes=release_pose[1],
        chin=release_pose[2],
        is_calibrated=True,
        current_time=1.05,
    )
    assert debug2.scroll_state == "NEUTRAL"
    assert not gestures._scroll_rearmed  # Rearm guard active!

    # 3. Fluctuate back up to +0.022 without visiting deadband
    _, _, _, _, debug3 = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        is_calibrated=True,
        current_time=1.10,
    )
    # Must NOT re-trigger DOWN because neutral deadband was not visited!
    assert debug3.scroll_state == "NEUTRAL"
    assert "AWAITING_NEUTRAL_RETURN" in debug3.scroll_transition_reason

    # 4. Visit neutral deadband (+0.003 <= 0.008)
    near_zero_pose = make_head_pose(pitch_delta=+0.003)
    _, _, _, _, debug4 = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=near_zero_pose[0],
        mid_eyes=near_zero_pose[1],
        chin=near_zero_pose[2],
        is_calibrated=True,
        current_time=1.15,
    )
    assert debug4.scroll_state == "NEUTRAL"
    assert gestures._scroll_rearmed  # Now rearmed!

    # 5. Intentional tilt again (+0.025) cleanly triggers DOWN
    action5, _, _, _, debug5 = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        is_calibrated=True,
        current_time=1.20,
    )
    assert debug5.scroll_state == "DOWN"
    assert action5 == GestureAction.SCROLL_DOWN

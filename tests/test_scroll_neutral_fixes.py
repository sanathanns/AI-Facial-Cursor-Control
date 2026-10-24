"""Regression tests verifying scroll-neutral calibration, pitch-cursor decoupling,
immediate stop on return to neutral, and suppression telemetry accuracy.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureRecognizer, GestureAction, ScrollState, calculate_head_pitch_ratio
from tests.test_gestures import make_eye_landmarks, make_head_pose


@pytest.fixture(autouse=True)
def mock_pyautogui():
    with patch("pyautogui.moveTo") as m_move, patch("pyautogui.click") as m_click, patch("pyautogui.scroll") as m_scroll:
        yield m_move


# Scenario 1: Sitting still after calibration
def test_scenario_1_sitting_still_stays_neutral_allows_cursor():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Simulate sitting still over multiple frames
    for i in range(5):
        action, _, _, _, debug = gestures.update(
            face_detected=True,
            left_eye=open_eye,
            right_eye=open_eye,
            nose_tip=neutral_pose[0],
            mid_eyes=neutral_pose[1],
            chin=neutral_pose[2],
            is_calibrated=True,
            current_time=1.0 + i * 0.033,
        )
        assert abs(debug.head_pitch) < 0.001
        assert debug.scroll_state == "NEUTRAL"
        assert action == GestureAction.NONE

        is_scrolling = (debug.scroll_state != "NEUTRAL")
        # Moving slightly beyond deadzone permits cursor movement
        pos = controller.update_cursor_position(
            neutral_pose[0][0] + 0.025,
            neutral_pose[0][1],
            timestamp=1.0 + i * 0.033,
            is_scrolling=is_scrolling,
        )
        assert pos is not None
        assert controller.last_mouse_suppressed is False


# Scenario 2: Normal horizontal head movement does NOT alter pitch delta or trigger scroll
def test_scenario_2_horizontal_movement_no_pitch_crosstalk_and_moves_cursor():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Shift head far to the right (simulating target_x = 2091 on a 1920 screen)
    shift_x = 0.08  # Substantial horizontal displacement
    right_nose = (neutral_pose[0][0] + shift_x, neutral_pose[0][1])
    right_eyes = (neutral_pose[1][0] + shift_x, neutral_pose[1][1])
    right_chin = (neutral_pose[2][0] + shift_x, neutral_pose[2][1])

    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=right_nose,
        mid_eyes=right_eyes,
        chin=right_chin,
        is_calibrated=True,
        current_time=1.0,
    )
    # Vector projection ensures pitch delta is invariant to horizontal shift
    assert abs(debug.head_pitch) < 0.001
    assert debug.scroll_state == "NEUTRAL"
    assert action == GestureAction.NONE

    # Cursor moves right freely without being suppressed by scroll
    is_scrolling = (debug.scroll_state != "NEUTRAL")
    with patch("pyautogui.moveTo") as mock_move:
        pos = controller.update_cursor_position(
            right_nose[0], right_nose[1], timestamp=1.0, is_scrolling=is_scrolling
        )
        assert pos is not None
        assert pos[0] > controller.screen_center_x
        assert controller.last_mouse_moved is True
        assert controller.last_mouse_suppressed is False


# Scenario 3: Normal vertical head movement controls cursor and does not trigger scrolling unless intentional
def test_scenario_3_normal_vertical_cursor_does_not_trigger_scroll():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, enable_vertical_cursor=True)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Translation down (whole head shifts down without angular pitch tilt)
    shift_y = 0.020
    trans_nose = (neutral_pose[0][0], neutral_pose[0][1] + shift_y)
    trans_eyes = (neutral_pose[1][0], neutral_pose[1][1] + shift_y)
    trans_chin = (neutral_pose[2][0], neutral_pose[2][1] + shift_y)

    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=trans_nose,
        mid_eyes=trans_eyes,
        chin=trans_chin,
        is_calibrated=True,
        current_time=1.0,
    )
    # Head translation does not change the ratio of nose to chin-eyes span
    assert abs(debug.head_pitch) < 0.001
    assert debug.scroll_state == "NEUTRAL"
    assert action == GestureAction.NONE

    is_scrolling = (debug.scroll_state != "NEUTRAL")
    pos = controller.update_cursor_position(
        trans_nose[0], trans_nose[1], timestamp=1.0, is_scrolling=is_scrolling
    )
    assert pos is not None
    assert pos[1] > controller.screen_center_y
    assert controller.last_mouse_suppressed is False


# Scenario 4: Pitch down gesture triggers SCROLL DOWN and freezes cursor
def test_scenario_4_pitch_down_triggers_scroll_and_freezes_cursor():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Intentional angular pitch down (+0.030 > +0.020)
    down_pose = make_head_pose(pitch_delta=0.030)
    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        is_calibrated=True,
        current_time=1.0,
    )
    assert debug.scroll_state == "DOWN"
    assert action == GestureAction.SCROLL_DOWN

    # Cursor freezes during active scroll
    is_scrolling = (debug.scroll_state != "NEUTRAL")
    pos = controller.update_cursor_position(
        down_pose[0][0] + 0.05, down_pose[0][1], timestamp=1.0, is_scrolling=is_scrolling
    )
    assert pos is None
    assert controller.last_mouse_moved is False
    assert controller.last_mouse_suppressed is True
    assert controller.last_mouse_suppress_reason == "SCROLLING"


# Scenario 5: Returning to neutral from scroll down stops scrolling and resumes cursor immediately
def test_scenario_5_return_to_neutral_resumes_cursor_immediately():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # 1. Pitch DOWN
    down_pose = make_head_pose(pitch_delta=0.030)
    gestures.update(True, open_eye, open_eye, down_pose[0], down_pose[1], down_pose[2], is_calibrated=True, current_time=1.0)
    assert gestures.scroll_state == ScrollState.DOWN

    # 2. Return to neutral (pitch delta = 0.0)
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
    assert debug.scroll_state == "NEUTRAL"
    assert action == GestureAction.NONE

    # Cursor movement must resume immediately without freeze
    is_scrolling = (debug.scroll_state != "NEUTRAL")
    pos = controller.update_cursor_position(
        neutral_pose[0][0] + 0.03, neutral_pose[0][1], timestamp=1.033, is_scrolling=is_scrolling
    )
    assert pos is not None
    assert controller.last_mouse_moved is True
    assert controller.last_mouse_suppressed is False


# Scenario 6: Pitch up gesture triggers SCROLL UP and freezes cursor
def test_scenario_6_pitch_up_triggers_scroll_and_freezes_cursor():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # Intentional angular pitch up (-0.030 < -0.020)
    up_pose = make_head_pose(pitch_delta=-0.030)
    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=up_pose[0],
        mid_eyes=up_pose[1],
        chin=up_pose[2],
        is_calibrated=True,
        current_time=1.0,
    )
    assert debug.scroll_state == "UP"
    assert action == GestureAction.SCROLL_UP

    is_scrolling = (debug.scroll_state != "NEUTRAL")
    pos = controller.update_cursor_position(
        up_pose[0][0] + 0.05, up_pose[0][1], timestamp=1.0, is_scrolling=is_scrolling
    )
    assert pos is None
    assert controller.last_mouse_suppressed is True
    assert controller.last_mouse_suppress_reason == "SCROLLING"


# Scenario 7: Returning to neutral from scroll up stops scrolling and resumes cursor immediately
def test_scenario_7_return_from_up_resumes_cursor_immediately():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020)
    gestures = GestureRecognizer(cfg)
    controller = CursorController(cfg)
    open_eye = make_eye_landmarks()

    neutral_pose = make_head_pose(pitch_delta=0.0)
    controller.calibrate(neutral_pose[0][0], neutral_pose[0][1])
    gestures.calibrate_pitch(*neutral_pose)

    # 1. Pitch UP
    up_pose = make_head_pose(pitch_delta=-0.030)
    gestures.update(True, open_eye, open_eye, up_pose[0], up_pose[1], up_pose[2], is_calibrated=True, current_time=1.0)
    assert gestures.scroll_state == ScrollState.UP

    # 2. Return to neutral
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
    assert debug.scroll_state == "NEUTRAL"
    assert action == GestureAction.NONE

    is_scrolling = (debug.scroll_state != "NEUTRAL")
    pos = controller.update_cursor_position(
        neutral_pose[0][0] + 0.03, neutral_pose[0][1], timestamp=1.033, is_scrolling=is_scrolling
    )
    assert pos is not None
    assert controller.last_mouse_moved is True
    assert controller.last_mouse_suppressed is False


# Scenario 8: Eye winks while head is neutral trigger click without altering scroll state
def test_scenario_8_eye_wink_does_not_affect_scroll_state():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, wink_min_duration_s=0.10)
    gestures = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks(height=0.03)
    closed_eye = make_eye_landmarks(height=0.005)

    neutral_pose = make_head_pose(pitch_delta=0.0)
    gestures.calibrate_pitch(*neutral_pose)
    from src.gestures import calculate_ear
    gestures.calibrate_eyes(calculate_ear(open_eye), calculate_ear(open_eye))

    # Hold left wink for 120ms
    gestures.update(True, closed_eye, open_eye, neutral_pose[0], neutral_pose[1], neutral_pose[2], is_calibrated=True, current_time=1.00)
    action, _, _, _, debug = gestures.update(
        face_detected=True,
        left_eye=closed_eye,
        right_eye=open_eye,
        nose_tip=neutral_pose[0],
        mid_eyes=neutral_pose[1],
        chin=neutral_pose[2],
        is_calibrated=True,
        current_time=1.12,
    )
    assert action == GestureAction.LEFT_CLICK
    assert debug.scroll_state == "NEUTRAL"
    assert abs(debug.head_pitch) < 0.001


# Scenario 9: Distinguishing suppressed mouse command from resting
def test_scenario_9_distinguish_suppressed_from_resting():
    cfg = AppConfig(dead_zone_radius=0.010, cursor_min_movement_px=2.0)
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    # 1. At rest inside deadzone -> not suppressed, but resting
    controller.update_cursor_position(0.502, 0.502, timestamp=1.0, is_scrolling=False)
    tel = controller.get_telemetry()
    assert tel["mouse_moved"] is False
    assert tel["mouse_suppressed"] is False
    assert tel["mouse_suppress_reason"] == "AT_REST"

    # 2. Actively scrolling -> suppressed with reason SCROLLING
    controller.update_cursor_position(0.550, 0.50, timestamp=1.05, is_scrolling=True)
    tel = controller.get_telemetry()
    assert tel["mouse_moved"] is False
    assert tel["mouse_suppressed"] is True
    assert tel["mouse_suppress_reason"] == "SCROLLING"

    # 3. Paused -> suppressed with reason PAUSED
    controller.toggle_pause()
    controller.update_cursor_position(0.550, 0.50, timestamp=1.10, is_scrolling=False)
    tel = controller.get_telemetry()
    assert tel["mouse_moved"] is False
    assert tel["mouse_suppressed"] is True
    assert tel["mouse_suppress_reason"] == "PAUSED"
    controller.toggle_pause()

    # 4. Emergency stop -> suppressed with reason EMERGENCY_STOP
    controller.trigger_emergency_stop()
    controller.update_cursor_position(0.550, 0.50, timestamp=1.15, is_scrolling=False)
    tel = controller.get_telemetry()
    assert tel["mouse_moved"] is False
    assert tel["mouse_suppressed"] is True
    assert tel["mouse_suppress_reason"] == "EMERGENCY_STOP"
    controller.resume_from_emergency_stop()

    # 5. Face lost -> suppressed with reason NO_FACE
    controller.on_face_lost()
    tel = controller.get_telemetry()
    assert tel["mouse_moved"] is False
    assert tel["mouse_suppressed"] is True
    assert tel["mouse_suppress_reason"] == "NO_FACE"


# Scenario 10: Multi-frame stable calibration rejects startup outlier and prevents overwrite while moving
def test_scenario_10_multi_frame_calibration_rejects_outlier_and_prevents_overwrite():
    import statistics
    cfg = AppConfig()
    controller = CursorController(cfg)
    gestures = GestureRecognizer(cfg)

    # Frame 1 has transient downward jitter (pitch ratio 0.437)
    # Frames 2..10 have stable neutral pitch (0.400)
    samples_pitch = [0.437] + [0.400] * 9
    samples_x = [0.450] + [0.500] * 9
    samples_y = [0.550] + [0.500] * 9

    med_pitch = float(statistics.median(samples_pitch))
    med_x = float(statistics.median(samples_x))
    med_y = float(statistics.median(samples_y))

    # Median perfectly filters out the single-frame outlier
    assert med_pitch == 0.400
    assert med_x == 0.500
    assert med_y == 0.500

    controller.calibrate(med_x, med_y)
    gestures.neutral_pitch = med_pitch
    gestures.is_pitch_calibrated = True

    assert controller.is_calibrated is True
    assert gestures.is_pitch_calibrated is True
    assert gestures.neutral_pitch == 0.400

    # Moving head across 5 frames must NEVER overwrite neutral baselines
    for step in range(5):
        move_x = 0.500 + step * 0.015
        move_y = 0.500 + step * 0.010
        controller.update_cursor_position(move_x, move_y, timestamp=1.0 + step * 0.033, is_scrolling=False)
        assert controller.neutral_x == 0.500
        assert controller.neutral_y == 0.500
        assert gestures.neutral_pitch == 0.400


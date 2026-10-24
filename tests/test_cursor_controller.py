"""Unit tests for cursor controller logic, calibration, and safety."""
import pytest
from unittest.mock import patch, MagicMock
import pyautogui
from src.config import AppConfig
from src.cursor_controller import CursorController
from src.gestures import GestureAction


def test_cursor_controller_initialization():
    cfg = AppConfig()
    controller = CursorController(cfg)
    assert not controller.is_calibrated
    assert controller.neutral_x is None
    assert not controller.is_paused
    assert not controller.is_emergency_stopped


def test_cursor_controller_calibration():
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.calibrate(0.48, 0.52)
    assert controller.is_calibrated
    assert controller.neutral_x == 0.48
    assert controller.neutral_y == 0.52

    controller.reset_neutral()
    assert not controller.is_calibrated
    assert controller.neutral_x is None


def test_cursor_controller_pause_and_emergency_stop():
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.calibrate(0.5, 0.5)

    # When paused, update returns None
    controller.toggle_pause()
    assert controller.is_paused
    res = controller.update_cursor_position(0.55, 0.55, timestamp=1.0)
    assert res is None

    # Resume
    controller.toggle_pause()
    assert not controller.is_paused

    # When emergency stopped, update returns None
    controller.trigger_emergency_stop()
    assert controller.is_emergency_stopped
    res = controller.update_cursor_position(0.55, 0.55, timestamp=1.1)
    assert res is None

    controller.resume_from_emergency_stop()
    assert not controller.is_emergency_stopped


def test_cursor_controller_movement_within_screen_bounds():
    cfg = AppConfig(sensitivity_x=2.0, sensitivity_y=2.0, dead_zone_radius=0.01)
    controller = CursorController(cfg)
    controller.calibrate(0.5, 0.5)

    with patch("pyautogui.moveTo") as mock_move:
        # Move right
        pos = controller.update_cursor_position(0.60, 0.50, timestamp=1.0)
        assert pos is not None
        assert 0 <= pos[0] < controller.screen_width
        assert 0 <= pos[1] < controller.screen_height
        mock_move.assert_called_once()


def test_cursor_controller_actions():
    cfg = AppConfig()
    controller = CursorController(cfg)

    with patch("pyautogui.click") as mock_click, patch("pyautogui.scroll") as mock_scroll:
        controller.execute_action(GestureAction.LEFT_CLICK)
        mock_click.assert_called_with(button="left")

        controller.execute_action(GestureAction.RIGHT_CLICK)
        mock_click.assert_called_with(button="right")

        controller.execute_action(GestureAction.SCROLL_UP)
        mock_scroll.assert_called_with(cfg.scroll_speed)

        controller.execute_action(GestureAction.SCROLL_DOWN)
        mock_scroll.assert_called_with(-cfg.scroll_speed)


def test_cursor_controller_calibration_bounds():
    cfg = AppConfig()
    controller = CursorController(cfg)

    # Valid coordinates
    assert controller.calibrate(0.5, 0.5) is True
    # Out of bounds
    assert controller.calibrate(-0.1, 0.5) is False
    assert controller.calibrate(0.5, 1.2) is False
    # NaN
    assert controller.calibrate(float("nan"), 0.5) is False


def test_cursor_controller_failsafe_triggers_emergency_stop():
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.calibrate(0.5, 0.5)

    with patch("pyautogui.moveTo", side_effect=pyautogui.FailSafeException("Corner hit")):
        controller.update_cursor_position(0.6, 0.6, timestamp=1.0)
        assert controller.is_emergency_stopped is True


def test_cursor_controller_scrolling_freezes_cursor():
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.calibrate(0.5, 0.5)

    with patch("pyautogui.moveTo") as mock_move:
        # While scrolling, cursor must NOT move
        res = controller.update_cursor_position(0.6, 0.6, timestamp=1.0, is_scrolling=True)
        assert res is None
        mock_move.assert_not_called()


def test_cursor_controller_face_loss_cleans_state():
    cfg = AppConfig()
    controller = CursorController(cfg)
    controller.is_dragging = True

    with patch("pyautogui.mouseUp") as mock_up:
        controller.on_face_lost()
        assert controller.is_dragging is False
        mock_up.assert_called_once_with(button="left")


def test_cursor_controller_telemetry_lifecycle():
    """Verify that all telemetry fields exist from initialization and update consistently."""
    cfg = AppConfig()
    controller = CursorController(cfg)

    # 1. Guaranteed safe defaults at initialization (before any frames)
    assert hasattr(controller, "last_raw_dx")
    assert hasattr(controller, "last_raw_dy")
    assert hasattr(controller, "last_calibrated_dx")
    assert hasattr(controller, "last_calibrated_dy")
    assert hasattr(controller, "last_target_x")
    assert hasattr(controller, "last_target_y")
    assert hasattr(controller, "last_filtered_x")
    assert hasattr(controller, "last_filtered_y")
    assert hasattr(controller, "last_mouse_moved")

    assert controller.last_raw_dx == 0.0
    assert controller.last_raw_dy == 0.0
    assert controller.last_calibrated_dx == 0.0
    assert controller.last_calibrated_dy == 0.0
    assert controller.last_mouse_moved is False

    telemetry = controller.get_telemetry()
    assert isinstance(telemetry, dict)
    assert telemetry["raw_dx"] == 0.0
    assert telemetry["raw_dy"] == 0.0
    assert telemetry["calibrated_dx"] == 0.0
    assert telemetry["calibrated_dy"] == 0.0
    assert telemetry["mouse_moved"] is False
    assert "target_x" in telemetry
    assert "target_y" in telemetry
    assert "filtered_x" in telemetry
    assert "filtered_y" in telemetry
    assert "cursor_pos" in telemetry

    # 2. Update during cursor movement
    controller.calibrate(0.50, 0.50)
    with patch("pyautogui.moveTo"):
        controller.update_cursor_position(0.54, 0.46, timestamp=1.0)
        assert controller.last_raw_dx == pytest.approx(0.04, abs=1e-4)
        assert controller.last_raw_dy == pytest.approx(-0.04, abs=1e-4)
        assert controller.last_calibrated_dx > 0.0
        assert controller.last_calibrated_dy < 0.0
        assert controller.last_mouse_moved is True

    # 3. Reset on face loss
    controller.on_face_lost()
    assert controller.last_raw_dx == 0.0
    assert controller.last_raw_dy == 0.0
    assert controller.last_calibrated_dx == 0.0
    assert controller.last_calibrated_dy == 0.0
    assert controller.last_mouse_moved is False


def test_left_head_movement_moves_cursor_left():
    """Requirement 1: Head movement left produces cursor movement to the left."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos = controller.update_cursor_position(0.47, 0.50, timestamp=1.0)
        assert pos is not None
        assert pos[0] < controller.screen_center_x
        assert controller.last_raw_dx < 0
        assert controller.last_calibrated_dx < 0


def test_right_head_movement_moves_cursor_right():
    """Requirement 2: Head movement right produces cursor movement to the right."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos = controller.update_cursor_position(0.53, 0.50, timestamp=1.0)
        assert pos is not None
        assert pos[0] > controller.screen_center_x
        assert controller.last_raw_dx > 0
        assert controller.last_calibrated_dx > 0


def test_upward_head_movement_moves_cursor_up():
    """Requirement 3: Head movement up produces cursor movement upward (smaller y)."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos = controller.update_cursor_position(0.50, 0.47, timestamp=1.0)
        assert pos is not None
        assert pos[1] < controller.screen_center_y
        assert controller.last_raw_dy < 0
        assert controller.last_calibrated_dy < 0


def test_downward_head_movement_moves_cursor_down():
    """Requirement 4: Head movement down produces cursor movement downward (larger y)."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos = controller.update_cursor_position(0.50, 0.53, timestamp=1.0)
        assert pos is not None
        assert pos[1] > controller.screen_center_y
        assert controller.last_raw_dy > 0
        assert controller.last_calibrated_dy > 0


def test_small_movements_cause_measurable_cursor_movement():
    """Requirement 5: Small comfortable head movements produce measurable cursor movement."""
    cfg = AppConfig(
        sensitivity_x=2.5,
        sensitivity_y=2.5,
        comfort_range_x=0.045,
        comfort_range_y=0.045,
        dead_zone_radius=0.008,
        enable_vertical_cursor=True,
    )
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos = controller.update_cursor_position(0.520, 0.480, timestamp=1.0)
        assert pos is not None
        assert abs(pos[0] - controller.screen_center_x) > 150
        assert abs(pos[1] - controller.screen_center_y) > 150


def test_neutral_posture_does_not_cause_drift():
    """Requirement 6: Neutral posture inside dead zone produces zero resting drift."""
    cfg = AppConfig(dead_zone_radius=0.010, enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo") as mock_move:
        pos = controller.update_cursor_position(0.50, 0.50, timestamp=1.0)
        assert pos == (int(controller.screen_center_x), int(controller.screen_center_y))

        mock_move.reset_mock()
        pos_noise = controller.update_cursor_position(0.504, 0.496, timestamp=1.033)
        assert pos_noise == pos
        assert controller.last_calibrated_dx == 0.0
        assert controller.last_calibrated_dy == 0.0
        assert controller.last_mouse_moved is False
        mock_move.assert_not_called()


def test_cursor_remains_clamped_within_screen_bounds():
    """Requirement 7: Extreme head displacements remain safely clamped within screen bounds."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo"):
        pos_tl = controller.update_cursor_position(0.10, 0.10, timestamp=1.0)
        assert pos_tl is not None
        assert 3 <= pos_tl[0] < controller.screen_width
        assert 3 <= pos_tl[1] < controller.screen_height

        pos_br = controller.update_cursor_position(0.90, 0.90, timestamp=1.033)
        assert pos_br is not None
        assert 3 <= pos_br[0] <= controller.screen_width - 4
        assert 3 <= pos_br[1] <= controller.screen_height - 4


def test_cursor_movement_resumes_immediately_after_scrolling_ends():
    """Requirement 8: Cursor movement resumes immediately and smoothly after scrolling stops."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo") as mock_move:
        res1 = controller.update_cursor_position(0.53, 0.53, timestamp=1.0, is_scrolling=True)
        assert res1 is None
        mock_move.assert_not_called()

        res2 = controller.update_cursor_position(0.53, 0.53, timestamp=1.05, is_scrolling=False)
        assert res2 is not None
        assert controller.last_mouse_moved is True
        mock_move.assert_called_once()


def test_head_pitch_scrolling_works_in_both_up_and_down_directions():
    """Requirement 9: Head-pitch scrolling triggers both SCROLL_UP and SCROLL_DOWN without mode switch."""
    from src.gestures import GestureRecognizer, GestureAction
    from tests.test_gestures import make_eye_landmarks, make_head_pose

    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.020, scroll_interval_s=0.10)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()
    neutral_pose = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(*neutral_pose)

    # 1. Tilt UP (negative pitch delta)
    up_pose = make_head_pose(pitch_delta=-0.025)
    action_up, _, _, _, debug_up = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=up_pose[0],
        mid_eyes=up_pose[1],
        chin=up_pose[2],
        current_time=1.0,
    )
    assert action_up == GestureAction.SCROLL_UP
    assert debug_up.scroll_state == "UP"

    # Return to neutral
    recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=neutral_pose[0],
        mid_eyes=neutral_pose[1],
        chin=neutral_pose[2],
        current_time=1.10,
    )

    # 2. Tilt DOWN (positive pitch delta)
    down_pose = make_head_pose(pitch_delta=+0.025)
    action_down, _, _, _, debug_down = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=down_pose[0],
        mid_eyes=down_pose[1],
        chin=down_pose[2],
        current_time=1.25,
    )
    assert action_down == GestureAction.SCROLL_DOWN
    assert debug_down.scroll_state == "DOWN"


def test_pause_and_emergency_stop_block_movement():
    """Requirement 10: Pause and Emergency Stop immediately block all cursor movement."""
    cfg = AppConfig(enable_vertical_cursor=True)
    controller = CursorController(cfg)
    controller.calibrate(0.50, 0.50)

    with patch("pyautogui.moveTo") as mock_move:
        controller.toggle_pause()
        assert controller.is_paused is True
        assert controller.update_cursor_position(0.55, 0.55, timestamp=1.0) is None
        mock_move.assert_not_called()

        controller.toggle_pause()
        assert controller.is_paused is False

        controller.trigger_emergency_stop()
        assert controller.is_emergency_stopped is True
        assert controller.update_cursor_position(0.55, 0.55, timestamp=1.1) is None
        mock_move.assert_not_called()

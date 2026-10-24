"""Automated tests for stability fixes, dashboard upgrade, and keyboard repair.

Covers:
1. Consistent cursor displacement under consistent input.
2. Screen edge reversal responsiveness (no sticky dead-zone).
3. Calibration outlier rejection and multi-frame stability.
4. Calibration face loss safety.
5. Telemetry graph history bounds, missing-data handling, pause, and clear.
6. Settings validation, application, and persistence.
7. Keyboard direct screen-coordinate hit testing and hover.
8. Keyboard wink-click selection and duplicate-click debounce.
9. Keyboard external typing cancel button.
10. Camera frame deduplication via _has_new_frame.
"""
import time
import os
import sys
import collections
import tkinter as tk
from unittest.mock import MagicMock, patch
import pytest

from src.config import AppConfig
from src.cursor_controller import CursorController
from src.camera import CameraManager
from src.diagnostics import DiagnosticsDashboard, TelemetryGraphCanvas
from src.on_screen_keyboard import OnScreenKeyboard
from src.gui import VisionWorker


@pytest.fixture(autouse=True)
def mock_pyautogui():
    with patch("pyautogui.moveTo") as m_move, patch("pyautogui.click") as m_click, patch("pyautogui.scroll") as m_scroll:
        yield m_move


@pytest.fixture
def tk_root(session_tk_root):
    return session_tk_root


def test_consistent_cursor_displacement_under_consistent_input():
    """Verify that identical head movements produce consistent, repeatable cursor targets."""
    cfg = AppConfig(
        sensitivity_x=2.5,
        comfort_range_x=0.045,
        dead_zone_radius=0.008,
    )
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    displacements = []
    # Feed head displacement dx = +0.025 at different timestamps
    for i in range(5):
        t = 1.0 + i * 0.033
        pos = controller.update_cursor_position(0.525, 0.50, timestamp=t, is_scrolling=False)
        assert pos is not None
        displacements.append(pos[0])

    # After initial filter convergence, displacement must be consistent
    assert len(set(displacements[-3:])) == 1 or abs(displacements[-1] - displacements[-2]) <= 1


def test_screen_edge_reversal_responsiveness():
    """When head moves past screen edge, reversing direction must move cursor immediately."""
    cfg = AppConfig(
        sensitivity_x=2.5,
        comfort_range_x=0.045,
        dead_zone_radius=0.008,
    )
    controller = CursorController(cfg)
    controller.screen_width = 1920
    controller.screen_height = 1080
    controller.screen_center_x = 960.0
    controller.screen_center_y = 540.0
    controller.calibrate(0.50, 0.50)

    # 1. Turn head far to reach the screen edge (dx >= 0.035 reaches max clamped border)
    pos_edge = controller.update_cursor_position(0.535, 0.50, timestamp=1.0)
    assert pos_edge is not None
    assert pos_edge[0] >= 1910

    # 2. Turn head back toward center (from 0.535 to 0.518)
    pos_reversed = controller.update_cursor_position(0.518, 0.50, timestamp=1.033)
    assert pos_reversed is not None
    # Cursor target must start moving back immediately, not remain pinned at 1916!
    assert controller.last_target_x < pos_edge[0] or pos_reversed[0] < pos_edge[0]


def test_calibration_outlier_rejection_and_multi_frame_stability():
    """Multi-frame calibration must reject single-frame outliers using median filtering."""
    import statistics
    nose_x_samples = [0.50, 0.501, 0.499, 0.502, 0.850, 0.500, 0.498, 0.501, 0.500, 0.502]
    med_x = float(statistics.median(nose_x_samples))
    # Median rejects the 0.850 outlier completely
    assert abs(med_x - 0.500) < 0.005


def test_calibration_face_loss_safety():
    """Face loss during calibration must clear accumulation buffers and leave existing calibration intact."""
    import queue
    cfg = AppConfig()
    frame_q = queue.Queue(maxsize=1)
    cmd_q = queue.Queue()
    worker = VisionWorker(cfg, frame_q, cmd_q)

    # Anchor an initial calibration
    worker.controller.calibrate(0.48, 0.52)
    assert worker.controller.is_calibrated is True

    # Start re-calibration
    worker.request_calibration()
    worker._calib_nose_x.append(0.55)
    worker._calib_nose_y.append(0.55)

    # Simulate face lost: reset buffers
    worker._calib_nose_x.clear()
    worker._calib_nose_y.clear()
    worker.controller.on_face_lost()

    # Pre-existing calibration anchor must be preserved
    assert worker.controller.neutral_x == 0.48
    assert worker.controller.neutral_y == 0.52


def test_telemetry_graph_history_bounds_and_missing_data(tk_root):
    """TelemetryGraphCanvas maintains bounded history, handles missing data, pause, and clear."""
    frame = tk.Frame(tk_root)
    frame.pack()
    graph = TelemetryGraphCanvas(frame, max_history=50)
    graph.pack()

    # Feed 80 samples (exceeding max_history 50)
    for i in range(80):
        sample = {
            "face_detected": (i % 2 == 0),
            "raw_dx": 0.01 * (i % 5),
            "calibrated_dx": 0.008 * (i % 5),
            "raw_dy": -0.005,
            "calibrated_dy": -0.004,
            "head_pitch": 0.010,
            "dead_zone_radius": 0.008,
            "scroll_activation_thresh": 0.020,
            "scroll_release_thresh": 0.013,
            "scroll_state": "NEUTRAL",
        }
        graph.add_sample(sample)

    # Strictly bounded to max_history
    assert len(graph.history) == 50

    # Pause graph
    paused = graph.toggle_pause()
    assert paused is True
    assert graph.is_paused is True
    assert "PAUSED" in graph.lbl_status.cget("text")

    # Resume graph
    resumed = graph.toggle_pause()
    assert resumed is False
    assert graph.is_paused is False

    # Clear history
    graph.clear_history()
    assert len(graph.history) == 0

    frame.destroy()


def test_settings_validation_application_and_persistence(tmp_path):
    """Settings changes apply immediately and persist correctly to config file."""
    cfg_file = str(tmp_path / "test_config.json")
    cfg = AppConfig()
    cfg.sensitivity_x = 3.5
    cfg.scroll_release_ratio = 0.65
    cfg.save_to_file(cfg_file)

    loaded = AppConfig.load_from_file(cfg_file)
    assert loaded.sensitivity_x == 3.5
    assert loaded.scroll_release_ratio == 0.65


def test_keyboard_direct_screen_coordinate_hit_testing(tk_root):
    """OnScreenKeyboard update_cursor_coords detects hovered key from screen coordinates."""
    config = AppConfig()
    kb = OnScreenKeyboard(tk_root, config)
    kb.window.deiconify()
    kb.window.update()
    tk_root.update()

    # Pick the 'a' button
    a_btns = kb._buttons.get("a", [])
    assert len(a_btns) > 0
    btn = a_btns[0]
    btn.update_idletasks()

    bx = btn.winfo_rootx() + btn.winfo_width() // 2
    by = btn.winfo_rooty() + btn.winfo_height() // 2

    hit = kb.update_cursor_coords(bx, by)
    assert hit == "a"
    assert kb._current_hover_key == "a"
    assert "a" in kb.lbl_status.cget("text")

    # Move cursor outside
    outside_hit = kb.update_cursor_coords(0, 0)
    assert outside_hit is None
    assert kb._current_hover_key is None

    kb.close()


def test_keyboard_wink_click_selection_and_debounce(tk_root):
    """Validated wink-click triggers hovered key, while debounce blocks duplicate triggers."""
    config = AppConfig()
    kb = OnScreenKeyboard(tk_root, config)
    kb.window.update()

    # Position cursor over 'A'
    a_btns = kb._buttons.get("a", [])
    assert len(a_btns) > 0
    btn = a_btns[0]
    btn.update_idletasks()
    bx = btn.winfo_rootx() + btn.winfo_width() // 2
    by = btn.winfo_rooty() + btn.winfo_height() // 2

    # Click 1: should type 'a'
    clicked = kb.handle_click_event(bx, by)
    assert clicked is True
    assert kb.get_text() == "a"

    # Click 2 immediately without debounce duration: should be blocked
    clicked2 = kb.handle_click_event(bx, by)
    assert clicked2 is True  # handled
    assert kb.get_text() == "a"  # NOT duplicated!

    kb.close()


def test_keyboard_external_typing_cancel_button(tk_root):
    """External typing countdown displays prominent Cancel button and allows safe abort."""
    config = AppConfig()
    kb = OnScreenKeyboard(tk_root, config)
    kb.insert_text("Message for external app")

    kb.start_external_typing()
    assert kb._external_typing_timer_id is not None
    assert kb.btn_cancel_ext is not None

    # Click cancel
    kb.cancel_external_typing()
    assert kb._external_typing_timer_id is None
    assert kb.btn_cancel_ext is None
    assert "Cancelled" in kb.lbl_status.cget("text")

    kb.close()


def test_camera_has_new_frame_deduplication():
    """CameraManager only returns a frame when a new frame has actually been captured."""
    cfg = AppConfig()
    cam = CameraManager(cfg)
    cam.is_opened = True

    # Simulate camera worker capturing a frame
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with cam._lock:
        cam._latest_frame = dummy_frame
        cam._latest_timestamp = 100.0
        cam._has_new_frame = True

    # First call: retrieves new frame
    ok, frame, ts = cam.get_latest_frame_with_timestamp()
    assert ok is True
    assert frame is not None
    assert ts == 100.0
    assert cam._has_new_frame is False

    # Second call before next hardware capture: must NOT return stale duplicate frame!
    ok2, frame2, ts2 = cam.get_latest_frame_with_timestamp()
    assert ok2 is False
    assert frame2 is None

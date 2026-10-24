"""Unit tests for Live Tracking Diagnostics Dashboard."""
import sys
import os
import tkinter as tk
import pytest

from src.diagnostics import DiagnosticsDashboard

# Ensure Tk can find standard library files under Python 3.13 on Windows
base = sys.base_prefix
tcl_dir = os.path.join(base, "tcl", "tcl8.6")
tk_dir = os.path.join(base, "tcl", "tk8.6")
if os.path.exists(tcl_dir):
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir):
    os.environ["TK_LIBRARY"] = tk_dir


@pytest.fixture
def tk_root(session_tk_root):
    """Reuses session Tk root to prevent Tcl interpreter recreation issues."""
    return session_tk_root


@pytest.fixture
def dashboard(tk_root):
    """Provides a fresh DiagnosticsDashboard widget for each test."""
    frame = tk.Frame(tk_root)
    frame.pack()
    dash = DiagnosticsDashboard(frame)
    dash.pack()
    yield dash
    try:
        frame.destroy()
    except Exception:
        pass


def test_diagnostics_initialization(dashboard):
    assert dashboard is not None
    assert dashboard.lbl_face is not None
    assert dashboard.lbl_op_state is not None
    assert dashboard.banner_lbl is not None
    assert dashboard.lbl_mouse_moved is not None


def test_diagnostics_update_tracking_active(dashboard):
    sample_telemetry = {
        "face_detected": True,
        "is_calibrated": True,
        "is_paused": False,
        "is_emergency_stopped": False,
        "operational_state": "TRACKING",
        "cam_fps": 30.0,
        "proc_fps": 29.5,
        "gui_fps": 60.0,
        "latency_ms": 14.2,
        "raw_dx": 0.0125,
        "raw_dy": -0.0084,
        "calibrated_dx": 0.250,
        "calibrated_dy": -0.168,
        "target_x": 1150.0,
        "target_y": 480.0,
        "cursor_pos": (1150, 480),
        "mouse_moved": True,
        "mouse_suppressed": False,
        "mouse_suppress_reason": "NONE",
        "input_enabled": True,
        "left_ear": 0.265,
        "right_ear": 0.270,
        "baseline_left": 0.270,
        "baseline_right": 0.270,
        "thresh_left": 0.190,
        "thresh_right": 0.190,
        "left_open": True,
        "right_open": True,
        "eye_state": "OPEN",
        "gesture_event": "NONE",
        "head_pitch": 0.395,
        "scroll_state": "NEUTRAL",
        "camera_status": "Connected",
        "camera_id": 0,
        "landmarks_count": 478,
    }

    dashboard.update_dashboard(sample_telemetry)

    # Banner should reflect Tracking
    assert "TRACKING" in dashboard.banner_lbl.cget("text")
    # Face status should show DETECTED
    assert "DETECTED" in dashboard.lbl_face.cget("text")
    # Pointer moved should be YES
    assert "YES" in dashboard.lbl_mouse_moved.cget("text")
    # Coordinates should match target
    assert "1150" in dashboard.lbl_target_coords.cget("text")


def test_diagnostics_banner_warning_priority(dashboard):
    # 1. Emergency stop has top priority
    dashboard.update_dashboard({"is_emergency_stopped": True, "face_detected": True})
    assert "EMERGENCY STOP" in dashboard.banner_lbl.cget("text")

    # 2. Paused has next priority
    dashboard.update_dashboard({"is_emergency_stopped": False, "is_paused": True, "face_detected": True})
    assert "PAUSED" in dashboard.banner_lbl.cget("text")

    # 3. Face lost has high warning priority
    dashboard.update_dashboard({"is_emergency_stopped": False, "is_paused": False, "face_detected": False})
    assert "FACE NOT DETECTED" in dashboard.banner_lbl.cget("text")
    assert "NOT DETECTED" in dashboard.lbl_face.cget("text")

    # 4. Calibrating
    dashboard.update_dashboard({
        "is_emergency_stopped": False,
        "is_paused": False,
        "face_detected": True,
        "operational_state": "CALIBRATING (5/10)",
    })
    assert "CALIBRATING" in dashboard.banner_lbl.cget("text")

    # 5. Scrolling
    dashboard.update_dashboard({
        "is_emergency_stopped": False,
        "is_paused": False,
        "face_detected": True,
        "operational_state": "SCROLLING_DOWN",
        "scroll_state": "DOWN",
    })
    assert "SCROLLING" in dashboard.banner_lbl.cget("text")


def test_diagnostics_pointer_suppression_and_at_rest(dashboard):
    # Pointer suppressed during scrolling
    dashboard.update_dashboard({
        "face_detected": True,
        "mouse_moved": False,
        "mouse_suppressed": True,
        "mouse_suppress_reason": "SCROLLING",
    })
    assert "Suppressed: SCROLLING" in dashboard.lbl_mouse_moved.cget("text")

    # Pointer suppressed during deadzone rest
    dashboard.update_dashboard({
        "face_detected": True,
        "mouse_moved": False,
        "mouse_suppressed": False,
        "mouse_suppress_reason": "DEADZONE",
    })
    assert "DEADZONE" in dashboard.lbl_mouse_moved.cget("text")


def test_diagnostics_details_expand_collapse(dashboard):
    assert dashboard.details_visible is False
    # Toggle expand
    dashboard._toggle_details()
    assert dashboard.details_visible is True
    assert dashboard.btn_toggle_details.cget("text") == "[-] Details"

    # Toggle collapse
    dashboard._toggle_details()
    assert dashboard.details_visible is False
    assert dashboard.btn_toggle_details.cget("text") == "[+] Details"


def test_diagnostics_handles_missing_keys_gracefully(dashboard):
    # Empty dictionary should not raise an exception
    dashboard.update_dashboard({})
    assert dashboard.lbl_face is not None

    # Partial keys with unusual types
    dashboard.update_dashboard({
        "face_detected": "yes",
        "raw_dx": None,
        "cursor_pos": None,
    })
    assert dashboard.lbl_face is not None

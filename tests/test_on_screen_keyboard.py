"""Unit tests for On-Screen Virtual Keyboard."""
import sys
import os
import time
import tkinter as tk
import pytest

from src.config import AppConfig
from src.on_screen_keyboard import OnScreenKeyboard

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
def keyboard(tk_root):
    config = AppConfig()
    kb = OnScreenKeyboard(tk_root, config)
    yield kb
    try:
        kb.close()
    except Exception:
        pass


def test_keyboard_initialization(keyboard):
    assert keyboard is not None
    assert keyboard.is_open is True
    assert keyboard.get_text() == ""
    assert keyboard.text_preview is not None
    assert keyboard.dwell_progress is not None


def test_typing_printable_characters(keyboard):
    keyboard._on_key_click("H")
    time.sleep(0.26)  # Exceed debounce
    keyboard._on_key_click("i")
    time.sleep(0.26)
    keyboard._on_key_click("!")

    assert keyboard.get_text() == "Hi!"


def test_special_keys_space_backspace_enter_tab(keyboard):
    keyboard.insert_text("Cat")
    assert keyboard.get_text() == "Cat"

    # Backspace
    time.sleep(0.26)
    keyboard._on_key_click("Backspace")
    assert keyboard.get_text() == "Ca"

    # Space
    time.sleep(0.26)
    keyboard._on_key_click("Space")
    assert keyboard.get_text() == "Ca "

    # Enter
    time.sleep(0.26)
    keyboard._on_key_click("Enter")
    assert keyboard.get_text() == "Ca \n"

    # Tab
    time.sleep(0.26)
    keyboard._on_key_click("Tab")
    assert keyboard.get_text() == "Ca \n    "


def test_clear_text(keyboard):
    keyboard.insert_text("Hello")
    assert len(keyboard.get_text()) > 0
    keyboard.handle_clear()
    assert keyboard.get_text() == ""


def test_shift_and_caps_lock(keyboard):
    assert keyboard.is_caps is False
    assert keyboard.is_shift is False

    # Toggle Caps
    keyboard._on_key_click("Caps")
    assert keyboard.is_caps is True
    time.sleep(0.26)
    keyboard._on_key_click("Caps")
    assert keyboard.is_caps is False

    # Toggle Shift
    time.sleep(0.26)
    keyboard._on_key_click("Shift")
    assert keyboard.is_shift is True

    # After typing one character, shift should auto-revert
    time.sleep(0.26)
    keyboard._on_key_click("A")
    assert keyboard.is_shift is False


def test_click_debounce_prevents_double_activations(keyboard):
    # First click accepted
    keyboard._on_key_click("X")
    assert keyboard.get_text() == "X"

    # Immediate second click (within debounce window) rejected
    keyboard._on_key_click("X")
    assert keyboard.get_text() == "X"

    # Waiting past debounce window allows next click
    time.sleep(0.26)
    keyboard._on_key_click("X")
    assert keyboard.get_text() == "XX"


def test_scrolling_suppression_blocks_typing(keyboard):
    keyboard.set_scrolling_suppressed(True)
    assert keyboard.is_scrolling_suppressed is True

    # Clicks should be completely blocked while head-pitch scrolling is active
    time.sleep(0.26)
    keyboard._on_key_click("Z")
    assert keyboard.get_text() == ""

    # Resuming normal head posture re-enables typing
    keyboard.set_scrolling_suppressed(False)
    time.sleep(0.26)
    keyboard._on_key_click("Z")
    assert keyboard.get_text() == "Z"


def test_dwell_cancellation_on_mouse_leave(keyboard):
    keyboard.dwell_enabled = True
    keyboard.dwell_duration_s = 0.5

    # Simulate hovering over key 'K'
    btn = list(keyboard._btn_widget_map.keys())[0]
    keyboard._on_key_enter(btn, "K")
    assert keyboard._current_hover_key == "K"
    assert keyboard._dwell_timer_id is not None

    # Simulate cursor leaving before dwell completes
    keyboard._on_key_leave(btn, "K")
    assert keyboard._current_hover_key is None
    assert keyboard._dwell_timer_id is None
    assert keyboard.dwell_progress["value"] == 0


def test_external_typing_countdown_and_cancel(keyboard):
    keyboard.insert_text("Message")
    assert keyboard._external_typing_timer_id is None

    # Start external typing
    keyboard.start_external_typing()
    assert keyboard._external_typing_countdown == 3
    assert keyboard._external_typing_timer_id is not None

    # Cancel external typing
    keyboard.cancel_external_typing()
    assert keyboard._external_typing_timer_id is None
    assert "Cancelled" in keyboard.lbl_status.cget("text")


def test_close_keyboard(tk_root):
    closed = []
    config = AppConfig()
    kb = OnScreenKeyboard(tk_root, config, on_close_callback=lambda: closed.append(True))
    assert kb.is_open is True

    kb.close()
    assert kb.is_open is False
    assert len(closed) == 1

"""Unit tests for gesture detection, EAR calculation, and event state machine."""
import pytest
from src.config import AppConfig
from src.gestures import (
    calculate_ear,
    calculate_mar,
    calculate_head_pitch_ratio,
    GestureRecognizer,
    GestureAction,
    EyeState,
    GestureState,
    ScrollState,
)


def make_eye_landmarks(width: float = 0.10, height: float = 0.03):
    """Generates synthetic 6 eye landmarks with given width and opening height."""
    cx, cy = 0.5, 0.5
    half_w = width / 2.0
    half_h = height / 2.0
    return [
        (cx - half_w, cy),              # 0: outer corner (p1)
        (cx - half_w / 2, cy - half_h), # 1: upper eyelid outer (p2)
        (cx + half_w / 2, cy - half_h), # 2: upper eyelid inner (p3)
        (cx + half_w, cy),              # 3: inner corner (p4)
        (cx + half_w / 2, cy + half_h), # 4: lower eyelid inner (p5)
        (cx - half_w / 2, cy + half_h), # 5: lower eyelid outer (p6)
    ]


def make_head_pose(pitch_delta: float = 0.0):
    """Generates (nose_tip, mid_eyes, chin) matching a specific pitch delta.

    Neutral pitch ratio = 0.400 (span=0.25, offset=0.100).
    A non-zero pitch_delta changes the nose position relative to the eye-to-chin span.
    """
    mid_eyes = (0.50, 0.40)
    chin = (0.50, 0.65)
    # offset / 0.25 = 0.400 + pitch_delta => offset = 0.100 + pitch_delta * 0.25
    nose_y = 0.40 + 0.100 + (pitch_delta * 0.25)
    nose_tip = (0.50, nose_y)
    return nose_tip, mid_eyes, chin


# -------------------------------------------------------------
# EAR & Landmark Tests
# -------------------------------------------------------------

def test_calculate_ear_open_vs_closed():
    open_eye = make_eye_landmarks(width=0.10, height=0.03)
    closed_eye = make_eye_landmarks(width=0.10, height=0.005)

    open_ear = calculate_ear(open_eye)
    closed_ear = calculate_ear(closed_eye)

    assert open_ear > 0.25
    assert closed_ear < 0.10
    assert open_ear > closed_ear * 2.5


def test_calculate_ear_malformed():
    assert calculate_ear([]) == 0.0
    assert calculate_ear([(0.0, 0.0)]) == 0.0
    zero_width = [(0.5, 0.5)] * 6
    assert calculate_ear(zero_width) == 0.0


def test_calculate_head_pitch_ratio_translation_invariance():
    """Whole-head translation up/down must NOT change the calculated pitch ratio."""
    nose1, eyes1, chin1 = make_head_pose(pitch_delta=0.0)
    ratio1 = calculate_head_pitch_ratio(nose1, eyes1, chin1)

    # Translate head up by 0.15 (e.g. moving cursor to top of screen)
    nose2 = (nose1[0], nose1[1] - 0.15)
    eyes2 = (eyes1[0], eyes1[1] - 0.15)
    chin2 = (chin1[0], chin1[1] - 0.15)
    ratio2 = calculate_head_pitch_ratio(nose2, eyes2, chin2)

    assert pytest.approx(ratio1, rel=1e-4) == ratio2


# -------------------------------------------------------------
# 14 User Scenarios for Gestures, Clicks, and Scrolls
# -------------------------------------------------------------

# 1. Both eyes open -> no click
def test_1_both_eyes_open_no_click():
    cfg = AppConfig()
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks(height=0.03)

    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        current_time=1.0,
    )
    assert action == GestureAction.NONE
    assert debug.eye_state in ("OPEN", "BOTH OPEN")
    assert debug.gesture_state == "IDLE"


# 2. Both eyes closed -> no click (natural blink rejection)
def test_2_both_eyes_closed_no_click():
    cfg = AppConfig()
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)

    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=closed_eye,
        right_eye=closed_eye,
        current_time=1.0,
    )
    assert action == GestureAction.NONE
    assert debug.eye_state in ("BLINK", "BOTH CLOSED")
    assert debug.gesture_state == "BILATERAL_BLINK"


# 3. Left wink -> exactly one left click
def test_3_left_wink_exactly_one_click():
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # Frame 1: wink begins at t=1.0 (no click yet, must hold)
    action1, _, _, _, debug1 = recognizer.update(
        face_detected=True,
        left_eye=closed_eye,
        right_eye=open_eye,
        current_time=1.0,
    )
    assert action1 == GestureAction.NONE
    assert debug1.gesture_state == "WINK_LEFT_HOLDING"

    # Frame 2: held for 150ms (> 100ms) -> triggers exactly one LEFT_CLICK
    action2, _, _, _, debug2 = recognizer.update(
        face_detected=True,
        left_eye=closed_eye,
        right_eye=open_eye,
        current_time=1.15,
    )
    assert action2 == GestureAction.LEFT_CLICK
    assert debug2.gesture_state == "WAIT_REOPEN"


# 4. Right wink -> exactly one right click
def test_4_right_wink_exactly_one_click():
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # Frame 1: wink begins
    recognizer.update(True, open_eye, closed_eye, current_time=1.0)

    # Frame 2: held for 150ms -> triggers exactly one RIGHT_CLICK
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=closed_eye,
        current_time=1.15,
    )
    assert action == GestureAction.RIGHT_CLICK
    assert debug.gesture_state == "WAIT_REOPEN"


# 5. Holding wink -> still only one click
def test_5_holding_wink_still_only_one_click():
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    recognizer.update(True, closed_eye, open_eye, current_time=1.0)
    # First click fires at 1.15s
    action, _, _, _, _ = recognizer.update(True, closed_eye, open_eye, current_time=1.15)
    assert action == GestureAction.LEFT_CLICK

    # Keep holding closed at 1.40s, 1.70s, 2.50s (past any cooldown)
    for t in [1.40, 1.70, 2.00, 2.50]:
        action_held, _, _, _, debug_held = recognizer.update(True, closed_eye, open_eye, current_time=t)
        assert action_held == GestureAction.NONE
        assert debug_held.gesture_state == "WAIT_REOPEN"


# 6. Eye reopening -> resets gesture state
def test_6_eye_reopening_resets_gesture_state():
    cfg = AppConfig(wink_min_duration_s=0.10, click_cooldown_s=0.20)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    recognizer.update(True, closed_eye, open_eye, current_time=1.0)
    recognizer.update(True, closed_eye, open_eye, current_time=1.15)  # Clicks, enters WAIT_REOPEN

    # Reopen both eyes after cooldown
    action, _, _, _, debug = recognizer.update(True, open_eye, open_eye, current_time=1.45)
    assert action == GestureAction.NONE
    assert debug.gesture_state == "IDLE"


# 7. Second wink after cooldown -> second click allowed
def test_7_second_wink_after_cooldown_allowed():
    cfg = AppConfig(wink_min_duration_s=0.10, click_cooldown_s=0.20)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # First wink
    recognizer.update(True, closed_eye, open_eye, current_time=1.0)
    action1, _, _, _, _ = recognizer.update(True, closed_eye, open_eye, current_time=1.15)
    assert action1 == GestureAction.LEFT_CLICK

    # Reopen
    recognizer.update(True, open_eye, open_eye, current_time=1.40)

    # Second wink
    recognizer.update(True, closed_eye, open_eye, current_time=1.50)
    action2, _, _, _, _ = recognizer.update(True, closed_eye, open_eye, current_time=1.65)
    assert action2 == GestureAction.LEFT_CLICK


# 8. Small head movement -> no scroll
def test_8_small_head_movement_no_scroll():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.05)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    nose_neutral, eyes_neutral, chin_neutral = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(nose_neutral, eyes_neutral, chin_neutral)

    # Slight natural pitch variation (+0.02, < 0.05)
    nose_small, eyes_small, chin_small = make_head_pose(pitch_delta=0.02)
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=nose_small,
        mid_eyes=eyes_small,
        chin=chin_small,
        current_time=1.0,
    )
    assert action == GestureAction.NONE
    assert debug.scroll_state == "NEUTRAL"
    assert debug.scroll_event == "NONE"


# 9. Upward pitch -> scroll up
def test_9_upward_pitch_scrolls_up():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.05)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    nose_neutral, eyes_neutral, chin_neutral = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(nose_neutral, eyes_neutral, chin_neutral)

    # Intentional head pitch UP (-0.08 < -0.05)
    nose_up, eyes_up, chin_up = make_head_pose(pitch_delta=-0.08)
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=nose_up,
        mid_eyes=eyes_up,
        chin=chin_up,
        current_time=1.0,
    )
    assert action == GestureAction.SCROLL_UP
    assert debug.scroll_state == "UP"
    assert debug.scroll_event == "UP"


# 10. Downward pitch -> scroll down
def test_10_downward_pitch_scrolls_down():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.05)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    nose_neutral, eyes_neutral, chin_neutral = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(nose_neutral, eyes_neutral, chin_neutral)

    # Intentional head pitch DOWN (+0.08 > +0.05)
    nose_down, eyes_down, chin_down = make_head_pose(pitch_delta=0.08)
    action, _, _, _, debug = recognizer.update(
        face_detected=True,
        left_eye=open_eye,
        right_eye=open_eye,
        nose_tip=nose_down,
        mid_eyes=eyes_down,
        chin=chin_down,
        current_time=1.0,
    )
    assert action == GestureAction.SCROLL_DOWN
    assert debug.scroll_state == "DOWN"
    assert debug.scroll_event == "DOWN"


# 11. Holding pitch -> rate limited
def test_11_holding_pitch_rate_limited():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.05, scroll_interval_s=0.20)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    nose_neutral, eyes_neutral, chin_neutral = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(nose_neutral, eyes_neutral, chin_neutral)

    nose_up, eyes_up, chin_up = make_head_pose(pitch_delta=-0.08)

    # t=1.00: First scroll emits
    action1, _, _, _, _ = recognizer.update(True, open_eye, open_eye, nose_up, eyes_up, chin_up, current_time=1.00)
    assert action1 == GestureAction.SCROLL_UP

    # t=1.05: 50ms later (less than interval 0.20s) -> NO scroll (rate limited)
    action2, _, _, _, _ = recognizer.update(True, open_eye, open_eye, nose_up, eyes_up, chin_up, current_time=1.05)
    assert action2 == GestureAction.NONE

    # t=1.22: 220ms later (exceeds interval) -> Second scroll emits
    action3, _, _, _, _ = recognizer.update(True, open_eye, open_eye, nose_up, eyes_up, chin_up, current_time=1.22)
    assert action3 == GestureAction.SCROLL_UP


# 12. Returning to neutral -> resets scroll state
def test_12_returning_to_neutral_resets_scroll_state():
    cfg = AppConfig(scroll_enabled=True, scroll_pitch_threshold=0.05)
    recognizer = GestureRecognizer(cfg)
    open_eye = make_eye_landmarks()

    nose_neutral, eyes_neutral, chin_neutral = make_head_pose(pitch_delta=0.0)
    recognizer.calibrate_pitch(nose_neutral, eyes_neutral, chin_neutral)

    # Enter UP
    nose_up, eyes_up, chin_up = make_head_pose(pitch_delta=-0.08)
    recognizer.update(True, open_eye, open_eye, nose_up, eyes_up, chin_up, current_time=1.0)
    assert recognizer.scroll_state == ScrollState.UP

    # Return to neutral (pitch_delta = 0.01)
    nose_back, eyes_back, chin_back = make_head_pose(pitch_delta=0.01)
    action, _, _, _, debug = recognizer.update(True, open_eye, open_eye, nose_back, eyes_back, chin_back, current_time=1.1)
    assert action == GestureAction.NONE
    assert debug.scroll_state == "NEUTRAL"
    assert recognizer.scroll_state == ScrollState.NEUTRAL


# 13. Face loss -> no click/scroll
def test_13_face_loss_no_click_or_scroll():
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # Wink was pending
    recognizer.update(True, closed_eye, open_eye, current_time=1.0)

    # Face lost before completion
    action, _, _, _, debug = recognizer.update(
        face_detected=False,
        left_eye=None,
        right_eye=None,
        current_time=1.15,
    )
    assert action == GestureAction.NONE
    assert debug.input_enabled is False
    assert recognizer.gesture_state == GestureState.IDLE


# 14. Emergency stop -> no click/scroll
def test_14_emergency_stop_no_click_or_scroll():
    cfg = AppConfig(wink_min_duration_s=0.10)
    recognizer = GestureRecognizer(cfg)
    closed_eye = make_eye_landmarks(height=0.005)
    open_eye = make_eye_landmarks(height=0.03)

    # Wink held past duration, but is_emergency_stopped=True
    recognizer.update(True, closed_eye, open_eye, is_emergency_stopped=True, current_time=1.0)
    action, _, _, _, debug = recognizer.update(
        True,
        closed_eye,
        open_eye,
        is_emergency_stopped=True,
        current_time=1.15,
    )
    assert action == GestureAction.NONE
    assert debug.input_enabled is False

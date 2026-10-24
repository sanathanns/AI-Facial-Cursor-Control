"""Gesture recognition and event state machine for facial cursor control."""
import math
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.config import AppConfig


class GestureAction(Enum):
    NONE = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    OPTIONAL_DRAG_TOGGLE = auto()
    OPTIONAL_PAUSE_TOGGLE = auto()


class EyeState(Enum):
    OPEN = "OPEN"
    LEFT_WINK = "LEFT WINK"
    RIGHT_WINK = "RIGHT WINK"
    BLINK = "BLINK"
    AMBIGUOUS = "AMBIGUOUS"

    # Aliases for backwards compatibility with tests and older code
    BOTH_OPEN = "OPEN"
    LEFT_CLOSED = "LEFT WINK"
    RIGHT_CLOSED = "RIGHT WINK"
    BOTH_CLOSED = "BLINK"


class GestureState(Enum):
    IDLE = "IDLE"
    WINK_LEFT_HOLDING = "WINK_LEFT_HOLDING"
    WINK_RIGHT_HOLDING = "WINK_RIGHT_HOLDING"
    BILATERAL_BLINK = "BILATERAL_BLINK"
    WAIT_REOPEN = "WAIT_REOPEN"


class ScrollState(Enum):
    NEUTRAL = "NEUTRAL"
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class GestureDebugInfo:
    left_ear: float = 0.0
    right_ear: float = 0.0
    baseline_left: float = 0.25
    baseline_right: float = 0.25
    thresh_left: float = 0.18
    thresh_right: float = 0.18
    left_open: bool = True
    right_open: bool = True
    eye_state: str = "OPEN"
    gesture_state: str = "IDLE"
    gesture_event: str = "NONE"
    head_pitch: float = 0.0
    scroll_state: str = "NEUTRAL"
    scroll_event: str = "NONE"
    input_enabled: bool = False
    raw_dx: float = 0.0
    raw_dy: float = 0.0
    calibrated_dx: float = 0.0
    calibrated_dy: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    filtered_x: float = 0.0
    filtered_y: float = 0.0
    mouse_moved: bool = False
    scroll_mode_active: bool = False
    operational_state: str = "IDLE"
    in_click_cooldown: bool = False
    cooldown_remaining_s: float = 0.0
    bilateral_blink: bool = False
    raw_pitch: float = 0.0
    calibrated_pitch: float = 0.0
    scroll_activation_thresh: float = 0.020
    scroll_release_thresh: float = 0.009
    scroll_transition_reason: str = "STEADY"


def calculate_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """Calculates Eye Aspect Ratio (EAR) from 6 canonical eyelid points.

    Points:
    0: Corner 1 (p1)
    1: Upper eyelid 1 (p2)
    2: Upper eyelid 2 (p3)
    3: Corner 2 (p4)
    4: Lower eyelid 2 (p5)
    5: Lower eyelid 1 (p6)

    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    if len(eye_landmarks) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    horizontal_dist = calculate_euclidean_distance(p1, p4)
    if horizontal_dist < 1e-6:
        return 0.0

    v1 = calculate_euclidean_distance(p2, p6)
    v2 = calculate_euclidean_distance(p3, p5)
    return (v1 + v2) / (2.0 * horizontal_dist)


def calculate_mar(mouth_landmarks: List[Tuple[float, float]]) -> float:
    """Calculates Mouth Aspect Ratio (MAR) from 4 points."""
    if len(mouth_landmarks) < 4:
        return 0.0

    upper, lower, left, right = mouth_landmarks[:4]
    horizontal = calculate_euclidean_distance(left, right)
    if horizontal < 1e-6:
        return 0.0

    vertical = calculate_euclidean_distance(upper, lower)
    return vertical / horizontal


def calculate_head_pitch_ratio(
    nose_tip: Tuple[float, float],
    mid_eyes: Tuple[float, float],
    chin: Optional[Tuple[float, float]] = None,
) -> float:
    """Calculates rotational head pitch invariant to whole-head translation and roll.

    Uses vector projection of (nose_tip - mid_eyes) onto the facial centerline (chin - mid_eyes).
    Returns (N . V) / ||V||^2.
    Tilting head UP foreshortens nose towards eyes (ratio decreases).
    Tilting head DOWN extends nose towards chin (ratio increases).
    Pure translation (X, Y) and in-plane roll cancel out completely.
    """
    if chin is not None:
        vx = chin[0] - mid_eyes[0]
        vy = chin[1] - mid_eyes[1]
        v_sq = vx * vx + vy * vy
        if v_sq > 1e-6:
            nx = nose_tip[0] - mid_eyes[0]
            ny = nose_tip[1] - mid_eyes[1]
            return (nx * vx + ny * vy) / v_sq

    # Fallback to direct normalized vertical span if chin is unavailable
    nominal_span = 0.25
    return (nose_tip[1] - mid_eyes[1]) / nominal_span


class GestureRecognizer:
    """Deterministic state machine detecting intentional winks and vertical pitch scrolling."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Wink State Machine
        self.gesture_state: GestureState = GestureState.IDLE
        self._wink_start_time: Optional[float] = None
        self._wink_grace_frames: int = 0
        self._last_click_time: float = 0.0
        self._last_bilateral_blink_time: float = 0.0

        # Eye Calibration & Adaptive Thresholds (Part 5)
        self.baseline_left_ear: float = 0.25
        self.baseline_right_ear: float = 0.25
        self.adaptive_blink_left: float = config.ear_blink_threshold
        self.adaptive_blink_right: float = config.ear_blink_threshold
        self.adaptive_open_left: float = config.ear_open_threshold
        self.adaptive_open_right: float = config.ear_open_threshold
        self.is_eye_calibrated: bool = False

        # Auto-calibration accumulator for first steady frames
        self._open_samples_left: List[float] = []
        self._open_samples_right: List[float] = []
        self._auto_calibrated: bool = False

        # Scroll State Machine with Intentional Hysteresis & Debounce
        self.scroll_state: ScrollState = ScrollState.NEUTRAL
        self.neutral_pitch: float = 0.38  # Default nominal neutral pitch ratio
        self.is_pitch_calibrated: bool = False
        self._filtered_pitch: Optional[float] = None
        self._last_scroll_time: float = 0.0
        self._scroll_rearmed: bool = True
        self._scroll_release_time: float = 0.0
        self._scroll_transition_reason: str = "INITIALIZED"

        # Optional Mouth Gesture Tracker
        self._mouth_open_start_time: Optional[float] = None
        self._last_mouth_action_time: float = 0.0

        # Face-loss Safety
        self._last_face_seen_time: float = 0.0
        self.is_face_lost_safe: bool = True

        # Periodic debug logging throttle
        self._last_debug_log_time: float = 0.0

    def calibrate_pitch(
        self,
        nose_tip: Tuple[float, float],
        mid_eyes: Tuple[float, float],
        chin: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Anchors the user's current comfortable head pitch as neutral."""
        self.neutral_pitch = calculate_head_pitch_ratio(nose_tip, mid_eyes, chin)
        self.is_pitch_calibrated = True
        self._filtered_pitch = None
        self.scroll_state = ScrollState.NEUTRAL
        self._scroll_rearmed = True
        self._scroll_transition_reason = "CALIBRATED_NEUTRAL"
        print(f"[CALIBRATION] Neutral Head Pitch anchored: {self.neutral_pitch:.3f}")

    def calibrate_eyes(self, left_ear: float, right_ear: float) -> None:
        """Calibrates resting open-eye EAR baseline and calculates adaptive wink thresholds (Part 5)."""
        if left_ear > 0.05 and right_ear > 0.05:
            self.baseline_left_ear = left_ear
            self.baseline_right_ear = right_ear
            self.is_eye_calibrated = True

            # Adaptive thresholds: closed eye <= baseline * ear_blink_ratio, open eye >= baseline * ear_open_ratio
            self.adaptive_blink_left = max(0.08, left_ear * self.config.ear_blink_ratio)
            self.adaptive_blink_right = max(0.08, right_ear * self.config.ear_blink_ratio)
            self.adaptive_open_left = min(left_ear * self.config.ear_open_ratio, left_ear - 0.02)
            self.adaptive_open_right = min(right_ear * self.config.ear_open_ratio, right_ear - 0.02)

            print(f"[CALIBRATION] Baseline Open EAR: Left={left_ear:.3f}, Right={right_ear:.3f}")
            print(f"[CALIBRATION] Adaptive Blink Cutoff: Left<{self.adaptive_blink_left:.3f}, Right<{self.adaptive_blink_right:.3f}")
            print(f"[CALIBRATION] Adaptive Open Cutoff: Left>={self.adaptive_open_left:.3f}, Right>={self.adaptive_open_right:.3f}")

    def reset_trackers(self) -> None:
        """Resets all transient gesture timers and states upon face loss or stop."""
        self.gesture_state = GestureState.IDLE
        self.scroll_state = ScrollState.NEUTRAL
        self._wink_start_time = None
        self._wink_grace_frames = 0
        self._mouth_open_start_time = None
        self._filtered_pitch = None
        self._scroll_rearmed = True
        self._scroll_transition_reason = "RESET"

    def update(
        self,
        face_detected: bool,
        left_eye: Optional[List[Tuple[float, float]]],
        right_eye: Optional[List[Tuple[float, float]]],
        nose_tip: Optional[Tuple[float, float]] = None,
        mid_eyes: Optional[Tuple[float, float]] = None,
        chin: Optional[Tuple[float, float]] = None,
        mouth: Optional[List[Tuple[float, float]]] = None,
        is_calibrated: bool = True,
        is_paused: bool = False,
        is_emergency_stopped: bool = False,
        current_time: Optional[float] = None,
        is_scroll_mode: bool = False,
    ) -> Tuple[GestureAction, float, float, Optional[float], GestureDebugInfo]:
        """Processes frame landmarks and outputs action, EARs, MAR, and live debug info."""
        now = time.monotonic() if current_time is None else current_time
        input_enabled = face_detected and (not is_paused) and (not is_emergency_stopped)

        # 1. Face-Loss Safety Check
        if not face_detected or left_eye is None or right_eye is None:
            if now - self._last_face_seen_time > self.config.face_loss_freeze_s:
                self.is_face_lost_safe = True
            self.reset_trackers()
            debug_info = GestureDebugInfo(
                left_ear=0.0,
                right_ear=0.0,
                baseline_left=self.baseline_left_ear,
                baseline_right=self.baseline_right_ear,
                thresh_left=self.adaptive_blink_left,
                thresh_right=self.adaptive_blink_right,
                left_open=False,
                right_open=False,
                eye_state="NO FACE",
                gesture_state=self.gesture_state.value,
                gesture_event="NONE",
                head_pitch=0.0,
                scroll_state=self.scroll_state.value,
                scroll_event="NONE",
                input_enabled=False,
                scroll_mode_active=is_scroll_mode,
                raw_pitch=0.0,
                calibrated_pitch=0.0,
                scroll_activation_thresh=self.config.scroll_pitch_threshold,
                scroll_release_thresh=self.config.scroll_pitch_threshold * getattr(self.config, "scroll_release_ratio", 0.45),
                scroll_transition_reason="NO_FACE",
            )
            return GestureAction.NONE, 0.0, 0.0, None, debug_info

        self._last_face_seen_time = now
        self.is_face_lost_safe = False

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        mar = calculate_mar(mouth) if (mouth and self.config.enable_mouth_gestures) else None

        # Auto-seed open eye baseline over first 20 steady frames if uncalibrated
        if not self.is_eye_calibrated and not self._auto_calibrated:
            if left_ear > 0.16 and right_ear > 0.16 and abs(left_ear - right_ear) < 0.06:
                self._open_samples_left.append(left_ear)
                self._open_samples_right.append(right_ear)
                if len(self._open_samples_left) >= 20:
                    avg_l = sum(self._open_samples_left) / len(self._open_samples_left)
                    avg_r = sum(self._open_samples_right) / len(self._open_samples_right)
                    self.calibrate_eyes(avg_l, avg_r)
                    self._auto_calibrated = True

        # 2. Eye State Classification using adaptive or config thresholds (Part 5 & Part 6)
        blink_thresh_l = self.adaptive_blink_left if self.is_eye_calibrated else self.config.ear_blink_threshold
        blink_thresh_r = self.adaptive_blink_right if self.is_eye_calibrated else self.config.ear_blink_threshold

        open_thresh_l = self.adaptive_open_left if self.is_eye_calibrated else self.config.ear_open_threshold
        open_thresh_r = self.adaptive_open_right if self.is_eye_calibrated else self.config.ear_open_threshold

        left_closed = left_ear < blink_thresh_l
        right_closed = right_ear < blink_thresh_r

        left_open = left_ear >= open_thresh_l
        right_open = right_ear >= open_thresh_r

        asym_min = min(0.020, self.config.ear_asymmetry_min_diff)

        # Robust asymmetry classification:
        # If both eyes drop below blink threshold, verify whether it's a bilateral blink (symmetric)
        # or a hard wink where facial coupling causes the open eye to squint.
        if left_closed and right_closed:
            if (right_ear - left_ear) >= asym_min or (right_ear > 0 and left_ear / right_ear <= 0.82):
                eye_state = EyeState.LEFT_WINK
            elif (left_ear - right_ear) >= asym_min or (left_ear > 0 and right_ear / left_ear <= 0.82):
                eye_state = EyeState.RIGHT_WINK
            else:
                eye_state = EyeState.BLINK
        elif left_closed and (right_open or (right_ear >= left_ear + asym_min) or (right_ear > 0 and left_ear / right_ear <= 0.82)):
            eye_state = EyeState.LEFT_WINK
        elif right_closed and (left_open or (left_ear >= right_ear + asym_min) or (left_ear > 0 and right_ear / left_ear <= 0.82)):
            eye_state = EyeState.RIGHT_WINK
        elif left_open and right_open:
            eye_state = EyeState.OPEN
        else:
            eye_state = EyeState.AMBIGUOUS

        # 3. Wink Detection State Machine (Instant click at min duration, no hold repeat)
        detected_action = GestureAction.NONE
        time_since_click = now - self._last_click_time
        in_click_cooldown = time_since_click < self.config.click_cooldown_s
        post_blink_guard = (now - self._last_bilateral_blink_time) < 0.15

        if self.gesture_state == GestureState.IDLE:
            if eye_state == EyeState.BLINK:
                self.gesture_state = GestureState.BILATERAL_BLINK
                self._last_bilateral_blink_time = now
            elif eye_state == EyeState.LEFT_WINK and not in_click_cooldown and not post_blink_guard:
                self.gesture_state = GestureState.WINK_LEFT_HOLDING
                self._wink_start_time = now
                self._wink_grace_frames = 0
                print("[DEBUG] Left wink candidate")
            elif eye_state == EyeState.RIGHT_WINK and not in_click_cooldown and not post_blink_guard:
                self.gesture_state = GestureState.WINK_RIGHT_HOLDING
                self._wink_start_time = now
                self._wink_grace_frames = 0
                print("[DEBUG] Right wink candidate")

        elif self.gesture_state == GestureState.WINK_LEFT_HOLDING:
            if eye_state == EyeState.LEFT_WINK:
                self._wink_grace_frames = 0
                duration = now - (self._wink_start_time or now)
                if duration >= self.config.wink_min_duration_s:
                    print("[DEBUG] Left wink recognized")
                    print("[GESTURE] LEFT_WINK")
                    print("[EVENT] CLICK_LEFT")
                    if input_enabled:
                        detected_action = GestureAction.LEFT_CLICK
                    self._last_click_time = now
                    self.gesture_state = GestureState.WAIT_REOPEN
            else:
                self._wink_grace_frames += 1
                duration = now - (self._wink_start_time or now)
                if duration >= self.config.wink_min_duration_s:
                    print("[DEBUG] Left wink recognized on reopen")
                    print("[GESTURE] LEFT_WINK")
                    print("[EVENT] CLICK_LEFT")
                    if input_enabled:
                        detected_action = GestureAction.LEFT_CLICK
                    self._last_click_time = now
                    self.gesture_state = GestureState.WAIT_REOPEN
                elif self._wink_grace_frames > 2:
                    if eye_state == EyeState.BLINK:
                        self.gesture_state = GestureState.BILATERAL_BLINK
                        self._last_bilateral_blink_time = now
                    else:
                        self.gesture_state = GestureState.IDLE

        elif self.gesture_state == GestureState.WINK_RIGHT_HOLDING:
            if eye_state == EyeState.RIGHT_WINK:
                self._wink_grace_frames = 0
                duration = now - (self._wink_start_time or now)
                if duration >= self.config.wink_min_duration_s:
                    print("[DEBUG] Right wink recognized")
                    print("[GESTURE] RIGHT_WINK")
                    print("[EVENT] CLICK_RIGHT")
                    if input_enabled:
                        detected_action = GestureAction.RIGHT_CLICK
                    self._last_click_time = now
                    self.gesture_state = GestureState.WAIT_REOPEN
            else:
                self._wink_grace_frames += 1
                duration = now - (self._wink_start_time or now)
                if duration >= self.config.wink_min_duration_s:
                    print("[DEBUG] Right wink recognized on reopen")
                    print("[GESTURE] RIGHT_WINK")
                    print("[EVENT] CLICK_RIGHT")
                    if input_enabled:
                        detected_action = GestureAction.RIGHT_CLICK
                    self._last_click_time = now
                    self.gesture_state = GestureState.WAIT_REOPEN
                elif self._wink_grace_frames > 2:
                    if eye_state == EyeState.BLINK:
                        self.gesture_state = GestureState.BILATERAL_BLINK
                        self._last_bilateral_blink_time = now
                    else:
                        self.gesture_state = GestureState.IDLE

        elif self.gesture_state == GestureState.BILATERAL_BLINK:
            if eye_state != EyeState.BLINK:
                self._last_bilateral_blink_time = now
                self.gesture_state = GestureState.IDLE

        elif self.gesture_state == GestureState.WAIT_REOPEN:
            # Must wait until eyes are reopened and click cooldown elapsed before accepting new clicks
            if (eye_state == EyeState.OPEN or (left_open and right_open)) and not in_click_cooldown:
                self.gesture_state = GestureState.IDLE

        # 4. Intentional Head-Pitch Scrolling (Direct & Automatic with Hysteresis & Rearm)
        raw_pitch = 0.0
        head_pitch_delta = 0.0
        scroll_event_name = "NONE"
        pitch_thresh = self.config.scroll_pitch_threshold
        release_ratio = getattr(self.config, "scroll_release_ratio", 0.60)
        release_thresh = pitch_thresh * release_ratio
        neutral_deadband = pitch_thresh * 0.40

        if nose_tip is not None and mid_eyes is not None and self.config.scroll_enabled:
            raw_pitch = calculate_head_pitch_ratio(nose_tip, mid_eyes, chin)
            if not self.is_pitch_calibrated:
                self.neutral_pitch = raw_pitch
                self.is_pitch_calibrated = True

            self._filtered_pitch = raw_pitch
            head_pitch_delta = raw_pitch - self.neutral_pitch

            # State Transitions with Separate Activation/Release Hysteresis & Neutral Rearm
            if self.scroll_state == ScrollState.NEUTRAL:
                # Must return inside neutral deadband before subsequent scroll can trigger
                if not self._scroll_rearmed:
                    if abs(head_pitch_delta) <= neutral_deadband:
                        self._scroll_rearmed = True
                        self._scroll_transition_reason = "REARMED_IN_NEUTRAL"

                if self._scroll_rearmed:
                    if head_pitch_delta < -pitch_thresh:
                        self.scroll_state = ScrollState.UP
                        self._last_scroll_time = 0.0  # allow immediate first scroll
                        self._scroll_transition_reason = f"TRIGGER_UP (delta {head_pitch_delta:+.3f} < -{pitch_thresh:.3f})"
                        print(f"[SCROLL] Active: SCROLL UP ({self._scroll_transition_reason})")
                    elif head_pitch_delta > pitch_thresh:
                        self.scroll_state = ScrollState.DOWN
                        self._last_scroll_time = 0.0  # allow immediate first scroll
                        self._scroll_transition_reason = f"TRIGGER_DOWN (delta {head_pitch_delta:+.3f} > +{pitch_thresh:.3f})"
                        print(f"[SCROLL] Active: SCROLL DOWN ({self._scroll_transition_reason})")
                    else:
                        self._scroll_transition_reason = "NEUTRAL_WITHIN_DEADBAND"
                else:
                    self._scroll_transition_reason = "AWAITING_NEUTRAL_RETURN"

            elif self.scroll_state == ScrollState.UP:
                # Release promptly only when pitch clearly returns within release threshold
                if head_pitch_delta > -release_thresh:
                    self.scroll_state = ScrollState.NEUTRAL
                    self._scroll_rearmed = (abs(head_pitch_delta) <= neutral_deadband)
                    self._scroll_release_time = now
                    self._scroll_transition_reason = f"RELEASE_UP_TO_NEUTRAL (delta {head_pitch_delta:+.3f} > -{release_thresh:.3f})"
                    print(f"[SCROLL] Released -> NEUTRAL ({self._scroll_transition_reason})")
                else:
                    self._scroll_transition_reason = "HOLDING_UP"

            elif self.scroll_state == ScrollState.DOWN:
                # Release promptly only when pitch clearly returns within release threshold
                if head_pitch_delta < release_thresh:
                    self.scroll_state = ScrollState.NEUTRAL
                    self._scroll_rearmed = (abs(head_pitch_delta) <= neutral_deadband)
                    self._scroll_release_time = now
                    self._scroll_transition_reason = f"RELEASE_DOWN_TO_NEUTRAL (delta {head_pitch_delta:+.3f} < +{release_thresh:.3f})"
                    print(f"[SCROLL] Released -> NEUTRAL ({self._scroll_transition_reason})")
                else:
                    self._scroll_transition_reason = "HOLDING_DOWN"

            # Emit rate-limited scroll actions
            if input_enabled and detected_action == GestureAction.NONE:
                if self.scroll_state == ScrollState.UP:
                    if now - self._last_scroll_time >= self.config.scroll_interval_s:
                        detected_action = GestureAction.SCROLL_UP
                        scroll_event_name = "UP"
                        self._last_scroll_time = now
                elif self.scroll_state == ScrollState.DOWN:
                    if now - self._last_scroll_time >= self.config.scroll_interval_s:
                        detected_action = GestureAction.SCROLL_DOWN
                        scroll_event_name = "DOWN"
                        self._last_scroll_time = now

        # 5. Optional Mouth Gestures (Only if explicitly enabled)
        if (
            self.config.enable_mouth_gestures
            and mar is not None
            and detected_action == GestureAction.NONE
            and input_enabled
        ):
            if mar > self.config.mar_threshold:
                if self._mouth_open_start_time is None:
                    self._mouth_open_start_time = now
                if (now - self._mouth_open_start_time > 0.6) and (now - self._last_mouth_action_time > 1.0):
                    detected_action = GestureAction.OPTIONAL_DRAG_TOGGLE
                    self._last_mouth_action_time = now
                    self._mouth_open_start_time = None
            else:
                self._mouth_open_start_time = None

        gesture_event_name = detected_action.name if detected_action in (
            GestureAction.LEFT_CLICK,
            GestureAction.RIGHT_CLICK,
        ) else "NONE"

        debug_info = GestureDebugInfo(
            left_ear=left_ear,
            right_ear=right_ear,
            baseline_left=self.baseline_left_ear,
            baseline_right=self.baseline_right_ear,
            thresh_left=blink_thresh_l,
            thresh_right=blink_thresh_r,
            left_open=left_open,
            right_open=right_open,
            eye_state=eye_state.value,
            gesture_state=self.gesture_state.value,
            gesture_event=gesture_event_name,
            head_pitch=head_pitch_delta,
            scroll_state=self.scroll_state.value,
            scroll_event=scroll_event_name,
            input_enabled=input_enabled,
            scroll_mode_active=is_scroll_mode,
            in_click_cooldown=in_click_cooldown,
            cooldown_remaining_s=max(0.0, self.config.click_cooldown_s - time_since_click) if in_click_cooldown else 0.0,
            bilateral_blink=(eye_state == EyeState.BLINK),
            raw_pitch=raw_pitch,
            calibrated_pitch=head_pitch_delta,
            scroll_activation_thresh=pitch_thresh,
            scroll_release_thresh=release_thresh,
            scroll_transition_reason=self._scroll_transition_reason,
        )

        # Periodic console debug logging (once every 1.5 seconds if state is quiet)
        if now - self._last_debug_log_time >= 1.5:
            self._last_debug_log_time = now
            print(
                f"[DEBUG] L_EAR: {left_ear:.3f} [{'OPEN' if left_open else 'CLOSED'}] (base: {self.baseline_left_ear:.3f}, thr: {blink_thresh_l:.3f}) | "
                f"R_EAR: {right_ear:.3f} [{'OPEN' if right_open else 'CLOSED'}] (base: {self.baseline_right_ear:.3f}, thr: {blink_thresh_r:.3f}) | "
                f"Eye: {eye_state.value:<12} | State: {self.gesture_state.value:<17} | "
                f"Pitch: {head_pitch_delta:+.3f} ({self.scroll_state.value}) | "
                f"Input: {'ENABLED' if input_enabled else 'DISABLED'}"
            )

        return detected_action, left_ear, right_ear, mar, debug_info

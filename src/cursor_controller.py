"""Cursor controller integrating calibration, filtering, sensitivity, and mouse events."""
import time
from typing import Optional, Tuple
import pyautogui

from src.config import AppConfig
from src.filters import PointOneEuroFilter, DeadZoneFilter
from src.gestures import GestureAction


class CursorController:
    """Controls the operating system mouse cursor based on head pose and facial gestures."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Set PyAutoGUI properties for responsive and safe control (Part 3)
        pyautogui.FAILSAFE = config.pyautogui_failsafe
        pyautogui.PAUSE = 0.0

        # Screen dimensions
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception:
            # Fallback for headless environments
            self.screen_width, self.screen_height = 1920, 1080

        self.screen_center_x = self.screen_width / 2.0
        self.screen_center_y = self.screen_height / 2.0

        # Calibration anchor point (normalized image space)
        self.neutral_x: Optional[float] = None
        self.neutral_y: Optional[float] = None
        self.is_calibrated: bool = False

        # Filters
        self.point_filter = PointOneEuroFilter(
            min_cutoff=config.cursor_min_cutoff,
            beta=config.cursor_beta,
            d_cutoff=config.cursor_derivative_cutoff,
        )
        self.dead_zone_filter = DeadZoneFilter(radius=config.dead_zone_radius)

        # Operational States
        self.is_paused: bool = False
        self.is_emergency_stopped: bool = False
        self.is_dragging: bool = False
        self.last_cursor_pos: Tuple[int, int] = (int(self.screen_center_x), int(self.screen_center_y))
        self._was_scrolling: bool = False

        # Consistent Telemetry Interface (Guaranteed defaults for GUI, HUD & Diagnostics)
        self.last_raw_dx: float = 0.0
        self.last_raw_dy: float = 0.0
        self.last_calibrated_dx: float = 0.0
        self.last_calibrated_dy: float = 0.0
        self.last_target_x: float = float(self.screen_center_x)
        self.last_target_y: float = float(self.screen_center_y)
        self.last_filtered_x: float = float(self.screen_center_x)
        self.last_filtered_y: float = float(self.screen_center_y)
        self.last_mouse_moved: bool = False
        self.last_mouse_suppressed: bool = False
        self.last_mouse_suppress_reason: str = "NONE"

    def get_telemetry(self) -> dict:
        """Returns a consistent cursor telemetry dictionary with guaranteed defaults."""
        return {
            "raw_dx": self.last_raw_dx,
            "raw_dy": self.last_raw_dy,
            "calibrated_dx": self.last_calibrated_dx,
            "calibrated_dy": self.last_calibrated_dy,
            "target_x": self.last_target_x,
            "target_y": self.last_target_y,
            "filtered_x": self.last_filtered_x,
            "filtered_y": self.last_filtered_y,
            "mouse_moved": self.last_mouse_moved,
            "mouse_suppressed": self.last_mouse_suppressed,
            "mouse_suppress_reason": self.last_mouse_suppress_reason,
            "cursor_pos": self.last_cursor_pos,
            "is_calibrated": self.is_calibrated,
            "is_paused": self.is_paused,
            "is_emergency_stopped": self.is_emergency_stopped,
            "is_dragging": self.is_dragging,
        }

    def update_config_params(self) -> None:
        """Propagates updated config values into active filters."""
        self.point_filter.update_params(
            min_cutoff=self.config.cursor_min_cutoff,
            beta=self.config.cursor_beta,
            d_cutoff=self.config.cursor_derivative_cutoff,
        )
        self.dead_zone_filter.radius = self.config.dead_zone_radius

    def calibrate(self, x: float, y: float) -> bool:
        """Calibrates the current head position as the neutral center with boundary checks."""
        import math
        if not (math.isfinite(x) and math.isfinite(y) and 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95):
            return False

        self.neutral_x = x
        self.neutral_y = y
        self.is_calibrated = True
        self.last_raw_dx = 0.0
        self.last_raw_dy = 0.0
        self.last_calibrated_dx = 0.0
        self.last_calibrated_dy = 0.0
        self.point_filter.reset()
        return True

    def reset_neutral(self) -> None:
        """Clears neutral calibration and resets movement telemetry."""
        self.neutral_x = None
        self.neutral_y = None
        self.is_calibrated = False
        self.last_raw_dx = 0.0
        self.last_raw_dy = 0.0
        self.last_calibrated_dx = 0.0
        self.last_calibrated_dy = 0.0
        self.last_mouse_moved = False
        self.last_mouse_suppressed = False
        self.last_mouse_suppress_reason = "NONE"
        self.point_filter.reset()

    def on_face_lost(self) -> None:
        """Invoked when face tracking is lost to release drag, clear smoothing, and reset telemetry."""
        self.release_drag()
        self.point_filter.reset()
        self.last_mouse_moved = False
        self.last_mouse_suppressed = True
        self.last_mouse_suppress_reason = "NO_FACE"
        self.last_raw_dx = 0.0
        self.last_raw_dy = 0.0
        self.last_calibrated_dx = 0.0
        self.last_calibrated_dy = 0.0

    def toggle_pause(self) -> bool:
        """Toggles tracking pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.last_mouse_moved = False
            self.last_mouse_suppressed = True
            self.last_mouse_suppress_reason = "PAUSED"
            if self.is_dragging:
                self.release_drag()
        else:
            self.last_mouse_suppressed = False
            self.last_mouse_suppress_reason = "NONE"
        return self.is_paused

    def trigger_emergency_stop(self) -> None:
        """Instantly halts all mouse control."""
        self.is_emergency_stopped = True
        self.last_mouse_moved = False
        self.last_mouse_suppressed = True
        self.last_mouse_suppress_reason = "EMERGENCY_STOP"
        if self.is_dragging:
            self.release_drag()

    def resume_from_emergency_stop(self) -> None:
        """Resets emergency stop condition."""
        self.is_emergency_stopped = False
        self.last_mouse_moved = False
        self.last_mouse_suppressed = False
        self.last_mouse_suppress_reason = "NONE"
        self.point_filter.reset()

    def release_drag(self) -> None:
        """Releases drag if mouse button was held down."""
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button='left')
            except Exception:
                pass
            self.is_dragging = False

    def update_cursor_position(
        self,
        current_x: float,
        current_y: float,
        timestamp: Optional[float] = None,
        is_scrolling: bool = False,
        is_scroll_mode: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Calculates and updates cursor position based on head pose.
        
        Horizontal and vertical head movements control cursor X and Y with smooth local comfort mapping.
        Freezes completely when tilting head to scroll, preventing cursor jumps during scrolling,
        and cleanly resumes normal tracking the instant scrolling finishes.
        """
        currently_scrolling = is_scrolling or is_scroll_mode

        # If transitioning from active scrolling back to normal tracking, cleanly reset filter
        # so cursor resumes immediately without stale lag or velocity hitch
        if self._was_scrolling and not currently_scrolling:
            self.point_filter.reset()
        self._was_scrolling = currently_scrolling

        # If not calibrated yet, auto-calibrate on the first valid coordinate
        if not self.is_calibrated or self.neutral_x is None or self.neutral_y is None:
            if not self.calibrate(current_x, current_y):
                self.last_raw_dx = 0.0
                self.last_raw_dy = 0.0
                self.last_calibrated_dx = 0.0
                self.last_calibrated_dy = 0.0
                self.last_mouse_moved = False
                self.last_mouse_suppressed = True
                self.last_mouse_suppress_reason = "NEEDS_CALIBRATION"
                return None

        # Calculate horizontal and vertical displacements relative to calibrated neutral position
        raw_dx = current_x - self.neutral_x
        raw_dy = current_y - self.neutral_y
        dz_radius = max(0.001, self.config.dead_zone_radius)

        # Horizontal deadzone with continuous smooth ease-in to prevent boundary chatter
        if abs(raw_dx) <= dz_radius:
            filtered_dx = 0.0
        else:
            excess_x = abs(raw_dx) - dz_radius
            ramp_x = min(1.0, excess_x / max(1e-4, dz_radius * 0.5))
            sign_x = 1.0 if raw_dx > 0 else -1.0
            filtered_dx = sign_x * excess_x * ramp_x

        # Normalize horizontal displacement against local comfort range
        comfort_x = max(1e-4, getattr(self.config, "comfort_range_x", 0.045))
        norm_dx = filtered_dx / comfort_x

        # Map to screen horizontal displacement with gain, limiting extreme overshoot
        # to prevent sticky dead zones when reversing direction from screen borders
        gain_x = max(-1.02, min(1.02, norm_dx * self.config.sensitivity_x))
        target_x = self.screen_center_x + (gain_x * (self.screen_width / 2.0))

        # Clamp horizontal target safely (keep slightly off extreme 0/1 corners to avoid accidental OS corner failsafe)
        clamped_x = max(3.0, min(float(self.screen_width - 4), target_x))

        # Vertical cursor control
        if getattr(self.config, "enable_vertical_cursor", True):
            if abs(raw_dy) <= dz_radius:
                filtered_dy = 0.0
            else:
                excess_y = abs(raw_dy) - dz_radius
                ramp_y = min(1.0, excess_y / max(1e-4, dz_radius * 0.5))
                sign_y = 1.0 if raw_dy > 0 else -1.0
                filtered_dy = sign_y * excess_y * ramp_y

            comfort_y = max(1e-4, getattr(self.config, "comfort_range_y", 0.045))
            norm_dy = filtered_dy / comfort_y
            gain_y = max(-1.02, min(1.02, norm_dy * self.config.sensitivity_y))
            target_y = self.screen_center_y + (gain_y * (self.screen_height / 2.0))
            clamped_y = max(3.0, min(float(self.screen_height - 4), target_y))
        else:
            filtered_dy = 0.0
            clamped_y = float(self.screen_center_y)

        # Always update telemetry tracking with live coordinates so telemetry and HUD remain accurate
        self.last_raw_dx = raw_dx
        self.last_raw_dy = raw_dy
        self.last_calibrated_dx = filtered_dx
        self.last_calibrated_dy = filtered_dy
        self.last_target_x = clamped_x
        self.last_target_y = clamped_y

        # If paused, emergency stopped, or actively scrolling via head pitch: freeze mouse cursor movement
        if self.is_paused or self.is_emergency_stopped or currently_scrolling:
            self.last_mouse_moved = False
            self.last_mouse_suppressed = True
            if self.is_emergency_stopped:
                self.last_mouse_suppress_reason = "EMERGENCY_STOP"
            elif self.is_paused:
                self.last_mouse_suppress_reason = "PAUSED"
            elif currently_scrolling:
                self.last_mouse_suppress_reason = "SCROLLING"
            return None

        # Apply 1-Euro adaptive smoothing filter
        smooth_x, smooth_y = self.point_filter.filter(clamped_x, clamped_y, timestamp)
        screen_coord = (
            max(0, min(self.screen_width - 1, int(round(smooth_x)))),
            max(0, min(self.screen_height - 1, int(round(smooth_y)))),
        )

        mouse_moved = False
        try:
            # Check minimum movement threshold to avoid spamming OS
            dx_px = abs(screen_coord[0] - self.last_cursor_pos[0])
            dy_px = abs(screen_coord[1] - self.last_cursor_pos[1])
            if dx_px >= self.config.cursor_min_movement_px or dy_px >= self.config.cursor_min_movement_px:
                pyautogui.moveTo(screen_coord[0], screen_coord[1], duration=0)
                self.last_cursor_pos = screen_coord
                mouse_moved = True
                self.last_mouse_suppressed = False
                self.last_mouse_suppress_reason = "NONE"
            else:
                self.last_mouse_suppressed = False
                self.last_mouse_suppress_reason = "AT_REST"
        except pyautogui.FailSafeException:
            self.trigger_emergency_stop()
            self.last_mouse_suppressed = True
            self.last_mouse_suppress_reason = "EMERGENCY_STOP"
        except Exception:
            pass

        self.last_filtered_x = float(screen_coord[0])
        self.last_filtered_y = float(screen_coord[1])
        self.last_mouse_moved = mouse_moved

        return screen_coord

    def execute_action(self, action: GestureAction) -> None:
        """Executes cursor clicks and scrolls safely with explicit pipeline tracing."""
        if self.is_paused or self.is_emergency_stopped:
            return

        try:
            if action == GestureAction.LEFT_CLICK:
                print(f"[ACTION] LEFT CLICK at {self.last_cursor_pos}")
                pyautogui.click(button='left')
            elif action == GestureAction.RIGHT_CLICK:
                print(f"[ACTION] RIGHT CLICK at {self.last_cursor_pos}")
                pyautogui.click(button='right')
            elif action == GestureAction.SCROLL_UP:
                pyautogui.scroll(self.config.scroll_speed)
                print(f"[ACTION] SCROLL UP (+{self.config.scroll_speed})")
            elif action == GestureAction.SCROLL_DOWN:
                pyautogui.scroll(-self.config.scroll_speed)
                print(f"[ACTION] SCROLL DOWN (-{self.config.scroll_speed})")
            elif action == GestureAction.OPTIONAL_DRAG_TOGGLE:
                if self.is_dragging:
                    pyautogui.mouseUp(button='left')
                    self.is_dragging = False
                else:
                    pyautogui.mouseDown(button='left')
                    self.is_dragging = True
            elif action == GestureAction.OPTIONAL_PAUSE_TOGGLE:
                self.toggle_pause()
        except pyautogui.FailSafeException:
            self.trigger_emergency_stop()
        except Exception:
            pass

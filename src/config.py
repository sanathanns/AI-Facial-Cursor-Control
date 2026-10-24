"""Configuration module for Facial Cursor Control System."""
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


@dataclass
class AppConfig:
    # Model Configuration (resolved relative to project directory)
    model_path: str = DEFAULT_MODEL_PATH

    # Camera Settings
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    flip_horizontal: bool = True

    # Cursor Sensitivity & Motion (Direct horizontal pointer control)
    sensitivity_x: float = 2.5            # Horizontal movement gain
    sensitivity_y: float = 2.5            # Vertical movement gain (used if enable_vertical_cursor is True)
    comfort_range_x: float = 0.045        # Local normalized movement range for horizontal screen span
    comfort_range_y: float = 0.045        # Local normalized movement range for vertical screen span
    dead_zone_radius: float = 0.008       # Normalized distance from neutral point (low threshold for responsive start)
    cursor_min_movement_px: float = 1.0   # Minimum pixel movement to issue OS moveTo
    enable_vertical_cursor: bool = True   # Enable responsive 2-axis head movement cursor control by default

    # One-Euro Filter Smoothing Parameters (Part 2: responsive tuning)
    cursor_min_cutoff: float = 1.0        # Hz: Frequency cutoff at rest (prevents stationary jitter)
    cursor_beta: float = 0.6              # Speed coefficient: dynamically raises cutoff during movement (eliminates lag)
    cursor_derivative_cutoff: float = 1.0 # Hz: Cutoff for velocity estimation

    # Gesture Thresholds (EAR = Eye Aspect Ratio)
    # Absolute fallbacks when not calibrated
    ear_blink_threshold: float = 0.19     # Below this EAR is considered closed
    ear_open_threshold: float = 0.22      # Above this EAR is considered open
    # Adaptive EAR Ratios (relative to calibrated resting open-eye baseline)
    ear_blink_ratio: float = 0.75         # Closed eye is <= 75% of open baseline (reliable for natural winks)
    ear_open_ratio: float = 0.82          # Open eye is >= 82% of open baseline
    ear_asymmetry_min_diff: float = 0.020 # Minimum EAR difference between closed and open eye

    wink_min_duration_s: float = 0.12     # Minimum duration to count as intentional wink (120ms)
    wink_max_duration_s: float = 0.80     # Maximum duration before it is considered a hold (800ms)
    click_cooldown_s: float = 0.35        # Time between consecutive click events

    # Automatic Head-Pitch Scrolling Parameters (Direct, without mode switches)
    scroll_enabled: bool = True
    scroll_pitch_threshold: float = 0.020 # Head pitch ratio deviation from neutral (triggers automatic scroll)
    scroll_release_ratio: float = 0.65    # Release threshold ratio (0.65 * pitch_threshold = 0.013 prompt resumption)
    scroll_speed: int = 35                # PyAutoGUI scroll units
    scroll_sensitivity: float = 1.0       # Scroll speed multiplier
    scroll_interval_s: float = 0.12       # Rate of repeated scrolling when held (8 ticks/s)

    # Optional Mouth Gestures (MAR = Mouth Aspect Ratio, optional as requested)
    enable_mouth_gestures: bool = False
    mar_threshold: float = 0.55

    # Dwell-Click & Accessibility Settings
    dwell_click_enabled: bool = False
    dwell_duration_s: float = 1.0
    keyboard_dwell_enabled: bool = False
    keyboard_dwell_time_s: float = 1.0
    active_profile_name: str = "Default Profile"

    # Safety & Failsafe Parameters
    face_loss_freeze_s: float = 0.15     # If face is lost for this duration, immediately freeze cursor
    emergency_stop_key: str = "Escape"
    pyautogui_failsafe: bool = True

    # Legacy filter parameter aliases for compatibility
    @property
    def filter_min_cutoff(self) -> float:
        return self.cursor_min_cutoff

    @filter_min_cutoff.setter
    def filter_min_cutoff(self, val: float) -> None:
        self.cursor_min_cutoff = val

    @property
    def filter_beta(self) -> float:
        return self.cursor_beta

    @filter_beta.setter
    def filter_beta(self, val: float) -> None:
        self.cursor_beta = val

    @property
    def filter_d_cutoff(self) -> float:
        return self.cursor_derivative_cutoff

    @filter_d_cutoff.setter
    def filter_d_cutoff(self, val: float) -> None:
        self.cursor_derivative_cutoff = val

    def save_to_file(self, filepath: str = DEFAULT_CONFIG_PATH) -> None:
        """Save config parameters to JSON file."""
        data = asdict(self)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str = DEFAULT_CONFIG_PATH) -> "AppConfig":
        """Load config parameters from JSON file or return defaults if missing."""
        if not os.path.exists(filepath):
            config = cls()
            return config

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Filter unknown keys to remain forward compatible
            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            cfg = cls(**filtered)
            if not os.path.exists(cfg.model_path):
                cfg.model_path = DEFAULT_MODEL_PATH
            return cfg
        except Exception as e:
            print(f"[Config] Warning: Failed to parse {filepath} ({e}), using default config.")
            return cls()

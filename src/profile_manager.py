"""Profile manager for Facial Cursor Control System.

Handles creating, loading, saving, validating, and switching multiple user profiles.
Each profile stores personal accessibility settings and calibration data.
Global safety settings are kept strictly isolated from user profile modification.
"""
import os
import json
import time
import copy
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILES_PATH = os.path.join(PROJECT_ROOT, "profiles.json")

# Default settings template for a new profile
DEFAULT_PROFILE_SETTINGS: Dict[str, Any] = {
    # Cursor sensitivity & movement
    "sensitivity_x": 2.5,
    "sensitivity_y": 2.5,
    "comfort_range_x": 0.045,
    "comfort_range_y": 0.045,
    "dead_zone_radius": 0.008,
    "cursor_min_movement_px": 2.0,
    "enable_vertical_cursor": True,

    # 1-Euro Filter smoothing
    "cursor_min_cutoff": 1.0,
    "cursor_beta": 0.08,
    "cursor_derivative_cutoff": 1.0,

    # Eye Gestures
    "ear_blink_threshold": 0.19,
    "ear_open_threshold": 0.22,
    "ear_blink_ratio": 0.75,
    "ear_open_ratio": 0.82,
    "ear_asymmetry_min_diff": 0.020,
    "wink_min_duration_s": 0.12,
    "wink_max_duration_s": 0.80,
    "click_cooldown_s": 0.35,

    # Scrolling
    "scroll_enabled": True,
    "scroll_pitch_threshold": 0.020,
    "scroll_release_ratio": 0.45,
    "scroll_speed": 35,
    "scroll_sensitivity": 1.0,
    "scroll_interval_s": 0.12,

    # Optional Mouth Gestures
    "enable_mouth_gestures": False,
    "mar_threshold": 0.55,

    # Dwell-Click Settings
    "dwell_click_enabled": False,
    "dwell_duration_s": 1.0,

    # On-Screen Keyboard Preferences
    "keyboard_dwell_enabled": False,
    "keyboard_dwell_time_s": 1.0,
}

# Bounds for validating user profile values to prevent extreme/unsafe behavior
VALIDATION_BOUNDS = {
    "sensitivity_x": (0.1, 10.0),
    "sensitivity_y": (0.1, 10.0),
    "comfort_range_x": (0.01, 0.20),
    "comfort_range_y": (0.01, 0.20),
    "dead_zone_radius": (0.001, 0.050),
    "cursor_min_movement_px": (0.1, 10.0),
    "cursor_min_cutoff": (0.05, 5.0),
    "cursor_beta": (0.01, 5.0),
    "cursor_derivative_cutoff": (0.1, 5.0),
    "ear_blink_threshold": (0.10, 0.35),
    "ear_open_threshold": (0.15, 0.45),
    "ear_blink_ratio": (0.50, 0.90),
    "ear_open_ratio": (0.60, 0.95),
    "ear_asymmetry_min_diff": (0.005, 0.100),
    "wink_min_duration_s": (0.05, 0.50),
    "wink_max_duration_s": (0.30, 2.00),
    "click_cooldown_s": (0.10, 1.50),
    "scroll_pitch_threshold": (0.005, 0.100),
    "scroll_speed": (5, 150),
    "scroll_sensitivity": (0.2, 5.0),
    "scroll_interval_s": (0.03, 0.50),
    "mar_threshold": (0.20, 0.90),
    "dwell_duration_s": (0.3, 4.0),
    "keyboard_dwell_time_s": (0.3, 4.0),
}


@dataclass
class UserProfile:
    """Represents a single user profile with custom settings and calibration."""
    profile_id: str
    name: str
    created_at: float
    updated_at: float
    settings: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PROFILE_SETTINGS))
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "is_calibrated": False,
        "neutral_x": None,
        "neutral_y": None,
        "neutral_pitch": None,
        "baseline_left_ear": None,
        "baseline_right_ear": None,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            profile_id=data.get("profile_id", data.get("id", "default")),
            name=data.get("name", "Default"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            settings=data.get("settings", copy.deepcopy(DEFAULT_PROFILE_SETTINGS)),
            calibration=data.get("calibration", {}),
        )


class ProfileManager:
    """Manages loading, validating, persisting, creating, and switching user profiles."""

    def __init__(
        self,
        config: Optional[Any] = None,
        filepath: Optional[str] = None,
        profiles_path: Optional[str] = None,
    ):
        self.config = config
        self.filepath = profiles_path or filepath or DEFAULT_PROFILES_PATH
        self.profiles: Dict[str, UserProfile] = {}
        self.active_profile_id: str = "default"
        self.load()

    def _create_default_profile(self) -> UserProfile:
        return UserProfile(
            profile_id="default",
            name="Default",
            created_at=time.time(),
            updated_at=time.time(),
            settings=copy.deepcopy(DEFAULT_PROFILE_SETTINGS),
            calibration={
                "is_calibrated": False,
                "neutral_x": None,
                "neutral_y": None,
                "neutral_pitch": None,
                "baseline_left_ear": None,
                "baseline_right_ear": None,
            },
        )

    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes and clamps settings to safe bounds, supplying defaults for missing keys."""
        cleaned = copy.deepcopy(DEFAULT_PROFILE_SETTINGS)
        if not isinstance(settings, dict):
            return cleaned

        for k, v in settings.items():
            if k in VALIDATION_BOUNDS:
                min_val, max_val = VALIDATION_BOUNDS[k]
                try:
                    num_val = float(v)
                    if min_val <= num_val <= max_val:
                        if isinstance(DEFAULT_PROFILE_SETTINGS[k], int):
                            cleaned[k] = int(round(num_val))
                        else:
                            cleaned[k] = num_val
                    else:
                        clamped = max(min_val, min(max_val, num_val))
                        cleaned[k] = int(round(clamped)) if isinstance(DEFAULT_PROFILE_SETTINGS[k], int) else clamped
                except (ValueError, TypeError):
                    pass
            elif k in cleaned and isinstance(cleaned[k], bool):
                cleaned[k] = bool(v)

        return cleaned

    def _validate_calibration(self, calib: Any) -> Dict[str, Any]:
        """Validates calibration fields."""
        default_calib = {
            "is_calibrated": False,
            "neutral_x": None,
            "neutral_y": None,
            "neutral_pitch": None,
            "baseline_left_ear": None,
            "baseline_right_ear": None,
        }
        if not isinstance(calib, dict):
            return default_calib

        validated = dict(default_calib)
        validated["is_calibrated"] = bool(calib.get("is_calibrated", False))

        for coord in ["neutral_x", "neutral_y"]:
            val = calib.get(coord)
            if val is not None:
                try:
                    fval = float(val)
                    if 0.05 <= fval <= 0.95:
                        validated[coord] = fval
                except (ValueError, TypeError):
                    pass

        pitch = calib.get("neutral_pitch")
        if pitch is not None:
            try:
                fp = float(pitch)
                if 0.1 <= fp <= 0.9:
                    validated["neutral_pitch"] = fp
            except (ValueError, TypeError):
                pass

        for ear in ["baseline_left_ear", "baseline_right_ear"]:
            val = calib.get(ear)
            if val is not None:
                try:
                    fe = float(val)
                    if 0.10 <= fe <= 0.50:
                        validated[ear] = fe
                except (ValueError, TypeError):
                    pass

        if validated["is_calibrated"] and (
            validated["neutral_x"] is None or validated["neutral_y"] is None or validated["neutral_pitch"] is None
        ):
            validated["is_calibrated"] = False

        return validated

    def load(self) -> None:
        """Loads profiles from JSON, validating each and falling back safely if corrupted."""
        self.profiles = {}
        if not os.path.exists(self.filepath):
            default_p = self._create_default_profile()
            self.profiles[default_p.profile_id] = default_p
            self.active_profile_id = default_p.profile_id
            self.save()
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Profiles file root must be an object")

            raw_profiles = data.get("profiles", {})
            if not isinstance(raw_profiles, dict) or len(raw_profiles) == 0:
                raise ValueError("No valid profiles found")

            for pid, pdata in raw_profiles.items():
                if not isinstance(pdata, dict):
                    continue
                name = str(pdata.get("name", pid)).strip() or pid
                created_at = float(pdata.get("created_at", time.time()))
                updated_at = float(pdata.get("updated_at", time.time()))
                settings = self._validate_settings(pdata.get("settings", {}))
                calibration = self._validate_calibration(pdata.get("calibration", {}))

                self.profiles[pid] = UserProfile(
                    profile_id=pid,
                    name=name,
                    created_at=created_at,
                    updated_at=updated_at,
                    settings=settings,
                    calibration=calibration,
                )

            active_id = str(data.get("active_profile", "")).strip()
            # Match active profile either by ID or Name
            found_id = None
            if active_id in self.profiles:
                found_id = active_id
            else:
                for pid, p in self.profiles.items():
                    if p.name == active_id:
                        found_id = pid
                        break

            if found_id:
                self.active_profile_id = found_id
            elif self.profiles:
                self.active_profile_id = next(iter(self.profiles.keys()))
            else:
                default_p = self._create_default_profile()
                self.profiles[default_p.profile_id] = default_p
                self.active_profile_id = default_p.profile_id

        except Exception as e:
            print(f"[ProfileManager] Warning: Corrupted profiles file ({e}). Backing up and resetting to default.")
            try:
                if os.path.exists(self.filepath):
                    backup_path = f"{self.filepath}.corrupt"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(self.filepath, backup_path)
            except Exception as b_err:
                print(f"[ProfileManager] Error creating backup: {b_err}")
            default_p = self._create_default_profile()
            self.profiles[default_p.profile_id] = default_p
            self.active_profile_id = default_p.profile_id
            self.save()

    def save(self) -> bool:
        """Persists all profiles and the active profile ID to disk atomically."""
        try:
            data = {
                "active_profile": self.active_profile_id,
                "profiles": {
                    pid: {
                        "id": p.profile_id,
                        "name": p.name,
                        "created_at": p.created_at,
                        "updated_at": p.updated_at,
                        "settings": p.settings,
                        "calibration": p.calibration,
                    }
                    for pid, p in self.profiles.items()
                },
            }
            temp_path = f"{self.filepath}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            if os.path.exists(self.filepath):
                os.replace(temp_path, self.filepath)
            else:
                os.rename(temp_path, self.filepath)
            return True
        except Exception as e:
            print(f"[ProfileManager] Error saving profiles: {e}")
            return False

    def list_profiles(self) -> List[str]:
        """Returns sorted list of user-facing profile names."""
        return [p.name for p in self.profiles.values()]

    def get_profile_names(self) -> List[str]:
        """Alias for list_profiles()."""
        return self.list_profiles()

    def get_active_profile(self) -> UserProfile:
        """Returns the currently active profile (guaranteed to exist)."""
        if self.active_profile_id not in self.profiles:
            if not self.profiles:
                default_p = self._create_default_profile()
                self.profiles[default_p.profile_id] = default_p
            self.active_profile_id = next(iter(self.profiles.keys()))
        return self.profiles[self.active_profile_id]

    def get_active_profile_name(self) -> str:
        """Returns the user-facing name of the currently active profile."""
        return self.get_active_profile().name

    def get_profile(self, profile_id_or_name: str) -> Optional[UserProfile]:
        """Finds a profile by ID or by user-facing name."""
        if profile_id_or_name in self.profiles:
            return self.profiles[profile_id_or_name]
        return self.get_profile_by_name(profile_id_or_name)

    def get_profile_by_name(self, name: str) -> Optional[UserProfile]:
        """Finds a profile by its user-facing name (case-insensitive)."""
        name_lower = name.strip().lower()
        for p in self.profiles.values():
            if p.name.strip().lower() == name_lower:
                return p
        return None

    def create_profile(
        self,
        name: str,
        clone_active: bool = True,
        copy_from_active: bool = True,
    ) -> UserProfile:
        """Creates a new user profile with a unique ID."""
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Profile name cannot be empty")

        existing_names = [p.name.lower() for p in self.profiles.values()]
        if cleaned_name.lower() in existing_names:
            raise ValueError(f"Profile '{cleaned_name}' already exists")

        import uuid
        new_id = f"user_{uuid.uuid4().hex[:8]}"

        should_copy = clone_active or copy_from_active
        active = self.get_active_profile() if should_copy else None
        settings = copy.deepcopy(active.settings) if active else copy.deepcopy(DEFAULT_PROFILE_SETTINGS)

        new_profile = UserProfile(
            profile_id=new_id,
            name=cleaned_name,
            created_at=time.time(),
            updated_at=time.time(),
            settings=settings,
            calibration={
                "is_calibrated": False,
                "neutral_x": None,
                "neutral_y": None,
                "neutral_pitch": None,
                "baseline_left_ear": None,
                "baseline_right_ear": None,
            },
        )
        self.profiles[new_id] = new_profile
        self.save()
        return new_profile

    def switch_profile(self, profile_id_or_name: str) -> bool:
        """Switches the active profile by ID or name and updates config if present."""
        target_prof = None
        if profile_id_or_name in self.profiles:
            target_prof = self.profiles[profile_id_or_name]
        else:
            target_prof = self.get_profile_by_name(profile_id_or_name)

        if not target_prof:
            return False

        self.active_profile_id = target_prof.profile_id
        if self.config:
            self.config.active_profile_name = target_prof.name
            for k, v in target_prof.settings.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        self.save()
        return True

    def rename_profile(self, profile_id_or_name: str, new_name: str) -> bool:
        """Renames an existing profile safely. Protects Default profile."""
        cleaned_name = new_name.strip()
        if not cleaned_name:
            raise ValueError("New profile name cannot be empty")

        p = self.get_profile(profile_id_or_name)
        if not p:
            return False

        if p.profile_id == "default" or p.name.lower() == "default":
            raise ValueError("Cannot rename the 'Default' profile")

        for other_id, other_p in self.profiles.items():
            if other_id != p.profile_id and other_p.name.lower() == cleaned_name.lower():
                raise ValueError(f"Profile '{cleaned_name}' already exists")

        p.name = cleaned_name
        p.updated_at = time.time()
        if self.config and self.active_profile_id == p.profile_id:
            self.config.active_profile_name = cleaned_name

        self.save()
        return True

    def delete_profile(self, profile_id_or_name: str) -> bool:
        """Deletes a profile with safety: cannot delete Default or the last profile."""
        p = self.get_profile(profile_id_or_name)
        if not p:
            return False

        if p.profile_id == "default" or p.name.lower() == "default":
            raise ValueError("Cannot delete the 'Default' profile")

        if len(self.profiles) <= 1:
            raise ValueError("Cannot delete the only remaining profile")

        pid = p.profile_id
        del self.profiles[pid]

        if self.active_profile_id == pid:
            self.active_profile_id = "default" if "default" in self.profiles else next(iter(self.profiles.keys()))
            if self.config:
                active = self.get_active_profile()
                self.config.active_profile_name = active.name
                for k, v in active.settings.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)

        self.save()
        return True

    def save_active_profile_settings(self) -> None:
        """Captures settings from self.config (if provided) and persists them to active profile."""
        if not self.config:
            self.save()
            return

        active = self.get_active_profile()
        for k in DEFAULT_PROFILE_SETTINGS.keys():
            if hasattr(self.config, k):
                active.settings[k] = getattr(self.config, k)
        active.updated_at = time.time()
        self.save()

    def update_active_settings(self, settings_dict: Dict[str, Any]) -> None:
        """Updates and persists settings for the active profile."""
        active = self.get_active_profile()
        valid = self._validate_settings(settings_dict)
        active.settings.update(valid)
        active.updated_at = time.time()
        self.save()

    def update_active_profile_calibration(
        self,
        neutral_x: float,
        neutral_y: float,
        neutral_pitch: float,
        baseline_left_ear: float,
        baseline_right_ear: float,
    ) -> None:
        """Updates and persists neutral calibration data for the active profile."""
        self.update_active_calibration(
            neutral_x=neutral_x,
            neutral_y=neutral_y,
            neutral_pitch=neutral_pitch,
            baseline_left_ear=baseline_left_ear,
            baseline_right_ear=baseline_right_ear,
        )

    def update_active_calibration(
        self,
        neutral_x: float,
        neutral_y: float,
        neutral_pitch: float,
        baseline_left_ear: float,
        baseline_right_ear: float,
    ) -> None:
        """Updates and persists neutral calibration data for the active profile."""
        active = self.get_active_profile()
        active.calibration = self._validate_calibration({
            "is_calibrated": True,
            "neutral_x": neutral_x,
            "neutral_y": neutral_y,
            "neutral_pitch": neutral_pitch,
            "baseline_left_ear": baseline_left_ear,
            "baseline_right_ear": baseline_right_ear,
        })
        active.updated_at = time.time()
        self.save()

    def clear_active_calibration(self) -> None:
        """Resets calibration for the active profile."""
        active = self.get_active_profile()
        active.calibration = {
            "is_calibrated": False,
            "neutral_x": None,
            "neutral_y": None,
            "neutral_pitch": None,
            "baseline_left_ear": None,
            "baseline_right_ear": None,
        }
        active.updated_at = time.time()
        self.save()

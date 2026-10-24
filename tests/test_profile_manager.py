"""Unit tests for User Profiles and ProfileManager."""
import os
import json
import tempfile
import pytest

from src.config import AppConfig
from src.profile_manager import ProfileManager, UserProfile, VALIDATION_BOUNDS


@pytest.fixture
def temp_profile_env(tmp_path):
    """Fixture providing a temporary filepath for profiles and an AppConfig."""
    profiles_path = str(tmp_path / "test_profiles.json")
    config = AppConfig()
    return config, profiles_path


def test_default_profile_creation(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    assert pm.get_active_profile_name() == "Default"
    assert "Default" in pm.list_profiles()
    assert os.path.exists(path)

    prof = pm.get_active_profile()
    assert prof is not None
    assert prof.name == "Default"
    assert "sensitivity_x" in prof.settings
    assert "sensitivity_y" in prof.settings


def test_create_and_switch_profile(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    new_prof = pm.create_profile("User_Alice")
    assert "User_Alice" in pm.list_profiles()
    assert new_prof.name == "User_Alice"

    # Switch to User_Alice
    assert pm.switch_profile("User_Alice") is True
    assert pm.get_active_profile_name() == "User_Alice"
    assert config.active_profile_name == "User_Alice"

    # Switch to non-existent profile returns False
    assert pm.switch_profile("NonExistent") is False
    assert pm.get_active_profile_name() == "User_Alice"


def test_cannot_create_duplicate_profile(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    pm.create_profile("Bob")
    with pytest.raises(ValueError, match="already exists"):
        pm.create_profile("Bob")


def test_rename_profile(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    pm.create_profile("Charlie")
    pm.switch_profile("Charlie")

    # Renaming Default should fail
    with pytest.raises(ValueError, match="Cannot rename the 'Default' profile"):
        pm.rename_profile("Default", "Primary")

    # Rename Charlie to Charles
    assert pm.rename_profile("Charlie", "Charles") is True
    assert "Charles" in pm.list_profiles()
    assert "Charlie" not in pm.list_profiles()
    assert pm.get_active_profile_name() == "Charles"


def test_delete_profile(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    pm.create_profile("Dave")
    pm.switch_profile("Dave")

    # Cannot delete Default
    with pytest.raises(ValueError, match="Cannot delete the 'Default' profile"):
        pm.delete_profile("Default")

    # Delete Dave
    assert pm.delete_profile("Dave") is True
    assert "Dave" not in pm.list_profiles()
    # Active profile should fall back to Default
    assert pm.get_active_profile_name() == "Default"


def test_persistence_across_instances(temp_profile_env):
    config, path = temp_profile_env
    pm1 = ProfileManager(config, profiles_path=path)
    pm1.create_profile("PersistentUser")
    pm1.switch_profile("PersistentUser")

    config.sensitivity_x = 5.5
    config.comfort_range_x = 0.08
    pm1.save_active_profile_settings()

    # Create a new ProfileManager pointing to the same file
    config2 = AppConfig()
    pm2 = ProfileManager(config2, profiles_path=path)

    assert pm2.get_active_profile_name() == "PersistentUser"
    prof = pm2.get_active_profile()
    assert prof.settings["sensitivity_x"] == 5.5
    assert prof.settings["comfort_range_x"] == 0.08


def test_calibration_storage_per_profile(temp_profile_env):
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    pm.create_profile("Gamer")
    pm.switch_profile("Gamer")

    pm.update_active_profile_calibration(
        neutral_x=0.485,
        neutral_y=0.512,
        neutral_pitch=0.420,
        baseline_left_ear=0.28,
        baseline_right_ear=0.29,
    )

    prof = pm.get_active_profile()
    assert prof.calibration["is_calibrated"] is True
    assert prof.calibration["neutral_x"] == 0.485
    assert prof.calibration["neutral_pitch"] == 0.420

    # Default profile calibration should remain uncalibrated
    default_prof = pm.get_profile("Default")
    assert default_prof.calibration.get("is_calibrated") is False


def test_bounds_validation_and_clamping(temp_profile_env):
    config, path = temp_profile_env

    # Write a profiles.json with extreme out-of-bounds values
    corrupt_data = {
        "active_profile": "Default",
        "profiles": {
            "Default": {
                "name": "Default",
                "settings": {
                    "sensitivity_x": 999.0,  # Max allowed is 10.0
                    "sensitivity_y": -50.0,  # Min allowed is 0.1
                    "dead_zone_radius": 5.0,  # Max allowed is 0.1
                },
                "calibration": {},
            }
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corrupt_data, f)

    pm = ProfileManager(config, profiles_path=path)
    prof = pm.get_active_profile()

    # Verify clamping
    min_x, max_x = VALIDATION_BOUNDS["sensitivity_x"]
    assert prof.settings["sensitivity_x"] == max_x
    min_y, max_y = VALIDATION_BOUNDS["sensitivity_y"]
    assert prof.settings["sensitivity_y"] == min_y
    _, max_dz = VALIDATION_BOUNDS["dead_zone_radius"]
    assert prof.settings["dead_zone_radius"] == max_dz


def test_corrupted_json_recovery(temp_profile_env):
    config, path = temp_profile_env

    # Write broken non-JSON garbage
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken_json: None, NOT_VALID")

    # Manager should gracefully detect corruption, create a .corrupt backup, and initialize Default
    pm = ProfileManager(config, profiles_path=path)
    assert pm.get_active_profile_name() == "Default"
    assert "Default" in pm.list_profiles()

    # Check that backup file was created
    backup_file = f"{path}.corrupt"
    assert os.path.exists(backup_file)


def test_safety_settings_isolated_from_profiles(temp_profile_env):
    """Verify emergency stop and failsafe settings cannot be corrupted by user profile settings."""
    config, path = temp_profile_env
    pm = ProfileManager(config, profiles_path=path)

    prof = pm.get_active_profile()
    # Profile settings must only contain ergonomic / accessibility values, NOT safety overrides
    assert "failsafe_enabled" not in prof.settings
    assert "emergency_stop_key" not in prof.settings
    assert "model_path" not in prof.settings

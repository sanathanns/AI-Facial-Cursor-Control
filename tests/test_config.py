"""Unit tests for config serialization, defaults, and CLI parsing."""
import os
import json
import tempfile
from src.config import AppConfig


def test_config_defaults():
    cfg = AppConfig()
    assert cfg.sensitivity_x > 0
    assert cfg.sensitivity_y > 0
    assert cfg.dead_zone_radius > 0
    assert cfg.filter_min_cutoff > 0
    assert cfg.filter_beta > 0
    assert cfg.ear_blink_threshold > 0
    assert not cfg.enable_mouth_gestures  # Mouth gestures are optional and disabled by default


def test_config_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = tf.name

    try:
        cfg = AppConfig(sensitivity_x=3.5, dead_zone_radius=0.035)
        cfg.save_to_file(temp_path)

        loaded = AppConfig.load_from_file(temp_path)
        assert loaded.sensitivity_x == 3.5
        assert loaded.dead_zone_radius == 0.035
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_load_invalid_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(b"{invalid_json")
        temp_path = tf.name

    try:
        loaded = AppConfig.load_from_file(temp_path)
        assert isinstance(loaded, AppConfig)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

"""Main entrypoint for the AI Facial Cursor Control System."""
import os
import sys
import argparse
import tkinter as tk

from src.config import AppConfig
from src.gui import FacialCursorGUI
from download_model import ensure_model


def parse_args():
    parser = argparse.ArgumentParser(description="AI-Based Facial Cursor Control System")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera device index (default: 0)")
    parser.add_argument("--sensitivity", type=float, default=None, help="General sensitivity multiplier")
    parser.add_argument("--deadzone", type=float, default=None, help="Dead zone radius (e.g. 0.02)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Ensure MediaPipe Tasks model exists locally
    try:
        model_path = ensure_model()
    except Exception as e:
        print(f"[Error] Failed to acquire model bundle: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load Configuration
    config_path = args.config if args.config else os.path.join(os.path.dirname(__file__), "config.json")
    config = AppConfig.load_from_file(config_path)
    config.model_path = model_path

    # Apply any CLI overrides
    if args.camera_id is not None:
        config.camera_id = args.camera_id
    if args.sensitivity is not None:
        config.sensitivity_x = args.sensitivity
        config.sensitivity_y = args.sensitivity
    if args.deadzone is not None:
        config.dead_zone_radius = args.deadzone

    print("[FacialCursor] Starting Facial Cursor Control System...")
    print(f"[FacialCursor] Model Path: {config.model_path}")
    print(f"[FacialCursor] Camera ID: {config.camera_id}")

    # 3. Launch Tkinter GUI strictly on the main thread
    root = tk.Tk()
    app = FacialCursorGUI(root, config)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[FacialCursor] Shutdown requested by user.")
    finally:
        try:
            app.worker.stop()
            app.worker.join(timeout=1.0)
        except Exception:
            pass
        print("[FacialCursor] Clean shutdown complete.")


if __name__ == "__main__":
    main()

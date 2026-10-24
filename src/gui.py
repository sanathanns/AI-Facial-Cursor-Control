"""Graphical User Interface using Tkinter (strictly accessed from main GUI thread)."""
import time
import queue
import threading
from typing import Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from src.config import AppConfig
from src.camera import CameraManager
from src.face_tracker import FaceTracker
from src.filters import PointOneEuroFilter, DeadZoneFilter
from src.gestures import GestureRecognizer, GestureAction, GestureDebugInfo, ScrollState
from src.cursor_controller import CursorController
from src.profile_manager import ProfileManager, UserProfile
from src.on_screen_keyboard import OnScreenKeyboard
from src.diagnostics import DiagnosticsDashboard


class VisionWorker(threading.Thread):
    """Background worker thread that runs camera acquisition, tracking, and cursor control.

    Communicates with the main Tkinter thread strictly via thread-safe queues.
    """

    def __init__(self, config: AppConfig, frame_queue: queue.Queue, command_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.command_queue = command_queue

        self.camera = CameraManager(config)
        self.face_tracker: Optional[FaceTracker] = None
        self.gestures = GestureRecognizer(config)
        self.controller = CursorController(config)

        self._running = False
        self._request_calibration = False
        self._request_eye_calibration = False

        # Multi-frame stable calibration buffers (filters out single-frame startup jitter)
        self._calib_frames_target: int = 10
        self._calib_nose_x: list = []
        self._calib_nose_y: list = []
        self._calib_pitch: list = []
        self._calib_l_ear: list = []
        self._calib_r_ear: list = []
        self._is_calibrating: bool = True  # Capture stable baseline on startup
        self._just_calibrated_payload: Optional[Dict[str, Any]] = None

    def _handle_switch_profile(self, data: Dict[str, Any]) -> None:
        """Safely applies profile settings and calibration to worker components."""
        # 1. Release any mouse button or dragging safely
        self.controller.release_drag()

        # 2. Reset smoothing filters and gesture trackers
        self.controller.point_filter.reset()
        self.gestures.reset_trackers()

        # 3. Apply profile settings to self.config
        settings = data.get("settings", {})
        for k, v in settings.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.controller.update_config_params()

        # 4. Apply or request calibration
        calib = data.get("calibration", {})
        if (
            calib
            and calib.get("is_calibrated")
            and calib.get("neutral_x") is not None
            and calib.get("neutral_y") is not None
        ):
            self.controller.calibrate(calib["neutral_x"], calib["neutral_y"])
            if calib.get("neutral_pitch") is not None:
                self.gestures.neutral_pitch = calib["neutral_pitch"]
                self.gestures.is_pitch_calibrated = True
                self.gestures.scroll_state = ScrollState.NEUTRAL
                self.gestures._last_scroll_time = 0.0
            if calib.get("baseline_left_ear") and calib.get("baseline_right_ear"):
                self.gestures.calibrate_eyes(calib["baseline_left_ear"], calib["baseline_right_ear"])
            self._is_calibrating = False
            self._request_calibration = False
        else:
            self.controller.reset_neutral()
            self.gestures.is_pitch_calibrated = False
            self.request_calibration()

    def request_calibration(self) -> None:
        self._request_calibration = True
        self._is_calibrating = True
        self._calib_nose_x.clear()
        self._calib_nose_y.clear()
        self._calib_pitch.clear()
        self._calib_l_ear.clear()
        self._calib_r_ear.clear()

    def request_eye_calibration(self) -> None:
        self._request_eye_calibration = True

    def toggle_pause(self) -> bool:
        return self.controller.toggle_pause()

    def emergency_stop(self) -> None:
        self.controller.trigger_emergency_stop()

    def resume_from_stop(self) -> None:
        self.controller.resume_from_emergency_stop()

    def _handle_recalibrate_camera(self) -> None:
        """Flushes camera buffers, resets tracking timestamp, and requests recalibration."""
        try:
            if self.face_tracker:
                self.face_tracker._frame_timestamp_ms = 0
            self.request_calibration()
            self._calib_status_text = "Camera tracking reinitialized"
        except Exception as e:
            print(f"[VisionWorker] Camera recalibrate error: {e}")

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True

        try:
            self.face_tracker = FaceTracker(self.config.model_path)
        except Exception as e:
            self.frame_queue.put({"error": f"Failed to initialize FaceTracker: {e}"})
            return

        if not self.camera.start():
            self.frame_queue.put({"error": f"Failed to start camera (ID {self.config.camera_id})"})
            return

        try:
            fps_timer = time.monotonic()
            frame_count = 0
            current_fps = 0.0

            while self._running:
                try:
                    # Process any commands from GUI thread
                    while not self.command_queue.empty():
                        try:
                            cmd, data = self.command_queue.get_nowait()
                            if cmd == "CALIBRATE":
                                self.request_calibration()
                            elif cmd == "CALIBRATE_EYES":
                                self._request_eye_calibration = True
                                self._calib_l_ear.clear()
                                self._calib_r_ear.clear()
                            elif cmd == "RESET_NEUTRAL":
                                self.controller.reset_neutral()
                                self.gestures.is_pitch_calibrated = False
                                self.request_calibration()
                            elif cmd == "RECALIBRATE_CAMERA":
                                self._handle_recalibrate_camera()
                            elif cmd == "TOGGLE_PAUSE":
                                self.controller.toggle_pause()
                            elif cmd == "EMERGENCY_STOP":
                                self.controller.trigger_emergency_stop()
                            elif cmd == "RESUME":
                                self.controller.resume_from_emergency_stop()
                            elif cmd == "UPDATE_CONFIG":
                                self.controller.update_config_params()
                            elif cmd == "SWITCH_PROFILE":
                                self._handle_switch_profile(data)
                        except queue.Empty:
                            break

                    # Get newest available frame with capture timestamp (Part 1: zero buffering)
                    success, frame, frame_ts = self.camera.get_latest_frame_with_timestamp()
                    if not success or frame is None:
                        time.sleep(0.005)
                        continue

                    frame_count += 1
                    now = time.monotonic()
                    latency_ms = (now - frame_ts) * 1000.0 if frame_ts > 0 else 0.0

                    if now - fps_timer >= 1.0:
                        current_fps = frame_count / (now - fps_timer)
                        frame_count = 0
                        fps_timer = now

                    h, w, _ = frame.shape
                    tracking_result = self.face_tracker.process_frame(frame)

                    active_action = GestureAction.NONE
                    cursor_pos = None

                    if tracking_result.face_detected and tracking_result.nose_tip is not None:
                        nose_x, nose_y = tracking_result.nose_tip

                        # Multi-frame stable calibration (filters out single-frame startup jitter & posture settling)
                        if (not self.controller.is_calibrated) or self._is_calibrating or self._request_calibration:
                            self._is_calibrating = True
                            self._calib_nose_x.append(nose_x)
                            self._calib_nose_y.append(nose_y)
                            if tracking_result.mid_eyes is not None:
                                from src.gestures import calculate_head_pitch_ratio
                                p = calculate_head_pitch_ratio(
                                    tracking_result.nose_tip,
                                    tracking_result.mid_eyes,
                                    tracking_result.chin,
                                )
                                self._calib_pitch.append(p)
                            if tracking_result.left_eye and tracking_result.right_eye:
                                from src.gestures import calculate_ear
                                self._calib_l_ear.append(calculate_ear(tracking_result.left_eye))
                                self._calib_r_ear.append(calculate_ear(tracking_result.right_eye))

                            if len(self._calib_nose_x) >= self._calib_frames_target:
                                import statistics
                                med_nx = float(statistics.median(self._calib_nose_x))
                                med_ny = float(statistics.median(self._calib_nose_y))
                                self.controller.calibrate(med_nx, med_ny)

                                if self._calib_pitch:
                                    med_pitch = float(statistics.median(self._calib_pitch))
                                    self.gestures.neutral_pitch = med_pitch
                                    self.gestures.is_pitch_calibrated = True
                                    self.gestures.scroll_state = ScrollState.NEUTRAL
                                    self.gestures._last_scroll_time = 0.0

                                if self._calib_l_ear and self._calib_r_ear:
                                    med_lear = float(statistics.median(self._calib_l_ear))
                                    med_rear = float(statistics.median(self._calib_r_ear))
                                    self.gestures.calibrate_eyes(med_lear, med_rear)

                                self._is_calibrating = False
                                self._request_calibration = False
                                self._calib_nose_x.clear()
                                self._calib_nose_y.clear()
                                self._calib_pitch.clear()
                                self._calib_l_ear.clear()
                                self._calib_r_ear.clear()
                                print(f"[CALIBRATION] Stable neutral calibration complete over {self._calib_frames_target} frames.")
                                print(f"[CALIBRATION] Neutral nose: ({med_nx:.4f}, {med_ny:.4f}), Pitch: {getattr(self.gestures, 'neutral_pitch', 0.0):.4f}")
                                self._just_calibrated_payload = {
                                    "neutral_x": med_nx,
                                    "neutral_y": med_ny,
                                    "neutral_pitch": getattr(self.gestures, "neutral_pitch", 0.400),
                                    "baseline_left_ear": getattr(self.gestures, "baseline_left_ear", 0.25),
                                    "baseline_right_ear": getattr(self.gestures, "baseline_right_ear", 0.25),
                                }

                        if self._request_eye_calibration:
                            if tracking_result.left_eye and tracking_result.right_eye:
                                from src.gestures import calculate_ear
                                self._calib_l_ear.append(calculate_ear(tracking_result.left_eye))
                                self._calib_r_ear.append(calculate_ear(tracking_result.right_eye))
                                if len(self._calib_l_ear) >= self._calib_frames_target:
                                    import statistics
                                    med_lear = float(statistics.median(self._calib_l_ear))
                                    med_rear = float(statistics.median(self._calib_r_ear))
                                    self.gestures.calibrate_eyes(med_lear, med_rear)
                                    self._calib_l_ear.clear()
                                    self._calib_r_ear.clear()
                                    self._request_eye_calibration = False
                                    self._calib_status_text = "Eye calibration complete"
                                    print(f"[CALIBRATION] Eye baseline calibrated over {self._calib_frames_target} frames: L={med_lear:.3f}, R={med_rear:.3f}")

                        if self._is_calibrating or self._request_eye_calibration:
                            # While accumulating calibration frames, suppress cursor motion and gestures
                            active_action, left_ear, right_ear, mar, debug_info = self.gestures.update(
                                face_detected=True,
                                left_eye=tracking_result.left_eye,
                                right_eye=tracking_result.right_eye,
                                nose_tip=tracking_result.nose_tip,
                                mid_eyes=tracking_result.mid_eyes,
                                chin=tracking_result.chin,
                                mouth=tracking_result.mouth,
                                is_calibrated=False,
                                is_paused=self.controller.is_paused,
                                is_emergency_stopped=self.controller.is_emergency_stopped,
                                current_time=now,
                            )
                            active_action = GestureAction.NONE
                            cursor_pos = None
                            self.controller.last_mouse_moved = False
                            self.controller.last_mouse_suppressed = True
                            if self._is_calibrating:
                                self.controller.last_mouse_suppress_reason = f"CALIBRATING ({len(self._calib_nose_x)}/{self._calib_frames_target})"
                            else:
                                self.controller.last_mouse_suppress_reason = f"CALIBRATING_EYES ({len(self._calib_l_ear)}/{self._calib_frames_target})"
                        else:
                            # 1. Update gestures first (detects winks, head pitch, and scroll states)
                            active_action, left_ear, right_ear, mar, debug_info = self.gestures.update(
                                face_detected=True,
                                left_eye=tracking_result.left_eye,
                                right_eye=tracking_result.right_eye,
                                nose_tip=tracking_result.nose_tip,
                                mid_eyes=tracking_result.mid_eyes,
                                chin=tracking_result.chin,
                                mouth=tracking_result.mouth,
                                is_calibrated=self.controller.is_calibrated,
                                is_paused=self.controller.is_paused,
                                is_emergency_stopped=self.controller.is_emergency_stopped,
                                current_time=now,
                            )

                            is_scrolling = (debug_info.scroll_state != "NEUTRAL")

                            # 2. Update cursor position (automatically freezes while head is tilted to scroll)
                            cursor_pos = self.controller.update_cursor_position(
                                nose_x, nose_y, now, is_scrolling=is_scrolling
                            )

                            # 3. Execute actions (Clicks & Scrolls)
                            if active_action == GestureAction.LEFT_CLICK:
                                print("[WORKER] CLICK_LEFT RECEIVED")
                                self.controller.execute_action(active_action)
                            elif active_action == GestureAction.RIGHT_CLICK:
                                print("[WORKER] CLICK_RIGHT RECEIVED")
                                self.controller.execute_action(active_action)
                            elif active_action != GestureAction.NONE:
                                self.controller.execute_action(active_action)
                    else:
                        # Face lost: trigger immediate safety reset & freeze
                        if self._is_calibrating or self._request_eye_calibration:
                            self._calib_nose_x.clear()
                            self._calib_nose_y.clear()
                            self._calib_pitch.clear()
                            self._calib_l_ear.clear()
                            self._calib_r_ear.clear()
                            self._calib_status_text = "Face lost during calibration. Please look at camera."
                        self.controller.on_face_lost()
                        active_action, left_ear, right_ear, mar, debug_info = self.gestures.update(
                            face_detected=False,
                            left_eye=None,
                            right_eye=None,
                            is_calibrated=self.controller.is_calibrated,
                            is_paused=self.controller.is_paused,
                            is_emergency_stopped=self.controller.is_emergency_stopped,
                            current_time=now,
                        )

                    # Unified telemetry query from cursor controller
                    cursor_telemetry = self.controller.get_telemetry()

                    # Determine active operational state for live telemetry
                    if cursor_telemetry["is_emergency_stopped"]:
                        op_state = "EMERGENCY_STOPPED"
                    elif cursor_telemetry["is_paused"]:
                        op_state = "PAUSED"
                    elif not tracking_result.face_detected:
                        op_state = "NO_FACE"
                    elif self._is_calibrating:
                        count = len(self._calib_nose_x)
                        op_state = f"CALIBRATING ({count}/{self._calib_frames_target})"
                    elif getattr(debug_info, "scroll_state", "NEUTRAL") != "NEUTRAL":
                        op_state = f"SCROLLING_{debug_info.scroll_state}"
                    elif not cursor_telemetry["is_calibrated"]:
                        op_state = "NEEDS_CALIBRATION"
                    else:
                        op_state = "TRACKING"

                    # Attach cursor metrics safely to debug_info for HUD display
                    debug_info.raw_dx = cursor_telemetry["raw_dx"]
                    debug_info.raw_dy = cursor_telemetry["raw_dy"]
                    debug_info.calibrated_dx = cursor_telemetry["calibrated_dx"]
                    debug_info.calibrated_dy = cursor_telemetry["calibrated_dy"]
                    debug_info.target_x = cursor_telemetry["target_x"]
                    debug_info.target_y = cursor_telemetry["target_y"]
                    debug_info.filtered_x = cursor_telemetry["filtered_x"]
                    debug_info.filtered_y = cursor_telemetry["filtered_y"]
                    debug_info.mouse_moved = cursor_telemetry["mouse_moved"]
                    debug_info.mouse_suppressed = cursor_telemetry.get("mouse_suppressed", False)
                    debug_info.mouse_suppress_reason = cursor_telemetry.get("mouse_suppress_reason", "NONE")
                    debug_info.operational_state = op_state

                    # Draw HUD annotations on frame copy for GUI preview
                    preview_frame = self._draw_hud(
                        frame=frame.copy(),
                        tracking_result=tracking_result,
                        active_action=active_action,
                        debug_info=debug_info,
                        cam_fps=getattr(self.camera, "camera_fps", 0.0),
                        proc_fps=current_fps,
                        latency_ms=latency_ms,
                        queue_size=self.frame_queue.qsize(),
                    )

                    just_calib = self._just_calibrated_payload
                    self._just_calibrated_payload = None

                    # Package telemetry safely for GUI thread
                    telemetry = {
                        "face_detected": bool(tracking_result.face_detected),
                        "is_calibrated": cursor_telemetry["is_calibrated"],
                        "is_paused": cursor_telemetry["is_paused"],
                        "is_emergency_stopped": cursor_telemetry["is_emergency_stopped"],
                        "is_face_lost_safe": getattr(self.gestures, "is_face_lost_safe", False),
                        "operational_state": op_state,
                        "left_ear": getattr(debug_info, "left_ear", 0.0),
                        "right_ear": getattr(debug_info, "right_ear", 0.0),
                        "baseline_left": getattr(debug_info, "baseline_left", 0.25),
                        "baseline_right": getattr(debug_info, "baseline_right", 0.25),
                        "thresh_left": getattr(debug_info, "thresh_left", 0.18),
                        "thresh_right": getattr(debug_info, "thresh_right", 0.18),
                        "left_open": getattr(debug_info, "left_open", True),
                        "right_open": getattr(debug_info, "right_open", True),
                        "eye_state": getattr(debug_info, "eye_state", "OPEN"),
                        "gesture_state": getattr(debug_info, "gesture_state", "IDLE"),
                        "gesture_event": getattr(debug_info, "gesture_event", "NONE"),
                        "head_pitch": getattr(debug_info, "head_pitch", 0.0),
                        "scroll_state": getattr(debug_info, "scroll_state", "NEUTRAL"),
                        "scroll_event": getattr(debug_info, "scroll_event", "NONE"),
                        "input_enabled": getattr(debug_info, "input_enabled", False),
                        "active_action": active_action.name if hasattr(active_action, "name") else "NONE",
                        "raw_dx": cursor_telemetry["raw_dx"],
                        "raw_dy": cursor_telemetry["raw_dy"],
                        "calibrated_dx": cursor_telemetry["calibrated_dx"],
                        "calibrated_dy": cursor_telemetry["calibrated_dy"],
                        "target_x": cursor_telemetry["target_x"],
                        "target_y": cursor_telemetry["target_y"],
                        "filtered_x": cursor_telemetry["filtered_x"],
                        "filtered_y": cursor_telemetry["filtered_y"],
                        "mouse_moved": cursor_telemetry["mouse_moved"],
                        "mouse_suppressed": cursor_telemetry.get("mouse_suppressed", False),
                        "mouse_suppress_reason": cursor_telemetry.get("mouse_suppress_reason", "NONE"),
                        "cam_fps": getattr(self.camera, "camera_fps", 0.0),
                        "proc_fps": current_fps,
                        "latency_ms": latency_ms,
                        "queue_size": self.frame_queue.qsize(),
                        "cursor_pos": cursor_telemetry["cursor_pos"],
                        # Enhanced diagnostics fields
                        "camera_status": "Connected" if getattr(self.camera, "is_opened", False) else "Disconnected",
                        "camera_id": getattr(self.config, "camera_id", 0),
                        "landmarks_count": len(tracking_result.raw_landmarks) if (tracking_result.face_detected and tracking_result.raw_landmarks is not None) else 0,
                        "in_click_cooldown": getattr(debug_info, "in_click_cooldown", False),
                        "cooldown_remaining_s": getattr(debug_info, "cooldown_remaining_s", 0.0),
                        "bilateral_blink": getattr(debug_info, "bilateral_blink", False),
                        "dead_zone_radius": getattr(self.config, "dead_zone_radius", 0.008),
                        "sensitivity_x": getattr(self.config, "sensitivity_x", 2.5),
                        "sensitivity_y": getattr(self.config, "sensitivity_y", 2.5),
                        "camera_errors_count": 0,
                        "just_calibrated": just_calib,
                        "raw_pitch": getattr(debug_info, "raw_pitch", 0.0),
                        "calibrated_pitch": getattr(debug_info, "calibrated_pitch", getattr(debug_info, "head_pitch", 0.0)),
                        "scroll_activation_thresh": getattr(debug_info, "scroll_activation_thresh", 0.020),
                        "scroll_release_thresh": getattr(debug_info, "scroll_release_thresh", 0.013),
                        "scroll_transition_reason": getattr(debug_info, "scroll_transition_reason", "STEADY"),
                        "timestamp": frame_ts if frame_ts > 0 else now,
                        "calib_state": "CALIBRATING_NEUTRAL" if self._is_calibrating else ("CALIBRATING_EYES" if self._request_eye_calibration else "IDLE"),
                        "calib_progress": (len(self._calib_nose_x) / max(1, self._calib_frames_target)) if self._is_calibrating else ((len(self._calib_l_ear) / max(1, self._calib_frames_target)) if self._request_eye_calibration else (1.0 if self.controller.is_calibrated else 0.0)),
                        "calib_message": getattr(self, "_calib_status_text", "Ready"),
                    }

                    # Discard stale frames so queue only contains the newest frame (Part 1 & 9)
                    try:
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        self.frame_queue.put_nowait((preview_frame, telemetry))
                    except queue.Full:
                        pass
                except Exception as frame_err:
                    print(f"[VisionWorker] Frame processing error: {frame_err}")
                    time.sleep(0.01)
        finally:
            self.camera.release()
            if self.face_tracker:
                self.face_tracker.close()

    def _draw_hud(
        self,
        frame: np.ndarray,
        tracking_result: Any,
        active_action: GestureAction,
        debug_info: GestureDebugInfo,
        cam_fps: float,
        proc_fps: float,
        latency_ms: float,
        queue_size: int,
    ) -> np.ndarray:
        """Renders face landmarks, tracking reticle, deadzone, and live debug overlay onto frame."""
        h, w, _ = frame.shape

        if tracking_result.face_detected and tracking_result.nose_tip is not None:
            # Draw eye contours with live OPEN (Green) vs CLOSED (Red) indication
            left_color = (0, 255, 0) if debug_info.left_open else (0, 0, 255)
            right_color = (0, 255, 0) if debug_info.right_open else (0, 0, 255)

            if tracking_result.left_eye:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in tracking_result.left_eye], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=left_color, thickness=2)

            if tracking_result.right_eye:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in tracking_result.right_eye], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=right_color, thickness=2)

            # Draw neutral center, comfort boundary, and deadzone circle
            if self.controller.is_calibrated and self.controller.neutral_x is not None:
                nx = int(self.controller.neutral_x * w)
                ny = int(self.controller.neutral_y * h)
                deadzone_px = int(self.config.dead_zone_radius * min(w, h))
                cw = int(getattr(self.config, "comfort_range_x", 0.050) * w)
                ch = int(getattr(self.config, "comfort_range_y", 0.045) * h)

                # Comfort zone boundary (accessibility movement area)
                cv2.rectangle(frame, (nx - cw, ny - ch), (nx + cw, ny + ch), (140, 140, 60), 1, cv2.LINE_AA)
                # Deadzone boundary
                cv2.circle(frame, (nx, ny), deadzone_px, (80, 80, 80), 1, cv2.LINE_AA)
                # Neutral crosshair
                cv2.drawMarker(frame, (nx, ny), (0, 200, 0), cv2.MARKER_CROSS, 12, 1)

            # Draw nose tip tracking reticle
            px = int(tracking_result.nose_tip[0] * w)
            py = int(tracking_result.nose_tip[1] * h)
            is_scrolling = (debug_info.scroll_state != "NEUTRAL")
            reticle_color = (255, 0, 255) if is_scrolling else (0, 0, 255)
            cv2.circle(frame, (px, py), 5, reticle_color, -1, cv2.LINE_AA)

        # Draw live DEBUG overlay box in top-left
        overlay = frame.copy()
        box_y = 8
        box_w, box_h = 475, 175
        cv2.rectangle(overlay, (8, box_y), (box_w, box_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (8, box_y), (box_w, box_y + box_h), (80, 80, 80), 1)

        # Line 1: Telemetry (CAM FPS, PROC FPS, LATENCY, STATE)
        op_state = getattr(debug_info, "operational_state", "IDLE")
        fps_text = f"CAM:{cam_fps:.0f} | PROC:{proc_fps:.0f} | LATENCY:{latency_ms:.0f}ms | STATE:{op_state}"
        cv2.putText(frame, fps_text, (15, box_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Line 2: 2-Axis Pointer Movement (X Movement)
        x_text = f"X MOVE: raw={debug_info.raw_dx:+.4f} | norm={debug_info.calibrated_dx:+.3f}"
        cv2.putText(frame, x_text, (15, box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Line 3: 2-Axis Pointer Movement (Y Movement)
        y_text = f"Y MOVE: raw={debug_info.raw_dy:+.4f} | norm={debug_info.calibrated_dy:+.3f}"
        cv2.putText(frame, y_text, (15, box_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Line 4: Target Coordinates & Command Issued
        if debug_info.mouse_moved:
            moved_str = "YES"
            cmd_col = (0, 255, 0)
        elif getattr(debug_info, "mouse_suppressed", False):
            reason = getattr(debug_info, "mouse_suppress_reason", "SUPPRESSED")
            moved_str = f"NO (Suppressed: {reason})"
            cmd_col = (0, 165, 255) if reason == "SCROLLING" else (0, 0, 255)
        else:
            reason = getattr(debug_info, "mouse_suppress_reason", "AT_REST")
            moved_str = f"NO ({reason})"
            cmd_col = (180, 180, 180)

        inp_str = "ON" if debug_info.input_enabled else "OFF"
        coords_text = f"TARGET: ({debug_info.target_x:.0f}, {debug_info.target_y:.0f}) | MOVED: {moved_str} | INP: {inp_str}"
        cv2.putText(frame, coords_text, (15, box_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cmd_col, 1, cv2.LINE_AA)

        # Line 5: LEFT EAR
        l_stat = "OPEN" if debug_info.left_open else "CLOSED"
        l_col = (0, 255, 0) if debug_info.left_open else (0, 0, 255)
        l_text = f"LEFT EAR:  {debug_info.left_ear:.3f} [{l_stat}] (base:{debug_info.baseline_left:.2f}, thr:{debug_info.thresh_left:.2f})"
        cv2.putText(frame, l_text, (15, box_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, l_col, 1, cv2.LINE_AA)

        # Line 6: RIGHT EAR
        r_stat = "OPEN" if debug_info.right_open else "CLOSED"
        r_col = (0, 255, 0) if debug_info.right_open else (0, 0, 255)
        r_text = f"RIGHT EAR: {debug_info.right_ear:.3f} [{r_stat}] (base:{debug_info.baseline_right:.2f}, thr:{debug_info.thresh_right:.2f})"
        cv2.putText(frame, r_text, (15, box_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_col, 1, cv2.LINE_AA)

        # Line 7: Eye State & Click Event
        evt_col = (0, 255, 255) if debug_info.gesture_event == "NONE" else (0, 255, 0)
        state_text = f"EYE: {debug_info.eye_state} | CLICK: {debug_info.gesture_event}"
        cv2.putText(frame, state_text, (15, box_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, evt_col, 1, cv2.LINE_AA)

        # Line 8: Head Pitch & Scroll State
        scroll_col = (255, 0, 255) if debug_info.scroll_state != "NEUTRAL" else (200, 200, 200)
        act_th = getattr(debug_info, "scroll_activation_thresh", 0.020)
        rel_th = getattr(debug_info, "scroll_release_thresh", 0.009)
        pitch_text = f"PITCH: {debug_info.head_pitch:+.3f} (trig:+-{act_th:.3f}, rel:+-{rel_th:.3f}) | SCROLL: {debug_info.scroll_state}"
        cv2.putText(frame, pitch_text, (15, box_y + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, scroll_col, 1, cv2.LINE_AA)

        # Scroll Direction Indicator on Screen
        if debug_info.scroll_state == "UP":
            cv2.putText(frame, "^ SCROLLING UP", (w // 2 - 80, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        elif debug_info.scroll_state == "DOWN":
            cv2.putText(frame, "v SCROLLING DOWN", (w // 2 - 95, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Action notification badge if active
        if active_action != GestureAction.NONE:
            action_text = f"ACTION: {active_action.name}"
            cv2.putText(frame, action_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame


class FacialCursorGUI:
    """Tkinter Application Window running strictly on the main thread."""

    def __init__(self, root: tk.Tk, config: AppConfig):
        self.root = root
        self.config = config
        self.root.title("AI Facial Cursor Control System")
        self.root.geometry("1080x740")
        self.root.minsize(920, 620)

        # Profile manager (loads existing profiles or creates Default)
        self.profile_manager = ProfileManager(self.config)
        active_prof = self.profile_manager.get_active_profile()
        if active_prof:
            for k, v in active_prof.settings.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # Communication queues: maxsize=1 enforces newest-frame delivery (Part 1 & Part 9)
        self.frame_queue = queue.Queue(maxsize=1)
        self.command_queue = queue.Queue()

        # GUI FPS tracking
        self._gui_fps: float = 0.0
        self._gui_frame_count: int = 0
        self._gui_fps_timer: float = time.monotonic()

        # Keyboard reference
        self.keyboard: Optional[OnScreenKeyboard] = None

        # Start worker thread
        self.worker = VisionWorker(self.config, self.frame_queue, self.command_queue)
        if active_prof and active_prof.calibration.get("is_calibrated"):
            self.worker._handle_switch_profile({
                "settings": active_prof.settings,
                "calibration": active_prof.calibration,
            })
        self.worker.start()

        # Build UI layout
        self._build_ui()

        # Bind keyboard shortcuts (Accessibility-friendly hotkeys)
        self.root.bind("<c>", lambda e: self._on_calibrate())
        self.root.bind("<C>", lambda e: self._on_calibrate())
        self.root.bind("<e>", lambda e: self._on_calibrate_eyes())
        self.root.bind("<E>", lambda e: self._on_calibrate_eyes())
        self.root.bind("<p>", lambda e: self._on_toggle_pause())
        self.root.bind("<P>", lambda e: self._on_toggle_pause())
        self.root.bind("<space>", lambda e: self._on_toggle_pause())
        self.root.bind("<k>", lambda e: self._on_toggle_keyboard())
        self.root.bind("<K>", lambda e: self._on_toggle_keyboard())
        self.root.bind("<Escape>", lambda e: self._on_emergency_stop())
        self.root.bind("<q>", lambda e: self._on_emergency_stop())
        self.root.bind("<Q>", lambda e: self._on_emergency_stop())

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Start polling loop on Tkinter main thread
        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self.root.after(10, self._poll_worker)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        # Top Control & Status Bar
        top_bar = ttk.Frame(self.root, padding="10")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_calibrate = ttk.Button(top_bar, text="Calibrate Neutral (C)", command=self._on_calibrate)
        self.btn_calibrate.pack(side=tk.LEFT, padx=3)

        self.btn_calibrate_eyes = ttk.Button(top_bar, text="Calibrate Eyes (E)", command=self._on_calibrate_eyes)
        self.btn_calibrate_eyes.pack(side=tk.LEFT, padx=3)

        self.btn_pause = ttk.Button(top_bar, text="Pause (P / Space)", command=self._on_toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=3)

        self.btn_keyboard = ttk.Button(top_bar, text="Virtual Keyboard (K)", command=self._on_toggle_keyboard)
        self.btn_keyboard.pack(side=tk.LEFT, padx=3)

        self.btn_emergency = tk.Button(
            top_bar,
            text="EMERGENCY STOP (Esc / Q)",
            bg="#cc0000",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            command=self._on_emergency_stop,
        )
        self.btn_emergency.pack(side=tk.LEFT, padx=6)

        self.lbl_status = ttk.Label(
            top_bar,
            text="Status: Initializing...",
            font=("Segoe UI", 10, "bold"),
            foreground="#0066cc",
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # Main Split Frame: Left (Camera Preview), Right (Controls & Gauges)
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Video Feed Canvas
        left_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(left_frame, weight=3)

        self.canvas = tk.Canvas(left_frame, bg="#1e1e1e", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: User Profile + Tabbed Dashboard/Controls
        right_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(right_frame, weight=2)

        # --- Profile Management Section ---
        grp_profile = ttk.LabelFrame(right_frame, text="User Profile", padding="6")
        grp_profile.pack(fill=tk.X, pady=(0, 4))

        prof_row1 = ttk.Frame(grp_profile)
        prof_row1.pack(fill=tk.X, pady=2)
        ttk.Label(prof_row1, text="Active:").pack(side=tk.LEFT, padx=(0, 4))

        self.cb_profile = ttk.Combobox(
            prof_row1,
            state="readonly",
            values=self.profile_manager.list_profiles(),
            width=16,
        )
        self.cb_profile.set(self.profile_manager.get_active_profile_name())
        self.cb_profile.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.cb_profile.bind("<<ComboboxSelected>>", self._on_profile_selected)

        prof_row2 = ttk.Frame(grp_profile)
        prof_row2.pack(fill=tk.X, pady=2)
        ttk.Button(prof_row2, text="New", width=6, command=self._on_new_profile).pack(side=tk.LEFT, padx=1)
        ttk.Button(prof_row2, text="Rename", width=8, command=self._on_rename_profile).pack(side=tk.LEFT, padx=1)
        ttk.Button(prof_row2, text="Delete", width=7, command=self._on_delete_profile).pack(side=tk.LEFT, padx=1)
        ttk.Button(prof_row2, text="Save Profile", width=11, command=self._save_active_profile).pack(side=tk.RIGHT, padx=1)

        # --- Tabs: 1) Diagnostics & Telemetry, 2) Tuning Controls ---
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=2)

        # Tab 1: Diagnostics Dashboard
        tab_diag = ttk.Frame(self.notebook, padding="4")
        self.notebook.add(tab_diag, text="Diagnostics")
        self.diagnostics = DiagnosticsDashboard(tab_diag)
        self.diagnostics.pack(fill=tk.BOTH, expand=True)

        # Expose label aliases for backward compatibility with tests/inspectors
        self.lbl_face = self.diagnostics.lbl_face
        self.lbl_op_state = self.diagnostics.lbl_op_state
        self.lbl_raw_dx = self.diagnostics.lbl_raw_dx
        self.lbl_raw_dy = self.diagnostics.lbl_raw_dy
        self.lbl_calibrated_dx = self.diagnostics.lbl_calibrated_dx
        self.lbl_calibrated_dy = self.diagnostics.lbl_calibrated_dy
        self.lbl_target_coords = self.diagnostics.lbl_target_coords
        self.lbl_mouse_moved = self.diagnostics.lbl_mouse_moved
        self.lbl_ear_left = self.diagnostics.lbl_ear_left
        self.lbl_ear_right = self.diagnostics.lbl_ear_right
        self.lbl_eye_state = self.diagnostics.lbl_eye_state
        self.lbl_gesture_event = self.diagnostics.lbl_gesture_event
        self.lbl_pitch = self.diagnostics.lbl_pitch
        self.lbl_scroll_state = self.diagnostics.lbl_scroll_state
        self.lbl_fps = self.diagnostics.lbl_fps
        self.lbl_latency = self.diagnostics.lbl_latency
        self.lbl_input_enabled = self.diagnostics.lbl_input_enabled

        # Tab 2: Settings & Calibration
        tab_tuning = ttk.Frame(self.notebook, padding="4")
        self.notebook.add(tab_tuning, text="Settings & Calibration")

        # Scrollable container for Settings & Calibration
        canvas_settings = tk.Canvas(tab_tuning, highlightthickness=0)
        scrollbar_settings = ttk.Scrollbar(tab_tuning, orient=tk.VERTICAL, command=canvas_settings.yview)
        scroll_frame = ttk.Frame(canvas_settings)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_settings.configure(scrollregion=canvas_settings.bbox("all"))
        )
        canvas_settings.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas_settings.configure(yscrollcommand=scrollbar_settings.set)

        canvas_settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_settings.pack(side=tk.RIGHT, fill=tk.Y)

        def _make_row(parent, title, from_, to, init_val, fmt_func):
            hdr = ttk.Frame(parent)
            hdr.pack(fill=tk.X, pady=(2, 0))
            ttk.Label(hdr, text=title, font=("Segoe UI", 8)).pack(side=tk.LEFT)
            v_lbl = ttk.Label(hdr, text=fmt_func(init_val), font=("Segoe UI", 8, "bold"), foreground="#0066cc")
            v_lbl.pack(side=tk.RIGHT)
            scale = ttk.Scale(parent, from_=from_, to=to, value=init_val, command=self._on_slider_change)
            scale.pack(fill=tk.X, pady=(0, 2))
            return scale, v_lbl

        # 1. Posture & Tracking Calibration Section
        grp_calib = ttk.LabelFrame(scroll_frame, text="1. Posture & Tracking Calibration", padding="6")
        grp_calib.pack(fill=tk.X, padx=4, pady=3)

        ttk.Label(
            grp_calib,
            text="Posture Guidance: Sit upright and look directly at the center of your screen in your normal comfortable resting posture. Press 'Calibrate Neutral' and hold still for 1 second.",
            wraplength=330,
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, pady=2)

        row_cbtns = ttk.Frame(grp_calib)
        row_cbtns.pack(fill=tk.X, pady=2)
        ttk.Button(row_cbtns, text="Calibrate Neutral (C)", command=self._on_calibrate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cbtns, text="Calibrate Eyes (E)", command=self._on_calibrate_eyes).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        row_cbtns2 = ttk.Frame(grp_calib)
        row_cbtns2.pack(fill=tk.X, pady=2)
        ttk.Button(row_cbtns2, text="Reset Calibration", command=self._on_reset_neutral).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(row_cbtns2, text="Recalibrate Camera", command=self._on_recalibrate_camera).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self.calib_progress = ttk.Progressbar(grp_calib, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.calib_progress.pack(fill=tk.X, pady=2)

        self.calib_status_lbl = ttk.Label(
            grp_calib, text="Calibration Status: Ready", font=("Segoe UI", 8, "bold"), foreground="#0066cc"
        )
        self.calib_status_lbl.pack(anchor=tk.W, pady=1)

        # 2. Pointer Movement & Sensitivity Section
        grp_cursor = ttk.LabelFrame(scroll_frame, text="2. Pointer Movement & Sensitivity", padding="6")
        grp_cursor.pack(fill=tk.X, padx=4, pady=3)

        self.slider_sens_x, self.lbl_sens_x_val = _make_row(
            grp_cursor, "Horizontal Sensitivity (X):", 0.5, 8.0, self.config.sensitivity_x, lambda v: f"{v:.2f}"
        )
        self.slider_sens_y, self.lbl_sens_y_val = _make_row(
            grp_cursor, "Vertical Sensitivity (Y):", 0.5, 8.0, self.config.sensitivity_y, lambda v: f"{v:.2f}"
        )
        self.slider_deadzone, self.lbl_deadzone_val = _make_row(
            grp_cursor, "Dead Zone Radius (Stability):", 0.002, 0.030, self.config.dead_zone_radius, lambda v: f"{v:.3f}"
        )
        self.slider_comfort_x, self.lbl_comfort_x_val = _make_row(
            grp_cursor, "Comfort Range X (Movement Area):", 0.020, 0.100, getattr(self.config, "comfort_range_x", 0.045), lambda v: f"{v:.3f}"
        )
        self.slider_comfort_y, self.lbl_comfort_y_val = _make_row(
            grp_cursor, "Comfort Range Y (Movement Area):", 0.020, 0.100, getattr(self.config, "comfort_range_y", 0.045), lambda v: f"{v:.3f}"
        )

        # 3. Smoothing & Filter Responsiveness Section
        grp_filter = ttk.LabelFrame(scroll_frame, text="3. Smoothing & Jitter Filter", padding="6")
        grp_filter.pack(fill=tk.X, padx=4, pady=3)

        self.slider_cutoff, self.lbl_cutoff_val = _make_row(
            grp_filter, "Filter Min Cutoff (Rest Stability):", 0.1, 3.0, self.config.cursor_min_cutoff, lambda v: f"{v:.1f} Hz"
        )
        self.slider_beta, self.lbl_beta_val = _make_row(
            grp_filter, "Filter Beta (Motion Responsiveness):", 0.01, 1.50, self.config.cursor_beta, lambda v: f"{v:.2f}"
        )

        # 4. Head-Tilt Scrolling Section
        grp_scroll = ttk.LabelFrame(scroll_frame, text="4. Head-Tilt Scrolling & Thresholds", padding="6")
        grp_scroll.pack(fill=tk.X, padx=4, pady=3)

        self.slider_scroll_pitch, self.lbl_scroll_pitch_val = _make_row(
            grp_scroll, "Scroll Activation Threshold (Effort):", 0.010, 0.050, self.config.scroll_pitch_threshold, lambda v: f"{v:.3f}"
        )
        self.slider_scroll_release, self.lbl_scroll_release_val = _make_row(
            grp_scroll, "Scroll Release Ratio (Hysteresis):", 0.40, 0.85, getattr(self.config, "scroll_release_ratio", 0.65), lambda v: f"{v:.2f}"
        )
        self.slider_scroll_speed, self.lbl_scroll_speed_val = _make_row(
            grp_scroll, "Scroll Speed / Step:", 10, 80, self.config.scroll_speed, lambda v: f"{int(v)}"
        )
        self.slider_scroll_interval, self.lbl_scroll_interval_val = _make_row(
            grp_scroll, "Scroll Step Interval (s):", 0.05, 0.30, getattr(self.config, "scroll_interval_s", 0.12), lambda v: f"{v:.2f}s"
        )

        # 5. Gestures & Dwell-Click Section
        grp_dwell = ttk.LabelFrame(scroll_frame, text="5. Gestures & Dwell-Click", padding="6")
        grp_dwell.pack(fill=tk.X, padx=4, pady=3)

        self.slider_ear_ratio, self.lbl_ear_ratio_val = _make_row(
            grp_dwell, "Wink Blink EAR Ratio:", 0.60, 0.85, getattr(self.config, "ear_blink_ratio", 0.75), lambda v: f"{v:.2f}"
        )

        dwell_row = ttk.Frame(grp_dwell)
        dwell_row.pack(fill=tk.X, pady=2)
        self.chk_dwell_var = tk.BooleanVar(value=getattr(self.config, "keyboard_dwell_enabled", False))
        self.chk_dwell_mode = ttk.Checkbutton(
            dwell_row, text="Enable Dwell-Click Mode", variable=self.chk_dwell_var, command=self._on_slider_change
        )
        self.chk_dwell_mode.pack(side=tk.LEFT)

        self.slider_dwell, self.lbl_dwell_val = _make_row(
            grp_dwell, "Keyboard Dwell Time (s):", 0.4, 3.0, getattr(self.config, "keyboard_dwell_time_s", 1.2), lambda v: f"{v:.1f}s"
        )

        # 6. Action Buttons
        btn_box = ttk.Frame(scroll_frame)
        btn_box.pack(fill=tk.X, padx=4, pady=6)

        ttk.Button(btn_box, text="Save Settings", command=self._save_settings).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_box, text="Restore Defaults", command=self._reset_defaults).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_box, text="Re-Center (C)", command=self._on_calibrate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    def _on_calibrate(self) -> None:
        self.command_queue.put(("CALIBRATE", None))

    def _on_calibrate_eyes(self) -> None:
        self.command_queue.put(("CALIBRATE_EYES", None))

    def _on_reset_neutral(self) -> None:
        self.command_queue.put(("RESET_NEUTRAL", None))

    def _on_recalibrate_camera(self) -> None:
        self.command_queue.put(("RECALIBRATE_CAMERA", None))
        self.command_queue.put(("CALIBRATE_EYES", None))

    def _on_toggle_pause(self) -> None:
        self.command_queue.put(("TOGGLE_PAUSE", None))

    def _on_emergency_stop(self) -> None:
        if self.worker.controller.is_emergency_stopped:
            self.command_queue.put(("RESUME", None))
            self.btn_emergency.config(text="EMERGENCY STOP (Esc / Q)", bg="#cc0000")
        else:
            self.command_queue.put(("EMERGENCY_STOP", None))
            self.btn_emergency.config(text="RESUME FROM STOP (Esc / Q)", bg="#ff8800")

    def _on_toggle_keyboard(self) -> None:
        if self.keyboard is None or not self.keyboard.is_open:
            self.keyboard = OnScreenKeyboard(self.root, self.config, on_close_callback=self._on_keyboard_closed)
            self.btn_keyboard.config(text="Hide Keyboard (K)")
        else:
            self.keyboard.close()
            self.btn_keyboard.config(text="Virtual Keyboard (K)")

    def _on_keyboard_closed(self) -> None:
        self.keyboard = None
        self.btn_keyboard.config(text="Virtual Keyboard (K)")

    def _on_profile_selected(self, event=None) -> None:
        prof_name = self.cb_profile.get()
        if not prof_name:
            return
        success = self.profile_manager.switch_profile(prof_name)
        if success:
            active_prof = self.profile_manager.get_active_profile()
            if active_prof:
                self._sync_sliders_from_config()
                self.command_queue.put(("SWITCH_PROFILE", {
                    "settings": active_prof.settings,
                    "calibration": active_prof.calibration,
                }))

    def _on_new_profile(self) -> None:
        name = simpledialog.askstring("New Profile", "Enter name for new user profile:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        try:
            self.profile_manager.create_profile(name, clone_active=True)
            self._update_profile_dropdown(selected=name)
            self._on_profile_selected()
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self.root)

    def _on_rename_profile(self) -> None:
        current = self.profile_manager.get_active_profile_name()
        if current == "Default":
            messagebox.showwarning("Warning", "The 'Default' profile cannot be renamed.", parent=self.root)
            return
        new_name = simpledialog.askstring(
            "Rename Profile", f"Enter new name for profile '{current}':", initialvalue=current, parent=self.root
        )
        if not new_name or not new_name.strip() or new_name.strip() == current:
            return
        new_name = new_name.strip()
        try:
            self.profile_manager.rename_profile(current, new_name)
            self._update_profile_dropdown(selected=new_name)
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self.root)

    def _on_delete_profile(self) -> None:
        current = self.profile_manager.get_active_profile_name()
        if current == "Default":
            messagebox.showwarning("Warning", "The 'Default' profile cannot be deleted.", parent=self.root)
            return
        confirm = messagebox.askyesno(
            "Confirm Delete", f"Are you sure you want to delete profile '{current}'?", parent=self.root
        )
        if not confirm:
            return
        try:
            self.profile_manager.delete_profile(current)
            self._update_profile_dropdown(selected=self.profile_manager.get_active_profile_name())
            self._on_profile_selected()
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self.root)

    def _save_active_profile(self) -> None:
        self.profile_manager.save_active_profile_settings()
        self.config.save_to_file()
        messagebox.showinfo(
            "Profiles",
            f"Profile '{self.profile_manager.get_active_profile_name()}' saved successfully!",
            parent=self.root,
        )

    def _update_profile_dropdown(self, selected: Optional[str] = None) -> None:
        profiles = self.profile_manager.list_profiles()
        self.cb_profile["values"] = profiles
        if selected and selected in profiles:
            self.cb_profile.set(selected)
        elif profiles:
            self.cb_profile.set(self.profile_manager.get_active_profile_name())

    def _update_slider_labels(self) -> None:
        if hasattr(self, "lbl_sens_x_val"):
            self.lbl_sens_x_val.config(text=f"{self.config.sensitivity_x:.2f}")
        if hasattr(self, "lbl_sens_y_val"):
            self.lbl_sens_y_val.config(text=f"{self.config.sensitivity_y:.2f}")
        if hasattr(self, "lbl_comfort_x_val"):
            self.lbl_comfort_x_val.config(text=f"{self.config.comfort_range_x:.3f}")
        if hasattr(self, "lbl_comfort_y_val"):
            self.lbl_comfort_y_val.config(text=f"{self.config.comfort_range_y:.3f}")
        if hasattr(self, "lbl_deadzone_val"):
            self.lbl_deadzone_val.config(text=f"{self.config.dead_zone_radius:.3f}")
        if hasattr(self, "lbl_cutoff_val"):
            self.lbl_cutoff_val.config(text=f"{self.config.cursor_min_cutoff:.1f} Hz")
        if hasattr(self, "lbl_beta_val"):
            self.lbl_beta_val.config(text=f"{self.config.cursor_beta:.2f}")
        if hasattr(self, "lbl_scroll_pitch_val"):
            self.lbl_scroll_pitch_val.config(text=f"{self.config.scroll_pitch_threshold:.3f}")
        if hasattr(self, "lbl_scroll_release_val"):
            self.lbl_scroll_release_val.config(text=f"{getattr(self.config, 'scroll_release_ratio', 0.65):.2f}")
        if hasattr(self, "lbl_scroll_speed_val"):
            self.lbl_scroll_speed_val.config(text=f"{self.config.scroll_speed}")
        if hasattr(self, "lbl_scroll_interval_val"):
            self.lbl_scroll_interval_val.config(text=f"{getattr(self.config, 'scroll_interval_s', 0.12):.2f}s")
        if hasattr(self, "lbl_ear_ratio_val"):
            self.lbl_ear_ratio_val.config(text=f"{self.config.ear_blink_ratio:.2f}")
        if hasattr(self, "lbl_dwell_val"):
            self.lbl_dwell_val.config(text=f"{getattr(self.config, 'keyboard_dwell_time_s', 1.2):.1f}s")

    def _sync_sliders_from_config(self) -> None:
        self.slider_sens_x.set(self.config.sensitivity_x)
        self.slider_sens_y.set(self.config.sensitivity_y)
        self.slider_comfort_x.set(self.config.comfort_range_x)
        self.slider_comfort_y.set(self.config.comfort_range_y)
        self.slider_deadzone.set(self.config.dead_zone_radius)
        self.slider_cutoff.set(self.config.cursor_min_cutoff)
        self.slider_beta.set(self.config.cursor_beta)
        self.slider_ear_ratio.set(self.config.ear_blink_ratio)
        self.slider_scroll_pitch.set(self.config.scroll_pitch_threshold)
        if hasattr(self, "slider_scroll_release"):
            self.slider_scroll_release.set(getattr(self.config, "scroll_release_ratio", 0.65))
        self.slider_scroll_speed.set(self.config.scroll_speed)
        if hasattr(self, "slider_scroll_interval"):
            self.slider_scroll_interval.set(getattr(self.config, "scroll_interval_s", 0.12))
        if hasattr(self, "slider_dwell"):
            self.slider_dwell.set(getattr(self.config, "keyboard_dwell_time_s", 1.2))
        if hasattr(self, "chk_dwell_var"):
            self.chk_dwell_var.set(getattr(self.config, "keyboard_dwell_enabled", False))
        self._update_slider_labels()

    def _on_slider_change(self, event=None) -> None:
        self.config.sensitivity_x = float(self.slider_sens_x.get())
        self.config.sensitivity_y = float(self.slider_sens_y.get())
        self.config.comfort_range_x = float(self.slider_comfort_x.get())
        self.config.comfort_range_y = float(self.slider_comfort_y.get())
        self.config.dead_zone_radius = float(self.slider_deadzone.get())
        self.config.cursor_min_cutoff = float(self.slider_cutoff.get())
        self.config.cursor_beta = float(self.slider_beta.get())
        self.config.ear_blink_ratio = float(self.slider_ear_ratio.get())
        self.config.scroll_pitch_threshold = float(self.slider_scroll_pitch.get())
        if hasattr(self, "slider_scroll_release"):
            self.config.scroll_release_ratio = float(self.slider_scroll_release.get())
        self.config.scroll_speed = int(self.slider_scroll_speed.get())
        if hasattr(self, "slider_scroll_interval"):
            self.config.scroll_interval_s = float(self.slider_scroll_interval.get())
        if hasattr(self, "slider_dwell"):
            self.config.keyboard_dwell_time_s = float(self.slider_dwell.get())
        if hasattr(self, "chk_dwell_var"):
            self.config.keyboard_dwell_enabled = bool(self.chk_dwell_var.get())
            self.config.dwell_click_enabled = self.config.keyboard_dwell_enabled

        self._update_slider_labels()
        self.command_queue.put(("UPDATE_CONFIG", None))

    def _save_settings(self) -> None:
        self.profile_manager.save_active_profile_settings()
        self.config.save_to_file()
        messagebox.showinfo(
            "Settings",
            f"Configuration and profile '{self.profile_manager.get_active_profile_name()}' saved successfully!",
        )

    def _reset_defaults(self) -> None:
        default_cfg = AppConfig()
        self.config.sensitivity_x = default_cfg.sensitivity_x
        self.config.sensitivity_y = default_cfg.sensitivity_y
        self.config.comfort_range_x = default_cfg.comfort_range_x
        self.config.comfort_range_y = default_cfg.comfort_range_y
        self.config.dead_zone_radius = default_cfg.dead_zone_radius
        self.config.enable_vertical_cursor = default_cfg.enable_vertical_cursor
        self.config.cursor_min_cutoff = default_cfg.cursor_min_cutoff
        self.config.cursor_beta = default_cfg.cursor_beta
        self.config.ear_blink_threshold = default_cfg.ear_blink_threshold
        self.config.ear_blink_ratio = default_cfg.ear_blink_ratio
        self.config.scroll_pitch_threshold = default_cfg.scroll_pitch_threshold
        self.config.scroll_release_ratio = default_cfg.scroll_release_ratio
        self.config.scroll_speed = default_cfg.scroll_speed
        self.config.scroll_interval_s = default_cfg.scroll_interval_s
        self.config.keyboard_dwell_time_s = default_cfg.keyboard_dwell_time_s
        self.config.keyboard_dwell_enabled = default_cfg.keyboard_dwell_enabled
        self.config.dwell_click_enabled = default_cfg.dwell_click_enabled

        self._sync_sliders_from_config()
        self.command_queue.put(("UPDATE_CONFIG", None))
        self.profile_manager.save_active_profile_settings()
        self.config.save_to_file()

    def _poll_worker(self) -> None:
        """Pulls latest frame and telemetry from queue strictly on main thread (Part 1: latest frame wins)."""
        try:
            latest_item = None
            while not self.frame_queue.empty():
                try:
                    item = self.frame_queue.get_nowait()
                    if isinstance(item, dict) and "error" in item:
                        self.lbl_status.config(text=f"Error: {item['error']}", foreground="red")
                        continue
                    latest_item = item
                except queue.Empty:
                    break

            if latest_item is not None:
                frame_bgr, telemetry = latest_item

                # Calculate GUI FPS
                self._gui_frame_count += 1
                now = time.monotonic()
                if now - self._gui_fps_timer >= 1.0:
                    self._gui_fps = self._gui_frame_count / (now - self._gui_fps_timer)
                    self._gui_frame_count = 0
                    self._gui_fps_timer = now

                telemetry["gui_fps"] = self._gui_fps

                # Update Status Badge with priority (Section 3)
                if telemetry.get("is_emergency_stopped", False):
                    self.lbl_status.config(text="EMERGENCY STOPPED", foreground="#cc0000")
                elif telemetry.get("is_paused", False):
                    self.lbl_status.config(text="PAUSED", foreground="#ff8800")
                elif not telemetry.get("face_detected", False):
                    self.lbl_status.config(text="NO FACE", foreground="#888888")
                elif "CALIBRATING" in telemetry.get("operational_state", ""):
                    self.lbl_status.config(text="CALIBRATING...", foreground="#cc8800")
                elif telemetry.get("active_action") == "LEFT_CLICK":
                    self.lbl_status.config(text="LEFT CLICK", foreground="#00aa00")
                elif telemetry.get("active_action") == "RIGHT_CLICK":
                    self.lbl_status.config(text="RIGHT CLICK", foreground="#0088cc")
                elif telemetry.get("scroll_state") == "UP":
                    self.lbl_status.config(text="SCROLL UP", foreground="#aa00cc")
                elif telemetry.get("scroll_state") == "DOWN":
                    self.lbl_status.config(text="SCROLL DOWN", foreground="#aa00cc")
                elif not telemetry.get("is_calibrated", False):
                    self.lbl_status.config(text="NEEDS CALIBRATION (Press 'C')", foreground="#cc8800")
                elif telemetry.get("face_detected", False):
                    self.lbl_status.config(text="FACE DETECTED", foreground="#009900")
                else:
                    self.lbl_status.config(text="IDLE", foreground="#666666")

                # Update Unified Diagnostics Dashboard
                self.diagnostics.update_dashboard(telemetry)

                # Calibration UI feedback
                if hasattr(self, "calib_progress") and hasattr(self, "calib_status_lbl"):
                    prog = float(telemetry.get("calib_progress") or 0.0)
                    self.calib_progress["value"] = prog * 100.0
                    msg = telemetry.get("calib_message", "Ready")
                    self.calib_status_lbl.config(text=f"Calibration: {msg}")

                # Feed scrolling suppression and cursor position to OnScreenKeyboard if open
                if self.keyboard and self.keyboard.is_open:
                    is_scrolling = (telemetry.get("scroll_state", "NEUTRAL") != "NEUTRAL")
                    self.keyboard.set_scrolling_suppressed(is_scrolling)
                    cur_pos = telemetry.get("cursor_pos")
                    if cur_pos and isinstance(cur_pos, (tuple, list)) and len(cur_pos) >= 2:
                        self.keyboard.update_cursor_coords(cur_pos[0], cur_pos[1])
                    if telemetry.get("active_action") == "LEFT_CLICK" and cur_pos:
                        self.keyboard.handle_click_event(cur_pos[0], cur_pos[1])

                # Save new calibration to active profile automatically
                just_calib = telemetry.get("just_calibrated")
                if just_calib:
                    self.profile_manager.update_active_profile_calibration(
                        neutral_x=just_calib.get("neutral_x"),
                        neutral_y=just_calib.get("neutral_y"),
                        neutral_pitch=just_calib.get("neutral_pitch"),
                        baseline_left_ear=just_calib.get("baseline_left_ear"),
                        baseline_right_ear=just_calib.get("baseline_right_ear"),
                    )

                # Render ONLY the newest frame to Canvas
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                if canvas_w > 10 and canvas_h > 10:
                    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    fh, fw, _ = rgb_frame.shape
                    scale = min(canvas_w / fw, canvas_h / fh)
                    new_w = max(1, int(fw * scale))
                    new_h = max(1, int(fh * scale))
                    resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    img = Image.fromarray(resized)
                    self._current_photo = ImageTk.PhotoImage(image=img)
                    self.canvas.delete("all")
                    self.canvas.create_image(
                        canvas_w // 2,
                        canvas_h // 2,
                        anchor=tk.CENTER,
                        image=self._current_photo,
                    )
        except Exception as e:
            print(f"[GUI] Render error: {e}")

        # Schedule next poll on main thread (10ms = 100Hz responsiveness)
        self.root.after(10, self._poll_worker)

    def _on_closing(self) -> None:
        """Handles graceful shutdown."""
        if self.keyboard and self.keyboard.is_open:
            try:
                self.keyboard.close()
            except Exception:
                pass
        self.worker.stop()
        self.worker.join(timeout=1.0)
        self.root.destroy()

    def _on_closing(self) -> None:
        """Handles graceful shutdown."""
        self.worker.stop()
        self.worker.join(timeout=1.0)
        self.root.destroy()

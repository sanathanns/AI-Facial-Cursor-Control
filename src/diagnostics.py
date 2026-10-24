"""Live Tracking Diagnostics Dashboard component for Facial Cursor Control System.

Displays real-time telemetry from the tracking and cursor control pipeline:
- Camera connection and face landmark tracking status with visual warnings
- Cursor movement telemetry (raw, calibrated, filtered, target, and issued status)
- Eye gesture states, bilateral blink protection, and cooldown/debounce indicators
- Head-pitch scrolling status and actions
- Performance metrics (FPS, processing latency, queue size, warnings)
- Expandable / Collapsible detailed telemetry view
"""
import collections
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, Deque


class TelemetryGraphCanvas(ttk.Frame):
    """Real-time rolling analysis graph plotting movement, deadzone, pitch, and scroll states.
    
    Provides bounded rolling history (10-30s), live X/Y movement, dead-zone band,
    pitch thresholds, scroll state shading, pause/resume, and stale warning detection.
    """

    def __init__(self, parent: tk.Widget, max_history: int = 300, **kwargs):
        super().__init__(parent, **kwargs)
        self.max_history = max_history
        self.history: Deque[Dict[str, Any]] = collections.deque(maxlen=max_history)
        self.is_paused: bool = False
        self.last_update_time: float = time.monotonic()

        self._build_ui()

    def _build_ui(self) -> None:
        # Control header
        ctrl_bar = ttk.Frame(self)
        ctrl_bar.pack(fill=tk.X, pady=(0, 2))

        self.btn_pause = ttk.Button(ctrl_bar, text="Pause Graph", width=12, command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=2)

        self.btn_clear = ttk.Button(ctrl_bar, text="Clear History", width=12, command=self.clear_history)
        self.btn_clear.pack(side=tk.LEFT, padx=2)

        self.lbl_status = ttk.Label(
            ctrl_bar, text="Graph: LIVE (15s)", font=("Segoe UI", 8, "bold"), foreground="#009900"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=4)

        # Tkinter Canvas
        self.canvas_width = 380
        self.canvas_height = 190
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#181818",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=2)
        self.canvas.bind("<Configure>", self._on_resize)

        # Legend footer
        legend_frame = ttk.Frame(self)
        legend_frame.pack(fill=tk.X, pady=(1, 0))

        items = [
            ("Raw X", "#00c0ff"),
            ("Filt X", "#0066ff"),
            ("Raw Y", "#ffaa44"),
            ("Filt Y", "#ff6600"),
            ("Pitch", "#00ff66"),
            ("Deadzone", "#2d4a2d"),
            ("Scroll", "#cc33ff"),
        ]
        for name, col in items:
            lbl = tk.Label(legend_frame, text=f"■ {name}", font=("Segoe UI", 7, "bold"), fg=col, bg="#222222")
            lbl.pack(side=tk.LEFT, padx=2)

    def _on_resize(self, event):
        self.canvas_width = max(100, event.width)
        self.canvas_height = max(80, event.height)
        self.redraw()

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="Resume Graph")
            self.lbl_status.config(text="Graph: PAUSED", foreground="#ff8800")
        else:
            self.btn_pause.config(text="Pause Graph")
            self.lbl_status.config(text="Graph: LIVE", foreground="#009900")
            self.redraw()
        return self.is_paused

    def clear_history(self) -> None:
        self.history.clear()
        self.redraw()

    def add_sample(self, telemetry: Dict[str, Any]) -> None:
        now = time.monotonic()
        self.last_update_time = now

        face_detected = bool(telemetry.get("face_detected", False))
        sample = {
            "time": now,
            "face_detected": face_detected,
            "raw_dx": float(telemetry.get("raw_dx") or 0.0),
            "calibrated_dx": float(telemetry.get("calibrated_dx") or 0.0),
            "raw_dy": float(telemetry.get("raw_dy") or 0.0),
            "calibrated_dy": float(telemetry.get("calibrated_dy") or 0.0),
            "head_pitch": float(telemetry.get("head_pitch") or 0.0),
            "dead_zone": float(telemetry.get("dead_zone_radius") or 0.008),
            "act_thresh": float(telemetry.get("scroll_activation_thresh") or 0.020),
            "rel_thresh": float(telemetry.get("scroll_release_thresh") or 0.013),
            "scroll_state": str(telemetry.get("scroll_state", "NEUTRAL")),
        }
        self.history.append(sample)

        if not self.is_paused:
            self.redraw()

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")

        w = self.canvas_width
        h = self.canvas_height

        pad_top = 8
        pad_bottom = 12
        pad_left = 32
        pad_right = 8

        plot_w = max(10, w - pad_left - pad_right)
        total_h = max(20, h - pad_top - pad_bottom)
        h1 = int(total_h * 0.52)
        gap = 12
        h2 = total_h - h1 - gap

        y1_mid = pad_top + h1 // 2
        y2_mid = pad_top + h1 + gap + h2 // 2

        # Backgrounds
        c.create_rectangle(pad_left, pad_top, pad_left + plot_w, pad_top + h1, fill="#121212", outline="#2a2a2a")
        c.create_rectangle(pad_left, pad_top + h1 + gap, pad_left + plot_w, pad_top + h1 + gap + h2, fill="#121212", outline="#2a2a2a")

        # Subplot 1: Movement range [-0.05, +0.05]
        max_d = 0.050
        last_dz = self.history[-1]["dead_zone"] if self.history else 0.008
        dz_px = int((last_dz / max_d) * (h1 / 2))
        c.create_rectangle(
            pad_left, y1_mid - dz_px, pad_left + plot_w, y1_mid + dz_px,
            fill="#1e2d1e", outline=""
        )
        c.create_line(pad_left, y1_mid, pad_left + plot_w, y1_mid, fill="#383838", dash=(2, 2))
        c.create_text(pad_left - 2, y1_mid, text="0", anchor="e", fill="#888888", font=("Segoe UI", 6))
        c.create_text(pad_left - 2, pad_top + 4, text="+0.05", anchor="e", fill="#888888", font=("Segoe UI", 6))
        c.create_text(pad_left - 2, pad_top + h1 - 4, text="-0.05", anchor="e", fill="#888888", font=("Segoe UI", 6))
        c.create_text(pad_left + 4, pad_top + 2, text="Head Displacement (X/Y)", anchor="nw", fill="#777777", font=("Segoe UI", 7, "bold"))

        # Subplot 2: Pitch & Scroll range [-0.035, +0.035]
        max_p = 0.035
        act_th = self.history[-1]["act_thresh"] if self.history else 0.020
        rel_th = self.history[-1]["rel_thresh"] if self.history else 0.013
        act_px = int((act_th / max_p) * (h2 / 2))
        rel_px = int((rel_th / max_p) * (h2 / 2))

        c.create_line(pad_left, y2_mid, pad_left + plot_w, y2_mid, fill="#383838", dash=(2, 2))
        c.create_line(pad_left, y2_mid - act_px, pad_left + plot_w, y2_mid - act_px, fill="#cc3333", dash=(4, 2))
        c.create_line(pad_left, y2_mid + act_px, pad_left + plot_w, y2_mid + act_px, fill="#cc3333", dash=(4, 2))
        c.create_line(pad_left, y2_mid - rel_px, pad_left + plot_w, y2_mid - rel_px, fill="#9933cc", dash=(2, 2))
        c.create_line(pad_left, y2_mid + rel_px, pad_left + plot_w, y2_mid + rel_px, fill="#9933cc", dash=(2, 2))

        c.create_text(pad_left - 2, y2_mid, text="0", anchor="e", fill="#888888", font=("Segoe UI", 6))
        c.create_text(pad_left - 2, y2_mid - act_px, text="+Act", anchor="e", fill="#cc3333", font=("Segoe UI", 6))
        c.create_text(pad_left - 2, y2_mid + act_px, text="-Act", anchor="e", fill="#cc3333", font=("Segoe UI", 6))
        c.create_text(pad_left + 4, pad_top + h1 + gap + 2, text="Head Pitch & Scroll State", anchor="nw", fill="#777777", font=("Segoe UI", 7, "bold"))

        c.create_text(pad_left, h - 2, text="-15s", anchor="w", fill="#666666", font=("Segoe UI", 6))
        c.create_text(pad_left + plot_w, h - 2, text="Now", anchor="e", fill="#666666", font=("Segoe UI", 6))

        n = len(self.history)
        if n < 2:
            return

        # Stale warning
        if (time.monotonic() - self.last_update_time > 0.5) and not self.is_paused:
            c.create_text(
                pad_left + plot_w // 2, y1_mid,
                text="[!] TELEMETRY STALE / NO FACE DETECTED",
                fill="#ffaa00", font=("Segoe UI", 8, "bold")
            )
            self.lbl_status.config(text="Graph: STALE", foreground="#ff8800")
        elif not self.is_paused:
            self.lbl_status.config(text="Graph: LIVE (15s)", foreground="#009900")

        pts_raw_x = []
        pts_flt_x = []
        pts_raw_y = []
        pts_flt_y = []
        pts_pitch = []

        step_x = plot_w / max(1, self.max_history - 1)
        start_col = pad_left + (self.max_history - n) * step_x

        for i, s in enumerate(self.history):
            if not s["face_detected"]:
                continue
            x_coord = start_col + i * step_x

            def map_y1(val):
                clamped = max(-max_d, min(max_d, val))
                return y1_mid - (clamped / max_d) * (h1 / 2)

            pts_raw_x.extend([x_coord, map_y1(s["raw_dx"])])
            pts_flt_x.extend([x_coord, map_y1(s["calibrated_dx"])])
            pts_raw_y.extend([x_coord, map_y1(s["raw_dy"])])
            pts_flt_y.extend([x_coord, map_y1(s["calibrated_dy"])])

            def map_y2(val):
                clamped = max(-max_p, min(max_p, val))
                return y2_mid - (clamped / max_p) * (h2 / 2)

            pts_pitch.extend([x_coord, map_y2(s["head_pitch"])])

            st = s["scroll_state"]
            if st == "DOWN":
                c.create_line(x_coord, pad_top + h1 + gap, x_coord, pad_top + h1 + gap + h2, fill="#331122")
            elif st == "UP":
                c.create_line(x_coord, pad_top + h1 + gap, x_coord, pad_top + h1 + gap + h2, fill="#112233")

        if len(pts_raw_x) >= 4:
            c.create_line(pts_raw_x, fill="#00c0ff", dash=(2, 2), width=1)
        if len(pts_raw_y) >= 4:
            c.create_line(pts_raw_y, fill="#ffaa44", dash=(2, 2), width=1)
        if len(pts_flt_x) >= 4:
            c.create_line(pts_flt_x, fill="#0066ff", width=2)
        if len(pts_flt_y) >= 4:
            c.create_line(pts_flt_y, fill="#ff6600", width=2)
        if len(pts_pitch) >= 4:
            c.create_line(pts_pitch, fill="#00ff66", width=2)


class DiagnosticsDashboard(ttk.Frame):
    """Reusable, responsive Tkinter dashboard displaying live pipeline diagnostics."""

    def __init__(self, parent: tk.Widget, show_details: bool = False):
        super().__init__(parent, padding="4")
        self.show_details: bool = show_details

        self._build_ui()

    def _build_ui(self) -> None:
        """Constructs the dashboard sections, warning banner, and telemetry widgets."""
        # 1. Prominent Warning & Operational Status Banner
        self.banner_frame = tk.Frame(self, bg="#28a745", padx=8, pady=5, relief=tk.RIDGE, bd=1)
        self.banner_frame.pack(fill=tk.X, pady=2)

        self.lbl_banner = tk.Label(
            self.banner_frame,
            text="STATUS: INITIALIZING...",
            font=("Segoe UI", 9, "bold"),
            bg="#28a745",
            fg="#ffffff",
        )
        self.lbl_banner.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_toggle_details = tk.Button(
            self.banner_frame,
            text="[+] Details" if not self.show_details else "[-] Details",
            font=("Segoe UI", 8, "bold"),
            bg="#ffffff",
            fg="#333333",
            relief=tk.RAISED,
            bd=1,
            command=self.toggle_details,
        )
        self.btn_toggle_details.pack(side=tk.RIGHT, padx=2)

        # 2. Main Diagnostics Grid Container
        self.grid_container = ttk.Frame(self)
        self.grid_container.pack(fill=tk.BOTH, expand=True, pady=2)

        # Group A: System & Face Tracking Health
        grp_face = ttk.LabelFrame(self.grid_container, text="1. Tracking & Camera Status", padding="4")
        grp_face.pack(fill=tk.X, pady=2)

        self.lbl_camera = ttk.Label(grp_face, text="Camera: Connected (ID 0) | Hardware: -- FPS", font=("Segoe UI", 8))
        self.lbl_camera.pack(anchor=tk.W)

        self.lbl_face = ttk.Label(grp_face, text="Face Tracking: Searching...", font=("Segoe UI", 8, "bold"))
        self.lbl_face.pack(anchor=tk.W)

        self.lbl_op_state = ttk.Label(grp_face, text="Operational State: IDLE", font=("Segoe UI", 8, "bold"), foreground="#009900")
        self.lbl_op_state.pack(anchor=tk.W)

        # Group B: Cursor Movement Telemetry
        grp_cursor = ttk.LabelFrame(self.grid_container, text="2. Pointer Movement Telemetry", padding="4")
        grp_cursor.pack(fill=tk.X, pady=2)

        self.lbl_cursor_coords = ttk.Label(grp_cursor, text="Target: (---, ---) | Screen: (---, ---)", font=("Segoe UI", 8))
        self.lbl_cursor_coords.pack(anchor=tk.W)

        self.lbl_mouse_command = ttk.Label(grp_cursor, text="Mouse Command Issued: NO", font=("Segoe UI", 8, "bold"))
        self.lbl_mouse_command.pack(anchor=tk.W)

        self.lbl_calibrated_move = ttk.Label(grp_cursor, text="Calibrated Move: norm_dx=+0.000, norm_dy=+0.000", font=("Segoe UI", 8))
        self.lbl_calibrated_move.pack(anchor=tk.W)

        # Group C: Eye Gestures & Head-Pitch Scrolling
        grp_gestures = ttk.LabelFrame(self.grid_container, text="3. Eye Gestures & Head Pitch", padding="4")
        grp_gestures.pack(fill=tk.X, pady=2)

        self.lbl_ear_status = ttk.Label(grp_gestures, text="EAR: L=0.000 [OPEN] | R=0.000 [OPEN]", font=("Segoe UI", 8))
        self.lbl_ear_status.pack(anchor=tk.W)

        self.lbl_eye_state = ttk.Label(grp_gestures, text="Eye State: OPEN | Blinks: Normal", font=("Segoe UI", 8))
        self.lbl_eye_state.pack(anchor=tk.W)

        self.lbl_click_action = ttk.Label(grp_gestures, text="Click Command: NONE | Cooldown: Ready", font=("Segoe UI", 8, "bold"), foreground="#0066cc")
        self.lbl_click_action.pack(anchor=tk.W)

        self.lbl_scroll_info = ttk.Label(grp_gestures, text="Pitch: +0.000 | Scroll: NEUTRAL", font=("Segoe UI", 8, "bold"), foreground="#aa00cc")
        self.lbl_scroll_info.pack(anchor=tk.W)

        # Group D: Performance Metrics
        grp_perf = ttk.LabelFrame(self.grid_container, text="4. Real-Time Performance", padding="4")
        grp_perf.pack(fill=tk.X, pady=2)

        self.lbl_performance = ttk.Label(grp_perf, text="PROC: -- FPS | GUI: -- FPS | Latency: -- ms", font=("Segoe UI", 8))
        self.lbl_performance.pack(anchor=tk.W)

        # Group E: Live Pipeline Analysis Graph
        grp_graph = ttk.LabelFrame(self.grid_container, text="5. Live Pipeline Analysis Graph", padding="4")
        grp_graph.pack(fill=tk.BOTH, expand=True, pady=2)

        self.graph = TelemetryGraphCanvas(grp_graph)
        self.graph.pack(fill=tk.BOTH, expand=True)

        self.btn_pause_graph = self.graph.btn_pause
        self.btn_clear_graph = self.graph.btn_clear
        self.lbl_graph_status = self.graph.lbl_status

        # 3. Expandable Detailed Telemetry Frame
        self.detail_frame = ttk.LabelFrame(self, text="Detailed Telemetry Diagnostics", padding="6")
        if self.show_details:
            self.detail_frame.pack(fill=tk.X, pady=4)

        self.lbl_detail_raw = ttk.Label(self.detail_frame, text="Raw Head Movement: dx=+0.0000, dy=+0.0000", font=("Segoe UI", 8))
        self.lbl_detail_raw.pack(anchor=tk.W)

        self.lbl_detail_filtered = ttk.Label(self.detail_frame, text="Filtered Point: x=0.0, y=0.0", font=("Segoe UI", 8))
        self.lbl_detail_filtered.pack(anchor=tk.W)

        self.lbl_detail_ear_baseline = ttk.Label(self.detail_frame, text="Eye Baselines: Base_L=0.25, Base_R=0.25", font=("Segoe UI", 8))
        self.lbl_detail_ear_baseline.pack(anchor=tk.W)

        self.lbl_detail_tuning = ttk.Label(self.detail_frame, text="Settings: Deadzone=0.008, Sens=(2.5, 2.5)", font=("Segoe UI", 8))
        self.lbl_detail_tuning.pack(anchor=tk.W)

        self.lbl_detail_queue = ttk.Label(self.detail_frame, text="Frame Queue: 0 items | Errors: 0", font=("Segoe UI", 8))
        self.lbl_detail_queue.pack(anchor=tk.W)

        # Backward compatibility aliases for GUI and automated tests
        self.banner_lbl = self.lbl_banner
        self.lbl_mouse_moved = self.lbl_mouse_command
        self.lbl_target_coords = self.lbl_cursor_coords
        self.lbl_raw_dx = self.lbl_detail_raw
        self.lbl_raw_dy = self.lbl_detail_raw
        self.lbl_calibrated_dx = self.lbl_calibrated_move
        self.lbl_calibrated_dy = self.lbl_calibrated_move
        self.lbl_ear_left = self.lbl_ear_status
        self.lbl_ear_right = self.lbl_ear_status
        self.lbl_pitch = self.lbl_scroll_info
        self.lbl_scroll_state = self.lbl_scroll_info
        self.lbl_gesture_event = self.lbl_click_action
        self.lbl_fps = self.lbl_performance
        self.lbl_latency = self.lbl_performance
        self.lbl_input_enabled = self.lbl_mouse_command
        self._toggle_details = self.toggle_details

    @property
    def details_visible(self) -> bool:
        return self.show_details

    def toggle_details(self) -> None:
        """Toggles visibility of the detailed telemetry frame."""
        self.show_details = not self.show_details
        if self.show_details:
            self.detail_frame.pack(fill=tk.X, pady=4)
            self.btn_toggle_details.config(text="[-] Details")
        else:
            self.detail_frame.pack_forget()
            self.btn_toggle_details.config(text="[+] Details")

    def update_dashboard(self, telemetry: Dict[str, Any]) -> None:
        """Thread-safe update of all telemetry indicators from pipeline data."""
        if not isinstance(telemetry, dict):
            return

        # Feed sample to Live Analysis Graph
        if hasattr(self, "graph") and self.graph is not None:
            self.graph.add_sample(telemetry)

        # 1. Camera & Tracking Status
        cam_status = telemetry.get("camera_status", "Connected")
        cam_fps = telemetry.get("cam_fps", 0.0)
        cam_id = telemetry.get("camera_id", 0)
        self.lbl_camera.config(text=f"Camera: {cam_status} (ID {cam_id}) | Hardware: {cam_fps:.0f} FPS")

        face_detected = telemetry.get("face_detected", False)
        lmarks = telemetry.get("landmarks_count", 478 if face_detected else 0)
        if face_detected:
            self.lbl_face.config(text=f"Face Tracking: DETECTED ({lmarks} landmarks)", foreground="#009900")
        else:
            self.lbl_face.config(text="Face Tracking: NOT DETECTED (Searching...)", foreground="#cc0000")

        op_state = telemetry.get("operational_state", "IDLE")
        self.lbl_op_state.config(text=f"Operational State: {op_state}")

        # Update Visual Warning Banner based on priority state
        if telemetry.get("is_emergency_stopped", False):
            self.banner_frame.config(bg="#cc0000")
            self.lbl_banner.config(text="[!] EMERGENCY STOP ACTIVE - ALL INPUT DISABLED", bg="#cc0000", fg="#ffffff")
        elif telemetry.get("is_paused", False):
            self.banner_frame.config(bg="#ff8800")
            self.lbl_banner.config(text="[PAUSED] MOUSE CONTROL PAUSED (Press 'P' to Resume)", bg="#ff8800", fg="#ffffff")
        elif cam_status == "Error":
            self.banner_frame.config(bg="#cc0000")
            self.lbl_banner.config(text="[!] CAMERA ERROR - VERIFY WEBCAM CONNECTION", bg="#cc0000", fg="#ffffff")
        elif not face_detected:
            self.banner_frame.config(bg="#cc0000")
            self.lbl_banner.config(text="[!] WARNING: FACE NOT DETECTED - INPUT FROZEN", bg="#cc0000", fg="#ffffff")
        elif op_state.startswith("CALIBRATING"):
            self.banner_frame.config(bg="#007bff")
            self.lbl_banner.config(text=f"[CALIBRATING] {op_state} - HOLD STILL", bg="#007bff", fg="#ffffff")
        elif telemetry.get("scroll_state") in ("UP", "DOWN"):
            self.banner_frame.config(bg="#6f42c1")
            self.lbl_banner.config(text=f"[SCROLLING] Active: SCROLL {telemetry.get('scroll_state')}", bg="#6f42c1", fg="#ffffff")
        elif telemetry.get("is_calibrated", False):
            self.banner_frame.config(bg="#28a745")
            self.lbl_banner.config(text="SYSTEM ACTIVE: TRACKING & INPUT ENABLED", bg="#28a745", fg="#ffffff")
        else:
            self.banner_frame.config(bg="#ff8800")
            self.lbl_banner.config(text="NEEDS CALIBRATION (Press 'C')", bg="#ff8800", fg="#ffffff")

        # 2. Pointer Movement
        tgt_x = telemetry.get("target_x")
        tgt_x = float(tgt_x) if tgt_x is not None else 0.0
        tgt_y = telemetry.get("target_y")
        tgt_y = float(tgt_y) if tgt_y is not None else 0.0

        raw_cpos = telemetry.get("cursor_pos")
        if isinstance(raw_cpos, (tuple, list)) and len(raw_cpos) >= 2:
            cur_x, cur_y = raw_cpos[0], raw_cpos[1]
        else:
            cur_x, cur_y = 0, 0
        self.lbl_cursor_coords.config(text=f"Target: ({tgt_x:.0f}, {tgt_y:.0f}) | Screen: ({cur_x}, {cur_y})")

        moved = telemetry.get("mouse_moved", False)
        suppressed = telemetry.get("mouse_suppressed", False)
        suppress_reason = telemetry.get("mouse_suppress_reason", "NONE")
        if moved:
            self.lbl_mouse_command.config(text="Mouse Command Issued: YES", foreground="#009900")
        elif suppressed:
            col = "#ff8800" if suppress_reason == "SCROLLING" else "#cc0000"
            self.lbl_mouse_command.config(text=f"Mouse Command Issued: NO (Suppressed: {suppress_reason})", foreground=col)
        else:
            self.lbl_mouse_command.config(text=f"Mouse Command Issued: NO ({suppress_reason})", foreground="#666666")

        cal_x = telemetry.get("calibrated_dx", 0.0)
        cal_y = telemetry.get("calibrated_dy", 0.0)
        self.lbl_calibrated_move.config(text=f"Calibrated Move: norm_dx={cal_x:+.3f}, norm_dy={cal_y:+.3f}")

        # 3. Eye Gestures & Head Pitch
        l_ear = telemetry.get("left_ear", 0.0)
        r_ear = telemetry.get("right_ear", 0.0)
        l_stat = "OPEN" if telemetry.get("left_open", True) else "CLOSED"
        r_stat = "OPEN" if telemetry.get("right_open", True) else "CLOSED"
        self.lbl_ear_status.config(text=f"EAR: L={l_ear:.3f} [{l_stat}] | R={r_ear:.3f} [{r_stat}]")

        eye_state = telemetry.get("eye_state", "OPEN")
        blink_note = "Blinking (Ignored as click)" if eye_state == "BLINK" else "Normal"
        self.lbl_eye_state.config(text=f"Eye State: {eye_state} | Blinks: {blink_note}")

        act = telemetry.get("active_action", "NONE")
        in_cooldown = telemetry.get("in_click_cooldown", False)
        cd_rem = telemetry.get("cooldown_remaining_s", 0.0)
        cd_str = f"Cooldown: Active ({cd_rem:.2f}s)" if in_cooldown else "Cooldown: Ready"
        self.lbl_click_action.config(text=f"Click Command: {act} | {cd_str}")

        pitch = float(telemetry.get("head_pitch") or 0.0)
        raw_p = float(telemetry.get("raw_pitch") or 0.0)
        act_th = float(telemetry.get("scroll_activation_thresh") or 0.020)
        rel_th = float(telemetry.get("scroll_release_thresh") or 0.009)
        scr_st = telemetry.get("scroll_state", "NEUTRAL")
        trans_reason = telemetry.get("scroll_transition_reason", "STEADY")
        self.lbl_scroll_info.config(
            text=f"Pitch: {pitch:+.3f} (raw:{raw_p:.3f}, trig:+-{act_th:.3f}, rel:+-{rel_th:.3f}) | Scroll: {scr_st} [{trans_reason}]"
        )

        # 4. Performance
        proc_f = telemetry.get("proc_fps", 0.0)
        gui_f = telemetry.get("gui_fps", 0.0)
        lat_m = telemetry.get("latency_ms", 0.0)
        self.lbl_performance.config(text=f"PROC: {proc_f:.0f} FPS | GUI: {gui_f:.0f} FPS | Latency: {lat_m:.0f} ms")

        # 5. Expandable Details Update
        if self.show_details:
            raw_x = float(telemetry.get("raw_dx") or 0.0)
            raw_y = float(telemetry.get("raw_dy") or 0.0)
            self.lbl_detail_raw.config(text=f"Raw Head Movement: dx={raw_x:+.4f}, dy={raw_y:+.4f}")

            fil_x = float(telemetry.get("filtered_x") or 0.0)
            fil_y = float(telemetry.get("filtered_y") or 0.0)
            self.lbl_detail_filtered.config(text=f"Filtered Point: x={fil_x:.1f}, y={fil_y:.1f}")

            bl = float(telemetry.get("baseline_left") or 0.25)
            br = float(telemetry.get("baseline_right") or 0.25)
            tl = float(telemetry.get("thresh_left") or 0.18)
            tr = float(telemetry.get("thresh_right") or 0.18)
            self.lbl_detail_ear_baseline.config(text=f"Eye Baselines: Base_L={bl:.2f} (thr:{tl:.2f}), Base_R={br:.2f} (thr:{tr:.2f})")

            dz = float(telemetry.get("dead_zone_radius") or 0.008)
            sx = float(telemetry.get("sensitivity_x") or 2.5)
            sy = float(telemetry.get("sensitivity_y") or 2.5)
            self.lbl_detail_tuning.config(text=f"Settings: Deadzone={dz:.3f}, Sens=({sx:.1f}, {sy:.1f})")

            q_sz = int(telemetry.get("queue_size") or 0)
            err_cnt = int(telemetry.get("camera_errors_count") or 0)
            self.lbl_detail_queue.config(text=f"Frame Queue: {q_sz} items | Camera Errors: {err_cnt}")

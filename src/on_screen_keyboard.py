"""On-Screen Virtual Keyboard for Facial Cursor Control System.

Provides an accessible, professional on-screen keyboard supporting:
- Alphabet A-Z, Numbers 0-9, Symbols and Punctuation
- Space, Backspace, Enter, Shift, Caps Lock, Clear with confirmation
- Multiline text preview area
- Cursor hover highlighting
- Wink-click selection with single-activation debounce
- Configurable dwell-click mode with visual progress and leave-cancellation
- Scroll-state protection against accidental typing
- Explicit, cancelable "Type into Active Application" action
"""
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Dict, List, Tuple, Any


# Standard Keyboard Layouts: Normal vs Shifted
KEYBOARD_LAYOUT_NORMAL = [
    ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
    ["Tab", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", "\\"],
    ["Caps", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "Enter"],
    ["Shift", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "Shift"],
    ["Clear", "Space", "Type into App", "Minimize", "Close"],
]

KEYBOARD_LAYOUT_SHIFT = [
    ["~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "Backspace"],
    ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "{", "}", "|"],
    ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "\"", "Enter"],
    ["Shift", "Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "Shift"],
    ["Clear", "Space", "Type into App", "Minimize", "Close"],
]


class OnScreenKeyboard:
    """Accessible On-Screen Virtual Keyboard window."""

    def __init__(
        self,
        master: tk.Tk,
        config: Optional[Any] = None,
        on_close_callback: Optional[Callable[[], None]] = None,
        dwell_enabled: Optional[bool] = None,
        dwell_duration_s: Optional[float] = None,
    ):
        self.master = master

        # Support flexible calling conventions:
        # OnScreenKeyboard(master, config, on_close_callback=cb)
        # OnScreenKeyboard(master, on_close_callback=cb)
        # OnScreenKeyboard(master, cb)
        if callable(config) and on_close_callback is None:
            on_close_callback = config
            config = None

        self.config = config
        self.on_close_callback = on_close_callback

        if dwell_enabled is None:
            if config is not None:
                dwell_enabled = getattr(config, "keyboard_dwell_enabled", getattr(config, "dwell_click_enabled", False))
            else:
                dwell_enabled = False
        self.dwell_enabled: bool = bool(dwell_enabled)

        if dwell_duration_s is None:
            if config is not None:
                dwell_duration_s = getattr(config, "keyboard_dwell_time_s", getattr(config, "dwell_duration_s", 1.0))
            else:
                dwell_duration_s = 1.0
        self.dwell_duration_s: float = max(0.3, float(dwell_duration_s))

        # Keyboard State
        self.is_shift: bool = False
        self.is_caps: bool = False
        self.is_minimized: bool = False
        self.is_open: bool = True
        self.is_scrolling_suppressed: bool = False

        # Single-activation debounce
        self._last_key_press_time: float = 0.0
        self._click_debounce_s: float = 0.25

        # Dwell tracking
        self._current_hover_key: Optional[str] = None
        self._current_hover_btn: Optional[tk.Button] = None
        self._dwell_start_time: float = 0.0
        self._dwell_timer_id: Optional[str] = None
        self._dwell_triggered_for_current: bool = False

        # External typing state
        self._external_typing_countdown: int = 0
        self._external_typing_timer_id: Optional[str] = None
        self.btn_cancel_ext: Optional[tk.Button] = None

        # Build Toplevel Window
        self.window = tk.Toplevel(self.master)
        self.window.title("Virtual Accessible Keyboard")
        self.window.geometry("820x460")
        self.window.minsize(680, 380)
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        # Key button references for dynamic updating and visual highlighting
        self._buttons: Dict[str, List[tk.Button]] = {}
        self._btn_widget_map: Dict[tk.Button, str] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Constructs text preview, keyboard control bar, and key grid."""
        # Top Container: Multiline Text Area with Scrollbar
        text_frame = ttk.Frame(self.window, padding="6")
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.text_preview = tk.Text(
            text_frame,
            height=4,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg="#fdfdfd",
            fg="#222222",
            padx=8,
            pady=6,
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.text_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_preview.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_preview.config(yscrollcommand=scrollbar.set)

        # Dwell & Status Bar
        status_bar = ttk.Frame(self.window, padding="4 2")
        status_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = ttk.Label(
            status_bar,
            text="Ready | Hover or Wink-Click keys",
            font=("Segoe UI", 9, "bold"),
            foreground="#0066cc",
        )
        self.lbl_status.pack(side=tk.LEFT, padx=6)

        # Dwell Click Toggle Checkbox
        self.chk_dwell_var = tk.BooleanVar(value=self.dwell_enabled)
        self.chk_dwell = ttk.Checkbutton(
            status_bar,
            text="Dwell-Click",
            variable=self.chk_dwell_var,
            command=self._on_toggle_dwell,
        )
        self.chk_dwell.pack(side=tk.RIGHT, padx=4)

        # Dwell Progress Bar
        self.dwell_progress = ttk.Progressbar(
            status_bar,
            orient=tk.HORIZONTAL,
            length=140,
            mode="determinate",
            maximum=100,
        )
        self.dwell_progress.pack(side=tk.RIGHT, padx=6)
        ttk.Label(status_bar, text="Dwell:", font=("Segoe UI", 8)).pack(side=tk.RIGHT)

        # Key Grid Container
        self.grid_frame = ttk.Frame(self.window, padding="4")
        self.grid_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self._render_key_grid()

    def _render_key_grid(self) -> None:
        """Renders the grid of virtual keys based on current Shift/Caps mode."""
        # Clear existing grid
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._buttons.clear()
        self._btn_widget_map.clear()

        layout = KEYBOARD_LAYOUT_SHIFT if (self.is_shift ^ self.is_caps) else KEYBOARD_LAYOUT_NORMAL

        for row_idx, row in enumerate(layout):
            row_frame = ttk.Frame(self.grid_frame)
            row_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)

            for col_idx, key in enumerate(row):
                # Calculate relative key weight for layout aesthetics
                key_text = key
                btn_bg = "#f0f0f0"
                btn_fg = "#111111"
                btn_font = ("Segoe UI", 10, "bold")
                width = 3

                if key in ("Backspace", "Enter", "Shift", "Caps", "Tab"):
                    width = 6
                    btn_bg = "#e3e3e3"
                elif key == "Space":
                    width = 18
                    btn_bg = "#e8e8e8"
                elif key == "Type into App":
                    width = 12
                    btn_bg = "#d9edf7"
                    btn_fg = "#004085"
                elif key == "Clear":
                    width = 6
                    btn_bg = "#f8d7da"
                    btn_fg = "#721c24"
                elif key in ("Minimize", "Close"):
                    width = 6
                    btn_bg = "#e2e3e5"

                # Indicate active Caps Lock or Shift
                if key == "Caps" and self.is_caps:
                    btn_bg = "#28a745"
                    btn_fg = "#ffffff"
                elif key == "Shift" and self.is_shift:
                    btn_bg = "#007bff"
                    btn_fg = "#ffffff"

                btn = tk.Button(
                    row_frame,
                    text=key_text,
                    font=btn_font,
                    bg=btn_bg,
                    fg=btn_fg,
                    activebackground="#0078d7",
                    activeforeground="#ffffff",
                    relief=tk.RAISED,
                    bd=2,
                    cursor="hand2",
                    command=lambda k=key: self._on_key_click(k),
                )
                btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

                # Bind hover events for visual highlighting and dwell-click tracking
                btn.bind("<Enter>", lambda e, b=btn, k=key: self._on_key_enter(b, k))
                btn.bind("<Leave>", lambda e, b=btn, k=key: self._on_key_leave(b, k))

                if key not in self._buttons:
                    self._buttons[key] = []
                self._buttons[key].append(btn)
                self._btn_widget_map[btn] = key

    def _on_key_enter(self, btn: tk.Button, key: str) -> None:
        """Handles cursor entering a key boundary."""
        if not self.is_open:
            return

        # Visual highlighting
        btn.config(bg="#cce5ff", relief=tk.SOLID)
        self._current_hover_key = key
        self._current_hover_btn = btn
        self.lbl_status.config(text=f"Hovering: [{key}]")

        # Start dwell tracking if dwell mode is active and not currently scrolling
        if self.dwell_enabled and not self.is_scrolling_suppressed:
            self._dwell_triggered_for_current = False
            self._dwell_start_time = time.monotonic()
            self._start_dwell_tick(btn, key)

    def _on_key_leave(self, btn: tk.Button, key: str) -> None:
        """Handles cursor leaving a key boundary."""
        # Restore normal styling
        is_active_caps = (key == "Caps" and self.is_caps)
        is_active_shift = (key == "Shift" and self.is_shift)

        if is_active_caps:
            btn.config(bg="#28a745", fg="#ffffff", relief=tk.RAISED)
        elif is_active_shift:
            btn.config(bg="#007bff", fg="#ffffff", relief=tk.RAISED)
        elif key == "Type into App":
            btn.config(bg="#d9edf7", fg="#004085", relief=tk.RAISED)
        elif key == "Clear":
            btn.config(bg="#f8d7da", fg="#721c24", relief=tk.RAISED)
        else:
            btn.config(bg="#f0f0f0", fg="#111111", relief=tk.RAISED)

        if self._current_hover_key == key or self._current_hover_btn == btn:
            self._current_hover_key = None
            self._current_hover_btn = None
            self._cancel_dwell()

    def _start_dwell_tick(self, btn: tk.Button, key: str) -> None:
        """Schedules recurring dwell ticks to animate progress bar and trigger key."""
        self._cancel_dwell()

        def _tick():
            if not self.dwell_enabled or self.is_scrolling_suppressed or self._current_hover_key != key:
                self._cancel_dwell()
                return

            if self._dwell_triggered_for_current:
                return

            now = time.monotonic()
            elapsed = now - self._dwell_start_time
            progress_pct = min(100.0, (elapsed / self.dwell_duration_s) * 100.0)
            self.dwell_progress["value"] = progress_pct

            if progress_pct >= 100.0:
                # Dwell complete: trigger key action
                self._dwell_triggered_for_current = True
                self.dwell_progress["value"] = 100.0
                self._on_key_click(key)
                if self.is_open:
                    self.lbl_status.config(text=f"Activated [{key}] via Dwell")
                # Reset progress bar after brief visual confirmation
                if self.is_open:
                    self._reset_progress_timer_id = self.window.after(
                        150, lambda: self.dwell_progress.config(value=0) if self.is_open else None
                    )
            else:
                self._dwell_timer_id = self.window.after(30, _tick)

        self._dwell_timer_id = self.window.after(30, _tick)

    def _cancel_dwell(self) -> None:
        """Cancels any pending dwell timer and resets progress bar."""
        if self._dwell_timer_id:
            try:
                self.window.after_cancel(self._dwell_timer_id)
            except Exception:
                pass
            self._dwell_timer_id = None
        if hasattr(self, "_reset_progress_timer_id") and self._reset_progress_timer_id:
            try:
                self.window.after_cancel(self._reset_progress_timer_id)
            except Exception:
                pass
            self._reset_progress_timer_id = None
        self.dwell_progress["value"] = 0
        if not self._external_typing_timer_id and self.is_open:
            self.lbl_status.config(text="Ready")

    def _on_key_click(self, key: str) -> None:
        """Dispatches key actions with single-activation debounce and scrolling guard."""
        now = time.monotonic()
        # Single-activation debounce
        if now - self._last_key_press_time < self._click_debounce_s:
            return
        self._last_key_press_time = now

        # Prevent scrolling from accidentally activating keys
        if self.is_scrolling_suppressed:
            return

        # Special Action Dispatch
        if key == "Backspace":
            self.handle_backspace()
        elif key == "Space":
            self.insert_text(" ")
        elif key == "Enter":
            self.insert_text("\n")
        elif key == "Tab":
            self.insert_text("    ")
        elif key == "Caps":
            self.is_caps = not self.is_caps
            self._render_key_grid()
        elif key == "Shift":
            self.is_shift = not self.is_shift
            self._render_key_grid()
        elif key == "Clear":
            self.handle_clear()
        elif key == "Type into App":
            self.start_external_typing()
        elif key == "Minimize":
            self.toggle_minimize()
        elif key == "Close":
            self.close()
        else:
            # Printable characters
            self.insert_text(key)
            if self.is_shift:
                self.is_shift = False
                self._render_key_grid()

    def insert_text(self, text: str) -> None:
        """Appends text to the multiline preview area."""
        self.text_preview.insert(tk.END, text)
        self.text_preview.see(tk.END)

    def handle_backspace(self) -> None:
        """Deletes the character before the current cursor position."""
        content = self.get_text()
        if len(content) > 0:
            self.text_preview.delete("end-2c", "end-1c")

    def handle_clear(self) -> None:
        """Clears composed text, prompting confirmation if text is substantial."""
        content = self.get_text().strip()
        if len(content) > 10:
            confirm = messagebox.askyesno(
                "Clear Text",
                f"Are you sure you want to clear your message ({len(content)} characters)?",
                parent=self.window,
            )
            if not confirm:
                return
        self.text_preview.delete("1.0", tk.END)
        self.lbl_status.config(text="Text cleared.")

    def get_text(self) -> str:
        """Returns the full text currently composed in the preview area."""
        # tk.Text always adds a trailing newline, so slice to exclude it
        return self.text_preview.get("1.0", "end-1c")

    def set_scrolling_suppressed(self, is_scrolling: bool) -> None:
        """Notified by GUI when head-pitch scrolling is active to suppress accidental key clicks."""
        self.is_scrolling_suppressed = is_scrolling
        if is_scrolling:
            self._cancel_dwell()

    def _on_toggle_dwell(self) -> None:
        """Toggles dwell-click mode from GUI checkbox."""
        self.dwell_enabled = self.chk_dwell_var.get()
        if not self.dwell_enabled:
            self._cancel_dwell()

    def start_external_typing(self) -> None:
        """Initiates explicit 'Type into Active Application' with 3-second countdown and Cancel button."""
        text = self.get_text()
        if not text:
            messagebox.showinfo("Type into App", "There is no composed text to type.", parent=self.window)
            return

        if self._external_typing_timer_id:
            self.cancel_external_typing()
            return

        self._external_typing_countdown = 3
        self._show_typing_overlay(text)

    def _show_typing_overlay(self, text_to_type: str) -> None:
        """Displays countdown overlay allowing the user to switch focus or cancel."""
        self.lbl_status.config(
            text=f"Typing into external app in {self._external_typing_countdown}s... (Click Cancel to abort)",
            foreground="#d9534f",
        )
        if not hasattr(self, "btn_cancel_ext") or self.btn_cancel_ext is None:
            self.btn_cancel_ext = tk.Button(
                self.window,
                text="[X] CANCEL TYPING",
                font=("Segoe UI", 9, "bold"),
                bg="#d9534f",
                fg="#ffffff",
                command=self.cancel_external_typing,
            )
            self.btn_cancel_ext.pack(side=tk.TOP, pady=2)

        def _step():
            if self._external_typing_countdown > 1:
                self._external_typing_countdown -= 1
                self.lbl_status.config(
                    text=f"Typing in {self._external_typing_countdown}s... (Click 'Type into App' to Cancel)",
                    foreground="#d9534f",
                )
                self._external_typing_timer_id = self.window.after(1000, _step)
            else:
                self.lbl_status.config(text="Sending keystrokes...", foreground="#28a745")
                self._external_typing_timer_id = None
                if hasattr(self, "btn_cancel_ext") and self.btn_cancel_ext is not None:
                    try:
                        self.btn_cancel_ext.destroy()
                    except Exception:
                        pass
                    self.btn_cancel_ext = None
                self._execute_external_typing(text_to_type)

        self._external_typing_timer_id = self.window.after(1000, _step)

    def cancel_external_typing(self) -> None:
        """Cancels an in-progress external typing countdown."""
        if self._external_typing_timer_id:
            try:
                self.window.after_cancel(self._external_typing_timer_id)
            except Exception:
                pass
            self._external_typing_timer_id = None
        if hasattr(self, "btn_cancel_ext") and self.btn_cancel_ext is not None:
            try:
                self.btn_cancel_ext.destroy()
            except Exception:
                pass
            self.btn_cancel_ext = None
        if self.is_open:
            self.lbl_status.config(text="External typing Cancelled.", foreground="#0066cc")

    def _execute_external_typing(self, text: str) -> None:
        """Safely types composed text using PyAutoGUI without storing text."""
        if hasattr(self, "btn_cancel_ext") and self.btn_cancel_ext is not None:
            try:
                self.btn_cancel_ext.destroy()
            except Exception:
                pass
            self.btn_cancel_ext = None
        try:
            import pyautogui
            # Send text line by line to preserve newlines cleanly
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line:
                    pyautogui.write(line, interval=0.01)
                if i < len(lines) - 1:
                    pyautogui.press("enter")
            if self.is_open:
                self.lbl_status.config(text="Text typed into external application successfully!", foreground="#28a745")
        except Exception as e:
            if self.is_open:
                self.lbl_status.config(text=f"Typing failed: {e}", foreground="#d9534f")

    def update_cursor_coords(self, screen_x: int, screen_y: int) -> Optional[str]:
        """Direct hit-testing using screen coordinates from facial cursor controller.
        
        Provides reliable key hover highlighting and dwell progress even when
        OS cursor moveTo events bypass standard Tkinter mouse enter/leave events.
        """
        if not self.is_open:
            return None

        try:
            if not self.window.winfo_exists():
                return None
            win_x = self.window.winfo_rootx()
            win_y = self.window.winfo_rooty()
            win_w = self.window.winfo_width()
            win_h = self.window.winfo_height()
        except Exception:
            return None

        # Check if cursor is within keyboard window
        if not (win_x <= screen_x <= win_x + win_w and win_y <= screen_y <= win_y + win_h):
            if self._current_hover_btn is not None:
                self._on_key_leave(self._current_hover_btn, self._current_hover_key or "")
            return None

        # Hit-test button widgets
        hit_btn: Optional[tk.Button] = None
        hit_key: Optional[str] = None
        for btn, key in self._btn_widget_map.items():
            try:
                bx = btn.winfo_rootx()
                by = btn.winfo_rooty()
                bw = btn.winfo_width()
                bh = btn.winfo_height()
                if bx <= screen_x <= bx + bw and by <= screen_y <= by + bh:
                    hit_btn = btn
                    hit_key = key
                    break
            except Exception:
                continue

        if hit_btn is not None and hit_key is not None:
            if hit_btn != self._current_hover_btn:
                if self._current_hover_btn is not None:
                    self._on_key_leave(self._current_hover_btn, self._current_hover_key or "")
                self._on_key_enter(hit_btn, hit_key)
            return hit_key
        else:
            if self._current_hover_btn is not None:
                self._on_key_leave(self._current_hover_btn, self._current_hover_key or "")
            return None

    def handle_click_event(self, screen_x: Optional[int] = None, screen_y: Optional[int] = None) -> bool:
        """Invoked when a validated wink-click occurs. Activates hovered key if cursor is over keyboard."""
        if not self.is_open or self.is_scrolling_suppressed:
            return False

        key = None
        if screen_x is not None and screen_y is not None:
            key = self.update_cursor_coords(screen_x, screen_y)
        if not key:
            key = self._current_hover_key

        if key and self._current_hover_btn:
            self._on_key_click(key)
            # Brief visual confirmation flash
            btn = self._current_hover_btn
            try:
                btn.config(bg="#28a745", fg="#ffffff")
                self.window.after(120, lambda b=btn, k=key: (
                    b.config(bg="#cce5ff", fg="#111111") if self.is_open and b == self._current_hover_btn else None
                ))
            except Exception:
                pass
            return True
        return False

    def toggle_minimize(self) -> None:
        """Toggles between minimized floating bar and full keyboard layout."""
        if not self.is_minimized:
            self.is_minimized = True
            self.grid_frame.pack_forget()
            self.window.geometry("450x120")
            self.lbl_status.config(text="Keyboard Minimized")
        else:
            self.is_minimized = False
            self.grid_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            self.window.geometry("820x460")
            self.lbl_status.config(text="Keyboard Restored")

    def close(self) -> None:
        """Closes the keyboard window and invokes the on_close callback safely."""
        if not self.is_open:
            return
        self.is_open = False

        # 1. Cancel pending dwell & typing countdown timers
        self._cancel_dwell()
        self.cancel_external_typing()

        if hasattr(self, "btn_cancel_ext") and self.btn_cancel_ext is not None:
            try:
                self.btn_cancel_ext.destroy()
            except Exception:
                pass
            self.btn_cancel_ext = None

        # 2. Reset keyboard shift/caps states safely
        self.is_shift = False
        self.is_caps = False

        # 3. Destroy toplevel window
        try:
            self.window.destroy()
        except Exception:
            pass

        # 4. Invoke close callback exactly once
        cb = self.on_close_callback
        self.on_close_callback = None
        if cb:
            try:
                cb()
            except Exception:
                pass
            except Exception as e:
                print(f"[OnScreenKeyboard] Error in on_close_callback: {e}")

"""Camera input manager for webcam frame acquisition with zero-latency threaded capture."""
import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from src.config import AppConfig


class CameraManager:
    """Manages webcam capture, threaded frame acquisition, and latest-frame retrieval."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

        # Threaded capture synchronization
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._has_new_frame: bool = False

        # Camera hardware FPS telemetry
        self.camera_fps: float = 0.0
        self._fps_frame_count: int = 0
        self._fps_timer: float = time.monotonic()

    def start(self) -> bool:
        """Initializes the webcam device and starts the capture thread."""
        if self.is_opened and self._thread is not None and self._thread.is_alive():
            return True

        # cv2.CAP_DSHOW on Windows provides fast initialization without hang
        self._cap = cv2.VideoCapture(self.config.camera_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self.config.camera_id)

        if not self._cap.isOpened():
            print(f"[Camera] Error: Could not open camera {self.config.camera_id}")
            self.is_opened = False
            return False

        # Request resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        # Force internal camera driver buffer size to 1 to eliminate driver buffering lag
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.is_opened = True
        self._stop_event.clear()

        # Start dedicated thread to continuously drain driver frames
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

        # Wait briefly for first frame to arrive
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.02)

        return True

    def _capture_worker(self) -> None:
        """Continuously pulls frames from OpenCV at hardware speed to prevent buffer queuing."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Apply horizontal mirroring immediately
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = now
                self._has_new_frame = True

                self._fps_frame_count += 1
                if now - self._fps_timer >= 1.0:
                    self.camera_fps = self._fps_frame_count / (now - self._fps_timer)
                    self._fps_frame_count = 0
                    self._fps_timer = now

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the newest available frame without waiting for driver buffers."""
        if not self.is_opened:
            return False, None

        with self._lock:
            if not self._has_new_frame or self._latest_frame is None:
                return False, None
            self._has_new_frame = False
            return True, self._latest_frame.copy()

    def get_latest_frame_with_timestamp(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns the newest frame and its acquisition timestamp for latency tracking."""
        if not self.is_opened:
            return False, None, 0.0

        with self._lock:
            if not self._has_new_frame or self._latest_frame is None:
                return False, None, 0.0
            frame = self._latest_frame.copy()
            ts = self._latest_timestamp
            self._has_new_frame = False
            return True, frame, ts

    def release(self) -> None:
        """Stops background thread and releases the camera device."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.is_opened = False
        self._latest_frame = None

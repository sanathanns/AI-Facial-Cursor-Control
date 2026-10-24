"""Face tracking module using the MediaPipe Tasks Face Landmarker API."""
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canonical landmark indices in MediaPipe Face Mesh
# Left Eye (subject perspective / standard canonical mesh)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Right Eye (subject perspective / standard canonical mesh)
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Nose anchor points
NOSE_TIP_IDX = 1
MID_EYES_IDX = 168
CHIN_IDX = 152

# Optional Mouth landmarks (Upper, Lower, Left corner, Right corner)
MOUTH_LANDMARKS = [13, 14, 78, 308]


@dataclass
class FaceLandmarksResult:
    face_detected: bool
    timestamp_ms: int
    raw_landmarks: Optional[List[Tuple[float, float, float]]] = None
    nose_tip: Optional[Tuple[float, float]] = None
    mid_eyes: Optional[Tuple[float, float]] = None
    chin: Optional[Tuple[float, float]] = None
    left_eye: Optional[List[Tuple[float, float]]] = None
    right_eye: Optional[List[Tuple[float, float]]] = None
    mouth: Optional[List[Tuple[float, float]]] = None


class FaceTracker:
    """Wrapper around MediaPipe Face Landmarker Tasks API."""

    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        # Ensure strictly monotonically increasing timestamps
        self._last_timestamp_ms = 0

    def _next_monotonic_timestamp(self) -> int:
        """Enforces strictly monotonically increasing timestamps for MediaPipe Tasks."""
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        else:
            ts = now_ms
        self._last_timestamp_ms = ts
        return ts

    def process_frame(self, bgr_frame: np.ndarray) -> FaceLandmarksResult:
        """Processes a BGR image frame and returns detected landmarks."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = self._next_monotonic_timestamp()
        detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return FaceLandmarksResult(
                face_detected=False,
                timestamp_ms=timestamp_ms,
            )

        face_landmarks = detection_result.face_landmarks[0]
        raw_list = [(lm.x, lm.y, lm.z) for lm in face_landmarks]

        nose_tip = (face_landmarks[NOSE_TIP_IDX].x, face_landmarks[NOSE_TIP_IDX].y)
        mid_eyes = (face_landmarks[MID_EYES_IDX].x, face_landmarks[MID_EYES_IDX].y)
        chin = (face_landmarks[CHIN_IDX].x, face_landmarks[CHIN_IDX].y)

        eye_a = [(face_landmarks[i].x, face_landmarks[i].y) for i in LEFT_EYE_LANDMARKS]
        eye_b = [(face_landmarks[i].x, face_landmarks[i].y) for i in RIGHT_EYE_LANDMARKS]
        mouth = [(face_landmarks[i].x, face_landmarks[i].y) for i in MOUTH_LANDMARKS]

        # In mirrored camera view, the user's left eye appears on the screen's left (smaller x)
        # and user's right eye appears on the screen's right (larger x).
        # We ensure screen-left eye is always mapped to left_eye and screen-right to right_eye.
        avg_xa = sum(p[0] for p in eye_a) / len(eye_a)
        avg_xb = sum(p[0] for p in eye_b) / len(eye_b)
        if avg_xa <= avg_xb:
            left_eye = eye_a
            right_eye = eye_b
        else:
            left_eye = eye_b
            right_eye = eye_a

        return FaceLandmarksResult(
            face_detected=True,
            timestamp_ms=timestamp_ms,
            raw_landmarks=raw_list,
            nose_tip=nose_tip,
            mid_eyes=mid_eyes,
            chin=chin,
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
        )

    def close(self) -> None:
        """Closes the underlying MediaPipe landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

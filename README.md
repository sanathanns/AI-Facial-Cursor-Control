# AI-Based Facial Cursor Control System

An assistive computer vision system that maps head pose and intentional facial gestures into smooth, responsive mouse movements and click actions in real-time.

Built with **MediaPipe Tasks Face Landmarker API**, **OpenCV**, **PyAutoGUI**, and **Tkinter**.

---

## Features

- **Head Pose Tracking**: Real-time 3D facial landmark detection tracking nose and head motion.
- **Neutral Calibration**: One-click / one-key calibration (`C` key) to align user's comfortable resting posture with the center of the screen.
- **Configurable One-Euro Filter**: Eliminates resting cursor jitter while providing instant, lag-free response to rapid head movements.
- **Configurable Dead Zone**: Prevents resting drift so the user can easily hold the cursor steady over buttons and UI elements.
- **Dual-Wink Click Detection**:
  - **Left Wink**: Triggers Left Click (with blink-duration validation).
  - **Right Wink**: Triggers Right Click.
  - **Bilateral Blink Rejection**: Ignores natural two-eye involuntary blinks to prevent unintended clicks.
- **Intentional Scrolling**: Clearly separated Scroll Mode (`S` key / button or prolonged dual-eye squeeze) freezes pointer movement and maps vertical head tilt directly to document scrolling, guaranteeing normal cursor movements cannot accidentally scroll.
- **Face-Loss Safety**: Instantly freezes the cursor and releases active mouse clicks if face tracking is lost.
- **Emergency Failsafe**: Instant emergency stop via `Escape` key, Emergency Stop GUI button, or moving the physical mouse to any screen corner (PyAutoGUI failsafe).
- **Main-Thread Tkinter GUI**:
  - Real-time video canvas with live face mesh and tracking HUD overlay.
  - Live EAR (Eye Aspect Ratio) gauges and action status.
  - Runtime tuning sliders for Sensitivity, Dead Zone, 1-Euro Cutoff, and Wink Thresholds.
  - Settings persistence to `config.json`.

---

## Quick Start

### 1. Requirements

- Python 3.10+ (tested and verified on Python 3.13.5)
- Webcam (built-in or USB)

### 2. Activate Virtual Environment & Install Dependencies

```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Ensure Model Bundle is Present

The system automatically downloads the official Google MediaPipe `face_landmarker.task` model into the `models/` folder on first launch, or you can run:

```powershell
python download_model.py
```

### 4. Launch Application

```powershell
python main.py
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **`C`** | Calibrate neutral head position |
| **`S`** | Toggle Scroll Mode (freezes cursor and scrolls via head tilt) |
| **`Space`** | Pause / Resume facial cursor tracking |
| **`Escape`** | Emergency Stop (halts all cursor actions immediately) |

---

## Automated Test Suite

Run the full pytest suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

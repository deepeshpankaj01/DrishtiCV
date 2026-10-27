# DrishtiCV — Real-Time Object Detection & Tracking

> **Tagline:** *See Objects. Track Movement. Understand the Scene.*  
> **Developer:** Deepesh Kumar Pankaj  
> **Category:** Computer Vision / Artificial Intelligence / Machine Learning  
> **Framework:** Python 3.11, PySide6 (Qt6), OpenCV, Ultralytics YOLO26  

---

## 1. Project Overview

**DrishtiCV** is a high-performance desktop computer-vision application engineered for real-time object detection, multi-object tracking, and spatial scene analytics. Developed by **Deepesh Kumar Pankaj**, the application processes computer webcams, recorded video files, and still images locally on the user's computer with zero cloud telemetry.

The application features a restrained technical dark UI designed in PySide6, providing real-time visual overlays, interactive virtual-line crossing counters, rectangular region-of-interest (ROI) monitoring, snapshot exports, annotated MP4 video recording, and structured CSV event logging.

---

## 2. Features Implemented

- **Pretrained YOLO Object Detection:** Powered by Ultralytics YOLO26 nano (`yolo26n.pt`) with real-time bounding box annotations, class labels, and confidence percentages.
- **Persistent Multi-Object Tracking:** Tracks objects across consecutive video frames using ByteTrack (default) or BoT-SORT, preserving persistent numerical IDs (`#ID`).
- **Dynamic Movement Trails:** Visualizes motion trajectories by rendering fading trail lines behind moving tracked objects.
- **Interactive Virtual-Line Counting:**
  - Define a directional counting line on the video feed with two mouse clicks.
  - Computes exact segment crossings using 2D cross-product signed determinant mathematics.
  - Tracks separate **Inbound (Side A → Side B)** and **Outbound (Side B → Side A)** cumulative totals.
  - Implements temporal debouncing to eliminate duplicate counts from stationary or hovering objects.
- **Monitored Region of Interest (ROI):**
  - Draw a rectangular zone over the video feed using click-and-drag.
  - Detects boundary transitions from outside to inside (`REGION_ENTRY`) and inside to outside (`REGION_EXIT`).
  - Guards against duplicate entry triggers while an object remains within the zone.
- **Three Supported Input Modes:**
  1. *Live Webcam:* Auto-discovers connected webcam devices (index 0, 1, 2...) with graceful disconnection handling.
  2. *Video File:* Plays local video files (MP4, AVI, MKV, MOV) with natural frame pacing, pause/resume, and scrubbing seek slider.
  3. *Single Image:* Inspects local image files (JPG, PNG, BMP) with instant object detection and class count breakdowns.
- **Non-Blocking Multithreaded Architecture:** Runs neural network inference, OpenCV video capture, and trajectory calculations in a background `QThread` worker, ensuring zero UI freezing.
- **Structured CSV Event Logging:**
  - Automatically isolates each run in a timestamped folder: `outputs/session_YYYYMMDD_HHMMSS/`.
  - Appends every meaningful event to `events.csv` with standard escaping.
  - Captures source starts/stops, line crossings, zone entries/exits, snapshots, and recordings.
- **Annotated Video Recording & Snapshots:**
  - Start and stop video recording on demand to high-quality MP4 using robust fallback codecs (`mp4v`, `avc1`, `XVID`).
  - Capture instantaneous full-resolution annotated JPEG snapshots.
- **Custom Model Compatibility:** Easily load custom-trained PyTorch (`.pt`) YOLO weights via GUI file picker with automatic class label extraction.
- **Class Filtering:** Filter detections to specific object classes (e.g., monitor only `person` and `car`).

---

## 3. Technology Stack

- **Language:** Python 3.11 (strictly recommended for binary wheel compatibility on Windows)
- **GUI Framework:** PySide6 6.12+ (Qt 6 for Python)
- **Computer Vision:** OpenCV 5.0+ (`opencv-python`)
- **Neural Network Inference:** Ultralytics 8.4+ (YOLO26, YOLO11, YOLOv8)
- **Deep Learning Runtime:** PyTorch 2.14+ (`torch`, `torchvision`)
- **Numerical Processing:** NumPy 2.4+
- **Automated Testing:** pytest 9.1+

---

## 4. Architecture & Data Flow

```
[ Input Source ] ──► [ VideoProcessingWorker (QThread) ]
(Webcam / Video / Image)              │
                                      ▼
                             [ YOLO26 Inference ]
                           (Bounding Boxes, Classes)
                                      │
                                      ▼
                            [ ByteTrack Tracker ]
                            (Persistent Track IDs)
                                      │
                                      ▼
                        [ Spatial Geometry Engine ]
                    (which_side, check_line_crossing,
                        point_in_rect, debouncing)
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   [ Scene Analytics ]       [ OpenCV Renderer ]      [ Output Pipeline ]
   - LineCrossTracker        - Bounding Boxes          - VideoRecorder
   - RegionMonitor           - Badges & Trails         - SessionManager
   - Cumulative Counts       - Line / Zone Overlays    - events.csv Logging
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                       [ Qt Signal-Slot Dispatch ]
                                      │
                                      ▼
                          [ PySide6 Main Window ]
                    (Video Canvas, Metrics, Event Log)
```

### Module Breakdown
- `app.py`: Application entrypoint; configures logging and launches PySide6 application.
- `src/drishticv/config.py`: Central configuration, styling constants, dark theme QSS, and default parameters.
- `src/drishticv/detection/detector.py`: Wraps Ultralytics YOLO models; handles detection, tracking, structured dataclasses (`DetectedObject`, `DetectionResult`), and OpenCV frame annotation.
- `src/drishticv/core/geometry.py`: Pure mathematical geometry routines for cross-products, line segment intersections, and rectangle boundary tests. No UI dependencies.
- `src/drishticv/core/events.py`: Higher-level stateful managers for virtual-line counting, region monitoring, and movement trail deques.
- `src/drishticv/core/recording.py`: OpenCV `VideoWriter` wrapper with codec fallbacks, dimension checks, and snapshot utilities.
- `src/drishticv/core/event_logger.py`: Thread-safe session folder generation and structured CSV event logger.
- `src/drishticv/ui/video_canvas.py`: Custom `QWidget` rendering letterboxed video frames, interactive two-click line definition, and drag-and-drop zone setup.
- `src/drishticv/ui/workers.py`: `QThread` worker managing OpenCV capture loops, frame pacing, inference execution, and Qt signal dispatching.
- `src/drishticv/ui/main_window.py`: Complete user interface assembling header, sidebar controls, canvas, playback controls, analytics cards, and status bar.

---

## 5. Installation & Setup Instructions

### Prerequisites
- Operating System: Windows 10/11 (64-bit), macOS, or Linux.
- Python: **Python 3.11** is recommended.

### Step 1: Clone or Navigate to the Workspace
```bash
cd C:\Users\deepe\Desktop\Fii\DRISHTICV
```

### Step 2: Create a Virtual Environment
```bash
# Windows
py -3.11 -m venv .venv

# Linux / macOS
python3.11 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. How to Run the Application

### Option A: Windows Batch Launcher (Fastest)
Double-click `run.bat` or execute in terminal:
```cmd
run.bat
```
`run.bat` automatically verifies the local `.venv`, activates it, and starts the application.

### Option B: Direct Python Command
With `.venv` active:
```bash
python app.py
```

---

## 7. Model Download & Offline Behavior

- **Initial Run:** On the first execution with the default `yolo26n.pt` model, Ultralytics downloads the lightweight 5.3 MB weights checkpoint automatically to `models/` or the cache. An active internet connection is required only for this initial download.
- **Subsequent Runs:** DrishtiCV operates **completely offline**.
- **Pre-downloading:** You can manually place `yolo26n.pt`, `yolo11n.pt`, or `yolov8n.pt` directly into `models/`.

---

## 8. Operating Guide

### A. Live Webcam Monitoring
1. In the left sidebar under **Input Source**, select the **Webcam** tab.
2. Select your camera device (e.g., `Webcam Device 0`). Click **Refresh** if your webcam was plugged in after starting the app.
3. Click **Start Webcam**.
4. Adjust the **Confidence** slider to tune detection sensitivity (default: 40%).

### B. Video File Playback & Seeking
1. Switch to the **Video File** tab.
2. Click **Select Video File...** and choose an MP4/AVI/MKV video.
3. Click **Play Video**.
4. Use the playback bar beneath the video canvas to **Pause**, **Resume**, **Stop**, or drag the **Seek Slider** to scrub across timestamps.

### C. Single Image Inspection
1. Switch to the **Image** tab.
2. Click **Select Image...** and pick a JPG/PNG file.
3. Click **Detect Objects**. Review detected bounding boxes and the right-hand **Detected Classes Breakdown** table.

### D. Setting up Virtual-Line Counting
1. While a video or camera is active, click **Define Line (2 Clicks)** in the sidebar.
2. Click on the video canvas to place the **Line Start Point (A)**.
3. Move the mouse and click to place the **Line End Point (B)**.
4. The canvas immediately overlays the line with **Side A** and **Side B** badges.
5. Tracked objects crossing from A to B increment **Inbound Crossings**; objects crossing from B to A increment **Outbound Crossings**.
6. Click **Clear** to remove the line.

### E. Setting up a Monitored Zone (ROI)
1. Click **Define Monitored Zone**.
2. Click and drag a rectangle over the region you wish to observe.
3. Release the mouse button. The zone will be highlighted with a semi-transparent teal overlay.
4. When any tracked object enters the boundary, a `REGION_ENTRY` event is recorded. When it leaves, a `REGION_EXIT` event is recorded.

### F. Capturing Snapshots & Recording Video
- **Snapshot:** Click **Save Snapshot**. The annotated frame is instantly saved to `outputs/session_YYYYMMDD_HHMMSS/snapshots/`.
- **Record Video:** Click **Start Recording Video**. The button turns red and shows the output filename. Click **Stop Recording** when finished.

---

## 9. Output Structure & Event Logging

For every session, DrishtiCV creates an isolated directory in `outputs/` named after the session start timestamp:

```
outputs/
└── session_20261027_010225/
    ├── events.csv          # Structured CSV event log
    ├── session.log         # Detailed application debug log
    ├── snapshots/          # Exported JPEG images
    │   └── snapshot_20261027_010530_123456.jpg
    └── recordings/         # Exported MP4 annotated videos
        └── recording_20261027_010540.mp4
```

### `events.csv` Schema
| Field | Description | Example |
|---|---|---|
| `timestamp` | High-precision ISO timestamp | `2026-10-27 01:05:32.418` |
| `event_type` | Category of event | `LINE_CROSSING`, `REGION_ENTRY`, `SNAPSHOT_SAVED` |
| `source_name` | Name of the active video stream | `Webcam #0`, `traffic.mp4` |
| `track_id` | Numerical persistent object tracking ID | `3` |
| `class_name` | Detected YOLO object class | `person`, `car`, `bicycle` |
| `confidence` | Model prediction confidence score | `0.875` |
| `direction` | Movement direction (for line crossings) | `Inbound (A→B)` |
| `coordinates` | Native video coordinates | `(320.5,240.0)` |
| `details` | Contextual metadata / file path | `Direction: A_TO_B` |

---

## 10. Automated Tests

Run the complete test suite with:

```bash
pytest -v
```

All 29 tests verify geometry math, line-crossing directional logic, duplicate event debouncing, region-entry/exit state machines, CSV writing, mock frame annotation, and VideoWriter lifecycle headlessly. Refer to [`docs/TESTING.md`](docs/TESTING.md) for full coverage details and the manual webcam checklist.

---

## 11. Troubleshooting

| Symptom | Probable Cause | Recommended Action |
|---|---|---|
| `Could not open webcam index #0` | Camera in use by another app, or permissions disabled. | Close Zoom/Teams/OBS; open Windows Settings -> Privacy & Security -> Camera and enable desktop app access. |
| Model download fails at startup | No internet connection on first run. | Connect to the internet briefly to download `yolo26n.pt`, or manually place the `.pt` file in `models/`. |
| Video playback runs too fast | Frame rate pacing omitted. | DrishtiCV automatically paces video playback using OpenCV source FPS. Ensure the file is not corrupted. |
| Recording fails to open VideoWriter | Missing MP4 media codecs. | DrishtiCV falls back across `mp4v`, `avc1`, and `XVID`. Ensure Windows Media Feature Pack or standard codecs are installed. |

---

## 12. Scope Limitations & Future Improvements

- **General-Purpose Model Scope:** The initial model is trained on the standard 80 COCO classes. It does not detect semiconductor wafer defects, PCB solder bridges, or microscopic electronic components without training a domain-specific model. Refer to [`docs/CUSTOM_MODEL_GUIDE.md`](docs/CUSTOM_MODEL_GUIDE.md) for custom model training instructions.
- **Future Enhancements:**
  - GPU acceleration acceleration auto-selector (CUDA / DirectML / TensorRT).
  - Multi-line polygonal counting zones.
  - Heatmap velocity accumulation overlays.
  - Automated report generation in PDF format.

---

## 13. Developer & License Notes

- **Application Title:** DrishtiCV — Real-Time Object Detection & Tracking
- **Developer Credit:** Deepesh Kumar Pankaj
- **License:** Educational / Engineering Use. Built using open-source libraries: PySide6 (LGPLv3), OpenCV (Apache 2.0), Ultralytics (AGPLv3), and PyTorch (BSD-3).
#   D r i s h t i C V  
 
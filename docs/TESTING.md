# DrishtiCV Testing Guide & Verification Checklist

**Application:** DrishtiCV — Real-Time Object Detection & Tracking  
**Developer:** Deepesh Kumar Pankaj  
**Tagline:** *See Objects. Track Movement. Understand the Scene.*

---

## 1. Automated Test Suite

DrishtiCV includes an automated unit test suite built with `pytest`. The tests run independently of physical webcam hardware or GPU availability by verifying mathematical, algorithmic, state-machine, and file logging logic directly.

### Running the Tests

Ensure your virtual environment is active:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

Execute the full test suite:

```bash
pytest -v
```

### Automated Test Coverage Summary

| Test File | Test Case | Purpose & Validation |
|---|---|---|
| `tests/test_geometry.py` | `test_box_centroid` | Validates accurate midpoint calculation for bounding boxes. |
| `tests/test_geometry.py` | `test_euclidean_distance` | Verifies point-to-point distance calculations. |
| `tests/test_geometry.py` | `test_which_side_horizontal_line` | Validates 2D cross-product signed area for horizontal lines. |
| `tests/test_geometry.py` | `test_which_side_vertical_line` | Validates signed determinant side classification for vertical lines. |
| `tests/test_geometry.py` | `test_which_side_deadzone` | Confirms deadzone margin prevents boundary jitter. |
| `tests/test_geometry.py` | `test_segments_intersect` | Tests segment intersection detection using counter-clockwise orientation. |
| `tests/test_geometry.py` | `test_check_line_crossing_valid_transitions` | Verifies `A_TO_B` and `B_TO_A` crossing detection. |
| `tests/test_geometry.py` | `test_check_line_crossing_no_intersection` | Verifies that movements crossing the infinite line beyond the segment are rejected. |
| `tests/test_geometry.py` | `test_check_line_crossing_same_side` | Confirms no event triggers if object stays on one side. |
| `tests/test_geometry.py` | `test_point_in_rect` | Tests rectangular containment and boundary normalization. |
| `tests/test_geometry.py` | `test_check_region_transition` | Verifies `ENTRY` and `EXIT` state machine transitions. |
| `tests/test_events.py` | `test_line_crossing_tracker_basic_crossing` | End-to-end line tracker incrementation and event emission. |
| `tests/test_events.py` | `test_line_crossing_debounce_prevents_duplicate_counts` | Confirms debounce window prevents multiple counts on hovering objects. |
| `tests/test_events.py` | `test_line_crossing_stationary_object` | Verifies zero false positive counts on stationary objects. |
| `tests/test_events.py` | `test_line_crossing_multiple_classes` | Verifies independent class-based cumulative counting. |
| `tests/test_events.py` | `test_line_crossing_reset_behavior` | Verifies state wiping vs full metric reset. |
| `tests/test_events.py` | `test_region_monitor_entry_and_exit` | Confirms exactly 1 entry event when entering, and zero duplicates while remaining inside. |
| `tests/test_events.py` | `test_region_monitor_reset` | Verifies clean reset of zone occupancy and cumulative counts. |
| `tests/test_events.py` | `test_trail_manager_lifecycle` | Confirms maximum trail length deque capping and eviction of lost track IDs. |
| `tests/test_event_logger.py` | `test_session_manager_directory_creation` | Verifies session folder generation and subdirectory hierarchy. |
| `tests/test_event_logger.py` | `test_event_logger_csv_header_and_row_consistency` | Confirms field alignment across CSV headers and data rows. |
| `tests/test_event_logger.py` | `test_event_logger_special_characters_and_commas` | Verifies standard CSV escaping for details containing quotes/commas. |
| `tests/test_event_logger.py` | `test_event_logger_missing_track_id` | Verifies graceful handling of untracked objects (`track_id=None`). |
| `tests/test_event_logger.py` | `test_event_logger_recent_events_memory` | Tests bounded FIFO memory for GUI event display. |
| `tests/test_event_logger.py` | `test_event_logger_line_and_region_helpers` | Verifies typed conversion of domain events into CSV logs. |
| `tests/test_detector_mock.py` | `test_detection_result_properties` | Verifies aggregations (count, tracked count, class distribution). |
| `tests/test_detector_mock.py` | `test_annotate_frame_renders_cleanly` | Verifies bounding box, badge, and trail rendering on frame. |
| `tests/test_recording.py` | `test_save_snapshot` | Verifies JPEG snapshot writing to disk. |
| `tests/test_recording.py` | `test_video_recording_lifecycle` | Tests VideoWriter startup, multi-frame writing, flush, and release. |

---

## 2. Manual Verification Checklist (Physical Webcam & Video)

When running on a local desktop with an attached camera and sample video files, perform the following manual test sequence:

### Phase A: Application Startup & Model Verification
- [ ] Launch `run.bat` or `python app.py`.
- [ ] Confirm window opens with dark technical theme at minimum 1120 × 720 resolution.
- [ ] Confirm header displays **DrishtiCV**, subtitle, and developer credit (**Deepesh Kumar Pankaj**).
- [ ] Confirm initial status pill transitions from `LOADING MODEL` to `READY` once `yolo26n.pt` is initialized.
- [ ] Confirm status bar displays active model (`Model: yolo26n.pt`) and active output session folder.

### Phase B: Webcam Input Mode
- [ ] On the **Webcam** tab, click **Refresh** and verify detected camera indices (e.g., `Webcam Device 0`).
- [ ] Click **Start Webcam**.
- [ ] Verify live camera feed renders smoothly in the center canvas.
- [ ] Verify status pill updates to green `STREAMING`.
- [ ] Hold an object (e.g., a cup, cell phone, or person) in front of the camera:
  - Bounding box appears around the object with high-contrast badge.
  - Class label, confidence percentage (e.g., `person 88%`), and track ID (e.g., `#1`) appear above box.
- [ ] Move the object across the frame:
  - Confirm persistent track ID does not rapidly flicker or reassign.
  - Confirm fading movement trails follow the object centroid.
- [ ] Click **Stop Webcam**. Confirm camera stream stops cleanly and canvas returns to idle reticle.

### Phase C: Virtual Line Counting
- [ ] Start video or webcam stream.
- [ ] Click **Define Line (2 Clicks)**.
- [ ] Click point 1 on the canvas to place start, then click point 2 to complete line.
- [ ] Verify amber virtual line appears with **Side A** and **Side B** badges and endpoint circles.
- [ ] Pass an object across the line from Side A to Side B:
  - Verify **Inbound Crossings (A→B)** counter increments by 1.
  - Verify an event row appears in the **Recent Scene Events** table.
- [ ] Pass the object back from Side B to Side A:
  - Verify **Outbound Crossings (B→A)** counter increments by 1.
- [ ] Leave the object stationary near the line:
  - Confirm the counter does not repeatedly increment.
- [ ] Click **Clear** next to line controls: confirm line disappears and state resets.

### Phase D: Monitored Zone (ROI)
- [ ] Click **Define Monitored Zone**.
- [ ] Click and drag a rectangle over a section of the video feed.
- [ ] Verify the zone displays with a subtle cyan tint, dashed/solid border, and "Monitored Zone" badge.
- [ ] Move an object from outside the rectangle into the rectangle:
  - Verify **Monitored Zone Entries** counter increments by 1.
  - Verify `REGION_ENTRY` event is logged in the events table.
- [ ] Keep the object inside the rectangle:
  - Confirm no additional entry events are fired while remaining inside.
- [ ] Move the object outside the rectangle:
  - Verify `REGION_EXIT` is logged.

### Phase E: Video File Playback & Seeking
- [ ] Switch to **Video File** tab. Click **Select Video File...** and pick an MP4 video.
- [ ] Click **Play Video**.
- [ ] Confirm playback runs at natural source FPS without running excessively fast.
- [ ] Click **Pause** and verify playback halts; click **Resume** to continue.
- [ ] Drag the seek slider to scrub forward and backward through the video.
- [ ] Allow video to reach end-of-file: verify stream closes cleanly without crash.

### Phase F: Single Image Mode
- [ ] Switch to **Image** tab. Select a JPG/PNG image.
- [ ] Click **Detect Objects**.
- [ ] Confirm single frame detection results and class breakdown table populate accurately.

### Phase G: Snapshots, Video Recording & CSV Log Verification
- [ ] During an active stream, click **Save Snapshot**.
  - Notification appears indicating snapshot saved.
- [ ] Click **Start Recording Video**.
  - Status updates to `Recording: recording_YYYYMMDD_HHMMSS.mp4`.
- [ ] Let 5–10 seconds of video record, then click **Stop Recording**.
- [ ] Click **Open Session Folder** in the header.
  - Confirm Windows Explorer opens the active session directory:
    - `snapshots/` contains the saved image.
    - `recordings/` contains a playable MP4 video of the annotated feed.
    - `events.csv` contains structured rows with timestamp, event type, object ID, and class name.
    - `session.log` contains detailed timestamped technical logs.

### Phase H: Application Shutdown
- [ ] Close the application using the window close button (`X`) during active video streaming and recording.
- [ ] Verify application exits cleanly, all video writer handles are flushed, and no zombie Python processes remain.

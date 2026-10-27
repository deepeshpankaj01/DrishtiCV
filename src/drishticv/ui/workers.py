"""Background worker threads for video capture, YOLO inference, and scene analytics.

Decouples camera I/O and neural network computation from the PySide6 UI event loop,
ensuring zero GUI stutter and safe signal-slot message passing.
"""

from __future__ import annotations
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from drishticv.detection.detector import DrishtiDetector, DetectionResult
from drishticv.core.events import (
    LineCrossingTracker,
    RegionMonitor,
    TrailManager,
    TrackItem,
    LineCrossingEvent,
    RegionMonitoringEvent,
)
from drishticv.core.recording import VideoRecorder

logger = logging.getLogger(__name__)


def list_available_cameras(max_to_test: int = 5) -> List[int]:
    """Scan and return indices of active webcam devices."""
    available = []
    for index in range(max_to_test):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(index)
            cap.release()
        else:
            # Fallback test with default backend
            cap_default = cv2.VideoCapture(index)
            if cap_default.isOpened():
                ret, _ = cap_default.read()
                if ret:
                    available.append(index)
                cap_default.release()
    return available


class VideoProcessingWorker(QThread):
    """Worker thread that executes video capture, YOLO inference, tracking, and scene logic."""

    # Qt Signals
    frame_ready = Signal(object, object, dict)  # (annotated_frame, DetectionResult, metrics)
    metrics_updated = Signal(dict)
    crossing_event_occurred = Signal(object)     # LineCrossingEvent
    region_event_occurred = Signal(object)       # RegionMonitoringEvent
    source_started = Signal(str, int, int, float) # (name, width, height, fps)
    source_finished = Signal()
    source_error = Signal(str, str)              # (error_message, recovery_hint)
    status_changed = Signal(str)

    def __init__(
        self,
        detector: DrishtiDetector,
        line_tracker: LineCrossingTracker,
        region_monitor: RegionMonitor,
        recorder: VideoRecorder,
    ):
        super().__init__()
        self.detector = detector
        self.line_tracker = line_tracker
        self.region_monitor = region_monitor
        self.recorder = recorder
        self.trail_manager = TrailManager()

        # Thread Control
        self._running: bool = False
        self._paused: bool = False
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()

        # Source Configuration
        self.source_mode: str = "IDLE"  # "WEBCAM", "VIDEO", "IMAGE"
        self.source_target: Any = None
        self.source_name: str = ""

        # Runtime Parameters (safely updated from GUI)
        self.conf_threshold: float = 0.40
        self.allowed_classes: Optional[List[int]] = None
        self.tracking_enabled: bool = True
        self.tracker_type: str = "bytetrack.yaml"
        self.draw_trails: bool = True
        self.draw_labels: bool = True
        self.draw_conf: bool = True
        self.draw_tracks: bool = True

        # Video File Playback Seeking
        self._seek_requested: bool = False
        self._seek_target_frame: int = 0

        # Performance Tracking
        self._fps_window: deque[float] = deque(maxlen=20)
        self._last_frame_timestamp: float = time.perf_counter()

    def set_source_webcam(self, camera_index: int) -> None:
        """Configure worker to stream from a webcam device."""
        self._mutex.lock()
        self.source_mode = "WEBCAM"
        self.source_target = camera_index
        self.source_name = f"Webcam Device #{camera_index}"
        self._mutex.unlock()

    def set_source_video(self, video_path: str) -> None:
        """Configure worker to play back a local video file."""
        self._mutex.lock()
        self.source_mode = "VIDEO"
        self.source_target = video_path
        self.source_name = f"Video: {video_path}"
        self._mutex.unlock()

    def set_source_image(self, image_path: str) -> None:
        """Configure worker to inspect a single image file."""
        self._mutex.lock()
        self.source_mode = "IMAGE"
        self.source_target = image_path
        self.source_name = f"Image: {image_path}"
        self._mutex.unlock()

    def update_settings(
        self,
        conf_threshold: float,
        allowed_classes: Optional[List[int]],
        tracking_enabled: bool,
        tracker_type: str,
        draw_trails: bool,
        draw_labels: bool = True,
        draw_conf: bool = True,
        draw_tracks: bool = True,
    ) -> None:
        """Thread-safely update detection and rendering settings."""
        self._mutex.lock()
        self.conf_threshold = conf_threshold
        self.allowed_classes = allowed_classes
        self.tracking_enabled = tracking_enabled
        self.tracker_type = tracker_type
        self.draw_trails = draw_trails
        self.draw_labels = draw_labels
        self.draw_conf = draw_conf
        self.draw_tracks = draw_tracks
        self._mutex.unlock()

    def request_seek(self, frame_index: int) -> None:
        """Request seek to specific frame in video file."""
        self._mutex.lock()
        self._seek_requested = True
        self._seek_target_frame = frame_index
        self._mutex.unlock()

    def pause(self) -> None:
        """Pause video playback."""
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self) -> None:
        """Resume video playback."""
        self._mutex.lock()
        self._paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def is_paused(self) -> bool:
        self._mutex.lock()
        val = self._paused
        self._mutex.unlock()
        return val

    def stop(self) -> None:
        """Stop worker execution and join thread safely."""
        self._mutex.lock()
        self._running = False
        self._paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.wait(3000)

    def run(self) -> None:
        """Main worker loop."""
        self._running = True
        self._paused = False
        self._fps_window.clear()
        self.trail_manager.clear()
        self.detector.reset_tracker()

        mode = self.source_mode
        target = self.source_target

        try:
            if mode == "WEBCAM":
                self._run_webcam_stream(target)
            elif mode == "VIDEO":
                self._run_video_stream(target)
            elif mode == "IMAGE":
                self._run_single_image(target)
        except Exception as e:
            logger.error(f"Worker runtime exception: {e}", exc_info=True)
            self.source_error.emit(
                f"Unexpected runtime error: {str(e)}",
                "Restart the input source or verify the model configuration.",
            )
        finally:
            self._running = False
            self.status_changed.emit("IDLE")

    def _run_single_image(self, image_path: str) -> None:
        """Process a single still image."""
        self.status_changed.emit("PROCESSING")
        img = cv2.imread(image_path)
        if img is None:
            self.source_error.emit(
                f"Failed to read image at: {image_path}",
                "Verify that the file is a valid PNG, JPG, or BMP image.",
            )
            return

        h, w = img.shape[:2]
        self.source_started.emit(self.source_name, w, h, 0.0)

        # Run detection
        conf = self.conf_threshold
        classes = self.allowed_classes
        res = self.detector.detect(img, conf_threshold=conf, allowed_classes=classes)

        annotated = self.detector.annotate_frame(
            img,
            res,
            trails=None,
            draw_trails=False,
            show_labels=self.draw_labels,
            show_conf=self.draw_conf,
            show_tracks=False,
        )

        metrics = {
            "fps": 0.0,
            "inference_ms": res.inference_time_ms,
            "frame_index": 1,
            "total_frames": 1,
            "object_count": res.count,
            "tracked_count": 0,
        }

        self.frame_ready.emit(annotated, res, metrics)
        self.status_changed.emit("IMAGE_READY")

    def _run_webcam_stream(self, cam_index: int) -> None:
        """Stream and process frames from local webcam."""
        self.status_changed.emit("OPENING_CAMERA")
        # Try DSHOW first for Windows speed, then default
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            self.source_error.emit(
                f"Could not open webcam index #{cam_index}.",
                "Check that the webcam is securely plugged in, not in use by another program, "
                "and that Windows camera permissions are granted.",
            )
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_started.emit(self.source_name, w, h, 30.0)
        self.status_changed.emit("STREAMING")

        frame_count = 0

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame received from webcam. Stream may have disconnected.")
                self.msleep(10)
                continue

            frame_count += 1
            self._process_and_dispatch_frame(frame, frame_count, total_frames=0, target_fps=30.0)

        cap.release()
        logger.info(f"Released webcam device #{cam_index}.")

    def _run_video_stream(self, video_path: str) -> None:
        """Stream, pace, and seek frames from a video file."""
        self.status_changed.emit("OPENING_VIDEO")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.source_error.emit(
                f"Could not open video file: {video_path}",
                "Ensure the video exists and is encoded in a supported format (MP4, AVI, MKV, MOV).",
            )
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if src_fps <= 0 or src_fps > 120:
            src_fps = 30.0

        target_frame_time = 1.0 / src_fps
        self.source_started.emit(self.source_name, w, h, src_fps)
        self.status_changed.emit("STREAMING")

        frame_idx = 0

        while self._running:
            # Handle seeking
            if self._seek_requested:
                self._mutex.lock()
                target_frame = max(0, min(total_frames - 1, self._seek_target_frame))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_idx = target_frame
                self._seek_requested = False
                self._mutex.unlock()

            # Handle pause
            self._mutex.lock()
            if self._paused:
                self._pause_cond.wait(self._mutex, 100)
                self._mutex.unlock()
                continue
            self._mutex.unlock()

            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                # End of file
                logger.info("Video playback reached end of stream.")
                self.source_finished.emit()
                break

            frame_idx += 1
            self._process_and_dispatch_frame(frame, frame_idx, total_frames, target_fps=src_fps)

            # Frame pacing to playback at real video speed
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0.002:
                self.msleep(int(sleep_time * 1000))

        cap.release()
        logger.info(f"Released video file: {video_path}.")

    def _process_and_dispatch_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        target_fps: float,
    ) -> None:
        """Run YOLO inference/tracking, evaluate scene events, annotate, and dispatch."""
        conf = self.conf_threshold
        classes = self.allowed_classes
        tracking_on = self.tracking_enabled
        tracker = self.tracker_type

        # 1. Inference / Tracking
        if tracking_on:
            res = self.detector.track(
                frame,
                conf_threshold=conf,
                allowed_classes=classes,
                tracker=tracker,
                persist=True,
            )
        else:
            res = self.detector.detect(
                frame,
                conf_threshold=conf,
                allowed_classes=classes,
            )

        # 2. Extract active tracks for scene analytics
        track_items: List[TrackItem] = []
        for obj in res.objects:
            if obj.track_id is not None:
                track_items.append(
                    TrackItem(
                        track_id=obj.track_id,
                        class_id=obj.class_id,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        box=obj.box,
                    )
                )

        # 3. Evaluate Virtual Line Crossings
        if self.line_tracker.is_configured() and track_items:
            crossing_events = self.line_tracker.update(track_items)
            for evt in crossing_events:
                self.crossing_event_occurred.emit(evt)

        # 4. Evaluate Monitored Region Transitions
        if self.region_monitor.is_configured() and track_items:
            region_events = self.region_monitor.update(track_items)
            for revt in region_events:
                self.region_event_occurred.emit(revt)

        # 5. Update Movement Trails
        if self.draw_trails and track_items:
            self.trail_manager.update(track_items)
        elif not self.draw_trails:
            self.trail_manager.clear()

        # 6. Render Annotations
        annotated = self.detector.annotate_frame(
            frame,
            res,
            trails=self.trail_manager.trails if self.draw_trails else None,
            draw_trails=self.draw_trails,
            show_labels=self.draw_labels,
            show_conf=self.draw_conf,
            show_tracks=self.draw_tracks,
        )

        # 7. Record if recorder is active
        if self.recorder.is_recording:
            self.recorder.write_frame(annotated)

        # 8. Compute rolling FPS
        now = time.perf_counter()
        delta = now - self._last_frame_timestamp
        self._last_frame_timestamp = now
        if delta > 0:
            self._fps_window.append(1.0 / delta)

        rolling_fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0.0

        metrics = {
            "fps": rolling_fps,
            "inference_ms": res.inference_time_ms,
            "frame_index": frame_idx,
            "total_frames": total_frames,
            "object_count": res.count,
            "tracked_count": res.tracked_count,
        }

        self.frame_ready.emit(annotated, res, metrics)

"""YOLO detection and multi-object tracking engine for DrishtiCV.

Wraps Ultralytics YOLO26 / YOLO models, provides consistent structured outputs,
supports ByteTrack and BoT-SORT, and renders clean bounding-box annotations.
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import cv2
import numpy as np

from drishticv.config import (
    DEFAULT_MODEL_NAME,
    MODELS_DIR,
    DEFAULT_TRACKER,
    COLOR_ACCENT,
    CV_COLOR_ACCENT,
    CV_COLOR_TRAIL,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Represents a single detected or tracked object in a frame."""
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    centroid: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class DetectionResult:
    """Aggregated detection and tracking outputs for a single frame."""
    objects: List[DetectedObject]
    inference_time_ms: float
    frame_width: int
    frame_height: int

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def tracked_count(self) -> int:
        return sum(1 for obj in self.objects if obj.track_id is not None)

    @property
    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self.objects:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts


class DrishtiDetector:
    """Loads and executes YOLO models for object detection and tracking."""

    # Curated palette of distinguishable, professional bounding-box colors (BGR)
    COLOR_PALETTE: List[Tuple[int, int, int]] = [
        (178, 214, 55),   # Muted Teal / Green (#37D6B2)
        (255, 178, 55),   # Soft Sky Blue
        (85, 153, 255),   # Warm Orange
        (180, 105, 255),  # Violet Rose
        (75, 215, 238),   # Amber Yellow
        (205, 220, 57),   # Lime Green
        (225, 130, 100),  # Steel Blue
        (120, 200, 150),  # Jade
    ]

    def __init__(self, model_identifier: str = DEFAULT_MODEL_NAME):
        self.model_identifier: str = model_identifier
        self.model: Any = None
        self.class_names: Dict[int, str] = {}
        self.is_loaded: bool = False
        self.active_tracker: str = DEFAULT_TRACKER

    def resolve_model_path(self, identifier: str) -> str:
        """Resolve model path from local models directory or direct file path."""
        p = Path(identifier)
        if p.is_file():
            return str(p.resolve())

        # Check in models directory
        candidate = MODELS_DIR / identifier
        if candidate.is_file():
            return str(candidate.resolve())

        # Check in workspace root
        root_candidate = Path(identifier)
        if root_candidate.is_file():
            return str(root_candidate.resolve())

        # Return original identifier (Ultralytics auto-download by name)
        return identifier

    def load_model(self, model_identifier: Optional[str] = None) -> Tuple[bool, str]:
        """Load the YOLO model from local file or official checkpoint.

        Returns (success: bool, message: str).
        """
        if model_identifier is not None:
            self.model_identifier = model_identifier

        resolved_target = self.resolve_model_path(self.model_identifier)
        logger.info(f"Loading YOLO model from: {resolved_target}")

        try:
            from ultralytics import YOLO

            # Load model
            self.model = YOLO(resolved_target)
            self.class_names = dict(self.model.names) if hasattr(self.model, "names") else {}
            self.is_loaded = True

            class_count = len(self.class_names)
            msg = f"Model loaded successfully: {Path(resolved_target).name} ({class_count} classes)"
            logger.info(msg)
            return True, msg

        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.class_names = {}
            error_msg = (
                f"Failed to load model '{self.model_identifier}'.\n"
                f"Reason: {str(e)}\n\n"
                f"Recovery tips:\n"
                f"1. Ensure you have an active internet connection if downloading a pretrained model for the first time.\n"
                f"2. Verify the model file exists and is a valid PyTorch (.pt) checkpoint.\n"
                f"3. For custom models, verify compatibility in docs/CUSTOM_MODEL_GUIDE.md."
            )
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def reset_tracker(self) -> None:
        """Reset internal tracker state when changing input sources."""
        if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []
            logger.debug("Tracker internal predictor state reset.")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.40,
        allowed_classes: Optional[List[int]] = None,
    ) -> DetectionResult:
        """Run single-frame object detection without tracking."""
        if not self.is_loaded or self.model is None:
            return DetectionResult([], 0.0, frame.shape[1], frame.shape[0])

        start_time = time.perf_counter()
        results = self.model(
            frame,
            conf=conf_threshold,
            classes=allowed_classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_ultralytics_results(results, elapsed_ms, frame.shape)

    def track(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.40,
        allowed_classes: Optional[List[int]] = None,
        tracker: str = DEFAULT_TRACKER,
        persist: bool = True,
    ) -> DetectionResult:
        """Run object detection and multi-object tracking across video frames."""
        if not self.is_loaded or self.model is None:
            return DetectionResult([], 0.0, frame.shape[1], frame.shape[0])

        self.active_tracker = tracker
        start_time = time.perf_counter()

        try:
            results = self.model.track(
                frame,
                conf=conf_threshold,
                classes=allowed_classes,
                tracker=tracker,
                persist=persist,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Tracker error occurred: {e}, falling back to plain detection.")
            return self.detect(frame, conf_threshold, allowed_classes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_ultralytics_results(results, elapsed_ms, frame.shape)

    def _parse_ultralytics_results(
        self,
        results: Any,
        elapsed_ms: float,
        frame_shape: Tuple[int, ...],
    ) -> DetectionResult:
        """Parse raw Ultralytics Results object into structured DetectionResult."""
        h, w = frame_shape[:2]
        detected_objects: List[DetectedObject] = []

        if not results:
            return DetectionResult([], elapsed_ms, w, h)

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult([], elapsed_ms, w, h)

        # Extract coordinates, confidences, classes, and optional track IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if boxes.id is not None:
            try:
                track_ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                track_ids = None

        for i in range(len(xyxy)):
            box = (float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3]))
            cls_id = int(classes[i])
            cls_name = self.class_names.get(cls_id, f"Class {cls_id}")
            conf = float(confs[i])
            tid = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else None

            detected_objects.append(
                DetectedObject(
                    box=box,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    track_id=tid,
                )
            )

        return DetectionResult(
            objects=detected_objects,
            inference_time_ms=elapsed_ms,
            frame_width=w,
            frame_height=h,
        )

    def get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Return a deterministic BGR color for an object class."""
        return self.COLOR_PALETTE[class_id % len(self.COLOR_PALETTE)]

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        trails: Optional[Dict[int, deque]] = None,
        draw_trails: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_tracks: bool = True,
    ) -> np.ndarray:
        """Render polished bounding boxes, text badges, and movement trails onto frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw trails if enabled
        if draw_trails and trails:
            for tid, pts in trails.items():
                if len(pts) < 2:
                    continue
                pts_list = list(pts)
                n = len(pts_list)
                for j in range(1, n):
                    alpha = float(j) / float(n)
                    thickness = max(1, int(round(alpha * 2.5)))
                    p_start = (int(pts_list[j - 1][0]), int(pts_list[j - 1][1]))
                    p_end = (int(pts_list[j][0]), int(pts_list[j][1]))
                    # Faint fading trail
                    cv2.line(annotated, p_start, p_end, CV_COLOR_TRAIL, thickness, cv2.LINE_AA)

        # 2. Draw bounding boxes and text badges
        for obj in result.objects:
            x1, y1, x2, y2 = map(int, obj.box)
            # Clip coordinates to frame boundary
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)

            color = self.get_color_for_class(obj.class_id)

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw corner accent marks for technical look
            corner_len = min(15, max(5, int((x2 - x1) * 0.2)))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2, cv2.LINE_AA)

            # Draw centroid point
            cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
            cv2.circle(annotated, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Construct label text
            parts = []
            if show_tracks and obj.track_id is not None:
                parts.append(f"#{obj.track_id}")
            if show_labels:
                parts.append(obj.class_name)
            if show_conf:
                parts.append(f"{int(round(obj.confidence * 100))}%")

            if not parts:
                continue

            label_text = " ".join(parts)

            # Measure text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Position badge above box, or inside if near frame top
            pad = 4
            badge_h = text_h + baseline + pad * 2
            badge_w = text_w + pad * 2

            badge_y1 = y1 - badge_h if y1 - badge_h >= 0 else y1
            badge_y2 = badge_y1 + badge_h
            badge_x1 = x1
            badge_x2 = min(w - 1, badge_x1 + badge_w)

            # Dark solid background pill with colored left bar
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 24, 32), -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x1 + 3, badge_y2), color, -1)

            text_pos = (badge_x1 + pad + 3, badge_y1 + text_h + pad)
            cv2.putText(annotated, label_text, text_pos, font, font_scale, (240, 244, 248), thickness, cv2.LINE_AA)

        return annotated

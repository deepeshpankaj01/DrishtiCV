"""Session management, structured CSV event logging, and application logging.

Creates isolated, timestamped session output directories:
    outputs/session_YYYYMMDD_HHMMSS/
        events.csv
        session.log
        snapshots/
        recordings/
"""

from __future__ import annotations
import os
import csv
import sys
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from drishticv.config import OUTPUTS_DIR
from drishticv.core.events import LineCrossingEvent, RegionMonitoringEvent

logger = logging.getLogger(__name__)

CSV_HEADERS: List[str] = [
    "timestamp",
    "event_type",
    "source_name",
    "track_id",
    "class_name",
    "confidence",
    "direction",
    "coordinates",
    "details",
]


class SessionManager:
    """Manages unique timestamped session directories and application file logs."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else OUTPUTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / session_id
        self.snapshots_dir = self.session_dir / "snapshots"
        self.recordings_dir = self.session_dir / "recordings"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.events_csv_path = self.session_dir / "events.csv"
        self.session_log_path = self.session_dir / "session.log"

        self._configure_file_logging()

    def _configure_file_logging(self) -> None:
        """Attach a rotating or standard file handler to the root logger."""
        file_handler = logging.FileHandler(str(self.session_log_path), encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info(f"Initialized DrishtiCV session directory at: {self.session_dir}")

    def generate_snapshot_path(self, prefix: str = "snapshot") -> Path:
        """Generate a unique timestamped image path in the snapshots directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        return self.snapshots_dir / f"{prefix}_{ts}.jpg"

    def generate_recording_path(self, prefix: str = "recording", ext: str = ".mp4") -> Path:
        """Generate a unique timestamped video path in the recordings directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.recordings_dir / f"{prefix}_{ts}{ext}"

    def open_session_directory_in_explorer(self) -> bool:
        """Open the active session directory in the native OS file explorer."""
        try:
            path_str = str(self.session_dir.resolve())
            if sys.platform == "win32":
                os.startfile(path_str)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", path_str])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path_str])
            return True
        except Exception as e:
            logger.error(f"Failed to open session folder: {e}")
            return False


class EventLogger:
    """Thread-safe CSV event logger for scene analytics and system lifecycle events."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.csv_path = session_manager.events_csv_path
        self._lock = threading.Lock()
        self.recent_events: List[Dict[str, Any]] = []
        self.max_recent: int = 100

        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Create the CSV file and write header row if not existing."""
        with self._lock:
            if not self.csv_path.is_file():
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)

    def log_event(
        self,
        event_type: str,
        source_name: str = "",
        track_id: Optional[int] = None,
        class_name: str = "",
        confidence: float = 0.0,
        direction: str = "",
        coordinates: str = "",
        details: str = "",
    ) -> Dict[str, Any]:
        """Record a structured event row into CSV and recent memory."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
        tid_str = str(track_id) if track_id is not None else ""
        conf_str = f"{confidence:.3f}" if confidence > 0.0 else ""

        row = [
            now_str,
            event_type,
            source_name,
            tid_str,
            class_name,
            conf_str,
            direction,
            coordinates,
            details,
        ]

        with self._lock:
            try:
                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                logger.error(f"Error writing to events CSV: {e}")

        event_dict: Dict[str, Any] = {
            "timestamp": now_str,
            "event_type": event_type,
            "source_name": source_name,
            "track_id": track_id,
            "class_name": class_name,
            "confidence": confidence,
            "direction": direction,
            "coordinates": coordinates,
            "details": details,
        }

        with self._lock:
            self.recent_events.append(event_dict)
            if len(self.recent_events) > self.max_recent:
                self.recent_events.pop(0)

        logger.info(f"Event: [{event_type}] {class_name} {tid_str} {direction} {details}".strip())
        return event_dict

    def log_source_started(self, source_name: str, resolution: str = "") -> Dict[str, Any]:
        return self.log_event(
            event_type="SOURCE_STARTED",
            source_name=source_name,
            details=f"Resolution: {resolution}" if resolution else "Started",
        )

    def log_source_stopped(self, source_name: str, total_frames: int = 0) -> Dict[str, Any]:
        return self.log_event(
            event_type="SOURCE_STOPPED",
            source_name=source_name,
            details=f"Frames processed: {total_frames}",
        )

    def log_line_crossing(self, event: LineCrossingEvent, source_name: str) -> Dict[str, Any]:
        coords_str = f"({event.crossing_point[0]:.1f},{event.crossing_point[1]:.1f})"
        return self.log_event(
            event_type="LINE_CROSSING",
            source_name=source_name,
            track_id=event.track_id,
            class_name=event.class_name,
            confidence=event.confidence,
            direction=event.human_direction,
            coordinates=coords_str,
            details=f"Direction: {event.direction}",
        )

    def log_region_event(self, event: RegionMonitoringEvent, source_name: str) -> Dict[str, Any]:
        coords_str = f"({event.centroid[0]:.1f},{event.centroid[1]:.1f})"
        return self.log_event(
            event_type=event.event_type,
            source_name=source_name,
            track_id=event.track_id,
            class_name=event.class_name,
            confidence=event.confidence,
            coordinates=coords_str,
            details=f"Region: {event.region_box}",
        )

    def log_recording_started(self, video_path: str, source_name: str) -> Dict[str, Any]:
        return self.log_event(
            event_type="RECORDING_STARTED",
            source_name=source_name,
            details=f"File: {Path(video_path).name}",
        )

    def log_recording_stopped(self, video_path: str, duration_sec: float, frames: int) -> Dict[str, Any]:
        return self.log_event(
            event_type="RECORDING_STOPPED",
            details=f"File: {Path(video_path).name}, Duration: {duration_sec:.1f}s, Frames: {frames}",
        )

    def log_snapshot_saved(self, snapshot_path: str, source_name: str) -> Dict[str, Any]:
        return self.log_event(
            event_type="SNAPSHOT_SAVED",
            source_name=source_name,
            details=f"File: {Path(snapshot_path).name}",
        )

    def log_error(self, error_type: str, message: str, source_name: str = "") -> Dict[str, Any]:
        return self.log_event(
            event_type=f"ERROR_{error_type.upper()}",
            source_name=source_name,
            details=message,
        )

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return a copy of the most recent logged events."""
        with self._lock:
            return list(self.recent_events[-limit:])

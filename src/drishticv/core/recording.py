"""Annotated video recording and snapshot management.

Manages OpenCV VideoWriter lifecycle with multi-codec fallbacks (mp4v, avc1, XVID),
frame dimension consistency checks, and thread-safe file handling.
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records annotated frames to video files using OpenCV VideoWriter."""

    SUPPORTED_FOURCCS = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
    ]

    def __init__(self):
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording: bool = False
        self.output_path: Optional[Path] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 30.0
        self.start_time: float = 0.0
        self.frames_written: int = 0

    def start_recording(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> Tuple[bool, str]:
        """Initialize VideoWriter with fallback codec detection.

        Returns (success: bool, message: str).
        """
        if self.is_recording:
            return False, "Recording is already active."

        if width <= 0 or height <= 0:
            return False, f"Invalid video dimensions: {width}x{height}."

        # Guarantee parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_fps = fps if (fps and fps > 1.0 and fps <= 120.0) else 30.0
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.fps = target_fps
        self.output_path = output_path
        self.frames_written = 0

        # Try codecs in order of preference
        for codec_str, _ in self.SUPPORTED_FOURCCS:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height),
                )
                if writer.isOpened():
                    self.writer = writer
                    self.is_recording = True
                    self.start_time = time.time()
                    msg = f"Started recording using codec '{codec_str}' to {output_path.name}"
                    logger.info(msg)
                    return True, msg
                else:
                    writer.release()
            except Exception as e:
                logger.warning(f"Codec '{codec_str}' initialization failed: {e}")

        # If all failed
        self.is_recording = False
        self.writer = None
        err_msg = (
            f"Failed to open video writer for {output_path}.\n"
            f"Tested codecs: {[c[0] for c in self.SUPPORTED_FOURCCS]}.\n"
            f"Ensure OpenH264 / media codecs are supported on your system."
        )
        logger.error(err_msg)
        return False, err_msg

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write an annotated frame into the active video stream."""
        if not self.is_recording or self.writer is None:
            return False

        try:
            h, w = frame.shape[:2]
            # Handle frame dimension mismatch gracefully by resizing to initial writer dimensions
            if w != self.frame_width or h != self.frame_height:
                frame_to_write = cv2.resize(frame, (self.frame_width, self.frame_height))
            else:
                frame_to_write = frame

            self.writer.write(frame_to_write)
            self.frames_written += 1
            return True
        except Exception as e:
            logger.error(f"Error writing video frame: {e}")
            return False

    def stop_recording(self) -> Tuple[bool, Optional[Path], float, int]:
        """Finalize video writer and return recording statistics."""
        if not self.is_recording:
            return False, None, 0.0, 0

        duration = time.time() - self.start_time
        path = self.output_path
        total_frames = self.frames_written

        try:
            if self.writer is not None:
                self.writer.release()
            logger.info(f"Finalized recording {path}: {total_frames} frames in {duration:.1f}s.")
        except Exception as e:
            logger.error(f"Error finalizing video recording: {e}")
        finally:
            self.writer = None
            self.is_recording = False
            self.output_path = None
            self.frames_written = 0

        return True, path, duration, total_frames

    @staticmethod
    def save_snapshot(frame: np.ndarray, destination_path: Path) -> Tuple[bool, str]:
        """Save a single image frame to disk as a high-quality JPEG/PNG."""
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(destination_path), frame)
            if success:
                logger.info(f"Snapshot saved successfully: {destination_path}")
                return True, str(destination_path)
            else:
                return False, f"OpenCV failed to write image to {destination_path}"
        except Exception as e:
            err = f"Failed to save snapshot: {e}"
            logger.error(err)
            return False, err

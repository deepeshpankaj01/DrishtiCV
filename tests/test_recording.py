"""Tests for VideoRecorder and snapshot output."""

from pathlib import Path
import numpy as np
import pytest

from drishticv.core.recording import VideoRecorder


def test_save_snapshot(tmp_path):
    dest = tmp_path / "test_snapshot.jpg"
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    success, path_str = VideoRecorder.save_snapshot(frame, dest)
    assert success is True
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_video_recording_lifecycle(tmp_path):
    dest = tmp_path / "test_video.mp4"
    recorder = VideoRecorder()

    success, msg = recorder.start_recording(dest, width=320, height=240, fps=30.0)
    assert success is True
    assert recorder.is_recording is True

    # Write 15 synthetic frames
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    for _ in range(15):
        assert recorder.write_frame(frame) is True

    stop_success, path, duration, total_frames = recorder.stop_recording()
    assert stop_success is True
    assert total_frames == 15
    assert recorder.is_recording is False
    assert dest.is_file()
    assert dest.stat().st_size > 0

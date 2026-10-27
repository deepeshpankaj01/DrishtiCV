"""Unit tests for drishticv.core.event_logger module."""

import csv
import tempfile
from pathlib import Path
import pytest

from drishticv.core.event_logger import SessionManager, EventLogger, CSV_HEADERS
from drishticv.core.events import LineCrossingEvent, RegionMonitoringEvent


@pytest.fixture
def temp_session_manager(tmp_path):
    return SessionManager(base_dir=tmp_path)


def test_session_manager_directory_creation(temp_session_manager):
    sm = temp_session_manager
    assert sm.session_dir.is_dir()
    assert sm.snapshots_dir.is_dir()
    assert sm.recordings_dir.is_dir()
    assert sm.session_log_path.parent.is_dir()


def test_event_logger_csv_header_and_row_consistency(temp_session_manager):
    sm = temp_session_manager
    logger = EventLogger(sm)

    # Log an event
    logger.log_event(
        event_type="TEST_EVENT",
        source_name="Webcam 0",
        track_id=42,
        class_name="person",
        confidence=0.892,
        direction="Inbound",
        coordinates="(150.0,200.0)",
        details="Normal test event",
    )

    # Verify CSV file contents
    assert sm.events_csv_path.is_file()

    with open(sm.events_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2  # Header + 1 row
        assert reader[0] == CSV_HEADERS
        row = reader[1]
        assert row[1] == "TEST_EVENT"
        assert row[2] == "Webcam 0"
        assert row[3] == "42"
        assert row[4] == "person"
        assert row[5] == "0.892"
        assert row[6] == "Inbound"
        assert row[7] == "(150.0,200.0)"
        assert row[8] == "Normal test event"


def test_event_logger_special_characters_and_commas(temp_session_manager):
    sm = temp_session_manager
    logger = EventLogger(sm)

    # Ensure details containing commas, quotes, and newlines don't corrupt CSV structure
    complex_details = 'Coordinates: "100, 200", status: ok, comment: "tested, verified"'
    logger.log_event(
        event_type="SPECIAL_EVENT",
        details=complex_details,
    )

    with open(sm.events_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2
        # Reading must parse exactly 9 fields (same length as CSV_HEADERS)
        assert len(reader[1]) == len(CSV_HEADERS)
        assert reader[1][8] == complex_details


def test_event_logger_missing_track_id(temp_session_manager):
    sm = temp_session_manager
    logger = EventLogger(sm)

    # Untracked object (track_id = None)
    logger.log_event(
        event_type="UNTRACKED_DETECTION",
        track_id=None,
        class_name="bottle",
    )

    with open(sm.events_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2
        assert reader[1][3] == ""  # Empty string for track_id
        assert reader[1][4] == "bottle"


def test_event_logger_recent_events_memory(temp_session_manager):
    sm = temp_session_manager
    logger = EventLogger(sm)
    logger.max_recent = 5

    for i in range(10):
        logger.log_event(event_type=f"EVENT_{i}")

    recent = logger.get_recent_events(limit=10)
    assert len(recent) == 5
    assert recent[-1]["event_type"] == "EVENT_9"
    assert recent[0]["event_type"] == "EVENT_5"


def test_event_logger_line_and_region_helpers(temp_session_manager):
    sm = temp_session_manager
    logger = EventLogger(sm)

    line_evt = LineCrossingEvent(
        timestamp=1000.0,
        track_id=5,
        class_name="car",
        confidence=0.91,
        direction="A_TO_B",
        human_direction="Inbound (A→B)",
        crossing_point=(120.0, 150.0),
    )
    logged_line = logger.log_line_crossing(line_evt, "video.mp4")
    assert logged_line["event_type"] == "LINE_CROSSING"
    assert logged_line["track_id"] == 5

    reg_evt = RegionMonitoringEvent(
        timestamp=1001.0,
        event_type="REGION_ENTRY",
        track_id=7,
        class_name="person",
        confidence=0.88,
        centroid=(210.0, 320.0),
        region_box=(100.0, 100.0, 400.0, 400.0),
    )
    logged_reg = logger.log_region_event(reg_evt, "video.mp4")
    assert logged_reg["event_type"] == "REGION_ENTRY"
    assert logged_reg["track_id"] == 7

"""Unit tests for drishticv.core.events module."""

import pytest
from drishticv.core.events import (
    LineCrossingTracker,
    RegionMonitor,
    TrailManager,
    TrackItem,
)


def _make_track(track_id: int, cx: float, cy: float, class_name: str = "person", conf: float = 0.85) -> TrackItem:
    """Helper to create a TrackItem centered at (cx, cy)."""
    half = 10.0
    box = (cx - half, cy - half, cx + half, cy + half)
    return TrackItem(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=conf,
        box=box,
    )


def test_line_crossing_tracker_basic_crossing():
    tracker = LineCrossingTracker(debounce_frames=3)
    # Line from (100, 0) to (100, 200). Side A is left (x < 100), Side B is right (x > 100).
    tracker.set_line((100.0, 0.0), (100.0, 200.0))

    # Frame 1: Track #1 on Left (Side A)
    t1_f1 = _make_track(1, cx=80.0, cy=100.0)
    events_f1 = tracker.update([t1_f1])
    assert len(events_f1) == 0
    assert tracker.count_a_to_b == 0
    assert tracker.count_b_to_a == 0

    # Frame 2: Track #1 crosses to Right (Side B)
    t1_f2 = _make_track(1, cx=120.0, cy=100.0)
    events_f2 = tracker.update([t1_f2])
    assert len(events_f2) == 1
    assert events_f2[0].direction == "A_TO_B"
    assert events_f2[0].track_id == 1
    assert events_f2[0].class_name == "person"
    assert tracker.count_a_to_b == 1
    assert tracker.count_b_to_a == 0


def test_line_crossing_debounce_prevents_duplicate_counts():
    tracker = LineCrossingTracker(debounce_frames=5)
    tracker.set_line((100.0, 0.0), (100.0, 200.0))

    # Frame 1: on left
    tracker.update([_make_track(1, cx=80.0, cy=100.0)])
    # Frame 2: crosses to right -> crossing 1
    ev_2 = tracker.update([_make_track(1, cx=120.0, cy=100.0)])
    assert len(ev_2) == 1
    assert tracker.count_a_to_b == 1

    # Frame 3: rapid jitter back to left within debounce window (2 frames after)
    ev_3 = tracker.update([_make_track(1, cx=80.0, cy=100.0)])
    # Should be rejected by debounce filter
    assert len(ev_3) == 0
    assert tracker.count_b_to_a == 0


def test_line_crossing_stationary_object():
    tracker = LineCrossingTracker(debounce_frames=3)
    tracker.set_line((100.0, 0.0), (100.0, 200.0))

    # Stationary object near the line
    for _ in range(10):
        events = tracker.update([_make_track(1, cx=95.0, cy=100.0)])
        assert len(events) == 0

    assert tracker.count_a_to_b == 0
    assert tracker.count_b_to_a == 0


def test_line_crossing_multiple_classes():
    tracker = LineCrossingTracker(debounce_frames=2)
    tracker.set_line((100.0, 0.0), (100.0, 200.0))

    # Frame 1
    tracker.update([
        _make_track(1, cx=80.0, cy=50.0, class_name="person"),
        _make_track(2, cx=80.0, cy=150.0, class_name="car"),
    ])

    # Frame 2
    events = tracker.update([
        _make_track(1, cx=120.0, cy=50.0, class_name="person"),
        _make_track(2, cx=120.0, cy=150.0, class_name="car"),
    ])

    assert len(events) == 2
    assert tracker.class_counts["person"]["A_TO_B"] == 1
    assert tracker.class_counts["car"]["A_TO_B"] == 1
    assert tracker.count_a_to_b == 2


def test_line_crossing_reset_behavior():
    tracker = LineCrossingTracker()
    tracker.set_line((100.0, 0.0), (100.0, 200.0))
    tracker.update([_make_track(1, cx=80.0, cy=50.0)])
    tracker.update([_make_track(1, cx=120.0, cy=50.0)])
    assert tracker.count_a_to_b == 1

    # reset_all wipes cumulative counters
    tracker.reset_all()
    assert tracker.count_a_to_b == 0
    assert tracker.count_b_to_a == 0
    assert len(tracker.class_counts) == 0


def test_region_monitor_entry_and_exit():
    monitor = RegionMonitor()
    # Region between 50 and 150 on both axes
    monitor.set_region((50.0, 50.0, 150.0, 150.0))

    # Frame 1: Object outside at (20, 20)
    ev_1 = monitor.update([_make_track(1, cx=20.0, cy=20.0)])
    assert len(ev_1) == 0
    assert monitor.total_entries == 0

    # Frame 2: Object moves inside to (100, 100) -> ENTRY
    ev_2 = monitor.update([_make_track(1, cx=100.0, cy=100.0)])
    assert len(ev_2) == 1
    assert ev_2[0].event_type == "REGION_ENTRY"
    assert ev_2[0].track_id == 1
    assert monitor.total_entries == 1

    # Frame 3, 4, 5: Object remains inside at (105, 105), (110, 110)
    # Must NOT generate repeated entry events!
    for pos in [105.0, 110.0, 115.0]:
        ev_rem = monitor.update([_make_track(1, cx=pos, cy=pos)])
        assert len(ev_rem) == 0
    assert monitor.total_entries == 1

    # Frame 6: Object moves outside to (180, 180) -> EXIT
    ev_exit = monitor.update([_make_track(1, cx=180.0, cy=180.0)])
    assert len(ev_exit) == 1
    assert ev_exit[0].event_type == "REGION_EXIT"
    assert monitor.total_entries == 1  # Total entries stays at 1


def test_region_monitor_reset():
    monitor = RegionMonitor()
    monitor.set_region((50.0, 50.0, 150.0, 150.0))
    monitor.update([_make_track(1, cx=20.0, cy=20.0)])
    monitor.update([_make_track(1, cx=100.0, cy=100.0)])
    assert monitor.total_entries == 1

    monitor.reset_all()
    assert monitor.total_entries == 0
    assert len(monitor.class_entries) == 0
    assert len(monitor._inside_state) == 0


def test_trail_manager_lifecycle():
    tm = TrailManager(max_length=5)

    # Add points over 7 frames
    for i in range(7):
        tm.update([_make_track(1, cx=float(i * 10), cy=float(i * 10))])

    # Should be capped at max_length (5)
    assert len(tm.trails[1]) == 5
    # Oldest retained point should be from frame 2 (i=2, cx=20)
    assert tm.trails[1][0] == (20.0, 20.0)

    # Lost track should be evicted
    tm.update([])
    assert 1 not in tm.trails

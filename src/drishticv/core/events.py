"""Scene analytics state managers: virtual-line counting, region monitoring, and movement trails.

Maintains temporal track state, handles hysteresis/debouncing, and records
cumulative metrics without UI dependencies.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from drishticv.config import MAX_TRAIL_LENGTH
from drishticv.core.geometry import (
    Point,
    Rect,
    BoundingBox,
    CrossingDirection,
    box_centroid,
    which_side,
    check_line_crossing,
    point_in_rect,
)


@dataclass
class TrackItem:
    """Represents a single tracked object in the current frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    centroid: Point = field(init=False)

    def __post_init__(self):
        self.centroid = box_centroid(self.box)


@dataclass
class LineCrossingEvent:
    """Event data recorded when a tracked object crosses a virtual line."""
    timestamp: float
    track_id: int
    class_name: str
    confidence: float
    direction: CrossingDirection
    human_direction: str  # e.g., "Inbound (A→B)" or "Outbound (B→A)"
    crossing_point: Point


@dataclass
class RegionMonitoringEvent:
    """Event data recorded when a tracked object transitions into/out of a monitored region."""
    timestamp: float
    event_type: str  # "REGION_ENTRY" or "REGION_EXIT"
    track_id: int
    class_name: str
    confidence: float
    centroid: Point
    region_box: Rect


class LineCrossingTracker:
    """Evaluates and counts objects crossing a user-defined virtual line segment."""

    def __init__(self, debounce_frames: int = 5):
        self.line_p1: Optional[Point] = None
        self.line_p2: Optional[Point] = None
        self.debounce_frames: int = debounce_frames

        # State tracking per track_id
        self._last_positions: Dict[int, Point] = {}
        self._last_sides: Dict[int, int] = {}
        self._last_crossing_frame: Dict[int, int] = {}
        self._frame_index: int = 0

        # Cumulative counters
        self.count_a_to_b: int = 0
        self.count_b_to_a: int = 0
        self.class_counts: Dict[str, Dict[str, int]] = {}

    def set_line(self, p1: Point, p2: Point) -> None:
        """Set or update the virtual line endpoints. Resets per-track side tracking."""
        self.line_p1 = (float(p1[0]), float(p1[1]))
        self.line_p2 = (float(p2[0]), float(p2[1]))
        self.reset_state_only()

    def clear_line(self) -> None:
        """Remove the virtual line and reset crossing state."""
        self.line_p1 = None
        self.line_p2 = None
        self.reset_state_only()

    def is_configured(self) -> bool:
        """Return True if a valid virtual line has been configured."""
        return self.line_p1 is not None and self.line_p2 is not None

    def reset_state_only(self) -> None:
        """Reset temporal state without wiping cumulative totals."""
        self._last_positions.clear()
        self._last_sides.clear()
        self._last_crossing_frame.clear()

    def reset_all(self) -> None:
        """Full reset of line counting state and cumulative metrics."""
        self.reset_state_only()
        self.count_a_to_b = 0
        self.count_b_to_a = 0
        self.class_counts.clear()
        self._frame_index = 0

    def update(self, tracks: List[TrackItem]) -> List[LineCrossingEvent]:
        """Update tracker with the current frame's tracked objects."""
        self._frame_index += 1
        if not self.is_configured():
            return []

        assert self.line_p1 is not None and self.line_p2 is not None
        events: List[LineCrossingEvent] = []
        active_ids = {t.track_id for t in tracks}

        for track in tracks:
            tid = track.track_id
            curr_pt = track.centroid
            prev_pt = self._last_positions.get(tid)

            curr_side = which_side(self.line_p1, self.line_p2, curr_pt, deadzone=1.0)

            if prev_pt is not None:
                # Check for crossing
                direction = check_line_crossing(
                    self.line_p1,
                    self.line_p2,
                    prev_pt,
                    curr_pt,
                    require_segment_intersection=True,
                    deadzone=1.0,
                )

                # Check debouncing to prevent jitter re-triggers
                last_frame = self._last_crossing_frame.get(tid, -999)
                frames_since_last = self._frame_index - last_frame

                if direction is not None and frames_since_last > self.debounce_frames:
                    self._last_crossing_frame[tid] = self._frame_index

                    if direction == "A_TO_B":
                        self.count_a_to_b += 1
                        human_dir = "Inbound (A→B)"
                    else:
                        self.count_b_to_a += 1
                        human_dir = "Outbound (B→A)"

                    # Update class-specific stats
                    if track.class_name not in self.class_counts:
                        self.class_counts[track.class_name] = {"A_TO_B": 0, "B_TO_A": 0}
                    self.class_counts[track.class_name][direction] += 1

                    event = LineCrossingEvent(
                        timestamp=time.time(),
                        track_id=tid,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        direction=direction,
                        human_direction=human_dir,
                        crossing_point=curr_pt,
                    )
                    events.append(event)

            # Update position and side
            self._last_positions[tid] = curr_pt
            if curr_side != 0:
                self._last_sides[tid] = curr_side

        # Clean up stale track histories
        stale_ids = [tid for tid in self._last_positions if tid not in active_ids]
        for sid in stale_ids:
            # Keep for a short window or delete if absent
            del self._last_positions[sid]
            self._last_sides.pop(sid, None)
            self._last_crossing_frame.pop(sid, None)

        return events


class RegionMonitor:
    """Monitors a user-defined rectangular region of interest and records entry/exit events."""

    def __init__(self):
        self.region_rect: Optional[Rect] = None
        # Track ID -> current state (True = inside, False = outside)
        self._inside_state: Dict[int, bool] = {}
        self.total_entries: int = 0
        self.class_entries: Dict[str, int] = {}

    def set_region(self, rect: Rect) -> None:
        """Set monitored rectangular region (x1, y1, x2, y2). Resets state."""
        rx1, ry1, rx2, ry2 = rect
        self.region_rect = (min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2))
        self.reset_state_only()

    def clear_region(self) -> None:
        """Remove monitored region."""
        self.region_rect = None
        self.reset_state_only()

    def is_configured(self) -> bool:
        """Return True if a monitored region is configured."""
        return self.region_rect is not None

    def reset_state_only(self) -> None:
        """Clear active object occupancy state."""
        self._inside_state.clear()

    def reset_all(self) -> None:
        """Full reset of region state and cumulative metrics."""
        self.reset_state_only()
        self.total_entries = 0
        self.class_entries.clear()

    def update(self, tracks: List[TrackItem]) -> List[RegionMonitoringEvent]:
        """Update monitor with current tracked objects and detect entry/exit events."""
        if not self.is_configured():
            return []

        assert self.region_rect is not None
        events: List[RegionMonitoringEvent] = []
        active_ids = {t.track_id for t in tracks}

        for track in tracks:
            tid = track.track_id
            is_inside = point_in_rect(self.region_rect, track.centroid)
            prev_inside = self._inside_state.get(tid, None)

            if prev_inside is not None:
                # Transition detected
                if not prev_inside and is_inside:
                    # Transition: OUTSIDE -> INSIDE (Entry Event)
                    self.total_entries += 1
                    self.class_entries[track.class_name] = self.class_entries.get(track.class_name, 0) + 1

                    events.append(
                        RegionMonitoringEvent(
                            timestamp=time.time(),
                            event_type="REGION_ENTRY",
                            track_id=tid,
                            class_name=track.class_name,
                            confidence=track.confidence,
                            centroid=track.centroid,
                            region_box=self.region_rect,
                        )
                    )
                elif prev_inside and not is_inside:
                    # Transition: INSIDE -> OUTSIDE (Exit Event)
                    events.append(
                        RegionMonitoringEvent(
                            timestamp=time.time(),
                            event_type="REGION_EXIT",
                            track_id=tid,
                            class_name=track.class_name,
                            confidence=track.confidence,
                            centroid=track.centroid,
                            region_box=self.region_rect,
                        )
                    )

            # Record current state
            self._inside_state[tid] = is_inside

        # Clean up stale track IDs
        stale_ids = [tid for tid in self._inside_state if tid not in active_ids]
        for sid in stale_ids:
            del self._inside_state[sid]

        return events


class TrailManager:
    """Maintains recent coordinate histories for active tracked objects."""

    def __init__(self, max_length: int = MAX_TRAIL_LENGTH):
        self.max_length = max_length
        self.trails: Dict[int, deque[Point]] = {}

    def update(self, tracks: List[TrackItem]) -> None:
        """Update trails with new positions and remove lost tracks."""
        active_ids = set()
        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)
            if tid not in self.trails:
                self.trails[tid] = deque(maxlen=self.max_length)
            self.trails[tid].append(track.centroid)

        # Evict inactive tracks
        lost_ids = [tid for tid in self.trails if tid not in active_ids]
        for lid in lost_ids:
            del self.trails[lid]

    def clear(self) -> None:
        """Clear all active trails."""
        self.trails.clear()

"""Core processing and scene analytics modules for DrishtiCV."""

from drishticv.core.geometry import (
    which_side,
    check_line_crossing,
    point_in_rect,
    check_region_transition,
    box_centroid,
    euclidean_distance,
)
from drishticv.core.events import (
    LineCrossingTracker,
    RegionMonitor,
    TrailManager,
    TrackItem,
    LineCrossingEvent,
    RegionMonitoringEvent,
)
from drishticv.core.event_logger import SessionManager, EventLogger
from drishticv.core.recording import VideoRecorder

__all__ = [
    "which_side",
    "check_line_crossing",
    "point_in_rect",
    "check_region_transition",
    "box_centroid",
    "euclidean_distance",
    "LineCrossingTracker",
    "RegionMonitor",
    "TrailManager",
    "TrackItem",
    "LineCrossingEvent",
    "RegionMonitoringEvent",
    "SessionManager",
    "EventLogger",
    "VideoRecorder",
]

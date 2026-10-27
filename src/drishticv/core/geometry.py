"""Core computational geometry routines for line crossings and region monitoring.

Pure Python implementation without GUI or hardware dependencies, allowing
direct, rigorous unit testing.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Literal
from dataclasses import dataclass

Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Rect = Tuple[float, float, float, float]         # x1, y1, x2, y2 (normalized or absolute)

CrossingDirection = Literal["A_TO_B", "B_TO_A"]
RegionState = Literal["OUTSIDE", "INSIDE"]
RegionEvent = Literal["ENTRY", "EXIT"]


def box_centroid(box: BoundingBox) -> Point:
    """Calculate the geometric center (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def which_side(line_p1: Point, line_p2: Point, pt: Point, deadzone: float = 0.5) -> int:
    """Determine which side of a directed line (p1 -> p2) a point lies on.

    Uses the 2D cross product (signed area determinant):
        d = (p2.x - p1.x) * (pt.y - p1.y) - (p2.y - p1.y) * (pt.x - p1.x)

    Returns:
        +1: Side A (Left of directed vector p1 -> p2)
        -1: Side B (Right of directed vector p1 -> p2)
         0: On the line (or within the deadzone margin)
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    det = dx * (pt[1] - line_p1[1]) - dy * (pt[0] - line_p1[0])

    line_len = math.hypot(dx, dy)
    if line_len < 1e-6:
        return 0

    # Normalised perpendicular distance
    dist = det / line_len

    if dist > deadzone:
        return 1
    elif dist < -deadzone:
        return -1
    return 0


def _ccw(a: Point, b: Point, c: Point) -> bool:
    """Return True if points a, b, c are in counter-clockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Return True if segment (p1, p2) intersects segment (p3, p4)."""
    return (
        _ccw(p1, p3, p4) != _ccw(p2, p3, p4)
        and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)
    )


def check_line_crossing(
    line_p1: Point,
    line_p2: Point,
    prev_pt: Point,
    curr_pt: Point,
    require_segment_intersection: bool = True,
    deadzone: float = 0.5,
) -> Optional[CrossingDirection]:
    """Evaluate whether an object movement from prev_pt to curr_pt crossed the virtual line.

    Args:
        line_p1: Starting endpoint of the virtual line (defines vector direction).
        line_p2: Ending endpoint of the virtual line.
        prev_pt: Previous centroid location of the object.
        curr_pt: Current centroid location of the object.
        require_segment_intersection: If True, the movement vector must physically
            intersect the virtual line segment. If False, crossing the infinite line
            in the vicinity is evaluated.
        deadzone: Perpendicular deadzone margin to prevent jitter counts.

    Returns:
        "A_TO_B" if crossed from side +1 to -1.
        "B_TO_A" if crossed from side -1 to +1.
        None if no crossing occurred or transition landed in deadzone.
    """
    if prev_pt == curr_pt:
        return None

    side_prev = which_side(line_p1, line_p2, prev_pt, deadzone=deadzone)
    side_curr = which_side(line_p1, line_p2, curr_pt, deadzone=deadzone)

    # Must transition cleanly between opposing sides (not from or to deadzone 0)
    if side_prev == 0 or side_curr == 0 or side_prev == side_curr:
        return None

    if require_segment_intersection:
        if not segments_intersect(line_p1, line_p2, prev_pt, curr_pt):
            return None

    if side_prev == 1 and side_curr == -1:
        return "A_TO_B"
    elif side_prev == -1 and side_curr == 1:
        return "B_TO_A"

    return None


def point_in_rect(rect: Rect, pt: Point) -> bool:
    """Check if point pt is inside the bounding rectangle (x1, y1, x2, y2).

    Normalizes coordinate boundaries so that (x1 <= x2 and y1 <= y2).
    """
    rx1, ry1, rx2, ry2 = rect
    min_x, max_x = min(rx1, rx2), max(rx1, rx2)
    min_y, max_y = min(ry1, ry2), max(ry1, ry2)

    return (min_x <= pt[0] <= max_x) and (min_y <= pt[1] <= max_y)


def check_region_transition(
    rect: Rect,
    prev_pt: Optional[Point],
    curr_pt: Point,
) -> Optional[RegionEvent]:
    """Determine if movement between prev_pt and curr_pt enters or exits a monitored rectangle.

    Args:
        rect: (x1, y1, x2, y2) bounds of the monitored region.
        prev_pt: Previous point (can be None if first detected).
        curr_pt: Current point.

    Returns:
        "ENTRY" if moving from outside to inside.
        "EXIT" if moving from inside to outside.
        None if remaining inside, remaining outside, or first frame inside.
    """
    curr_inside = point_in_rect(rect, curr_pt)

    if prev_pt is None:
        # Initial point detection does not trigger transition event
        return None

    prev_inside = point_in_rect(rect, prev_pt)

    if not prev_inside and curr_inside:
        return "ENTRY"
    elif prev_inside and not curr_inside:
        return "EXIT"

    return None

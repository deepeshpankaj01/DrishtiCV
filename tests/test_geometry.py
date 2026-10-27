"""Unit tests for drishticv.core.geometry module."""

import pytest
from drishticv.core.geometry import (
    which_side,
    segments_intersect,
    check_line_crossing,
    point_in_rect,
    check_region_transition,
    box_centroid,
    euclidean_distance,
)


def test_box_centroid():
    box = (100.0, 200.0, 300.0, 400.0)
    cx, cy = box_centroid(box)
    assert cx == 200.0
    assert cy == 300.0


def test_euclidean_distance():
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    assert euclidean_distance(p1, p2) == 5.0


def test_which_side_horizontal_line():
    # Vector from (0, 100) to (200, 100) -> points pointing right
    # Looking from start to end:
    # Points with y > 100 are to the right (Side B: -1 in standard screen coordinates where +y is down)
    # or left depending on det:
    # dx = 200, dy = 0.
    # det = 200 * (pt.y - 100) - 0.
    # If pt.y = 150 -> det = +10000 > 0 -> Side A (+1)
    # If pt.y = 50 -> det = -10000 < 0 -> Side B (-1)
    line_p1 = (0.0, 100.0)
    line_p2 = (200.0, 100.0)

    side_above = which_side(line_p1, line_p2, (100.0, 50.0))
    side_below = which_side(line_p1, line_p2, (100.0, 150.0))
    side_on_line = which_side(line_p1, line_p2, (100.0, 100.0))

    assert side_below == 1
    assert side_above == -1
    assert side_on_line == 0


def test_which_side_vertical_line():
    # Vector from (100, 0) to (100, 200) -> points downwards
    # dx = 0, dy = 200.
    # det = 0 - 200 * (pt.x - 100) = -200 * (pt.x - 100)
    # If pt.x = 50 -> det = -200 * (-50) = +10000 -> Side A (+1)
    # If pt.x = 150 -> det = -200 * (+50) = -10000 -> Side B (-1)
    line_p1 = (100.0, 0.0)
    line_p2 = (100.0, 200.0)

    side_left = which_side(line_p1, line_p2, (50.0, 100.0))
    side_right = which_side(line_p1, line_p2, (150.0, 100.0))
    side_on = which_side(line_p1, line_p2, (100.0, 100.0))

    assert side_left == 1
    assert side_right == -1
    assert side_on == 0


def test_which_side_deadzone():
    line_p1 = (100.0, 0.0)
    line_p2 = (100.0, 200.0)
    # pt.x is 100.2 (distance is 0.2, within deadzone 0.5)
    side = which_side(line_p1, line_p2, (100.2, 100.0), deadzone=0.5)
    assert side == 0


def test_segments_intersect():
    # Intersecting cross
    assert segments_intersect((0, 50), (100, 50), (50, 0), (50, 100)) is True

    # Parallel disjoint segments
    assert segments_intersect((0, 0), (100, 0), (0, 20), (100, 20)) is False

    # Disjoint non-intersecting
    assert segments_intersect((0, 0), (10, 10), (20, 20), (30, 30)) is False


def test_check_line_crossing_valid_transitions():
    line_p1 = (100.0, 0.0)
    line_p2 = (100.0, 200.0)

    # Moving from left (Side A) to right (Side B)
    prev_pt = (80.0, 100.0)
    curr_pt = (120.0, 100.0)

    direction = check_line_crossing(line_p1, line_p2, prev_pt, curr_pt)
    assert direction == "A_TO_B"

    # Moving from right (Side B) to left (Side A)
    direction_rev = check_line_crossing(line_p1, line_p2, curr_pt, prev_pt)
    assert direction_rev == "B_TO_A"


def test_check_line_crossing_no_intersection():
    line_p1 = (100.0, 0.0)
    line_p2 = (100.0, 100.0)

    # Movement crosses infinite line at y=150, but beyond segment (y in [0, 100])
    prev_pt = (80.0, 150.0)
    curr_pt = (120.0, 150.0)

    direction = check_line_crossing(line_p1, line_p2, prev_pt, curr_pt, require_segment_intersection=True)
    assert direction is None


def test_check_line_crossing_same_side():
    line_p1 = (100.0, 0.0)
    line_p2 = (100.0, 200.0)

    prev_pt = (80.0, 50.0)
    curr_pt = (70.0, 150.0)

    direction = check_line_crossing(line_p1, line_p2, prev_pt, curr_pt)
    assert direction is None


def test_point_in_rect():
    rect = (50.0, 50.0, 150.0, 150.0)
    assert point_in_rect(rect, (100.0, 100.0)) is True
    assert point_in_rect(rect, (50.0, 50.0)) is True  # on boundary
    assert point_in_rect(rect, (150.0, 150.0)) is True
    assert point_in_rect(rect, (40.0, 100.0)) is False
    assert point_in_rect(rect, (200.0, 100.0)) is False

    # Inverted rectangle coordinates (x2 < x1, y2 < y1) should be normalized
    rect_inverted = (150.0, 150.0, 50.0, 50.0)
    assert point_in_rect(rect_inverted, (100.0, 100.0)) is True


def test_check_region_transition():
    rect = (50.0, 50.0, 150.0, 150.0)

    # Outside to Inside -> ENTRY
    assert check_region_transition(rect, (20.0, 20.0), (100.0, 100.0)) == "ENTRY"

    # Inside to Outside -> EXIT
    assert check_region_transition(rect, (100.0, 100.0), (200.0, 200.0)) == "EXIT"

    # Inside to Inside -> None
    assert check_region_transition(rect, (70.0, 70.0), (120.0, 120.0)) is None

    # Outside to Outside -> None
    assert check_region_transition(rect, (20.0, 20.0), (30.0, 30.0)) is None

    # Initial frame (prev_pt is None) -> None
    assert check_region_transition(rect, None, (100.0, 100.0)) is None

"""Tests for DrishtiDetector result structures and frame annotation routines."""

import numpy as np
from drishticv.detection.detector import DrishtiDetector, DetectedObject, DetectionResult


def test_detection_result_properties():
    obj1 = DetectedObject(
        box=(10.0, 20.0, 50.0, 60.0),
        class_id=0,
        class_name="person",
        confidence=0.88,
        track_id=1,
    )
    obj2 = DetectedObject(
        box=(70.0, 80.0, 110.0, 120.0),
        class_id=2,
        class_name="car",
        confidence=0.75,
        track_id=None,  # Untracked object
    )

    res = DetectionResult(
        objects=[obj1, obj2],
        inference_time_ms=12.5,
        frame_width=640,
        frame_height=480,
    )

    assert res.count == 2
    assert res.tracked_count == 1
    assert res.counts_by_class == {"person": 1, "car": 1}
    assert obj1.centroid == (30.0, 40.0)


def test_annotate_frame_renders_cleanly():
    detector = DrishtiDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    obj = DetectedObject(
        box=(50.0, 50.0, 150.0, 150.0),
        class_id=0,
        class_name="person",
        confidence=0.92,
        track_id=3,
    )
    res = DetectionResult([obj], 10.0, 640, 480)

    trails = {3: [(50.0, 50.0), (70.0, 70.0), (100.0, 100.0)]}

    annotated = detector.annotate_frame(
        frame,
        res,
        trails=trails,
        draw_trails=True,
        show_labels=True,
        show_conf=True,
        show_tracks=True,
    )

    assert annotated.shape == (480, 640, 3)
    # The output should have non-zero pixels where the box, label, and trails were drawn
    assert np.count_nonzero(annotated) > 0

import numpy as np

from perception.overlay import draw_detections
from perception.tracker import TrackedObject


def _tracked(bbox, stale=False):
    return TrackedObject(
        track_id=0, position=np.zeros(3), shape_label="cube", bbox=bbox, frames_since_seen=1 if stale else 0
    )


def test_draw_detections_returns_new_array_same_shape():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    annotated = draw_detections(frame, [_tracked((10, 10, 20, 20))])
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert not np.array_equal(annotated, frame)


def test_draw_detections_does_not_mutate_input():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    original = frame.copy()
    draw_detections(frame, [_tracked((10, 10, 20, 20))])
    np.testing.assert_array_equal(frame, original)


def test_draw_detections_draws_bbox_border_pixels():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    annotated = draw_detections(frame, [_tracked((10, 10, 20, 20))])
    assert annotated[10, 10:21].any()


def test_no_detections_returns_unchanged_frame():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    annotated = draw_detections(frame, [])
    np.testing.assert_array_equal(annotated, frame)

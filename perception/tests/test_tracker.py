import numpy as np

from perception.pipeline import Detection
from perception.tracker import ObjectTracker, find_by_shape


def _det(x, y, z, shape_label="cube"):
    return Detection(position=np.array([x, y, z]), shape_label=shape_label, bbox=(0, 0, 1, 1), pixel_count=4)


def test_new_detection_creates_track():
    tracker = ObjectTracker()
    tracked = tracker.update([_det(0.0, 0.0, 0.0)])
    assert len(tracked) == 1
    assert tracked[0].frames_since_seen == 0
    assert not tracked[0].is_stale


def test_matching_detection_keeps_same_track_id():
    tracker = ObjectTracker(match_distance=0.03)
    first = tracker.update([_det(0.0, 0.0, 0.0)])
    second = tracker.update([_det(0.001, 0.001, 0.0)])
    assert second[0].track_id == first[0].track_id
    assert second[0].frames_since_seen == 0


def test_missing_detection_persists_last_known_pose_and_goes_stale():
    tracker = ObjectTracker(match_distance=0.03, max_missed_frames=15)
    first = tracker.update([_det(0.1, 0.2, 0.0)])
    tracked = tracker.update([])
    assert len(tracked) == 1
    assert tracked[0].track_id == first[0].track_id
    np.testing.assert_array_equal(tracked[0].position, first[0].position)
    assert tracked[0].frames_since_seen == 1
    assert tracked[0].is_stale


def test_reappearing_detection_resumes_same_track_and_clears_staleness():
    tracker = ObjectTracker(match_distance=0.03)
    first = tracker.update([_det(0.1, 0.2, 0.0)])
    tracker.update([])
    tracker.update([])
    resumed = tracker.update([_det(0.101, 0.199, 0.0)])
    assert resumed[0].track_id == first[0].track_id
    assert resumed[0].frames_since_seen == 0


def test_track_dropped_after_too_many_missed_frames():
    tracker = ObjectTracker(match_distance=0.03, max_missed_frames=2)
    tracker.update([_det(0.1, 0.2, 0.0)])
    tracker.update([])
    tracker.update([])
    tracked = tracker.update([])
    assert tracked == []


def test_distant_detections_stay_separate_tracks():
    tracker = ObjectTracker(match_distance=0.03)
    tracked = tracker.update([_det(0.0, 0.0, 0.0), _det(1.0, 1.0, 0.0)])
    assert len(tracked) == 2
    assert tracked[0].track_id != tracked[1].track_id


def test_find_by_shape_prefers_freshest_match():
    tracker = ObjectTracker(match_distance=0.03)
    tracker.update([_det(0.0, 0.0, 0.0, shape_label="cube")])
    tracker.update([_det(0.5, 0.5, 0.0, shape_label="cube")])
    tracked = tracker.update([_det(0.5, 0.5, 0.0, shape_label="cube")])
    match = find_by_shape(tracked, "cube")
    assert match is not None
    assert match.frames_since_seen == 0


def test_find_by_shape_returns_none_when_absent():
    tracker = ObjectTracker()
    tracked = tracker.update([_det(0.0, 0.0, 0.0, shape_label="sphere")])
    assert find_by_shape(tracked, "cube") is None

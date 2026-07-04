"""Frame-to-frame nearest-centroid tracker with last-known-pose persistence.

An object that briefly disappears (occluded by the arm passing over it, or
by another object) keeps reporting its last-known position with a
`frames_since_seen` staleness counter, rather than vanishing from the
tracked-objects list entirely - see the design doc's occlusion-handling
discussion.
"""

from dataclasses import dataclass

import numpy as np

from perception.pipeline import Detection


@dataclass
class TrackedObject:
    track_id: int
    position: np.ndarray
    shape_label: str
    bbox: tuple[int, int, int, int]
    frames_since_seen: int = 0

    @property
    def is_stale(self) -> bool:
        return self.frames_since_seen > 0


class ObjectTracker:
    def __init__(self, match_distance: float = 0.03, max_missed_frames: int = 15):
        self.match_distance = match_distance
        self.max_missed_frames = max_missed_frames
        self._tracks: list[TrackedObject] = []
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        candidate_pairs = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                dist = float(np.linalg.norm(track.position - det.position))
                if dist <= self.match_distance:
                    candidate_pairs.append((dist, ti, di))
        candidate_pairs.sort(key=lambda p: p[0])

        matched_tracks: dict[int, int] = {}
        matched_detections: set[int] = set()
        for _, ti, di in candidate_pairs:
            if ti in matched_tracks or di in matched_detections:
                continue
            matched_tracks[ti] = di
            matched_detections.add(di)

        new_tracks: list[TrackedObject] = []
        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks:
                det = detections[matched_tracks[ti]]
                new_tracks.append(TrackedObject(track.track_id, det.position, det.shape_label, det.bbox, 0))
            elif track.frames_since_seen + 1 <= self.max_missed_frames:
                new_tracks.append(
                    TrackedObject(
                        track.track_id, track.position, track.shape_label, track.bbox, track.frames_since_seen + 1
                    )
                )
            # else: dropped, missed too many consecutive frames

        for di, det in enumerate(detections):
            if di not in matched_detections:
                new_tracks.append(TrackedObject(self._next_id, det.position, det.shape_label, det.bbox, 0))
                self._next_id += 1

        self._tracks = new_tracks
        return list(self._tracks)


def find_by_shape(tracked: list[TrackedObject], shape_label: str) -> TrackedObject | None:
    candidates = [t for t in tracked if t.shape_label == shape_label]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t.frames_since_seen)

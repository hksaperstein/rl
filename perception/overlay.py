"""Draws each tracked object's pixel bbox and shape label onto a video frame -
shared by the calibration clip, eval video, and interactive-demo video, so
perception correctness is visible during actual policy runs, not just in
isolation.
"""

import cv2
import numpy as np

FRESH_COLOR = (0, 255, 0)  # BGR: green
STALE_COLOR = (0, 165, 255)  # BGR: orange


def draw_detections(frame_rgb: np.ndarray, tracked_objects: list) -> np.ndarray:
    """Returns an annotated copy of `frame_rgb` (H, W, 3) uint8. Stale
    (not-currently-visible) detections are drawn in a different color so
    staleness is visible in the recorded video, not just in the underlying data.
    """
    annotated = frame_rgb.copy()
    for obj in tracked_objects:
        row_min, col_min, row_max, col_max = obj.bbox
        color = STALE_COLOR if obj.is_stale else FRESH_COLOR
        cv2.rectangle(annotated, (col_min, row_min), (col_max, row_max), color, 1)
        label = obj.shape_label + (" (stale)" if obj.is_stale else "")
        text_origin = (col_min, max(row_min - 4, 10))
        cv2.putText(annotated, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return annotated

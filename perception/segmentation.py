"""Ground-plane removal + connected-component clustering on a single depth image.

The camera looks straight down at a flat ground plane, so "ground removal" is a
robust plane-depth estimate (the median depth, since the ground dominates the
pixel count) rather than a full 3D RANSAC plane fit, and "clustering" is
connected-component labeling directly on the resulting 2D binary mask (the
depth image is an organized/structured grid, so this is equivalent to Euclidean
clustering for this top-down, non-stacking-objects setup, without needing a
point-cloud clustering library).
"""

import numpy as np
from scipy import ndimage


def estimate_ground_depth(depth: np.ndarray) -> float:
    """Robust ground-plane depth estimate: the median of all finite depth values."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("Depth image has no finite values.")
    return float(np.median(finite))


def segment_objects(
    depth: np.ndarray, ground_margin: float = 0.003, min_cluster_pixels: int = 12
) -> tuple[np.ndarray, list[int]]:
    """Labels non-ground pixels into connected-component clusters.

    Args:
        depth: (H, W) orthogonal depth image (distance from the camera's image
            plane), in meters.
        ground_margin: pixels within this many meters of the estimated ground
            depth are considered ground, not object. 3mm default.
        min_cluster_pixels: clusters smaller than this are treated as noise and
            excluded from the returned valid ids.

    Returns:
        (labeled, valid_ids): `labeled` is an (H, W) int array (0 = ground/background,
        1..N = cluster ids); `valid_ids` lists the cluster ids that passed the
        `min_cluster_pixels` filter.
    """
    ground_depth = estimate_ground_depth(depth)
    object_mask = np.isfinite(depth) & (depth < (ground_depth - ground_margin))
    labeled, num_labels = ndimage.label(object_mask)
    valid_ids = [
        label_id
        for label_id in range(1, num_labels + 1)
        if np.count_nonzero(labeled == label_id) >= min_cluster_pixels
    ]
    return labeled, valid_ids

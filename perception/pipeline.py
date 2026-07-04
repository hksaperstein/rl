"""Ties ground-plane removal, clustering, and shape classification together
into one per-frame perception pass, plus the depth-to-world-point math used
to localize each detected object in 3D.

Deliberately reimplements depth unprojection in plain numpy (rather than
importing Isaac Lab's `isaaclab.utils.math.unproject_depth`, which is
torch-based) so this whole package stays free of any Isaac Sim/torch
dependency and is testable with plain pytest.
"""

from dataclasses import dataclass

import numpy as np

from perception.segmentation import segment_objects
from perception.shape_classifier import classify_shape


@dataclass
class Detection:
    position: np.ndarray  # (3,), world frame, meters
    shape_label: str
    bbox: tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max), inclusive
    pixel_count: int


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def unproject_depth_image(depth: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """Converts an orthogonal depth image (H, W) into a (H, W, 3) grid of camera-local
    points, using the ROS camera-frame convention (x right, y down, z forward) - matching
    Isaac Lab's `CameraData.quat_w_ros`."""
    h, w = depth.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.stack([x, y, depth], axis=-1)


def build_world_point_grid(
    depth: np.ndarray, intrinsic_matrix: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray
) -> np.ndarray:
    """Returns a (H, W, 3) grid of world-frame points for every pixel in `depth`."""
    points_cam = unproject_depth_image(depth, intrinsic_matrix)
    h, w = depth.shape
    rot = quat_to_rot_matrix(cam_quat_w_ros)
    points_world = points_cam.reshape(-1, 3) @ rot.T + cam_pos_w
    return points_world.reshape(h, w, 3)


def run_perception(
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    ground_z: float = 0.0,
    ground_margin: float = 0.003,
    min_cluster_pixels: int = 12,
) -> list[Detection]:
    """Runs the full perception pipeline on one depth frame: ground removal,
    clustering, per-cluster shape classification, and 3D localization."""
    labeled, valid_ids = segment_objects(depth, ground_margin=ground_margin, min_cluster_pixels=min_cluster_pixels)
    world_points = build_world_point_grid(depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros)

    detections = []
    for label_id in valid_ids:
        mask = labeled == label_id
        cluster_points = world_points[mask]
        rows, cols = np.nonzero(mask)
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
        classification = classify_shape(cluster_points, ground_z=ground_z)
        centroid_xy = cluster_points[:, :2].mean(axis=0)
        position = np.array([centroid_xy[0], centroid_xy[1], ground_z + classification.height])
        detections.append(
            Detection(position=position, shape_label=classification.label, bbox=bbox, pixel_count=int(mask.sum()))
        )
    return detections

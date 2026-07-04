import numpy as np

from perception.pipeline import build_world_point_grid, quat_to_rot_matrix, run_perception

CAM_POS = np.array([0.0, 0.0, 0.5])
CAM_QUAT_ROS = np.array([0.0, 1.0, 0.0, 0.0])  # straight down: local +Z (forward) -> world -Z
FX = FY = 500.0
H = W = 60
CX = CY = 30.0
INTRINSICS = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def test_quat_to_rot_matrix_straight_down():
    rot = quat_to_rot_matrix(CAM_QUAT_ROS)
    forward_world = rot @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(forward_world, [0.0, 0.0, -1.0], atol=1e-9)


def test_run_perception_reports_correct_height_and_shape():
    depth = np.full((H, W), 0.5)
    depth[20:30, 20:30] = 0.5 - 0.018
    detections = run_perception(depth, INTRINSICS, CAM_POS, CAM_QUAT_ROS, ground_z=0.0)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.shape_label == "cube"
    assert abs(detection.position[2] - 0.018) < 1e-6
    assert detection.bbox == (20, 20, 29, 29)


def test_run_perception_world_spacing_matches_pixel_spacing():
    depth = np.full((H, W), 0.5)
    depth[10:14, 5:9] = 0.48
    depth[10:14, 25:29] = 0.48
    detections = run_perception(depth, INTRINSICS, CAM_POS, CAM_QUAT_ROS, ground_z=0.0, min_cluster_pixels=4)
    assert len(detections) == 2
    positions = sorted((d.position for d in detections), key=lambda p: p[0])
    spacing = np.linalg.norm(positions[1] - positions[0])
    expected_spacing = 20 / FX * 0.48
    assert abs(spacing - expected_spacing) < 1e-3


def test_build_world_point_grid_center_pixel_hits_ground_below_camera():
    depth = np.full((H, W), 0.5)
    world_points = build_world_point_grid(depth, INTRINSICS, CAM_POS, CAM_QUAT_ROS)
    center = world_points[int(CY), int(CX)]
    np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-9)

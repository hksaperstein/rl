# AR4 Perception Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real top-down RGB-D camera and a classical-CV perception pipeline (segmentation, geometric shape classification, occlusion-tolerant tracking) to the AR4 pick-and-place task, keep training on privileged simulation state for speed, wire perception in only at eval/demo time, scale training to `num_envs=4096`, and ship a calibration clip, an interactive drag-the-cube demo, and a README tying it all together.

**Architecture:** A new `rl/perception/` package (pure numpy/scipy/cv2, zero Isaac Sim or torch dependency) implements ground-plane removal + connected-component clustering (`segmentation.py`), per-cluster geometric shape classification (`shape_classifier.py`), depth-to-world unprojection + the end-to-end per-frame pipeline (`pipeline.py`), frame-to-frame tracking with last-known-pose/staleness persistence (`tracker.py`), and detection overlay drawing (`overlay.py`) — all independently unit-tested with plain `pytest`, no Isaac Sim launch required. A new `Ar4PickPlacePerceptionSceneCfg`/`EnvCfg` in `pickplace_env_cfg.py` adds a static top-down `CameraCfg`, used only by `perception_calibration.py`, `eval_loop.py --perception`, and `interactive_demo.py` — never by `train.py`, which keeps training exactly as it already works, just at `num_envs=4096`. A shared `rl/scripts/_perception_adapter.py` bridges Isaac Lab tensors to the perception package and substitutes the perception-derived cube position into the same observation slot the policy was trained on (computed from the observation manager, not hardcoded).

**Tech Stack:** Isaac Lab (`ManagerBasedRLEnv`, `CameraCfg`), `rsl_rl` PPO (unchanged), numpy/scipy/opencv-python/imageio (perception + video), plain `pytest` (perception unit tests), PyTorch.

## Global Constraints

- `rl/perception/` is pure numpy/scipy/cv2 with no Isaac Sim/torch import anywhere in its non-test code — test it with plain `python3 -m pytest rl/perception/tests/ -v` from the repo root, never via `isaaclab.sh` (no sim startup needed, and mixing the two would be slower and wrong).
- All Isaac-Sim-dependent scripts run via the explicit form: `cd /home/saps/projects/6DoF` then `/home/saps/IsaacLab/isaaclab.sh -p <path>` — not the ambiguous `./isaaclab.sh -p <path>` shorthand the two existing scripts currently use (fixed in Task 6). `isaaclab.sh` lives in the separate IsaacLab install, not this repo.
- Every Isaac Sim GUI/headless run takes ~15-45s to start and does not reliably flush trailing `print()` output before shutdown — verify completion via exit code and output-artifact existence, not trailing prints (see `project_ar4_scene_config_lessons` memory).
- Training (`train.py`, `Ar4PickPlaceEnvCfg`/`Ar4PickPlaceSceneCfg`) is never modified to add camera rendering — perception is only ever consumed by `perception_calibration.py`, `eval_loop.py --perception`, and `interactive_demo.py`.
- Ground plane is world Z=0 (`Ar4SceneCfg`'s `GroundPlaneCfg` has no pose override) — treated as a known constant (`GROUND_Z`), not detected at runtime.
- No custom TensorBoard logging code is added. Isaac Lab's manager-based env already logs `Episode_Reward/<term_name>` and `Episode_Termination/<term_name>` (including the success-rate signal, `Episode_Termination/cube_reached_goal`) automatically via `rsl_rl`'s `OnPolicyRunner`; Task 11's README documents these tags.
- When an Isaac-Sim-side API detail (exact tensor shape, attribute name) can't be verified from source alone, the step says so explicitly and gives a concrete fallback to try — print/inspect the real runtime value rather than silently trusting a guess (same approach the prior AR4 RL plan used for the `ee_frame` prim path and `RecordVideo` trigger type).

---

### Task 1: Ground-plane removal + clustering (`perception/segmentation.py`)

**Files:**
- Create: `rl/perception/__init__.py`
- Create: `rl/perception/segmentation.py`
- Create: `rl/perception/tests/__init__.py`
- Create: `rl/perception/tests/conftest.py`
- Create: `rl/perception/tests/test_segmentation.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy).
- Produces: `estimate_ground_depth(depth: np.ndarray) -> float`, `segment_objects(depth: np.ndarray, ground_margin: float = 0.003, min_cluster_pixels: int = 12) -> tuple[np.ndarray, list[int]]` (returns a per-pixel integer label image and the list of valid cluster label ids) — consumed by Task 3's `pipeline.py`.

- [ ] **Step 1: Create the package init and test-path conftest**

```python
# rl/perception/__init__.py
```
(empty file — makes `perception` an importable package)

```python
# rl/perception/tests/__init__.py
```
(empty file)

```python
# rl/perception/tests/conftest.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This inserts `rl/` (the parent of `rl/perception/`) onto `sys.path`, so test files can do `from perception.segmentation import ...` when pytest is run from the repo root — matching this repo's existing convention of scripts inserting `rl/` onto `sys.path` themselves (see `rl/scripts/train.py`), rather than requiring a `PYTHONPATH` env var.

- [ ] **Step 2: Write the failing tests**

```python
# rl/perception/tests/test_segmentation.py
import numpy as np

from perception.segmentation import estimate_ground_depth, segment_objects


def test_estimate_ground_depth_is_dominant_value():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    assert estimate_ground_depth(depth) == 0.5


def test_segment_objects_finds_single_cluster():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    labeled, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 1
    cluster_mask = labeled == valid_ids[0]
    assert np.count_nonzero(cluster_mask) == 16
    assert cluster_mask[2:6, 2:6].all()
    assert not cluster_mask[0:2, :].any()


def test_segment_objects_separates_two_clusters():
    depth = np.full((30, 30), 0.5)
    depth[2:6, 2:6] = 0.4
    depth[20:24, 20:24] = 0.4
    _, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 2


def test_segment_objects_filters_small_noise_clusters():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    depth[10, 10] = 0.4
    _, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 1
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perception.segmentation'` (the module doesn't exist yet).

- [ ] **Step 4: Write the implementation**

```python
# rl/perception/segmentation.py
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_segmentation.py -v`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add rl/perception/__init__.py rl/perception/segmentation.py rl/perception/tests/__init__.py rl/perception/tests/conftest.py rl/perception/tests/test_segmentation.py
git commit -m "Add depth-based ground-plane removal and clustering for AR4 perception"
```

---

### Task 2: Geometric shape classification (`perception/shape_classifier.py`)

**Files:**
- Create: `rl/perception/shape_classifier.py`
- Create: `rl/perception/tests/test_shape_classifier.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy).
- Produces: `classify_shape(points: np.ndarray, ground_z: float = 0.0) -> ShapeClassification` (with `.label` in `{"cube", "rectangular_prism", "sphere", "wedge", "unknown"}`, plus `.height`, `.planarity_residual`, `.tilt_rad`, `.circularity`) — consumed by Task 3's `pipeline.py`.

- [ ] **Step 1: Write the failing tests**

```python
# rl/perception/tests/test_shape_classifier.py
import numpy as np

from perception.shape_classifier import CUBE, RECTANGULAR_PRISM, SPHERE, WEDGE, classify_shape

RNG = np.random.default_rng(0)


def _flat_square_top(height: float, half_extent: float = 0.008, n: int = 200) -> np.ndarray:
    xy = RNG.uniform(-half_extent, half_extent, size=(n, 2))
    z = height + RNG.normal(0.0, 0.0002, size=n)
    return np.column_stack([xy, z])


def test_classifies_short_flat_top_as_cube():
    points = _flat_square_top(height=0.018)
    assert classify_shape(points, ground_z=0.0).label == CUBE


def test_classifies_tall_flat_top_as_rectangular_prism():
    points = _flat_square_top(height=0.030)
    assert classify_shape(points, ground_z=0.0).label == RECTANGULAR_PRISM


def test_classifies_curved_round_cap_as_sphere():
    n = 300
    radius = 0.009
    center_z = radius
    theta = RNG.uniform(0, 2 * np.pi, size=n)
    r = radius * np.sqrt(RNG.uniform(0, 1, size=n)) * 0.9
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = center_z + np.sqrt(np.clip(radius**2 - r**2, 0.0, None))
    points = np.column_stack([x, y, z])
    assert classify_shape(points, ground_z=0.0).label == SPHERE


def test_classifies_tilted_plane_as_wedge():
    n = 200
    xy = RNG.uniform(-0.008, 0.008, size=(n, 2))
    z = 0.015 + xy[:, 0] * np.tan(np.radians(30.0))
    points = np.column_stack([xy, z])
    assert classify_shape(points, ground_z=0.0).label == WEDGE


def test_too_few_points_is_unknown():
    points = np.zeros((2, 3))
    result = classify_shape(points, ground_z=0.0)
    assert result.label == "unknown"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_shape_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Since a single top-down camera only ever sees each object's top-facing surface (not its full 3D shape), classification works from what that surface reveals: how high it sits above the ground (separates cube from rectangular prism — both have almost the same footprint, but the prism's top face is ~12mm higher), whether it's flat and horizontal (cube/prism) vs. tilted (wedge) vs. not well-fit by any single plane at all (sphere's curved cap).

```python
# rl/perception/shape_classifier.py
"""Geometric shape classification from a per-object 3D point cloud.

Classifies a cluster of world-frame points (as produced by segmenting a
single top-down depth image) into one of: cube, rectangular_prism, sphere,
wedge, or unknown (too few points to say). See module docstring reasoning
in docs/superpowers/specs/2026-07-04-ar4-perception-integration-design.md.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Thresholds tuned and verified against this project's actual object sizes
# (rl/tasks/ar4/objects_cfg.py): cube 18mm, rectangular prism 16x16x30mm,
# sphere 9mm radius. See the design doc's "Approaches considered" section -
# revisit these if real camera noise proves them too tight/loose.
HEIGHT_THRESHOLD = 0.024  # m, midpoint between the cube (0.018m) and prism (0.030m) heights
PLANARITY_RESIDUAL_THRESHOLD = 0.0008  # m, above this the top surface reads as curved, not flat
TILT_THRESHOLD_RAD = np.radians(15.0)  # above this a flat surface reads as a slanted (wedge) face
CIRCULARITY_THRESHOLD = 0.7  # footprint roundness (1.0 = circle) above which a curved cap reads as a sphere

UNKNOWN = "unknown"
CUBE = "cube"
RECTANGULAR_PRISM = "rectangular_prism"
SPHERE = "sphere"
WEDGE = "wedge"


@dataclass
class ShapeClassification:
    label: str
    height: float
    planarity_residual: float
    tilt_rad: float
    circularity: float


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fits a plane to `points` (P, 3) via SVD. Returns (unit normal, residual std)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residual = float(np.std(centered @ normal))
    return normal, residual


def _footprint_circularity(points_xy: np.ndarray) -> float:
    """4*pi*area/perimeter^2 of the convex hull of the XY footprint. 1.0 for a circle."""
    if len(points_xy) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        return 0.0
    area = hull.volume  # `volume` is the 2D area for a 2D ConvexHull
    hull_points = points_xy[hull.vertices]
    perimeter = sum(np.linalg.norm(hull_points[i] - hull_points[i - 1]) for i in range(len(hull_points)))
    if perimeter == 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter**2))


def classify_shape(points: np.ndarray, ground_z: float = 0.0) -> ShapeClassification:
    """Classifies a single object's world-frame point cloud (P, 3)."""
    if points.shape[0] < 3:
        return ShapeClassification(UNKNOWN, 0.0, 0.0, 0.0, 0.0)

    height = float(points[:, 2].max() - ground_z)
    normal, residual = _fit_plane(points)
    tilt_rad = float(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
    circularity = _footprint_circularity(points[:, :2])

    if residual > PLANARITY_RESIDUAL_THRESHOLD and circularity > CIRCULARITY_THRESHOLD:
        label = SPHERE
    elif tilt_rad > TILT_THRESHOLD_RAD:
        label = WEDGE
    else:
        label = CUBE if height < HEIGHT_THRESHOLD else RECTANGULAR_PRISM

    return ShapeClassification(label, height, residual, tilt_rad, circularity)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_shape_classifier.py -v`
Expected: `5 passed`.

(These exact thresholds and fixtures were prototyped and verified numerically before writing this plan — all four shapes classify correctly with `PLANARITY_RESIDUAL_THRESHOLD=0.0008`. An earlier attempt at `0.003` misclassified the sphere fixture as a cube because the visible cap's curvature at this small radius produces a smaller residual than a first guess would suggest — a concrete illustration of why the design doc flags real-camera-noise validation as an open risk.)

- [ ] **Step 5: Commit**

```bash
git add rl/perception/shape_classifier.py rl/perception/tests/test_shape_classifier.py
git commit -m "Add geometric shape classifier for AR4 perception"
```

---

### Task 3: Depth-to-world unprojection + pipeline glue (`perception/pipeline.py`)

**Files:**
- Create: `rl/perception/pipeline.py`
- Create: `rl/perception/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment_objects` (Task 1), `classify_shape` (Task 2).
- Produces: `Detection` dataclass (`position: np.ndarray(3,)`, `shape_label: str`, `bbox: tuple[int,int,int,int]`, `pixel_count: int`), `run_perception(depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros, ground_z=0.0, ...) -> list[Detection]`, `build_world_point_grid(...)`, `quat_to_rot_matrix(...)` — consumed by Task 4's `tracker.py` and Task 6's Isaac-Sim-side scripts.

- [ ] **Step 1: Write the failing tests**

```python
# rl/perception/tests/test_pipeline.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# rl/perception/pipeline.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_pipeline.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add rl/perception/pipeline.py rl/perception/tests/test_pipeline.py
git commit -m "Add depth-to-world unprojection and end-to-end perception pipeline"
```

---

### Task 4: Frame-to-frame tracking (`perception/tracker.py`)

**Files:**
- Create: `rl/perception/tracker.py`
- Create: `rl/perception/tests/test_tracker.py`

**Interfaces:**
- Consumes: `Detection` (Task 3).
- Produces: `TrackedObject` dataclass (`track_id`, `position`, `shape_label`, `bbox`, `frames_since_seen`, `.is_stale` property), `ObjectTracker.update(detections: list[Detection]) -> list[TrackedObject]`, `find_by_shape(tracked, shape_label) -> TrackedObject | None` — consumed by Task 8/9/10's Isaac-Sim-side scripts.

- [ ] **Step 1: Write the failing tests**

```python
# rl/perception/tests/test_tracker.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# rl/perception/tracker.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_tracker.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add rl/perception/tracker.py rl/perception/tests/test_tracker.py
git commit -m "Add frame-to-frame object tracker with staleness for AR4 perception"
```

---

### Task 5: Detection overlay drawing (`perception/overlay.py`)

**Files:**
- Create: `rl/perception/overlay.py`
- Create: `rl/perception/tests/test_overlay.py`

**Interfaces:**
- Consumes: `TrackedObject` (Task 4).
- Produces: `draw_detections(frame_rgb: np.ndarray, tracked_objects: list[TrackedObject]) -> np.ndarray` — consumed by Task 8/9/10's Isaac-Sim-side scripts (calibration clip, eval video, demo video).

- [ ] **Step 1: Write the failing tests**

```python
# rl/perception/tests/test_overlay.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# rl/perception/overlay.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/test_overlay.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Run the full perception test suite**

Run: `cd /home/saps/projects/6DoF && python3 -m pytest rl/perception/tests/ -v`
Expected: `25 passed`.

- [ ] **Step 6: Commit**

```bash
git add rl/perception/overlay.py rl/perception/tests/test_overlay.py
git commit -m "Add detection overlay drawing for AR4 perception videos"
```

---

### Task 6: Training scale-up + launch-command docstring fixes

**Files:**
- Modify: `rl/tasks/ar4/pickplace_env_cfg.py:179` (`Ar4PickPlaceEnvCfg.scene` default `num_envs`)
- Modify: `rl/scripts/train.py` (default `--num_envs`, docstring)
- Modify: `rl/scripts/eval_loop.py` (docstring only)

**Interfaces:**
- Consumes: existing `Ar4PickPlaceEnvCfg`, `train.py`, `eval_loop.py` (all already exist and work).
- Produces: no new interfaces — same classes/CLI, just a higher default env count and an unambiguous launch command in every script's docstring, used by Task 11's README.

- [ ] **Step 1: Bump `num_envs` to 4096**

In `rl/tasks/ar4/pickplace_env_cfg.py`, change:
```python
    scene: Ar4PickPlaceSceneCfg = Ar4PickPlaceSceneCfg(num_envs=512, env_spacing=2.5)
```
to:
```python
    scene: Ar4PickPlaceSceneCfg = Ar4PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
```

In `rl/scripts/train.py`, change:
```python
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
```
to:
```python
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
```

- [ ] **Step 2: Fix the launch-command docstrings**

In `rl/scripts/train.py`, change the module docstring's code block from:
```python
.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/train.py --num_envs 512
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    ./isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless
```
to:
```python
.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 4096
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

In `rl/scripts/eval_loop.py`, change:
```python
.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_1500.pt --episodes 10
```
to:
```python
.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_1500.pt --episodes 10
```

(`isaaclab.sh` lives in the separate IsaacLab install at `/home/saps/IsaacLab/`, not this repo — the old `./isaaclab.sh` shorthand silently assumed it was aliased onto the PATH from this repo's root, which it isn't.)

- [ ] **Step 3: Smoke-test the 4096-env VRAM footprint**

```bash
cd /home/saps/projects/6DoF
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 4096 --max_iterations 2 --headless > /tmp/train_smoke_4096.log 2>&1
echo "EXIT:$?"
grep -n "Traceback\|CUDA out of memory" /tmp/train_smoke_4096.log
find rl/logs/train -name "model_*.pt" -newer rl/scripts/train.py
```

Expected: `EXIT:0`, no `Traceback` or `CUDA out of memory` lines, at least one new `model_*.pt` file. If it fails with `CUDA out of memory`, that's a real VRAM constraint on this machine (RTX 5070 Ti, 16GB) — pick a smaller `num_envs` (e.g. 2048) instead of 4096 for both the env cfg default and `train.py`'s default, re-run this smoke test at that size, and note the reduced value in Task 11's README instead of 4096.

- [ ] **Step 4: Commit**

```bash
git add rl/tasks/ar4/pickplace_env_cfg.py rl/scripts/train.py rl/scripts/eval_loop.py
git commit -m "Scale AR4 pick-and-place training to num_envs=4096, fix launch-command docstrings"
```

---

### Task 7: Perception camera scene/env config

**Files:**
- Modify: `rl/tasks/ar4/pickplace_env_cfg.py` (add `Ar4PickPlacePerceptionSceneCfg`, `Ar4PickPlacePerceptionEnvCfg`, `GROUND_Z`, `WORKSPACE_BOUNDS`)

**Interfaces:**
- Consumes: `Ar4PickPlaceSceneCfg`, `Ar4PickPlaceEnvCfg` (existing, this file).
- Produces: `Ar4PickPlacePerceptionEnvCfg` (adds a `perception_camera` sensor, `num_envs=1`), `GROUND_Z: float`, `WORKSPACE_BOUNDS: dict` — consumed by Task 8 (`perception_calibration.py`), Task 9 (`eval_loop.py --perception`), Task 10 (`interactive_demo.py`).

- [ ] **Step 1: Add the camera import and scene/env cfg classes**

In `rl/tasks/ar4/pickplace_env_cfg.py`, change:
```python
from isaaclab.sensors import FrameTransformerCfg
```
to:
```python
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
```

Then, after `Ar4PickPlaceEnvCfg`'s class body (end of the file), append:

```python
# World-frame constants shared by perception-consuming entry points
# (eval_loop.py --perception, interactive_demo.py, perception_calibration.py).
# Ground plane is world Z=0 (Ar4SceneCfg's GroundPlaneCfg has no pose override).
GROUND_Z = 0.0

# A generous box around the objects' known spawn region (x:[-0.2,0.2], y:[0.28,0.34]
# in objects_cfg.py) - wide enough for the interactive demo to tolerate the cube
# being dragged well outside its training-time randomization range, while still
# rejecting positions the policy was never trained anywhere near.
WORKSPACE_BOUNDS = {"x": (-0.30, 0.30), "y": (0.10, 0.45), "z": (0.0, 0.05)}

# Mounted above the workspace center (x=0, y=0.31 - the midpoint of the object
# layout), 0.55m above the ground plane, looking straight down. Under CameraCfg's
# "world" offset convention (forward +X, up +Z), a +90deg rotation about Y maps
# local forward (+X) to world -Z (straight down): quat (cos45, 0, sin45, 0).
# Verified empirically by Task 7 Step 2's smoke test, not just derived on paper.
_PERCEPTION_CAMERA_POS = (0.0, 0.31, 0.55)
_PERCEPTION_CAMERA_QUAT_WORLD = (0.70710678, 0.0, 0.70710678, 0.0)


@configclass
class Ar4PickPlacePerceptionSceneCfg(Ar4PickPlaceSceneCfg):
    """Pick-and-place scene plus a static top-down RGB-D perception camera.

    Only used by eval/demo entry points - never by train.py, since camera
    rendering isn't free even when unused and training stays on privileged
    simulation state (see docs/superpowers/specs/2026-07-04-ar4-perception-integration-design.md).
    """

    perception_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PerceptionCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40.0, clipping_range=(0.2, 1.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_PERCEPTION_CAMERA_POS, rot=_PERCEPTION_CAMERA_QUAT_WORLD, convention="world"),
    )


@configclass
class Ar4PickPlacePerceptionEnvCfg(Ar4PickPlaceEnvCfg):
    """Ar4PickPlaceEnvCfg with the perception camera enabled, num_envs=1
    (eval/demo run one environment at a time in the GUI)."""

    scene: Ar4PickPlacePerceptionSceneCfg = Ar4PickPlacePerceptionSceneCfg(num_envs=1, env_spacing=2.5)
```

- [ ] **Step 2: Smoke-test the camera geometry**

This verifies both that the camera prim resolves and renders, and that it's actually looking straight down by unprojecting its center pixel and checking it lands on the ground directly below the camera (mirroring Task 3's already-passing `test_build_world_point_grid_center_pixel_hits_ground_below_camera` test, now against the real Isaac Sim camera instead of a synthetic one).

```bash
cat > /tmp/smoke_perception_camera.py << 'EOF'
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import sys
import torch

sys.path.insert(0, "/home/saps/projects/6DoF/rl")
from tasks.ar4.pickplace_env_cfg import Ar4PickPlacePerceptionEnvCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from perception.pipeline import build_world_point_grid  # noqa: E402

cfg = Ar4PickPlacePerceptionEnvCfg()
env = ManagerBasedRLEnv(cfg=cfg)

with torch.inference_mode():
    env.reset()
    for _ in range(5):  # let the camera render a few real frames before reading data
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        env.step(actions)

camera = env.scene["perception_camera"]
print("output keys:", list(camera.data.output.keys()))
depth_raw = camera.data.output["distance_to_image_plane"]
rgb_raw = camera.data.output["rgb"]
print("depth tensor shape:", depth_raw.shape, "rgb tensor shape:", rgb_raw.shape)

depth = depth_raw[0, ..., 0].cpu().numpy() if depth_raw.dim() == 4 else depth_raw[0].cpu().numpy()
intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
cam_pos = camera.data.pos_w[0].cpu().numpy()
cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()
print("cam_pos_w:", cam_pos, "cam_quat_w_ros:", cam_quat_ros)

world_grid = build_world_point_grid(depth, intrinsics, cam_pos, cam_quat_ros)
h, w = depth.shape
center = world_grid[h // 2, w // 2]
print("center pixel world xyz:", center, "(expect approx [0.0, 0.31, 0.0])")
print("SMOKE_TEST_OK")
env.close()
simulation_app.close()
EOF
cd /home/saps/projects/6DoF && /home/saps/IsaacLab/isaaclab.sh -p /tmp/smoke_perception_camera.py 2>&1 | tail -30
```

Expected: no `Traceback`, `SMOKE_TEST_OK` printed, and `center pixel world xyz` close to `[0.0, 0.31, 0.0]` (within a few cm — some deviation is expected since the center pixel may land on the cube or another object rather than bare ground, depending on where objects are at that reset).

If `depth_raw.shape`/`rgb_raw.shape` don't match the `[0, ..., 0]`/`[0, ..., :3]` indexing assumed here and in later tasks, adjust the indexing to match what's actually printed (e.g. drop the trailing `, 0]` if the channel dim is already squeezed) and re-run. If the center-pixel world Z is far from 0 (not near the ground) or the reported XY isn't near `(0.0, 0.31)`, the camera isn't actually looking straight down — inspect `camera.data.quat_w_ros` directly and adjust `_PERCEPTION_CAMERA_QUAT_WORLD` until this check passes (the derivation in Step 1's comment should be correct, but this is exactly the kind of geometry that's worth confirming empirically rather than trusting on paper, per this plan's Global Constraints).

```bash
rm /tmp/smoke_perception_camera.py
```

- [ ] **Step 3: Commit**

```bash
git add rl/tasks/ar4/pickplace_env_cfg.py
git commit -m "Add top-down RGB-D perception camera to the AR4 pick-and-place scene"
```

---

### Task 8: Perception calibration clip

**Files:**
- Create: `rl/scripts/perception_calibration.py`

**Interfaces:**
- Consumes: `Ar4PickPlacePerceptionEnvCfg`, `GROUND_Z` (Task 7); `run_perception` (Task 3); `ObjectTracker` (Task 4); `draw_detections` (Task 5).
- Produces: `rl/logs/videos/perception_calibration.mp4`.

- [ ] **Step 1: Write the calibration script**

The robot stays motionless (zero actions, gripper held open, same convention as `drive_joints_demo.py`) while the cube is teleported across the camera's field of view over the clip's duration. Reusing the existing `Ar4PickPlacePerceptionEnvCfg` (rather than building a bespoke robot-free scene) means this exercises the exact same scene/camera/asset code the eval and demo scripts use, and — as a bonus — the other three objects stay visible as a static-object sanity check on the same clip, not just the sliding cube.

```python
# rl/scripts/perception_calibration.py
"""Sanity-check the perception pipeline before trusting it in eval/demo scripts:
slides the cube across the perception camera's field of view for a few seconds
and writes an mp4 with the detected mask/bbox/shape-label burned into each frame.

Not run during training or as part of any automated flow - a one-time (or
re-run-when-something-changes) manual check. The robot is present but held
motionless throughout (Isaac Lab's manager framework needs at least one
action/observation term, and the existing pick-and-place env config already
provides a well-tested one) - only the cube moves.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/perception_calibration.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a perception calibration clip.")
parser.add_argument("--duration", type=float, default=6.0, help="Clip duration in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from perception.overlay import draw_detections  # noqa: E402
from perception.pipeline import run_perception  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlacePerceptionEnvCfg  # noqa: E402

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "perception_calibration.mp4"
)
CUBE_Y = 0.31  # camera center
CUBE_Z = 0.009  # cube's resting half-height, see objects_cfg.py's CUBE_CFG
SLIDE_X_RANGE = (-0.35, 0.35)  # sweeps across the camera's field of view


def main() -> None:
    env_cfg = Ar4PickPlacePerceptionEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    camera = env.scene["perception_camera"]
    tracker = ObjectTracker()

    num_steps = int(args_cli.duration / env.step_dt)
    action_dim = env.action_manager.total_action_dim
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = imageio.get_writer(OUTPUT_PATH, fps=int(1.0 / env.step_dt), codec="libx264")

    with torch.inference_mode():
        env.reset()
        for step in range(num_steps):
            frac = step / max(num_steps - 1, 1)
            cube_x = SLIDE_X_RANGE[0] + frac * (SLIDE_X_RANGE[1] - SLIDE_X_RANGE[0])
            pose = torch.tensor([[cube_x, CUBE_Y, CUBE_Z, 1.0, 0.0, 0.0, 0.0]], device=env.device)
            env.scene["cube"].write_root_pose_to_sim(pose)

            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            actions[:, -1] = 1.0  # keep the gripper open
            env.step(actions)

            depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
            cam_pos = camera.data.pos_w[0].cpu().numpy()
            cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

            detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=GROUND_Z)
            tracked = tracker.update(detections)
            writer.append_data(draw_detections(rgb, tracked))

    writer.close()
    env.close()
    print(f"Calibration clip written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the calibration clip**

```bash
cd /home/saps/projects/6DoF
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/perception_calibration.py --headless > /tmp/perception_calibration.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/perception_calibration.log
ls -la rl/logs/videos/perception_calibration.mp4
```

Expected: `EXIT:0`, no `Traceback`, and `perception_calibration.mp4` exists with a non-trivial size (a few hundred KB to a few MB for a ~6s clip at 480p).

If `camera.data.output["distance_to_image_plane"]`/`["rgb"]` indexing raises a shape error, apply the same fix Task 7 Step 2 already resolved (adjust the trailing index to match the actual tensor rank) here too.

- [ ] **Step 3: Watch the clip**

Play `rl/logs/videos/perception_calibration.mp4` (e.g. `xdg-open` or copy it out) and confirm: the sliding cube is masked, boxed, and labeled `"cube"` throughout its slide (not confused with another shape), and the three static objects are labeled correctly and consistently (not flickering between shapes frame to frame).

- [ ] **Step 4: Commit**

```bash
git add rl/scripts/perception_calibration.py
git commit -m "Add AR4 perception calibration clip script"
```

---

### Task 9: Perception adapter + `eval_loop.py --perception`

**Files:**
- Create: `rl/scripts/_perception_adapter.py`
- Modify: `rl/scripts/eval_loop.py`

**Interfaces:**
- Consumes: `run_perception` (Task 3), `find_by_shape` (Task 4), `Ar4PickPlacePerceptionEnvCfg`, `GROUND_Z` (Task 7).
- Produces: `cube_position_obs_slice(env)`, `perceive_cube(env, camera, tracker, ground_z)` (in `_perception_adapter.py`) — consumed here and by Task 10's `interactive_demo.py`. `eval_loop.py --perception` produces `rl/logs/videos/ar4_pickplace_perception.mp4`.

- [ ] **Step 1: Write the shared perception adapter**

The trained policy expects the cube's position in the robot's root frame (matching `isaaclab_tasks...lift.mdp.object_position_in_robot_root_frame`, used during training) — this uses the same `subtract_frame_transforms` utility to convert perception's world-frame position into that same frame, so the substituted value is directly comparable to what the policy saw during training.

```python
# rl/scripts/_perception_adapter.py
"""Shared helper for eval_loop.py --perception and interactive_demo.py: builds
the trained policy's cube-position observation slot from the real perception
pipeline instead of privileged simulation state.
"""

import numpy as np
import torch

from isaaclab.utils.math import subtract_frame_transforms

from perception.pipeline import run_perception
from perception.tracker import find_by_shape


def cube_position_obs_slice(env) -> tuple[int, int]:
    """Column range of the 'cube_position' term within the concatenated policy
    observation tensor, computed from the observation manager rather than
    hardcoded, so it can't silently drift out of sync with pickplace_env_cfg.py.
    """
    term_names = env.observation_manager.active_terms["policy"]
    term_dims = env.observation_manager.group_obs_term_dim["policy"]
    offset = 0
    for name, dim in zip(term_names, term_dims):
        size = int(np.prod(dim))
        if name == "cube_position":
            return offset, offset + size
        offset += size
    raise ValueError("cube_position term not found in the policy observation group.")


def perceive_cube(env, camera, tracker, ground_z: float):
    """Runs perception on the camera's current frame, updates `tracker`, and
    returns (cube_position_in_robot_root_frame_or_None, tracked_objects, rgb_frame).
    `env` must be the raw ManagerBasedRLEnv (e.g. `env.unwrapped`), not the
    rsl_rl-wrapped env."""
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos = camera.data.pos_w[0].cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

    detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=ground_z)
    tracked = tracker.update(detections)
    cube = find_by_shape(tracked, "cube")
    if cube is None:
        return None, tracked, rgb

    object_pos_w = torch.tensor(cube.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot = env.scene["robot"]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b, tracked, rgb
```

- [ ] **Step 2: Add `--perception` to `eval_loop.py`**

Add the import (near the top, with the other `argparse` setup) and CLI flag:
```python
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help="Use the real camera-based perception pipeline instead of privileged simulation state for the cube's observed position.",
)
```

Add `args_cli.enable_cameras = True` right after `if not os.path.isfile(...)` (it's already set unconditionally today for the existing video path — leave that line as-is; the perception path needs it too).

Add these imports after the existing `sys.path.insert(...)` line:
```python
import imageio  # noqa: E402

from _perception_adapter import cube_position_obs_slice, perceive_cube  # noqa: E402
from perception.overlay import draw_detections  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
```

Change the existing:
```python
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
```
to:
```python
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
```

Replace the body of `main()` with:
```python
def main() -> None:
    env_cfg_cls = Ar4PickPlacePerceptionEnvCfg if args_cli.perception else Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1

    agent_cfg = Ar4PickPlacePPORunnerCfg()

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    tracker = None
    perception_writer = None
    if args_cli.perception:
        tracker = ObjectTracker()
        perception_writer = imageio.get_writer(
            os.path.join(VIDEO_DIR, "ar4_pickplace_perception.mp4"), fps=int(1.0 / env.step_dt), codec="libx264"
        )
    else:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            episode_trigger=lambda episode_id: episode_id < args_cli.episodes,
            video_length=0,  # 0 = record the full episode, not a fixed step count
            name_prefix="ar4_pickplace",
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    camera = env.unwrapped.scene["perception_camera"] if args_cli.perception else None
    cube_slice = cube_position_obs_slice(env.unwrapped) if args_cli.perception else None

    obs = env.get_observations()
    completed_episodes = 0
    with torch.inference_mode():
        while completed_episodes < args_cli.episodes and simulation_app.is_running():
            if args_cli.perception:
                cube_pos_b, tracked, rgb = perceive_cube(env.unwrapped, camera, tracker, GROUND_Z)
                if cube_pos_b is not None:
                    col_start, col_end = cube_slice
                    obs[:, col_start:col_end] = cube_pos_b
                perception_writer.append_data(draw_detections(rgb, tracked))

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]):
                completed_episodes += 1
                print(f"[INFO] Completed episode {completed_episodes}/{args_cli.episodes}")

    if perception_writer is not None:
        perception_writer.close()
    env.close()
    print(f"Videos written to: {VIDEO_DIR}")
```

(The non-`--perception` path is byte-for-byte the same behavior as before this change — only the `if args_cli.perception:` branches are new.)

- [ ] **Step 3: Smoke-test `eval_loop.py --perception` against the Task 6 checkpoint**

```bash
cd /home/saps/projects/6DoF
CKPT=$(find rl/logs/train -name "model_*.pt" | sort | tail -1)
echo "Using checkpoint: $CKPT"
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint "$CKPT" --episodes 1 --perception --headless > /tmp/eval_perception_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/eval_perception_smoke.log
ls -la rl/logs/videos/ar4_pickplace_perception.mp4
```

Expected: `EXIT:0`, no `Traceback`, and `ar4_pickplace_perception.mp4` exists. Since this is only a smoke test against a 2-iteration training checkpoint (not a converged policy), the arm's behavior in the video is expected to look random/untrained — the goal here is confirming the perception-substituted observation path runs end-to-end without error, not that pick-and-place succeeds.

If `cube_position_obs_slice` raises `ValueError: cube_position term not found`, print `env.unwrapped.observation_manager.active_terms["policy"]` to see the actual term names and confirm `"cube_position"` matches exactly (it's defined in this same file's `ObservationsCfg.PolicyCfg`, so it should, but confirm rather than assume).

- [ ] **Step 4: Verify the original (non-perception) path still works**

```bash
cd /home/saps/projects/6DoF
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint "$CKPT" --episodes 1 --headless > /tmp/eval_original_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/eval_original_smoke.log
```

Expected: `EXIT:0`, no `Traceback` — confirms the `--perception` addition didn't break the existing default behavior.

- [ ] **Step 5: Commit**

```bash
git add rl/scripts/_perception_adapter.py rl/scripts/eval_loop.py
git commit -m "Add --perception flag to eval_loop.py, sourcing cube position from real perception"
```

---

### Task 10: Interactive demo

**Files:**
- Modify: `rl/tasks/ar4/pickplace_env_cfg.py` (add `Ar4PickPlaceDemoEnvCfg`)
- Create: `rl/scripts/interactive_demo.py`

**Interfaces:**
- Consumes: `Ar4PickPlacePerceptionEnvCfg`, `GROUND_Z`, `WORKSPACE_BOUNDS` (Task 7); `cube_position_obs_slice`, `perceive_cube` (Task 9); `draw_detections` (Task 5).
- Produces: `rl/logs/videos/ar4_interactive_demo.mp4`.

- [ ] **Step 1: Add a demo-specific env cfg that doesn't reset the cube**

The training/eval env's reset events randomize the cube's position on every episode reset (`reset_cube_position`) — exactly what training needs, but wrong for this demo, where the user (not the randomizer) controls the cube's position between rounds. This variant resets only the robot's joints, never touching the cube (or the other three objects).

In `rl/tasks/ar4/pickplace_env_cfg.py`, add near the other `EventCfg`:
```python
@configclass
class DemoEventCfg:
    """Only resets the robot's joints on episode end - the cube (and other
    objects) are left exactly where they physically are, since the interactive
    demo relies on the user placing the cube by hand rather than training-time
    randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Ar4PickPlaceDemoEnvCfg(Ar4PickPlacePerceptionEnvCfg):
    """Perception-enabled pick-and-place env for the interactive demo."""

    events: DemoEventCfg = DemoEventCfg()
```

- [ ] **Step 2: Write the interactive demo script**

```python
# rl/scripts/interactive_demo.py
"""Interactive AR4 pick-and-place demo: drag the cube anywhere in the Isaac Sim
GUI viewport (native drag gizmo), and once it settles the trained policy picks
it up and places it in the fixed target region on the other side - using the
real camera-based perception pipeline the whole time, exactly as
eval_loop.py --perception does at inference time.

An out-of-view or out-of-the-workspace cube position never triggers an
attempt - the arm just keeps watching and waiting.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint rl/logs/train/<run>/model_1500.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the interactive AR4 pick-and-place demo.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument(
    "--stable_seconds", type=float, default=1.0, help="How long the cube must stay put before the robot acts."
)
parser.add_argument("--stable_tolerance", type=float, default=0.005, help="Max drift (m) still considered 'stable'.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _perception_adapter import cube_position_obs_slice, perceive_cube  # noqa: E402
from perception.overlay import draw_detections  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, WORKSPACE_BOUNDS, Ar4PickPlaceDemoEnvCfg  # noqa: E402

VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "ar4_interactive_demo.mp4"
)


def _in_workspace_bounds(position: np.ndarray) -> bool:
    x, y, z = position
    return (
        WORKSPACE_BOUNDS["x"][0] <= x <= WORKSPACE_BOUNDS["x"][1]
        and WORKSPACE_BOUNDS["y"][0] <= y <= WORKSPACE_BOUNDS["y"][1]
        and WORKSPACE_BOUNDS["z"][0] <= z <= WORKSPACE_BOUNDS["z"][1]
    )


def main() -> None:
    env_cfg = Ar4PickPlaceDemoEnvCfg()
    env_cfg.scene.num_envs = 1

    agent_cfg = Ar4PickPlacePPORunnerCfg()

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    camera = env.unwrapped.scene["perception_camera"]
    cube_slice = cube_position_obs_slice(env.unwrapped)
    tracker = ObjectTracker()
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.unwrapped.step_dt), codec="libx264")

    stable_steps_needed = int(args_cli.stable_seconds / env.unwrapped.step_dt)
    last_stable_position = None
    stable_count = 0

    obs = env.get_observations()
    print("[INFO] Watching for the cube to be placed and settled. Drag it in the viewport.")
    with torch.inference_mode():
        while simulation_app.is_running():
            cube_pos_b, tracked, rgb = perceive_cube(env.unwrapped, camera, tracker, GROUND_Z)
            cube = next((t for t in tracked if t.shape_label == "cube"), None)

            ready_to_act = False
            if cube is not None and not cube.is_stale and _in_workspace_bounds(cube.position):
                if (
                    last_stable_position is not None
                    and np.linalg.norm(cube.position - last_stable_position) <= args_cli.stable_tolerance
                ):
                    stable_count += 1
                else:
                    stable_count = 0
                last_stable_position = cube.position
                ready_to_act = stable_count >= stable_steps_needed
            else:
                stable_count = 0
                last_stable_position = None

            video_writer.append_data(draw_detections(rgb, tracked))

            if not ready_to_act:
                action_dim = env.unwrapped.action_manager.total_action_dim
                actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
                obs, _, _, _ = env.step(actions)
                continue

            print("[INFO] Cube settled - picking it up.")
            stable_count = 0
            last_stable_position = None
            episode_done = False
            for _ in range(env.unwrapped.max_episode_length):
                if cube_pos_b is not None:
                    col_start, col_end = cube_slice
                    obs[:, col_start:col_end] = cube_pos_b
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                cube_pos_b, tracked, rgb = perceive_cube(env.unwrapped, camera, tracker, GROUND_Z)
                video_writer.append_data(draw_detections(rgb, tracked))
                if bool(dones[0]):
                    episode_done = True
                    break
            print(f"[INFO] Pick-and-place {'succeeded' if episode_done else 'timed out'}. Watching for the next drag.")

    video_writer.close()
    env.close()
    print(f"Demo video written to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 3: Smoke-test headlessly for a bounded number of steps**

The demo's main loop runs until the GUI window is closed, so a plain smoke test needs a time-boxed run rather than waiting for manual interaction:

```bash
cd /home/saps/projects/6DoF
CKPT=$(find rl/logs/train -name "model_*.pt" | sort | tail -1)
timeout 60 /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint "$CKPT" --headless > /tmp/demo_smoke.log 2>&1
echo "EXIT:$?"  # 124 = timeout fired (expected/healthy - the loop doesn't exit on its own headlessly)
grep -n "Traceback" /tmp/demo_smoke.log
grep -n "Watching for the cube" /tmp/demo_smoke.log
ls -la rl/logs/videos/ar4_interactive_demo.mp4
```

Expected: exit code `124` (from `timeout`, not a crash), no `Traceback`, the "Watching for the cube..." message printed, and a growing `ar4_interactive_demo.mp4`. Since nothing drags the cube in a headless run, the loop should stay in the idle branch the whole time — this smoke test only confirms the idle/watch loop runs cleanly, not the pick-and-place trigger path (that needs a live GUI session with an actual drag, which is Step 4).

If `env.unwrapped.max_episode_length` raises an `AttributeError`, print `dir(env.unwrapped)` to find the actual attribute name for the episode step budget and substitute it.

- [ ] **Step 4: Manually verify the full interactive flow**

Run without `--headless` (and without `timeout`), so the GUI is visible:

```bash
cd /home/saps/projects/6DoF
CKPT=$(find rl/logs/train -name "model_*.pt" | sort | tail -1)
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint "$CKPT"
```

In the viewport: drag the cube to a new position within the workspace and release it. Confirm: the arm stays idle while dragging, starts moving only after the cube has settled for about a second, correctly picks it up and moves it toward the target region, and returns to idle-watching afterward so it can be dragged again. Then drag the cube somewhere clearly outside `WORKSPACE_BOUNDS` (e.g. far off the table) and confirm the arm never attempts to reach for it.

- [ ] **Step 5: Commit**

```bash
git add rl/tasks/ar4/pickplace_env_cfg.py rl/scripts/interactive_demo.py
git commit -m "Add interactive AR4 pick-and-place demo driven by real perception"
```

---

### Task 11: README

**Files:**
- Create: `rl/README.md`

**Interfaces:**
- Consumes: every script and constant from Tasks 6-10 (exact commands, TensorBoard tag names).
- Produces: nothing consumed by other tasks — this is the end-user-facing walkthrough.

- [ ] **Step 1: Write the README**

```markdown
# rl/ - AR4 Pick-and-Place RL

Everything here runs through Isaac Lab's launcher, not plain `python`. `isaaclab.sh`
lives in the separate IsaacLab install, not this repo - always run from this
repo's root and reference it by absolute path:

```bash
cd /home/saps/projects/6DoF
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/<script>.py [args]
```

## 1. Build the robot/scene assets (one-time)

```bash
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/build_asset.py
```

## 2. Sanity-check perception before trusting it anywhere else

Slides the cube across the camera's view for a few seconds and writes a labeled
mp4 - watch it before running anything else that depends on perception:

```bash
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/perception_calibration.py --headless
```

Check `rl/logs/videos/perception_calibration.mp4`: the sliding cube should be
labeled `"cube"` throughout, and the three static objects (sphere, rectangular
prism, wedge) should be labeled correctly and consistently, without flickering
between shapes frame to frame.

The perception math itself (ground-plane removal, shape classification,
tracking) has its own fast unit test suite, independent of Isaac Sim:

```bash
python3 -m pytest rl/perception/tests/ -v
```

## 3. Train

```bash
# Quick smoke test first (~seconds, confirms the loop runs and writes a checkpoint):
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless

# Full training run:
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 4096 --headless
```

Checkpoints and TensorBoard logs are written to `rl/logs/train/<timestamp>/`.
Watch training with:

```bash
tensorboard --logdir rl/logs/train
```

What to look at:

- `Train/mean_reward` - overall trend; should climb and plateau.
- `Episode_Reward/lifting_cube` - climbing off zero means the policy is at
  least starting to lift the cube, independent of whether it's placing
  accurately yet.
- `Episode_Reward/cube_goal_tracking_fine_grained` - the sharpest signal that
  placement is getting precise, not just "close enough."
- `Episode_Termination/cube_reached_goal` - the success rate: fraction of
  episodes that ended by actually reaching the goal, rather than timing out.
  This is the clearest single "is it working" number - reward can climb from
  partial credit (reaching, lifting) while this stays at zero, which tells you
  the policy is exploring but not yet succeeding.
- `Episode_Termination/time_out` - the complement of the above (episodes that
  ran out the clock without success).

There's no fixed "enough training" iteration count - stop once
`Episode_Termination/cube_reached_goal` has climbed and plateaued, rather than
running to a predetermined number of iterations.

## 4. Evaluate a checkpoint

```bash
# Privileged simulation state (fast, matches how training worked):
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_<iter>.pt --episodes 10

# Real camera-based perception instead:
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_<iter>.pt --episodes 10 --perception
```

Videos are written to `rl/logs/videos/` (`ar4_pickplace-*.mp4` for the default
path, `ar4_pickplace_perception.mp4` for `--perception`, with the detection
overlay burned in for the latter). A healthy result: the cube is reliably
picked up and placed near the target region in most episodes.

## 5. Interactive demo

```bash
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint rl/logs/train/<run>/model_<iter>.pt
```

With the GUI open: drag the cube anywhere in the workspace using the
viewport's drag gizmo, then let go. Once it's settled for about a second, the
arm picks it up and moves it to the target region on the other side, using the
real perception pipeline the whole time (not privileged simulation state) -
including through the brief period where the arm itself blocks the camera's
view of the cube mid-grasp (the tracker holds its last-known position through
that). Drag the cube outside the workspace or camera's view and the arm stays
idle rather than reacting to it.

The session records to `rl/logs/videos/ar4_interactive_demo.mp4` with the
detection overlay burned in, and keeps running (watching for the next drag)
until you close the window.
```

- [ ] **Step 2: Commit**

```bash
git add rl/README.md
git commit -m "Add rl/README.md walkthrough for perception, training, eval, and the interactive demo"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1-5 cover the design's "Perception module" component. Task 6 covers "Training" (scale-up) and the "Script-launch convention fix." Task 7 covers "Camera + scene." Task 8 covers "Perception calibration clip." Task 9 covers "Inference-time perception adapter" and "Eval script (extended)." Task 10 covers "Interactive demo," including the validity-gating and demo-trigger requirements added during the design discussion. Task 11 covers "`rl/README.md`" and "Training diagnostics" (documented, not coded, since Isaac Lab already logs `Episode_Reward`/`Episode_Termination` automatically - confirmed by reading `reward_manager.py`/`termination_manager.py` source directly, not assumed). The design's "Out of scope" items (learned CV model, perception-in-the-loop training, multi-camera, ROS2/real hardware, non-cube interactive targets) are deliberately untouched by any task.
- **Type/interface consistency:** `Detection` (Task 3) is consumed identically by `tracker.py` (Task 4) and the Isaac-Sim-side scripts (Tasks 8-10). `TrackedObject` (Task 4) is consumed identically by `overlay.py` (Task 5) and the same scripts. `Ar4PickPlacePerceptionEnvCfg`/`GROUND_Z`/`WORKSPACE_BOUNDS` (Task 7) are imported identically in Tasks 8, 9, and 10. `cube_position_obs_slice`/`perceive_cube` (Task 9's `_perception_adapter.py`) are reused unchanged by Task 10 rather than reimplemented, per DRY.
- **No placeholders:** every step has complete, real code. The perception package's numeric thresholds and test fixtures (Tasks 1-4) were actually implemented and run against `pytest` while writing this plan (25 tests passing), not just written from memory - the `PLANARITY_RESIDUAL_THRESHOLD` value in Task 2 is the corrected value after an initial guess (`0.003`) was caught misclassifying the sphere fixture. Isaac-Sim-side uncertainty that couldn't be verified the same way (exact `camera.data.output[...]` tensor rank, `max_episode_length` attribute name, the camera's empirically-derived look-down quaternion) is flagged with a concrete fallback in the relevant step rather than silently assumed.

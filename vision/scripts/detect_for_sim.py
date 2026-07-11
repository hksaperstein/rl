"""Gate P (dice-pick demo) perception bridge: run the trained dice detector
on Gate A's saved Isaac Sim camera frame, deproject each detection to a 3D
world-frame position, and measure detection quality against ground truth.

See `.superpowers/sdd/dice-demo-task2-brief.md` for the full task spec and
`.superpowers/sdd/dice-demo-task2-report.md` for the write-up of what was
found running this script.

Runs entirely under `vision/.venv/bin/python` (ultralytics + numpy + PIL
only) - never Isaac's python, never bare python3. No Isaac Sim launch.

.. code-block:: bash

    vision/.venv/bin/python vision/scripts/detect_for_sim.py \\
        --input-dir outputs/dice_demo/gate_a --output-dir outputs/dice_demo/gate_p

KNOWN GATE-A BUG WORKAROUND (see report for full diagnosis): Gate A's
`camera_params.json` has `pos_w=[0,0,0]` and `quat_w_ros` containing NaNs.
Root cause: `scripts/dice_pick_demo.py` calls `sim.reset()` but never
`scene.reset()`, so `Camera._update_poses()` (which IsaacLab only calls once
at sensor `reset()` time when `update_latest_camera_pose=False`, the default)
never runs and the camera pose buffers stay at their zero-initialized
default. This is a real bug in Gate A's script, not something fixable here
without relaunching Isaac Sim (out of scope / forbidden for this task). This
script detects the invalid pose and falls back to the camera pose implied by
`tasks/franka/dice_scene_cfg.py`'s `DICE_CAMERA_POS`/`DICE_CAMERA_QUAT_WORLD`
constants (the exact values the scene config places the camera prim at -
these are plain Python literals, not Isaac Sim output, and not
`gt_dice.json`), converted from IsaacLab's camera "world" convention (local
+X forward, +Z up) to the "ros" convention (local +Z forward, -Y up) that
`perception/pipeline.py`'s unprojection math expects. The conversion below
mirrors `isaaclab.utils.math.convert_camera_frame_orientation_convention`
(see `/home/saps/IsaacLab/source/isaaclab/isaaclab/utils/math.py:1513-1592`)
and was independently validated: the "world"-convention forward axis
(R_world @ [1,0,0]) and the derived "ros"-convention forward axis
(R_ros @ [0,0,1]) were checked to agree to 1e-6, and R_ros was checked to be
a proper rotation matrix (orthonormal, det=1).

GATE G d10 FIX (2026-07-11, "d10 fix pass"): on the Gate G frame
(`outputs/dice_demo/gate_g/`, same dice layout as Gate A, brighter RTX
render), the detector boxed the table's mounting hole at image center as
"d10" (conf 0.52, deprojected world z=-0.0036m - below the table surface,
physically impossible for a die) while missing the real d10 die (small
white die, upper-center in frame) entirely. Two fixes were added to
`detect()`, always applied, in this order:

1. **Geometric plausibility filter** (`z_min`/`z_max`, default
   `[0.0, 0.05]` m): any detection whose deprojected world z falls outside
   this band is rejected - dice rest at 0.002-0.011m (measured, see
   `scripts/dice_pick_demo.py::_DIE_REST_HEIGHT_M`) and the tallest
   plausible surface point (e.g. a d20 vertex pointing up) is ~33mm, so
   0.05m leaves headroom above that while still excluding the table's
   mounting hole (whose recessed geometry makes the depth ray return a
   BELOW-table z). Rejected detections are never silently dropped - they
   are recorded in `meta["geometric_rejections"]` (class, confidence,
   world_pos, bbox, rejection_reason) for audit.
2. **Scene-contract recovery + one-per-class assignment** (`expected_classes`,
   default `("d4", "d8", "d10", "d12", "d20")`; `enable_scene_contract_recovery`,
   default `True`). This is SCENE-SPECIFIC knowledge, not general detector
   behavior: the dice-pick demo scene (`tasks/franka/dice_scene_cfg.py`)
   always places exactly one die of each of these five types on the table -
   never zero, never two of the same type. After the primary + mitigation
   passes and the plausibility filter, if any expected class has no
   plausible detection, `_recovery_sweep` runs additional passes (a
   higher-upscale, `recovery_upscale`=4x by default, crop of the known
   spawn region, plus a full-frame pass, at each rung of
   `recovery_conf_ladder`, default `(0.25, 0.10, 0.05)`), admitting only
   plausibility-passing candidates of the still-missing classes. Every
   attempt (rung, counts, which classes were recovered) is recorded in
   `meta["scene_contract"]["recovery_attempts"]` even when it finds
   nothing. `_one_per_class_resolve` then picks exactly one detection per
   expected class (highest confidence among plausible candidates); if two
   different expected classes' chosen detections sit within
   `colocation_tol_m` (default 0.02m/20mm) of each other in world space -
   i.e. they're almost certainly the same physical die claimed under two
   labels - the lower-confidence one is displaced and its class is
   re-searched via another recovery sweep. A class that still has no
   plausible candidate after all attempts is reported as missing
   (`meta["scene_contract"]["still_missing"]`) - never fabricated.
   Detections of classes NOT in `expected_classes` (e.g. a genuine `d6`
   false positive) pass through this stage untouched, so
   `d6_false_positives` accounting downstream is unaffected.

Both fixes operate purely on already-deprojected world positions and class
labels already available from the primary/mitigation passes (or new passes
run identically to them) - neither reads `gt_dice.json` or any other
ground-truth source; `expected_classes` is scene-contract knowledge (baked
into the demo scene's design, same category as the detector's own class
list), not measured ground truth.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Minimal unprojection math, copied (not imported) from perception/pipeline.py
# to avoid pulling in that package's scipy dependency (perception/pipeline.py
# imports perception/segmentation.py, which imports scipy - not present in
# vision/.venv, and this task's environment must stay vision/.venv-only).
# Source of truth for this math: /home/saps/projects/rl/perception/pipeline.py
# (`quat_to_rot_matrix`, `unproject_depth_image`) - copied verbatim, not
# re-derived, per this task's brief. If perception/pipeline.py's math ever
# changes, re-sync this copy.
# ---------------------------------------------------------------------------


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion. Copied from
    perception/pipeline.py::quat_to_rot_matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def unproject_pixel(u: float, v: float, depth: float, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """Single-pixel version of perception/pipeline.py::unproject_depth_image
    (ROS camera-frame convention: x right, y down, z forward)."""
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.array([x, y, depth])


def camera_point_to_world(point_cam: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray) -> np.ndarray:
    """Same transform as perception/pipeline.py::build_world_point_grid, for
    a single point."""
    rot = quat_to_rot_matrix(cam_quat_w_ros)
    return point_cam @ rot.T + cam_pos_w


def world_point_to_pixel(point_w: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray,
                          intrinsic_matrix: np.ndarray) -> tuple[float, float]:
    """Inverse of camera_point_to_world + pinhole projection - used only to
    draw GT crosses on the overlay image for visual sanity-checking."""
    rot = quat_to_rot_matrix(cam_quat_w_ros)
    point_cam = (point_w - cam_pos_w) @ rot  # rot.T inverse = rot for orthonormal rot, applied as (rot^-1) = rot.T; point_cam = rot.T @ (p - pos) -> as row-vec: (p-pos) @ rot
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    u = point_cam[0] / point_cam[2] * fx + cx
    v = point_cam[1] / point_cam[2] * fy + cy
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Gate-A camera-pose-bug workaround (see module docstring)
# ---------------------------------------------------------------------------

# Source: tasks/franka/dice_scene_cfg.py DICE_CAMERA_POS / DICE_CAMERA_QUAT_WORLD
# (scene-config constants, not Isaac Sim output, not gt_dice.json).
_DICE_CAMERA_POS = (0.5, -0.35353319, 0.45132444)
_DICE_CAMERA_QUAT_WORLD = (0.64085638, -0.29883624, 0.29883624, 0.64085638)  # (w, x, y, z)

# Rotation from ROS-local axes (x right, y down, z forward) to "world"-convention
# local axes (x forward, y left, z up) for the SAME physical camera orientation:
# ros_local_z(fwd) -> world_local_x(fwd); ros_local_x(right) -> -world_local_y(left);
# ros_local_y(down) -> -world_local_z(up). Derived and validated in this task
# (see module docstring) against isaaclab.utils.math.convert_camera_frame_orientation_convention.
_ROS_TO_WORLDCONV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _fallback_camera_pose() -> tuple[np.ndarray, np.ndarray]:
    """Returns (cam_pos_w, cam_quat_w_ros) computed from the scene-config
    camera placement constants, working around Gate A's broken
    camera_params.json pose fields (see module docstring)."""

    def rot_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
        tr = np.trace(rot)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    r_world = quat_to_rot_matrix(np.array(_DICE_CAMERA_QUAT_WORLD))
    r_ros = r_world @ _ROS_TO_WORLDCONV
    quat_ros = rot_matrix_to_quat(r_ros)
    return np.array(_DICE_CAMERA_POS), quat_ros


def _load_camera_pose(camera_params: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (cam_pos_w, cam_quat_w_ros, used_fallback)."""
    pos_w = np.array(camera_params["pos_w"], dtype=np.float64)
    quat_w_ros = np.array(camera_params["quat_w_ros"], dtype=np.float64)
    invalid = np.any(np.isnan(quat_w_ros)) or np.allclose(pos_w, 0.0) or np.allclose(quat_w_ros, 0.0)
    if invalid:
        pos_w, quat_w_ros = _fallback_camera_pose()
        return pos_w, quat_w_ros, True
    return pos_w, quat_w_ros, False


# ---------------------------------------------------------------------------
# Detection core (importable - Gate G will call `detect()` directly)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "vision", "models", "runs", "s_plus_r", "weights", "best.pt")
D10_ALIASES = {"d10", "d10_pct"}
EXPECTED_SCENE_CLASSES = ("d4", "d8", "d10", "d12", "d20")


def _canon_class(cls_name: str) -> str:
    """Maps a detector class name to its scene-contract class name (folds
    d10_pct into d10 - same physical die, see D10_ALIASES)."""
    return "d10" if cls_name in D10_ALIASES else cls_name


def _median_patch_depth(depth: np.ndarray, u: int, v: int, half: int = 2) -> float | None:
    h, w = depth.shape
    r0, r1 = max(0, v - half), min(h, v + half + 1)
    c0, c1 = max(0, u - half), min(w, u + half + 1)
    patch = depth[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _boxes_to_detections(
    boxes,
    names: dict,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    """Converts ultralytics `Boxes` (possibly detected on a cropped+upscaled
    copy of the original image, per `x_offset`/`y_offset`/`scale`) into
    detection dicts in ORIGINAL image pixel space, deprojected using the
    ORIGINAL (non-upscaled) `depth` array - upscaling is a detector-input
    trick only, never applied to the depth/geometry math."""
    detections = []
    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        cls_name = names[cls_idx]
        confidence = float(boxes.conf[i].item())
        xyxy_crop = boxes.xyxy[i].cpu().numpy()
        xyxy = [
            x_offset + xyxy_crop[0] / scale,
            y_offset + xyxy_crop[1] / scale,
            x_offset + xyxy_crop[2] / scale,
            y_offset + xyxy_crop[3] / scale,
        ]
        u = int(round((xyxy[0] + xyxy[2]) / 2))
        v = int(round((xyxy[1] + xyxy[3]) / 2))

        depth_val = _median_patch_depth(depth, u, v)
        if depth_val is None:
            world_pos = None
        else:
            point_cam = unproject_pixel(u, v, depth_val, intrinsic_matrix)
            world_pos = camera_point_to_world(point_cam, cam_pos_w, cam_quat_w_ros).tolist()

        detections.append(
            {
                "class": cls_name,
                "is_d10_alias": cls_name in D10_ALIASES,
                "confidence": confidence,
                "bbox_xyxy": [float(coord) for coord in xyxy],
                "center_px": [u, v],
                "depth_m": depth_val,
                "world_pos": world_pos,
            }
        )
    return detections


def _deduplicate_detections(primary_dets: list[dict], mitigation_dets: list[dict],
                             spatial_tol: float = 0.10) -> list[dict]:
    """Merge detections from primary and mitigation passes, removing
    spatial duplicates (same/compatible class, <spatial_tol apart in world
    coords). Keeps the higher-confidence detection of each duplicate pair.

    Compatible classes: d10 and d10_pct are considered the same die.

    Note on spatial_tol=0.10m: assumes one die per class. The scene's
    min spawn spacing is ~90mm, so 100mm tolerance exceeds it; in a
    future multi-set scene with multiple dice of the same type, this
    threshold would incorrectly merge nearby same-class dice as duplicates."""
    all_dets = primary_dets + mitigation_dets
    if not all_dets:
        return []

    # Sort by confidence descending, so when we encounter a detection,
    # any prior (higher-conf) detection of the same die is already in merged
    all_dets_sorted = sorted(all_dets, key=lambda d: d["confidence"], reverse=True)

    def classes_compatible(cls1: str, cls2: str) -> bool:
        """Check if two classes represent the same die (d10 and d10_pct are aliases)."""
        if cls1 == cls2:
            return True
        # d10 and d10_pct are the same die type
        if {cls1, cls2} == {"d10", "d10_pct"}:
            return True
        return False

    merged = []
    for det in all_dets_sorted:
        # Check if this detection is a duplicate of one already in merged
        is_duplicate = False
        if det["world_pos"] is not None:
            det_pos = np.array(det["world_pos"])
            for existing_det in merged:
                # Duplicates: compatible class + spatially close
                if (existing_det["world_pos"] is not None and
                    classes_compatible(existing_det["class"], det["class"])):
                    existing_pos = np.array(existing_det["world_pos"])
                    distance = float(np.linalg.norm(det_pos - existing_pos))
                    if distance < spatial_tol:
                        # This detection is a duplicate of an already-merged one
                        is_duplicate = True
                        break

        if not is_duplicate:
            merged.append(det)

    return merged


def _project_scene_region_to_image_bbox(
    region_bounds: tuple[tuple[float, float], tuple[float, float]],
    table_z: float,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    intrinsic_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Projects the 3D scene spawn region (XY bounds, fixed table Z) to
    image-space bbox. Region format: ((x_min, x_max), (y_min, y_max)).
    Returns (x0, y0, x1, y1) in image coordinates, or None if the region
    projects entirely outside the visible image."""
    (x_min, x_max), (y_min, y_max) = region_bounds

    # 4 corners of the scene region at table height
    corners_3d = [
        np.array([x_min, y_min, table_z]),
        np.array([x_max, y_min, table_z]),
        np.array([x_min, y_max, table_z]),
        np.array([x_max, y_max, table_z]),
    ]

    projected_pixels = []
    for corner in corners_3d:
        u, v = world_point_to_pixel(corner, cam_pos_w, cam_quat_w_ros, intrinsic_matrix)
        # Clamp to image bounds (may project outside frame at the edges)
        u = max(0.0, min(float(image_width - 1), u))
        v = max(0.0, min(float(image_height - 1), v))
        projected_pixels.append((u, v))

    if not projected_pixels:
        return None

    us = [p[0] for p in projected_pixels]
    vs = [p[1] for p in projected_pixels]
    x0 = int(min(us))
    x1 = int(max(us))
    y0 = int(min(vs))
    y1 = int(max(vs))

    # Sanity check: is the bbox non-empty and in-frame?
    if x0 >= image_width or x1 < 0 or y0 >= image_height or y1 < 0:
        return None
    if x0 == x1 or y0 == y1:
        # Degenerate projection (zero-area region)
        return None

    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Fix 1: geometric plausibility filter (see module docstring, "GATE G d10 FIX")
# ---------------------------------------------------------------------------


def _apply_plausibility_filter(
    dets: list[dict], z_min: float, z_max: float
) -> tuple[list[dict], list[dict]]:
    """Splits detections into (plausible, rejected) by deprojected world z.
    A detection with `world_pos is None` (invalid/missing depth) is passed
    through as plausible unchanged - there's no z to judge, and this
    matches prior behavior (such detections were already excluded from GT
    matching downstream). Rejected detections are recorded, not dropped
    silently - each rejected dict is a copy of the original detection plus
    a human-readable `rejection_reason`."""
    plausible, rejected = [], []
    for det in dets:
        world_pos = det.get("world_pos")
        if world_pos is None:
            plausible.append(det)
            continue
        z = world_pos[2]
        if z_min <= z <= z_max:
            plausible.append(det)
        else:
            rejected.append(
                {
                    "class": det["class"],
                    "is_d10_alias": det["is_d10_alias"],
                    "confidence": det["confidence"],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "center_px": det["center_px"],
                    "world_pos": world_pos,
                    "rejection_reason": (
                        f"deprojected world z={z:.4f}m outside plausible die-resting "
                        f"band [{z_min:.3f}, {z_max:.3f}]m - physically impossible for "
                        f"a die resting on the table (likely a depth-ray hit on a "
                        f"recessed/raised table feature, e.g. a mounting hole)"
                    ),
                }
            )
    return plausible, rejected


# ---------------------------------------------------------------------------
# Fix 2: scene-contract recovery + one-per-class assignment (see module
# docstring, "GATE G d10 FIX"). SCENE-SPECIFIC knowledge: the dice-pick demo
# scene always contains exactly one die of each of EXPECTED_SCENE_CLASSES.
# ---------------------------------------------------------------------------


def _one_per_class_resolve(
    pool: list[dict], expected_classes: tuple[str, ...], colocation_tol_m: float
) -> tuple[dict[str, dict], list[dict]]:
    """From a pool of plausible detections (possibly multiple per class,
    possibly classes with none), resolves to at most one detection per
    expected class: highest confidence wins within a class. Then checks for
    cross-class collisions - if two different expected classes' chosen
    detections are within `colocation_tol_m` of each other in world space,
    they're almost certainly claims on the same physical die under two
    labels; the lower-confidence one is displaced (dropped from `resolved`,
    its class becomes unresolved again) so the caller can re-search for it.
    Returns (resolved: {class -> det}, displaced: [det, ...])."""
    buckets: dict[str, list[dict]] = {c: [] for c in expected_classes}
    for det in pool:
        c = _canon_class(det["class"])
        if c in buckets:
            buckets[c].append(det)

    resolved: dict[str, dict] = {}
    for c, cands in buckets.items():
        if cands:
            resolved[c] = max(cands, key=lambda d: d["confidence"])

    displaced: list[dict] = []
    # Bounded loop: at most one displacement per expected class can ever
    # happen (each displacement permanently removes a class from `resolved`
    # or replaces it with a non-colocated candidate), so len(expected_classes)
    # iterations is a safe upper bound - never actually infinite.
    for _ in range(len(expected_classes)):
        conflict = None
        classes_with_pos = [c for c in resolved if resolved[c]["world_pos"] is not None]
        for i in range(len(classes_with_pos)):
            for j in range(i + 1, len(classes_with_pos)):
                c1, c2 = classes_with_pos[i], classes_with_pos[j]
                p1 = np.array(resolved[c1]["world_pos"])
                p2 = np.array(resolved[c2]["world_pos"])
                if float(np.linalg.norm(p1 - p2)) < colocation_tol_m:
                    conflict = (c1, c2)
                    break
            if conflict:
                break
        if conflict is None:
            break
        c1, c2 = conflict
        loser_c = c1 if resolved[c1]["confidence"] < resolved[c2]["confidence"] else c2
        loser_det = resolved.pop(loser_c)
        displaced.append(loser_det)
        remaining = [d for d in buckets[loser_c] if d is not loser_det]
        if remaining:
            resolved[loser_c] = max(remaining, key=lambda d: d["confidence"])
        # else loser_c has no other candidate - stays unresolved, caller re-searches

    return resolved, displaced


def _tight_recrop_detect(
    input_dir: str,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    model,
    bbox_xyxy: list[float],
    conf: float,
    pad_px: int = 50,
    scale: int = 6,
) -> list[dict]:
    """Crops the ORIGINAL rgb image tightly around a single candidate's own
    bbox (+padding) and upscales heavily, then re-runs the detector on just
    that patch. Distinct from `_crop_upscale_detect`'s whole-scene-region
    crop: empirically (see "GATE G d10 FIX" in the module docstring),
    upscaling the whole ~390x390px scene region does NOT change a small
    die's classification even at 8x, because ultralytics' `predict()`
    letterbox-resizes any input to the network's fixed inference size
    (e.g. 640x640) before inference - so what matters is what FRACTION of
    that fixed input the die occupies, not the crop's raw pixel count.
    A tight recrop directly around a specific candidate's own small bbox
    makes the die occupy a much larger fraction of the network's input,
    closer to the detector's training distribution - this is what actually
    flips a misclassified low-confidence box (e.g. a real d10 die
    classified `d8` at conf 0.06 in the whole-region crop) to its correct
    class at usable confidence (measured: same die, same pixels, conf 0.57
    as `d10` once tightly recropped)."""
    rgb_path = os.path.join(input_dir, "rgb.png")
    img = Image.open(rgb_path).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, int(x0 - pad_px))
    y0 = max(0, int(y0 - pad_px))
    x1 = min(w, int(x1 + pad_px))
    y1 = min(h, int(y1 + pad_px))
    if x1 <= x0 or y1 <= y0:
        return []
    crop = img.crop((x0, y0, x1, y1))
    crop_up = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    result = model.predict(crop_up, conf=conf, verbose=False)[0]
    return _boxes_to_detections(
        result.boxes, result.names, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros,
        x_offset=x0, y_offset=y0, scale=float(scale),
    )


def _recovery_sweep(
    input_dir: str,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    model,
    region_bounds: tuple[tuple[float, float], tuple[float, float]],
    table_z: float,
    missing_classes: set[str],
    conf_ladder: tuple[float, ...],
    upscale: int,
    z_min: float,
    z_max: float,
    resolved_positions: list[np.ndarray],
    colocation_tol_m: float,
    max_recrop_candidates: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Progressively lowers the confidence threshold across `conf_ladder`.
    At each rung: (1) a higher-upscale crop pass over the known scene spawn
    region (`_crop_upscale_detect` at `upscale`, higher than the standard
    mitigation pass's 2x) plus a full-frame pass, admitting any
    plausibility-passing (z-band) candidate whose class already matches a
    still-missing class; (2) a targeted tight-recrop pass
    (`_tight_recrop_detect`) around every OTHER plausibility-passing
    candidate from step 1 that isn't already claimed by a resolved class
    (within `colocation_tol_m`) - up to `max_recrop_candidates`, highest-
    confidence first - re-detecting just that small patch to see whether
    the detector's classification flips to a still-missing class once the
    candidate fills more of the network's fixed input (see
    `_tight_recrop_detect`'s docstring for why this is necessary, not
    optional: the region-wide crop alone does not recover a real,
    misclassified-at-low-confidence die). Stops early once every missing
    class has a candidate. Returns (recovered_detections, attempts) -
    `attempts` records every rung tried (even ones that find nothing), for
    honest reporting: this project never silently reports a class as found
    without a paper trail, and never fabricates a detection for a class
    that's never actually found."""
    still_missing = set(missing_classes)
    recovered: list[dict] = []
    attempts: list[dict] = []
    rgb_path = os.path.join(input_dir, "rgb.png")

    def _is_plausible(det: dict) -> bool:
        wp = det.get("world_pos")
        return wp is not None and z_min <= wp[2] <= z_max

    def _near_resolved(det: dict) -> bool:
        wp = det.get("world_pos")
        if wp is None:
            return False
        p = np.array(wp)
        return any(float(np.linalg.norm(p - rp)) < colocation_tol_m for rp in resolved_positions)

    for conf_thresh in conf_ladder:
        if not still_missing:
            break

        crop_dets, _crop_meta = _crop_upscale_detect(
            input_dir, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros, model,
            region_bounds, table_z=table_z, conf=conf_thresh, scale=upscale,
        )
        full_result = model.predict(rgb_path, conf=conf_thresh, verbose=False)[0]
        full_dets = _boxes_to_detections(
            full_result.boxes, full_result.names, depth, intrinsic_matrix,
            cam_pos_w, cam_quat_w_ros,
        )
        candidates = crop_dets + full_dets

        found_this_rung: dict[str, dict] = {}
        leftover_candidates: list[dict] = []
        for det in candidates:
            c = _canon_class(det["class"])
            plausible = _is_plausible(det)
            if plausible and c in still_missing:
                if c not in found_this_rung or det["confidence"] > found_this_rung[c]["confidence"]:
                    found_this_rung[c] = det
            elif plausible and not _near_resolved(det):
                leftover_candidates.append(det)

        # Step 2: tight recrop around the highest-confidence leftover
        # candidates not yet explained by a resolved or missing-class match.
        recrop_attempts = 0
        for det in sorted(leftover_candidates, key=lambda d: d["confidence"], reverse=True):
            if not still_missing or recrop_attempts >= max_recrop_candidates:
                break
            recrop_attempts += 1
            recrop_dets = _tight_recrop_detect(
                input_dir, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros, model,
                det["bbox_xyxy"], conf=conf_thresh,
            )
            for rdet in recrop_dets:
                rc = _canon_class(rdet["class"])
                if rc in still_missing and _is_plausible(rdet):
                    if rc not in found_this_rung or rdet["confidence"] > found_this_rung[rc]["confidence"]:
                        found_this_rung[rc] = rdet

        attempts.append(
            {
                "conf_threshold": conf_thresh,
                "upscale": upscale,
                "missing_before": sorted(still_missing),
                "candidates_considered": len(candidates),
                "leftover_candidates_recropped": recrop_attempts,
                "recovered_classes": sorted(found_this_rung.keys()),
            }
        )

        for c, det in found_this_rung.items():
            recovered.append(det)
            still_missing.discard(c)

    return recovered, attempts


def detect(
    input_dir: str,
    conf: float = 0.25,
    weights: str = DEFAULT_WEIGHTS,
    region_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    table_z: float = 0.01,
    model=None,
    z_min: float = 0.0,
    z_max: float = 0.05,
    expected_classes: tuple[str, ...] = EXPECTED_SCENE_CLASSES,
    enable_scene_contract_recovery: bool = True,
    recovery_conf_ladder: tuple[float, ...] = (0.25, 0.10, 0.05),
    recovery_upscale: int = 4,
    colocation_tol_m: float = 0.02,
) -> tuple[list[dict], dict]:
    """Runs the FULL detection pipeline: primary pass on full frame, then
    mitigation (crop the known spawn region and upscale), then deduplicates,
    then applies a geometric plausibility filter, then (if enabled) a
    scene-contract recovery sweep + one-per-class assignment. See the
    module docstring's "GATE G d10 FIX" section for the full rationale -
    this pipeline was extended to fix a reproduced failure mode where the
    detector boxed the table's mounting hole as "d10" while missing the
    real d10 die. Gate G will call this and must never get a degraded
    result - the mitigation, plausibility filter, and (by default) scene-
    contract recovery are ALWAYS applied.

    Returns (detections, meta) where detections is the final list: one
    physically-plausible detection per expected class (when
    `enable_scene_contract_recovery` is True) plus any detections of
    classes outside `expected_classes` (e.g. a genuine d6 false positive),
    unchanged from the merged+filtered pool.

    Does NOT read gt_dice.json and does NOT write any files (importable
    core for Gate G). `expected_classes` is scene-contract knowledge (this
    demo scene always has exactly one die per type), not ground truth.

    Each detection dict: {class, is_d10_alias, confidence, bbox_xyxy,
    center_px, depth_m, world_pos}. world_pos is [x, y, z] in meters in
    the world frame (or None if depth invalid); z is on the visible surface,
    not the die's volumetric center, so expect ~surface-offset error
    (up to ~die-radius, ≤~16mm). `meta` includes the camera pose, pass
    counts, region used, mitigation crop details, geometric rejections, and
    scene-contract recovery attempts, for audit.
    """
    from ultralytics import YOLO  # local import: keep module importable without ultralytics for pure math reuse

    rgb_path = os.path.join(input_dir, "rgb.png")
    depth_path = os.path.join(input_dir, "depth.npy")
    params_path = os.path.join(input_dir, "camera_params.json")
    for p in (rgb_path, depth_path, params_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Gate P input missing: {p}")

    depth = np.load(depth_path)
    with open(params_path) as f:
        camera_params = json.load(f)
    intrinsic_matrix = np.array(camera_params["intrinsic_matrix"], dtype=np.float64)
    cam_pos_w, cam_quat_w_ros, used_fallback = _load_camera_pose(camera_params)

    if model is None:
        model = YOLO(weights)

    if region_bounds is None:
        # Default: the scene's known spawn region (from dice_pick_demo.py)
        region_bounds = ((0.40, 0.60), (-0.15, 0.15))

    # -- Primary pass: full frame, default confidence
    primary_result = model.predict(rgb_path, conf=conf, verbose=False)[0]
    primary_dets = _boxes_to_detections(
        primary_result.boxes, primary_result.names, depth, intrinsic_matrix,
        cam_pos_w, cam_quat_w_ros
    )

    # -- Mitigation: crop the scene's known spawn region + upscale
    mitigation_dets, mitigation_meta = _crop_upscale_detect(
        input_dir, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros, model,
        region_bounds, table_z=table_z, conf=conf
    )

    # -- Deduplication: merge primary + mitigation, remove spatial duplicates
    merged_dets = _deduplicate_detections(primary_dets, mitigation_dets)

    # -- Fix 1: geometric plausibility filter (always applied - not gated by
    # enable_scene_contract_recovery, since a die can't physically sit below
    # the table regardless of scene-contract assumptions).
    plausible_dets, geometric_rejections = _apply_plausibility_filter(merged_dets, z_min, z_max)

    scene_contract_meta = {
        "enabled": enable_scene_contract_recovery,
        "expected_classes": list(expected_classes),
        "initially_missing": [],
        "recovery_attempts": [],
        "displaced_by_colocation": [],
        "still_missing": [],
    }

    if not enable_scene_contract_recovery:
        final_dets = plausible_dets
    else:
        # -- Fix 2: scene-contract recovery + one-per-class assignment
        non_expected_pool = [d for d in plausible_dets if _canon_class(d["class"]) not in expected_classes]
        expected_pool = [d for d in plausible_dets if _canon_class(d["class"]) in expected_classes]

        resolved, displaced = _one_per_class_resolve(expected_pool, expected_classes, colocation_tol_m)
        initially_missing = [c for c in expected_classes if c not in resolved]
        scene_contract_meta["initially_missing"] = initially_missing

        all_displaced = list(displaced)
        missing = [c for c in expected_classes if c not in resolved]

        # Up to 2 recovery rounds: round 1 covers classes missing from the
        # start; round 2 covers any class newly displaced by a colocation
        # conflict inside round 1's re-resolve. Bounded, not unbounded
        # recursion - a colocation conflict discovered in round 2 is
        # reported as still_missing rather than chased indefinitely.
        for _round in range(2):
            if not missing:
                break
            resolved_positions = [
                np.array(d["world_pos"]) for d in resolved.values() if d["world_pos"] is not None
            ]
            recovered, attempts = _recovery_sweep(
                input_dir, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros, model,
                region_bounds, table_z, set(missing), recovery_conf_ladder, recovery_upscale,
                z_min, z_max, resolved_positions, colocation_tol_m,
            )
            scene_contract_meta["recovery_attempts"].extend(attempts)
            if recovered:
                expected_pool = expected_pool + recovered
                resolved, displaced = _one_per_class_resolve(expected_pool, expected_classes, colocation_tol_m)
                all_displaced.extend(displaced)
            missing = [c for c in expected_classes if c not in resolved]

        scene_contract_meta["displaced_by_colocation"] = [
            {
                "class": d["class"],
                "confidence": d["confidence"],
                "world_pos": d["world_pos"],
            }
            for d in all_displaced
        ]
        scene_contract_meta["still_missing"] = missing

        final_dets = non_expected_pool + sorted(resolved.values(), key=lambda d: d["confidence"], reverse=True)

    meta = {
        "cam_pos_w_used": cam_pos_w.tolist(),
        "cam_quat_w_ros_used": cam_quat_w_ros.tolist(),
        "used_gate_a_pose_fallback": used_fallback,
        "conf_threshold": conf,
        "weights": str(weights),
        "region_bounds": list(region_bounds),
        "table_z": table_z,
        "primary_detections_count": len(primary_dets),
        "mitigation_detections_count": len(mitigation_dets),
        "merged_detections_count": len(merged_dets),
        "final_merged_count": len(final_dets),
        "mitigation_crop_region_xyz": mitigation_meta["crop_region_xyz"],
        "mitigation_crop_xyxy": mitigation_meta["crop_xyxy"],
        "mitigation_upscale": mitigation_meta["upscale"],
        "z_band": [z_min, z_max],
        "geometric_rejections": geometric_rejections,
        "scene_contract": scene_contract_meta,
    }
    return final_dets, meta


def _crop_upscale_detect(
    input_dir: str,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    model,
    region_bounds: tuple[tuple[float, float], tuple[float, float]],
    table_z: float = 0.01,
    conf: float = 0.25,
    pad_px: int = 40,
    scale: int = 2,
) -> tuple[list[dict], dict]:
    """Mitigation: derive crop region from the scene's known spawn bounds
    (not from the detector's own output), upscale 2x, and re-run the detector.
    This removes the blind spot where a die entirely outside the primary
    detections' hull couldn't be recovered.

    Region_bounds: ((x_min, x_max), (y_min, y_max)). Projects through the
    camera pose to image space, applies padding, crops, upscales, detects,
    and maps results back to original frame coordinates."""
    rgb_path = os.path.join(input_dir, "rgb.png")
    img = Image.open(rgb_path).convert("RGB")
    w, h = img.size

    # Project the scene region to image space
    bbox = _project_scene_region_to_image_bbox(
        region_bounds, table_z, cam_pos_w, cam_quat_w_ros, intrinsic_matrix, w, h
    )
    if bbox is None:
        # Region projects entirely out of frame - fall back to full image
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = bbox
        # Apply padding
        x0 = max(0, int(x0 - pad_px))
        y0 = max(0, int(y0 - pad_px))
        x1 = min(w, int(x1 + pad_px))
        y1 = min(h, int(y1 + pad_px))

    # Crop, upscale, detect
    crop = img.crop((x0, y0, x1, y1))
    crop_up = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)

    result = model.predict(crop_up, conf=conf, verbose=False)[0]
    detections = _boxes_to_detections(
        result.boxes, result.names, depth, intrinsic_matrix, cam_pos_w, cam_quat_w_ros,
        x_offset=x0, y_offset=y0, scale=float(scale),
    )
    meta = {
        "crop_region_xyz": [list(region_bounds[0]), list(region_bounds[1]), table_z],
        "crop_xyxy": [x0, y0, x1, y1],
        "upscale": scale,
        "conf_threshold": conf,
    }
    return detections, meta


# ---------------------------------------------------------------------------
# GT comparison (the ONLY place gt_dice.json is read) + reporting outputs
# ---------------------------------------------------------------------------


def _match_to_gt(detections: list[dict], gt_dice: dict) -> list[dict]:
    """3D nearest-neighbor greedy matching of detections (with a valid
    world_pos) to GT dice, closest-distance-first, each GT die claimed at
    most once."""
    gt_items = list(gt_dice.items())  # [(die_type, [x,y,z]), ...]
    candidates = []
    for det_idx, det in enumerate(detections):
        if det["world_pos"] is None:
            continue
        wp = np.array(det["world_pos"])
        for gt_idx, (die_type, gt_pos) in enumerate(gt_items):
            dist = float(np.linalg.norm(wp - np.array(gt_pos)))
            candidates.append((dist, det_idx, gt_idx))
    candidates.sort(key=lambda c: c[0])

    matched_det = set()
    matched_gt = set()
    matches = []
    for dist, det_idx, gt_idx in candidates:
        if det_idx in matched_det or gt_idx in matched_gt:
            continue
        matched_det.add(det_idx)
        matched_gt.add(gt_idx)
        die_type, gt_pos = gt_items[gt_idx]
        matches.append({"det_idx": det_idx, "die_type": die_type, "gt_pos": gt_pos, "distance_m": dist})

    unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
    unmatched_gt = [gt_items[i][0] for i in range(len(gt_items)) if i not in matched_gt]
    return matches, unmatched_dets, unmatched_gt


def _build_table(detections: list[dict], matches: list[dict], unmatched_dets: list[int],
                  unmatched_gt: list[str], pos_tol: float = 0.03) -> list[dict]:
    rows = []
    for m in matches:
        det = detections[m["det_idx"]]
        die_type = m["die_type"]
        class_ok = det["class"] == die_type or (die_type in D10_ALIASES and det["is_d10_alias"])
        pos_ok = m["distance_m"] <= pos_tol
        verdict = "PASS" if (class_ok and pos_ok) else ("CLASS_MISMATCH" if not class_ok else "POSITION_ERROR")
        rows.append(
            {
                "die_type": die_type,
                "detected_class": det["class"],
                "confidence": det["confidence"],
                "deprojected_world_pos": det["world_pos"],
                "gt_world_pos": m["gt_pos"],
                "position_error_m": m["distance_m"],
                "verdict": verdict,
            }
        )
    for i in unmatched_dets:
        det = detections[i]
        rows.append(
            {
                "die_type": None,
                "detected_class": det["class"],
                "confidence": det["confidence"],
                "deprojected_world_pos": det["world_pos"],
                "gt_world_pos": None,
                "position_error_m": None,
                "verdict": "UNMATCHED_DETECTION (false positive or depth-invalid)",
            }
        )
    for die_type in unmatched_gt:
        rows.append(
            {
                "die_type": die_type,
                "detected_class": None,
                "confidence": None,
                "deprojected_world_pos": None,
                "gt_world_pos": None,
                "position_error_m": None,
                "verdict": "MISSED (no detection matched)",
            }
        )
    return rows


def _evaluate(detections: list[dict], gt_dice: dict, pos_tol: float = 0.03) -> dict:
    """Runs the full GT comparison for one candidate detection set and
    returns everything needed to report + decide pass/fail."""
    matches, unmatched_dets, unmatched_gt = _match_to_gt(detections, gt_dice)
    table = _build_table(detections, matches, unmatched_dets, unmatched_gt, pos_tol=pos_tol)
    n_pass = sum(1 for r in table if r["verdict"] == "PASS")
    d6_false_positives = [d for d in detections if d["class"] == "d6"]
    all_pass = (
        n_pass == len(gt_dice)
        and len(unmatched_gt) == 0
        and len(unmatched_dets) == 0
        and len(d6_false_positives) == 0
    )
    return {
        "detections": detections,
        "comparison_table": table,
        "n_pass": n_pass,
        "n_gt_dice": len(gt_dice),
        "unmatched_detections": unmatched_dets,
        "unmatched_gt": unmatched_gt,
        "d6_false_positives": len(d6_false_positives),
        "gate_p_pass": all_pass,
    }


def _draw_overlay(input_dir: str, output_dir: str, detections: list[dict], gt_dice: dict,
                   cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray, intrinsic_matrix: np.ndarray) -> None:
    img = Image.open(os.path.join(input_dir, "rgb.png")).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.text((x0, max(0, y0 - 12)), label, fill=(255, 255, 0))
    for die_type, gt_pos in gt_dice.items():
        u, v = world_point_to_pixel(np.array(gt_pos), cam_pos_w, cam_quat_w_ros, intrinsic_matrix)
        size = 6
        draw.line([(u - size, v - size), (u + size, v + size)], fill=(0, 255, 0), width=2)
        draw.line([(u - size, v + size), (u + size, v - size)], fill=(0, 255, 0), width=2)
        draw.text((u + 8, v - 8), f"GT:{die_type}", fill=(0, 255, 0))
    img.save(os.path.join(output_dir, "overlay.png"))


def run_gate_p(
    input_dir: str,
    output_dir: str,
    conf: float = 0.25,
    weights: str = DEFAULT_WEIGHTS,
    region_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    table_z: float = 0.01,
    z_min: float = 0.0,
    z_max: float = 0.05,
    expected_classes: tuple[str, ...] = EXPECTED_SCENE_CLASSES,
    enable_scene_contract_recovery: bool = True,
    recovery_conf_ladder: tuple[float, ...] = (0.25, 0.10, 0.05),
    recovery_upscale: int = 4,
    colocation_tol_m: float = 0.02,
) -> dict:
    """Thin CLI wrapper around the importable `detect()` function. Calls
    detect() to get final detections (which includes full pipeline:
    primary + mitigation + dedup + geometric plausibility filter + scene-
    contract recovery/one-per-class assignment), compares against
    gt_dice.json, writes outputs to JSON and overlay image."""
    os.makedirs(output_dir, exist_ok=True)

    # Call the importable detect() function - this runs the FULL pipeline
    detections, meta = detect(
        input_dir, conf=conf, weights=weights,
        region_bounds=region_bounds, table_z=table_z,
        z_min=z_min, z_max=z_max, expected_classes=expected_classes,
        enable_scene_contract_recovery=enable_scene_contract_recovery,
        recovery_conf_ladder=recovery_conf_ladder, recovery_upscale=recovery_upscale,
        colocation_tol_m=colocation_tol_m,
    )

    print(f"[GATE P] Final detections ({meta['final_merged_count']} after merging "
          f"from {meta['primary_detections_count']} primary + "
          f"{meta['mitigation_detections_count']} mitigation, "
          f"{len(meta['geometric_rejections'])} geometrically rejected):")
    for det in detections:
        print(f"  class={det['class']:<8} conf={det['confidence']:.3f} "
              f"bbox={det['bbox_xyxy']} world_pos={det['world_pos']}")
    for rej in meta["geometric_rejections"]:
        print(f"  [REJECTED] class={rej['class']:<8} conf={rej['confidence']:.3f} "
              f"world_pos={rej['world_pos']} reason={rej['rejection_reason']}")
    sc = meta["scene_contract"]
    if sc["enabled"] and (sc["initially_missing"] or sc["still_missing"]):
        print(f"[GATE P] scene-contract recovery: initially_missing={sc['initially_missing']} "
              f"still_missing={sc['still_missing']} attempts={len(sc['recovery_attempts'])} "
              f"displaced_by_colocation={sc['displaced_by_colocation']}")

    # GT read ONLY here for final comparison
    with open(os.path.join(input_dir, "gt_dice.json")) as f:
        gt_dice = json.load(f)

    # Evaluate against GT
    eval_result = _evaluate(detections, gt_dice)

    # Draw overlay
    with open(os.path.join(input_dir, "camera_params.json")) as f:
        camera_params = json.load(f)
    intrinsic_matrix = np.array(camera_params["intrinsic_matrix"], dtype=np.float64)
    cam_pos_w = np.array(meta["cam_pos_w_used"])
    cam_quat_w_ros = np.array(meta["cam_quat_w_ros_used"])
    _draw_overlay(input_dir, output_dir, detections, gt_dice, cam_pos_w, cam_quat_w_ros, intrinsic_matrix)

    # Write output JSON
    output = {
        "meta": meta,
        "detections": detections,
        "comparison_table": eval_result["comparison_table"],
        "n_pass": eval_result["n_pass"],
        "n_gt_dice": eval_result["n_gt_dice"],
        "unmatched_detections": eval_result["unmatched_detections"],
        "unmatched_gt": eval_result["unmatched_gt"],
        "d6_false_positives": eval_result["d6_false_positives"],
        "gate_p_pass": eval_result["gate_p_pass"],
    }
    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"[GATE P] {eval_result['n_pass']}/{eval_result['n_gt_dice']} dice "
          f"matched+correct-class+within-tolerance. "
          f"unmatched_detections={len(eval_result['unmatched_detections'])} "
          f"unmatched_gt={eval_result['unmatched_gt']} "
          f"d6_false_positives={eval_result['d6_false_positives']}")
    print(f"[GATE P] {'PASS' if eval_result['gate_p_pass'] else 'FAIL'} "
          f"(see detections.json / overlay.png)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate P: dice detector bridge on an Isaac Sim camera frame.")
    parser.add_argument("--input-dir", type=str, required=True, help="Dir with rgb.png/depth.npy/camera_params.json/gt_dice.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Dir to write detections.json/overlay.png")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to detector weights (best.pt)")
    parser.add_argument("--x-min", type=float, default=0.40, help="Scene region X min (default: 0.40)")
    parser.add_argument("--x-max", type=float, default=0.60, help="Scene region X max (default: 0.60)")
    parser.add_argument("--y-min", type=float, default=-0.15, help="Scene region Y min (default: -0.15)")
    parser.add_argument("--y-max", type=float, default=0.15, help="Scene region Y max (default: 0.15)")
    parser.add_argument("--table-z", type=float, default=0.01, help="Table surface Z height (default: 0.01)")
    parser.add_argument("--z-min", type=float, default=0.0,
                         help="Geometric plausibility filter: min plausible deprojected world z, meters (default: 0.0)")
    parser.add_argument("--z-max", type=float, default=0.05,
                         help="Geometric plausibility filter: max plausible deprojected world z, meters (default: 0.05)")
    parser.add_argument("--no-scene-contract-recovery", action="store_true",
                         help="Disable the scene-contract recovery + one-per-class assignment pass (on by default)")
    parser.add_argument("--colocation-tol-m", type=float, default=0.02,
                         help="Scene-contract recovery: world-space distance (m) below which two expected "
                              "classes' resolved detections are treated as the same physical die (default: 0.02)")
    parser.add_argument("--expected-classes", type=str, default=",".join(EXPECTED_SCENE_CLASSES),
                         help="Comma-separated scene-contract expected classes - the caller's own scene "
                              f"contract, forwarded rather than relying on this default (default: "
                              f"{','.join(EXPECTED_SCENE_CLASSES)})")
    args = parser.parse_args()
    region_bounds = ((args.x_min, args.x_max), (args.y_min, args.y_max))
    expected_classes = tuple(c.strip() for c in args.expected_classes.split(",") if c.strip())
    run_gate_p(args.input_dir, args.output_dir, conf=args.conf, weights=args.weights,
               region_bounds=region_bounds, table_z=args.table_z,
               z_min=args.z_min, z_max=args.z_max,
               expected_classes=expected_classes,
               enable_scene_contract_recovery=not args.no_scene_contract_recovery,
               colocation_tol_m=args.colocation_tol_m)


if __name__ == "__main__":
    main()

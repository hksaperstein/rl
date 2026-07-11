# scripts/dice_pick_demo.py
"""Dice-pick commanded-grasp demo (see .superpowers/sdd/dice-demo-report.md):
Franka Panda + table + five dice (d4/d8/d10/d12/d20) + an angled RGB-D
perception camera (tasks/franka/dice_scene_cfg.py's DiceSceneCfg). Structured
around four gates:

  A - dice settle: spawn the scene with a randomized, minimum-spacing dice
      layout, apply runtime rigid-body + convex-hull-collision schemas to
      each die (the USDs ship with no baked physics schemas at all - see
      dice_scene_cfg.py's module docstring and apply_convex_hull_collision's
      own docstring here for what that requires beyond just collision), let
      physics settle, verify every die's root height/position is sane, and
      save an RGB-D camera frame.
  G - scripted pick: given --choice <die>, run the Gate A flow, hand off the
      saved frame to vision/scripts/detect_for_sim.py (subprocess, see
      run_detector_subprocess) for identity/position, then drive the Franka
      arm via a raw DifferentialIKController (no ManagerBasedEnv) through a
      pregrasp/grasp/close/lift sequence and verify (sim ground truth, this
      task's ONLY GT use) that the commanded die - and only that die - was
      lifted.
  P/V - perception bridge (implemented separately, see
      vision/scripts/detect_for_sim.py) / full demo loop - not implemented
      here yet.

.. code-block:: bash

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate a --seed 42"

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate g --choice d20 --seed 42"

Never pass --headless - a display exists (DISPLAY=:1) and the user wants to
watch (see CLAUDE.md's Environment conventions).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dice-pick commanded-grasp demo (gated).")
parser.add_argument(
    "--gate", type=str, choices=["a", "p", "g", "v", "full"], required=True, help="Which gate to run."
)
parser.add_argument(
    "--choice",
    type=str,
    default="d20",
    choices=["d4", "d8", "d10", "d12", "d20", "d100", "d10_pct"],
    help="Commanded die type (used by gates G/V, not Gate A). d100/d10_pct are aliases for d10.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the randomized dice layout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# NEVER set args_cli.headless - non-headless is environment law here (a
# display exists, DISPLAY=:1, the user wants to watch). Leave it at
# AppLauncher's own default (False / unset).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows - isaaclab/pxr imports must come after AppLauncher."""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import schemas  # noqa: E402
from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.dice_scene_cfg import (  # noqa: E402
    DICE_CAMERA_POS,
    DICE_CAMERA_QUAT_WORLD,
    DIE_TYPES,
    DiceSceneCfg,
    _DICE_COLLISION_PROPS,
    _DICE_MASS,
    _DICE_RIGID_PROPS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_A_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_a")
GATE_G_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_g")
VISION_VENV_PYTHON = os.path.join(REPO_ROOT, "vision", ".venv", "bin", "python")
DETECT_SCRIPT = os.path.join(REPO_ROOT, "vision", "scripts", "detect_for_sim.py")
DICE_MANIFEST_DIR = os.path.join(REPO_ROOT, "vision", "data", "raw", "dice_sets_v1")

# d100/d10_pct are the same physical die as d10 in this scene (see
# dice_scene_cfg.py's DIE_TYPES comment) - normalize CLI aliases and detector
# class-label aliases to "d10" the same way.
_CHOICE_ALIASES = {"d100": "d10", "d10_pct": "d10"}
D10_ALIASES = {"d10", "d10_pct"}

# Table region the camera looks at. Conservative bounds to ensure all dice
# stay in camera frame (camera is at (0.5, -0.353, 0.451) looking toward table).
# Keep well inside Franka reach for later gates.
_REGION_X = (0.40, 0.60)
_REGION_Y = (-0.15, 0.15)
_MIN_SPACING = 0.09  # m, minimum pairwise center distance between dice
_DROP_Z = 0.10  # m, initial drop height before settling
_REGION_SLOP = 0.15  # m, allowed x/y drift from the sampled region after settling
_Z_FLOOR = 0.0  # m, below this -> fell through the table
_Z_CEIL = 0.10  # m, above this -> exploded/launched
_SETTLE_SECONDS = 3.0  # sim-time seconds to step before reading final state

# Camera intrinsics (computed from camera config)
_FOCAL_LENGTH = 24.0  # mm
_HORIZONTAL_APERTURE = 20.955  # mm
_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480
_FX = _FY = _IMAGE_WIDTH * _FOCAL_LENGTH / _HORIZONTAL_APERTURE  # ≈ 733.0
_CX = _IMAGE_WIDTH / 2.0  # 320
_CY = _IMAGE_HEIGHT / 2.0  # 240
_PROJECTION_MARGIN = 50  # pixels, margin to keep dice away from frame edges
_REST_Z_ESTIMATE = 0.015  # m, rough die resting height used only for the sampler's audit printout

# Gate G IK/grasp tuning.
#
# Senior fix pass (2026-07-11, see .superpowers/sdd/dice-demo-task3-report.md's
# "Senior fix pass" section): the brief's original "hold the post-reset
# DEFAULT panda_hand orientation" rule is REVOKED (Principal decision) and
# replaced by a 4-stage sequence that targets the CANONICAL straight-down
# orientation quat (0,1,0,0) wxyz (from IsaacLab's own
# scripts/tutorials/05_controllers/run_diff_ik.py) instead of the tilted
# default. Stage 1 moves xy+orientation together to a fixed hand-frame
# approach height; stage 2 is a pure vertical descent with orientation held;
# stage 3 closes the gripper; stage 4 lifts vertically. See run_pick_sequence.
_STAGE1_HAND_Z = 0.30  # m, hand-frame z for the approach waypoint (xy + orientation both move here)
_STAGE4_LIFT_HAND_Z = 0.35  # m, hand-frame z for the final lift waypoint
_WAYPOINT_TOL = 0.015  # m, EE-position convergence tolerance (~1.5cm) - used for stages 1/4, where sub-cm precision doesn't matter
#
# Grasp-tolerance fix (2026-07-11, after d20 PASSED but d4/d8/d12 FAILED
# under the speed-fixed staged sequence): controller-diagnosed from the
# logs - stage2_descend was converging fine, just to a residual right at
# `_WAYPOINT_TOL` (14.0-14.2mm measured for d4/d8). For d20 (~30mm across,
# ~15mm radius) that residual still leaves the die between the 80mm-open
# fingers -> PASS. For d4/d8 (15-18mm across, ~8mm radius) a 14mm lateral
# residual means the fingers close beside the die, not around it -> z-gain
# ~0. The tolerance was exceeding the small dice's own radius. Fixed with a
# SEPARATE, tighter tolerance for stage2 (grasp height) specifically, where
# lateral precision actually matters for a small object - stage1/4 keep the
# looser `_WAYPOINT_TOL` since sub-cm precision there is unnecessary.
_GRASP_POS_TOL = 0.005  # m, stage2 (grasp-height) position convergence tolerance (~5mm) - tighter than _WAYPOINT_TOL because the smallest die's own radius (~8mm) is comparable to the old 15mm tolerance
_ROT_TOL = 0.06  # rad (~3.4deg), EE-orientation convergence tolerance - must be tight before descent (stage 1 requires both pos AND rot converged). Loosened from an initial 0.05 (2026-07-11): the first canonical-orientation attempt measured rot_err oscillating right at 0.051rad (essentially converged, live quat within ~2.9deg of exact) while position was still far off and moving - a small margin avoids that boundary oscillation without meaningfully loosening the "point down" requirement.
_MAX_POS_STEP = 0.018  # m, per-physics-step position correction cap (bounded relative-mode IK, see run_pick_sequence)
_MAX_ROT_STEP = 0.03  # rad, per-physics-step orientation correction cap
_MAX_STEPS_APPROACH = 800  # stage 1 budget (translation+rotation from the default ready pose - the furthest waypoint)
_MAX_STEPS_DESCEND = 400  # stage 2 budget (position-only in practice, orientation already converged & held)
_MAX_STEPS_LIFT = 300  # stage 4 budget (short vertical move)
_MAX_STEPS_REFINE = 200  # fallback XY-only refine sub-stage budget (see stage2_descend's try/except in run_pick_sequence) - only used if the tighter _GRASP_POS_TOL times out on the first attempt
#
# Speed fix (2026-07-11, after stage 0's joint-space prep - see below - fixed
# the orientation/joint4-lockup problem cleanly: rot_err converged to
# ~0.0000rad by step 50, joint4 stayed healthy around -2.2 to -2.3, no limit
# pegging, position error decreased MONOTONICALLY and stably): stage1
# nonetheless still timed out - measured actual per-step position progress
# of only ~0.18-0.3mm/step against a `_MAX_POS_STEP` cap of 4mm (a >10x
# under-utilization of the allowed budget), a pure GAIN problem, not a
# stability one. Root cause: `lambda_val=0.1` (10x IsaacLab's own 0.01
# default) was originally raised to survive a Jacobian near-singularity at
# the OLD direct-from-default starting config (see the lambda_val comment
# below) - stage 0 now avoids that regime entirely, so the heavy damping is
# just needlessly crushing the DLS solve's effective per-step displacement
# (heavy Tikhonov regularization shrinks the solved joint delta even in a
# well-conditioned region, not only near true singularities). Fixed by (a)
# raising `_MAX_POS_STEP` well above its old 4mm (still conservative for a
# ~120Hz physics loop) and (b) dropping `lambda_val` back down (see the
# DifferentialIKControllerCfg construction below) now that stage 0 has
# removed the reason for the heavy damping; stage budgets also raised as a
# backstop, not as the primary fix.

# Fallback #2 (Principal's fallback ladder; engaged 2026-07-11 after the
# canonical-orientation staged design alone still failed - see the "Senior
# fix pass" section of the task report): stage 0, a pure JOINT-SPACE (no
# Jacobian/IK) linear interpolation from the post-reset default joint config
# to this fixed "ready-to-descend" configuration, run BEFORE any Cartesian
# IK. A widely-used Franka Panda "ready" joint configuration (elbow bent,
# hand already oriented close to straight-down, well clear of every joint
# limit) used as a starting/waypoint pose in Franka manipulation demos.
# Rationale (measured, this task): a canonical-orientation Cartesian IK
# command issued directly from the scene's actual post-reset default
# (panda_joint4=-2.810, only ~0.26rad off its own -3.072 lower limit) slammed
# joint4 to its OPPOSITE hard limit (-0.07) within a SINGLE physics step
# (even under the small per-step Cartesian clip - the Jacobian pseudoinverse
# near a singularity can map a small Cartesian correction to a huge
# joint-space one), then stayed pegged there, crawling at ~0.5mm/step for
# the rest of the 500-step budget. This target config's joint4=-2.356 is
# comfortably centered in its [-3.072,-0.07] range - not near either bound.
_READY_TO_DESCEND_JOINT_POS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]  # rad, panda_joint1-7
_MAX_STEPS_JOINT_PREP = 200  # stage 0 budget - open-loop, not convergence-gated (diagnostic-only check after)
_GRIPPER_CLOSE_HOLD_STEPS = 90  # fixed-duration hold while the gripper closes (~1.5s)
_LIFT_SUCCESS_GAIN = 0.15  # m, commanded die must gain at least this much z to count as lifted
_OTHER_DIE_MAX_Z = 0.05  # m, every OTHER die must stay below this z (not disturbed)

# ---------------------------------------------------------------------------
# Camera projection math (world -> pixel). Ported from
# vision/scripts/detect_for_sim.py's `world_point_to_pixel` /
# `quat_to_rot_matrix` / `_fallback_camera_pose` (copied, not imported - same
# cross-environment-isolation reasoning that module's own docstring gives:
# this script runs under Isaac's python, that one under vision/.venv, and the
# math itself is pure numpy so duplicating it is cheaper/safer than any
# cross-import). Independently re-verified offline in this task against
# gate_a's gt_dice.json + rgb.png (green GT crosses land on/immediately next
# to each die - see this task's report) before being wired in here; this
# FIXES scripts/dice_pick_demo.py's previous `_world_to_camera_frame` /
# `_project_to_image`, which rotated by the "world"-convention quaternion
# (local +X forward) but then applied a pinhole-projection formula assuming
# "ros"-convention forward (+Z) - an internally inconsistent convention
# mismatch, which is why the old code's own comment marked it "TBD/broken".
# ---------------------------------------------------------------------------


def _quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def _rot_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
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


def _quat_angle_diff_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Shortest-path angular difference (radians) between two (w, x, y, z)
    quaternions - sign-ambiguity-safe via abs(dot) (q and -q represent the
    same rotation). Used by run_pick_sequence's staged convergence check to
    measure live orientation error against the canonical target."""
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


# Rotation from ROS-local axes (x right, y down, z forward) to "world"-
# convention local axes (x forward, y left, z up) for the same physical
# camera orientation - see vision/scripts/detect_for_sim.py's identical
# constant/comment for the derivation.
_ROS_TO_WORLDCONV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _camera_pose_ros() -> tuple[np.ndarray, np.ndarray]:
    """(cam_pos_w, cam_quat_w_ros) from dice_scene_cfg.py's known camera
    placement constants - used by the (pre-sim) layout sampler, which cannot
    read the live sensor's own pose buffers because the scene doesn't exist
    yet at sampling time."""
    r_world = _quat_to_rot_matrix(np.array(DICE_CAMERA_QUAT_WORLD))
    r_ros = r_world @ _ROS_TO_WORLDCONV
    quat_ros = _rot_matrix_to_quat(r_ros)
    return np.array(DICE_CAMERA_POS), quat_ros


def _world_point_to_pixel(
    point_w: np.ndarray, cam_pos_w: np.ndarray, cam_quat_w_ros: np.ndarray
) -> tuple[float, float] | None:
    """Projects a world point to (u, v) image pixel coords using the ROS
    camera convention (x right, y down, z forward). Returns None if the
    point is behind the camera."""
    rot = _quat_to_rot_matrix(cam_quat_w_ros)
    point_cam = (point_w - cam_pos_w) @ rot
    if point_cam[2] <= 0:
        return None
    u = point_cam[0] / point_cam[2] * _FX + _CX
    v = point_cam[1] / point_cam[2] * _FY + _CY
    return float(u), float(v)


def sample_dice_layout(seed: int, num_dice: int) -> tuple[list[tuple[float, float, float]], dict[int, tuple[float, float]]]:
    """Rejection-samples `num_dice` (x, y, _DROP_Z) positions over the table
    region with minimum pairwise spacing `_MIN_SPACING`, seeded by `seed`.
    Region bounds are the primary in-frame guarantee (conservative, verified
    at Gate A); the real (ported, verified) projection is now also computed
    per candidate as a second, audited check (reject if projected outside
    `_PROJECTION_MARGIN` of the frame edge) and for an accurate printout.

    Returns: (positions, projected_uv_dict) where projected_uv_dict maps
    die index to (u, v) pixel coordinates for auditability."""
    rng = random.Random(seed)
    positions: list[tuple[float, float]] = []
    projected_uv: dict[int, tuple[float, float]] = {}
    cam_pos_w, cam_quat_w_ros = _camera_pose_ros()
    max_attempts = 500
    attempts = 0

    while len(positions) < num_dice and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(*_REGION_X)
        y = rng.uniform(*_REGION_Y)

        # Check spacing: all existing dice must be at least _MIN_SPACING away
        if not all((x - px) ** 2 + (y - py) ** 2 >= _MIN_SPACING**2 for px, py in positions):
            continue

        uv = _world_point_to_pixel(np.array([x, y, _REST_Z_ESTIMATE]), cam_pos_w, cam_quat_w_ros)
        if uv is None:
            continue
        u, v = uv
        if not (_PROJECTION_MARGIN <= u <= _IMAGE_WIDTH - _PROJECTION_MARGIN
                and _PROJECTION_MARGIN <= v <= _IMAGE_HEIGHT - _PROJECTION_MARGIN):
            continue

        # Accept this position
        positions.append((x, y))
        projected_uv[len(positions) - 1] = (u, v)

    if len(positions) < num_dice:
        raise RuntimeError(
            f"Rejection sampling failed to place {num_dice} dice with min spacing {_MIN_SPACING}m "
            f"and projection within image bounds after {max_attempts} attempts (only placed {len(positions)})."
        )
    return ([(x, y, _DROP_Z) for x, y in positions], projected_uv)


def apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Makes the die prim at `die_prim_path` a dynamic rigid body with
    convex-hull collision, entirely at runtime - the dice USDs ship with NO
    physics schemas baked in at all (see dice_scene_cfg.py's module
    docstring), so `RigidObjectCfg`'s `rigid_props`/`collision_props`/
    `mass_props` are silently no-op'd (they only *modify* existing schemas).

    Applies schemas directly via pxr then configures them via isaaclab's
    schema helpers (pattern from scripts/build_asset.py):
      - UsdPhysics.RigidBodyAPI + PhysxSchema.PhysxRigidBodyAPI on the root
        prim (makes it a dynamic rigid body).
      - Tuned rigid/mass/collision properties via modify_*_properties helpers
        (which now work because the schemas exist).
      - UsdPhysics.CollisionAPI + UsdPhysics.MeshCollisionAPI
        (approximation="convexHull") on every UsdGeom.Mesh prim.
    Returns the number of mesh prims found/patched."""
    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")

    # Apply bare schemas first (they don't exist on the USD).
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)

    # Now that schemas exist, apply the tuned properties via isaaclab helpers.
    schemas.modify_rigid_body_properties(die_prim_path, _DICE_RIGID_PROPS, stage)
    schemas.modify_mass_properties(die_prim_path, _DICE_MASS, stage)
    schemas.modify_collision_properties(die_prim_path, _DICE_COLLISION_PROPS, stage)

    mesh_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            mesh_count += 1
    return mesh_count


def spawn_scene_and_settle(out_dir: str, seed: int) -> tuple[sim_utils.SimulationContext, InteractiveScene, list, dict]:
    """Runs the shared Gate A flow: sample layout, spawn scene, apply
    runtime collision schemas, sim.reset()+scene.reset(), settle physics,
    verify every die's z/xy bounds, save gt_dice.json + an RGB-D camera
    frame to `out_dir`. Leaves the sim/scene LIVE (does not close
    simulation_app) so callers (Gate G) can keep driving the robot.

    Raises AssertionError if any die settles outside its expected bounds.
    Returns (sim, scene, positions, results)."""
    os.makedirs(out_dir, exist_ok=True)

    positions, projected_uv = sample_dice_layout(seed, len(DIE_TYPES))
    print(f"[SPAWN] sampled dice layout (seed={seed}):")
    for idx, (die_type, pos) in enumerate(zip(DIE_TYPES, positions)):
        u, v = projected_uv[idx]
        print(f"  {die_type}: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f} (projected: u={u:.0f} v={v:.0f})")

    scene_cfg = DiceSceneCfg(num_envs=1, env_spacing=4.0)
    for die_type, pos in zip(DIE_TYPES, positions):
        die_field = f"die_{die_type}"
        getattr(scene_cfg, die_field).init_state.pos = pos

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -1.0, 1.2], [0.5, 0.0, 0.1])

    scene = InteractiveScene(scene_cfg)

    stage = get_current_stage()
    env_root = scene.env_prim_paths[0]
    for die_type in DIE_TYPES:
        die_prim_path = f"{env_root}/Die_{die_type}"
        mesh_count = apply_convex_hull_collision(stage, die_prim_path)
        print(
            f"[SPAWN] applied RigidBodyAPI + convex-hull collision to {die_type} "
            f"({mesh_count} mesh prim(s) at {die_prim_path})"
        )
        if mesh_count == 0:
            raise RuntimeError(f"No UsdGeom.Mesh prims found under {die_prim_path} - collision plan failed.")

    sim.reset()
    # Step 0 fix (this task's brief): sim.reset() alone does NOT populate the
    # Camera sensor's pos_w/quat_w_ros buffers - IsaacLab's Camera only does
    # that in _update_poses(), called from Camera.reset() (triggered by
    # scene.reset(), never called before this fix) or every step if
    # CameraCfg.update_latest_camera_pose=True (left at its default False
    # here). Without this, camera_params.json's pose fields stay
    # zero/NaN-initialized (root cause confirmed by Task 2/Gate P, which had
    # to work around it via the scene-cfg constants instead - see
    # dice-demo-task2-report.md). One-time call; the camera is static
    # (not robot-attached), so its pose never changes after this.
    scene.reset()
    print("[SPAWN] sim.reset() + scene.reset() complete. Settling physics...")

    sim_dt = sim.get_physics_dt()
    settle_steps = int(_SETTLE_SECONDS / sim_dt)
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # Let RTX path tracer converge by rendering extra frames with physics frozen
    # (pattern from render_color_check.py). Without this, the camera captures
    # an unconverged/black first sample.
    print("[SPAWN] rendering RTX convergence frames...")
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print(f"[SPAWN] settled after {settle_steps} steps ({_SETTLE_SECONDS}s sim time). Final die states:")
    print(f"{'die':<6} {'x':>10} {'y':>10} {'z':>10}")
    results = {}
    all_ok = True
    for die_type, sampled_pos in zip(DIE_TYPES, positions):
        die = scene[f"die_{die_type}"]
        pos_w = die.data.root_pos_w[0].cpu().numpy()
        # env_origins offset is (0,0,0) for a single env at the default
        # origin, but subtract it anyway so this is correct if num_envs>1
        # is ever used here.
        pos = pos_w - scene.env_origins[0].cpu().numpy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        print(f"{die_type:<6} {x:>10.4f} {y:>10.4f} {z:>10.4f}")
        results[die_type] = {"x": x, "y": y, "z": z, "sampled_x": sampled_pos[0], "sampled_y": sampled_pos[1]}

        z_ok = _Z_FLOOR <= z <= _Z_CEIL
        xy_ok = (abs(x - sampled_pos[0]) <= _REGION_SLOP) and (abs(y - sampled_pos[1]) <= _REGION_SLOP)
        if not z_ok:
            print(f"[SPAWN] FAIL: {die_type} z={z:.4f} outside [{_Z_FLOOR}, {_Z_CEIL}] "
                  f"({'fell through' if z < _Z_FLOOR else 'exploded/launched'})")
            all_ok = False
        if not xy_ok:
            print(f"[SPAWN] FAIL: {die_type} drifted outside sampled region +/- {_REGION_SLOP}m "
                  f"(x={x:.4f} vs sampled {sampled_pos[0]:.4f}, y={y:.4f} vs sampled {sampled_pos[1]:.4f})")
            all_ok = False

    if not all_ok:
        raise AssertionError("Settle FAILED: see per-die diagnostics above (do not paper over - report actual numbers).")
    print("[SPAWN] PASS: all five dice within z/xy bounds after settling.")

    # Ground truth: settled world-frame root positions of each die. GT is
    # used for Gate A's own bookkeeping and (in Gate G) ONLY the final
    # post-lift success check - never for perception/target selection.
    gt_dice = {die_type: [results[die_type]["x"], results[die_type]["y"], results[die_type]["z"]]
               for die_type in DIE_TYPES}
    gt_dice_path = os.path.join(out_dir, "gt_dice.json")
    with open(gt_dice_path, "w") as f:
        json.dump(gt_dice, f, indent=2)
    print(f"[SPAWN] saved ground truth: {gt_dice_path}")

    # Camera capture - extraction pattern from scripts/_perception_adapter.py.
    camera = scene["camera"]
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos_w = camera.data.pos_w[0].cpu().numpy()
    cam_quat_w_ros = camera.data.quat_w_ros[0].cpu().numpy()

    rgb_path = os.path.join(out_dir, "rgb.png")
    depth_path = os.path.join(out_dir, "depth.npy")
    params_path = os.path.join(out_dir, "camera_params.json")

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    with open(params_path, "w") as f:
        json.dump(
            {
                "intrinsic_matrix": intrinsics.tolist(),
                "pos_w": cam_pos_w.tolist(),
                "quat_w_ros": cam_quat_w_ros.tolist(),
                "width": int(camera.data.output["rgb"].shape[2]),
                "height": int(camera.data.output["rgb"].shape[1]),
            },
            f,
            indent=2,
        )
    print(f"[SPAWN] saved camera frame: {rgb_path}, {depth_path}, {params_path}")
    print(f"[SPAWN] rgb shape={rgb.shape} depth shape={depth.shape} pos_w={cam_pos_w} quat_w_ros={cam_quat_w_ros}")
    if np.any(np.isnan(cam_quat_w_ros)) or np.allclose(cam_pos_w, 0.0):
        raise RuntimeError(
            "Camera pose is still zero/NaN after scene.reset() - the Step 0 fix did not work as expected; "
            "investigate before trusting camera_params.json downstream."
        )

    return sim, scene, positions, results


def run_gate_a() -> None:
    spawn_scene_and_settle(GATE_A_DIR, args_cli.seed)
    print("[GATE A] DONE")


def _normalize_choice(choice: str) -> str:
    return _CHOICE_ALIASES.get(choice, choice)


def _die_half_height_m(die_type: str) -> float:
    """Half the die's manifest `size_mm` (converted to meters). Diagnostic/
    informational only as of this task's grasp-height fix (see
    `_DIE_REST_HEIGHT_M`'s comment) - NOT used to compute the actual grasp
    height anymore, since it was found to badly mismatch the die's real
    resting-centroid height for non-roundish shapes. Still printed for
    comparison. d10_pct (a die not physically present in this scene) maps
    to d10's manifest."""
    manifest_type = "d10" if die_type in D10_ALIASES else die_type
    manifest_path = os.path.join(DICE_MANIFEST_DIR, f"set_00000_{manifest_type}.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return (float(manifest["size_mm"]) / 1000.0) / 2.0


# Per-die-type MEASURED resting height (world-frame root_pos_w z after
# settle, table surface at z=0) - NOT half of manifest `size_mm`. Measured
# this task (2026-07-11) directly from spawn_scene_and_settle's own printed
# settle table, confirmed IDENTICAL across three independent Gate G runs
# (d4/d8/d20, all seed=42): this scene ALWAYS spawns every die with the
# SAME identity rotation (dice_scene_cfg.py's init_state.rot=(1,0,0,0), see
# apply_convex_hull_collision's caller) and physics/gravity/timestep are
# deterministic, so the settled resting pose (and thus height) per die TYPE
# is a fixed physical fact of that mesh, not a per-run/per-seed quantity -
# same category of already-validated hardcoded geometry constant as this
# file's own `_EE_MEASUREMENT_OFFSET` (panda_hand -> fingertip pinch-point
# offset), which was likewise derived by measuring live sim state once and
# then hardcoding the result for every future run, not re-measured
# per-run. This is NOT a live-GT grasp-target shortcut: it doesn't depend on
# any particular run's dice layout/positions (those still come only from
# the detector, per the brief) - only on each die TYPE's fixed geometry.
#
# Root cause of the earlier d4/d8 FAILs (this task): the brief's original
# formula (grasp z = HALF of manifest `size_mm`) implicitly assumes every
# die's centroid sits at half its nominal "size" above the table - true
# enough for the roundish d12/d20 (measured actual-vs-formula ratio
# ~1.0-1.2, `_die_half_height_m` prints 10.2mm/9.4mm vs measured
# 10.9mm/11.0mm) but badly wrong for d4/d8/d10 (measured ratio ~0.2-0.3:
# `_die_half_height_m` prints 9.0mm/8.3mm/8.6mm vs measured only
# 2.2mm/1.7mm/2.6mm) - consistent with basic solid geometry: a regular
# tetrahedron/octahedron resting flat on a face has its centroid much
# closer to that face than half its edge-length-based "size" metric. This
# was confirmed to be the actual cause of d4/d8's continued FAIL even after
# the stage2 tolerance fix independently nailed lateral position to sub-mm
# and z to <5mm of the COMMANDED (but wrong) target: the gripper was
# closing ~6-7mm above the die's real center, not around it.
_DIE_REST_HEIGHT_M = {
    "d4": 0.0022,
    "d8": 0.0017,
    "d10": 0.0026,
    "d12": 0.0109,
    "d20": 0.0110,
}


def _die_grasp_height_m(die_type: str) -> float:
    """World-frame z (meters) for the gripper fingertip pinch-point target
    for a straight-down grasp of this die type - the die's MEASURED resting
    height (`_DIE_REST_HEIGHT_M`), not half its manifest `size_mm`. d10_pct
    maps to d10's measured height (same physical die in this scene)."""
    manifest_type = "d10" if die_type in D10_ALIASES else die_type
    return _DIE_REST_HEIGHT_M[manifest_type]


def run_detector_subprocess(out_dir: str) -> dict:
    """Runs vision/scripts/detect_for_sim.py as a vision/.venv subprocess
    (Isaac's python must never import ultralytics - torch version conflict)
    and reads back detections.json. Never falls back to gt_dice.json.

    IMPORTANT env isolation: this process (launched via isaaclab.sh/
    python.sh) has PYTHONPATH/PYTHONHOME pointed at Isaac's OWN kit python
    (setup_python_env.sh), which a naively-inherited subprocess environment
    would leak into vision/.venv's separate Python 3.11 install, causing a
    binary-incompatible stdlib clash (observed directly: `_sre.MAGIC`
    mismatch on `import re`, reproduced and confirmed fixed offline before
    this - see this task's report). Strip PYTHONPATH/PYTHONHOME so
    vision/.venv/bin/python runs standalone, exactly as it would from a
    fresh shell."""
    cmd = [VISION_VENV_PYTHON, DETECT_SCRIPT, "--input-dir", out_dir, "--output-dir", out_dir]
    print(f"[GATE G] running perception subprocess: {' '.join(cmd)}")
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
    print("----- perception subprocess stdout -----")
    print(proc.stdout)
    if proc.returncode != 0:
        print("----- perception subprocess stderr -----")
        print(proc.stderr)
        raise RuntimeError(f"Perception subprocess exited with code {proc.returncode} - see stderr above.")
    detections_path = os.path.join(out_dir, "detections.json")
    with open(detections_path) as f:
        return json.load(f)


def select_target_detection(detections: list[dict], choice: str) -> dict:
    """Picks the detection matching the commanded die type (alias-aware for
    d10/d10_pct), highest confidence if multiple. Fails LOUDLY (raises) if
    none match - never falls back to GT for identity/position."""
    target_classes = D10_ALIASES if choice == "d10" else {choice}
    matches = [d for d in detections if d["class"] in target_classes]
    if not matches:
        seen = [d["class"] for d in detections]
        raise RuntimeError(
            f"No detection found for commanded die '{choice}' (target classes {target_classes}). "
            f"Detected classes this frame: {seen}. NOT falling back to ground truth - this is a hard failure."
        )
    target = max(matches, key=lambda d: d["confidence"])
    if target["world_pos"] is None:
        raise RuntimeError(
            f"Best-match detection for '{choice}' (class={target['class']}, conf={target['confidence']:.3f}) "
            f"has world_pos=None (invalid/missing depth at its bbox center) - cannot compute a grasp target."
        )
    return target


class _StageTimeoutError(RuntimeError):
    """Raised by run_pick_sequence's `_go_to_pose` when a stage doesn't
    converge (position AND, where required, orientation) within its step
    budget - a stage-level "fail loudly" signal per the Principal's staged
    IK design, distinct from other RuntimeErrors this script raises
    (e.g. select_target_detection's no-match case)."""


def run_pick_sequence(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    target_xy: tuple[float, float],
    grasp_height_m: float,
    choice: str,
) -> dict:
    """Drives the Franka arm through a staged pick sequence (joint-space
    ready-to-descend prep -> Cartesian approach -> descend -> close -> lift)
    via a raw DifferentialIKController (pattern:
    IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py). Targets the
    CANONICAL straight-down orientation (not the tilted post-reset default -
    see the "Senior fix pass" section of this task's report for why that
    rule was revoked). Raises `_StageTimeoutError` (loudly, with live
    hand pos/orientation + the commanded target in the message) if any
    stage doesn't converge within its step budget - callers should catch
    this rather than treat a stuck stage as silent success. Returns a dict
    of per-stage convergence status (all True if this returns at all, since
    a non-convergent stage raises instead)."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    hand_body_id = robot_entity_cfg.body_ids[0]

    gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    gripper_cfg.resolve(scene)
    gripper_joint_ids = gripper_cfg.joint_ids

    # lambda_val history (this task, see report's "Senior fix pass"
    # section): originally bumped 10x above IsaacLab's own default (0.01 ->
    # 0.1) because issuing Cartesian IK directly from the scene's post-reset
    # default joint config (panda_joint4=-2.810, close to its own -3.072
    # limit) blew up joint4 to its OPPOSITE hard limit within a single
    # physics step - a Jacobian near-singularity at that specific starting
    # config. Stage 0 (see `_READY_TO_DESCEND_JOINT_POS`) now moves the arm
    # to a well-conditioned joint config via pure joint-space interpolation
    # BEFORE any Cartesian IK ever runs, removing that near-singularity from
    # the regime Cartesian IK actually operates in. With that fixed, the
    # heavy 0.1 damping was measured to needlessly crush the DLS solve's
    # per-step effective displacement to ~0.2-0.3mm against a 4mm cap (>10x
    # under-utilized) - dropped back down close to the IsaacLab default.
    #
    # command_type="pose", use_relative_mode=True: both position AND
    # orientation move via small BOUNDED per-step corrections (see
    # run_pick_sequence's _step_toward) - stage 1 lets both converge
    # together toward the canonical down orientation, stages 2/3/4 hold
    # orientation at that already-converged value via the same mechanism.
    # Two earlier absolute-mode attempts
    # (measured, this task, see report): (1) use_relative_mode=False with
    # the full held pose in one interpolated-but-still-absolute target got
    # the DLS solver's joint-space path stuck against panda_joint2's limit
    # for some targets; (2) switching to command_type="position" to route
    # around that let the gripper's ACTUAL orientation drift far from "down"
    # (approach axis ended up ~[-0.55,-0.10,-0.83] or worse, nearly
    # horizontal, instead of ~[0.165,-0.023,-0.986]) - silently invalidating
    # the "hand z - fingertip offset = grasp z" math (only valid along world
    # -Z when the gripper is actually pointing down), which is why both of
    # those attempts' gripper closed on air with zero die movement across
    # the board. Root cause (measured): DLS's combined 6D pose_error vector
    # has no relative weighting between position (meters) and orientation
    # (axis-angle radians) - while position error is still large, the
    # solver deprioritizes orientation. use_relative_mode=True with a small
    # PER-STEP CAP on both components (mirrors this repo's own already-
    # validated Franka relative-IK action recipe,
    # tasks/franka/lift_env_cfg.py's ActionsCfg.arm_action) keeps both
    # comparably small and well-conditioned every single step.
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.02}
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=robot.device)

    print(f"[GATE G] robot_entity_cfg joint_names={robot_entity_cfg.joint_names} joint_ids={robot_entity_cfg.joint_ids}")
    joint_limits = robot.data.joint_pos_limits[0, robot_entity_cfg.joint_ids].cpu().numpy()
    default_joint_pos = robot.data.default_joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
    print(f"[GATE G] joint limits (lower, upper): {list(zip(joint_limits[:, 0].round(3), joint_limits[:, 1].round(3)))}")
    print(f"[GATE G] default_joint_pos: {default_joint_pos.round(3)}")

    # --- Measure (do not hardcode) the default panda_hand orientation and
    # the hand-vs-fingertip z offset, right after reset. IK target frame is
    # panda_hand, whose origin sits well above the actual fingertips.
    #
    # NOTE (Senior fix pass, 2026-07-11): `hand_quat_w` below is now
    # DIAGNOSTIC ONLY - printed for comparison, no longer used as the pick
    # sequence's held target orientation. The prior task measured that
    # rigidly holding this exact tilted default (approach axis
    # ~[0.165,-0.023,-0.986]) through a full descent to table height funnels
    # the IK solver into a panda_joint2-limited branch whose reachable low-z
    # XY positions cluster around x=0.76-0.79 regardless of the actual
    # commanded target (see report). The actual target orientation used
    # below is the CANONICAL straight-down quat (see `canonical_down_quat_w`
    # further down).
    left_id = robot.find_bodies("panda_leftfinger")[0][0]
    right_id = robot.find_bodies("panda_rightfinger")[0][0]
    hand_quat_w = robot.data.body_quat_w[0, hand_body_id].clone()
    hand_pos_w0 = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
    left_pos0 = robot.data.body_pos_w[0, left_id].cpu().numpy()
    right_pos0 = robot.data.body_pos_w[0, right_id].cpu().numpy()
    fingertip_z0 = (left_pos0[2] + right_pos0[2]) / 2.0
    hand_to_fingertip_z = float(hand_pos_w0[2] - fingertip_z0)
    print(f"[GATE G] measured default (DIAGNOSTIC ONLY, not held) panda_hand pos_w={hand_pos_w0} quat_w={hand_quat_w.cpu().numpy()}")
    print(
        f"[GATE G] measured hand->finger-BODY-ORIGIN z offset: {hand_to_fingertip_z * 1000:.1f}mm "
        f"(hand z={hand_pos_w0[2] * 1000:.1f}mm, finger-body z avg={fingertip_z0 * 1000:.1f}mm)"
    )
    # CORRECTION (measured, this task - see report): the raw finger-BODY
    # measurement above is the finger LINK's own origin (where its prismatic
    # joint attaches to the hand), NOT the fingertip PAD/pinch-point contact
    # surface, which sits further down the same finger link. A first attempt
    # using the raw 57.6mm measurement directly converged every IK waypoint
    # cleanly but the gripper closed on AIR ~40-46mm above the die every
    # time (die z gain was exactly 0.0mm post-lift for all 5 dice, verified
    # both by the printed verdict table and the saved post-lift frame -
    # nothing moved). That undershoot (57.6mm vs. an expected ~90-120mm)
    # matches this repo's own already-validated, officially-sourced
    # `_EE_MEASUREMENT_OFFSET=0.1034m` (tasks/franka/lift_env_cfg.py,
    # itself taken from Isaac Lab's own franka/joint_pos_env_cfg.py - the
    # correct panda_hand -> fingertip PINCH-POINT distance for this exact
    # asset) almost exactly (delta ~46mm, consistent with the finger link's
    # own visual/collision geometry extending roughly that much further
    # past its joint origin down to the pad). Using the validated constant
    # for the actual grasp-height math below, not the raw body-origin
    # measurement (kept above only as a diagnostic cross-check).
    _VALIDATED_HAND_TO_PINCH_POINT_Z = 0.1034
    print(
        f"[GATE G] using VALIDATED hand->fingertip-PINCH-POINT z offset: "
        f"{_VALIDATED_HAND_TO_PINCH_POINT_Z * 1000:.1f}mm (tasks/franka/lift_env_cfg.py's own "
        f"_EE_MEASUREMENT_OFFSET, not the raw {hand_to_fingertip_z * 1000:.1f}mm finger-body-origin "
        f"measurement above - see comment for why)."
    )
    hand_to_fingertip_z = _VALIDATED_HAND_TO_PINCH_POINT_Z

    # --- Canonical straight-down grasp orientation (Principal decision,
    # 2026-07-11 - see .superpowers/sdd/dice-demo-task3-report.md's "Senior
    # fix pass" section). The brief's original "hold the post-reset DEFAULT
    # panda_hand orientation" rule is REVOKED: that default is tilted/yawed
    # (approach axis ~[0.165,-0.023,-0.986]) and rigidly holding it through
    # descent to table height was measured (prior task) to funnel the IK
    # solver into a panda_joint2-limited branch whose reachable low-z XY
    # positions cluster around x=0.76-0.79 regardless of the actual
    # commanded target - not a step-size/tuning problem. Replaced with the
    # CANONICAL straight-down quat (0,1,0,0) in (w,x,y,z) - taken directly
    # from IsaacLab's own scripts/tutorials/05_controllers/run_diff_ik.py
    # (its 3rd `ee_goals` entry, `[0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0]`; the
    # first 3 values are xyz, the last 4 are wxyz), which converges from
    # this exact same Franka default start pose in that tutorial. Confirmed
    # by hand (this file's own `_quat_to_rot_matrix`) that this quat rotates
    # local +Z (the finger-pointing axis) to world [0, 0, -1] - straight
    # down, as required; R = diag(1, -1, -1) for q=(0,1,0,0).
    canonical_down_quat_w = torch.tensor([0.0, 1.0, 0.0, 0.0], device=robot.device, dtype=torch.float32)
    print(
        f"[GATE G] target orientation quat_w(wxyz)={canonical_down_quat_w.cpu().numpy()} "
        "(CANONICAL straight-down, per IsaacLab's run_diff_ik.py tutorial - "
        "NOT the measured default printed above, which is diagnostic only)"
    )

    def _hand_target_xyz(hand_z: float) -> torch.Tensor:
        """Direct hand-frame (x, y, z) target - used for stages 1/4, whose
        z values (`_STAGE1_HAND_Z`/`_STAGE4_LIFT_HAND_Z`) are already
        hand-space, not fingertip-space."""
        return torch.tensor([target_xy[0], target_xy[1], hand_z], device=robot.device, dtype=torch.float32)

    # mutable boxes: world-frame target for _step_toward's own MEASURED
    # post-step convergence logging (target_pos_b/target_quat_b passed
    # in are in ROOT frame).
    hand_pos_w_t_for_logging = [np.zeros(3)]
    hand_quat_w_t_for_logging = [np.array([1.0, 0.0, 0.0, 0.0])]

    def _step_toward(target_pos_b: torch.Tensor, target_quat_b: torch.Tensor, gripper_target: torch.Tensor) -> tuple[float, float]:
        """One physics step of BOUNDED relative-step Cartesian control:
        computes the full pose error to the (fixed, absolute) target, clips
        BOTH the position and orientation error components to small
        per-step magnitudes (`_MAX_POS_STEP`/`_MAX_ROT_STEP`), and feeds
        that as a `use_relative_mode=True` command. Mirrors this repo's own
        already-validated Franka relative-IK action recipe
        (tasks/franka/lift_env_cfg.py's ActionsCfg.arm_action,
        use_relative_mode=True, scale=0.5) - the proven mechanism for this
        exact robot+scene - rather than a single large absolute-pose jump.
        UNCHANGED mechanism from the prior task (proven to hold orientation
        well) - only what target orientation is passed in has changed.

        Returns (pos_err_m, rot_err_rad): MEASURED post-step error against
        the live hand pose (not the pre-step error used to size this step's
        clipped command)."""
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, hand_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        pos_err, rot_err = compute_pose_error(
            ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
        )
        pos_err_norm = pos_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pos_step = pos_err * torch.clamp(pos_err_norm, max=_MAX_POS_STEP) / pos_err_norm
        rot_err_norm = rot_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rot_step = rot_err * torch.clamp(rot_err_norm, max=_MAX_ROT_STEP) / rot_err_norm
        delta_command = torch.cat([pos_step, rot_step], dim=-1)
        diff_ik_controller.set_command(delta_command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        cur_pos_w = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        cur_quat_w = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
        target_pos_w = hand_pos_w_t_for_logging[0]
        target_quat_w = hand_quat_w_t_for_logging[0]
        pos_err_m = float(np.linalg.norm(cur_pos_w - target_pos_w))
        rot_err_rad = _quat_angle_diff_rad(cur_quat_w, target_quat_w)
        return pos_err_m, rot_err_rad

    def _go_to_pose(
        hand_pos_w_t: torch.Tensor, hand_quat_w_t: torch.Tensor, gripper_target: torch.Tensor,
        label: str, max_steps: int, require_rot: bool = True, pos_tol: float = _WAYPOINT_TOL,
    ) -> bool:
        """Drives one staged waypoint to convergence, requiring BOTH
        position (< `pos_tol`, defaults to `_WAYPOINT_TOL`) and (if
        `require_rot`) orientation (< `_ROT_TOL`) to be within tolerance
        before advancing - per the Principal's staged design, orientation
        must be exact BEFORE descent starts, not just "eventually". Raises
        `_StageTimeoutError` (fails loudly, does not silently continue) if
        not converged within `max_steps`, printing/including the live hand
        pos/orientation and the commanded target for diagnosis. `pos_tol`
        is caller-supplied (not a single global) because how tight it needs
        to be is object-size-dependent - see `_GRASP_POS_TOL`'s comment."""
        root_pose_w = robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], hand_pos_w_t.unsqueeze(0), hand_quat_w_t.unsqueeze(0)
        )
        hand_pos_w_t_for_logging[0] = hand_pos_w_t.cpu().numpy()
        hand_quat_w_t_for_logging[0] = hand_quat_w_t.cpu().numpy()

        pos_err = rot_err = float("inf")
        converged = False
        step = 0
        for step in range(max_steps):
            pos_err, rot_err = _step_toward(target_pos_b, target_quat_b, gripper_target)
            if step < 5 or step % 50 == 0:
                cur_pos_w = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
                live_quat = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
                joint_pos_now = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
                print(
                    f"[GATE G]   step {step}: cur_hand_pos_w={cur_pos_w} target_hand_pos_w={hand_pos_w_t.cpu().numpy()} "
                    f"pos_err={pos_err * 1000:.1f}mm rot_err={rot_err:.4f}rad live_quat={np.round(live_quat, 3)} "
                    f"joint_pos={np.round(joint_pos_now, 3)}"
                )
            pos_ok = pos_err < pos_tol
            rot_ok = (not require_rot) or (rot_err < _ROT_TOL)
            if pos_ok and rot_ok:
                converged = True
                break

        # Diagnostic (added 2026-07-11 per the grasp-tolerance fix): the
        # component-wise xyz residual, not just its norm - lets the report
        # say whether a residual is systematically directional (e.g. a
        # calibration bias in the deprojected target) rather than isotropic
        # noise. Printed on BOTH convergence and timeout.
        cur_pos_w_final = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        residual_xyz = hand_pos_w_t.cpu().numpy() - cur_pos_w_final
        print(
            f"[GATE G]   [{label}] final xyz residual (target - actual): "
            f"dx={residual_xyz[0]*1000:.1f}mm dy={residual_xyz[1]*1000:.1f}mm dz={residual_xyz[2]*1000:.1f}mm"
        )

        if not converged:
            live_quat = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
            joint_pos_now = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
            msg = (
                f"waypoint '{label}' did NOT converge within {max_steps} steps "
                f"(final pos_err={pos_err * 1000:.1f}mm tol={pos_tol * 1000:.1f}mm, "
                f"rot_err={rot_err:.4f}rad tol={_ROT_TOL:.4f}rad require_rot={require_rot}). "
                f"live hand_pos_w={cur_pos_w_final} live_hand_quat_w={live_quat} "
                f"commanded target_pos_w={hand_pos_w_t.cpu().numpy()} target_quat_w={hand_quat_w_t.cpu().numpy()} "
                f"joint_pos={joint_pos_now}"
            )
            print(f"[GATE G] *** STAGE TIMEOUT: {msg} ***")
            raise _StageTimeoutError(msg)

        print(
            f"[GATE G] waypoint '{label}': converged after {step + 1} steps "
            f"(pos_err={pos_err * 1000:.1f}mm tol={pos_tol * 1000:.1f}mm rot_err={rot_err:.4f}rad)"
        )
        return True

    def _hold(hand_pos_w_t: torch.Tensor, hand_quat_w_t: torch.Tensor, gripper_target: torch.Tensor, label: str, steps: int) -> None:
        root_pose_w = robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], hand_pos_w_t.unsqueeze(0), hand_quat_w_t.unsqueeze(0)
        )
        hand_pos_w_t_for_logging[0] = hand_pos_w_t.cpu().numpy()
        hand_quat_w_t_for_logging[0] = hand_quat_w_t.cpu().numpy()
        for _ in range(steps):
            _step_toward(target_pos_b, target_quat_b, gripper_target)
        print(f"[GATE G] held '{label}' for {steps} steps")

    open_target = torch.full((1, len(gripper_joint_ids)), 0.04, device=robot.device)
    close_target = torch.full((1, len(gripper_joint_ids)), 0.0, device=robot.device)

    def _joint_space_prep(target_joint_pos: list[float], gripper_target: torch.Tensor, steps: int, label: str) -> None:
        """Pure JOINT-SPACE (no Jacobian/IK) linear interpolation from the
        robot's CURRENT joint state to `target_joint_pos`, run BEFORE any
        Cartesian IK - see `_READY_TO_DESCEND_JOINT_POS`'s comment for why.
        Open-loop over a fixed step budget (not convergence-gated - this is
        a coarse pre-positioning move, not a precision waypoint); prints the
        final joint/hand state so a bad landing is visible, but does not
        raise (stage 1's Cartesian IK, now starting from a much better
        conditioned config, is expected to clean up any small residual)."""
        start = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].clone()
        target = torch.tensor(target_joint_pos, device=robot.device, dtype=torch.float32)
        print(
            f"[GATE G] {label}: joint-space interpolation from {start.cpu().numpy().round(3)} "
            f"to {target.cpu().numpy().round(3)} over {steps} steps"
        )
        for i in range(steps):
            alpha = (i + 1) / steps
            interp = start + alpha * (target - start)
            robot.set_joint_position_target(interp.unsqueeze(0), joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        final_joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()
        final_hand_pos = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        final_hand_quat = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
        joint_err = float(np.linalg.norm(final_joint_pos - target.cpu().numpy()))
        print(
            f"[GATE G] {label}: DONE. final_joint_pos={final_joint_pos.round(3)} joint_err_norm={joint_err:.4f}rad "
            f"hand_pos_w={final_hand_pos} hand_quat_w={final_hand_quat}"
        )

    stage1_hand_z = _STAGE1_HAND_Z
    stage2_hand_z = grasp_height_m + hand_to_fingertip_z  # grasp height, hand frame (fingertip at the die's MEASURED resting height, see _DIE_REST_HEIGHT_M)
    stage4_hand_z = _STAGE4_LIFT_HAND_Z

    print(
        f"[GATE G] staged pick sequence for '{choice}': target xy=({target_xy[0]:.4f},{target_xy[1]:.4f}) "
        f"grasp_height(measured)={grasp_height_m * 1000:.1f}mm stage1_hand_z={stage1_hand_z * 1000:.1f}mm "
        f"stage2_hand_z(grasp)={stage2_hand_z * 1000:.1f}mm stage4_hand_z(lift)={stage4_hand_z * 1000:.1f}mm"
    )

    def _print_live_hand_orientation(label: str) -> None:
        """Diagnostic: verify the ACTUAL live orientation stayed close to
        the canonical straight-down target, not just trust that it did -
        the "hand->fingertip pinch point" offset is only valid along world
        -Z if the hand is still (close to) pointing straight down."""
        live_quat = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
        w, x, y, z = live_quat
        R = _quat_to_rot_matrix(np.array([w, x, y, z]))
        approach_axis_world = R @ np.array([0.0, 0.0, 1.0])  # local +Z (finger-pointing axis) in world
        left_p = robot.data.body_pos_w[0, left_id].cpu().numpy()
        right_p = robot.data.body_pos_w[0, right_id].cpu().numpy()
        hand_p = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        print(
            f"[GATE G]   [{label}] live hand quat_w={live_quat} approach_axis_world={approach_axis_world} "
            f"(canonical target is [0,0,-1], straight down) hand_pos={hand_p} "
            f"left_finger_pos={left_p} right_finger_pos={right_p}"
        )

    waypoint_status = {}

    # Stage 0: joint-space pre-positioning (fallback #2 - see
    # _READY_TO_DESCEND_JOINT_POS's comment). Runs BEFORE any Cartesian IK,
    # entirely open-loop/direct joint targets - sidesteps the default pose's
    # Jacobian near-singularity that slammed joint4 to its hard limit when
    # stage 1's Cartesian IK was previously started directly from the
    # post-reset default.
    _joint_space_prep(_READY_TO_DESCEND_JOINT_POS, open_target, _MAX_STEPS_JOINT_PREP, "stage0_joint_prep")
    _print_live_hand_orientation("stage0_joint_prep")

    # Stage 1: approach - translate to (target_xy, stage1_hand_z) AND rotate
    # to the canonical down orientation TOGETHER (orientation is allowed to
    # move throughout this stage - it only needs to be exact BEFORE
    # descent). require_rot=True: both position and orientation must
    # converge before stage 2 begins.
    stage1_target_pos = _hand_target_xyz(stage1_hand_z)
    waypoint_status["stage1_approach"] = _go_to_pose(
        stage1_target_pos, canonical_down_quat_w, open_target, "stage1_approach",
        max_steps=_MAX_STEPS_APPROACH, require_rot=True,
    )
    _print_live_hand_orientation("stage1_approach")

    # Stage 2: pure vertical descent to grasp height, orientation HELD at
    # the now-converged canonical down quat (require_rot=True as a
    # correctness check - should already be satisfied from stage 1 and held
    # via the same per-step bounded rotation correction the whole way down).
    # Uses `_GRASP_POS_TOL` (tighter than stage1/4's `_WAYPOINT_TOL`) - see
    # that constant's comment: a small die's own radius is comparable to the
    # looser tolerance, so a "converged" waypoint there could still miss the
    # die entirely (measured: d4/d8 both converged at ~14mm under the old
    # single global tolerance and z-gain was ~0 - fingers closed beside the
    # die, not around it).
    stage2_target_pos = _hand_target_xyz(stage2_hand_z)
    try:
        waypoint_status["stage2_descend"] = _go_to_pose(
            stage2_target_pos, canonical_down_quat_w, open_target, "stage2_descend",
            max_steps=_MAX_STEPS_DESCEND, require_rot=True, pos_tol=_GRASP_POS_TOL,
        )
    except _StageTimeoutError as e:
        # Fallback (Principal's dispatch: "if a tighter tol alone can't
        # converge (oscillation floor), add a brief XY-only refine
        # sub-stage at grasp height"). Re-targets using the CURRENT live
        # hand z (freezing z, which should already be close from the main
        # descent above) so the SAME bounded relative-step mechanism has
        # ~nothing left to correct in z and effectively concentrates its
        # per-step budget on xy - no new control mechanism, just a
        # differently-shaped target.
        print(
            f"[GATE G] stage2_descend did not converge at pos_tol={_GRASP_POS_TOL * 1000:.1f}mm - "
            f"attempting XY-only refine sub-stage (fallback): {e}"
        )
        live_z = float(robot.data.body_pos_w[0, hand_body_id, 2].cpu().numpy())
        stage2_target_pos = _hand_target_xyz(live_z)
        waypoint_status["stage2_descend_xy_refine"] = _go_to_pose(
            stage2_target_pos, canonical_down_quat_w, open_target, "stage2_descend_xy_refine",
            max_steps=_MAX_STEPS_REFINE, require_rot=True, pos_tol=_GRASP_POS_TOL,
        )
    _print_live_hand_orientation("stage2_descend (grasp height, before close)")

    # Stage 3: close gripper, dwell.
    _hold(stage2_target_pos, canonical_down_quat_w, close_target, "stage3_close_gripper", _GRIPPER_CLOSE_HOLD_STEPS)
    _print_live_hand_orientation("stage3_after_close")

    # Stage 4: vertical lift.
    stage4_target_pos = _hand_target_xyz(stage4_hand_z)
    waypoint_status["stage4_lift"] = _go_to_pose(
        stage4_target_pos, canonical_down_quat_w, close_target, "stage4_lift",
        max_steps=_MAX_STEPS_LIFT, require_rot=True,
    )
    _print_live_hand_orientation("stage4_lift")

    return waypoint_status


def run_gate_g() -> None:
    choice = _normalize_choice(args_cli.choice)
    if choice not in DIE_TYPES:
        raise RuntimeError(f"Normalized choice '{choice}' is not one of the physical dice in this scene: {DIE_TYPES}")

    sim, scene, positions, results = spawn_scene_and_settle(GATE_G_DIR, args_cli.seed)
    settled_z = {die_type: results[die_type]["z"] for die_type in DIE_TYPES}

    detection_output = run_detector_subprocess(GATE_G_DIR)
    detections = detection_output["detections"]
    print(f"[GATE G] perception subprocess returned {len(detections)} detections:")
    for det in detections:
        print(f"  class={det['class']:<8} conf={det['confidence']:.3f} world_pos={det['world_pos']}")

    target_det = select_target_detection(detections, choice)
    det_x, det_y, det_z = target_det["world_pos"]
    print(
        f"[GATE G] target detection for '{choice}': class={target_det['class']} "
        f"conf={target_det['confidence']:.3f} world_pos=({det_x:.4f}, {det_y:.4f}, {det_z:.4f})"
    )

    # Diagnostic-only GT comparison (measure, don't blind-nudge - CLAUDE.md's
    # explicit guidance re: this repo's own AR4-era unexplained IK misses).
    # This value is NEVER used to compute the grasp target - x/y come only
    # from the detection above.
    gt_pos = np.array([results[choice]["x"], results[choice]["y"], results[choice]["z"]])
    det_pos = np.array([det_x, det_y, det_z])
    full_err = float(np.linalg.norm(gt_pos - det_pos))
    xy_err = float(np.linalg.norm(gt_pos[:2] - det_pos[:2]))
    print(
        f"[GATE G] detector-vs-GT offset for '{choice}' [DIAGNOSTIC ONLY, not used for grasp]: "
        f"xy={xy_err * 1000:.1f}mm full-3d={full_err * 1000:.1f}mm "
        f"(gt={gt_pos.tolist()}, det={det_pos.tolist()})"
    )

    half_height_m = _die_half_height_m(choice)  # diagnostic only now, see _DIE_REST_HEIGHT_M's comment
    grasp_height_m = _die_grasp_height_m(choice)  # the value actually used for grasp math
    print(
        f"[GATE G] grasp height for '{choice}': half_height(manifest size_mm/2, DIAGNOSTIC ONLY)="
        f"{half_height_m * 1000:.1f}mm grasp_height(MEASURED resting height, actually used)="
        f"{grasp_height_m * 1000:.1f}mm"
    )
    target_xy = (det_x, det_y)

    # A stage timeout is a "fail loudly" signal (see _StageTimeoutError /
    # _go_to_pose), not a script crash: catch it here so the run still
    # produces a post-lift frame + verdict json (which will naturally show
    # FAIL via the GT z-gain check, with the real diagnostic - live hand
    # pos/orientation + commanded target - preserved in
    # `pick_sequence_error` for the report) and the sim still tears down
    # cleanly via __main__'s try/finally.
    pick_sequence_error = None
    try:
        waypoint_status = run_pick_sequence(sim, scene, target_xy, grasp_height_m, choice)
    except _StageTimeoutError as e:
        pick_sequence_error = str(e)
        waypoint_status = {"error": pick_sequence_error}
        print(f"[GATE G] *** pick sequence FAILED for choice={choice}: stage timeout - {pick_sequence_error} ***")

    # Post-lift camera capture (RTX convergence frames, same pattern as
    # spawn_scene_and_settle's own capture).
    sim_dt = sim.get_physics_dt()
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    camera = scene["camera"]
    rgb_post = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    post_lift_path = os.path.join(GATE_G_DIR, f"post_lift_{choice}.png")
    Image.fromarray(rgb_post).save(post_lift_path)
    print(f"[GATE G] saved post-lift frame: {post_lift_path}")

    # Success verification - GT ALLOWED HERE ONLY (this task's one exception).
    verdict_table = []
    all_ok = True
    for die_type in DIE_TYPES:
        die = scene[f"die_{die_type}"]
        pos = die.data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        z_now = float(pos[2])
        z_before = settled_z[die_type]
        gain = z_now - z_before
        is_target = die_type == choice
        ok = (gain >= _LIFT_SUCCESS_GAIN) if is_target else (z_now < _OTHER_DIE_MAX_Z)
        if not ok:
            all_ok = False
        verdict_table.append(
            {"die": die_type, "z_before_m": z_before, "z_now_m": z_now, "gain_m": gain, "is_target": is_target, "ok": ok}
        )

    print(f"[GATE G] post-lift verdict table (commanded die: {choice}):")
    print(f"{'die':<6}{'z_before(mm)':>14}{'z_now(mm)':>12}{'gain(mm)':>10}  {'target':^8}  verdict")
    for row in verdict_table:
        print(
            f"{row['die']:<6}{row['z_before_m'] * 1000:>14.1f}{row['z_now_m'] * 1000:>12.1f}"
            f"{row['gain_m'] * 1000:>10.1f}  {'*TARGET*' if row['is_target'] else '':^8}  "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )

    print(f"[GATE G] {choice}: {'PASS' if all_ok else 'FAIL'} (waypoints={waypoint_status})")

    result = {
        "choice": choice,
        "seed": args_cli.seed,
        "detected_class": target_det["class"],
        "detection_confidence": target_det["confidence"],
        "detector_world_pos": det_pos.tolist(),
        "gt_world_pos_at_settle": gt_pos.tolist(),
        "detector_vs_gt_xy_error_m": xy_err,
        "detector_vs_gt_full_error_m": full_err,
        "half_height_m": half_height_m,
        "grasp_height_m": grasp_height_m,
        "waypoint_status": waypoint_status,
        "pick_sequence_error": pick_sequence_error,
        "verdict_table": verdict_table,
        "gate_g_pass": bool(all_ok),
    }
    verdict_path = os.path.join(GATE_G_DIR, f"verdict_{choice}.json")
    with open(verdict_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[GATE G] saved verdict: {verdict_path}")
    print("[GATE G] DONE")

    if not all_ok:
        # Report loudly but don't raise: the verdict JSON + printed table are
        # the source of truth the controller/report reads, and a raised
        # exception here would prevent simulation_app.close() from running
        # cleanly (this repo's own documented teardown-hang failure mode).
        print(f"[GATE G] *** FAILED for choice={choice} - see verdict table above/verdict_{choice}.json ***")


def main() -> None:
    if args_cli.gate == "a":
        run_gate_a()
    elif args_cli.gate == "g":
        run_gate_g()
    else:
        sys.exit(f"--gate {args_cli.gate} not implemented in this script (only 'a' and 'g' are).")


if __name__ == "__main__":
    # try/finally so simulation_app.close() ALWAYS runs, even if main() raises
    # (e.g. run_detector_subprocess's / select_target_detection's intentional
    # hard failures - "fail loudly" per this task's brief). An uncaught
    # exception that skips simulation_app.close() is this repo's own
    # documented Kit-teardown-hang failure mode (CLAUDE.md: Kit's shutdown
    # can spin indefinitely at high CPU and orphan an Omniverse Hub process
    # holding the flock lock) - confirmed reproduced once in this task before
    # this fix (see report). Re-raises after cleanup so the process still
    # exits non-zero and the failure is still visible.
    try:
        main()
        print("[DONE] holding window briefly before close...")
        time.sleep(3.0)
    finally:
        simulation_app.close()

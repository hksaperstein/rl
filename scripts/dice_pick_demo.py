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
  P - perception bridge (implemented separately, see
      vision/scripts/detect_for_sim.py).
  V - demo video: runs Gate G's flow wholesale (see run_gate_g) and, via
      run_pick_sequence's optional `on_step` hook, records the whole
      pre-pick/approach/descend/close/lift/post-lift-dwell sequence from the
      scene's DiceCamera to outputs/dice_demo/gate_v/dice_pick_<choice>.mp4
      (imageio/libx264), with a post-hoc PIL bbox+label overlay (from that
      run's own detections.json) drawn on the opening ~1.5s showing which
      die was commanded and what the detector localized. Same GT verdict
      check as Gate G (must pass/fail the same way).

.. code-block:: bash

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate a --seed 42"

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate g --choice d20 --seed 42"

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate v --choice d20 --seed 42"

Colored-dice repeat (2026-07-11, see
.superpowers/sdd/dice-demo-colored-report.md): --colored-dice runtime-applies
each die's own manifest-derived body material (default OFF, white/near-white
baseline unaffected) and redirects output to outputs/dice_demo/colored/
instead of outputs/dice_demo/. --light-scale (default 1.0) independently
scales both scene lights, for testing the render-exposure-blowout hypothesis
(see apply_colored_material's and --light-scale's own docstrings/help for why
the body material alone was measured to NOT be the missing piece).

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate a --seed 42 \\
        --colored-dice --light-scale 0.3"

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
parser.add_argument(
    "--colored-dice",
    action="store_true",
    default=False,
    help=(
        "Apply each die's own manifest-derived (hue/saturation/value -> RGB) UsdPreviewSurface body "
        "material at spawn time, runtime-patched onto the die's mesh prim (see apply_colored_material). "
        "Default OFF - the white/near-white baseline (unmodified dice USD materials, as measured pre-fix) "
        "stays reproducible without this flag. Outputs are redirected to outputs/dice_demo/colored/ "
        "instead of outputs/dice_demo/ when this is set, so colored-dice runs never overwrite the "
        "white-baseline gate outputs."
    ),
)
parser.add_argument(
    "--light-scale",
    type=float,
    default=1.0,
    help=(
        "Multiplies BOTH scene lights' (DomeLight, DistantLight - tasks/franka/dice_scene_cfg.py) "
        "intensity by this factor at spawn time. Default 1.0 (scene's authored default, unchanged). "
        "Diagnostic/experimental knob (2026-07-11 colored-dice repeat task): the dice USDs' own body "
        "material is measured (scripts/_diag_dice_material_check.py) to already be correctly authored "
        "and bound (diffuseColor exactly matches colorsys.hsv_to_rgb of each die's manifest "
        "hue/saturation/value) - the baseline's near-white rendered appearance is hypothesized to be a "
        "render-time exposure/blowout artifact, not a missing/lost material, since dice_scene_cfg.py's "
        "DistantLight (intensity 3000, added on top of the DomeLight already present at the same "
        "intensity in the validated tasks/franka/lift_env_cfg.py baseline) roughly doubles that "
        "baseline's total light energy. This flag lets one capture test that hypothesis without "
        "permanently changing the scene's default lighting."
    ),
)
parser.add_argument(
    "--gt-xy-bypass",
    action="store_true",
    default=False,
    help=(
        "Grasp-mechanism-isolation bypass for Gates G/V (see "
        "docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md's 'Addendum: ground-truth "
        "XY-bypass'). Default OFF - every die type's/existing call's target_xy still comes ONLY from "
        "select_target_detection's detector-sourced result, byte-identical to pre-flag behavior. When "
        "set, target_xy is instead sourced from the commanded die's own settled ground-truth (x, y) "
        "(the same value already computed, diagnostic-only, as gt_pos in both gates) - the detector "
        "subprocess still runs and select_target_detection's own 'fails loudly, never falls back to "
        "ground truth' contract is UNCHANGED (this bypass is a separate, explicit branch in the caller, "
        "not a fallback inside that function), so the diagnostic detector-vs-GT comparison print/verdict "
        "fields stay populated either way. This isolates the grasp-mechanism variable from the (currently "
        "under separate investigation) d4 perception weakness - same isolation principle rung 0 already "
        "used for orientation. NOT a fix for detection and must not be represented as one; a bypassed run "
        "claims only 'the grasp mechanism works', not 'the perception-driven demo can pick this die'."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# NEVER set args_cli.headless - non-headless is environment law here (a
# display exists, DISPLAY=:1, the user wants to watch). Leave it at
# AppLauncher's own default (False / unset).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows - isaaclab/pxr imports must come after AppLauncher."""

import colorsys  # noqa: E402

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

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
from tasks.franka.notch_fixture import (  # noqa: E402
    grip_height_above_table_m,
    joint_local_pos0_m,
    joint_local_rot1_wxyz,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_A_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_a")
GATE_G_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_g")
GATE_V_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_v")
# --colored-dice redirects every gate's output dir under here instead, so a
# colored-dice run NEVER overwrites the white-baseline gate_a/gate_g/gate_v
# outputs above (the report needs both, side by side, for comparison).
COLORED_ROOT_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "colored")
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
_D4_CONTACT_FORCE_THRESHOLD_N = 0.05  # N, minimum net contact force to count as "touching" - matches this repo's
# own precedent for the AR4 gripper-vs-sphere case (scripts/classical_grasp_contact_check.py's
# FORCE_THRESHOLD/BILATERAL_CONTACT_THRESHOLD), not a new value invented for this task.
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
_OTHER_DIE_MIN_Z = -0.02  # m, plausible on-table lower bound - below this means knocked off the
# table onto the ground plane (z=-1.05), not merely settled/jostled in place.
_OTHER_DIE_MAX_XY_DRIFT = 0.05  # m, every OTHER die's xy displacement from its settled position
# must stay under this - a z-only/upper-bound-only check alone would silently pass a die swept
# sideways without ever crossing _OTHER_DIE_MAX_Z (final whole-branch review finding 1).

# Gate V (demo video) tuning. Reuses Gate G's flow wholesale (spawn/settle ->
# detector subprocess -> target select -> run_pick_sequence -> GT verdict)
# and adds per-step frame capture via run_pick_sequence's optional `on_step`
# hook (purely additive - Gate G's own call site passes on_step=None, so
# Gate G's already-validated behavior is completely unchanged).
_VIDEO_FRAME_STRIDE = 2  # capture every 2nd physics step (halves frame count/memory vs. every step; 60Hz physics -> 30fps video, still smooth enough to check motion continuity around the grasp moment per the brief)
_PRE_PICK_SECONDS = 1.5  # s, idle-scene segment captured BEFORE run_pick_sequence starts (arm stationary at its post-reset default) - this is ALSO the overlay window (see run_gate_v): the brief asks for the bbox overlay on "the opening ~1.5s", so making the pre-pick segment exactly that long means the overlay covers precisely the pre-pick segment, no separate frame-count math needed
_POST_LIFT_DWELL_SECONDS = 2.0  # s, held-lift segment captured AFTER run_pick_sequence returns (arm holds its last commanded lift pose/closed gripper via the PD controller - no new commands issued), so the video doesn't cut off right at the last waypoint

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


# d4 rung-1 V-notch fingertip fixture attachment (2026-07-15, see
# docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md and
# .superpowers/sdd/task-1-brief.md). Rung-0's tilted-axis edge-grasp branch
# (read_die_local_vertices/edge_pair_grasp_axes/best_reachable_pair/
# stage_waypoints_world - all previously imported/called here) has been
# REMOVED from this file, not extended: rung-0 was falsified at the
# reachability level (spec's own "Prior result"), and rung 1's whole premise
# is that the ORIGINAL straight-down approach (already reachable/working for
# d8/d10/d12/d20) plus this notch fixture is sufficient - no per-die tilted
# waypoint computation is needed anymore. `tasks/franka/antipodal_edge_grasp.py`
# and its own unit tests (tests/test_antipodal_edge_grasp.py) are NOT
# deleted (kept on record per the spec's "Future work" - the asymmetric
# edge/face-ramp refinement, if ever built, may reuse its mesh-vertex-
# extraction utilities), just no longer imported/called from this file.


def attach_notch_fixtures(stage, robot_prim_path: str) -> None:
    """Rigidly attaches the notch fixture (tasks/franka/notch_fixture.py,
    authored offline by scripts/build_notch_fixture_asset.py into
    `{ENV_REGEX_NS}/NotchFixtureLeft`/`NotchFixtureRight`, see
    dice_scene_cfg.py) to `panda_leftfinger`/`panda_rightfinger` via a new
    `UsdPhysics.FixedJoint` per side - UNCONDITIONAL (called for every die
    type, not gated on `choice`), per the spec's North Star call. Must run
    BEFORE `sim.reset()` (matches `apply_convex_hull_collision`'s own
    runtime-schema-patching timing), so PhysX cooks the joint constraint
    into its very first physics step rather than needing to correct an
    already-settled mismatch.

    Each joint prim is authored as a CHILD OF THE FIXTURE prim (a plain,
    non-instanced prim this file fully owns), NOT as a child of the finger
    prim - `panda_leftfinger`/`panda_rightfinger` are instance proxies
    (Task 0's own finding: this Franka asset is instanceable), and USD does
    not allow authoring new child prims directly under an instance proxy's
    own path. `body0`/`body1` RELATIONSHIPS on the joint prim can still
    TARGET an instance-proxy path with no such restriction (this repo's own
    `d4_leftfinger_contact`/`d4_rightfinger_contact` `ContactSensorCfg`
    already relies on exactly that - unaffected by where the joint prim
    itself happens to live).

    Before creating each joint, this function overwrites the fixture
    prim's own initial WORLD transform (translate only) to the finger's
    OWN live tip-attachment-point world position (via `UsdGeom.XformCache`,
    the same local-to-world technique
    scripts/_diag_franka_fingertip_geometry.py already established for
    Task 0) - a stability nicety (avoids relying on
    dice_scene_cfg.py's own placeholder `init_state.pos` guess), not a
    correctness requirement (the fixed joint's own `localPos0`/`localPos1`
    offsets are what define the enforced pose once physics starts, not the
    initial placement)."""
    xform_cache = UsdGeom.XformCache()
    local_pos0 = joint_local_pos0_m()

    for finger_name, fixture_name, mirror in [
        ("panda_leftfinger", "NotchFixtureLeft", False),
        ("panda_rightfinger", "NotchFixtureRight", True),
    ]:
        finger_prim = stage.GetPrimAtPath(f"{robot_prim_path}/{finger_name}")
        if not finger_prim.IsValid():
            raise RuntimeError(f"attach_notch_fixtures: finger prim not found: {robot_prim_path}/{finger_name}")
        # fixture prims live at the env root (siblings of Robot/Die_*, see
        # dice_scene_cfg.py), not under robot_prim_path.
        env_root = os.path.dirname(robot_prim_path)
        fixture_prim = stage.GetPrimAtPath(f"{env_root}/{fixture_name}")
        if not fixture_prim.IsValid():
            raise RuntimeError(f"attach_notch_fixtures: fixture prim not found: {env_root}/{fixture_name}")

        finger_to_world = xform_cache.GetLocalToWorldTransform(finger_prim)
        attach_point_world = finger_to_world.Transform(Gf.Vec3d(*local_pos0))
        UsdGeom.XformCommonAPI(fixture_prim).SetTranslate(attach_point_world)
        print(
            f"[SPAWN] attach_notch_fixtures: {fixture_name} initial world pos set to "
            f"{tuple(round(c, 5) for c in attach_point_world)} (from {finger_name}'s live tip transform)"
        )

        joint_path = f"{fixture_prim.GetPath()}/attach_joint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([finger_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([fixture_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rot1_wxyz = joint_local_rot1_wxyz(mirror=mirror)
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(rot1_wxyz[0], Gf.Vec3f(*rot1_wxyz[1:])))
        print(
            f"[SPAWN] attach_notch_fixtures: created FixedJoint {joint_path} "
            f"(body0={finger_prim.GetPath()}, body1={fixture_prim.GetPath()}, "
            f"localPos0={local_pos0}, localRot1_wxyz={rot1_wxyz})"
        )


def _die_material_params(die_type: str) -> dict:
    """Reads `material_params` (hue/saturation/value/roughness/...) from this
    die type's own set_00000_<type>.json manifest. d10_pct maps to d10's
    manifest (same physical die in this scene, see D10_ALIASES)."""
    manifest_type = "d10" if die_type in D10_ALIASES else die_type
    manifest_path = os.path.join(DICE_MANIFEST_DIR, f"set_00000_{manifest_type}.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["material_params"]


def apply_colored_material(stage, die_prim_path: str, die_type: str) -> tuple[int, tuple[float, float, float]]:
    """--colored-dice helper: creates a UsdPreviewSurface material from this
    die's own manifest `material_params` (hue/saturation/value ->
    colorsys.hsv_to_rgb for diffuseColor, roughness passed through directly)
    and binds it to every UsdGeom.Mesh prim under `die_prim_path` (same
    runtime-patching pattern as apply_convex_hull_collision above).

    NOTE (2026-07-11 investigation, scripts/_diag_dice_material_check.py):
    every set_00000 die USD was measured to ALREADY carry a correctly
    authored + bound UsdPreviewSurface body material whose diffuseColor is
    an EXACT match (to 6 decimal places) for colorsys.hsv_to_rgb of that
    die's own manifest hue/saturation/value - material authorship was never
    lost in the Blender->USD export. This function's practical effect is
    therefore mostly a no-op on color (it recomputes and rebinds the same
    RGB the source USD already has) - kept as a deliberate belt-and-suspenders
    guarantee (works even if a future asset regenerate ever drops the body
    material) and because binding at the mesh-prim level here does NOT
    disturb the per-face numeral-decal materials, which are bound at a more
    specific UsdGeomSubset level that wins USD's material-binding-strength
    resolution for their own face indices. The actual near-white baseline
    symptom this task investigates is a RENDER-time exposure/blowout effect
    (see --light-scale's help text), not a missing-material one - this
    function alone does not fix that; see the --light-scale flag.

    Returns (mesh_count_bound, rgb) for the caller's own logging."""
    params = _die_material_params(die_type)
    hue, sat, val = float(params["hue"]), float(params["saturation"]), float(params["value"])
    roughness = float(params["roughness"])
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

    mat_path = f"{die_prim_path}/ColoredBodyMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((r, g, b))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")
    mesh_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            mesh_count += 1
    return mesh_count, (r, g, b)


def spawn_scene_and_settle(
    out_dir: str, seed: int, colored_dice: bool = False, light_scale: float = 1.0
) -> tuple[sim_utils.SimulationContext, InteractiveScene, list, dict]:
    """Runs the shared Gate A flow: sample layout, spawn scene, apply
    runtime collision schemas, sim.reset()+scene.reset(), settle physics,
    verify every die's z/xy bounds, save gt_dice.json + an RGB-D camera
    frame to `out_dir`. Leaves the sim/scene LIVE (does not close
    simulation_app) so callers (Gate G) can keep driving the robot.

    `colored_dice` (default False, preserves the white/near-white baseline):
    after collision-schema patching, additionally runs apply_colored_material
    on every die - see that function's docstring for what it does and does
    NOT fix.

    `light_scale` (default 1.0, preserves the scene's authored default):
    multiplies both DomeLight and DistantLight intensity in scene_cfg BEFORE
    scene construction - a diagnostic/experimental knob for testing the
    render-exposure-blowout hypothesis (see --light-scale's argparse help),
    not a permanent scene default change.

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

    if light_scale != 1.0:
        orig_dome = scene_cfg.light.spawn.intensity
        orig_distant = scene_cfg.sun.spawn.intensity
        scene_cfg.light.spawn.intensity = orig_dome * light_scale
        scene_cfg.sun.spawn.intensity = orig_distant * light_scale
        print(
            f"[SPAWN] --light-scale={light_scale}: DomeLight intensity {orig_dome} -> "
            f"{scene_cfg.light.spawn.intensity}, DistantLight intensity {orig_distant} -> "
            f"{scene_cfg.sun.spawn.intensity} (scene default NOT permanently changed, "
            "this is a per-run override only)"
        )

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
        if colored_dice:
            mat_mesh_count, rgb = apply_colored_material(stage, die_prim_path, die_type)
            print(
                f"[SPAWN] --colored-dice: applied manifest-derived UsdPreviewSurface "
                f"(diffuseColor={tuple(round(c, 4) for c in rgb)}) to {die_type} "
                f"({mat_mesh_count} mesh prim(s) bound)"
            )

    # d4 rung-1 V-notch fingertip fixture (2026-07-15) - UNCONDITIONAL, every
    # scene spawn regardless of --choice, per the spec's North Star call (see
    # attach_notch_fixtures's own docstring). Must run before sim.reset(),
    # same ordering as the dice collision-schema patching above.
    attach_notch_fixtures(stage, f"{env_root}/Robot")

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
        # quat_wxyz: settled world-frame ORIENTATION (w,x,y,z), added
        # alongside x/y/z (d4 edge-grasp rung-0 task, 2026-07-13) - purely
        # additive to this dict (every existing "x"/"y"/"z"/"sampled_x"/
        # "sampled_y" key/consumer is untouched). No longer read by
        # run_pick_sequence at all (rung 0's tilted-axis branch, the one
        # consumer that needed it, was removed in the 2026-07-15 rung-1
        # task - see that function's own docstring) - kept in this dict as
        # diagnostic-only for every die type, harmless to leave in place.
        quat_wxyz = die.data.root_quat_w[0].cpu().numpy()
        results[die_type] = {
            "x": x, "y": y, "z": z, "sampled_x": sampled_pos[0], "sampled_y": sampled_pos[1],
            "quat_wxyz": quat_wxyz.tolist(),
        }

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

    # Diagnostic-only whole-arm view (2026-07-11 colored-dice repeat task's
    # Franka material check - see dice_scene_cfg.py's ARM_CAMERA_POS comment
    # for why this is a SEPARATE camera from DiceCamera, never used for any
    # gate's pass/fail logic). Always captured (cheap - one more sensor
    # readout), independent of --colored-dice/--light-scale.
    arm_camera = scene["arm_camera"]
    arm_rgb = arm_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    arm_rgb_path = os.path.join(out_dir, "arm_camera_rgb.png")
    Image.fromarray(arm_rgb).save(arm_rgb_path)
    print(f"[SPAWN] saved whole-arm diagnostic camera frame: {arm_rgb_path}")

    return sim, scene, positions, results


def run_gate_a() -> None:
    out_dir = os.path.join(COLORED_ROOT_DIR, "gate_a") if args_cli.colored_dice else GATE_A_DIR
    spawn_scene_and_settle(out_dir, args_cli.seed, colored_dice=args_cli.colored_dice, light_scale=args_cli.light_scale)
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
    for a straight-down grasp of this die type.

    For d8/d10/d12/d20: the die's MEASURED resting height
    (`_DIE_REST_HEIGHT_M`, unchanged from before this task) - close enough
    to these dice's own centroid that gripping there gives real contact
    area against their roundish geometry.

    For d4 (rung-1 V-notch fixture, 2026-07-15, see
    docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md):
    `_DIE_REST_HEIGHT_M["d4"]` (2.2mm, the die's own centroid height) is
    deliberately NOT used here anymore - that number describes where a
    FLAT pad would need to close to straddle the die's centroid, but the
    notch fixture's whole design point is to grip HIGHER on the pyramid
    (`tasks.franka.notch_fixture.grip_height_above_table_m`, ~9.3mm above
    the table - 10mm below the d4's own apex), where the local
    cross-section is wide enough for the notch's two walls to make flush
    facet contact instead of a near-point contact at the flat centroid
    height. d10_pct maps to d10's measured height (same physical die in
    this scene, not affected by the d4 special-case)."""
    if die_type == "d4":
        return grip_height_above_table_m()
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
    fresh shell.

    Scene-contract plumbing (final whole-branch review finding 2): forwards
    THIS demo's own _REGION_X/_REGION_Y and DIE_TYPES to the detector's
    --x-min/--x-max/--y-min/--y-max/--expected-classes flags, rather than
    relying on detect_for_sim.py's own hardcoded argparse defaults/
    EXPECTED_SCENE_CLASSES to happen to stay in sync with this file's copies
    of the same scene contract."""
    cmd = [
        VISION_VENV_PYTHON, DETECT_SCRIPT,
        "--input-dir", out_dir, "--output-dir", out_dir,
        "--x-min", str(_REGION_X[0]), "--x-max", str(_REGION_X[1]),
        "--y-min", str(_REGION_Y[0]), "--y-max", str(_REGION_Y[1]),
        "--expected-classes", ",".join(DIE_TYPES),
    ]
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
    on_step: "callable | None" = None,
    results: dict | None = None,
) -> dict:
    """Drives the Franka arm through a staged pick sequence (joint-space
    ready-to-descend prep -> Cartesian approach -> descend -> close -> lift)
    via a raw DifferentialIKController (pattern:
    IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py). Targets the
    CANONICAL straight-down orientation for EVERY die type, including d4
    (2026-07-15 rung-1 change - see
    docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md).
    Rung 0's separate tilted-axis d4 branch (a different target orientation,
    computed per-run from the die's own resting pose) has been REMOVED, not
    kept dormant: rung 0 was falsified at the reachability level (the
    spec's own "Prior result"), and rung 1's whole premise is that the
    ORIGINAL straight-down approach - already reachable/working for
    d8/d10/d12/d20 - is sufficient once the fingertips carry the notch
    fixture (`attach_notch_fixtures`, unconditional, every die type); d4 is
    therefore no longer geometrically special relative to the other four
    dice at the CONTROL-PATH level (only its own `_die_grasp_height_m`
    target height differs, same as every other die type already has its
    own height). Raises `_StageTimeoutError` (loudly, with live hand
    pos/orientation + the commanded target in the message) if any stage
    doesn't converge within its step budget - callers should catch this
    rather than treat a stuck stage as silent success. Returns a dict of
    per-stage convergence status (all True if this returns at all, since a
    non-convergent stage raises instead) - when `choice == "d4"`, this dict
    additionally carries the closure-window contact/displacement
    instrumentation keys (see the `if choice == "d4":` blocks around
    stage 3/stage 4 below), reading the fixture-targeted contact sensors
    dice_scene_cfg.py retargeted for this same rung.

    `on_step`, if given, is called once after EVERY physics step this
    function takes (stage 0's joint-space prep and every `_step_toward` call
    inside stages 1/2/3/4) - Gate V's frame-capture hook (see run_gate_v).
    Purely additive: Gate G's own call site passes on_step=None (the
    default), so Gate G's already-validated mechanism/timing is completely
    unchanged by this parameter's existence.

    `results` is accepted for call-site compatibility with the (now
    removed) rung-0 d4 branch's own signature - no longer read by this
    function at all (the common straight-down path never needed the die's
    resting orientation; only `target_xy`/`grasp_height_m`, both
    detector/measured-height-derived, are used)."""
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
        if on_step is not None:
            on_step()
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
            if on_step is not None:
                on_step()
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
    # post-reset default. Shared by every die type (choice-independent).
    _joint_space_prep(_READY_TO_DESCEND_JOINT_POS, open_target, _MAX_STEPS_JOINT_PREP, "stage0_joint_prep")
    _print_live_hand_orientation("stage0_joint_prep")

    # 2026-07-15 rung-1 change: d4 now takes the EXACT SAME common
    # straight-down path as d8/d10/d12/d20 below (rung 0's separate tilted-
    # axis branch, previously here, has been REMOVED - see this function's
    # own docstring and .superpowers/sdd/task-1-report.md for why). Only the
    # d4-specific closure/post-lift CONTACT INSTRUMENTATION (reading the
    # notch-fixture-targeted `d4_leftfinger_contact`/`d4_rightfinger_contact`
    # sensors dice_scene_cfg.py already retargeted for this rung) is still
    # gated on `choice == "d4"`, appended additively around stage 3/4 below -
    # not a separate control-flow branch.

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

    # Stage 3: close gripper, dwell. d4-only instrumentation bracketing the
    # hold (die-position before/after, per the spec's "closure-window
    # displacement" verification requirement) - additive, not a separate
    # code path; every other die type skips straight to `_hold` below.
    d4_die_pos_before_close = None
    if choice == "d4":
        d4_die_pos_before_close = (
            scene["die_d4"].data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        )
    _hold(stage2_target_pos, canonical_down_quat_w, close_target, "stage3_close_gripper", _GRIPPER_CLOSE_HOLD_STEPS)
    _print_live_hand_orientation("stage3_after_close")

    if choice == "d4":
        # Closure-window displacement + contact-force/point instrumentation
        # (2026-07-13, Task 2; RETAINED across the 2026-07-15 rung-1 branch
        # removal above - this is verification instrumentation, not part of
        # the removed tilted-axis control path). scene.update() runs every
        # sim step inside `_hold` above, so these sensors' buffers already
        # reflect the just-completed closure hold - no extra step needed.
        # force_matrix_w shape is (num_envs=1, num_bodies=1, num_filters=1,
        # 3); contact_pos_w is NaN when no contact is registered
        # (track_contact_points=True). `d4_leftfinger_contact`/
        # `d4_rightfinger_contact` now target the NOTCH FIXTURE prims, not
        # the bare finger bodies (dice_scene_cfg.py, this same rung-1 task) -
        # see that module's own comment for why.
        die_pos_after_close = (
            scene["die_d4"].data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        )
        closure_delta = die_pos_after_close - d4_die_pos_before_close
        closure_lateral_ejection_m = float(np.linalg.norm(closure_delta[:2]))
        closure_full_displacement_m = float(np.linalg.norm(closure_delta))
        ee_pos_at_closure = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        ee_quat_at_closure = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()

        left_contact = scene["d4_leftfinger_contact"]
        right_contact = scene["d4_rightfinger_contact"]
        left_force_vec_n = left_contact.data.force_matrix_w[0, 0, 0].cpu().numpy()
        right_force_vec_n = right_contact.data.force_matrix_w[0, 0, 0].cpu().numpy()
        left_force_n = float(np.linalg.norm(left_force_vec_n))
        right_force_n = float(np.linalg.norm(right_force_vec_n))
        left_contact_point_w = left_contact.data.contact_pos_w[0, 0, 0].cpu().numpy()
        right_contact_point_w = right_contact.data.contact_pos_w[0, 0, 0].cpu().numpy()
        bilateral_contact = (
            left_force_n > _D4_CONTACT_FORCE_THRESHOLD_N and right_force_n > _D4_CONTACT_FORCE_THRESHOLD_N
        )
        print(
            f"[GATE G] d4 closure-window die displacement: lateral(xy)={closure_lateral_ejection_m * 1000:.2f}mm "
            f"full3d={closure_full_displacement_m * 1000:.2f}mm (pos before={d4_die_pos_before_close} "
            f"after={die_pos_after_close}) ee_pos_at_closure={ee_pos_at_closure} "
            f"ee_quat_at_closure={ee_quat_at_closure}"
        )
        print(
            f"[GATE G] d4 closure-window contact: left_force={left_force_n:.4f}N right_force={right_force_n:.4f}N "
            f"bilateral={bilateral_contact} (threshold={_D4_CONTACT_FORCE_THRESHOLD_N}N) "
            f"left_contact_pt_w={left_contact_point_w} right_contact_pt_w={right_contact_point_w}"
        )
        waypoint_status["closure_lateral_ejection_m"] = closure_lateral_ejection_m
        waypoint_status["closure_full_displacement_m"] = closure_full_displacement_m
        waypoint_status["closure_ee_pos_w"] = ee_pos_at_closure.tolist()
        waypoint_status["closure_ee_quat_w"] = ee_quat_at_closure.tolist()
        waypoint_status["closure_left_finger_force_n"] = left_force_n
        waypoint_status["closure_right_finger_force_n"] = right_force_n
        waypoint_status["closure_bilateral_contact"] = bilateral_contact
        waypoint_status["closure_left_contact_pt_w"] = (
            None if np.isnan(left_contact_point_w).any() else left_contact_point_w.tolist()
        )
        waypoint_status["closure_right_contact_pt_w"] = (
            None if np.isnan(right_contact_point_w).any() else right_contact_point_w.tolist()
        )

    # Stage 4: vertical lift.
    stage4_target_pos = _hand_target_xyz(stage4_hand_z)
    waypoint_status["stage4_lift"] = _go_to_pose(
        stage4_target_pos, canonical_down_quat_w, close_target, "stage4_lift",
        max_steps=_MAX_STEPS_LIFT, require_rot=True,
    )
    _print_live_hand_orientation("stage4_lift")

    if choice == "d4":
        # Post-lift contact re-read (sustained-grip evidence, distinct from
        # the closure-window reading above) - same sensors, same threshold.
        post_lift_left_force_n = float(
            np.linalg.norm(scene["d4_leftfinger_contact"].data.force_matrix_w[0, 0, 0].cpu().numpy())
        )
        post_lift_right_force_n = float(
            np.linalg.norm(scene["d4_rightfinger_contact"].data.force_matrix_w[0, 0, 0].cpu().numpy())
        )
        post_lift_bilateral_contact = (
            post_lift_left_force_n > _D4_CONTACT_FORCE_THRESHOLD_N
            and post_lift_right_force_n > _D4_CONTACT_FORCE_THRESHOLD_N
        )
        print(
            f"[GATE G] d4 post-lift contact: left_force={post_lift_left_force_n:.4f}N "
            f"right_force={post_lift_right_force_n:.4f}N bilateral={post_lift_bilateral_contact}"
        )
        waypoint_status["post_lift_left_finger_force_n"] = post_lift_left_force_n
        waypoint_status["post_lift_right_finger_force_n"] = post_lift_right_force_n
        waypoint_status["post_lift_bilateral_contact"] = post_lift_bilateral_contact

    return waypoint_status


def _compute_verdict_table(scene: InteractiveScene, results: dict, choice: str) -> tuple[list[dict], bool]:
    """Shared post-lift GT verdict-table computation - previously duplicated
    byte-for-byte inline in both run_gate_g and run_gate_v (final
    whole-branch review finding 8); now a single implementation called from
    both.

    `results` is spawn_scene_and_settle's per-die dict of settled x/y/z
    (GT, world minus env-origin frame) - GT ALLOWED HERE ONLY, this task's
    one exception.

    Target die: success requires an upward z gain of at least
    _LIFT_SUCCESS_GAIN.

    Every OTHER (non-target) die: "undisturbed" requires BOTH (a) xy
    displacement from its settled position under _OTHER_DIE_MAX_XY_DRIFT,
    AND (b) z still within the plausible on-table band
    [_OTHER_DIE_MIN_Z, _OTHER_DIE_MAX_Z). The prior z-only/upper-bound-only
    check (`z_now < _OTHER_DIE_MAX_Z` alone) would silently record ok=True
    for a die swept sideways by the gripper, or knocked off the table
    entirely onto the ground plane at z=-1.05 (which is still < the old
    0.05 upper bound) - final whole-branch review finding 1."""
    verdict_table = []
    all_ok = True
    for die_type in DIE_TYPES:
        die = scene[f"die_{die_type}"]
        pos = die.data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        x_now, y_now, z_now = float(pos[0]), float(pos[1]), float(pos[2])
        x_before = results[die_type]["x"]
        y_before = results[die_type]["y"]
        z_before = results[die_type]["z"]
        gain = z_now - z_before
        xy_drift = float(np.hypot(x_now - x_before, y_now - y_before))
        is_target = die_type == choice
        if is_target:
            ok = gain >= _LIFT_SUCCESS_GAIN
        else:
            ok = (xy_drift < _OTHER_DIE_MAX_XY_DRIFT) and (_OTHER_DIE_MIN_Z <= z_now < _OTHER_DIE_MAX_Z)
        if not ok:
            all_ok = False
        verdict_table.append(
            {
                "die": die_type,
                "z_before_m": z_before,
                "z_now_m": z_now,
                "gain_m": gain,
                "xy_drift_m": xy_drift,
                "is_target": is_target,
                "ok": ok,
            }
        )
    return verdict_table, all_ok


def run_gate_g() -> None:
    choice = _normalize_choice(args_cli.choice)
    if choice not in DIE_TYPES:
        raise RuntimeError(f"Normalized choice '{choice}' is not one of the physical dice in this scene: {DIE_TYPES}")

    out_dir = os.path.join(COLORED_ROOT_DIR, "gate_g") if args_cli.colored_dice else GATE_G_DIR
    sim, scene, positions, results = spawn_scene_and_settle(
        out_dir, args_cli.seed, colored_dice=args_cli.colored_dice, light_scale=args_cli.light_scale
    )

    detection_output = run_detector_subprocess(out_dir)
    detections = detection_output["detections"]
    print(f"[GATE G] perception subprocess returned {len(detections)} detections:")
    for det in detections:
        print(f"  class={det['class']:<8} conf={det['confidence']:.3f} world_pos={det['world_pos']}")

    # 2026-07-15 fix: the first --gt-xy-bypass implementation called
    # select_target_detection() unconditionally above this point, so a
    # TOTAL detection miss (0 candidates for `choice`, exactly d4's
    # observed failure mode) raised RuntimeError before the bypass branch
    # below ever ran - the bypass only ever helped an INACCURATE
    # detection, never a MISSING one, which is the dominant failure mode
    # it exists to route around (found live on a cloud rerun, seed 42,
    # see .superpowers/sdd/task-2-report.md). Fix: only let the exception
    # propagate when the bypass is OFF (preserves today's exact fail-loud
    # behavior for every non-bypass call); when the bypass is ON, catch
    # it, record that no real detection exists, and continue - GT alone
    # is sufficient to compute target_xy below, no detection required.
    target_det: dict | None = None
    det_x = det_y = det_z = None
    try:
        target_det = select_target_detection(detections, choice)
        det_x, det_y, det_z = target_det["world_pos"]
        print(
            f"[GATE G] target detection for '{choice}': class={target_det['class']} "
            f"conf={target_det['confidence']:.3f} world_pos=({det_x:.4f}, {det_y:.4f}, {det_z:.4f})"
        )
    except RuntimeError as e:
        if not args_cli.gt_xy_bypass:
            raise
        print(
            f"[GATE G] detector FAILED to find '{choice}' ({e}) - continuing because "
            f"--gt-xy-bypass is active; target_xy will be sourced from ground truth below, "
            f"detector-vs-GT diagnostic print skipped (nothing to compare against)."
        )

    gt_pos = np.array([results[choice]["x"], results[choice]["y"], results[choice]["z"]])

    # Diagnostic-only GT comparison (measure, don't blind-nudge - CLAUDE.md's
    # explicit guidance re: this repo's own AR4-era unexplained IK misses).
    # This value is NEVER used to compute the grasp target - x/y come only
    # from the detection above. Only meaningful when a detection exists;
    # None when the bypass caught a total detection miss (nothing to
    # compare GT against) - the verdict JSON below records this explicitly
    # rather than crashing on a None.
    det_pos = None
    xy_err = None
    full_err = None
    if target_det is not None:
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
    # Ground-truth XY-bypass (2026-07-15, see this task's brief and the
    # spec's "Addendum: ground-truth XY-bypass"): --gt-xy-bypass is OFF by
    # default, so target_xy stays sourced from the detector above -
    # byte-identical to pre-flag behavior (target_det is guaranteed
    # non-None here in that case, since the except-block above re-raises
    # when the bypass is off). select_target_detection's own "fails
    # loudly, never falls back to GT" contract is untouched for the
    # non-bypass path - this is a separate, explicit override of
    # target_xy in the CALLER, not a fallback inside that function.
    if args_cli.gt_xy_bypass:
        target_xy = (float(gt_pos[0]), float(gt_pos[1]))
        print(
            f"[GATE G] target_xy SOURCED FROM GROUND TRUTH (bypass active) for '{choice}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f}) - grasp-mechanism isolation only, NOT a "
            f"perception result; detector-vs-GT diagnostic above (if printed) still reflects the "
            f"real detector output."
        )
    else:
        target_xy = (det_x, det_y)
        print(
            f"[GATE G] target_xy sourced from DETECTOR for '{choice}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f})"
        )

    # A stage timeout is a "fail loudly" signal (see _StageTimeoutError /
    # _go_to_pose), not a script crash: catch it here so the run still
    # produces a post-lift frame + verdict json (which will naturally show
    # FAIL via the GT z-gain check, with the real diagnostic - live hand
    # pos/orientation + commanded target - preserved in
    # `pick_sequence_error` for the report) and the sim still tears down
    # cleanly via __main__'s try/finally.
    pick_sequence_error = None
    try:
        waypoint_status = run_pick_sequence(sim, scene, target_xy, grasp_height_m, choice, results=results)
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
    post_lift_path = os.path.join(out_dir, f"post_lift_{choice}.png")
    Image.fromarray(rgb_post).save(post_lift_path)
    print(f"[GATE G] saved post-lift frame: {post_lift_path}")

    # Success verification - GT ALLOWED HERE ONLY (this task's one exception).
    verdict_table, all_ok = _compute_verdict_table(scene, results, choice)

    print(f"[GATE G] post-lift verdict table (commanded die: {choice}):")
    print(
        f"{'die':<6}{'z_before(mm)':>14}{'z_now(mm)':>12}{'gain(mm)':>10}{'xy_drift(mm)':>14}  "
        f"{'target':^8}  verdict"
    )
    for row in verdict_table:
        print(
            f"{row['die']:<6}{row['z_before_m'] * 1000:>14.1f}{row['z_now_m'] * 1000:>12.1f}"
            f"{row['gain_m'] * 1000:>10.1f}{row['xy_drift_m'] * 1000:>14.1f}  "
            f"{'*TARGET*' if row['is_target'] else '':^8}  "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )

    print(f"[GATE G] {choice}: {'PASS' if all_ok else 'FAIL'} (waypoints={waypoint_status})")

    result = {
        "choice": choice,
        "seed": args_cli.seed,
        "gt_xy_bypass_active": bool(args_cli.gt_xy_bypass),
        "detected_class": target_det["class"] if target_det is not None else None,
        "detection_confidence": target_det["confidence"] if target_det is not None else None,
        "detector_world_pos": det_pos.tolist() if det_pos is not None else None,
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
    verdict_path = os.path.join(out_dir, f"verdict_{choice}.json")
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


def _load_overlay_font() -> "ImageFont.ImageFont":
    """A slightly-larger-than-tiny-default bitmap font for the video overlay
    label, so class+confidence text is actually readable on a 640x480 frame.
    `ImageFont.load_default(size=...)` is a newer Pillow API (confirmed
    supported by vision/.venv's Pillow 12.2.0 this task) - guarded with a
    try/except in case Isaac's own bundled Pillow is older, since this runs
    inside Isaac's python (see run_gate_v's docstring for why that's fine -
    PIL alone needs no torch/ultralytics)."""
    try:
        return ImageFont.load_default(size=16)
    except TypeError:
        return ImageFont.load_default()


def _draw_video_overlay_frame(
    frame: np.ndarray, choice: str, det_class: str, confidence: float, font: "ImageFont.ImageFont"
) -> np.ndarray:
    """Draws a "commanded vs detected" class/confidence text label onto one
    already-captured video frame - Gate V's post-hoc overlay (see
    run_gate_v). Style mirrors vision/scripts/detect_for_sim.py's own
    `_draw_overlay` text label, but uses a filled text background since this
    needs to read clearly on a compressed video frame, not just a static
    PNG.

    2026-07-15: this used to also draw `bbox_xyxy` as a rectangle - dropped
    (user-reported "detection box is not oriented correctly"). The bbox is
    computed against the perception detector's own camera (DiceCamera) but
    this video's frames come from the separate ArmCamera (different pose/
    FOV, switched earlier this session so the grip is actually visible) -
    drawing one camera's pixel-space box onto the other camera's frame is
    meaningless, not just imprecise. A geometrically correct fix would
    reproject the detection's known 3D world_pos into ArmCamera's own image
    plane; not done here - the text label alone (class/confidence, no
    spatial claim) stays accurate regardless of which camera captured the
    frame, which is the safe/quick fix for now."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    label = f"COMMANDED: {choice}  ->  detected: {det_class} ({confidence:.2f})"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = 8
    text_y = 8
    draw.rectangle([text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2], fill=(0, 0, 0))
    draw.text((text_x, text_y), label, fill=(0, 255, 0), font=font)
    return np.array(img)


def run_gate_v() -> None:
    """Gate V: the demo video. Reuses Gate G's flow WHOLESALE (spawn/settle
    -> detector subprocess -> target select -> run_pick_sequence -> GT
    verdict - see run_gate_g, which this deliberately mirrors line-for-line
    except for the video capture/encode/overlay additions below) and adds:
      1. Per-physics-step frame capture (every `_VIDEO_FRAME_STRIDE`'th step)
         via run_pick_sequence's `on_step` hook, spanning a `_PRE_PICK_SECONDS`
         idle-scene segment BEFORE the pick sequence, the ENTIRE pick
         sequence itself (stage 0 through stage 4), and a
         `_POST_LIFT_DWELL_SECONDS` held-lift segment AFTER it.
      2. A post-hoc PIL bbox+label overlay (no torch/ultralytics - safe to
         do directly in Isaac's python, no vision/.venv subprocess needed)
         drawn on the pre-pick segment's own frames, showing which die was
         commanded and what the detector localized for it.
      3. imageio mp4 encoding (pattern: scripts/graspgoal_closeup_video.py),
         done AFTER the run (frames are captured as raw numpy arrays during
         the run, kept out of the IK control-loop's own per-step cost).
    The GT verdict table/pass criteria are byte-for-byte the same check as
    Gate G's own (both call the shared `_compute_verdict_table` helper)."""
    choice = _normalize_choice(args_cli.choice)
    if choice not in DIE_TYPES:
        raise RuntimeError(f"Normalized choice '{choice}' is not one of the physical dice in this scene: {DIE_TYPES}")

    out_dir = os.path.join(COLORED_ROOT_DIR, "gate_v") if args_cli.colored_dice else GATE_V_DIR
    os.makedirs(out_dir, exist_ok=True)

    sim, scene, positions, results = spawn_scene_and_settle(
        out_dir, args_cli.seed, colored_dice=args_cli.colored_dice, light_scale=args_cli.light_scale
    )

    detection_output = run_detector_subprocess(out_dir)
    detections = detection_output["detections"]
    print(f"[GATE V] perception subprocess returned {len(detections)} detections:")
    for det in detections:
        print(f"  class={det['class']:<8} conf={det['confidence']:.3f} world_pos={det['world_pos']}")

    # 2026-07-15 fix: identical bug/fix as Gate G's own (see that gate's
    # comment for the full rationale) - select_target_detection() was
    # called unconditionally before the bypass branch, so a TOTAL
    # detection miss raised before the bypass ever got a chance to run.
    target_det: dict | None = None
    det_x = det_y = det_z = None
    try:
        target_det = select_target_detection(detections, choice)
        det_x, det_y, det_z = target_det["world_pos"]
        print(
            f"[GATE V] target detection for '{choice}': class={target_det['class']} "
            f"conf={target_det['confidence']:.3f} world_pos=({det_x:.4f}, {det_y:.4f}, {det_z:.4f}) "
            f"bbox_xyxy={target_det['bbox_xyxy']}"
        )
    except RuntimeError as e:
        if not args_cli.gt_xy_bypass:
            raise
        print(
            f"[GATE V] detector FAILED to find '{choice}' ({e}) - continuing because "
            f"--gt-xy-bypass is active; target_xy will be sourced from ground truth below, "
            f"detector-vs-GT diagnostic print and video overlay both skipped (nothing to "
            f"compare/draw against)."
        )

    # Diagnostic-only GT comparison - identical to Gate G's, never used for
    # the grasp target itself. Only meaningful when a detection exists.
    gt_pos = np.array([results[choice]["x"], results[choice]["y"], results[choice]["z"]])
    det_pos = None
    xy_err = None
    full_err = None
    if target_det is not None:
        det_pos = np.array([det_x, det_y, det_z])
        full_err = float(np.linalg.norm(gt_pos - det_pos))
        xy_err = float(np.linalg.norm(gt_pos[:2] - det_pos[:2]))
        print(
            f"[GATE V] detector-vs-GT offset for '{choice}' [DIAGNOSTIC ONLY, not used for grasp]: "
            f"xy={xy_err * 1000:.1f}mm full-3d={full_err * 1000:.1f}mm (gt={gt_pos.tolist()}, det={det_pos.tolist()})"
        )

    half_height_m = _die_half_height_m(choice)
    grasp_height_m = _die_grasp_height_m(choice)
    print(
        f"[GATE V] grasp height for '{choice}': half_height(manifest size_mm/2, DIAGNOSTIC ONLY)="
        f"{half_height_m * 1000:.1f}mm grasp_height(MEASURED resting height, actually used)="
        f"{grasp_height_m * 1000:.1f}mm"
    )
    # Ground-truth XY-bypass (2026-07-15) - identical branch/contract to
    # Gate G's own (see that gate's comment for the full rationale); kept
    # as a separate, explicit override here rather than a shared helper to
    # match this file's existing convention of Gate V mirroring Gate G's
    # flow line-for-line rather than factoring it out. target_det is
    # guaranteed non-None in the non-bypass branch (the except-block above
    # re-raises when the bypass is off).
    if args_cli.gt_xy_bypass:
        target_xy = (float(gt_pos[0]), float(gt_pos[1]))
        print(
            f"[GATE V] target_xy SOURCED FROM GROUND TRUTH (bypass active) for '{choice}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f}) - grasp-mechanism isolation only, NOT a "
            f"perception result; detector-vs-GT diagnostic above (if printed) still reflects the "
            f"real detector output."
        )
    else:
        target_xy = (det_x, det_y)
        print(
            f"[GATE V] target_xy sourced from DETECTOR for '{choice}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f})"
        )

    # --- Video capture setup ---
    # 2026-07-15: uses arm_camera (dedicated video/diagnostic camera), NOT
    # scene["camera"] (DiceCamera) - DiceCamera's pose is pinned to the
    # perception detector's training distribution and must stay untouched;
    # the video-capture role only ever needed a camera aimed for human
    # viewing. See tasks/franka/dice_scene_cfg.py's ARM_CAMERA_POS/QUAT_WORLD
    # comment for the re-aim rationale (fixing the fingers'-closing-motion
    # side-profile-occlusion complaint). scene["camera"] itself is untouched
    # and still used above for perception (run_detector_subprocess reads its
    # own saved rgb.png from spawn_scene_and_settle, not this local var).
    camera = scene["arm_camera"]
    sim_dt = sim.get_physics_dt()
    video_frames: list[np.ndarray] = []
    step_counter = [0]

    def _on_step() -> None:
        step_counter[0] += 1
        if step_counter[0] % _VIDEO_FRAME_STRIDE == 0:
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            video_frames.append(rgb)

    # Pre-pick idle segment: the arm is still at its post-reset default (no
    # joint targets touched yet by this gate), dice already settled - this
    # is the segment the overlay below draws onto.
    pre_pick_steps = int(_PRE_PICK_SECONDS / sim_dt)
    for _ in range(pre_pick_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _on_step()
    pre_pick_frame_count = len(video_frames)
    print(f"[GATE V] captured {pre_pick_frame_count} pre-pick idle frames ({_PRE_PICK_SECONDS}s)")

    # A stage timeout is a "fail loudly" signal (see _StageTimeoutError), not
    # a script crash - caught here exactly as Gate G catches it, so a stuck
    # stage still produces a full video + verdict (showing the honest failed
    # attempt) rather than losing all evidence.
    pick_sequence_error = None
    try:
        waypoint_status = run_pick_sequence(sim, scene, target_xy, grasp_height_m, choice, on_step=_on_step, results=results)
    except _StageTimeoutError as e:
        pick_sequence_error = str(e)
        waypoint_status = {"error": pick_sequence_error}
        print(f"[GATE V] *** pick sequence FAILED for choice={choice}: stage timeout - {pick_sequence_error} ***")

    # Post-lift dwell: keep stepping WITHOUT touching any joint targets (the
    # PD controller holds the last commanded arm/gripper targets from
    # whichever stage run_pick_sequence last executed) while continuing to
    # capture frames, so the video shows the (attempted) lift held for a
    # couple of seconds rather than cutting off right at the last waypoint.
    dwell_steps = int(_POST_LIFT_DWELL_SECONDS / sim_dt)
    for _ in range(dwell_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _on_step()
    print(f"[GATE V] captured post-lift dwell ({_POST_LIFT_DWELL_SECONDS}s); total frames so far={len(video_frames)}")

    # Post-lift still frame + RTX convergence steps - same pattern as Gate G
    # (kept separate from the dwell above/not fed into _on_step's stride
    # counter, so this doesn't perturb the video's own frame cadence).
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    rgb_post = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    post_lift_path = os.path.join(out_dir, f"post_lift_{choice}.png")
    Image.fromarray(rgb_post).save(post_lift_path)
    print(f"[GATE V] saved post-lift frame: {post_lift_path}")

    # Success verification - GT ALLOWED HERE ONLY, byte-for-byte the same
    # check as Gate G's (same shared helper/thresholds/logic) so Gate V's
    # pass/fail must match Gate G's per the brief.
    verdict_table, all_ok = _compute_verdict_table(scene, results, choice)

    print(f"[GATE V] post-lift verdict table (commanded die: {choice}):")
    print(
        f"{'die':<6}{'z_before(mm)':>14}{'z_now(mm)':>12}{'gain(mm)':>10}{'xy_drift(mm)':>14}  "
        f"{'target':^8}  verdict"
    )
    for row in verdict_table:
        print(
            f"{row['die']:<6}{row['z_before_m'] * 1000:>14.1f}{row['z_now_m'] * 1000:>12.1f}"
            f"{row['gain_m'] * 1000:>10.1f}{row['xy_drift_m'] * 1000:>14.1f}  "
            f"{'*TARGET*' if row['is_target'] else '':^8}  "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )
    print(f"[GATE V] {choice}: {'PASS' if all_ok else 'FAIL'} (waypoints={waypoint_status})")

    # --- Overlay: draw the commanded die's detection class/confidence label
    # (text only, no spatial bbox - see _draw_video_overlay_frame's own
    # docstring for why the bbox was dropped) on the pre-pick segment's
    # frames (post-hoc, after capture, before encode). Skipped when the
    # bypass caught a total detection miss (target_det is None) - there is
    # no real detection to label, and labeling one would misrepresent a
    # bypassed run as a working perception result. ---
    if target_det is not None:
        overlay_font = _load_overlay_font()
        for i in range(pre_pick_frame_count):
            video_frames[i] = _draw_video_overlay_frame(
                video_frames[i], choice, target_det["class"], target_det["confidence"], overlay_font
            )
        print(
            f"[GATE V] drew detection label on {pre_pick_frame_count} opening frames "
            f"(class={target_det['class']}, conf={target_det['confidence']:.3f})"
        )
    else:
        print(
            f"[GATE V] no detection overlay drawn for '{choice}' - bypass active, no real "
            f"detection exists to label."
        )

    # --- Encode video (imageio, writer pattern from
    # scripts/graspgoal_closeup_video.py) ---
    fps = max(1, round(1.0 / (sim_dt * _VIDEO_FRAME_STRIDE)))
    video_path = os.path.join(out_dir, f"dice_pick_{choice}.mp4")
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264")
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    print(
        f"[GATE V] wrote video: {video_path} "
        f"({len(video_frames)} frames @ {fps}fps, ~{len(video_frames) / fps:.1f}s)"
    )

    result = {
        "choice": choice,
        "seed": args_cli.seed,
        "gt_xy_bypass_active": bool(args_cli.gt_xy_bypass),
        "detected_class": target_det["class"] if target_det is not None else None,
        "detection_confidence": target_det["confidence"] if target_det is not None else None,
        "detection_bbox_xyxy": target_det["bbox_xyxy"] if target_det is not None else None,
        "detector_world_pos": det_pos.tolist() if det_pos is not None else None,
        "gt_world_pos_at_settle": gt_pos.tolist(),
        "detector_vs_gt_xy_error_m": xy_err,
        "detector_vs_gt_full_error_m": full_err,
        "half_height_m": half_height_m,
        "grasp_height_m": grasp_height_m,
        "waypoint_status": waypoint_status,
        "pick_sequence_error": pick_sequence_error,
        "verdict_table": verdict_table,
        "gate_v_pass": bool(all_ok),
        "video_path": video_path,
        "num_video_frames": len(video_frames),
        "video_fps": fps,
        "video_frame_stride": _VIDEO_FRAME_STRIDE,
        "pre_pick_frame_count": pre_pick_frame_count,
        "overlay_frame_count": pre_pick_frame_count,
    }
    verdict_path = os.path.join(out_dir, f"verdict_{choice}.json")
    with open(verdict_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[GATE V] saved verdict: {verdict_path}")
    print("[GATE V] DONE")

    if not all_ok:
        # Report loudly but don't raise - same reasoning as Gate G's own
        # (a raised exception here would skip simulation_app.close() and hit
        # this repo's documented teardown-hang failure mode). d4 was
        # historically this demo's own documented permitted-failure case
        # (pre-rung-1); as of the 2026-07-15 V-notch fixture, a d4 FAIL here
        # is a real result to investigate (Task 2's seeded trials), not an
        # accepted/expected outcome anymore - still non-raising so the sim
        # tears down cleanly either way.
        print(f"[GATE V] *** FAILED for choice={choice} - see verdict table above/verdict_{choice}.json ***")


def main() -> None:
    if args_cli.gate == "a":
        run_gate_a()
    elif args_cli.gate == "g":
        run_gate_g()
    elif args_cli.gate == "v":
        run_gate_v()
    else:
        sys.exit(f"--gate {args_cli.gate} not implemented in this script (only 'a', 'g', and 'v' are).")


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

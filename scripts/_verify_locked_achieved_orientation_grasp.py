"""UPDATE 2026-07-24 (ar4-locked-achieved-orientation-grasp task): the SYNTHESIS
test this investigation's own prior session explicitly called for. Built
directly on ``scripts/_verify_vertical_position_ik_fixed_gripper.py`` (kept
byte-identical through Phase 2 - HOME -> PREGRASP_APPROACH -> GRASP_APPROACH,
all with the gripper OPEN, using the same grid-search-then-DLS-polish
position-only IK to compute ``grasp_q``/``pregrasp_q``), which itself found
this investigation's single biggest positive result: position-only IK
(``command_type="position"``, no orientation lock) reaching the cube's true
~9mm grasp height and producing real, SIMULTANEOUS two-sided contact on both
jaws during CLOSE for the first time ever - but the grasp did not survive
retreat, because position-only IK has no mechanism to hold whatever wrist
orientation it happened to land on once contact was made, and the arm's
orientation was free to drift the moment it moved again toward PREGRASP.

That session's own diagnosis: the orientation-LOCKED mechanism
(``command_type="pose"``, used throughout ``grasp_demo_v2.py``'s later
65-degree-tilt work) CAN hold a stable orientation once commanded to, but in
every prior use it was locked onto a pre-ASSUMED canonical vertical target
decided before any contact was ever made - and that pre-assumed target
cannot reach the true ~9mm grasp height at all under this arm's confirmed
real ``joint_3`` limit. Neither mechanism alone works. This script is the
combination neither prior mechanism tried: use position-only IK to reach the
true height and make real contact (exactly as
``_verify_vertical_position_ik_fixed_gripper.py`` did), then at the EXACT
moment real bilateral contact is detected (watched every physics step during
CLOSE, not assumed), read the arm's ACTUAL live wrist orientation - whatever
the free null-space actually settled into, not a pre-decided canonical
value - and LOCK THAT MEASURED ORIENTATION as the fixed target for a genuine
closed-loop ``command_type="pose"`` DLS controller (the same per-physics-step
continuous-resolve design ``grasp_demo_v2.py``'s own ``polish_from_seed``
already validated as stable, ported here via ``_pose_locked_step``) for the
remainder of CLOSE, the retreat to PREGRASP height, and the hold - instead of
letting orientation drift freely (the old position-only failure mode) or
reverting to a different pre-assumed canonical target (the old
orientation-locked mechanism's own, different, failure mode).

Phase 0-2 (HOME -> PREGRASP_APPROACH -> GRASP_APPROACH, gripper OPEN): kept
mechanically IDENTICAL to
``_verify_vertical_position_ik_fixed_gripper.py`` - one-shot fixed
joint-position commands, no orientation lock, since orientation lock only
becomes meaningful once real contact exists to lock onto.

Phase 3 (CLOSE at ``grasp_q``): still a fixed joint-position command (same
mechanism as before - the ARM doesn't move during this phase, only the
gripper closes), but now watched EVERY physics step (not just the previous
script's every-20-steps sampling cadence) for the first step where BOTH
``gripper_jaw1_contact``/``gripper_jaw2_contact`` register a real force above
``CONTACT_FORCE_THRESHOLD`` - the live wrist orientation (root-frame quat,
link_6) at that exact step is captured as ``locked_quat_b``.

Phase 4 (RETREAT to PREGRASP height) / Phase 5 (HOLD at PREGRASP height):
replace the old scripts' single fixed-joint-config command (which enforced
NOTHING about orientation - PD stiffness alone decided where the wrist ended
up) with ``_pose_locked_step``, a continuous per-physics-step closed-loop DLS
pose controller (position target = ``pregrasp_pos_b``, the same Cartesian
hover point as before; orientation target = ``locked_quat_b``, the
just-measured contact orientation, not a re-derived canonical one) - mirrors
``grasp_demo_v2.py``'s own ``polish_from_seed`` per-step design exactly
(bounded position/rotation step per step via ``POLISH_STEP_MAX``/
``POLISH_ROT_STEP_MAX``, world-to-root Jacobian rotation, joint-limit clamp),
the only proven-stable closed-loop pose-tracking mechanism already validated
in this codebase.

Phase 6 (RELEASE at PREGRASP height, gripper OPEN): same pose-locked
mechanism, target position/orientation unchanged, only the gripper command
flips to OPEN - keeps the arm from swinging away before releasing.

Phase 7 (return HOME): reverts to a plain fixed joint-position command (same
as ``_verify_vertical_position_ik_fixed_gripper.py``'s own final phase) -
once the gripper has released there is nothing left to hold an orientation
lock for.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p \\
        scripts/_verify_locked_achieved_orientation_grasp.py --video-suffix run1
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Locked-achieved-orientation grasp synthesis test.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--cube-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the cube's world (x, y) spawn position (z stays the scene default). Used to validate the "
    "result at more than one position, matching grasp_demo_v2.py's own --cube-xy convention.",
)
parser.add_argument(
    "--video-suffix", type=str, default=None,
    help="Suffix appended to the output video filename, so different --cube-xy runs don't overwrite each other.",
)
parser.add_argument(
    "--retreat-step-max", type=float, default=None,
    help="Per-physics-step Cartesian position bound (meters) for the Phase 4-6 closed-loop pose-locked controller "
    "ONLY (Phase 0-2's grasp_q/pregrasp_q solve is unaffected, still POLISH_STEP_MAX). Defaults to POLISH_STEP_MAX "
    "(0.03m) - the same bound grasp_demo_v2.py's polish_from_seed uses for converging to a STATIC target from a "
    "stationary start. Added (2026-07-24, same task, second run) after a first run at the default bound found "
    "contact dropping to exactly 0.0N on both jaws at the very first physics step of Phase 4 RETREAT, despite the "
    "locked orientation being held to <0.01rad throughout - a 0.03m single-physics-step bound lets a stiff "
    "(stiffness=4000) PD controller snap almost the whole way to a 50mm-away target in one step, a near-instant "
    "yank rather than a smooth carry motion. A much smaller value here tests directly whether that yank, not a "
    "fundamentally non-antipodal grip, is what breaks contact.",
)
parser.add_argument(
    "--retreat-rot-step-max", type=float, default=None,
    help="Per-physics-step rotation bound (radians) for the same Phase 4-6 controller, analogous to "
    "--retreat-step-max. Defaults to POLISH_ROT_STEP_MAX (0.03rad).",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_video_suffix = f"_{args_cli.video_suffix}" if args_cli.video_suffix else ""
VIDEO_PATH = os.path.join(LOG_DIR, "videos", f"ar4_grasp_locked_achieved_orientation{_video_suffix}.mp4")

# Same default cube position _verify_vertical_position_ik_fixed_gripper.py
# used (the config that achieved real bilateral contact) - see that script's
# own NOTE for why this specific value matters (Ar4PickPlaceMirrorSceneCfg's
# actual cube spawn, not the raw objects_cfg.py default). Overridable via
# --cube-xy for the multi-position validation this task asked for.
CUBE_POS_W = (args_cli.cube_xy[0], args_cli.cube_xy[1], 0.009) if args_cli.cube_xy else (0.0, 0.275, 0.009)
PREGRASP_HOVER = 0.05
GRASP_AT_HEIGHT = 0.009
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset

# link_6 -> gripper-jaw-pinch-point offset along link_6's own local +Z axis.
# See _verify_vertical_position_ik_fixed_gripper.py's own _EE_OFFSET
# docstring for the full derivation (measured directly via
# robot.data.body_pos_w for gripper_jaw1_link/gripper_jaw2_link).
_EE_OFFSET = (0.0, 0.0, 0.036)

# 20mm cube (tasks/ar4/objects_cfg.py CUBE_CFG, bumped from 12mm 2026-07-24)
# - half-extent along each of the cube's own local axes. Used only by the
# PINCH-GEOM diagnostic below to test whether a candidate pinch point is
# actually inside the cube's own volume.
CUBE_HALF_SIZE = 0.010

CANDIDATE_SEEDS = [
    (1.0, -0.5, -0.8),
    (1.2, 0.0, -1.2),
    (0.6, 0.3, 0.5),
    (1.3, -0.8, 0.9),
    (0.9, -0.9, -1.5),
    (1.4, 0.4, -0.3),
    (0.8, -0.85, -1.55),
    (1.3, 0.6, -1.7),
    (1.1, 0.9, -1.3),
    (0.5, -0.3, 0.8),
    (1.5, 0.2, 0.2),
    (0.7, 0.7, -0.9),
]

SEED_SETTLE_STEPS = 60
POLISH_STEP_MAX = 0.03
POLISH_ROUNDS = 60
POLISH_SETTLE_STEPS = 30
LAMBDA_VAL = 0.02
CONVERGENCE_THRESHOLD = 0.003

# Rotation-step bound for the Phase 4-6 closed-loop pose controller - EXACTLY
# grasp_demo_v2.py's own POLISH_ROT_STEP_MAX (itself matched to
# demo_franka_ik_dice_line.py's proven-stable _MAX_ROT_STEP), not re-derived
# here. See that module's own docstring for the full justification (an
# 8.6-degree bound was found to destabilize a tilted-target polish; 0.03rad
# (~1.7 degrees) did not).
POLISH_ROT_STEP_MAX = 0.03

# Real-contact detection threshold (Newtons). The prior session
# (_verify_vertical_position_ik_fixed_gripper.py) measured genuine bilateral
# contact in the 0.11-0.17N range on both jaws - this threshold is well below
# that real signal and well above float/sensor noise (this project's contact
# sensors have never reported a nonzero non-noise reading on a
# non-contacting body in any prior session).
CONTACT_FORCE_THRESHOLD = 0.01

# Phase durations (steps). Phases 0-2 and 7 match
# _verify_vertical_position_ik_fixed_gripper.py's own PHASES list exactly for
# direct comparability; phases 3-6 keep the same nominal durations as that
# script's own Phase 3 (CLOSE)/4 (retreat)/5 (hold)/6 (release), even though
# the MECHANISM driving phases 4-6 is now the closed-loop pose controller
# instead of a one-shot fixed joint command.
HOME_STEPS = 60
PREGRASP_APPROACH_STEPS = 150
GRASP_APPROACH_STEPS = 90
CLOSE_STEPS = 60
RETREAT_STEPS = 90
HOLD_STEPS = 120
RELEASE_STEPS = 60
HOME_RETURN_STEPS = 150


def _world_jacobian_to_root_frame(jacobian_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate a Jacobian returned by ``root_physx_view.get_jacobians()`` (WORLD
    frame) into the articulation's own root/base frame - see
    _verify_vertical_position_ik_fixed_gripper.py's identical helper for the
    full derivation (AR4's base carries a real 180-degree yaw, so this is NOT
    a no-op here)."""
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
    if jacobian_b.shape[1] > 3:
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])
    return jacobian_b


def _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_list, steps) -> None:
    """Teleport + explicit zero-velocity + hold, exactly as in
    _verify_vertical_position_ik_fixed_gripper.py - see that script's own
    docstring for why this is necessary for a trustworthy distance reading."""
    q = torch.tensor([q_list], device=env.device)
    robot.write_joint_position_to_sim(q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(steps):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = q
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)


def _ee_point_pos_and_jacobian(ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor, jacobian_b: torch.Tensor):
    """Actual gripper jaw pinch point's position/Jacobian, given link_6's own
    pose/Jacobian and the constant local offset _EE_OFFSET - identical to
    _verify_vertical_position_ik_fixed_gripper.py's own helper."""
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset

    skew = torch.zeros(ee_pos_b.shape[0], 3, 3, device=ee_pos_b.device)
    skew[:, 0, 1], skew[:, 0, 2] = -world_offset[:, 2], world_offset[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = world_offset[:, 2], -world_offset[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -world_offset[:, 1], world_offset[:, 0]
    jac_ang = jacobian_b[:, 3:6, :]
    point_jac_pos = jacobian_b[:, 0:3, :] - torch.bmm(skew, jac_ang)
    return point_pos_b, point_jac_pos


def _measure_dist(robot, robot_entity_cfg, target_pos_b) -> float:
    """Distance from the ACTUAL gripper jaw pinch point to target_pos_b -
    identical to _verify_vertical_position_ik_fixed_gripper.py's own
    helper."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    return torch.norm(point_pos_b[0] - target_pos_b[0]).item()


def _current_ee_quat_b(robot, robot_entity_cfg) -> torch.Tensor:
    """Live link_6 orientation in ROOT frame - the pinch point shares link_6's
    orientation exactly (the rigid offset only shifts position), so this is
    also the pinch point's own live orientation. Used both to capture
    ``locked_quat_b`` at the contact moment and inside the Phase 4-6
    closed-loop controller."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    _, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ee_quat_b


def _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, target_pos_b, seed_j1, extra_full_seeds=None):
    """Genuinely-settled multi-seed search - identical to
    _verify_vertical_position_ik_fixed_gripper.py's own helper."""
    best_dist = float("inf")
    best_q = None
    for (j2, j3, j5) in CANDIDATE_SEEDS:
        q = [seed_j1, j2, j3, 0.0, j5, 0.0]
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q, SEED_SETTLE_STEPS)
        dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        if dist < best_dist:
            best_dist = dist
            best_q = q
    for q in (extra_full_seeds or []):
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q, SEED_SETTLE_STEPS)
        dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        if dist < best_dist:
            best_dist = dist
            best_q = q
    print(f"  [SEED SEARCH] best settled seed dist: {best_dist:.5f}m, config: {best_q}")
    return best_q, best_dist


def polish_from_seed(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, seed_q, num_arm_joints, joint_pos_limits):
    """Bounded-step position-only DLS polish from an already-settled seed -
    identical to _verify_vertical_position_ik_fixed_gripper.py's own helper
    (used only for Phase 0-2's grasp_q/pregrasp_q solve, unchanged from that
    script)."""
    robot = env.scene["robot"]

    best_polish_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
    best_polish_q = list(seed_q)
    final_residual = best_polish_residual

    for round_num in range(POLISH_ROUNDS):
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        jacobian_b = _world_jacobian_to_root_frame(jacobian_w, robot.data.root_quat_w)

        point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)

        direction = target_pos_b - point_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = point_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=POLISH_STEP_MAX)

        ik_controller.set_command(step_target_b, ee_pos=point_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(point_pos_b, ee_quat_b, point_jac_pos, current_joint_pos)

        lo = joint_pos_limits[:, :, 0]
        hi = joint_pos_limits[:, :, 1]
        joint_pos_des = torch.clamp(joint_pos_des, min=lo, max=hi)

        for _ in range(POLISH_SETTLE_STEPS):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = joint_pos_des[:, j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        final_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        if final_residual < best_polish_residual:
            best_polish_residual = final_residual
            best_polish_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
        if final_residual < CONVERGENCE_THRESHOLD:
            break

    if best_polish_residual < final_residual:
        print(
            f"  [POLISH] last round ({final_residual:.5f}m) was worse than the best round found "
            f"({best_polish_residual:.5f}m) - restoring the best config instead of the last one"
        )
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, best_polish_q, POLISH_SETTLE_STEPS * 2)
        final_residual = best_polish_residual

    print(f"  [POLISH] final residual: {final_residual:.5f}m")
    return robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist(), final_residual


def _pose_locked_step(env, ik_pose_controller, robot, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, num_arm_joints, joint_pos_limits, gripper_cmd, step_max=None, rot_step_max=None):
    """One physics step of CLOSED-LOOP pose-locked DLS tracking - ports
    grasp_demo_v2.py's own polish_from_seed per-physics-step design (continuous
    re-solve every step, bounded position+rotation step, world-to-root
    Jacobian rotation, joint-limit clamp - the only closed-loop pose-tracking
    mechanism already validated stable in this codebase) rather than this
    investigation's older "compute a joint config once, hold it open-loop"
    phase pattern. Needed here specifically because the whole point of this
    experiment is to ACTIVELY correct any orientation drift back toward the
    just-measured contact orientation every single step, not just command a
    fixed joint target once and hope PD stiffness alone holds it - that is
    exactly the mechanism the position-only run's own failure (contact
    dropping to 0.0N the instant the arm moved again) showed does NOT work.

    Returns (pos_err, rot_err) for logging."""
    step_max = POLISH_STEP_MAX if step_max is None else step_max
    rot_step_max = POLISH_ROT_STEP_MAX if rot_step_max is None else rot_step_max
    current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
    jacobian_b = _world_jacobian_to_root_frame(jacobian_w, robot.data.root_quat_w)

    point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)
    jac_ang = jacobian_b[:, 3:6, :]
    full_jac = torch.cat([point_jac_pos, jac_ang], dim=1)

    pos_error, rot_error = compute_pose_error(
        point_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
    )
    pos_norm = torch.norm(pos_error, dim=-1, keepdim=True)
    pos_step = pos_error / (pos_norm + 1e-8) * torch.clamp(pos_norm, max=step_max)
    rot_norm = torch.norm(rot_error, dim=-1, keepdim=True)
    rot_step = rot_error / (rot_norm + 1e-8) * torch.clamp(rot_norm, max=rot_step_max)
    delta_command = torch.cat([pos_step, rot_step], dim=-1)

    ik_pose_controller.set_command(delta_command, ee_pos=point_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_pose_controller.compute(point_pos_b, ee_quat_b, full_jac, current_joint_pos)

    lo = joint_pos_limits[:, :, 0]
    hi = joint_pos_limits[:, :, 1]
    joint_pos_des = torch.clamp(joint_pos_des, min=lo, max=hi)

    action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
    action[:, :num_arm_joints] = joint_pos_des
    action[:, num_arm_joints] = gripper_cmd
    env.step(action)

    return torch.norm(pos_error[0]).item(), torch.norm(rot_error[0]).item()


def _jaw_forces(env) -> tuple[float, float]:
    jaw1_force = torch.linalg.vector_norm(
        env.scene["gripper_jaw1_contact"].data.force_matrix_w.view(1, 3)[0]
    ).item()
    jaw2_force = torch.linalg.vector_norm(
        env.scene["gripper_jaw2_contact"].data.force_matrix_w.view(1, 3)[0]
    ).item()
    return jaw1_force, jaw2_force


def _measure_pinch_geometry(label, robot, robot_entity_cfg, gripper_jaw_body_ids, gripper_base_body_id, cube):
    """Direct, LIVE measurement (robot.data.body_pos_w/body_pose_w on the
    real built asset - not this script's own root-frame IK-target
    bookkeeping) of the true fingertip geometry, dispatched to answer TWO
    SEPARATE questions the coordinator explicitly asked not to conflate
    (2026-07-24, this task):

      Q1 (offset-math correctness): does the _EE_OFFSET-derived "assumed
      pinch point" (link_6 world pos + the live-orientation-rotated local
      offset - exactly what _measure_dist/_ee_point_pos_and_jacobian above
      use as the IK target) match the TRUE geometric midpoint between the
      two real jaw fingertip links, AT THIS SPECIFIC achieved orientation?
      This is a pure math/frame-transform check, independent of whether
      the orientation itself is any good.

      Q2 (orientation sanity, independent of Q1): using the TRUE measured
      bisector (not the possibly-buggy offset point) as the candidate pinch
      point, is the ACHIEVED orientation even a geometrically sensible
      antipodal approach on the cube? Checked two ways: (a) is the true
      bisector actually inside the cube's own volume (cube-local-frame
      coordinates within +/-CUBE_HALF_SIZE on all 3 axes) - a pinch point
      outside the cube's own volume cannot be a sensible grasp point
      regardless of any offset bug; (b) is the jaw-opening axis (the
      direction from jaw1 to jaw2) reasonably aligned with one of the
      cube's own local axes (jaws closing along a face-normal direction,
      i.e. straddling two opposite faces - dot product near 1.0) rather
      than roughly tangent to a face (dot product near 0.0, a
      geometrically nonsensical pinch no matter where exactly the "target
      point" is), and whether the approach axis (link_6's local +Z, the
      _EE_OFFSET direction) comes in roughly PARALLEL to that same cube
      axis (face-on) rather than at a skewed/edge-on angle.

    Also reports the gripper BASE link's own distance to the cube (the
    user's specific visual hypothesis: does the base appear to rest on the
    cube rather than the fingertips straddling it?). gripper_base_link is
    coincident in POSITION with link_6 (both `ee_joint`/`gripper_base_joint`
    are zero-translation fixed joints per tasks/ar4/fk_verification.py -
    only the ROTATION differs, a -90deg roll) - so this is equivalently a
    check of link_6's own proximity to the cube vs. the fingertips'.
    """
    jaw_pos_w = robot.data.body_pos_w[0, gripper_jaw_body_ids].clone()
    jaw1_pos_w, jaw2_pos_w = jaw_pos_w[0], jaw_pos_w[1]
    true_bisector_w = (jaw1_pos_w + jaw2_pos_w) / 2.0
    jaw_sep_mm = torch.norm(jaw1_pos_w - jaw2_pos_w).item() * 1000.0

    base_pos_w = robot.data.body_pos_w[0, gripper_base_body_id[0]].clone()

    ee_pose_w = robot.data.body_pose_w[0, robot_entity_cfg.body_ids[0]]
    ee_pos_w, ee_quat_w = ee_pose_w[0:3], ee_pose_w[3:7]
    rot_w = matrix_from_quat(ee_quat_w.unsqueeze(0))[0]
    offset_local = torch.tensor(_EE_OFFSET, device=ee_pos_w.device)
    assumed_pinch_w = ee_pos_w + rot_w @ offset_local
    approach_axis_w = rot_w @ torch.tensor([0.0, 0.0, 1.0], device=ee_pos_w.device)

    offset_disc_mm = torch.norm(true_bisector_w - assumed_pinch_w).item() * 1000.0

    cube_pos_w = cube.data.root_pos_w[0].clone()
    cube_quat_w = cube.data.root_quat_w[0].clone()
    cube_rot_w = matrix_from_quat(cube_quat_w.unsqueeze(0))[0]  # columns = cube's own local axes, in world frame

    bisector_to_cube_mm = torch.norm(true_bisector_w - cube_pos_w).item() * 1000.0
    assumed_to_cube_mm = torch.norm(assumed_pinch_w - cube_pos_w).item() * 1000.0
    base_to_cube_mm = torch.norm(base_pos_w - cube_pos_w).item() * 1000.0
    jaw1_to_cube_mm = torch.norm(jaw1_pos_w - cube_pos_w).item() * 1000.0
    jaw2_to_cube_mm = torch.norm(jaw2_pos_w - cube_pos_w).item() * 1000.0

    bisector_local = cube_rot_w.T @ (true_bisector_w - cube_pos_w)
    inside_cube = bool((bisector_local.abs() <= CUBE_HALF_SIZE).all().item())

    jaw_axis_w = jaw2_pos_w - jaw1_pos_w
    jaw_axis_w = jaw_axis_w / (torch.norm(jaw_axis_w) + 1e-8)
    cube_axis_alignment = [abs(torch.dot(jaw_axis_w, cube_rot_w[:, k]).item()) for k in range(3)]
    best_axis = max(range(3), key=lambda k: cube_axis_alignment[k])
    approach_axis_alignment = abs(torch.dot(approach_axis_w, cube_rot_w[:, best_axis]).item())

    print(f"[PINCH-GEOM] {label}: jaw1_w={jaw1_pos_w.tolist()} jaw2_w={jaw2_pos_w.tolist()} sep={jaw_sep_mm:.4f}mm")
    print(f"[PINCH-GEOM] {label}: true_bisector_w={true_bisector_w.tolist()} assumed_pinch_w(_EE_OFFSET)={assumed_pinch_w.tolist()}")
    print(f"[PINCH-GEOM] {label}: Q1 offset-vs-true-bisector DISCREPANCY = {offset_disc_mm:.4f}mm")
    print(f"[PINCH-GEOM] {label}: gripper_base_link_w={base_pos_w.tolist()} cube_w={cube_pos_w.tolist()}")
    print(
        f"[PINCH-GEOM] {label}: dist-to-cube: true_bisector={bisector_to_cube_mm:.4f}mm "
        f"assumed_pinch={assumed_to_cube_mm:.4f}mm gripper_base={base_to_cube_mm:.4f}mm "
        f"jaw1={jaw1_to_cube_mm:.4f}mm jaw2={jaw2_to_cube_mm:.4f}mm"
    )
    print(
        f"[PINCH-GEOM] {label}: Q2 true_bisector INSIDE cube volume (local-frame half-size={CUBE_HALF_SIZE * 1000:.1f}mm)? "
        f"{inside_cube} (local coords mm={[round(x * 1000, 3) for x in bisector_local.tolist()]})"
    )
    print(
        f"[PINCH-GEOM] {label}: Q2 jaw-opening-axis vs cube local axes |dot|={[round(x, 4) for x in cube_axis_alignment]} "
        f"best_axis={best_axis} approach-axis(link_6 +Z) vs SAME axis |dot|={approach_axis_alignment:.4f} "
        "(1.0=face-on/antipodal-candidate, 0.0=jaws closing tangent to a face - geometrically wrong regardless of "
        "any offset bug)"
    )
    return {
        "offset_disc_mm": offset_disc_mm,
        "base_to_cube_mm": base_to_cube_mm,
        "bisector_to_cube_mm": bisector_to_cube_mm,
        "inside_cube": inside_cube,
        "jaw_axis_alignment": cube_axis_alignment[best_axis],
        "approach_axis_alignment": approach_axis_alignment,
    }


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Same test-local actuator-gain override as
    # _verify_vertical_position_ik_fixed_gripper.py - the arm's default gains
    # (stiffness=40, damping=4) were confirmed too weak to hold/track a
    # commanded pose under gravity for this class of classical-IK demo.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    # Resolved once here for the PINCH-GEOM diagnostic (this task,
    # 2026-07-24) - same robot.find_bodies pattern grasp_demo_v2.py's own
    # _measure_jaw_bisector_vs_ee_offset uses, real built-asset body ids,
    # not FK-model bookkeeping.
    gripper_jaw_body_ids, gripper_jaw_body_names_found = robot.find_bodies(["gripper_jaw1_link", "gripper_jaw2_link"])
    gripper_base_body_id, gripper_base_body_name_found = robot.find_bodies(["gripper_base_link"])
    print(
        f"[INFO] PINCH-GEOM diagnostic body ids resolved: jaws={gripper_jaw_body_names_found}->{gripper_jaw_body_ids} "
        f"base={gripper_base_body_name_found}->{gripper_base_body_id}"
    )

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": LAMBDA_VAL}
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    # Second controller, command_type="pose" (relative mode) - used ONLY for
    # Phase 4-6's closed-loop orientation-locked tracking, mirroring
    # grasp_demo_v2.py's own controller config exactly. Kept as a SEPARATE
    # instance from ik_controller (position-only, absolute mode, used for the
    # Phase 0-2 waypoint solves) rather than reconfiguring one controller
    # in place, since the two use different command_type/use_relative_mode
    # settings.
    ik_pose_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": LAMBDA_VAL}
    )
    ik_pose_controller = DifferentialIKController(ik_pose_cfg, num_envs=env.num_envs, device=env.device)

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["perception_camera"]

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Cube world pos: {CUBE_POS_W}, robot frame: {cube_pos_b[0].tolist()}, calibrated joint_1: {seed_j1:.4f}")

        cube = env.scene["cube"]
        cube_init_pos = cube.data.root_pos_w[0].clone()
        cube_init_quat = cube.data.root_quat_w[0].clone()
        print(f"[INFO] Cube initial pose (world): pos={cube_init_pos.tolist()}")

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

        # Known-good absolute configs from
        # _verify_vertical_position_ik_fixed_gripper.py's own prior verified
        # search at this exact default target (cube (0.0, 0.275, 0.009)) -
        # included directly (alongside this run's own live search) for the
        # same reason that script's own _find_best_seed docstring gives: a
        # live search alone is not perfectly reproducible run-to-run. NOT
        # meaningful as a seed at a different --cube-xy override (the live
        # search alone handles that case).
        KNOWN_GOOD_GRASP_Q = [-0.014429761096835136, 1.240863561630249, 0.3401874601840973, -0.08906537294387817, 1.1987247467041016, 0.0052983760833740234]
        KNOWN_GOOD_PREGRASP_Q = [0.00014747596287634224, 0.9648232460021973, 0.9025915265083313, -0.0006890640361234546, -0.6352600455284119, 0.008003178983926773]
        extra_grasp_seeds = [KNOWN_GOOD_GRASP_Q] if args_cli.cube_xy is None else []
        extra_pregrasp_seeds = [KNOWN_GOOD_PREGRASP_Q] if args_cli.cube_xy is None else []

        print("\n[INFO] Finding best seed for GRASP waypoint (multi-seed, genuinely settled)...")
        seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, grasp_pos_b, seed_j1, extra_full_seeds=extra_grasp_seeds)
        print("[INFO] Polishing GRASP waypoint (fixed-Jacobian DLS)...")
        grasp_q, grasp_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print("\n[INFO] Finding best seed for PREGRASP waypoint (multi-seed, genuinely settled)...")
        seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b, seed_j1, extra_full_seeds=extra_pregrasp_seeds)
        print("[INFO] Polishing PREGRASP waypoint (fixed-Jacobian DLS)...")
        pregrasp_q, pregrasp_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print(f"\n[SUMMARY] grasp_residual={grasp_residual:.5f}m pregrasp_residual={pregrasp_residual:.5f}m")
        print(f"[SUMMARY] grasp_q={grasp_q}")
        print(f"[SUMMARY] pregrasp_q={pregrasp_q}")

        cube_pos_now = cube.data.root_pos_w[0].tolist()
        print(f"[INFO] Cube pose before restore: pos={cube_pos_now}")
        restore_pose = torch.cat([cube_init_pos, cube_init_quat]).unsqueeze(0)
        cube.write_root_pose_to_sim(restore_pose, env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube pose after restore: pos={cube.data.root_pos_w[0].tolist()}")

        def _run_direct_phase(phase_name, duration, target_q, gripper_cmd):
            start_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(f"[{phase_name}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} start_q={['%.4f' % x for x in start_q]}")
            for _ in range(duration):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = target_q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))
            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            max_err = max(abs(a - t) for a, t in zip(achieved_q, target_q))
            print(f"  [{phase_name} END] max joint error: {max_err:.5f} rad")

        print("\n[INFO] Starting phased execution...\n")

        # Phase 0-2: identical mechanism to
        # _verify_vertical_position_ik_fixed_gripper.py - fixed joint-position
        # commands, no orientation lock (none exists yet - contact hasn't
        # happened).
        _run_direct_phase("PHASE 0 HOME", HOME_STEPS, HOME_Q, GRIPPER_OPEN)
        _run_direct_phase("PHASE 1 PREGRASP_APPROACH", PREGRASP_APPROACH_STEPS, pregrasp_q, GRIPPER_OPEN)
        _run_direct_phase("PHASE 2 GRASP_APPROACH", GRASP_APPROACH_STEPS, grasp_q, GRIPPER_OPEN)

        # Phase 3: CLOSE at grasp_q - fixed joint-position command (the arm
        # itself doesn't move here, only the gripper), but watched EVERY
        # physics step for the first real bilateral-contact moment, at which
        # the LIVE (not pre-assumed) wrist orientation is captured and locked
        # for every phase from here on.
        print("\n[INFO] PINCH-GEOM baseline (grasp_q reached, gripper still OPEN, before CLOSE)...")
        _measure_pinch_geometry(
            "PRE-CLOSE (grasp_q, gripper OPEN)", robot, robot_entity_cfg,
            gripper_jaw_body_ids, gripper_base_body_id, cube,
        )

        print(f"\n[PHASE 3 CLOSE] duration={CLOSE_STEPS} gripper=CLOSE watching for first bilateral contact...")
        locked_quat_b = None
        contact_step = None
        contact_forces_at_capture = None
        contact_pinch_geom = None
        for i in range(CLOSE_STEPS):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = grasp_q[j]
            action[:, num_arm_joints] = GRIPPER_CLOSE
            env.step(action)

            rgb = camera.data.output["rgb"][0].cpu().numpy()
            video_writer.append_data(rgb[:, :, :3].astype("uint8"))

            jaw1_force, jaw2_force = _jaw_forces(env)

            if locked_quat_b is None and jaw1_force > CONTACT_FORCE_THRESHOLD and jaw2_force > CONTACT_FORCE_THRESHOLD:
                locked_quat_b = _current_ee_quat_b(robot, robot_entity_cfg).clone()
                contact_step = i
                contact_forces_at_capture = (jaw1_force, jaw2_force)
                print(
                    f"  [CONTACT DETECTED] step {i}: jaw1={jaw1_force:.4f}N jaw2={jaw2_force:.4f}N -> "
                    f"LOCKING orientation quat_b={locked_quat_b[0].tolist()}"
                )
                # Critical measurement for this task (2026-07-24,
                # ar4-pinch-point-geometry-at-contact): AT THIS EXACT
                # physics step, not derived/assumed after the fact.
                contact_pinch_geom = _measure_pinch_geometry(
                    "CONTACT MOMENT (this exact step)", robot, robot_entity_cfg,
                    gripper_jaw_body_ids, gripper_base_body_id, cube,
                )

            if i % 20 == 0 or i == CLOSE_STEPS - 1:
                cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                print(
                    f"  [PHASE 3 step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]} "
                    f"jaw1_cube_force={jaw1_force:.4f}N jaw2_cube_force={jaw2_force:.4f}N"
                )

        achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
        max_err = max(abs(a - t) for a, t in zip(achieved_q, grasp_q))
        print(f"  [PHASE 3 END] max joint error: {max_err:.5f} rad")

        if locked_quat_b is None:
            print(
                "  [WARNING] No bilateral contact detected during CLOSE - falling back to the current "
                "(NOT contact-verified) orientation as the lock target. Treat any subsequent result as "
                "less conclusive than a real contact-verified lock."
            )
            locked_quat_b = _current_ee_quat_b(robot, robot_entity_cfg).clone()
            contact_step = -1
            contact_forces_at_capture = (0.0, 0.0)
            print("[INFO] PINCH-GEOM at fallback (non-contact-verified) orientation, for reference only...")
            contact_pinch_geom = _measure_pinch_geometry(
                "FALLBACK (no real contact detected)", robot, robot_entity_cfg,
                gripper_jaw_body_ids, gripper_base_body_id, cube,
            )

        print(
            f"[SUMMARY] locked_quat_b={locked_quat_b[0].tolist()} captured_at_step={contact_step} "
            f"forces_at_capture={contact_forces_at_capture}"
        )
        if contact_pinch_geom is not None:
            print(
                f"[SUMMARY] PINCH-GEOM at capture: Q1 offset_disc_mm={contact_pinch_geom['offset_disc_mm']:.4f}mm  "
                f"Q2 base_to_cube_mm={contact_pinch_geom['base_to_cube_mm']:.4f}mm  "
                f"bisector_to_cube_mm={contact_pinch_geom['bisector_to_cube_mm']:.4f}mm  "
                f"true_bisector_inside_cube={contact_pinch_geom['inside_cube']}  "
                f"jaw_axis_alignment={contact_pinch_geom['jaw_axis_alignment']:.4f}  "
                f"approach_axis_alignment={contact_pinch_geom['approach_axis_alignment']:.4f}"
            )

        def _run_pose_locked_phase(phase_name, duration, gripper_cmd, log_every=20):
            print(
                f"\n[{phase_name}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} "
                f"target_pos_b={pregrasp_pos_b[0].tolist()} locked_quat_b={locked_quat_b[0].tolist()}"
            )
            for i in range(duration):
                pos_err, rot_err = _pose_locked_step(
                    env, ik_pose_controller, robot, robot_entity_cfg, ik_jacobi_idx,
                    pregrasp_pos_b, locked_quat_b, num_arm_joints, joint_pos_limits, gripper_cmd,
                    step_max=args_cli.retreat_step_max, rot_step_max=args_cli.retreat_rot_step_max,
                )
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))
                if i % log_every == 0 or i == duration - 1:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    jaw1_force, jaw2_force = _jaw_forces(env)
                    print(
                        f"  [{phase_name} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]} "
                        f"jaw1_cube_force={jaw1_force:.4f}N jaw2_cube_force={jaw2_force:.4f}N "
                        f"pos_err={pos_err:.5f}m rot_err={rot_err:.4f}rad"
                    )
            cube_z_end = env.scene["cube"].data.root_pos_w[0, 2].item()
            height_gain = cube_z_end - cube_init_pos[2].item()
            print(f"  [{phase_name} END] cube z={cube_z_end:.4f}m height_gain_vs_resting={height_gain:.4f}m")

        # Phase 4-6: closed-loop orientation-LOCKED tracking (position target
        # = pregrasp_pos_b, orientation target = locked_quat_b measured at
        # real contact above) - the actual synthesis mechanism this script
        # exists to test.
        _run_pose_locked_phase("PHASE 4 RETREAT", RETREAT_STEPS, GRIPPER_CLOSE)
        _run_pose_locked_phase("PHASE 5 HOLD", HOLD_STEPS, GRIPPER_CLOSE)
        _run_pose_locked_phase("PHASE 6 RELEASE", RELEASE_STEPS, GRIPPER_OPEN)

        # Phase 7: return HOME - plain fixed joint-position command, no
        # orientation lock needed once the gripper has released.
        _run_direct_phase("PHASE 7 HOME_RETURN", HOME_RETURN_STEPS, HOME_Q, GRIPPER_OPEN)

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()

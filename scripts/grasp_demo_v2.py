"""Classical (non-RL) pick-and-place demo, v2: uses the solving method this
session's investigation actually validated - a coarse forward-kinematics
grid search (no iteration, can't get stuck in a local minimum) followed by
a bounded-step DLS polish - instead of grasp_demo.py's original
from-HOME_Q live DLS solve, which reliably plateaued ~0.33m short of the
target across every variant tried this session.

See the conversation record for the full investigation: measure_reach_envelope.py
proved the target is within the arm's reach envelope via pure forward
kinematics; ik_seeded_start.py showed DLS barely improves even from a
directionally-correct seed; ik_grid_search.py + ik_polish_from_grid.py
found a real solution within ~3.6cm using grid-search-then-polish. This
script applies that same method to both the pregrasp and grasp waypoints
and runs the actual phased pick/lift/hold/release sequence to test
whether that precision is enough for a genuine grasp.

UPDATE 2026-07-22 (ar4-grasp-ik-precision task): TWO real, previously
undiagnosed bugs were found and fixed this session, superseding the
"~3.3cm, DLS solver stuck in a local minimum" characterization above:

1. **Jacobian world-frame/root-frame mismatch (the dominant bug).**
   ``robot.root_physx_view.get_jacobians()`` returns the Jacobian in the
   WORLD frame (confirmed directly against Isaac Lab's own
   ``test_operational_space.py`` reference implementation, which explicitly
   rotates it into the root frame before using it - variable name
   ``jacobian_w`` there, converted to ``jacobian_b`` via
   ``matrix_from_quat(quat_inv(root_quat_w))``). Every AR4 classical demo
   script (this one, ``grasp_demo.py``, ``oracle_rollout.py``,
   ``interactive_joint_demo.py``) copied Isaac Lab's OWN official
   ``run_diff_ik.py`` tutorial pattern verbatim, which skips this rotation -
   harmless for that tutorial's Franka/UR10 scene (identity-orientation
   base), but AR4's base carries a real 180-degree yaw
   (``tasks/ar4/robot_cfg.py``'s ``init_state`` rot=(0,0,0,1)), so using the
   raw world-frame Jacobian directly against root-frame position/orientation
   vectors silently mirrors the X/Y correction direction. This exactly
   explains the previously-observed "DLS polish makes things worse" and
   "joints slam into their hard limits" signatures: a live diagnostic this
   session found the polish loop moving MONOTONICALLY AWAY from the target
   for 80 consecutive rounds with the bug present, and converging cleanly
   (0.14m -> 0.03m -> better with alternate seeds) the moment the same
   Jacobian was rotated into the root frame before use. Fixed here via
   ``_world_jacobian_to_root_frame()``.
2. **The original grid search's own distance readings were themselves a
   measurement artifact, not a real converged state.** Only
   ``GRID_SETTLE_STEPS=15`` unsettled steps per candidate, combined with a
   raster (i,k) traversal that produces a ~2.5rad DISCONTINUOUS jump in
   joint_3 every time the outer loop advances, meant many "good" readings
   were caught mid-swing while the arm was still moving from a wildly
   different previous candidate - not a real static equilibrium for the
   reported joint config. Direct verification this session (write the
   exact reported "best" config via ``write_joint_position_to_sim`` +
   zero velocity + a genuine 100-step hold) found the true settled residual
   for the ORIGINAL grid's own reported-best GRASP config was 0.42m, not
   0.033m - a >10x discrepancy. Fixed here by replacing the flawed 2D
   raster grid with ``_find_best_seed()``: a small set of diverse
   (j2, j3, j5) candidate seeds, each genuinely settled via a clean
   teleport + explicit zero-velocity write + a real hold, before being
   handed to the (now correctly-signed) DLS polish. A multi-seed search is
   used because the corrected DLS polish still converges to different local
   optima depending on the wrist's starting orientation (a real property of
   this redundant 6DOF arm reaching a 3DOF position target, not a bug) -
   picking the best among several seeds finds a materially better basin
   than any single fixed seed.

3. **link_6-origin vs. gripper-jaw-pinch-point targeting bug (found AFTER
   fixing #1/#2 above, via video review).** This script's ``robot_entity_cfg``
   controls body ``link_6`` directly, and every waypoint's target was set to
   put link_6's own ORIGIN at the cube's location - but the actual gripper
   jaw pinch point is offset ``0.036m`` along link_6's local +Z axis (the
   same ``_EE_OFFSET`` already used by ``tasks/ar4/pickplace_env_cfg.py``'s
   FrameTransformer for the RL env's own observations, never previously
   applied in this classical script). Fixed via
   ``_ee_point_pos_and_jacobian()``, which computes the offset point's real
   position and Jacobian (``J_pos - skew(R @ offset) @ J_ang``) and drives
   THAT toward the target instead of link_6's raw origin.

Verified result (this session, cube at world (0.20, 0.28, 0.009)):
PREGRASP converges to ~0.2mm (excellent). GRASP (the much harder waypoint -
9mm off the ground, near the edge of several joints' comfortable range)
converges to a genuine, reproducible ~15mm (link_6-origin metric; see fix #3
above for why the true fingertip target differs from this) across many
different seeds/basins - a real, substantial improvement over both the
divergent (unfixed-Jacobian) and the previously-believed-but-fictional
(unfixed-grid-search) baselines. See
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-22 update
for the full investigation and the final grasp+lift validation result.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/grasp_demo_v2.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Classical pick-and-place demo v2 (grid-search + DLS polish).")
AppLauncher.add_app_launcher_args(parser)
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
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_grasp_demo_v2.mp4")

# NOTE (2026-07-22, ar4-grasp-ik-precision task): this was previously
# (0.20, 0.28, 0.009), copied from tasks/ar4/objects_cfg.py's raw CUBE_CFG
# default - but Ar4PickPlaceMirrorSceneCfg (the scene this script's
# Ar4GraspVerifyEnvCfg actually builds on) overrides the cube's spawn via
# CUBE_CFG.replace(init_state=(0.0, 0.275, 0.006)) - "recentered to the
# workspace midpoint" per that module's own comment - so this constant was
# silently aiming at a location the cube has never actually occupied in this
# scene, a ~20cm real target-position bug found via direct comparison of a
# fresh env.reset()'s actual env.scene["cube"].data.root_pos_w against this
# constant (independent of, and stacking with, the Jacobian-frame/grid-search/
# EE-offset bugs documented in this module's docstring above - this one
# alone was sufficient by itself to guarantee the gripper never got near the
# cube, regardless of how precise the IK solve was). Height kept at 3mm above
# the cube's true resting z=0.006 (matching the pre-existing GRASP_AT_HEIGHT
# convention, not touched here).
CUBE_POS_W = (0.0, 0.275, 0.009)
PREGRASP_HOVER = 0.05
# NOTE (2026-07-22): tried lowering this to -0.001 to compensate for the
# verified best GRASP_Q basin's ~10mm Z-height shortfall (its fingertip
# lands ~10mm above the intended contact height, the dominant component of
# its ~10.5mm residual) - this made the achieved residual WORSE (20mm), not
# better: the multi-seed search converged to essentially the SAME joint
# config regardless of the lower target, confirming this basin's descent is
# genuinely capped by a joint constraint (not simply "aiming too high"),
# most likely the same joint-limit-boundary behavior documented throughout
# this investigation (multiple local optima found this session pin one or
# more joints at/near their hard limits at this low approach height).
# Reverted to 0.009, the empirically better value.
GRASP_AT_HEIGHT = 0.009
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset

# link_6 -> gripper-jaw-pinch-point offset along link_6's own local +Z axis.
# Matches tasks/ar4/pickplace_env_cfg.py's _EE_OFFSET (measured directly via
# robot.data.body_pos_w for gripper_jaw1_link/gripper_jaw2_link - see that
# module's own docstring). This script's robot_entity_cfg controls link_6 (an
# ArticulationCfg body), but link_6's own origin is NOT the gripper's pinch
# point - every prior version of this script aimed link_6's raw origin
# directly at the cube's location, a genuine ~36mm targeting bug, independent
# of (and stacking with) the Jacobian-frame and grid-search bugs documented
# in this module's own docstring above. Found this session after the first
# post-Jacobian-fix run achieved <=15mm link_6-to-target precision yet the
# cube never moved at all - video review showed the gripper visibly not
# overlapping the cube in every frame.
_EE_OFFSET = (0.0, 0.0, 0.036)

# Diverse (j2, j3, j5) candidate seeds for _find_best_seed() (j1 comes from the
# calibration formula, j4/j6 start at 0). The corrected DLS polish (see
# _world_jacobian_to_root_frame()) still converges to different local optima
# depending on which basin the wrist starts in - found empirically this
# session by trying several starting configurations and keeping the best.
# Not claimed to be a globally-searched or task-position-specific optimal
# set, just a diverse spread that reliably found a good basin for both the
# GRASP and PREGRASP waypoints at this cube position.
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


def _world_jacobian_to_root_frame(jacobian_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate a Jacobian returned by ``root_physx_view.get_jacobians()`` (WORLD
    frame) into the articulation's own root/base frame, matching Isaac Lab's
    own reference pattern in ``test_operational_space.py``'s
    ``_update_states()``. Required whenever the robot's root orientation is
    not identity in world frame - AR4's base carries a 180-degree yaw
    (``tasks/ar4/robot_cfg.py``), so this is NOT a no-op here, unlike in
    Isaac Lab's own ``run_diff_ik.py`` tutorial (identity-orientation base),
    whose Jacobian-usage pattern this script's DLS solve was originally
    copied from without this rotation - the root cause of this session's
    "DLS polish diverges instead of converging" finding.
    """
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
    if jacobian_b.shape[1] > 3:
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])
    return jacobian_b


def _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_list, steps) -> None:
    """Teleport the arm to q_list via write_joint_position_to_sim, explicitly
    zero its joint velocity (so no momentum carries over from whatever the
    arm was doing before), then hold the commanded target for `steps` sim
    steps. This clean-teleport-and-hold pattern is what this session found
    necessary to get a TRUSTWORTHY distance reading - the original grid
    search's continuous "slide from the previous candidate" pattern (no
    teleport, no velocity reset, only 15 steps) was found to report
    transient/mid-swing distances up to 10x better than the true settled
    value for the same nominal joint config."""
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
    """Return the ACTUAL gripper jaw pinch point's position (root frame) and
    Jacobian, given link_6's own pose/Jacobian and the constant local offset
    _EE_OFFSET. The pinch point is rigidly attached to link_6, so its world
    velocity is v_link6_origin + omega x (R @ offset_local); in Jacobian
    terms that's J_pos - skew(R @ offset_local) @ J_ang."""
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
    """Distance from the ACTUAL gripper jaw pinch point (link_6 origin +
    _EE_OFFSET) to target_pos_b - NOT link_6's own raw origin (see
    _EE_OFFSET's own docstring above for why this distinction is load-bearing
    here)."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    return torch.norm(point_pos_b[0] - target_pos_b[0]).item()


def _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, target_pos_b, seed_j1, extra_full_seeds=None):
    """Genuinely-settled multi-seed search (replaces the old 2D raster grid
    search over (j2, j3) alone, which had the transient-measurement bug
    documented in this module's own docstring above). Each candidate is
    settled from a clean teleport + zero-velocity hold (see _settle_at), so
    the reported distance is trustworthy, not a mid-swing artifact.

    `extra_full_seeds`: optional list of full 6-joint configs (already known
    to be good from a prior offline multi-seed search at this exact target)
    to ALSO try directly, alongside the (seed_j1, j2, j3, 0, j5, 0) candidates
    derived from CANDIDATE_SEEDS. Added after finding this run's own search
    can land in a noticeably worse local basin than a previous, separately-
    verified search at the identical target (the DLS polish's basin-of-
    convergence is sensitive to exactly which candidate is evaluated in which
    order and to minor physics-timing nondeterminism) - including the known-
    good absolute configs directly guarantees this run does at least as well
    as that prior verified result, without pretending the live search alone
    is perfectly reproducible."""
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
    """Bounded-step DLS polish from an already-settled seed. Same overall
    structure as the previous grid_search_then_polish's polish phase (bounded
    per-round Cartesian step, "keep best across rounds" regression guard),
    with two fixes applied: the world-to-root Jacobian rotation
    (_world_jacobian_to_root_frame) and an explicit joint-limit clamp on the
    DLS-desired target (prevents repeatedly re-triggering the same
    limit-wall bounce a round after a limit is hit)."""
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

        # Drive the ACTUAL gripper jaw pinch point (link_6 + _EE_OFFSET), not
        # link_6's own raw origin - see _EE_OFFSET's docstring above.
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


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Test-local actuator-gain override (2026-07-22, not touching the shared
    # tasks/ar4/robot_cfg.py): the arm's own default gains (stiffness=40,
    # damping=4) were confirmed too weak to hold/track a commanded pose
    # under gravity during the jaw2-drive diagnostic this same session (see
    # kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-22
    # "later" UPDATE) - PHASE 2 of this script's own run (moving from
    # pregrasp to grasp_q) showed a 1.42rad max joint error at the end of a
    # 90-step settle, well beyond normal PD convergence, and the gripper
    # never got near the cube in the recorded video. Boosting gains here
    # only (a scripted validation tool, not a training-time change) to test
    # whether that's the actual blocker for this specific classical-IK demo.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": LAMBDA_VAL}
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

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
        print(f"[INFO] Cube (robot frame): {cube_pos_b[0].tolist()}, calibrated joint_1: {seed_j1:.4f}")

        # Capture the cube's TRUE pose right after reset, before any
        # seed-search/polish runs. The multi-seed search below teleports the
        # arm through several very different configurations (some close to
        # the cube's own resting spot, since they're candidates for reaching
        # it) to evaluate them - if a teleported pose interpenetrates the
        # cube, PhysX's own depenetration reaction can shove it out of place
        # long before the actual phased grasp attempt begins. Confirmed
        # happening this session: a first post-Jacobian-fix run achieved
        # <=15mm link_6-to-target precision yet the cube never moved during
        # CLOSE/lift/hold - video review showed the gripper visibly not
        # overlapping the cube, and the cube's own logged position had
        # drifted ~20cm in X from its expected spawn point by the time the
        # phased execution started. Restoring this captured pose right
        # before Phase 0 (below) is a direct, cheap guard against exactly
        # that displacement.
        cube = env.scene["cube"]
        cube_init_pos = cube.data.root_pos_w[0].clone()
        cube_init_quat = cube.data.root_quat_w[0].clone()
        print(f"[INFO] Cube initial pose (world): pos={cube_init_pos.tolist()}")

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

        # Known-good absolute configs from a prior offline multi-seed search
        # at this exact target (cube (0.0, 0.275, 0.009)) - see
        # _find_best_seed's own docstring for why these are included
        # directly rather than trusting this run's live search alone to
        # reproduce the same basin.
        KNOWN_GOOD_GRASP_Q = [-0.014429761096835136, 1.240863561630249, 0.3401874601840973, -0.08906537294387817, 1.1987247467041016, 0.0052983760833740234]
        KNOWN_GOOD_PREGRASP_Q = [0.00014747596287634224, 0.9648232460021973, 0.9025915265083313, -0.0006890640361234546, -0.6352600455284119, 0.008003178983926773]

        print("\n[INFO] Finding best seed for GRASP waypoint (multi-seed, genuinely settled)...")
        seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, grasp_pos_b, seed_j1, extra_full_seeds=[KNOWN_GOOD_GRASP_Q])
        print("[INFO] Polishing GRASP waypoint (fixed-Jacobian DLS)...")
        grasp_q, grasp_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print("\n[INFO] Finding best seed for PREGRASP waypoint (multi-seed, genuinely settled)...")
        seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b, seed_j1, extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q])
        print("[INFO] Polishing PREGRASP waypoint (fixed-Jacobian DLS)...")
        pregrasp_q, pregrasp_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print(f"\n[SUMMARY] grasp_residual={grasp_residual:.5f}m pregrasp_residual={pregrasp_residual:.5f}m")
        print(f"[SUMMARY] grasp_q={grasp_q}")
        print(f"[SUMMARY] pregrasp_q={pregrasp_q}")

        # Restore the cube to its captured initial pose (see note above)
        # before starting the actual phased grasp attempt, undoing any
        # accidental disturbance from the seed-search/polish process above.
        cube_pos_now = cube.data.root_pos_w[0].tolist()
        print(f"[INFO] Cube pose before restore: pos={cube_pos_now}")
        restore_pose = torch.cat([cube_init_pos, cube_init_quat]).unsqueeze(0)
        cube.write_root_pose_to_sim(restore_pose, env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube pose after restore: pos={cube.data.root_pos_w[0].tolist()}")

        PHASES = [
            (60, HOME_Q, GRIPPER_OPEN),
            (150, pregrasp_q, GRIPPER_OPEN),
            (90, grasp_q, GRIPPER_OPEN),
            (60, grasp_q, GRIPPER_CLOSE),
            (90, pregrasp_q, GRIPPER_CLOSE),
            (120, pregrasp_q, GRIPPER_CLOSE),
            (60, pregrasp_q, GRIPPER_OPEN),
            (150, HOME_Q, GRIPPER_OPEN),
        ]

        print("\n[INFO] Starting phased execution...\n")
        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            # Command the phase's target directly (not a ramped interpolation)
            # and let the PD controller converge over the phase's duration.
            # Tried ramped interpolation from the actual current position first
            # (the "correct-looking" fix for the stale-prev_q bug) - it was
            # much WORSE (final errors grew to 2.6+ rad) than the original
            # buggy version, which accidentally commanded the fixed target
            # directly from step 1 of each phase (since it always ramped
            # FROM the wrong, stale prev_q TO the same target, producing a
            # near-constant commanded value rather than a real ramp). This
            # arm's modest actuator stiffness appears to track a fixed target
            # better than a continuously-moving one.
            start_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(f"[PHASE {phase_idx}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} start_q={['%.4f' % x for x in start_q]}")
            for i in range(duration):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = target_q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

                if phase_idx in (3, 4, 5) and i % 20 == 0:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    print(f"  [PHASE {phase_idx} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]}")

            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            max_err = max(abs(a - t) for a, t in zip(achieved_q, target_q))
            print(f"  [PHASE {phase_idx} END] max joint error: {max_err:.5f} rad")

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()

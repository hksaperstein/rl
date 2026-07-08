"""Experiment 24 Gate 1: pure, non-learned scripted oracle that follows
this repo's existing 5-waypoint geometric path (compute_path_waypoints,
tasks/ar4/mdp.py) via live differential IK, with a hand-coded
gripper-close schedule tied to waypoint index. No RL, no policy
network - establishes (or rules out) a demonstration source for
Experiment 24 Gate 2. See
docs/superpowers/specs/2026-07-08-ar4-experiment24-scripted-oracle-demonstration-bootstrap-design.md.

Reuses Ar4PickPlaceMirrorEnvCfg UNMODIFIED (tasks/ar4/pickplace_mirror_env_cfg.py) -
its existing JointPositionActionCfg (scale=0.5, offset=default_joint_pos)
action space means recorded (observation, raw_action) trajectories are
already in the format Gate 2's RL policy will use, with no format
conversion needed. compute_path_waypoints is called directly after each
env.reset() (not registered via EventCfg, since this script does not
modify the env cfg) to populate env._path_waypoints_w/env._path_waypoint_idx.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --episodes 50
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 24 Gate 1: scripted-oracle waypoint-following rollout.")
parser.add_argument("--episodes", type=int, default=50, help="Total number of episodes to run across all envs.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.pickplace_env_cfg import _EE_OFFSET  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

ADVANCE_TOLERANCE = 0.03
LIFT_MINIMAL_HEIGHT = 0.03
PREGRASP_HOVER = 0.05
LIFT_MARGIN = 0.02
CARRY_HEIGHT = 0.10
ARM_SCALE = 0.5  # matches Ar4PickPlaceMirrorEnvCfg's ActionsCfg.joint_positions scale
IK_PURSUIT_MAX_STEP = 0.05
"""Max per-step Cartesian pursuit distance (meters) fed to the live
differential-IK solve, before inverting its joint-target solution into a
raw action - identical value/convention to
tasks/ar4/residual_ik_action.py's own _BASE_MAX_STEP. Feeding the IK
solver the FULL remaining distance to a waypoint (which can be tens of cm
at episode start) breaks the differential-IK linearization's local-validity
assumption: a single DLS Newton step toward a far-away target produces
wildly unrealistic joint deltas (observed empirically: raw actions up to
~4.9 rad), which saturate the robot's joint limits and leave the arm stuck
(or diverging) rather than converging - confirmed by a smoke-test rerun
where waypoint_idx never advanced past 0 in 250 steps for any of 4 envs.
This mirrors exactly why every other live-IK controller in this repo
(residual_ik_action.py, classical_pickplace_demo.py) already bounds its
per-step Cartesian pursuit distance before calling IK. Note: raw actions
are NOT additionally clamped to [-1, 1] here - Ar4PickPlacePPORunnerCfg's
policy (tasks/ar4/agents/rsl_rl_ppo_cfg.py) is a plain Gaussian actor with
no tanh-squashing and no action clip configured anywhere in the pipeline
(ActionsCfg.joint_positions.clip is unset/None), so a real PPO rollout's
raw actions are NOT bounded to [-1, 1] either - clamping the oracle's
raw actions there was an earlier, incorrect assumption that pinned joints
at the clamp boundary and reproduced the exact same stuck-at-0 symptom."""

STALL_CHECK_INTERVAL = 25
STALL_THRESHOLD = 0.005
ESCAPE_STEPS = 15
ESCAPE_PERTURBATION_SCALE = 0.03
"""Stall-detection/escape-perturbation mechanism, direct port of
classical_pickplace_demo.py's own _STALL_CHECK_INTERVAL/_STALL_THRESHOLD/
_ESCAPE_STEPS/_ESCAPE_PERTURBATION_SCALE (same values, proven there),
vectorized here for the num_envs > 1 case. Added after a Gate 1 smoke-test
diagnostic (see task-1-report.md) showed pure bounded-pursuit differential
IK genuinely converges for the first ~15-20 steps of an episode, then
reverses and plateaus at a stable pose well short of the target - the
textbook signature of a purely reactive Jacobian-IK controller stalling at
a joint limit or kinematic singularity (no replanning, no escape from
local minima), not a logic bug. classical_pickplace_demo.py already solves
this for its single-env case; this ports the same mechanism per-env."""


def compute_ik_arm_raw_action(
    env: ManagerBasedRLEnv,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ik_jacobi_idx: int,
    arm_default_joint_pos: torch.Tensor,
) -> torch.Tensor:
    """Live differential-IK joint target toward a point bounded
    IK_PURSUIT_MAX_STEP meters from the current end-effector position, in
    the direction of the currently-active waypoint
    (env._path_waypoints_w[:, env._path_waypoint_idx]), with the
    pinch-point offset correction - same IK setup as ik_guided_path_bonus's
    (tasks/ar4/mdp.py), plus the bounded-pursuit-distance step every other
    live-IK controller in this repo already applies (see
    IK_PURSUIT_MAX_STEP's docstring for why this is required, not
    optional) - then inverted through JointPositionActionCfg's known
    processed_action = raw*scale+offset formula (offset=default_joint_pos,
    scale=ARM_SCALE) to produce the raw action env.step() expects. Not
    further clamped - see IK_PURSUIT_MAX_STEP's docstring for why an
    additional [-1, 1] clamp here would be both unnecessary and wrong.
    Envs with env._escape_steps_remaining > 0 (set by update_stall_check)
    get a random per-env perturbation added to the pursuit direction
    before it's converted to a body-frame IK target - direct port of
    classical_pickplace_demo.py's escape-perturbation mechanism, applied
    here instead of after the fact since this controller's "pursuit
    delta" plays the same role as that script's own local variable of the
    same name."""
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]

    current_waypoint_w = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

    offset_vec = torch.tensor(_EE_OFFSET, device=env.device).expand(env.num_envs, 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = current_waypoint_w - offset_w
    ik_target_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)

    direction = ik_target_b - ee_pos_b
    dist = torch.norm(direction, dim=-1, keepdim=True)
    step_mag = torch.clamp(dist, max=IK_PURSUIT_MAX_STEP)
    pursuit_delta = direction / (dist + 1e-8) * step_mag

    escaping = env._escape_steps_remaining > 0
    if escaping.any():
        perturbation = (torch.rand(env.num_envs, 3, device=env.device) * 2.0 - 1.0) * ESCAPE_PERTURBATION_SCALE
        pursuit_delta = pursuit_delta + perturbation * escaping.unsqueeze(-1).to(perturbation.dtype)
        env._escape_steps_remaining = torch.where(
            escaping, env._escape_steps_remaining - 1, env._escape_steps_remaining
        )

    pursuit_target_b = ee_pos_b + pursuit_delta

    ik_controller.set_command(pursuit_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    return (joint_pos_des - arm_default_joint_pos) / ARM_SCALE


def gripper_raw_action_for_waypoint(waypoint_idx: torch.Tensor) -> torch.Tensor:
    """+1.0 (open) at waypoint 0 (pregrasp) and 4 (place); -1.0 (closed)
    at waypoints 1-3 (grasp/lift/transit). Matches BinaryJointPositionAction's
    own raw-action convention (action < 0 -> close, action >= 0 -> open),
    isaaclab/envs/mdp/actions/binary_joint_actions.py."""
    closed = (waypoint_idx >= 1) & (waypoint_idx <= 3)
    return torch.where(closed, torch.full_like(waypoint_idx, -1.0, dtype=torch.float32), torch.ones_like(waypoint_idx, dtype=torch.float32))


def advance_waypoint_idx(env: ManagerBasedRLEnv) -> None:
    ee_frame = env.scene["ee_frame"]
    current_waypoint_w = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
    reached = dist_to_waypoint < ADVANCE_TOLERANCE
    env._path_waypoint_idx = torch.where(
        reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
    )


def reset_waypoints_for_envs(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Calls compute_path_waypoints directly (not via EventCfg, since
    this script does not modify the env cfg) for the given env_ids,
    immediately after those envs have been reset. Also (re)initializes
    per-env stall-detection state (env._stall_check_pos/
    env._escape_steps_remaining) for those env_ids, so a fresh episode
    starts with a clean stall-tracking reference point and no leftover
    escape perturbation from the previous episode - same convention as
    env._path_waypoint_idx itself being reset per-episode."""
    ar4_mdp.compute_path_waypoints(
        env,
        env_ids,
        object_cfg=SceneEntityCfg("cube"),
        lift_minimal_height=LIFT_MINIMAL_HEIGHT,
        pregrasp_hover=PREGRASP_HOVER,
        lift_margin=LIFT_MARGIN,
        carry_height=CARRY_HEIGHT,
    )

    if not hasattr(env, "_stall_check_pos"):
        env._stall_check_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._escape_steps_remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    root_pose_w = robot.data.root_pose_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w)

    env._stall_check_pos[env_ids] = ee_pos_b[env_ids]
    env._escape_steps_remaining[env_ids] = 0


def update_stall_check(env: ManagerBasedRLEnv, step: int) -> None:
    """Every STALL_CHECK_INTERVAL steps (skipping step 0, which only has
    the reset-time initialization in env._stall_check_pos to compare
    against - matching classical_pickplace_demo.py's own guard, there
    expressed as `if stall_check_pos is not None`), flags envs whose
    end-effector moved less than STALL_THRESHOLD in body-frame Cartesian
    distance since the last check as stalled, granting them ESCAPE_STEPS
    of randomized pursuit-direction perturbation (applied in
    compute_ik_arm_raw_action). Direct port of
    classical_pickplace_demo.py's stall-detection/escape mechanism,
    vectorized per-env for num_envs > 1."""
    if step == 0 or step % STALL_CHECK_INTERVAL != 0:
        return

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    root_pose_w = robot.data.root_pose_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w)

    moved = torch.norm(ee_pos_b - env._stall_check_pos, dim=-1)
    stalled = moved < STALL_THRESHOLD
    env._escape_steps_remaining = torch.where(
        stalled, torch.full_like(env._escape_steps_remaining, ESCAPE_STEPS), env._escape_steps_remaining
    )
    env._stall_check_pos = ee_pos_b


def main() -> None:
    env_cfg = Ar4PickPlaceMirrorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    robot = env.scene["robot"]

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    arm_default_joint_pos = robot.data.default_joint_pos[:, arm_joint_ids]

    obs, _ = env.reset()
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    reset_waypoints_for_envs(env, all_env_ids)

    print(f"[SETUP] num_envs={env.num_envs} arm_joint_ids={arm_joint_ids} gripper_joint_ids={gripper_joint_ids}")

    with torch.inference_mode():
        for step in range(250):
            update_stall_check(env, step)

            arm_raw = compute_ik_arm_raw_action(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, arm_default_joint_pos)
            # BinaryJointPositionAction's action_dim is always 1 regardless of how many
            # joints it internally controls (isaaclab/envs/mdp/actions/binary_joint_actions.py,
            # BinaryJointAction.action_dim / self._raw_actions is (num_envs, 1)) - do not
            # expand to len(gripper_joint_ids) here, that overcounts total_action_dim.
            gripper_raw = gripper_raw_action_for_waypoint(env._path_waypoint_idx).unsqueeze(-1)
            actions = torch.cat([arm_raw, gripper_raw], dim=-1)

            obs, _, terminated, truncated, _ = env.step(actions)
            advance_waypoint_idx(env)

            done = terminated | truncated
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                reset_waypoints_for_envs(env, done_ids)

            if step % 50 == 0:
                print(f"[STEP {step:3d}] waypoint_idx={env._path_waypoint_idx.tolist()} escape_steps_remaining={env._escape_steps_remaining.tolist()}")

    print(f"[FINAL] waypoint_idx={env._path_waypoint_idx.tolist()}")
    print("[SMOKE RUN COMPLETE]")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

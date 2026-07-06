# tasks/ar4/mdp.py
"""Local MDP reward terms for the AR4 pick-and-place task that don't exist
in any of Isaac Lab's built-in `mdp` modules.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer


def contact_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus when both gripper fingers register real contact force
    against the sphere specifically - a ground-truth grasp signal
    (ContactSensor, filtered via force_matrix_w), replacing the geometric
    position/closure proxies every prior experiment in this repo's
    grasp-reward history used (see ROADMAP.md's "grasp/lift never emerges"
    entry for why those failed: either reward-hackable via a loose distance
    check, or too sparse to discover via a tight alignment check). Adapted
    from isaaclab_tasks' manipulation/place/agibot task's object_grasped
    pattern (bilateral force-threshold check), using one sensor per jaw and
    the filtered force_matrix_w field rather than the unfiltered
    net_forces_w the reference used - see
    docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md
    for why both corrections were necessary.
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    # force_matrix_w shape: (num_envs, 1 body, 1 filter, 3) for each sensor.
    jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    both_fingers_contact = (jaw1_force > force_threshold) & (jaw2_force > force_threshold)
    return both_fingers_contact.float()


def _raw_lift_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Raw, per-step staged progress signal - NOT itself required to be
    monotonic (the monotonicity comes from the running-max wrapper that
    calls this, staged_potential_progress). Weighted so each higher stage
    dominates once reached: reach (0.1) < grasp (0.2) < lift (0.3) <
    goal-tracking (0.4), max 1.0. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    reach_dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    reach_term = 1.0 - torch.tanh(reach_dist / reach_std)

    # Reuse the already-tested contact_grasp_bonus directly (same bilateral
    # force-threshold check used by every prior experiment this session)
    # rather than re-deriving the same jaw-force logic inline.
    grasp_term = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg)

    lift_term = (object.data.root_pos_w[:, 2] > lift_minimal_height).float()

    # The command is generated in the robot's root frame (UniformPoseCommandCfg
    # with asset_name="robot") - must transform to world frame before comparing
    # against the object's world-frame position, exactly matching
    # isaaclab_tasks' own object_goal_distance (the function sphere_goal_tracking
    # used before this experiment replaced it).
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_dist = torch.norm(object.data.root_pos_w - des_pos_w, dim=-1)
    goal_term = 1.0 - torch.tanh(goal_dist / goal_std)

    return 0.1 * reach_term + 0.2 * grasp_term + 0.3 * lift_term + 0.4 * goal_term


def staged_potential_progress(
    env: ManagerBasedRLEnv,
    gamma: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Potential-based reward shaping (Ng, Harada, Russell, ICML 1999):
    F(s,s') = gamma*Phi(s') - Phi(s), where Phi is a per-episode running
    max of _raw_lift_progress. Because Phi never decreases within an
    episode, this reward is always >= 0 - a momentary drop in the raw
    signal (e.g. contact force dipping during a real lift attempt) cannot
    produce negative reward, structurally removing the incentive to avoid
    risky transitions that a plain additive combination of the same
    sub-signals would create. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_lift_progress(
        env, object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, robot_cfg,
        command_name, reach_std, force_threshold, lift_minimal_height, goal_std,
    )
    prev_potential = env._lift_potential_max.clone()
    new_potential = torch.maximum(env._lift_potential_max, raw)
    env._lift_potential_max = new_potential

    return gamma * new_potential - prev_potential


def reset_lift_potential(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max potential buffer
    for resetting envs, so a new episode starts with no carried-over
    progress. Must be registered in EventCfg alongside reset_scene_to_default.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)
    env._lift_potential_max[env_ids] = 0.0

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
from isaaclab.utils.math import combine_frame_transforms, sample_uniform, subtract_frame_transforms

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


def set_mirrored_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sphere_cfg: SceneEntityCfg,
    goal_y_range: tuple[float, float],
    goal_z_range: tuple[float, float],
) -> None:
    """Event term (mode="reset"): must be registered AFTER the sphere's
    own reset_root_state_uniform event in the same EventCfg (Isaac Lab's
    EventManager runs same-mode terms in registration order - confirmed
    against event_manager.py's apply(), which iterates
    self._mode_term_cfgs[mode] in a plain for loop over registration
    order) so this reads the sphere's freshly-randomized position, not
    the previous episode's. Computes the goal as the mirror image of the
    sphere's spawn across the robot's local x=0 plane (robot_cfg.py's
    AR4_MK5_CFG has no explicit init_state.pos, defaulting to (0,0,0) -
    the robot base sits at each env's own local origin, so negating
    local x is exactly "the other side of the robot"). goal_y is
    independently resampled (not mirrored) for a second degree of
    freedom. Stores the result in env._target_pos_w (world frame,
    per-env, shape (num_envs, 3)) - this stateful buffer replaces
    CommandsCfg/UniformPoseCommandCfg for this scene, since the command
    manager has no way to make one term's target a function of another
    term's own random draw within the same reset. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    sphere: RigidObject = env.scene[sphere_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)

    origins = env.scene.env_origins[env_ids]
    sphere_local_x = sphere.data.root_pos_w[env_ids, 0] - origins[:, 0]

    num = len(env_ids)
    goal_local_x = -sphere_local_x
    goal_local_y = sample_uniform(goal_y_range[0], goal_y_range[1], (num,), env.device)
    goal_local_z = sample_uniform(goal_z_range[0], goal_z_range[1], (num,), env.device)

    env._target_pos_w[env_ids, 0] = origins[:, 0] + goal_local_x
    env._target_pos_w[env_ids, 1] = origins[:, 1] + goal_local_y
    env._target_pos_w[env_ids, 2] = origins[:, 2] + goal_local_z


def mirrored_target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The mirrored goal position (env._target_pos_w, set by
    set_mirrored_goal) expressed in the robot's root frame - mirrors
    isaaclab_tasks' object_position_in_robot_root_frame pattern exactly,
    reading the stateful buffer instead of an object's own root_pos_w."""
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    target_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, env._target_pos_w)
    return target_pos_b


def object_reached_mirrored_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Termination: object within threshold of env._target_pos_w - same
    shape as isaaclab_tasks' object_reached_goal, but compares against
    the stateful mirrored-goal buffer instead of the command manager."""
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(object.data.root_pos_w - env._target_pos_w, dim=-1)
    return distance < threshold

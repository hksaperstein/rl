# tasks/ar4/mdp.py
"""Local MDP reward terms for the AR4 pick-and-place task that don't exist
in any of Isaac Lab's built-in `mdp` modules.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, sample_uniform, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
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
    object_cfg: SceneEntityCfg,
    goal_y_range: tuple[float, float],
    goal_z_range: tuple[float, float],
) -> None:
    """Event term (mode="reset"): must be registered AFTER the object's
    own reset_root_state_uniform event in the same EventCfg (Isaac Lab's
    EventManager runs same-mode terms in registration order - confirmed
    against event_manager.py's apply(), which iterates
    self._mode_term_cfgs[mode] in a plain for loop over registration
    order) so this reads the object's freshly-randomized position, not
    the previous episode's. Computes the goal as the mirror image of the
    object's spawn across the robot's local x=0 plane (robot_cfg.py's
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
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)

    origins = env.scene.env_origins[env_ids]
    object_local_x = object.data.root_pos_w[env_ids, 0] - origins[:, 0]

    num = len(env_ids)
    goal_local_x = -object_local_x
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


def _raw_lift_progress_mirrored(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Same staged reach/grasp/lift/goal signal as _raw_lift_progress,
    but the goal sub-term compares against env._target_pos_w directly
    (already world-frame, set by set_mirrored_goal) instead of
    transforming a CommandsCfg-generated command - this scene has no
    CommandsCfg. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    reach_dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    reach_term = 1.0 - torch.tanh(reach_dist / reach_std)

    grasp_term = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg)

    lift_term = (object.data.root_pos_w[:, 2] > lift_minimal_height).float()

    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_dist = torch.norm(object.data.root_pos_w - env._target_pos_w, dim=-1)
    goal_term = 1.0 - torch.tanh(goal_dist / goal_std)

    return 0.1 * reach_term + 0.2 * grasp_term + 0.3 * lift_term + 0.4 * goal_term


def staged_milestone_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Undiscounted running-max milestone bonus: reward = (new
    best-ever raw progress) - (previous best-ever raw progress) - 0 at a
    plateau, positive on any new milestone, never negative. Corrects a
    bug found in staged_potential_progress (tasks/ar4/mdp.py, used by
    the four-object scene in pickplace_env_cfg.py): that function's
    `gamma * new_potential - prev_potential` formula goes NEGATIVE
    whenever the agent holds a plateaued potential (since gamma < 1),
    making "never approach the object" the reward-minimizing policy -
    see
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md's
    "Why now" section for the full derivation. This version has no gamma
    at all - do not add one back.
    """
    if not hasattr(env, "_lift_milestone_max"):
        env._lift_milestone_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_lift_progress_mirrored(
        env, object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
        reach_std, force_threshold, lift_minimal_height, goal_std,
    )
    prev = env._lift_milestone_max.clone()
    env._lift_milestone_max = torch.maximum(env._lift_milestone_max, raw)
    return env._lift_milestone_max - prev


def reset_lift_milestone(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max milestone buffer
    for resetting envs, so a new episode starts with no carried-over
    progress. Must be registered in EventCfg alongside
    reset_scene_to_default. Uses a different buffer name
    (_lift_milestone_max) than reset_lift_potential's
    (_lift_potential_max) so the two scenes' state can never collide if
    both were ever imported in the same process.
    """
    if not hasattr(env, "_lift_milestone_max"):
        env._lift_milestone_max = torch.zeros(env.num_envs, device=env.device)
    env._lift_milestone_max[env_ids] = 0.0


def stillness_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    still_bound: float,
    patience_steps: int,
) -> torch.Tensor:
    """Grasp-gated penalty for the object failing to move beyond
    still_bound within patience_steps of its last significant movement.
    Targets the 'reach, grip, freeze' failure mode directly: 0 whenever
    grasp hasn't been achieved yet (pre-grasp settling isn't penalized),
    -1.0 once the object has been essentially stationary for too long
    while gripped. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_still_ref_pos"):
        env._still_ref_pos = object.data.root_pos_w.clone()
        env._still_steps = torch.zeros(env.num_envs, device=env.device)

    pos = object.data.root_pos_w
    moved = torch.norm(pos - env._still_ref_pos, dim=-1) > still_bound
    env._still_ref_pos = torch.where(moved.unsqueeze(-1), pos, env._still_ref_pos)
    env._still_steps = torch.where(moved, torch.zeros_like(env._still_steps), env._still_steps + 1)

    grasped = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg) > 0.5
    stagnant = env._still_steps > patience_steps
    return -(grasped & stagnant).float()


def reset_stillness_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor, object_cfg: SceneEntityCfg) -> None:
    """Event term (mode="reset"): must be registered after randomize_goal
    (EventCfg in pickplace_mirror_env_cfg.py) so the reference position
    reflects the new episode's spawn, not the prior episode's end state.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_still_ref_pos"):
        env._still_ref_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._still_steps = torch.zeros(env.num_envs, device=env.device)
    env._still_ref_pos[env_ids] = object.data.root_pos_w[env_ids]
    env._still_steps[env_ids] = 0.0


def ground_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ground_height_threshold: float,
) -> torch.Tensor:
    """Penalty for the object remaining on/near the ground, independent
    of grasp state - unlike stillness_penalty (grasp-gated, only fires
    after a freeze *following* a successful grasp), this applies from
    the start of every episode regardless of whether grasp has been
    achieved, giving constant pressure to lift the object off the
    ground as soon as possible rather than only kicking in after a
    stall post-grasp. Direct user request (2026-07-06): "give a
    negative reward when the cube is on the ground."
    """
    object: RigidObject = env.scene[object_cfg.name]
    on_ground = object.data.root_pos_w[:, 2] < ground_height_threshold
    return -on_ground.float()


def compute_path_waypoints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    lift_minimal_height: float,
    pregrasp_hover: float,
    lift_margin: float,
    carry_height: float,
) -> None:
    """Event term (mode="reset"): must be registered AFTER
    reset_object_position (object's spawn) and AFTER randomize_goal
    (env._target_pos_w) in the same EventCfg, since it reads both.
    Computes 5 Cartesian waypoints (pre-grasp, grasp, lift, transit,
    place) purely geometrically - no IK is used to define them. IK
    guidance happens later, live, per step (ik_guided_path_bonus), by
    asking what classical IK would suggest toward whichever waypoint is
    currently active - see
    docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md for
    why an offline joint-space path isn't computed here.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    object_pos = object.data.root_pos_w[env_ids]
    goal_pos = env._target_pos_w[env_ids]

    pregrasp = object_pos.clone()
    pregrasp[:, 2] += pregrasp_hover

    grasp = object_pos.clone()

    lift = object_pos.clone()
    lift[:, 2] = lift_minimal_height + lift_margin

    transit = torch.zeros_like(object_pos)
    transit[:, 0] = (object_pos[:, 0] + goal_pos[:, 0]) / 2.0
    transit[:, 1] = (object_pos[:, 1] + goal_pos[:, 1]) / 2.0
    transit[:, 2] = carry_height

    place = goal_pos.clone()

    env._path_waypoints_w[env_ids, 0] = pregrasp
    env._path_waypoints_w[env_ids, 1] = grasp
    env._path_waypoints_w[env_ids, 2] = lift
    env._path_waypoints_w[env_ids, 3] = transit
    env._path_waypoints_w[env_ids, 4] = place
    env._path_waypoint_idx[env_ids] = 0
    env._ik_milestone_max[env_ids] = 0.0


def ik_guided_path_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    proximity_std: float,
    advance_tolerance: float,
    ik_joint_std: float,
    gripper_tool_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Undiscounted running-max bonus (same corrected pattern as
    staged_milestone_bonus - see that function's docstring for the decay
    bug this avoids) combining two sub-signals:

    1. Cartesian proximity to the current path waypoint
       (env._path_waypoints_w[:, env._path_waypoint_idx]), weighted so
       later waypoints dominate - a direct generalization of the old
       reach/lift/goal staged terms into one continuous 5-stage signal.
    2. How closely the arm's actual joint configuration matches what a
       LIVE classical IK controller (DifferentialIKController) suggests
       as the next joint target toward that same waypoint, computed
       fresh every step from the real physics state (jacobian, joint
       pos, ee pose) - see
       docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md's
       "Important implementation refinement" section for why this is
       live rather than a precomputed offline path.

    The waypoint index itself advances (monotonically, capped at 4)
    whenever the end-effector comes within advance_tolerance of the
    current waypoint.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "_ik_controller"):
        ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
        env._ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        env._ik_robot_entity_cfg = SceneEntityCfg("robot", joint_names=robot_cfg.joint_names, body_names=["link_6"])
        env._ik_robot_entity_cfg.resolve(env.scene)
        env._ik_jacobi_idx = (
            env._ik_robot_entity_cfg.body_ids[0] - 1
            if robot.is_fixed_base
            else env._ik_robot_entity_cfg.body_ids[0]
        )

    current_waypoint = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint, dim=-1)

    reached = dist_to_waypoint < advance_tolerance
    env._path_waypoint_idx = torch.where(
        reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
    )

    proximity_term = (1.0 - torch.tanh(dist_to_waypoint / proximity_std)) * (
        env._path_waypoint_idx.float() + 1.0
    ) / 5.0

    jacobian = robot.root_physx_view.get_jacobians()[:, env._ik_jacobi_idx, :, env._ik_robot_entity_cfg.joint_ids]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, env._ik_robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    joint_pos = robot.data.joint_pos[:, env._ik_robot_entity_cfg.joint_ids]

    # The waypoint is defined for the gripper's pinch point (ee_frame's
    # target_pos_w - link_6 offset by gripper_tool_offset along its own
    # local +Z, see _EE_OFFSET in pickplace_env_cfg.py), but the IK
    # controller/Jacobian operate on the raw link_6 body. Subtract the
    # offset (rotated into world frame by link_6's current world
    # orientation) from the waypoint before commanding IK, so the
    # suggested joint target places the PINCH POINT - not link_6 itself -
    # at the waypoint. Without this, IK's suggested target is
    # systematically off by the offset's magnitude (3.6cm, larger than
    # advance_tolerance), so the IK-match sub-signal could never reach
    # its maximum even at the objectively correct grasp pose.
    offset_vec = torch.tensor(gripper_tool_offset, device=env.device).expand(env.num_envs, 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = current_waypoint - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)
    env._ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = env._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    joint_dist = torch.norm(joint_pos - joint_pos_des, dim=-1)
    ik_match_term = 1.0 - torch.tanh(joint_dist / ik_joint_std)

    raw = proximity_term + ik_match_term
    prev = env._ik_milestone_max.clone()
    env._ik_milestone_max = torch.maximum(env._ik_milestone_max, raw)
    return env._ik_milestone_max - prev


def gripper_schedule_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    open_pos: float,
    closed_pos: float,
) -> torch.Tensor:
    """Reward matching the classical plan's expected gripper state for
    the current path waypoint: open through waypoints 0-1 (pre-grasp,
    grasp-approach), closed from waypoint 2 onward (lift, transit,
    place). Uses the actual gripper joint position (not the commanded
    action) as ground truth, consistent with contact_grasp_bonus reading
    real physical state rather than commands. Returns 1.0/0.0
    (matches/doesn't) - the "+0.1" magnitude described in the design
    spec comes from this term's RewardsCfg weight (0.1), not from this
    function's own return value. See
    docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_path_waypoint_idx"):
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1)
    midpoint = (open_pos + closed_pos) / 2.0
    is_open = gripper_pos > midpoint

    expected_open = env._path_waypoint_idx < 2
    matches = is_open == expected_open
    return matches.float()


def antipodal_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    antipodal_cos_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bilateral force-closure grasp bonus: requires both jaw contact
    force magnitudes to exceed force_threshold AND their force
    directions to be nearly anti-parallel (cosine of the angle between
    them below antipodal_cos_threshold, e.g. -0.85 corresponds to a
    ~30deg friction-cone half-angle) - the classical two-contact
    force-closure necessary condition (Nguyen 1988, "Constructing
    Force-Closure Grasps"; Ponce & Faverjon, "On Computing Two-Finger
    Force-Closure Grasps of Curved 2D Objects," ICRA 1991/IJRR 1993).
    Unlike contact_grasp_bonus (magnitude-only, kept unchanged and still
    used by the original sphere-based pickplace_env_cfg.py task), this
    also checks force *direction*, which force_matrix_w already
    provides but contact_grasp_bonus discards via vector_norm. A real
    bilateral contact-force reading can register from a non-antipodal,
    physically-unstable pinch that classical theory says is not
    actually resistant to gravity's wrench even though it satisfies a
    magnitude-only check. See
    docs/superpowers/specs/research/2026-07-06-classical-manipulation-senior-a.md.
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    # force_matrix_w shape: (num_envs, 1 body, 1 filter, 3) for each sensor.
    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
    both_magnitude_ok = (jaw1_force_mag > force_threshold) & (jaw2_force_mag > force_threshold)

    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
    antipodal_ok = cos_angle < antipodal_cos_threshold

    return (both_magnitude_ok & antipodal_ok).float()


def path_proximity_bonus(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    proximity_std: float,
    advance_tolerance: float,
) -> torch.Tensor:
    """Undiscounted running-max bonus (same corrected pattern as
    staged_milestone_bonus/ik_guided_path_bonus - see those functions'
    docstrings for the decay bug this avoids) for Cartesian proximity to
    the current path waypoint (env._path_waypoints_w[:, env._path_waypoint_idx]),
    weighted so later waypoints dominate.

    Unlike ik_guided_path_bonus, this drops the IK-action-matching
    sub-signal entirely: this task's arm action is driven by a live
    DifferentialInverseKinematicsAction (see pickplace_taskspace_env_cfg.py's
    ActionsCfg), so the arm tracks IK's suggestion by construction -
    scoring "does the joint configuration match what IK suggests" would
    be close to tautological here. See
    docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md.

    The waypoint index itself advances (monotonically, capped at 4)
    whenever the end-effector comes within advance_tolerance of the
    current waypoint - identical mechanism to ik_guided_path_bonus.
    Reuses env._ik_milestone_max (initialized/reset by
    compute_path_waypoints, unchanged) as its running-max buffer rather
    than introducing a new one, since compute_path_waypoints already
    owns that buffer's lazy-init and per-episode reset.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    current_waypoint = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint, dim=-1)

    reached = dist_to_waypoint < advance_tolerance
    env._path_waypoint_idx = torch.where(
        reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
    )

    proximity_term = (1.0 - torch.tanh(dist_to_waypoint / proximity_std)) * (
        env._path_waypoint_idx.float() + 1.0
    ) / 5.0

    prev = env._ik_milestone_max.clone()
    env._ik_milestone_max = torch.maximum(env._ik_milestone_max, proximity_term)
    return env._ik_milestone_max - prev


def reset_arm_to_pregrasp_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    pregrasp_hover: float,
    gripper_tool_offset: tuple[float, float, float],
) -> None:
    """Event term (mode="reset"): one-shot IK solve that teleports the
    arm's joints so the gripper's pinch point starts AT the pregrasp
    waypoint (this episode's randomized cube position + pregrasp_hover in
    z), instead of starting from a fixed home pose every episode. Must be
    registered AFTER the cube's position has been randomized (reads
    object.data.root_pos_w) and BEFORE compute_path_waypoints - the full
    5-waypoint path is still computed unchanged; since the arm now starts
    already at/near waypoint 0, path_proximity_bonus's own
    advance-tolerance check naturally credits it almost immediately, with
    no reward-function change needed.

    Reuses the same live-DifferentialIKController construction and
    gripper-tool-offset correction ik_guided_path_bonus already uses
    (same file, above), but as a single one-shot solve for only env_ids
    at reset time - NOT cached across calls, since env_ids' length varies
    between calls (all envs on the very first reset, a smaller subset on
    later per-env resets during training), and DifferentialIKController
    allocates internal buffers sized to whatever num_envs it's
    constructed with. Only the env-agnostic SceneEntityCfg/jacobian-index
    lookups are cached; the controller itself is constructed fresh each
    call, sized to len(env_ids). See
    docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_reachskip_robot_entity_cfg"):
        env._reachskip_robot_entity_cfg = SceneEntityCfg(
            robot_cfg.name, joint_names=robot_cfg.joint_names, body_names=["link_6"]
        )
        env._reachskip_robot_entity_cfg.resolve(env.scene)
        env._reachskip_jacobi_idx = (
            env._reachskip_robot_entity_cfg.body_ids[0] - 1
            if robot.is_fixed_base
            else env._reachskip_robot_entity_cfg.body_ids[0]
        )

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=len(env_ids), device=env.device)

    object_pos_w = object.data.root_pos_w[env_ids]
    pregrasp_w = object_pos_w.clone()
    pregrasp_w[:, 2] += pregrasp_hover

    root_pose_w = robot.data.root_pose_w[env_ids]
    ee_pose_w = robot.data.body_pose_w[env_ids, env._reachskip_robot_entity_cfg.body_ids[0]]

    # Same gripper-tool-offset correction as ik_guided_path_bonus: the
    # waypoint targets the pinch point, but the IK controller/Jacobian
    # operate on the raw link_6 body - subtract the offset (rotated into
    # world frame by link_6's current orientation) before commanding IK.
    offset_vec = torch.tensor(gripper_tool_offset, device=env.device).expand(len(env_ids), 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = pregrasp_w - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    jacobian_all = robot.root_physx_view.get_jacobians()[env_ids]
    jacobian = jacobian_all[:, env._reachskip_jacobi_idx, :, env._reachskip_robot_entity_cfg.joint_ids]
    joint_pos = robot.data.joint_pos[env_ids][:, env._reachskip_robot_entity_cfg.joint_ids]

    ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    # Teleport the actual physics state AND set the PD drive's target to
    # the same value, so the drive doesn't immediately fight to move away
    # from the teleported pose on the very first control step.
    robot.write_joint_position_to_sim(
        joint_pos_des, joint_ids=env._reachskip_robot_entity_cfg.joint_ids, env_ids=env_ids
    )
    robot.set_joint_position_target(
        joint_pos_des, joint_ids=env._reachskip_robot_entity_cfg.joint_ids, env_ids=env_ids
    )

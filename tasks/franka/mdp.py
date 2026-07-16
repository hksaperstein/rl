# tasks/franka/mdp.py
"""Fresh MDP glue for the Franka Panda stock lift-recipe reproduction
(franka-panda-pivot, see CLAUDE.md's "Platform pivot (2026-07-09)" section).

Reads live simulated state (cube pose, end-effector FrameTransformer pose,
the randomized goal command) and delegates the actual reward arithmetic to
tasks/franka/lift_reward.py's pure-tensor functions - the same
math/observation/termination behavior as Isaac Lab's own official
isaaclab_tasks/manager_based/manipulation/lift/mdp/{rewards,observations,
terminations}.py, reimplemented from scratch here rather than imported,
per the pivot's "everything new, no reuse" instruction. This module
deliberately does NOT import `isaaclab_tasks.manager_based.manipulation.
lift.mdp` (Isaac Lab's own task-specific reference module) even though the
math it reproduces came from reading that module directly - only the fully
generic, robot/task-agnostic `isaaclab.envs.mdp` library (joint_pos_rel,
action_rate_l2, UniformPoseCommandCfg, BinaryJointPositionActionCfg,
reset_scene_to_default, root_height_below_minimum, etc.) is re-exported
below, the same kind of "official Isaac Lab library, not this repo's own
code" exception CLAUDE.md's pivot section explicitly grants for
`FRANKA_PANDA_CFG`.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# Generic, robot/task-agnostic manager-term library (joint_pos_rel, last_action,
# action_rate_l2, joint_vel_l2, time_out, reset_scene_to_default,
# reset_root_state_uniform, root_height_below_minimum, modify_reward_weight,
# UniformPoseCommandCfg, JointPositionActionCfg, BinaryJointPositionActionCfg,
# DifferentialInverseKinematicsActionCfg, generated_commands, ...) - this is
# Isaac Lab's own installed-package library, not this repo's AR4-era code.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .lift_reward import lifting_object_reward, object_goal_distance_reward, reaching_object_reward
from .shape_observations import geometry_descriptor_broadcast, shape_class_onehot

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Default shape class for env cfgs that never set `die_shape_class` (the
# plain DexCube recipe, FrankaLiftEnvCfg itself, has no shape in
# {d8,d10,d12,d20} at all - this observation term is only meaningful for
# the die-specialist subclasses; d20 is used as an arbitrary but
# historically-dominant fallback so the base class still produces a
# well-formed (num_envs, 4) one-hot rather than erroring). Task 1 of
# docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
# distillation.md - see tasks/franka/shape_observations.py for the pure
# math and docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-
# distillation-design.md for the spec.
_DEFAULT_SHAPE_CLASS = "d20"


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position expressed in the robot's own root frame (observation term)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward term: reaching_object. 1 - tanh(dist/std) kernel on cube-to-EE distance."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return reaching_object_reward(cube_pos_w, ee_pos_w, std)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward term: lifting_object. Binary reward once the cube clears minimal_height."""
    object: RigidObject = env.scene[object_cfg.name]
    return lifting_object_reward(object.data.root_pos_w[:, 2], minimal_height)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward term: object_goal_tracking / object_goal_tracking_fine_grained.
    Gated 1 - tanh(dist/std) kernel on cube-to-goal distance, gated on lift height."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return object_goal_distance_reward(
        object.data.root_pos_w, des_pos_w, object.data.root_pos_w[:, 2], minimal_height, std
    )


def object_shape_class_onehot(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Observation term: one-hot (num_envs, 4) over {d8, d10, d12, d20}.

    This is a per-env-cfg static property (which shape THIS training run's
    object is - a config-time choice, e.g. FrankaDieLiftJointD8StandardEnvCfg
    sets `self.die_shape_class = "d8"` in its own __post_init__), broadcast
    identically to every parallel env - NOT read off live simulated object
    state (the live sim only has raw mesh/pose, no semantic shape label).
    See tasks/franka/shape_observations.py's module docstring for the full
    scope rationale (every consumer of this task is single-shape-per-env-cfg;
    per-env-varying shape is explicitly out of scope here).
    """
    shape_class = getattr(env.cfg, "die_shape_class", _DEFAULT_SHAPE_CLASS)
    return shape_class_onehot(shape_class, env.num_envs, device=env.device)


def object_geometry_descriptor(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Observation term: (num_envs, K) continuous geometry-descriptor
    feature (K=1, Wadell sphericity - see tasks/franka/shape_observations.py's
    module docstring for the exact formula/derivation), same per-env-cfg
    static-property/broadcast treatment as object_shape_class_onehot above.
    """
    shape_class = getattr(env.cfg, "die_shape_class", _DEFAULT_SHAPE_CLASS)
    return geometry_descriptor_broadcast(shape_class, env.num_envs, device=env.device)

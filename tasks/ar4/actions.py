# tasks/ar4/actions.py
"""Custom task-space/gripper action terms for AR4 experiments 20-23:
VerticalLockDifferentialIKAction (Experiment 20), ProximityGatedBinaryJointPositionAction
and MirroredGripperAction (Experiments 21-22), and
WarmStartedResidualDifferentialIKAction (Experiment 23, residual RL with a
literature-grounded warm-start - see
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md).

Subclasses Isaac Lab's built-in DifferentialInverseKinematicsAction to
reuse its jacobian resolution, apply_actions, and reset logic unchanged -
only action_dim and process_actions are overridden. Isolated in its own
file (not tasks/ar4/mdp.py, which holds only reward/observation/event
functions, not ActionTerm/ActionTermCfg classes - a different
responsibility, kept separate).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer


class VerticalLockDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Task-space IK action that re-asserts a fixed end-effector
    orientation every step (not merely leaving it unperturbed), while
    the policy controls only a 3D Cartesian position delta.

    cfg.controller must be configured with command_type="pose",
    use_relative_mode=False - an absolute 7D pose command is required so
    the fixed orientation is actively re-targeted every step, not just
    left alone (which a relative/delta command would do, allowing drift
    to accumulate under contact forces without correction).

    Maintains a persistent position target (self._target_pos, seeded from
    the actual end-effector position on each reset, then accumulated by
    the policy's own deltas) rather than the stock action term's
    ee_pos_curr + delta convention. That stock convention is a
    self-referential target - with a zero action it silently means
    "wherever you currently are," providing zero restoring force if the
    actual pose drifts. Harmless for the stock 3-DOF position-only
    action (3 redundant joint DOF absorb any drift), but combined with
    this class's own fully-constrained 6-DOF pose lock (0 redundant DOF)
    it left nothing anchoring the solve - confirmed by direct
    instrumented testing: the real end-effector orientation converged to
    within ~9 degrees of several different fixed targets (including
    tilted ones, ruling out one specific singular target as the cause)
    and then drifted to 75-99 degrees off target within a single
    episode, under zero commanded action. A persistent, non-self-
    referential target removes that failure mode at its source.
    """

    cfg: VerticalLockDifferentialIKActionCfg

    def __init__(self, cfg: VerticalLockDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._fixed_quat = torch.tensor(cfg.fixed_orientation, device=self.device).repeat(self.num_envs, 1)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ee_pos_curr, _ = self._compute_frame_pose()
        if env_ids is None:
            self._target_pos[:] = ee_pos_curr
        else:
            self._target_pos[env_ids] = ee_pos_curr[env_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._target_pos += self._processed_actions
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        command = torch.cat([self._target_pos, self._fixed_quat], dim=1)
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)


@configclass
class VerticalLockDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Adds fixed_orientation to the stock IK action cfg. See
    VerticalLockDifferentialIKAction."""

    class_type: type[ActionTerm] = VerticalLockDifferentialIKAction
    fixed_orientation: tuple[float, float, float, float] = MISSING
    """Quaternion (w, x, y, z) the end-effector orientation is locked to,
    every step, regardless of policy output."""


class ProximityGatedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint-position gripper action that forces the gripper open
    regardless of the policy's own command, unless the cube is within
    cfg.proximity_threshold of the end-effector. Once within range, the
    policy's own open/close command passes through unchanged.

    Directly motivated by Experiment 20's own follow-up instrumented
    diagnostic (see
    docs/superpowers/specs/2026-07-07-ar4-experiment21-proximity-gated-gripper-design.md):
    across 750 rollout steps of the trained checkpoint,
    gripper_jaw1_joint's contact sensor registered zero force at every
    step while gripper_jaw2_joint registered contact intermittently - an
    asymmetric single-jaw-contact failure, not the both-jaws-wrong-angle
    wedge Experiment 17 found. Hard-gating closing to only be possible
    near the object tests whether premature/imprecise closing explains
    that asymmetry.
    """

    cfg: ProximityGatedBinaryJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        object: RigidObject = self._env.scene[self.cfg.object_cfg.name]
        ee_frame: FrameTransformer = self._env.scene[self.cfg.ee_frame_cfg.name]
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        dist = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
        out_of_range = dist > self.cfg.proximity_threshold
        self._processed_actions[out_of_range] = self._open_command


@configclass
class ProximityGatedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Adds object_cfg/ee_frame_cfg/proximity_threshold to the stock
    binary joint-position action cfg. See
    ProximityGatedBinaryJointPositionAction."""

    class_type: type[ActionTerm] = ProximityGatedBinaryJointPositionAction
    object_cfg: SceneEntityCfg = MISSING
    """The object (cube) the gate measures proximity to."""
    ee_frame_cfg: SceneEntityCfg = MISSING
    """The end-effector frame the gate measures proximity from."""
    proximity_threshold: float = MISSING
    """Distance (m) below which the policy's own gripper command passes
    through; at or above this distance the gripper is forced open."""


class MirroredGripperAction(ProximityGatedBinaryJointPositionAction):
    """Gripper action where gripper_jaw2_joint's commanded target
    continuously tracks gripper_jaw1_joint's ACTUAL measured position
    each step, rather than both jaws independently targeting a fixed
    open/closed constant. Subclasses ProximityGatedBinaryJointPositionAction
    (not plain BinaryJointPositionAction) so both mechanisms compose:
    the gate still decides whether closing is allowed at all
    (super().process_actions runs the gate logic first), then jaw2's
    target is overridden to jaw1's live actual position regardless.

    Implements mimic behavior as a software control-loop reference
    rather than a PhysX-level constraint - Experiment 19's
    PhysxMimicJointAPI-based approach was independently confirmed not
    viable (two tested configurations both made jaw-position divergence
    measurably worse than the uncoupled baseline). See
    docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md.

    Assumes joint_names orders gripper_jaw1_joint before
    gripper_jaw2_joint - verified against the resolved joint names at
    init, not assumed silently.
    """

    cfg: MirroredGripperActionCfg

    def __init__(self, cfg: MirroredGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        assert self._joint_names[0] == "gripper_jaw1_joint" and self._joint_names[1] == "gripper_jaw2_joint", (
            f"MirroredGripperAction assumes joint order [gripper_jaw1_joint, gripper_jaw2_joint], "
            f"got {self._joint_names}"
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        jaw1_actual_pos = self._asset.data.joint_pos[:, self._joint_ids[0]]
        self._processed_actions[:, 1] = jaw1_actual_pos


@configclass
class MirroredGripperActionCfg(ProximityGatedBinaryJointPositionActionCfg):
    """No new fields - reuses ProximityGatedBinaryJointPositionActionCfg's
    object_cfg/ee_frame_cfg/proximity_threshold. See
    MirroredGripperAction."""

    class_type: type[ActionTerm] = MirroredGripperAction


class WarmStartedResidualDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Residual RL over a classical waypoint-pursuit base controller
    (same _compute_base_delta mechanism as ResidualDifferentialIKAction
    in tasks/ar4/residual_ik_action.py - Experiment 13's original), with
    a literal-percentage-of-training linear warm-start ramp on the
    residual's authority: residual_authority = min(1.0, step_count /
    cfg.warmup_steps). Approximates Johannink et al. 2019's technique of
    holding the residual at zero for an initial period while training
    only the value function - Experiment 13's own diagnosed regression
    cause was the ABSENCE of any such warm-start (residual_authority
    implicitly 1.0 from step 0), which this class fixes. See
    docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

    Also performs the waypoint auto-advance side effect
    (env._path_waypoint_idx increments when the end-effector comes
    within cfg.advance_tolerance of the active waypoint) directly inside
    _compute_base_delta, rather than reusing ik_guided_path_bonus's
    bundled reward+advance logic (tasks/ar4/mdp.py) - this experiment's
    only new variable is the action space, not the reward function.

    self._step_count increments once per process_actions call (i.e. once
    per env.step, not per PPO iteration - num_steps_per_env=24 env steps
    make up one iteration, per tasks/ar4/agents/rsl_rl_ppo_cfg.py) and is
    NOT reset on episode reset - it tracks wall-clock training progress
    across the whole run, not per-episode progress, matching Johannink
    et al.'s framing of an initial TRAINING period, not an initial
    EPISODE period.
    """

    cfg: WarmStartedResidualDifferentialIKActionCfg

    def __init__(self, cfg: WarmStartedResidualDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._step_count = 0

    def _compute_base_delta(self) -> torch.Tensor:
        """Bounded proportional ("seek") step toward the currently-active
        waypoint, identical convention to ResidualDifferentialIKAction's
        own method (tasks/ar4/residual_ik_action.py) - returns zeros
        before the first compute_path_waypoints reset event has run.
        Additionally advances env._path_waypoint_idx here (moved from
        ik_guided_path_bonus, which this action term does not use)."""
        env = self._env
        if not hasattr(env, "_path_waypoints_w"):
            return torch.zeros(self.num_envs, 3, device=self.device)
        current_waypoint_w = torch.gather(
            env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        root_pose_w = self._asset.data.root_pose_w
        target_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], current_waypoint_w)
        ee_pos_curr, _ = self._compute_frame_pose()
        direction = target_b - ee_pos_curr
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step = torch.clamp(dist, max=self.cfg.base_max_step)

        ee_frame: FrameTransformer = env.scene[self.cfg.ee_frame_cfg.name]
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        dist_to_waypoint_w = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
        reached = dist_to_waypoint_w < self.cfg.advance_tolerance
        env._path_waypoint_idx = torch.where(
            reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
        )

        return direction / (dist + 1e-8) * step

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        base_delta = self._compute_base_delta()
        residual_authority = min(1.0, self._step_count / self.cfg.warmup_steps)
        self._processed_actions[:] = base_delta + residual_authority * self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        self._step_count += 1


@configclass
class WarmStartedResidualDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Adds ee_frame_cfg/base_max_step/advance_tolerance/warmup_steps to
    the stock IK action cfg. See WarmStartedResidualDifferentialIKAction."""

    class_type: type[ActionTerm] = WarmStartedResidualDifferentialIKAction
    ee_frame_cfg: SceneEntityCfg = MISSING
    """The end-effector frame used for the waypoint-advance distance
    check (target_pos_w, the same field ProximityGatedBinaryJointPositionAction
    already reads from this same cfg name elsewhere in this file)."""
    base_max_step: float = 0.05
    """Max per-step Cartesian pursuit distance (meters) the base
    controller contributes toward the active waypoint."""
    advance_tolerance: float = MISSING
    """Distance (m) below which the end-effector is considered to have
    reached the active waypoint, advancing env._path_waypoint_idx."""
    warmup_steps: int = MISSING
    """Number of env.step() calls (process_actions invocations) over
    which residual_authority ramps linearly from 0 to 1.0."""

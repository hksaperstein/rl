# tasks/ar4/actions.py
"""Custom task-space action term for Experiment 20: locks the AR4
gripper's approach orientation to a fixed, always-vertical (top-down)
target every step, exposing only 3D Cartesian position to the policy -
removing orientation discovery from the exploration problem entirely.
See docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md.

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

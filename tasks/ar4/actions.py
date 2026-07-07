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

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VerticalLockDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Task-space IK action that re-asserts a fixed end-effector
    orientation every step (not merely leaving it unperturbed), while
    the policy controls only a 3D Cartesian position delta.

    cfg.controller must be configured with command_type="pose",
    use_relative_mode=False - an absolute 7D pose command is required so
    the fixed orientation is actively re-targeted every step, not just
    left alone (which a relative/delta command would do, allowing drift
    to accumulate under contact forces without correction).
    """

    cfg: VerticalLockDifferentialIKActionCfg

    def __init__(self, cfg: VerticalLockDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._fixed_quat = torch.tensor(cfg.fixed_orientation, device=self.device).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        desired_pos = ee_pos_curr + self._processed_actions
        command = torch.cat([desired_pos, self._fixed_quat], dim=1)
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)


@configclass
class VerticalLockDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Adds fixed_orientation to the stock IK action cfg. See
    VerticalLockDifferentialIKAction."""

    class_type: type[ActionTerm] = VerticalLockDifferentialIKAction
    fixed_orientation: tuple[float, float, float, float] = MISSING
    """Quaternion (w, x, y, z) the end-effector orientation is locked to,
    every step, regardless of policy output."""

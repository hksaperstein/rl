"""Residual-over-classical-controller action term (Experiment 13): adds a
bounded proportional ("seek") step toward the currently-active waypoint
(env._path_waypoints_w[env._path_waypoint_idx], the same 5-waypoint path
compute_path_waypoints already computes and path_proximity_bonus/
ik_guided_path_bonus already track) to the policy's own scaled raw action,
before handing the combined Cartesian delta to the same live differential-
IK controller every other task-space action term already uses. Additive
superposition (base + residual), per Silver et al. 2018 "Residual Policy
Learning" (arXiv:1812.06298) and Johannink et al. 2019 "Residual
Reinforcement Learning for Robot Control" (ICRA). See
docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created (matches every other tasks/ar4/*.py module in this repo).
"""

import torch

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import subtract_frame_transforms

_BASE_MAX_STEP = 0.05
"""Max per-step Cartesian pursuit distance (meters) the base controller
contributes toward the active waypoint - deliberately identical to
ActionsCfg's own scale=0.05 for the policy's raw-action contribution, so
base and residual are comparably-sized (neither dominates by
construction). See design spec's "Design" section."""


class ResidualDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Same as DifferentialInverseKinematicsAction, except process_actions()
    adds a bounded pursuit step toward the active waypoint to the policy's
    own scaled action, instead of using the policy's action alone as the
    full commanded delta. apply_actions() (the actual IK solve + joint
    command) is inherited unchanged - only the Cartesian delta fed into it
    changes."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        base_delta = self._compute_base_delta()
        self._processed_actions[:] = base_delta + self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def _compute_base_delta(self) -> torch.Tensor:
        """Bounded proportional ("seek") step toward the currently-active
        waypoint, in the same body-frame convention process_actions expects
        (command_type="position", use_relative_mode=True - the controller
        adds this delta to the CURRENT ee pose each step). Returns zeros
        before the first reset event (compute_path_waypoints) has run, since
        env._path_waypoints_w doesn't exist yet at that point - matches the
        identical defensive pattern path_proximity_bonus/ik_guided_path_bonus
        already use in tasks/ar4/mdp.py."""
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
        step = torch.clamp(dist, max=_BASE_MAX_STEP)
        return direction / (dist + 1e-8) * step


@configclass
class ResidualDifferentialIKActionCfg(isaaclab_mdp.DifferentialInverseKinematicsActionCfg):
    """Same fields as DifferentialInverseKinematicsActionCfg - only
    class_type differs, pointing to ResidualDifferentialIKAction instead of
    the base DifferentialInverseKinematicsAction."""

    class_type: type[ActionTerm] = ResidualDifferentialIKAction

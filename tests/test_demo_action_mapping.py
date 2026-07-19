"""Sim-independent unit tests for tasks/franka/demo_action_mapping.py's
pure-torch closed-form action-space conversion - no Isaac Lab import
needed (same importable-without-Isaac-Sim split as
tests/test_mdp_shape_observations.py / tests/test_distillation_data_collection.py).
Run via:

/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_demo_action_mapping.py -v -p no:launch_testing

(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory.)

Task 1 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
implementation.md ("write failing unit tests ... before implementing
either function").
"""

from __future__ import annotations

import pytest
import torch

from tasks.franka.demo_action_mapping import (
    gripper_target_to_raw_action,
    joint_pos_to_raw_action,
)


class TestJointPosToRawAction:
    def test_round_trips_known_values_against_documented_formula(self):
        """isaaclab's JointPositionAction (JointAction.process_actions,
        confirmed by direct source read - see demo_action_mapping.py's own
        module docstring): processed_actions = raw_actions * scale + offset,
        offset = default_joint_pos when use_default_offset=True. Solving for
        raw_action: raw_action = (target_joint_pos - default_joint_pos) / scale.
        Pick concrete numbers and verify both directions agree."""
        default_joint_pos = torch.tensor([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
        scale = 0.5
        raw_action = torch.tensor([0.2, -0.1, 0.0, 0.4, -0.3, 0.05, 0.15])
        target_joint_pos = default_joint_pos + scale * raw_action

        recovered_raw_action = joint_pos_to_raw_action(target_joint_pos, default_joint_pos, scale=scale)
        assert torch.allclose(recovered_raw_action, raw_action, atol=1e-6)

    def test_zero_offset_from_default_gives_zero_raw_action(self):
        default_joint_pos = torch.tensor([0.1, 0.2, 0.3])
        target_joint_pos = default_joint_pos.clone()
        raw_action = joint_pos_to_raw_action(target_joint_pos, default_joint_pos, scale=0.5)
        assert torch.allclose(raw_action, torch.zeros(3), atol=1e-6)

    def test_default_scale_is_0_5(self):
        """FrankaDieLiftJointD8BigEnvCfg/...D10BigEnvCfg's own arm_action
        cfg (tasks/franka/dice_lift_joint_env_cfg.py) uses scale=0.5 -
        confirm this is the function's own default so call sites don't have
        to repeat it."""
        default_joint_pos = torch.zeros(7)
        target_joint_pos = torch.full((7,), 0.5)
        raw_action = joint_pos_to_raw_action(target_joint_pos, default_joint_pos)
        assert torch.allclose(raw_action, torch.ones(7), atol=1e-6)

    def test_scale_zero_rejected(self):
        default_joint_pos = torch.zeros(3)
        target_joint_pos = torch.ones(3)
        with pytest.raises(ValueError):
            joint_pos_to_raw_action(target_joint_pos, default_joint_pos, scale=0.0)

    def test_batched_input(self):
        default_joint_pos = torch.zeros(4)
        target_joint_pos = torch.stack([torch.zeros(4), torch.full((4,), 0.5)])
        raw_action = joint_pos_to_raw_action(target_joint_pos, default_joint_pos, scale=0.5)
        assert raw_action.shape == (2, 4)
        assert torch.allclose(raw_action[0], torch.zeros(4), atol=1e-6)
        assert torch.allclose(raw_action[1], torch.ones(4), atol=1e-6)


class TestGripperTargetToRawAction:
    """Per the plan's Global Constraints, the raw-action convention selecting
    open_command_expr vs close_command_expr was confirmed by direct source
    read of isaaclab.envs.mdp.actions.binary_joint_actions.BinaryJointAction.
    process_actions (see demo_action_mapping.py's own module docstring for
    the exact file/line citation): for float raw actions, `binary_mask =
    actions < 0` selects close; the complement (raw_action >= 0, including
    exactly 0.0) selects open. dice_pick_demo.py's own open_target=0.04 /
    close_target=0.0 gripper joint-position constants match
    lift_env_cfg.py's ActionsCfg.gripper_action open_command_expr/
    close_command_expr values exactly, so a demo-logged gripper_target is
    directly classifiable against those two values."""

    def test_open_joint_pos_maps_to_nonnegative_raw_action(self):
        gripper_target = torch.tensor([0.04, 0.04])  # dice_pick_demo.py's own open_target value
        raw_action = gripper_target_to_raw_action(gripper_target)
        assert (raw_action >= 0).all()

    def test_close_joint_pos_maps_to_negative_raw_action(self):
        gripper_target = torch.tensor([0.0, 0.0])  # dice_pick_demo.py's own close_target value
        raw_action = gripper_target_to_raw_action(gripper_target)
        assert (raw_action < 0).all()

    def test_open_and_close_map_to_different_signs(self):
        open_raw = gripper_target_to_raw_action(torch.tensor([0.04, 0.04]))
        close_raw = gripper_target_to_raw_action(torch.tensor([0.0, 0.0]))
        assert (open_raw >= 0).all()
        assert (close_raw < 0).all()
        assert not torch.equal(torch.sign(open_raw), torch.sign(close_raw))

    def test_batched_rows_open_and_close(self):
        """A (num_steps, num_finger_joints) trajectory tensor mixing both
        gripper states across rows - each row must classify independently."""
        gripper_target = torch.tensor(
            [
                [0.04, 0.04],
                [0.0, 0.0],
                [0.04, 0.04],
            ]
        )
        raw_action = gripper_target_to_raw_action(gripper_target)
        assert raw_action.shape[0] == 3
        assert (raw_action[0] >= 0).all()
        assert (raw_action[1] < 0).all()
        assert (raw_action[2] >= 0).all()

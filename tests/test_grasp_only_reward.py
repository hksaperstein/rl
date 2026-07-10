"""Sim-independent unit tests for tasks/ar4/grasp_only_reward.py's pure
progress math (hierarchical-decomposition research prototype) - no Isaac
Lab import needed. Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_grasp_only_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment)."""

import torch

from tasks.ar4.grasp_only_reward import grasp_lift_goal_progress

LIFT_TARGET_HEIGHT = 0.10
CUBE_TO_GOAL_DIST = 0.4251


def test_zero_before_grasp():
    """Before grasp, raw progress must be exactly 0.0 regardless of
    cube height / goal distance (no reach stage, no partial credit)."""
    n = 10
    grasped = torch.zeros(n, dtype=torch.bool)
    lifted = torch.zeros(n, dtype=torch.bool)
    cube_height = torch.linspace(0.0, LIFT_TARGET_HEIGHT, n)
    goal_dist = torch.linspace(CUBE_TO_GOAL_DIST, 0.0, n)

    raw = grasp_lift_goal_progress(grasped, lifted, cube_height, goal_dist, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST)
    assert torch.all(raw.abs() < 1e-6), f"expected all-zero before grasp, got {raw.tolist()}"


def test_grasp_jump_floor_is_one_third():
    """Achieving grasp (grasped flips true) must jump raw progress to
    exactly 1/3, regardless of cube height/goal distance at that instant."""
    grasped = torch.tensor([False, True])
    lifted = torch.tensor([False, False])
    cube_height = torch.tensor([0.0, 0.0])
    goal_dist = torch.tensor([CUBE_TO_GOAL_DIST, CUBE_TO_GOAL_DIST])

    raw = grasp_lift_goal_progress(grasped, lifted, cube_height, goal_dist, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST)
    assert raw[1].item() - raw[0].item() >= (1.0 / 3.0) - 1e-6, f"grasp jump too small: {(raw[1] - raw[0]).item()}"
    assert abs(raw[1].item() - 1.0 / 3.0) < 1e-5, f"grasp stage floor should be exactly 1/3, got {raw[1].item()}"


def test_lift_to_goal_transition_is_non_negative():
    """Achieving lift (lifted flips true) must produce a non-negative
    raw-progress jump to the goal stage."""
    grasped = torch.tensor([True, True])
    lifted = torch.tensor([False, True])
    cube_height = torch.tensor([LIFT_TARGET_HEIGHT / 2, LIFT_TARGET_HEIGHT / 2])
    goal_dist = torch.tensor([CUBE_TO_GOAL_DIST, CUBE_TO_GOAL_DIST])

    raw = grasp_lift_goal_progress(grasped, lifted, cube_height, goal_dist, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST)
    assert raw[1].item() - raw[0].item() >= -1e-6, f"lift transition jumped backward: {(raw[1] - raw[0]).item()}"
    assert abs(raw[1].item() - 2.0 / 3.0) < 1e-5, f"goal stage floor should be exactly 2/3, got {raw[1].item()}"


def test_lift_and_goal_stages_monotonic_and_bounded():
    """Once grasped (not yet lifted), raw progress ramps 1/3->2/3 with
    cube height. Once lifted, raw progress ramps 2/3->1.0 with
    cube-to-goal distance. Neither stage ever exceeds its ceiling."""
    n = 100
    cube_height = torch.linspace(0.0, LIFT_TARGET_HEIGHT, n)
    grasped = torch.ones(n, dtype=torch.bool)
    lifted = torch.zeros(n, dtype=torch.bool)
    zeros = torch.zeros(n)

    lift_raw = grasp_lift_goal_progress(grasped, lifted, cube_height, zeros, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST)
    lift_deltas = lift_raw[1:] - lift_raw[:-1]
    assert torch.all(lift_deltas >= -1e-6), f"lift stage decreased: min delta {lift_deltas.min().item()}"
    assert torch.all(lift_raw <= 2.0 / 3.0 + 1e-6), f"lift stage exceeded 2/3 ceiling: max {lift_raw.max().item()}"

    goal_dist = torch.linspace(CUBE_TO_GOAL_DIST, 0.0, n)
    lifted_all = torch.ones(n, dtype=torch.bool)

    goal_raw = grasp_lift_goal_progress(grasped, lifted_all, zeros, goal_dist, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST)
    goal_deltas = goal_raw[1:] - goal_raw[:-1]
    assert torch.all(goal_deltas >= -1e-6), f"goal stage decreased: min delta {goal_deltas.min().item()}"
    assert abs(goal_raw[-1].item() - 1.0) < 1e-5, f"goal stage should reach ~1.0 at goal_dist=0, got {goal_raw[-1].item()}"

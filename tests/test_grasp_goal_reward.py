"""Sim-independent unit tests for tasks/ar4/grasp_goal_reward.py's pure
progress math (Experiment 26) - no Isaac Lab import needed. Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_grasp_goal_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory)."""

import torch

from tasks.ar4.grasp_goal_reward import grasp_goal_progress

REACH_DIST_NORM = 0.3
LIFT_MINIMAL_HEIGHT = 0.03
LIFT_TARGET_HEIGHT = 0.10
CUBE_TO_GOAL_DIST = 0.4251


def test_reach_stage_monotonic_and_bounded():
    """Before grasp, raw progress must rise monotonically as reach_dist
    shrinks, and never exceed the 0.25 reach ceiling."""
    reach_dist = torch.linspace(REACH_DIST_NORM, 0.0, 100)
    grasped = torch.zeros(100, dtype=torch.bool)
    lifted = torch.zeros(100, dtype=torch.bool)
    zeros = torch.zeros(100)

    raw = grasp_goal_progress(
        reach_dist, grasped, lifted, zeros, zeros,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )

    deltas = raw[1:] - raw[:-1]
    assert torch.all(deltas >= -1e-6), f"reach stage decreased: min delta {deltas.min().item()}"
    assert torch.all(raw <= 0.25 + 1e-6), f"reach stage exceeded 0.25 ceiling: max {raw.max().item()}"
    assert abs(raw[-1].item() - 0.25) < 1e-5, f"reach stage should reach ~0.25 at reach_dist=0, got {raw[-1].item()}"


def test_grasp_jump_is_at_least_a_quarter():
    """Achieving grasp (grasped flips true) must produce a raw-progress
    jump of at least 0.25 relative to the reach ceiling, regardless of
    reach_dist at that instant."""
    reach_dist = torch.tensor([0.0, 0.0])
    grasped = torch.tensor([False, True])
    lifted = torch.tensor([False, False])
    cube_height = torch.tensor([0.0, 0.0])
    goal_dist = torch.tensor([CUBE_TO_GOAL_DIST, CUBE_TO_GOAL_DIST])

    raw = grasp_goal_progress(
        reach_dist, grasped, lifted, cube_height, goal_dist,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )

    assert raw[1].item() - raw[0].item() >= 0.25 - 1e-6, f"grasp jump too small: {(raw[1] - raw[0]).item()}"
    assert abs(raw[1].item() - 0.50) < 1e-5, f"grasp stage floor should be exactly 0.50, got {raw[1].item()}"


def test_lift_and_goal_stages_monotonic_and_bounded():
    """Once grasped (not yet lifted), raw progress ramps 0.50->0.75 with
    cube height. Once lifted, raw progress ramps 0.75->1.00 with
    cube-to-goal distance. Neither stage ever exceeds its ceiling."""
    n = 100
    cube_height = torch.linspace(0.0, LIFT_TARGET_HEIGHT, n)
    grasped = torch.ones(n, dtype=torch.bool)
    lifted = torch.zeros(n, dtype=torch.bool)
    zeros = torch.zeros(n)

    lift_raw = grasp_goal_progress(
        zeros, grasped, lifted, cube_height, zeros,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )
    lift_deltas = lift_raw[1:] - lift_raw[:-1]
    assert torch.all(lift_deltas >= -1e-6), f"lift stage decreased: min delta {lift_deltas.min().item()}"
    assert torch.all(lift_raw <= 0.75 + 1e-6), f"lift stage exceeded 0.75 ceiling: max {lift_raw.max().item()}"

    goal_dist = torch.linspace(CUBE_TO_GOAL_DIST, 0.0, n)
    lifted_all = torch.ones(n, dtype=torch.bool)

    goal_raw = grasp_goal_progress(
        zeros, grasped, lifted_all, zeros, goal_dist,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )
    goal_deltas = goal_raw[1:] - goal_raw[:-1]
    assert torch.all(goal_deltas >= -1e-6), f"goal stage decreased: min delta {goal_deltas.min().item()}"
    assert abs(goal_raw[-1].item() - 1.0) < 1e-5, f"goal stage should reach ~1.0 at goal_dist=0, got {goal_raw[-1].item()}"

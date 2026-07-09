"""Sim-independent unit tests for tasks/ar4/touch_goal_reward.py's pure
progress math (Experiment 25) - no Isaac Lab import needed, run with
plain pytest: `pytest tests/test_touch_goal_reward.py -v`
"""

import torch

from tasks.ar4.touch_goal_reward import touch_goal_progress

TOUCH_STD = 0.05
TOUCH_TO_GOAL_DIST = 0.4231437


def test_monotonic_non_decreasing_along_touch_to_goal_path():
    """Once touched, raw progress must never decrease as goal_dist
    decreases (walking straight from the touch point toward the goal) -
    the exact property final whole-branch review found broken in the
    original dual-tanh-sum formulation."""
    n_samples = 200
    goal_dist = torch.linspace(TOUCH_TO_GOAL_DIST, 0.0, n_samples)
    touch_dist = torch.full((n_samples,), TOUCH_TO_GOAL_DIST) - goal_dist
    touched = torch.ones(n_samples, dtype=torch.bool)

    raw = touch_goal_progress(touch_dist, goal_dist, touched, TOUCH_STD, TOUCH_TO_GOAL_DIST)

    deltas = raw[1:] - raw[:-1]
    assert torch.all(deltas >= -1e-6), f"raw progress decreased somewhere: min delta {deltas.min().item()}"
    assert abs(raw[0].item() - 0.3) < 1e-5, f"raw progress at touch point should be ~0.3, got {raw[0].item()}"
    assert abs(raw[-1].item() - 1.0) < 1e-5, f"raw progress at goal should be ~1.0, got {raw[-1].item()}"


def test_pre_touch_capped_below_post_touch_floor():
    """Before touch, raw progress should never exceed the post-touch
    floor of 0.3 (touch_term saturates at 1.0 as touch_dist -> 0, giving
    0.3 * 1.0 = 0.3 in the limit, never more)."""
    touch_dist = torch.linspace(0.0, 1.0, 100)
    goal_dist = torch.full((100,), TOUCH_TO_GOAL_DIST)
    touched = torch.zeros(100, dtype=torch.bool)

    raw = touch_goal_progress(touch_dist, goal_dist, touched, TOUCH_STD, TOUCH_TO_GOAL_DIST)

    assert torch.all(raw <= 0.3 + 1e-6), f"pre-touch raw progress exceeded 0.3: max {raw.max().item()}"


def test_touch_latches_regardless_of_subsequent_distance():
    """Once touched=True, formula must use the post-touch branch even if
    touch_dist has since grown large (latch semantics documented here at
    the formula level; the actual latching logic lives in
    tasks/ar4/mdp.py's stateful buffer, not tested here)."""
    touch_dist = torch.tensor([0.5])
    goal_dist = torch.tensor([TOUCH_TO_GOAL_DIST])
    touched = torch.tensor([True])

    raw = touch_goal_progress(touch_dist, goal_dist, touched, TOUCH_STD, TOUCH_TO_GOAL_DIST)

    assert abs(raw.item() - 0.3) < 1e-5, f"expected ~the post-touch floor (0.3) at goal_dist=touch_to_goal_dist, got {raw.item()}"

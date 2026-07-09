"""Pure-tensor touch-then-goal progress math for Experiment 25 - no Isaac
Lab dependency, so this is testable with plain pytest+torch (see
tests/test_touch_goal_reward.py), the same sim-independent-math pattern
this project's perception/tests/ already uses. tasks/ar4/mdp.py's
_raw_touch_goal_progress reads live simulated state (end-effector
position, cube position) and delegates the actual formula to
touch_goal_progress() below.
"""

import torch


def touch_goal_progress(
    touch_dist: torch.Tensor,
    goal_dist: torch.Tensor,
    touched: torch.Tensor,
    touch_std: float,
    touch_to_goal_dist: float,
) -> torch.Tensor:
    """Two-stage touch-then-goal progress signal. Pre-touch: dense
    tanh-shaped proximity to the touch point, capped at 0.3. Post-touch:
    a MONOTONIC linear potential from 0.3 (at the touch point) to 1.0 (at
    the goal point), driven by goal_dist/touch_to_goal_dist - not a
    second tanh proximity bump summed with the first (that combination,
    run through a running-max milestone mechanism, was found by final
    whole-branch review 2026-07-09 to leave a reward-free dead zone
    across most of the touch-to-goal traverse, since the two points are
    ~0.42m apart and both tanh terms saturate to ~0 well before reaching
    each other's neighborhood). This linear post-touch formulation is
    monotonically non-decreasing by construction along the straight
    touch->goal line, so a running-max mechanism built on top of it never
    stalls once touch is registered."""
    touch_term = 1.0 - torch.tanh(touch_dist / touch_std)
    goal_progress = torch.clamp(1.0 - goal_dist / touch_to_goal_dist, min=0.0, max=1.0)
    pre_touch_raw = 0.3 * touch_term
    post_touch_raw = 0.3 + 0.7 * goal_progress
    return torch.where(touched, post_touch_raw, pre_touch_raw)

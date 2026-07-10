"""Pure-tensor 3-stage (grasp/lift/goal) progress math for a hierarchical-
decomposition research prototype (not yet a numbered production
experiment): unlike tasks/ar4/grasp_goal_reward.py's grasp_goal_progress
(Experiment 26's 4-stage reach/grasp/lift/goal formula), this has NO reach
stage at all, because this reward is meant to run on episodes that already
start from a real, physically-simulated post-reach state (see
tasks/ar4/mdp.py's reset_arm_from_handoff_bank and
scripts/harvest_reach_handoff_states.py) - reach is assumed already solved
by a separate, frozen upstream skill, not something this reward needs to
shape at all. No Isaac Lab dependency - testable with plain pytest+torch,
same pattern grasp_goal_reward.py and touch_goal_reward.py established.
"""

import torch


def grasp_lift_goal_progress(
    grasped: torch.Tensor,
    lifted: torch.Tensor,
    cube_height_above_ground: torch.Tensor,
    goal_dist: torch.Tensor,
    lift_target_height: float,
    cube_to_goal_dist: float,
) -> torch.Tensor:
    """Three equal 1/3-wide stage segments: grasp (0.00-0.33, a discrete
    achievement jump on genuine bilateral antipodal contact - the caller
    computes `grasped` via antipodal_grasp_bonus's own condition, exactly
    as grasp_goal_progress's grasp stage does, no partial credit before
    the gate fires), lift (0.33-0.67, linear ramp on cube height once
    grasped), goal (0.67-1.00, linear ramp on cube-to-goal distance once
    lifted). Monotonically non-decreasing along any trajectory where
    grasped/lifted latch true then cube_height/goal_dist improve, by the
    same construction grasp_goal_progress/touch_goal_progress use to avoid
    the dual-tanh-sum dead-zone bug (see
    [[staged-reward-co-satisfiability]] in this repo's kb).

    `grasped`/`lifted` are latched booleans (once true, stay true for the
    episode) - the caller (tasks/ar4/mdp.py) owns that state, this function
    is a pure function of whatever it's passed each call."""
    lift_progress = torch.clamp(cube_height_above_ground / lift_target_height, min=0.0, max=1.0)
    goal_progress = torch.clamp(1.0 - goal_dist / cube_to_goal_dist, min=0.0, max=1.0)

    grasp_stage = (1.0 / 3.0) + (1.0 / 3.0) * lift_progress
    goal_stage = (2.0 / 3.0) + (1.0 / 3.0) * goal_progress

    return torch.where(lifted, goal_stage, torch.where(grasped, grasp_stage, torch.zeros_like(grasp_stage)))

"""Pure-tensor 4-stage (reach/grasp/lift/goal) progress math for
Experiment 26 - no Isaac Lab dependency, so this is testable with plain
pytest+torch (see tests/test_grasp_goal_reward.py), the same pattern
tasks/ar4/touch_goal_reward.py established for Experiment 25.
tasks/ar4/mdp.py reads live simulated state (end-effector position, cube
position, contact forces via antipodal_grasp_bonus's own condition) and
delegates the actual staging formula to grasp_goal_progress() below.
"""

import torch


def grasp_goal_progress(
    reach_dist: torch.Tensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
    cube_height_above_ground: torch.Tensor,
    goal_dist: torch.Tensor,
    reach_dist_norm: float,
    lift_minimal_height: float,
    lift_target_height: float,
    cube_to_goal_dist: float,
) -> torch.Tensor:
    """Four equal 0.25-wide stage segments: reach (0.00-0.25, dense tanh-
    free linear proximity ramp, always active), grasp (0.25-0.50, a
    discrete achievement jump - genuine bilateral antipodal contact,
    computed by the caller via antipodal_grasp_bonus's own condition, IS
    the gate, not a shaped sub-metric, per Experiment 18's falsified
    dense-pre-grasp-readiness-shaping hypothesis - no partial credit
    within this segment beyond the reach ceiling already banked), lift
    (0.50-0.75, linear ramp on cube height once grasped), goal
    (0.75-1.00, linear ramp on cube-to-goal distance once lifted).
    Monotonically non-decreasing along any trajectory where reach_dist
    shrinks then grasped/lifted latch true then cube_height/goal_dist
    improve, by the same construction Experiment 25's
    touch_goal_reward.touch_goal_progress() used to avoid the dual-tanh-
    sum dead-zone bug - see
    docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md
    and [[staged-reward-co-satisfiability]] for that lesson generalized.

    `grasped`/`lifted` are latched booleans (once true, stay true for the
    episode) - the caller (tasks/ar4/mdp.py) owns that state, this
    function is a pure function of whatever it's passed each call.
    `lift_minimal_height` is accepted for interface symmetry with
    _raw_lift_progress_mirrored's own param name but is not used in this
    function's own math (the caller uses it to decide when `lifted`
    latches true in the first place, before calling this function)."""
    reach_progress = torch.clamp(1.0 - reach_dist / reach_dist_norm, min=0.0, max=1.0)
    lift_progress = torch.clamp(cube_height_above_ground / lift_target_height, min=0.0, max=1.0)
    goal_progress = torch.clamp(1.0 - goal_dist / cube_to_goal_dist, min=0.0, max=1.0)

    reach_stage = 0.25 * reach_progress
    grasp_stage = 0.50 + 0.25 * lift_progress
    goal_stage = 0.75 + 0.25 * goal_progress

    return torch.where(lifted, goal_stage, torch.where(grasped, grasp_stage, reach_stage))

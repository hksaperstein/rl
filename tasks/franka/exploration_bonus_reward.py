# tasks/franka/exploration_bonus_reward.py
"""Pure-tensor reward math for the GRM-D=1 gripper-closure-attempt
exploration bonus (Task 1 of docs/superpowers/plans/2026-07-19-exploration-
bonus-grasp-discovery-implementation.md, spec: docs/superpowers/specs/
2026-07-19-exploration-bonus-grasp-discovery-design.md). NO isaaclab
import - pure torch only, mirrors tasks/franka/lift_reward.py's and
tasks/franka/distractor_observations.py's established split:
tasks/franka/mdp.py reads live simulated state and delegates the actual
computation to the two functions below.

Implements PBIM/GRM (Forbes, Villalobos-Arias, Wang, Jhala, Roberts,
"Potential-Based Intrinsic Motivation: Preserving Optimality With Complex,
Non-Markovian Shaping Rewards," arXiv:2410.12197), instantiated at delay
D=1 (ADOPS, Forbes, Wang, Villalobos-Arias, Jhala, Roberts, "Action-
Dependent Optimality-Preserving Reward Shaping," arXiv:2505.12611, Eq. 8's
own delay-parameterized family - D=1 is that paper's own empirically
best-performing value, Sec. 6.2/Appendix A.3).

Two functions:

- gripper_closure_attempt_bonus_raw: the raw, action-dependent F_t (reward
  term 1, `gripper_closure_attempt_bonus` in the env cfg):

      F_t = w_attempt * tanh(k * relu(-raw_gripper_action_t)) * (1 - tanh(d_t / std_gate))

  per the spec's "Exact mechanism proposed" section, verbatim.

- gripper_closure_attempt_bonus_correction: the GRM D=1 correction term
  (reward term 2, `gripper_closure_attempt_bonus_correction` in the env
  cfg).

Double-counting resolution (load-bearing - see the implementation plan's
own "Design notes" section for the full derivation; reproduced here so
this module stays self-contained). The spec's own literal formula gives
`F'_t` (the full GRM-D=1-corrected shaping reward) as a single 3-branch
piecewise function:

    F'_t = F_t                          if t = 0
         = F_t - (1/gamma) * F_{t-1}     if 1 <= t < N-1
         = -(1/gamma) * F_{N-2}          if t = N-1

But term 2 does NOT return F'_t verbatim - term 1 already returns F_t
unconditionally, every step (per the spec), so if term 2 also returned
F'_t verbatim, F_t would be double-counted (Term1 + Term2 = F_t + F'_t =
2*F_t at t=0, instead of the intended F'_t). Instead, term 2 returns ONLY
the correction piece:

    Correction_t := F'_t - F_t

which algebraically works out to:

    Correction_t = 0                             if is_first_step        (F'_0 - F_0 = 0)
                 = -(1/gamma) * F_{t-1}            if not first, not last  (F_t cancels)
                 = -(1/gamma) * F_{t-1} - F_t       if is_last_step         (extra -F_t: a bonus
                                                                             paid on the very last
                                                                             step never gets a future
                                                                             step to be matched at)

so that Term1_t + Term2_t == F'_t (the spec's own literal formula) exactly,
at every step including both boundaries. This identity is this module's
own safety-net test (tests/test_exploration_bonus_reward.py,
TestSafetyNetTerm1PlusTerm2EqualsSpecFPrime), checked against the spec's
3-branch formula independently re-derived in the test file (not by calling
this module's own functions to generate the "expected" value) - so it
catches a double-counting regression regardless of which side's algebra
turns out to have the bug.

Correction_t carries NO non-negativity (or any other sign) claim -
unlike Experiment 5's informal, and false, "always >= 0" claim about its
own hand-derived running-max potential
(kb/wiki/experiments/experiment-05-potential-based-reward-shaping.md).
This design's correctness rests entirely on GRM's own proved Theorem 1
(boundary condition, Eq. 25) + Assumption 1 (future-agnostic F_t) + the
delay-D matching-function conditions (Eqs. 46-49) holding by construction,
never on any per-step sign guarantee.
"""

from __future__ import annotations

import torch


def gripper_closure_attempt_bonus_raw(
    raw_gripper_action: torch.Tensor,
    cube_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    w_attempt: float,
    k: float,
    std_gate: float,
) -> torch.Tensor:
    """F_t = w_attempt * tanh(k * relu(-raw_gripper_action)) * (1 - tanh(d_t / std_gate)),
    per the spec's "Exact mechanism proposed" section, verbatim.

    Args:
        raw_gripper_action: the policy's own raw (pre-threshold) output for
            the gripper action dimension, shape (N,). Negative values
            command "close" (`BinaryJointAction`'s
            `binary_mask = actions < 0`, per the spec's own confirmed
            source read) - `relu(-raw_gripper_action)` is therefore
            positive exactly when the policy is attempting closure, zero
            otherwise ("open" or neutral/non-negative action).
        cube_pos: cube position in world frame, shape (N, 3). Same raw-
            position-input convention as
            `lift_reward.reaching_object_reward`'s `(cube_pos, ee_pos)`
            pair (not a pre-computed distance).
        ee_pos: end-effector position in world frame, shape (N, 3).
        w_attempt: bonus weight scalar (implementer-set starting value, not
            tuned by this module - see the plan's Global Constraints).
        k: relu-output tanh-saturation steepness scalar (implementer-set).
        std_gate: proximity-gate length scale, matching
            `lift_reward`'s `1 - tanh(distance / std)` kernel convention
            (implementer-set; the plan's starting value is `0.05`,
            matching `object_goal_tracking_fine_grained`'s existing
            fine-grained std, not `reaching_object`'s loose `0.1`).

    Returns:
        Tensor, shape (N,), values in [0, w_attempt) - strictly less than
        w_attempt since tanh saturates but never reaches 1, and always
        non-negative (relu + tanh-of-nonnegative + a [0, 1)-valued gate
        term are each individually non-negative).
    """
    d_t = torch.norm(cube_pos - ee_pos, dim=-1)
    closure_attempt = torch.tanh(k * torch.relu(-raw_gripper_action))
    proximity_gate = 1.0 - torch.tanh(d_t / std_gate)
    return w_attempt * closure_attempt * proximity_gate


def gripper_closure_attempt_bonus_correction(
    F_t: torch.Tensor,
    F_prev: torch.Tensor,
    is_first_step: torch.Tensor,
    is_last_step: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Correction_t := F'_t - F_t, per this module's own docstring
    derivation above (NOT F'_t verbatim - see "Double-counting resolution").

    Args:
        F_t: this step's raw bonus (`gripper_closure_attempt_bonus_raw`'s
            own output), shape (N,) - used only for the `is_last_step`
            branch's own `-F_t` term.
        F_prev: the previous step's raw bonus F_{t-1}, shape (N,) -
            caller-owned persistent state (see
            `GripperClosureAttemptBonusCorrection` in `tasks/franka/mdp.py`,
            Task 2). Value is irrelevant wherever `is_first_step` is True.
        is_first_step: (N,) bool tensor, caller-supplied - True for an
            env's episode-first control step (no real `F_{t-1}` exists
            yet). This function has zero knowledge of episode bookkeeping
            itself - pure arithmetic only, exactly the pattern this
            project's TDD discipline requires for something with no
            live-sim state to fake.
        is_last_step: (N,) bool tensor, caller-supplied - True for an
            env's episode-final control step. If both `is_first_step` and
            `is_last_step` are True for the same env (a degenerate case
            not expected in this project's 250-step episodes), the
            first-step rule takes priority and 0.0 is returned.
        gamma: MUST equal `FrankaLiftPPORunnerCfg.algorithm.gamma` exactly
            (`0.98`, `tasks/franka/agents/rsl_rl_ppo_cfg.py:50`) - see the
            plan's Global Constraints for why a mismatched gamma breaks the
            policy-invariance guarantee this mechanism exists to provide.

    Returns:
        Tensor, shape (N,), sign UNCONSTRAINED (see module docstring - this
        is a deliberate departure from Experiment 5's false "always >= 0"
        claim; no non-negativity property is claimed or relied upon here).
    """
    base_correction = -(1.0 / gamma) * F_prev
    terminal_extra = torch.where(is_last_step, -F_t, torch.zeros_like(F_t))
    correction = base_correction + terminal_extra
    return torch.where(is_first_step, torch.zeros_like(F_t), correction)

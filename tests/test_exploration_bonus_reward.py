"""Sim-independent unit tests for tasks/franka/exploration_bonus_reward.py's
pure GRM-D=1 gripper-closure-attempt-bonus math (Task 1 of
docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-discovery-
implementation.md, spec: docs/superpowers/specs/2026-07-19-exploration-
bonus-grasp-discovery-design.md). No Isaac Lab import needed - mirrors
tests/test_distractor_observations.py's own scope: this file tests the two
pure functions tasks/franka/mdp.py's thin wrappers delegate to directly,
with raw tensors, not the live-sim wiring (that's Task 2). Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_exploration_bonus_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory).

The single most important test class here is
TestSafetyNetTerm1PlusTerm2EqualsSpecFPrime - it independently re-derives
the spec's own literal 3-branch F'_t formula over a synthetic MULTI-EPISODE
trajectory (so the reset/boundary handling is exercised at a real episode
boundary, not just a single episode's start/end) and asserts
Term1_t + Term2_t == F'_t at every step. This is designed to catch a
double-counting regression regardless of which side of the implementation
plan's "Design notes" derivation is actually correct.
"""

import math

import torch

from tasks.franka.exploration_bonus_reward import (
    gripper_closure_attempt_bonus_correction,
    gripper_closure_attempt_bonus_raw,
)


def _pos(x, y, z):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# gripper_closure_attempt_bonus_raw (F_t)
# ---------------------------------------------------------------------------


class TestGripperClosureAttemptBonusRawNoAttempt:
    def test_zero_when_raw_action_is_positive_regardless_of_distance(self):
        raw_action = torch.tensor([0.5, 2.0, 7.77])
        cube_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 3, dim=0)
        ee_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 3, dim=0)  # distance 0 - closest possible
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        assert torch.all(out == 0.0)

    def test_zero_when_raw_action_is_exactly_zero(self):
        raw_action = torch.tensor([0.0])
        cube_pos = _pos(0.0, 0.0, 0.0)
        ee_pos = _pos(0.0, 0.0, 0.0)
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        assert out.item() == 0.0


class TestGripperClosureAttemptBonusRawMonotonicAndSaturating:
    def test_more_negative_action_gives_larger_bonus_near_object(self):
        actions = torch.tensor([-0.1, -1.0, -5.0, -20.0])
        cube_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 4, dim=0)
        ee_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 4, dim=0)
        out = gripper_closure_attempt_bonus_raw(actions, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        vals = out.tolist()
        assert vals[0] < vals[1] < vals[2] < vals[3]

    def test_saturates_below_w_attempt_for_very_large_negative_action(self):
        # action=-7.0 (k=1.0) puts tanh's argument at 7.0: 1 - tanh(7.0) ~= 1.8e-6,
        # still representable as distinct from exactly 1.0 in float32 (epsilon
        # ~1.19e-7). A much larger magnitude (e.g. -1000.0) underflows tanh's
        # output to the literal float32 value 1.0, which would make the
        # "strictly saturates below w_attempt" assertion fail on a float32
        # representation artifact rather than test anything about the formula.
        raw_action = torch.tensor([-7.0])
        cube_pos = _pos(0.0, 0.0, 0.0)
        ee_pos = _pos(0.0, 0.0, 0.0)
        w_attempt = 1.0
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=w_attempt, k=1.0, std_gate=0.05)
        assert out.item() < w_attempt
        assert out.item() > 0.999 * w_attempt  # saturated close to the bound

    def test_diminishing_returns_confirms_saturation_not_linear_growth(self):
        cube_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 3, dim=0)
        ee_pos = torch.cat([_pos(0.0, 0.0, 0.0)] * 3, dim=0)
        actions = torch.tensor([-1.0, -2.0, -4.0])
        out = gripper_closure_attempt_bonus_raw(actions, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        vals = out.tolist()
        step1 = vals[1] - vals[0]  # bonus gained going from action -1 -> -2
        step2 = vals[2] - vals[1]  # bonus gained going from action -2 -> -4 (twice the action delta)
        assert step2 < step1  # a linear (unbounded) term would give step2 > step1 here


class TestGripperClosureAttemptBonusRawProximityGate:
    def test_near_zero_far_from_object_even_with_strongly_negative_action(self):
        raw_action = torch.tensor([-50.0])
        cube_pos = _pos(0.0, 0.0, 0.0)
        ee_pos = _pos(5.0, 0.0, 0.0)  # 5m away, std_gate=0.05 -> d/std_gate=100
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        assert out.item() < 1e-6

    def test_gate_actually_gates_near_vs_far_same_action(self):
        raw_action = torch.tensor([-10.0, -10.0])
        cube_pos = torch.cat([_pos(0.0, 0.0, 0.0), _pos(0.0, 0.0, 0.0)], dim=0)
        ee_pos = torch.cat([_pos(0.0, 0.0, 0.0), _pos(1.0, 0.0, 0.0)], dim=0)  # env 0 at cube, env 1 1m away
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=1.0, k=1.0, std_gate=0.05)
        assert out[0].item() > out[1].item()
        assert out[1].item() < 1e-3


class TestGripperClosureAttemptBonusRawKnownValue:
    def test_known_value_at_specific_action_distance_pair(self):
        # w_attempt=2.0, k=1.0, action=-1.0, std_gate=0.1, distance=0.1 (d/std_gate=1)
        # F = 2.0 * tanh(1.0 * relu(1.0)) * (1 - tanh(0.1/0.1))
        #   = 2.0 * tanh(1.0) * (1 - tanh(1.0))
        raw_action = torch.tensor([-1.0])
        cube_pos = _pos(0.0, 0.0, 0.0)
        ee_pos = _pos(0.1, 0.0, 0.0)
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=2.0, k=1.0, std_gate=0.1)
        expected = 2.0 * math.tanh(1.0) * (1.0 - math.tanh(1.0))
        assert math.isclose(out.item(), expected, rel_tol=1e-5)

    def test_known_value_at_zero_distance(self):
        # distance 0 -> gate term is exactly 1 - tanh(0) = 1.0
        raw_action = torch.tensor([-3.0])
        cube_pos = _pos(1.0, 2.0, 3.0)
        ee_pos = _pos(1.0, 2.0, 3.0)
        out = gripper_closure_attempt_bonus_raw(raw_action, cube_pos, ee_pos, w_attempt=1.0, k=0.5, std_gate=0.05)
        expected = 1.0 * math.tanh(0.5 * 3.0) * 1.0
        assert math.isclose(out.item(), expected, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# gripper_closure_attempt_bonus_correction (Correction_t)
# ---------------------------------------------------------------------------


class TestCorrectionFirstStepBoundary:
    def test_zero_when_is_first_step_regardless_of_f_prev(self):
        for f_prev_val in [0.0, 5.0, -3.0, 999.0]:
            F_t = torch.tensor([0.3])
            F_prev = torch.tensor([f_prev_val])
            is_first = torch.tensor([True])
            is_last = torch.tensor([False])
            out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma=0.98)
            assert out.item() == 0.0, f"expected 0.0 at first step regardless of F_prev={f_prev_val}"

    def test_zero_at_first_step_even_if_also_flagged_last_step(self):
        # Degenerate case (shouldn't occur in this project's 250-step episodes,
        # but the function's own branch order should still resolve unambiguously
        # to the first-step rule).
        F_t = torch.tensor([0.7])
        F_prev = torch.tensor([0.4])
        is_first = torch.tensor([True])
        is_last = torch.tensor([True])
        out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma=0.98)
        assert out.item() == 0.0


class TestCorrectionMiddleStepFormula:
    def test_plain_discounted_f_prev_formula_for_non_first_non_last_step(self):
        F_t = torch.tensor([0.6])
        F_prev = torch.tensor([0.4])
        is_first = torch.tensor([False])
        is_last = torch.tensor([False])
        gamma = 0.98
        out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma=gamma)
        expected = -(1.0 / gamma) * 0.4
        assert math.isclose(out.item(), expected, rel_tol=1e-6)

    def test_middle_step_does_not_depend_on_f_t(self):
        F_prev = torch.tensor([0.4])
        is_first = torch.tensor([False])
        is_last = torch.tensor([False])
        gamma = 0.98
        out_a = gripper_closure_attempt_bonus_correction(torch.tensor([0.1]), F_prev, is_first, is_last, gamma)
        out_b = gripper_closure_attempt_bonus_correction(torch.tensor([0.9]), F_prev, is_first, is_last, gamma)
        assert math.isclose(out_a.item(), out_b.item(), rel_tol=1e-6)


class TestCorrectionLastStepFormula:
    def test_extra_minus_f_t_term_present_only_when_last_step(self):
        F_t = torch.tensor([0.6])
        F_prev = torch.tensor([0.4])
        is_first = torch.tensor([False])
        gamma = 0.98

        out_not_last = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, torch.tensor([False]), gamma)
        out_last = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, torch.tensor([True]), gamma)

        assert math.isclose(out_last.item() - out_not_last.item(), -0.6, rel_tol=1e-6)

    def test_last_step_known_value(self):
        F_t = torch.tensor([0.6])
        F_prev = torch.tensor([0.4])
        is_first = torch.tensor([False])
        is_last = torch.tensor([True])
        gamma = 0.98
        out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma)
        expected = -(1.0 / gamma) * 0.4 - 0.6
        assert math.isclose(out.item(), expected, rel_tol=1e-6)


class TestCorrectionSignIsNotConstrained:
    """Direct regression test against Experiment 5's own false "always >= 0"
    failure class (kb/wiki/experiments/experiment-05-potential-based-reward-
    shaping.md) - this design makes no sign claim about Correction_t at all,
    unlike Experiment 5's incorrect informal claim about its own hand-derived
    potential. The function is pure arithmetic with no episode-bookkeeping
    knowledge, so it must not clamp/clip to any particular sign regardless
    of the (here, deliberately atypical/negative) F_prev values fed in."""

    def test_correction_can_be_positive(self):
        # A negative F_prev (atypical for the real deployed F_t, which is
        # always >= 0 by construction - see gripper_closure_attempt_bonus_raw's
        # own docstring - but this function has no knowledge of that upstream
        # invariant and must not assume it) makes the correction positive.
        F_t = torch.tensor([0.0])
        F_prev = torch.tensor([-1.0])
        is_first = torch.tensor([False])
        is_last = torch.tensor([False])
        out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma=0.98)
        assert out.item() > 0.0

    def test_correction_can_be_negative(self):
        F_t = torch.tensor([0.0])
        F_prev = torch.tensor([1.0])
        is_first = torch.tensor([False])
        is_last = torch.tensor([False])
        out = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma=0.98)
        assert out.item() < 0.0


# ---------------------------------------------------------------------------
# Safety net: Term1_t + Term2_t == spec's own literal F'_t, over a synthetic
# multi-episode trajectory (the single most important test in this file).
# ---------------------------------------------------------------------------


def _spec_f_prime_reference(f_values, gamma):
    """Independent re-derivation of the spec's own literal 3-branch F'_t
    formula (docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-
    discovery-design.md, "Exact mechanism proposed"), applied to one
    episode's worth of F_t values (a plain Python list, one float per
    step, length N):

        F'_t = F_t                       if t = 0
             = F_t - (1/gamma)*F_{t-1}    if 1 <= t < N-1
             = -(1/gamma)*F_{N-2}         if t = N-1

    Computed independently of gripper_closure_attempt_bonus_correction -
    this reference does not call that function at all, so the safety-net
    test below catches a double-counting regression regardless of which
    side's algebra has the bug.
    """
    n = len(f_values)
    f_prime = []
    for t in range(n):
        if t == 0:
            f_prime.append(f_values[0])
        elif t < n - 1:
            f_prime.append(f_values[t] - (1.0 / gamma) * f_values[t - 1])
        else:  # t == n - 1
            f_prime.append(-(1.0 / gamma) * f_values[n - 2])
    return f_prime


class TestSafetyNetTerm1PlusTerm2EqualsSpecFPrime:
    def test_term1_plus_term2_equals_spec_f_prime_single_episode(self):
        gamma = 0.98
        # One synthetic 5-step episode's worth of raw bonus values.
        f_values = [0.2, 0.5, 0.9, 0.1, 0.4]
        n = len(f_values)
        expected_f_prime = _spec_f_prime_reference(f_values, gamma)

        f_prev = 0.0  # buffer starts at 0, matching the "reset to 0" convention
        for t in range(n):
            F_t = torch.tensor([f_values[t]])
            F_prev = torch.tensor([f_prev])
            is_first = torch.tensor([t == 0])
            is_last = torch.tensor([t == n - 1])

            term1 = F_t
            term2 = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma)
            summed = (term1 + term2).item()

            assert math.isclose(summed, expected_f_prime[t], rel_tol=1e-6, abs_tol=1e-9), (
                f"t={t}: Term1+Term2={summed} != spec F'_t={expected_f_prime[t]}"
            )

            f_prev = f_values[t]  # buffer update: "self._prev_raw = F_t for the next call"

    def test_term1_plus_term2_equals_spec_f_prime_multi_episode_multi_env(self):
        """The load-bearing test: TWO consecutive episodes, TWO envs with
        different F_t sequences and different episode lengths, run through
        a step-by-step simulation of the real GripperClosureAttemptBonusCorrection
        buffer's own lifecycle (reset F_prev to 0 at each episode boundary,
        then update it to F_t after every step) - so the reset/boundary
        handling is exercised at a REAL episode transition, not just a
        single episode's own start/end."""
        gamma = 0.98

        # Env A: two episodes, lengths 4 and 3.
        episodes_a = [[0.1, 0.6, 0.3, 0.8], [0.5, 0.2, 0.9]]
        # Env B: two episodes, lengths 5 and 2 (deliberately different from env A,
        # to confirm no cross-env leakage in the per-env boundary handling).
        episodes_b = [[0.05, 0.15, 0.55, 0.35, 0.65], [0.9, 0.1]]

        for episodes, label in [(episodes_a, "env A"), (episodes_b, "env B")]:
            f_prev = 0.0
            for episode in episodes:
                n = len(episode)
                expected_f_prime = _spec_f_prime_reference(episode, gamma)
                f_prev = 0.0  # buffer reset to 0 at episode start (event-manager reset hook)
                for t in range(n):
                    F_t = torch.tensor([episode[t]])
                    F_prev = torch.tensor([f_prev])
                    is_first = torch.tensor([t == 0])
                    is_last = torch.tensor([t == n - 1])

                    term1 = F_t
                    term2 = gripper_closure_attempt_bonus_correction(F_t, F_prev, is_first, is_last, gamma)
                    summed = (term1 + term2).item()

                    assert math.isclose(summed, expected_f_prime[t], rel_tol=1e-6, abs_tol=1e-9), (
                        f"{label}, t={t}: Term1+Term2={summed} != spec F'_t={expected_f_prime[t]}"
                    )

                    f_prev = episode[t]

    def test_term1_plus_term2_equals_spec_f_prime_batched_across_envs_same_call(self):
        """Same identity, but exercised through a single batched (N=3) call
        per step rather than one env at a time - confirms the vectorized
        implementation doesn't silently only work in the scalar/single-env
        case."""
        gamma = 0.98
        # 3 envs, all 4-step episodes, different F_t values per env.
        episode_env0 = [0.1, 0.4, 0.2, 0.7]
        episode_env1 = [0.9, 0.05, 0.6, 0.3]
        episode_env2 = [0.0, 0.0, 0.5, 0.0]
        episodes = [episode_env0, episode_env1, episode_env2]
        n = 4
        expected = [_spec_f_prime_reference(ep, gamma) for ep in episodes]

        f_prev = torch.zeros(3)
        for t in range(n):
            F_t = torch.tensor([ep[t] for ep in episodes])
            is_first = torch.tensor([t == 0] * 3)
            is_last = torch.tensor([t == n - 1] * 3)

            term1 = F_t
            term2 = gripper_closure_attempt_bonus_correction(F_t, f_prev, is_first, is_last, gamma)
            summed = term1 + term2

            for env_idx in range(3):
                assert math.isclose(summed[env_idx].item(), expected[env_idx][t], rel_tol=1e-6, abs_tol=1e-9), (
                    f"env {env_idx}, t={t}: {summed[env_idx].item()} != {expected[env_idx][t]}"
                )

            f_prev = F_t

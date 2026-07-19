"""Sim-independent unit tests for tasks/franka/distractor_observations.py's
pure distractor-distance-summary math (Task 2 of docs/superpowers/plans/
2026-07-19-target-selection-clutter-implementation.md, spec:
docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md - the
new observation term implementing DexSinGrasp's own `d_t^S` mechanism,
arXiv:2504.04516 §III-A Eq. 1). No Isaac Lab import needed - mirrors
tests/test_mdp_shape_observations.py's own scope: tasks/franka/mdp.py's
distractor_distance_summary thin wrapper imports isaaclab at module level
per its own docstring ("Import this module only after an Isaac Sim/Isaac
Lab AppLauncher has been created"), so it's exercised only by actually
running the env, not here; this file tests the pure function the wrapper
delegates to directly, with raw tensors. Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_distractor_observations.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory)."""

import math

import torch

from tasks.franka.distractor_observations import distractor_distance_summary


def _pos(x, y, z):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


class TestDistractorDistanceSummaryShape:
    def test_output_shape_is_num_envs_by_2(self):
        num_envs = 5
        target = torch.zeros(num_envs, 3)
        d1 = torch.ones(num_envs, 3)
        d2 = torch.ones(num_envs, 3) * 2.0
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=2)
        assert out.shape == (num_envs, 2)

    def test_num_envs_one(self):
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(1.0, 0.0, 0.0)
        d2 = _pos(0.0, 1.0, 0.0)
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=2)
        assert out.shape == (1, 2)


class TestDistractorDistanceSummaryActiveSlots:
    def test_correct_euclidean_distance_slot_0_active(self):
        """3-4-5 triangle: target at origin, distractor_1 at (3, 4, 0) ->
        distance exactly 5.0."""
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(3.0, 4.0, 0.0)
        d2 = _pos(0.0, 0.0, 0.0)  # inactive, value irrelevant
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        assert math.isclose(out[0, 0].item(), 5.0, rel_tol=1e-5)

    def test_correct_euclidean_distance_slot_1_active(self):
        """Both slots active: distractor_1 at distance 5 (3-4-5), distractor_2
        at distance 13 (5-12-13) from the target."""
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(3.0, 4.0, 0.0)
        d2 = _pos(5.0, 12.0, 0.0)
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=2)
        assert math.isclose(out[0, 0].item(), 5.0, rel_tol=1e-5)
        assert math.isclose(out[0, 1].item(), 13.0, rel_tol=1e-5)

    def test_correct_euclidean_distance_3d(self):
        target = _pos(1.0, 1.0, 1.0)
        d1 = _pos(1.0, 1.0, 3.0)  # distance 2.0 along z only
        d2 = _pos(1.0, 1.0, 1.0)  # inactive
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        assert math.isclose(out[0, 0].item(), 2.0, rel_tol=1e-5)


class TestDistractorDistanceSummaryHardZeroPadding:
    """The load-bearing correctness property: an inactive slot is HARD
    zeroed, never the real (possibly large, parked-off-table) distance -
    DexSinGrasp's own literal zero-padding convention, not a "far away"
    sentinel."""

    def test_active_distractor_count_zero_both_slots_hard_zero(self):
        target = _pos(0.0, 0.0, 0.0)
        # Both distractors placed FAR from the target - if zero-padding were
        # broken (e.g. accidentally reporting real distance), this would be
        # a large nonzero value instead of exactly 0.0.
        d1 = _pos(100.0, 100.0, 100.0)
        d2 = _pos(-500.0, 300.0, 50.0)
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=0)
        assert out[0, 0].item() == 0.0
        assert out[0, 1].item() == 0.0

    def test_active_distractor_count_one_slot_1_real_slot_2_hard_zero(self):
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(3.0, 4.0, 0.0)  # real, active -> 5.0
        d2 = _pos(-1000.0, 1000.0, 1000.0)  # far away, but INACTIVE -> must be exactly 0
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        assert math.isclose(out[0, 0].item(), 5.0, rel_tol=1e-5)
        assert out[0, 1].item() == 0.0, "inactive slot must be hard-zeroed, not the real parked-off-table distance"

    def test_active_distractor_count_two_both_slots_real_nonzero(self):
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(3.0, 4.0, 0.0)
        d2 = _pos(6.0, 8.0, 0.0)
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=2)
        assert out[0, 0].item() != 0.0
        assert out[0, 1].item() != 0.0

    def test_zero_and_nonzero_column_pattern_matches_active_count_across_0_1_2(self):
        target = _pos(0.0, 0.0, 0.0)
        d1 = _pos(3.0, 4.0, 0.0)  # nonzero real distance = 5.0
        d2 = _pos(6.0, 8.0, 0.0)  # nonzero real distance = 10.0

        out0 = distractor_distance_summary(target, d1, d2, active_distractor_count=0)
        assert out0[0, 0].item() == 0.0 and out0[0, 1].item() == 0.0

        out1 = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        assert out1[0, 0].item() != 0.0 and out1[0, 1].item() == 0.0

        out2 = distractor_distance_summary(target, d1, d2, active_distractor_count=2)
        assert out2[0, 0].item() != 0.0 and out2[0, 1].item() != 0.0


class TestDistractorDistanceSummaryPerEnvVarying:
    """Unlike shape_class/geometry_descriptor's per-env-cfg-constant
    broadcast, this term is genuinely per-environment-varying (each env's
    own live object positions differ) - a real behavioral difference worth
    asserting explicitly, not assumed."""

    def test_batch_of_multiple_envs_produces_per_row_varying_distances(self):
        # 3 envs, each with a different target/distractor_1 separation.
        target = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        d1 = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # distance 1.0
                [2.0, 0.0, 0.0],  # distance 2.0
                [3.0, 0.0, 0.0],  # distance 3.0
            ],
            dtype=torch.float32,
        )
        d2 = torch.zeros(3, 3)
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        assert math.isclose(out[0, 0].item(), 1.0, rel_tol=1e-5)
        assert math.isclose(out[1, 0].item(), 2.0, rel_tol=1e-5)
        assert math.isclose(out[2, 0].item(), 3.0, rel_tol=1e-5)
        # rows must genuinely differ from each other - not a broadcast constant
        assert out[0, 0].item() != out[1, 0].item()
        assert out[1, 0].item() != out[2, 0].item()

    def test_inactive_column_stays_zero_across_all_rows_of_a_varying_batch(self):
        num_envs = 4
        target = torch.zeros(num_envs, 3)
        d1 = torch.arange(num_envs, dtype=torch.float32).unsqueeze(1) * torch.tensor([[1.0, 0.0, 0.0]])
        d2 = torch.arange(num_envs, dtype=torch.float32).unsqueeze(1) * torch.tensor([[0.0, 5.0, 0.0]])
        out = distractor_distance_summary(target, d1, d2, active_distractor_count=1)
        for row in range(num_envs):
            assert out[row, 1].item() == 0.0

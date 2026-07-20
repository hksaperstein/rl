"""Sim-independent unit tests for tasks/franka/antipodal_grasp_reward.py's
pure bilateral force-closure/antipodal grasp-quality math (Task 1 of
docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-quality-implementation.md,
spec: docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md).
No Isaac Lab import needed - mirrors tests/test_exploration_bonus_reward.py's
own scope: this file tests the one pure function tasks/franka/mdp.py's thin
wrapper (antipodal_grasp_bonus) delegates to directly, with raw synthetic
(N, 3) force tensors, not the live-sim ContactSensor wiring (that is Task 1's
own separate empirical check, not a pytest test). Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_franka_antipodal_grasp_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment).

The single most important test here is
TestThresholdBoundaryRegression::test_cos_minus_0_80_fails_at_franka_threshold
- a synthetic cos_angle of exactly -0.80 sits strictly between AR4's own
mu=1.0 threshold (-0.7071) and this scene's real mu=0.5 threshold
(-0.894427). It would silently PASS (return 1.0) if this module accidentally
carried over AR4's looser -0.7071 threshold, but must FAIL (return 0.0) at
the correct -0.894427 value - the one test that actually catches an
accidental threshold-carryover regression, per the implementation plan's
own "Design notes" #2.
"""

import torch

from tasks.franka.antipodal_grasp_reward import antipodal_grasp_bonus_raw

# This scene's real, physically-derived threshold (mu=0.5 friction,
# -cos(arctan(0.5)) = -0.894427) - see antipodal_grasp_reward.py's own module
# docstring for the full derivation. NOT AR4's own mu=1.0 value (-0.7071).
FRANKA_THRESHOLD = -0.894427
FORCE_THRESHOLD = 0.05


def _unit_pair(cos_val: float, mag1: float = 1.0, mag2: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one (1, 3) force-vector pair whose direction cosine is exactly
    cos_val: jaw1 points along +X, jaw2 points at angle arccos(cos_val) from
    +X in the XY plane, each scaled to its own requested magnitude."""
    sin_val = (1.0 - cos_val**2) ** 0.5
    jaw1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32) * mag1
    jaw2 = torch.tensor([[cos_val, sin_val, 0.0]], dtype=torch.float32) * mag2
    return jaw1, jaw2


class TestKnownAntipodalPair:
    def test_opposing_x_forces_above_threshold_returns_one(self):
        jaw1 = torch.tensor([[-1.0, 0.0, 0.0]])  # along -X
        jaw2 = torch.tensor([[1.0, 0.0, 0.0]])  # along +X (exactly anti-parallel, cos=-1.0)
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 1.0


class TestKnownNonAntipodalPair:
    def test_perpendicular_forces_above_threshold_returns_zero(self):
        jaw1 = torch.tensor([[1.0, 0.0, 0.0]])
        jaw2 = torch.tensor([[0.0, 1.0, 0.0]])  # perpendicular, cos_angle == 0
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 0.0


class TestMagnitudeTooSmall:
    def test_perfect_antipodal_direction_but_one_magnitude_below_threshold(self):
        jaw1 = torch.tensor([[-1.0, 0.0, 0.0]]) * 0.01  # below FORCE_THRESHOLD=0.05
        jaw2 = torch.tensor([[1.0, 0.0, 0.0]]) * 1.0  # well above threshold
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 0.0


class TestThresholdBoundaryRegression:
    def test_cos_minus_0_80_fails_at_franka_threshold(self):
        # -0.80 sits strictly between AR4's own -0.7071 (mu=1.0) and this
        # scene's real -0.894427 (mu=0.5) - would wrongly pass under AR4's
        # looser threshold, must fail under the correct Franka one.
        jaw1, jaw2 = _unit_pair(-0.80)
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 0.0


class TestBoundaryExactness:
    def test_cos_exactly_at_threshold_returns_zero_strict_less_than(self):
        jaw1, jaw2 = _unit_pair(FRANKA_THRESHOLD)
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 0.0

    def test_cos_just_past_threshold_returns_one(self):
        jaw1, jaw2 = _unit_pair(-0.90)  # just past -0.894427 (more negative)
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 1.0


class TestZeroForceVector:
    def test_both_jaws_zero_force_returns_zero_not_nan(self):
        jaw1 = torch.zeros((1, 3))
        jaw2 = torch.zeros((1, 3))
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.item() == 0.0
        assert not torch.isnan(out).any()


class TestBatchProcessing:
    def test_mixed_batch_gives_correct_per_env_tensor(self):
        # env 0: antipodal, both magnitudes ok -> 1.0
        # env 1: non-antipodal (perpendicular), both magnitudes ok -> 0.0
        # env 2: antipodal direction, but jaw2 magnitude below threshold -> 0.0
        jaw1 = torch.stack(
            [
                torch.tensor([-1.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0]),
                torch.tensor([-1.0, 0.0, 0.0]),
            ]
        )
        jaw2 = torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0]) * 0.01,
            ]
        )
        out = antipodal_grasp_bonus_raw(jaw1, jaw2, force_threshold=FORCE_THRESHOLD, antipodal_cos_threshold=FRANKA_THRESHOLD)
        assert out.shape == (3,)
        assert out.tolist() == [1.0, 0.0, 0.0]

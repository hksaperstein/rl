"""Sim-independent unit tests for tasks/franka/lift_reward.py's pure reward
math — no Isaac Lab import needed. Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_franka_lift_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory)."""

import torch

from tasks.franka.lift_reward import (
    reaching_object_reward,
    lifting_object_reward,
    object_goal_distance_reward,
)

# Hyperparameters used in tests
REACHING_STD = 0.1
MINIMAL_HEIGHT = 0.03
GOAL_STD = 0.1


class TestReachingObjectReward:
    """Tests for reaching_object_reward (end-effector-to-cube proximity)."""

    def test_distance_zero_gives_one(self):
        """At distance 0, reward should be 1.0 (since 1 - tanh(0) = 1 - 0 = 1)."""
        cube_pos = torch.tensor([[0.0, 0.0, 0.0]])
        ee_pos = torch.tensor([[0.0, 0.0, 0.0]])
        reward = reaching_object_reward(cube_pos, ee_pos, REACHING_STD)
        assert abs(reward[0].item() - 1.0) < 1e-6, f"expected ~1.0 at distance 0, got {reward[0].item()}"

    def test_known_distance_matches_tanh_kernel(self):
        """Verify against direct tanh computation for a known configuration."""
        # Cube at origin, ee at [1, 0, 0], so distance = 1
        cube_pos = torch.tensor([[0.0, 0.0, 0.0]])
        ee_pos = torch.tensor([[1.0, 0.0, 0.0]])
        distance = 1.0
        expected = 1.0 - torch.tanh(torch.tensor(distance / REACHING_STD))

        reward = reaching_object_reward(cube_pos, ee_pos, REACHING_STD)

        assert abs(reward[0].item() - expected.item()) < 1e-6, (
            f"distance {distance}: expected {expected.item()}, got {reward[0].item()}"
        )

    def test_batch_processing(self):
        """Multiple environments (batch > 1) should produce corresponding reward batch."""
        cube_pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        ee_pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        reward = reaching_object_reward(cube_pos, ee_pos, REACHING_STD)

        assert reward.shape == (3,), f"expected shape (3,), got {reward.shape}"
        assert torch.all(reward >= 0.0) and torch.all(reward <= 1.0), "rewards should be in [0, 1]"
        # All positions match, so all should be ~1.0
        assert torch.all(reward > 0.99), f"all should be ~1.0, got {reward.tolist()}"

    def test_monotonic_in_distance(self):
        """Reward should monotonically decrease as distance increases."""
        distances = torch.linspace(0.0, 1.0, 50)
        cube_pos = torch.stack([torch.tensor([d, 0.0, 0.0]) for d in distances])
        ee_pos = torch.zeros(50, 3)

        reward = reaching_object_reward(cube_pos, ee_pos, REACHING_STD)
        deltas = reward[1:] - reward[:-1]

        assert torch.all(deltas <= 1e-5), f"reward should decrease monotonically, min delta {deltas.min().item()}"


class TestLiftingObjectReward:
    """Tests for lifting_object_reward (discrete binary lifting gate)."""

    def test_below_threshold_gives_zero(self):
        """Height strictly below minimal_height should give reward 0.0."""
        cube_height = torch.tensor([MINIMAL_HEIGHT - 0.01])
        reward = lifting_object_reward(cube_height, MINIMAL_HEIGHT)
        assert abs(reward[0].item() - 0.0) < 1e-6, f"expected 0.0 below threshold, got {reward[0].item()}"

    def test_above_threshold_gives_one(self):
        """Height strictly above minimal_height should give reward 1.0."""
        cube_height = torch.tensor([MINIMAL_HEIGHT + 0.01])
        reward = lifting_object_reward(cube_height, MINIMAL_HEIGHT)
        assert abs(reward[0].item() - 1.0) < 1e-6, f"expected 1.0 above threshold, got {reward[0].item()}"

    def test_at_threshold_gives_zero(self):
        """Height exactly at minimal_height should give 0.0 (strict > comparison)."""
        cube_height = torch.tensor([MINIMAL_HEIGHT])
        reward = lifting_object_reward(cube_height, MINIMAL_HEIGHT)
        assert abs(reward[0].item() - 0.0) < 1e-6, f"expected 0.0 at threshold (>= not used), got {reward[0].item()}"

    def test_batch_with_mixed_heights(self):
        """Batch with some above and some below threshold should produce
        corresponding 0.0 and 1.0 rewards per-element."""
        cube_height = torch.tensor([
            MINIMAL_HEIGHT - 0.01,  # below -> 0.0
            MINIMAL_HEIGHT + 0.01,  # above -> 1.0
            MINIMAL_HEIGHT + 0.05,  # above -> 1.0
            MINIMAL_HEIGHT - 0.02,  # below -> 0.0
        ])
        reward = lifting_object_reward(cube_height, MINIMAL_HEIGHT)

        assert reward.shape == (4,), f"expected shape (4,), got {reward.shape}"
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(reward, expected), f"expected {expected.tolist()}, got {reward.tolist()}"


class TestObjectGoalDistanceReward:
    """Tests for object_goal_distance_reward (gated cube-to-goal proximity)."""

    def test_not_lifted_always_zero(self):
        """When cube_height <= minimal_height, reward should be 0.0
        regardless of cube-to-goal distance."""
        cube_pos = torch.tensor([[0.0, 0.0, 0.0]])
        goal_pos = torch.tensor([[1.0, 0.0, 0.0]])  # 1.0 away
        cube_height = torch.tensor([MINIMAL_HEIGHT - 0.01])  # not lifted

        reward = object_goal_distance_reward(
            cube_pos, goal_pos, cube_height, MINIMAL_HEIGHT, GOAL_STD
        )

        assert abs(reward[0].item() - 0.0) < 1e-6, f"expected 0.0 when not lifted, got {reward[0].item()}"

    def test_lifted_follows_tanh_kernel(self):
        """When cube_height > minimal_height, reward should follow
        1 - tanh(distance / std) kernel."""
        cube_pos = torch.tensor([[0.0, 0.0, 0.0]])
        goal_pos = torch.tensor([[1.0, 0.0, 0.0]])
        cube_height = torch.tensor([MINIMAL_HEIGHT + 0.01])  # lifted

        distance = 1.0
        expected = 1.0 - torch.tanh(torch.tensor(distance / GOAL_STD))

        reward = object_goal_distance_reward(
            cube_pos, goal_pos, cube_height, MINIMAL_HEIGHT, GOAL_STD
        )

        assert abs(reward[0].item() - expected.item()) < 1e-6, (
            f"when lifted, expected {expected.item()}, got {reward[0].item()}"
        )

    def test_batch_mixed_lifted_and_not_lifted(self):
        """Batch with both lifted and not-lifted rows should gate per-element:
        not-lifted rows get 0.0, lifted rows get tanh-kernel reward."""
        cube_pos = torch.tensor([
            [0.0, 0.0, 0.0],  # not lifted (row 0)
            [0.0, 0.0, 0.0],  # lifted (row 1)
            [0.0, 0.0, 0.0],  # not lifted (row 2)
            [0.0, 0.0, 0.0],  # lifted (row 3)
        ])
        goal_pos = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        cube_height = torch.tensor([
            MINIMAL_HEIGHT - 0.01,  # not lifted
            MINIMAL_HEIGHT + 0.01,  # lifted
            MINIMAL_HEIGHT - 0.02,  # not lifted
            MINIMAL_HEIGHT + 0.05,  # lifted
        ])

        reward = object_goal_distance_reward(
            cube_pos, goal_pos, cube_height, MINIMAL_HEIGHT, GOAL_STD
        )

        # Rows 0 and 2 (not lifted) should be 0.0
        assert abs(reward[0].item() - 0.0) < 1e-6, f"row 0 not lifted, expected 0.0, got {reward[0].item()}"
        assert abs(reward[2].item() - 0.0) < 1e-6, f"row 2 not lifted, expected 0.0, got {reward[2].item()}"

        # Rows 1 and 3 (lifted) should follow tanh kernel
        distance = 1.0
        expected_lifted = 1.0 - torch.tanh(torch.tensor(distance / GOAL_STD)).item()
        assert abs(reward[1].item() - expected_lifted) < 1e-6, (
            f"row 1 lifted, expected {expected_lifted}, got {reward[1].item()}"
        )
        assert abs(reward[3].item() - expected_lifted) < 1e-6, (
            f"row 3 lifted, expected {expected_lifted}, got {reward[3].item()}"
        )

    def test_gating_is_per_row_not_global(self):
        """The lifting gate should apply independently per batch element,
        not globally across the batch."""
        # Two environments: one lifted, one not
        cube_pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        goal_pos = torch.tensor([
            [0.05, 0.0, 0.0],  # distance 0.05, should give meaningful reward
            [0.05, 0.0, 0.0],  # distance 0.05
        ])
        cube_height = torch.tensor([
            MINIMAL_HEIGHT + 0.01,  # lifted
            MINIMAL_HEIGHT - 0.01,  # not lifted
        ])

        reward = object_goal_distance_reward(
            cube_pos, goal_pos, cube_height, MINIMAL_HEIGHT, GOAL_STD
        )

        # First (lifted) should be nonzero, second (not lifted) should be zero
        assert reward[0].item() > 0.4, f"lifted row should be nonzero, got {reward[0].item()}"
        assert abs(reward[1].item() - 0.0) < 1e-6, f"not lifted row should be 0.0, got {reward[1].item()}"

    def test_zero_distance_when_lifted(self):
        """When cube is at goal position and lifted, reward should be ~1.0."""
        cube_pos = torch.tensor([[1.0, 2.0, 3.0]])
        goal_pos = torch.tensor([[1.0, 2.0, 3.0]])  # same position
        cube_height = torch.tensor([MINIMAL_HEIGHT + 0.01])  # lifted

        reward = object_goal_distance_reward(
            cube_pos, goal_pos, cube_height, MINIMAL_HEIGHT, GOAL_STD
        )

        # distance = 0, so 1 - tanh(0) = 1.0
        assert abs(reward[0].item() - 1.0) < 1e-6, f"expected ~1.0 at goal, got {reward[0].item()}"

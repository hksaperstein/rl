"""Sim-independent unit tests for tasks/franka/shape_observations.py's pure
shape-class one-hot / geometry-descriptor math - no Isaac Lab import
needed (mirrors tests/test_franka_lift_reward.py's scope: mdp.py itself
imports isaaclab at module level per its own docstring - "Import this
module only after an Isaac Sim/Isaac Lab AppLauncher has been created" -
so mdp.py's object_shape_class_onehot/object_geometry_descriptor thin
wrappers are exercised only by actually running the env, not here; this
file tests the pure functions those wrappers delegate to). Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_mdp_shape_observations.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory)."""

import math

import pytest
import torch

from tasks.franka.shape_observations import (
    GEOMETRY_DESCRIPTOR_K,
    SHAPE_CLASSES,
    SHAPE_GEOMETRY_DESCRIPTORS,
    geometry_descriptor_broadcast,
    geometry_descriptor_per_env,
    shape_class_onehot,
    shape_class_onehot_per_env,
)


class TestShapeClassOnehot:
    @pytest.mark.parametrize("shape_class", SHAPE_CLASSES)
    def test_correct_onehot_index_for_each_shape(self, shape_class):
        num_envs = 5
        onehot = shape_class_onehot(shape_class, num_envs)

        assert onehot.shape == (num_envs, 4)
        expected_index = SHAPE_CLASSES.index(shape_class)
        for row in range(num_envs):
            assert onehot[row, expected_index].item() == 1.0
            # every other column is exactly 0
            for col in range(4):
                if col != expected_index:
                    assert onehot[row, col].item() == 0.0
            assert onehot[row].sum().item() == 1.0

    def test_all_rows_identical_broadcast_constant(self):
        """Per-env-cfg constant, NOT per-environment-varying - every row of
        the batch must be byte-identical."""
        onehot = shape_class_onehot("d20", 8)
        for row in range(1, 8):
            assert torch.equal(onehot[0], onehot[row])

    def test_num_envs_one(self):
        onehot = shape_class_onehot("d8", 1)
        assert onehot.shape == (1, 4)

    def test_unknown_shape_class_raises(self):
        with pytest.raises(ValueError, match="unknown shape_class"):
            shape_class_onehot("d6", 4)


class TestGeometryDescriptorBroadcast:
    @pytest.mark.parametrize("shape_class", SHAPE_CLASSES)
    def test_shape_and_finiteness(self, shape_class):
        num_envs = 6
        descriptor = geometry_descriptor_broadcast(shape_class, num_envs)

        assert descriptor.shape == (num_envs, GEOMETRY_DESCRIPTOR_K)
        assert torch.isfinite(descriptor).all()

    def test_all_rows_identical_broadcast_constant(self):
        descriptor = geometry_descriptor_broadcast("d12", 7)
        for row in range(1, 7):
            assert torch.equal(descriptor[0], descriptor[row])

    def test_matches_registered_constant(self):
        for shape_class in SHAPE_CLASSES:
            descriptor = geometry_descriptor_broadcast(shape_class, 3)
            expected = SHAPE_GEOMETRY_DESCRIPTORS[shape_class]
            assert math.isclose(descriptor[0, 0].item(), expected, rel_tol=1e-5)

    def test_d8_and_d20_measurably_different(self):
        """The brief's own example: a shape with fewer faces (d8, more
        angular) and d20 (more faces, more sphere-like) should produce
        measurably different descriptor values - not degenerate to a
        constant regardless of shape."""
        d8_val = geometry_descriptor_broadcast("d8", 1)[0, 0].item()
        d20_val = geometry_descriptor_broadcast("d20", 1)[0, 0].item()
        assert abs(d8_val - d20_val) > 0.01, f"d8={d8_val}, d20={d20_val} not measurably different"

    def test_sphericity_bounded_by_one(self):
        """Wadell sphericity of any real convex body is <= 1.0 (== 1.0 only
        for a perfect sphere) and > 0."""
        for shape_class in SHAPE_CLASSES:
            value = SHAPE_GEOMETRY_DESCRIPTORS[shape_class]
            assert 0.0 < value <= 1.0, f"{shape_class}: sphericity {value} out of (0, 1] range"

    def test_unknown_shape_class_raises(self):
        with pytest.raises(ValueError, match="unknown shape_class"):
            geometry_descriptor_broadcast("d6", 4)


class TestShapeClassOnehotPerEnv:
    """Task 5 (BACKLOG.md's 2026-07-19 controller decision "(b) single
    mixed-population env"): tasks/franka/mdp.py's object_shape_class_onehot
    delegates to shape_class_onehot_per_env whenever the env cfg sets
    `die_shape_classes_per_env` (FrankaDieLiftJointD12D20MixedEnvCfg only -
    every other env cfg keeps the single-shape broadcast path above,
    unaffected). mdp.py's thin wrapper itself is not exercised here (it
    imports isaaclab at module level, same scope split this file's own
    docstring already documents for the existing single-shape wrappers) -
    this tests the pure function it delegates to, using
    FrankaDieLiftJointD12D20MixedEnvCfg's own real assets order ("d12",
    "d20") to confirm the exact per-env indexing this task's env cfg
    actually produces."""

    D12_D20_ORDER = ("d12", "d20")

    def test_env_0_gets_first_shape_env_1_gets_second_shape(self):
        """Directly the task brief's own example: with assets_cfg=[d12_cfg,
        d20_cfg] (this repo's real FrankaDieLiftJointD12D20MixedEnvCfg
        order), env 0 -> d12's onehot, env 1 -> d20's onehot."""
        onehot = shape_class_onehot_per_env(self.D12_D20_ORDER, num_envs=6)
        assert torch.equal(onehot[0], shape_class_onehot("d12", 1)[0])
        assert torch.equal(onehot[1], shape_class_onehot("d20", 1)[0])

    def test_round_robin_matches_index_mod_len(self):
        """Mirrors MultiAssetSpawnerCfg(random_choice=False)'s own live
        `proto_prim_paths[index % len(proto_prim_paths)]` formula
        (isaaclab/sim/spawners/wrappers/wrappers.py::spawn_multi_asset,
        confirmed by direct source read for this task) exactly."""
        num_envs = 9
        onehot = shape_class_onehot_per_env(self.D12_D20_ORDER, num_envs=num_envs)
        for env_idx in range(num_envs):
            expected_shape = self.D12_D20_ORDER[env_idx % len(self.D12_D20_ORDER)]
            assert torch.equal(onehot[env_idx], shape_class_onehot(expected_shape, 1)[0]), f"env {env_idx}"

    def test_shape_is_num_envs_by_4(self):
        onehot = shape_class_onehot_per_env(self.D12_D20_ORDER, num_envs=7)
        assert onehot.shape == (7, 4)

    def test_three_way_split(self):
        """Not just the 2-shape case - a 3-way assets order round-robins
        correctly too (generality check, not tied to this task's own
        2-shape env cfg)."""
        order = ("d8", "d10", "d12")
        onehot = shape_class_onehot_per_env(order, num_envs=7)
        expected_shapes = ["d8", "d10", "d12", "d8", "d10", "d12", "d8"]
        for env_idx, expected_shape in enumerate(expected_shapes):
            assert torch.equal(onehot[env_idx], shape_class_onehot(expected_shape, 1)[0]), f"env {env_idx}"

    def test_empty_shape_classes_per_env_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            shape_class_onehot_per_env((), num_envs=4)

    def test_unknown_shape_class_raises(self):
        with pytest.raises(ValueError, match="unknown shape_class"):
            shape_class_onehot_per_env(("d12", "d6"), num_envs=4)


class TestGeometryDescriptorPerEnv:
    D12_D20_ORDER = ("d12", "d20")

    def test_env_0_gets_first_shape_env_1_gets_second_shape(self):
        descriptor = geometry_descriptor_per_env(self.D12_D20_ORDER, num_envs=6)
        assert math.isclose(descriptor[0, 0].item(), SHAPE_GEOMETRY_DESCRIPTORS["d12"], rel_tol=1e-5)
        assert math.isclose(descriptor[1, 0].item(), SHAPE_GEOMETRY_DESCRIPTORS["d20"], rel_tol=1e-5)

    def test_round_robin_matches_index_mod_len(self):
        num_envs = 9
        descriptor = geometry_descriptor_per_env(self.D12_D20_ORDER, num_envs=num_envs)
        for env_idx in range(num_envs):
            expected_shape = self.D12_D20_ORDER[env_idx % len(self.D12_D20_ORDER)]
            assert math.isclose(descriptor[env_idx, 0].item(), SHAPE_GEOMETRY_DESCRIPTORS[expected_shape], rel_tol=1e-5)

    def test_shape_is_num_envs_by_k(self):
        descriptor = geometry_descriptor_per_env(self.D12_D20_ORDER, num_envs=7)
        assert descriptor.shape == (7, GEOMETRY_DESCRIPTOR_K)

    def test_empty_shape_classes_per_env_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            geometry_descriptor_per_env((), num_envs=4)

    def test_unknown_shape_class_raises(self):
        with pytest.raises(ValueError, match="unknown shape_class"):
            geometry_descriptor_per_env(("d12", "d6"), num_envs=4)

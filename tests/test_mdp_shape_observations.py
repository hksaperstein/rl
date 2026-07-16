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
    shape_class_onehot,
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

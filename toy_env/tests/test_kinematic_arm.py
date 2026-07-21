"""Sanity/regression tests for `toy_env.kinematic_arm`.

Run with: toy_env/.venv/bin/pytest toy_env/tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from toy_env import kinematic_arm as ka


def test_home_pose_is_finite_and_on_z_axis():
    theta = np.zeros(ka.N_JOINTS)
    fk = ka.forward_kinematics(theta)
    assert np.all(np.isfinite(fk.joint_positions))
    # All link offsets point along local z at theta=0 (no rotation applied
    # yet), so the fully-home pose is a degenerate straight-up singularity.
    assert np.allclose(fk.ee_pos[:2], 0.0, atol=1e-9)
    assert fk.ee_pos[2] == pytest.approx(ka.max_reach(), abs=1e-9)


def test_joint_positions_shape():
    theta = np.zeros(ka.N_JOINTS)
    fk = ka.forward_kinematics(theta)
    # base origin + N_JOINTS pivots + EE tip
    assert fk.joint_positions.shape == (ka.N_JOINTS + 2, 3)


def test_random_pose_within_max_reach():
    rng = np.random.default_rng(1)
    for _ in range(20):
        theta = rng.uniform(-ka.JOINT_LIMIT, ka.JOINT_LIMIT, size=ka.N_JOINTS)
        fk = ka.forward_kinematics(theta)
        assert np.linalg.norm(fk.ee_pos) <= ka.max_reach() + 1e-9


def test_jacobian_matches_finite_perturbation():
    rng = np.random.default_rng(2)
    theta = rng.uniform(-1.0, 1.0, size=ka.N_JOINTS)
    J = ka.jacobian_position(theta)
    assert J.shape == (3, ka.N_JOINTS)

    dq = rng.uniform(-0.01, 0.01, size=ka.N_JOINTS)
    ee0 = ka.forward_kinematics(theta).ee_pos
    ee1 = ka.forward_kinematics(theta + dq).ee_pos
    predicted = J @ dq
    actual = ee1 - ee0
    # First-order Taylor approximation should match closely for small dq.
    assert np.linalg.norm(predicted - actual) < 1e-4


def test_forward_kinematics_rejects_wrong_shape():
    with pytest.raises(AssertionError):
        ka.forward_kinematics(np.zeros(ka.N_JOINTS - 1))


def test_rot_matrix_orthonormal():
    for axis in ("z", "y"):
        for theta in (0.0, 0.3, -1.5, np.pi):
            R = ka._rot_matrix(axis, theta)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_unsupported_axis_raises():
    with pytest.raises(ValueError):
        ka._rot_matrix("x", 0.5)

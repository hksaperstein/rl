"""Finite-difference numerical verification of the axis-alignment IK
Jacobian added to scripts/grasp_demo_v2.py (2026-07-24, ar4-axis-align-ik
task) - REQUIRED to run and PASS before that Jacobian is trusted for any
real grasp attempt, per this task's own explicit instructions.

scripts/grasp_demo_v2.py is not importable as a module (its own top-level
argparse/AppLauncher construction would run - and conflict with this
script's own - at import time), so the small set of pure-math pieces this
check needs (`_skew_batch`, `_build_canonical_target_quat_w`/
`_build_canonical_target_quat_b`/`_build_canonical_axes_b`,
`_world_jacobian_to_root_frame`, `_ee_point_pos_and_jacobian`,
`_axis_align_error_and_jacobian`) are PORTED (copied) here verbatim rather
than imported - the same established pattern this project's other
`_verify_*`/`_record_*` scripts already use for the same reason (see e.g.
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-24
ar4-locked-achieved-orientation-grasp UPDATE: "_pose_locked_step - ported
directly from grasp_demo_v2.py's own polish_from_seed"). If you change the
derivation in grasp_demo_v2.py, update the copy here too and re-run this
check.

Method: teleport the AR4 arm (write_joint_position_to_sim, zero velocity,
a short PD-hold at the SAME commanded value to let cached .data refresh
without letting the joint actually drift) to several different joint
configs, read the REAL live Jacobian from Isaac Sim at each, and compare:

  1. The analytic axis-alignment Jacobian (`axis_jac_2d`, 2 rows) against a
     central-difference numerical derivative of the SAME tracked quantity
     (n_cur's own projection onto the (u_b, v_b) plane) obtained by
     perturbing each joint by +-EPS and re-measuring.
  2. The SAME check for the existing (already-trusted, but re-verified here
     per this task's literal instruction) position Jacobian
     (`point_jac_pos`, 3 rows) - the EE-offset-corrected pinch-point
     position.

A genuinely correct Jacobian should match the finite-difference estimate to
within a small tolerance set by the linearization/settle-noise floor, not
by construction - if the derivation has a sign error, a wrong basis, or a
transposed row, this WILL show up as a large, unambiguous mismatch (not a
subtle one), since a wrong-signed or wrong-axis Jacobian produces columns
that point in an entirely different direction from the numerical estimate.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_axis_align_jacobian.py
"""

import math
import os
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    subtract_frame_transforms,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

# === Ported verbatim from scripts/grasp_demo_v2.py (2026-07-24) - see this
# module's own docstring for why this is a copy, not an import. Keep in
# sync with that file if the derivation there changes. ===

_CANONICAL_Z_AXIS_W = (0.0, 0.0, -1.0)
_CANONICAL_X_AXIS_W = (0.0, 1.0, 0.0)
_CANONICAL_Y_AXIS_W = (1.0, 0.0, 0.0)
_EE_OFFSET = (0.0, 0.0, 0.036)


def _build_canonical_target_quat_w(device: str, tilt_deg: float = 0.0) -> torch.Tensor:
    theta = math.radians(tilt_deg)
    x_axis = torch.tensor(_CANONICAL_X_AXIS_W, device=device)
    base_y_axis = torch.tensor(_CANONICAL_Y_AXIS_W, device=device)
    base_z_axis = torch.tensor(_CANONICAL_Z_AXIS_W, device=device)
    z_axis = math.cos(theta) * base_z_axis + math.sin(theta) * base_y_axis
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    rot_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1).unsqueeze(0)
    return quat_from_matrix(rot_matrix)


def _build_canonical_target_quat_b(root_pos_w, root_quat_w, tilt_deg: float = 0.0) -> torch.Tensor:
    target_quat_w = _build_canonical_target_quat_w(str(root_pos_w.device), tilt_deg=tilt_deg)
    _, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, root_pos_w, target_quat_w)
    return target_quat_b


def _build_canonical_axes_b(root_pos_w, root_quat_w, tilt_deg: float = 0.0):
    target_quat_b = _build_canonical_target_quat_b(root_pos_w, root_quat_w, tilt_deg=tilt_deg)
    rot = matrix_from_quat(target_quat_b)
    u_b = rot[:, :, 0].clone()
    v_b = rot[:, :, 1].clone()
    n_des_b = rot[:, :, 2].clone()
    return n_des_b, u_b, v_b


def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    skew = torch.zeros(v.shape[0], 3, 3, device=v.device)
    skew[:, 0, 1], skew[:, 0, 2] = -v[:, 2], v[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = v[:, 2], -v[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -v[:, 1], v[:, 0]
    return skew


def _world_jacobian_to_root_frame(jacobian_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
    if jacobian_b.shape[1] > 3:
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])
    return jacobian_b


def _ee_point_pos_and_jacobian(ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor, jacobian_b: torch.Tensor):
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset

    skew = torch.zeros(ee_pos_b.shape[0], 3, 3, device=ee_pos_b.device)
    skew[:, 0, 1], skew[:, 0, 2] = -world_offset[:, 2], world_offset[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = world_offset[:, 2], -world_offset[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -world_offset[:, 1], world_offset[:, 0]
    jac_ang = jacobian_b[:, 3:6, :]
    point_jac_pos = jacobian_b[:, 0:3, :] - torch.bmm(skew, jac_ang)
    return point_pos_b, point_jac_pos


def _axis_align_error_and_jacobian(ee_quat_b, jac_ang, n_des_b, u_b, v_b):
    rot_cur = matrix_from_quat(ee_quat_b)
    n_cur_b = rot_cur[:, :, 2]

    cos_angle = torch.clamp((n_cur_b * n_des_b).sum(dim=-1), -1.0, 1.0)
    true_axis_angle = torch.acos(cos_angle)

    u_dot = (u_b * n_cur_b).sum(dim=-1, keepdim=True)
    v_dot = (v_b * n_cur_b).sum(dim=-1, keepdim=True)
    axis_error_2d = -torch.cat([u_dot, v_dot], dim=-1)

    d_ncur_dq = -torch.bmm(_skew_batch(n_cur_b), jac_ang)
    j_u = torch.bmm(u_b.unsqueeze(1), d_ncur_dq)
    j_v = torch.bmm(v_b.unsqueeze(1), d_ncur_dq)
    axis_jac_2d = torch.cat([j_u, j_v], dim=1)

    return axis_error_2d, axis_jac_2d, true_axis_angle


# === End ported section ===

EPS = 1e-4  # rad, central-difference perturbation
SETTLE_STEPS = 8  # physics steps to let cached .data refresh after a teleport, holding the SAME commanded target
TOL_ABS = 5e-3  # absolute mismatch tolerance (Jacobian units: dimensionless axis-projection or meters per rad)

# Test joint configs - arbitrary but within this arm's typical operating
# range (matching grasp_demo_v2.py's own CANDIDATE_SEEDS/KNOWN_GOOD_* scale,
# not hand-picked to be "nice"). A correct Jacobian derivation must hold at
# ANY joint config, not just a convenient one - deliberately includes
# HOME_Q (many quantities trivial/degenerate there - a weak test on its
# own) alongside two genuinely arbitrary non-trivial configs.
TEST_CONFIGS = {
    "HOME_Q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "MODERATE_A": [0.3, 0.9, -0.5, 0.4, 0.8, -0.3],
    "MODERATE_B": [-0.6, 1.1, 0.6, -0.4, -0.9, 0.5],
    "TILTED_GRASP_LIKE": [1.05, -0.55, -0.75, 0.2, -0.65, 0.15],
}


def _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_list, steps):
    q = torch.tensor([q_list], device=env.device)
    robot.write_joint_position_to_sim(q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(steps):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = q
        action[:, num_arm_joints] = 1.0  # gripper open, irrelevant here
        env.step(action)


def _measure_quantities(env, robot, robot_entity_cfg, ik_jacobi_idx, n_des_b, u_b, v_b):
    """Returns (point_pos_b (3,), axis_quantity_2d (2,) = [u.n_cur, v.n_cur],
    the LIVE analytic point_jac_pos (3,num_joints), axis_jac_2d (2,num_joints))
    at whatever joint config the robot is CURRENTLY settled at."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
    jacobian_b = _world_jacobian_to_root_frame(jacobian_w, robot.data.root_quat_w)
    point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)
    jac_ang = jacobian_b[:, 3:6, :]
    axis_error_2d, axis_jac_2d, true_axis_angle = _axis_align_error_and_jacobian(ee_quat_b, jac_ang, n_des_b, u_b, v_b)
    axis_quantity_2d = -axis_error_2d  # = [u.n_cur, v.n_cur], the tracked quantity axis_jac_2d is the Jacobian OF
    return point_pos_b[0].clone(), axis_quantity_2d[0].clone(), point_jac_pos[0].clone(), axis_jac_2d[0].clone(), true_axis_angle[0].item()


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    with torch.inference_mode():
        env.reset()
        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()

        # Safety clamp: TEST_CONFIGS above were hand-picked to plausibly fit
        # this arm's typical operating range, not read from the real joint
        # limits - clamp (with margin for the +-EPS perturbation) against
        # the LIVE limits here rather than trusting that guess blindly.
        joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
        lo = joint_pos_limits[0, :, 0].tolist()
        hi = joint_pos_limits[0, :, 1].tolist()
        print(f"[INFO] Live joint pos limits: lo={['%.4f' % v for v in lo]} hi={['%.4f' % v for v in hi]}")
        margin = 10 * EPS
        for name, q0 in TEST_CONFIGS.items():
            clamped = [min(max(v, l + margin), h - margin) for v, l, h in zip(q0, lo, hi)]
            if clamped != q0:
                print(f"[INFO] Clamped {name}: {q0} -> {clamped}")
            TEST_CONFIGS[name] = clamped

        # Verify at BOTH the pure-vertical canonical target (tilt_deg=0) and
        # a genuinely tilted one (65deg, this investigation's own
        # best-known real-attempt tilt - kb/wiki/concepts/ar4-vs-franka-
        # root-cause-comparison.md's 2026-07-24 UPDATEs) - a Jacobian
        # derivation that only happens to work for the untilted axis-
        # aligned-with-world-Z special case would be a much weaker check.
        overall_pass = True
        for tilt_deg in (0.0, 65.0):
            n_des_b, u_b, v_b = _build_canonical_axes_b(root_pos_w, root_quat_w, tilt_deg=tilt_deg)
            print(f"\n{'='*100}\n[TILT={tilt_deg} deg] n_des_b={n_des_b[0].tolist()} u_b={u_b[0].tolist()} v_b={v_b[0].tolist()}\n{'='*100}")

            for config_name, q0 in TEST_CONFIGS.items():
                _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q0, SETTLE_STEPS)
                pos0, axis0, analytic_pos_jac, analytic_axis_jac, axis_angle0 = _measure_quantities(
                    env, robot, robot_entity_cfg, ik_jacobi_idx, n_des_b, u_b, v_b
                )

                num_pos_jac = torch.zeros(3, num_arm_joints)
                num_axis_jac = torch.zeros(2, num_arm_joints)
                for j in range(num_arm_joints):
                    q_plus = list(q0)
                    q_plus[j] += EPS
                    _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_plus, SETTLE_STEPS)
                    pos_plus, axis_plus, _, _, _ = _measure_quantities(env, robot, robot_entity_cfg, ik_jacobi_idx, n_des_b, u_b, v_b)

                    q_minus = list(q0)
                    q_minus[j] -= EPS
                    _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_minus, SETTLE_STEPS)
                    pos_minus, axis_minus, _, _, _ = _measure_quantities(env, robot, robot_entity_cfg, ik_jacobi_idx, n_des_b, u_b, v_b)

                    num_pos_jac[:, j] = (pos_plus - pos_minus) / (2 * EPS)
                    num_axis_jac[:, j] = (axis_plus - axis_minus) / (2 * EPS)

                pos_err = (analytic_pos_jac.cpu() - num_pos_jac).abs()
                axis_err = (analytic_axis_jac.cpu() - num_axis_jac).abs()
                pos_max_err = pos_err.max().item()
                axis_max_err = axis_err.max().item()
                config_pass = pos_max_err < TOL_ABS and axis_max_err < TOL_ABS
                overall_pass = overall_pass and config_pass

                print(f"\n[{config_name}] q0={q0} axis_angle_to_target={math.degrees(axis_angle0):.2f}deg")
                print(f"  POSITION Jacobian (3x{num_arm_joints}): max|analytic-numeric| = {pos_max_err:.6f}  {'PASS' if pos_max_err < TOL_ABS else 'FAIL'}")
                print(f"    analytic:\n{analytic_pos_jac.cpu().numpy()}")
                print(f"    numeric (FD):\n{num_pos_jac.numpy()}")
                print(f"  AXIS-ALIGNMENT Jacobian (2x{num_arm_joints}): max|analytic-numeric| = {axis_max_err:.6f}  {'PASS' if axis_max_err < TOL_ABS else 'FAIL'}")
                print(f"    analytic:\n{analytic_axis_jac.cpu().numpy()}")
                print(f"    numeric (FD):\n{num_axis_jac.numpy()}")

        print(f"\n{'='*100}")
        print(f"[OVERALL] {'ALL CONFIGS/TILTS PASSED' if overall_pass else 'AT LEAST ONE CONFIG/TILT FAILED'} (tolerance={TOL_ABS})")
        print(f"{'='*100}")

    env.close()
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()

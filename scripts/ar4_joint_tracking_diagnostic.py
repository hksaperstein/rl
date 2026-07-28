"""Direct commanded-vs-achieved joint-tracking measurement at the FK-computed
graspable-workspace grasp pose (2026-07-27, ar4-joint-tracking-diagnostic task).

Motivation: this project's AR4 grasp investigation
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md) has repeatedly found
a kinematically-perfect, reachable, roll-constrained grasp pose (the
"ar4-graspable-roll-constraint task" UPDATE) still produces a real 45-61N
open-gripper collision with the cube in live physics, orientation-independent
across 3 distinct validation points. Every hypothesis tested so far (position,
orientation, workspace reachability, roll/heading) has assumed the physics arm
actually REACHES the FK-computed joint config it's commanded to. This script
tests that assumption directly and in isolation: command the arm DIRECTLY to
P0's exact FK-computed GRASP_Q (scripts/ar4_graspable_workspace_confirm_numeric.py's
own VALIDATION_POINTS["P0_recommended_bearing94"], reused verbatim, not
re-derived), gripper held OPEN throughout, cube parked far away (no obstruction
possible - isolates pure joint-tracking from any contact-resistance confound),
and reads back robot.data.joint_pos / robot.data.body_pos_w once the PD
controller has genuinely settled (a long dedicated EXTRA-SETTLE phase with
periodic error printouts to directly observe convergence, not just a fixed
step count assumed sufficient).

Two actuator-gain regimes tested in ONE Isaac Sim launch (write_joint_stiffness_to_sim
/write_joint_damping_to_sim called at runtime before each trial - same runtime
API already used by scripts/classical_grasp_contact_check.py and
scripts/interactive_joint_demo.py, avoids paying two separate cloud
provisioning costs):
  1. DEFAULT (tasks/ar4/robot_cfg.py's AR4_MK5_CFG "arm" actuator as currently
     shipped: stiffness=40, damping=4) - already documented
     (kb article's 2026-07-22 "later, same day" UPDATE) as producing a real
     1.42rad tracking error on a full multi-joint move; this script re-measures
     it directly at THIS specific pose/task's own commanded config, not just
     citing the old number.
  2. BOOSTED (stiffness=4000, damping=200) - the SAME test-local override
     already used by every confirm_numeric.py/grasp_demo_v2.py run in this
     investigation (never committed to the shared robot_cfg.py). Re-measures
     whether this override is sufficient to reach P0's specific GRASP_Q
     cleanly with NOTHING in the way, to isolate whether the 7.3deg/11deg
     plateau seen in the roll-constraint task's live confirmation (WITH the
     cube present, WITH this same boosted gain already active) was caused by
     genuine cube contact resistance, or by an inherent inability of even the
     boosted gain to reach this specific pose.

Also directly compares the achieved gripper pinch point (link_6 body pose +
the established _EE_OFFSET=(0,0,0.036) convention, scripts/grasp_demo_v2.py's
own pinch-point definition) and gripper_base_link body pose against the
FK-computed (pure kinematics, tasks/ar4/fk_verification.py) prediction for the
SAME commanded joint values - directly answers "is the physical gripper where
the kinematics said it would be."

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_joint_tracking_diagnostic.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Commanded-vs-achieved joint tracking measurement at the FK-computed P0 grasp pose, no cube obstruction.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402
from tasks.ar4.fk_verification import compute_link_pose_from_joint_values  # noqa: E402

# ----------------------------------------------------------------------
# P0's exact FK-computed configuration - copied verbatim from
# scripts/ar4_graspable_workspace_confirm_numeric.py's
# VALIDATION_POINTS["P0_recommended_bearing94"] (do not re-derive; this task
# is about the physics of REACHING this config, not about recomputing it).
# ----------------------------------------------------------------------
CUBE_XY_PARK_FAR_AWAY = (3.0, 3.0)  # nowhere near the arm's workspace (radius ~0.28-0.52m) - guarantees zero contact
P0_GRASP_Q_DEG = [-4.32433486449247, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994]
P0_PREGRASP_Q_DEG = [-4.32433486449247, 52.434178877076384, 27.87529831816333, -2.2493188573178595, 8.860705727687604, 100.52702739544871]
HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GRASP_Q = [math.radians(d) for d in P0_GRASP_Q_DEG]
PREGRASP_Q = [math.radians(d) for d in P0_PREGRASP_Q_DEG]
HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}

# Matches scripts/grasp_demo_v2.py's _EE_OFFSET / scripts/ar4_graspable_workspace.py's
# _EE_OFFSET_LOCAL exactly (link_6 -> gripper jaw pinch point, along link_6's
# own local +Z axis).
EE_OFFSET_LOCAL = np.array([0.0, 0.0, 0.036])

# World-frame base transform (tasks/ar4/robot_cfg.py's AR4_MK5_CFG.init_state:
# rot=(0,0,0,1) wxyz = a 180-degree rotation about Z, pos defaults to origin) -
# copied verbatim from scripts/ar4_graspable_workspace.py's base_to_world.
# Valid for both positions (no translation component) and free vectors.
def base_to_world(p_base: np.ndarray) -> np.ndarray:
    out = np.array(p_base, dtype=float).copy()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _quat_wxyz_to_matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fk_predicted_world(joint_values_rad: dict, link_name: str) -> np.ndarray:
    """Pure-kinematics (no physics) world-frame position of `link_name` at the
    given COMMANDED joint values, via tasks/ar4/fk_verification.py's already-
    verified vendor-URDF FK, transformed base->world exactly as
    scripts/ar4_graspable_workspace.py already does."""
    pos_b, _quat_b = compute_link_pose_from_joint_values(joint_values_rad, link_name)
    return base_to_world(pos_b)


def fk_predicted_pinch_point_world(joint_values_rad: dict) -> np.ndarray:
    pos_b, quat_b = compute_link_pose_from_joint_values(joint_values_rad, "link_6")
    rot_b = _quat_wxyz_to_matrix(quat_b)
    pinch_b = pos_b + rot_b @ EE_OFFSET_LOCAL
    return base_to_world(pinch_b)


def _drive(env, robot, arm_cfg, gripper_cfg, arm_target, gripper_target_expr, duration, label, print_every=20):
    """Direct joint-position-target driving, no render, no action-manager
    tensor - same low-level pattern already established by
    scripts/_verify_gripper_mirror_fix.py / ar4_graspable_workspace_confirm_numeric.py."""
    arm_target_t = torch.tensor([arm_target], device=env.device)
    gripper_target_t = torch.tensor(
        [[gripper_target_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device
    )
    last_err = None
    for i in range(duration):
        robot.set_joint_position_target(arm_target_t, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        if i % print_every == 0 or i == duration - 1:
            live_arm_q = robot.data.joint_pos[0, arm_cfg.joint_ids].tolist()
            err_deg = [math.degrees(a - b) for a, b in zip(live_arm_q, arm_target)]
            max_err_deg = max(abs(e) for e in err_deg)
            delta = "" if last_err is None else f" (delta from last print: {max_err_deg - last_err:+.4f}deg)"
            print(f"  [{label} step {i:3d}] max_joint_err={max_err_deg:.4f}deg per_joint_err_deg={['%.3f' % e for e in err_deg]}{delta}")
            last_err = max_err_deg
    return last_err


def _run_regime(env, robot, arm_cfg, gripper_cfg, jaw_body_ids, link6_body_id, gripper_base_body_id, cube, regime_label, stiffness, damping):
    print("\n" + "#" * 70)
    print(f"# REGIME: {regime_label} (stiffness={stiffness}, damping={damping})")
    print("#" * 70)

    env.reset()

    # Park the cube far away - guarantees zero possibility of contact
    # confounding this pure joint-tracking measurement (task's own explicit
    # instruction: "NO cube present (or cube parked far away) so nothing
    # obstructs the arm from freely reaching the pose").
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[CUBE_XY_PARK_FAR_AWAY[0], CUBE_XY_PARK_FAR_AWAY[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(
        torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
    )
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

    n_arm = len(arm_cfg.joint_ids)
    stiff_t = torch.full((1, n_arm), float(stiffness), device=env.device)
    damp_t = torch.full((1, n_arm), float(damping), device=env.device)
    robot.write_joint_stiffness_to_sim(stiff_t, joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(damp_t, joint_ids=arm_cfg.joint_ids)

    _drive(env, robot, arm_cfg, gripper_cfg, HOME_Q, GRIPPER_OPEN_EXPR, 60, "HOME")
    _drive(env, robot, arm_cfg, gripper_cfg, PREGRASP_Q, GRIPPER_OPEN_EXPR, 150, "PREGRASP")
    _drive(env, robot, arm_cfg, gripper_cfg, GRASP_Q, GRIPPER_OPEN_EXPR, 90, "GRASP")
    print(f"\n[{regime_label}] EXTRA-SETTLE phase (200 steps at GRASP_Q) - watching for genuine convergence, not assuming 90 steps was enough:")
    _drive(env, robot, arm_cfg, gripper_cfg, GRASP_Q, GRIPPER_OPEN_EXPR, 200, "EXTRA-SETTLE", print_every=20)

    # ---- Final measurement ----
    final_arm_q = robot.data.joint_pos[0, arm_cfg.joint_ids].tolist()
    per_joint_err_deg = [math.degrees(a - b) for a, b in zip(final_arm_q, GRASP_Q)]
    max_err_deg = max(abs(e) for e in per_joint_err_deg)

    link6_pose_w = robot.data.body_pose_w[0, link6_body_id]
    link6_pos_w, link6_quat_w = link6_pose_w[0:3], link6_pose_w[3:7]
    rot_w = matrix_from_quat(link6_quat_w.unsqueeze(0))[0]
    offset_t = torch.tensor(EE_OFFSET_LOCAL.tolist(), device=link6_pos_w.device)
    achieved_pinch_w = (link6_pos_w + rot_w @ offset_t).tolist()

    gripper_base_pos_w = robot.data.body_pos_w[0, gripper_base_body_id].tolist()

    jaw_pos_w = robot.data.body_pos_w[0, jaw_body_ids]
    jaw1_pos_w, jaw2_pos_w = jaw_pos_w[0].tolist(), jaw_pos_w[1].tolist()
    true_bisector_w = [(a + b) / 2.0 for a, b in zip(jaw1_pos_w, jaw2_pos_w)]

    joint_values_commanded = {name: GRASP_Q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
    fk_pred_link6_w = fk_predicted_world(joint_values_commanded, "link_6").tolist()
    fk_pred_gripper_base_w = fk_predicted_world(joint_values_commanded, "gripper_base_link").tolist()
    fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

    def _dist_mm(a, b):
        return math.dist(a, b) * 1000.0

    print(f"\n=== {regime_label} FINAL RESULT ===")
    print(f"Commanded GRASP_Q (deg): {P0_GRASP_Q_DEG}")
    print(f"Achieved  arm q   (deg): {[math.degrees(v) for v in final_arm_q]}")
    print(f"Per-joint error   (deg): {[round(e, 4) for e in per_joint_err_deg]}")
    print(f"MAX per-joint tracking error: {max_err_deg:.4f} deg")
    print(f"Achieved link_6 world pos:         {link6_pos_w.tolist()}")
    print(f"FK-predicted link_6 world pos:     {fk_pred_link6_w}")
    print(f"  link_6 discrepancy: {_dist_mm(link6_pos_w.tolist(), fk_pred_link6_w):.3f} mm")
    print(f"Achieved gripper_base_link world pos: {gripper_base_pos_w}")
    print(f"FK-predicted gripper_base_link world pos: {fk_pred_gripper_base_w}")
    print(f"  gripper_base_link discrepancy: {_dist_mm(gripper_base_pos_w, fk_pred_gripper_base_w):.3f} mm")
    print(f"Achieved pinch point (link_6 + EE_OFFSET) world pos: {achieved_pinch_w}")
    print(f"FK-predicted pinch point world pos:                  {fk_pred_pinch_w}")
    print(f"  pinch point discrepancy: {_dist_mm(achieved_pinch_w, fk_pred_pinch_w):.3f} mm")
    print(f"Real jaw1/jaw2 world pos: jaw1={jaw1_pos_w} jaw2={jaw2_pos_w} true_bisector={true_bisector_w}")
    print(f"  true_bisector vs FK-predicted pinch point: {_dist_mm(true_bisector_w, fk_pred_pinch_w):.3f} mm")
    print("=" * 70)

    return {
        "max_err_deg": max_err_deg,
        "per_joint_err_deg": per_joint_err_deg,
        "link6_discrepancy_mm": _dist_mm(link6_pos_w.tolist(), fk_pred_link6_w),
        "gripper_base_discrepancy_mm": _dist_mm(gripper_base_pos_w, fk_pred_gripper_base_w),
        "pinch_discrepancy_mm": _dist_mm(achieved_pinch_w, fk_pred_pinch_w),
    }


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    # Deliberately do NOT override actuator gains in the cfg itself here -
    # this script writes stiffness/damping AT RUNTIME per regime (see
    # _run_regime), starting from whatever tasks/ar4/robot_cfg.py currently
    # ships (stiffness=40, damping=4) for the "DEFAULT" regime.

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    cube = env.scene["cube"]

    jaw_body_ids, jaw_body_names_found = robot.find_bodies(["gripper_jaw1_link", "gripper_jaw2_link"])
    print(f"[INFO] jaw body ids resolved: {jaw_body_names_found} -> {jaw_body_ids}")
    link6_body_ids, link6_names_found = robot.find_bodies(["link_6"])
    gripper_base_body_ids, gripper_base_names_found = robot.find_bodies(["gripper_base_link"])
    print(f"[INFO] link_6 body id: {link6_names_found} -> {link6_body_ids}; gripper_base_link body id: {gripper_base_names_found} -> {gripper_base_body_ids}")
    link6_body_id = link6_body_ids[0]
    gripper_base_body_id = gripper_base_body_ids[0]

    with torch.inference_mode():
        results = {}
        results["DEFAULT_stiffness40_damping4"] = _run_regime(
            env, robot, arm_cfg, gripper_cfg, jaw_body_ids, link6_body_id, gripper_base_body_id, cube,
            "DEFAULT_stiffness40_damping4", stiffness=40.0, damping=4.0,
        )
        results["BOOSTED_stiffness4000_damping200"] = _run_regime(
            env, robot, arm_cfg, gripper_cfg, jaw_body_ids, link6_body_id, gripper_base_body_id, cube,
            "BOOSTED_stiffness4000_damping200", stiffness=4000.0, damping=200.0,
        )

        print("\n" + "%" * 70)
        print("FINAL SUMMARY (both regimes)")
        print("%" * 70)
        for label, r in results.items():
            print(
                f"  {label}: max_joint_err={r['max_err_deg']:.4f}deg  "
                f"link6_discrepancy={r['link6_discrepancy_mm']:.3f}mm  "
                f"gripper_base_discrepancy={r['gripper_base_discrepancy_mm']:.3f}mm  "
                f"pinch_discrepancy={r['pinch_discrepancy_mm']:.3f}mm"
            )
        print("%" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

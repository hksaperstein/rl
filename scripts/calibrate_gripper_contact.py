"""Calibrate the AR4 gripper's ContactSensor-based grasp reward
(tasks/ar4/mdp.py's contact_grasp_bonus) against a real closed grasp,
before spending a full training run on an untested force_threshold.

Commands the arm to hold its all-zero-joint pose (its default reset target)
and teleports the sphere every step to the gripper's *live* jaw pinch-point
position (read fresh from the ee_frame sensor each iteration, not read once
and frozen) - avoiding any dependency on scripts/grasp_demo.py's IK reach
waypoints, which a real run showed do not reliably bring the end-effector
near the object in this scene. Teleport pattern precedented in
scripts/perception_calibration.py and scripts/measure_planarity_residual.py.

Two corrections found empirically while validating this script (see
docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md's
"Major finding" and "Calibration method correction" sections for the full
diagnosis):
1. Commanding an all-zero joint target does not hold the arm rigidly in
   place - under this robot's actuator gains, the arm visibly sags/settles
   for well over 100 steps after reset before its pose stabilizes. Reading
   the pinch point once (right after reset, before the arm has settled)
   and freezing it there leaves the sphere floating in stale, empty space
   once the real arm moves away - this script now re-reads the pinch point
   every single step instead, so the sphere always tracks wherever the
   gripper actually is, regardless of settling dynamics.
2. `_EE_OFFSET` in tasks/ar4/pickplace_env_cfg.py was itself wrong (0.09m,
   corrected to the measured 0.036m) - this is a separate, more significant
   finding: that offset feeds the same ee_frame sensor reaching_sphere's
   reward has used in every experiment this session, so this was likely a
   root contributor to grasping never emerging, independent of this
   contact-sensor experiment.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/calibrate_gripper_contact.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate the AR4 gripper ContactSensor against a real sphere grasp.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.mdp import contact_grasp_bonus  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# (duration_steps, gripper_command, label, teleport_sphere_to_pinch_point)
# The 150-step settle (arm pose stabilization) runs separately, before this
# loop, so the link_6->jaw-midpoint measurement below reads a settled pose.
# "far" is a real negative control: the sphere is left at its normal spawn
# position (untouched, nowhere near the gripper) so this script's own
# committed output includes a genuine no-contact baseline, not just a claim
# carried over from separate, uncommitted runs.
PHASES = [
    (30, GRIPPER_OPEN, "far", False),  # negative control: sphere untouched, far from the gripper
    (60, GRIPPER_OPEN, "open", True),
    (60, GRIPPER_CLOSE, "close", True),
    (120, GRIPPER_CLOSE, "hold", True),
]

# Must match tasks/ar4/pickplace_env_cfg.py's grasp_contact RewTerm params.
FORCE_THRESHOLD = 0.05


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    settle_steps = 150
    total_steps = settle_steps + sum(duration for duration, _, _, _ in PHASES)
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = total_steps * step_dt + 2.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_joints = len(ARM_JOINT_NAMES)
    jaw1_cfg = SceneEntityCfg("gripper_jaw1_contact")
    jaw2_cfg = SceneEntityCfg("gripper_jaw2_contact")

    far_forces: list[list[float]] = []
    hold_forces: list[list[float]] = []
    hold_rewards: list[float] = []

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    with torch.inference_mode():
        env.reset()
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]

        # Bake the _EE_OFFSET measurement itself into this script's committed,
        # re-runnable output, rather than leaving it as an uncommitted,
        # narrative-only claim. Compares the real link_6-to-jaw-midpoint
        # distance (measured directly from body positions) against the
        # ee_frame sensor's offset-based estimate, once the arm has settled.
        for _ in range(settle_steps):
            action = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
            action[:, num_joints] = GRIPPER_OPEN
            env.step(action)
        link6_idx = robot.data.body_names.index("link_6")
        jaw1_idx = robot.data.body_names.index("gripper_jaw1_link")
        jaw2_idx = robot.data.body_names.index("gripper_jaw2_link")
        jaw_midpoint = (robot.data.body_pos_w[0, jaw1_idx] + robot.data.body_pos_w[0, jaw2_idx]) / 2.0
        link6_pos = robot.data.body_pos_w[0, link6_idx]
        measured_offset = torch.norm(jaw_midpoint - link6_pos).item()
        estimated_offset = torch.norm(ee_frame.data.target_pos_w[0, 0] - link6_pos).item()
        print(
            f"[info] measured link_6->jaw-midpoint distance: {measured_offset:.4f} m "
            f"(ee_frame's _EE_OFFSET-based estimate: {estimated_offset:.4f} m)"
        )

        for duration, gripper_cmd, label, track_pinch_point in PHASES:
            for i in range(duration):
                if track_pinch_point:
                    # Re-read the gripper's real jaw pinch-point position
                    # every step (not once, frozen) - the arm's commanded
                    # pose takes many steps to settle under this robot's
                    # actuator gains, so a stale reading leaves the sphere in
                    # empty space once the real arm moves away from where it
                    # started.
                    pinch_point = ee_frame.data.target_pos_w[0, 0].clone()
                    sphere_pose = torch.cat([pinch_point, identity_quat]).unsqueeze(0)
                    env.scene["sphere"].write_root_pose_to_sim(sphere_pose)

                action = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
                action[:, num_joints] = gripper_cmd
                env.step(action)

                jaw1_sensor = env.scene["gripper_jaw1_contact"]
                jaw2_sensor = env.scene["gripper_jaw2_contact"]
                jaw1_norm = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(1)
                jaw2_norm = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(1)
                force_norm = [jaw1_norm.item(), jaw2_norm.item()]
                reward = contact_grasp_bonus(env, FORCE_THRESHOLD, jaw1_cfg, jaw2_cfg)

                if label == "hold":
                    hold_forces.append(force_norm)
                    hold_rewards.append(reward.item())
                elif label == "far":
                    far_forces.append(force_norm)

            print(f"[phase done] {label}: last force_norm={force_norm}, reward={reward.item()}")

    print("\n=== Calibration summary ===")
    far_min = min(min(f) for f in far_forces)
    far_max = max(max(f) for f in far_forces)
    hold_min = min(min(f) for f in hold_forces)
    hold_max = max(max(f) for f in hold_forces)
    hold_success = sum(r == 1.0 for r in hold_rewards)
    print(f"far (sphere untouched, nowhere near gripper) force_norm: min={far_min:.4f}, max={far_max:.4f} N (real negative control)")
    print(f"hold (gripper closed on sphere)               force_norm: min={hold_min:.4f}, max={hold_max:.4f} N")
    print(
        f"hold reward==1.0 fraction: {hold_success}/{len(hold_rewards)} "
        f"(force_threshold={FORCE_THRESHOLD})"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

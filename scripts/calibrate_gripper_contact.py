"""Calibrate the AR4 gripper's ContactSensor-based grasp reward
(tasks/ar4/mdp.py's contact_grasp_bonus) against a real closed grasp,
before spending a full training run on an untested force_threshold.

Holds the arm motionless at its default (all-zero-joint) pose and
teleports the sphere directly to the gripper's real jaw pinch-point
position (read live from the ee_frame sensor), then closes the gripper on
it - avoiding any dependency on scripts/grasp_demo.py's IK reach waypoints,
which a real run showed do not reliably bring the end-effector near the
object in this scene (see
docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md's
"Calibration method correction" section). Teleport pattern precedented in
scripts/perception_calibration.py and scripts/measure_planarity_residual.py.

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

# (duration_steps, gripper_command, label)
PHASES = [
    (60, GRIPPER_OPEN, "open"),
    (60, GRIPPER_CLOSE, "close"),
    (120, GRIPPER_CLOSE, "hold"),
]

# Must match tasks/ar4/pickplace_env_cfg.py's grasp_contact RewTerm params.
FORCE_THRESHOLD = 0.05


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    total_steps = sum(duration for duration, _, _ in PHASES)
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = total_steps * step_dt + 2.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_joints = len(ARM_JOINT_NAMES)
    jaw1_cfg = SceneEntityCfg("gripper_jaw1_contact")
    jaw2_cfg = SceneEntityCfg("gripper_jaw2_contact")

    open_forces: list[list[float]] = []
    hold_forces: list[list[float]] = []
    hold_rewards: list[float] = []

    with torch.inference_mode():
        env.reset()
        # Read the gripper's real jaw pinch-point position at its resting
        # (all-zero-joint) pose, straight from the same sensor frame
        # reaching_sphere's reward already trusts in real training.
        ee_frame = env.scene["ee_frame"]
        pinch_point = ee_frame.data.target_pos_w[0, 0].clone()
        print(f"[info] gripper pinch-point (world frame): {pinch_point.tolist()}")
        sphere_pose = torch.cat(
            [pinch_point, torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)]
        ).unsqueeze(0)

        for duration, gripper_cmd, label in PHASES:
            for _ in range(duration):
                # Hold the sphere at the pinch point every step - the arm
                # never moves, so this is the same physical target throughout.
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
                elif label == "open":
                    open_forces.append(force_norm)

            print(f"[phase done] {label}: last force_norm={force_norm}, reward={reward.item()}")

    print("\n=== Calibration summary ===")
    open_min = min(min(f) for f in open_forces)
    open_max = max(max(f) for f in open_forces)
    hold_min = min(min(f) for f in hold_forces)
    hold_max = max(max(f) for f in hold_forces)
    hold_success = sum(r == 1.0 for r in hold_rewards)
    print(f"open (gripper open, sphere at pinch point) force_norm: min={open_min:.4f}, max={open_max:.4f} N (expect ~0.0)")
    print(f"hold (gripper closed on sphere)             force_norm: min={hold_min:.4f}, max={hold_max:.4f} N")
    print(
        f"hold reward==1.0 fraction: {hold_success}/{len(hold_rewards)} "
        f"(force_threshold={FORCE_THRESHOLD})"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

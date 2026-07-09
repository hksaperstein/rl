# scripts/_check_arm_ground_collision_pose.py
"""One-off diagnostic: sweep joint_2/joint_3 (shoulder/elbow) across their
full limit range, teleporting the robot directly (bypassing RL actions),
and print the resulting world-frame z-height of every upper-arm body
tracked by the new arm_ground_contact sensor (base_link, link_1..link_5).
Purpose: find a joint configuration that geometrically drives one of
those links below/near the ground plane, to then drive a genuine
contact-sensor-triggered termination test (see
scripts/smoke_test_graspgoal_ground_penalty.py's Phase 2, which guessed
an action and got zero contact force over 200 steps - this script finds
the actual colliding configuration directly via FK instead of guessing).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_check_arm_ground_collision_pose.py
"""

import itertools
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        env.reset()
        robot = env.scene["robot"]

        upper_arm_bodies = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5"]
        body_idx = {name: robot.data.body_names.index(name) for name in upper_arm_bodies}
        joint_idx = {name: robot.data.joint_names.index(name) for name in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]}

        limits = robot.data.joint_pos_limits[0]  # (num_joints, 2)
        j2_lo, j2_hi = limits[joint_idx["joint_2"]].tolist()
        j3_lo, j3_hi = limits[joint_idx["joint_3"]].tolist()
        print(f"[INFO] joint_2 limits: [{j2_lo:.3f}, {j2_hi:.3f}], joint_3 limits: [{j3_lo:.3f}, {j3_hi:.3f}]")

        best = None
        default_joint_pos = robot.data.default_joint_pos[0].clone()

        for j2, j3 in itertools.product(
            torch.linspace(j2_lo, j2_hi, 7).tolist(), torch.linspace(j3_lo, j3_hi, 7).tolist()
        ):
            joint_pos = default_joint_pos.clone().unsqueeze(0)
            joint_pos[0, joint_idx["joint_2"]] = j2
            joint_pos[0, joint_idx["joint_3"]] = j3
            robot.write_joint_position_to_sim(joint_pos)
            robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
            # Advance a tiny amount so FK/body_pos_w reflects the teleport
            # (Isaac Lab body_pos_w is only updated on the next
            # sim/data-refresh, not instantly on write).
            env.sim.forward()

            min_z = min(robot.data.body_pos_w[0, body_idx[name], 2].item() for name in upper_arm_bodies)
            if best is None or min_z < best[0]:
                best = (min_z, j2, j3)

        print(f"[RESULT] lowest achieved upper-arm-body z over sweep: {best[0]:.4f} at joint_2={best[1]:.3f}, joint_3={best[2]:.3f}")

        # Apply the best (lowest) configuration found and report per-body z.
        joint_pos = default_joint_pos.clone().unsqueeze(0)
        joint_pos[0, joint_idx["joint_2"]] = best[1]
        joint_pos[0, joint_idx["joint_3"]] = best[2]
        robot.write_joint_position_to_sim(joint_pos)
        robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
        env.sim.forward()
        for name in upper_arm_bodies:
            z = robot.data.body_pos_w[0, body_idx[name], 2].item()
            print(f"[RESULT] {name} z-height: {z:.4f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

"""One-off check: at the home/reset position, compare the gripper's real
simulated (forward-kinematics) position - the midpoint of
gripper_jaw1_link/gripper_jaw2_link, read directly from
robot.data.body_pos_w - against the ee_frame FrameTransformer's reported
target position (link_6 + _EE_OFFSET). Same method
scripts/calibrate_gripper_contact.py and today's earlier EE-offset
verifications already used.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_check_ee_vs_gripper_fk.py
"""

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
        ee_frame = env.scene["ee_frame"]

        jaw1_idx = robot.data.body_names.index("gripper_jaw1_link")
        jaw2_idx = robot.data.body_names.index("gripper_jaw2_link")
        link6_idx = robot.data.body_names.index("link_6")

        link6_pos = robot.data.body_pos_w[0, link6_idx]
        jaw1_pos = robot.data.body_pos_w[0, jaw1_idx]
        jaw2_pos = robot.data.body_pos_w[0, jaw2_idx]
        gripper_fk_pos = (jaw1_pos + jaw2_pos) / 2.0

        ee_pos = ee_frame.data.target_pos_w[0, 0, :]

        print("=" * 70)
        print("HOME POSITION: gripper (FK) vs ee_frame target")
        print("=" * 70)
        print(f"link_6 (wrist) world pos:        {link6_pos.cpu().tolist()}")
        print(f"gripper_jaw1_link world pos:      {jaw1_pos.cpu().tolist()}")
        print(f"gripper_jaw2_link world pos:      {jaw2_pos.cpu().tolist()}")
        print(f"gripper FK midpoint (x,y,z):      {gripper_fk_pos.cpu().tolist()}")
        print(f"ee_frame target (x,y,z):          {ee_pos.cpu().tolist()}")
        delta = torch.norm(gripper_fk_pos - ee_pos).item()
        print(f"|gripper_fk - ee_frame| distance: {delta:.6f} m")
        delta_link6 = torch.norm(link6_pos - ee_pos).item()
        print(f"|link_6 - ee_frame| distance:     {delta_link6:.6f} m (should be ~0.036, the offset magnitude)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

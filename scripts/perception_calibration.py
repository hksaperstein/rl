# rl/scripts/perception_calibration.py
"""Sanity-check the perception pipeline before trusting it in eval/demo scripts:
slides the cube across the perception camera's field of view for a few seconds
and writes an mp4 with the detected mask/bbox/shape-label burned into each frame.

Not run during training or as part of any automated flow - a one-time (or
re-run-when-something-changes) manual check. The robot is present but held
motionless throughout (Isaac Lab's manager framework needs at least one
action/observation term, and the existing pick-and-place env config already
provides a well-tested one) - only the cube moves.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/perception_calibration.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a perception calibration clip.")
parser.add_argument("--duration", type=float, default=6.0, help="Clip duration in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from perception.overlay import draw_detections  # noqa: E402
from perception.pipeline import run_perception  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlacePerceptionEnvCfg  # noqa: E402

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "perception_calibration.mp4"
)
CUBE_Y = 0.31  # camera center
CUBE_Z = 0.009  # cube's resting half-height, see objects_cfg.py's CUBE_CFG
SLIDE_X_RANGE = (-0.35, 0.35)  # sweeps across the camera's field of view


def main() -> None:
    env_cfg = Ar4PickPlacePerceptionEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    camera = env.scene["perception_camera"]
    tracker = ObjectTracker()

    num_steps = int(args_cli.duration / env.step_dt)
    action_dim = env.action_manager.total_action_dim
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = imageio.get_writer(OUTPUT_PATH, fps=int(1.0 / env.step_dt), codec="libx264")

    with torch.inference_mode():
        env.reset()
        for step in range(num_steps):
            frac = step / max(num_steps - 1, 1)
            cube_x = SLIDE_X_RANGE[0] + frac * (SLIDE_X_RANGE[1] - SLIDE_X_RANGE[0])
            pose = torch.tensor([[cube_x, CUBE_Y, CUBE_Z, 1.0, 0.0, 0.0, 0.0]], device=env.device)
            env.scene["cube"].write_root_pose_to_sim(pose)

            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            actions[:, -1] = 1.0  # keep the gripper open
            env.step(actions)

            depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
            cam_pos = camera.data.pos_w[0].cpu().numpy()
            cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

            detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=GROUND_Z)
            tracked = tracker.update(detections)
            writer.append_data(draw_detections(rgb, tracked))

    writer.close()
    env.close()
    print(f"Calibration clip written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()

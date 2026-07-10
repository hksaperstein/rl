"""Render a still frame of the AR4 arm from the close-up demo camera to visually
confirm the per-part URDF colors now show (aluminum / dark motors / gray
enclosure / blue covers), instead of a flat white silhouette. No trained policy
needed: builds the demo-cam env, holds the home pose, and saves a PNG.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/render_color_check.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render an AR4 color-check still frame.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.graspgoal_democam_env_cfg import Ar4GraspGoalDemoEnvCfg  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "color_check")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    env_cfg = Ar4GraspGoalDemoEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)

    camera = env.scene["demo_camera"]
    with torch.inference_mode():
        env.reset()
        # Hold home pose and let the render settle (RTX path tracer + physics).
        for _ in range(40):
            actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)
        # rgb comes back bottom-row-first (opengl); flip vertically like
        # graspgoal_closeup_video.py does.
        rgb = camera.data.output["rgb"][:, ..., :3].cpu().numpy()[:, ::-1, :, :]
        frame = rgb[0].astype("uint8")
        out_path = os.path.join(OUT_DIR, "ar4_color_check.png")
        imageio.imwrite(out_path, frame)
        print(f"[RENDER] saved color-check frame: {out_path}  shape={frame.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

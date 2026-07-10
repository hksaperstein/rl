"""Diagnose the reported upside-down demo-camera render by computing BOTH the
directly-derived quaternion for a given eye/target AND its 180-degree-roll
counterpart (negate the camera's local x_axis and y_axis, keep z_axis/forward
unchanged), then rendering one frame with each into separate files so the
correct one can be picked by direct visual inspection rather than more hand
math. New eye/target reflects "up and back" (raise + pull back further from
the last position).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_render_democam_variants.py
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.graspgoal_democam_env_cfg import Ar4GraspGoalDemoEnvCfg  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "color_check")
os.makedirs(OUT_DIR, exist_ok=True)

EYE = (0.0, 1.20, 0.55)
TARGET = (0.0, 0.0, 0.15)

eyes = torch.tensor([EYE])
targets = torch.tensor([TARGET])
R = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
quat_primary = quat_from_matrix(R)[0]

# 180-degree roll counterpart: negate x_axis and y_axis (columns 0 and 1 of
# the pre-transpose R, i.e. columns 0 and 1 of R itself since the function
# returns R.transpose(1,2) - forward/z_axis (column 2) stays the same).
R_rolled = R.clone()
R_rolled[:, :, 0] *= -1.0
R_rolled[:, :, 1] *= -1.0
quat_rolled = quat_from_matrix(R_rolled)[0]

print(f"EYE={EYE} TARGET={TARGET}")
print(f"QUAT_PRIMARY (w,x,y,z) = {tuple(quat_primary.tolist())}")
print(f"QUAT_ROLLED  (w,x,y,z) = {tuple(quat_rolled.tolist())}")


def render_with_quat(env_cfg_cls, quat, out_name):
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.demo_camera.offset = CameraCfg.OffsetCfg(pos=EYE, rot=tuple(quat.tolist()), convention="opengl")
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    camera = env.scene["demo_camera"]
    with torch.inference_mode():
        env.reset()
        for _ in range(40):
            actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)
        rgb = camera.data.output["rgb"][:, ..., :3].cpu().numpy()[:, ::-1, :, :]
        frame = rgb[0].astype("uint8")
        out_path = os.path.join(OUT_DIR, out_name)
        imageio.imwrite(out_path, frame)
        print(f"[RENDER] saved: {out_path}  shape={frame.shape}")
    env.close()


render_with_quat(Ar4GraspGoalDemoEnvCfg, quat_primary, "ar4_color_check_primary.png")
render_with_quat(Ar4GraspGoalDemoEnvCfg, quat_rolled, "ar4_color_check_rolled.png")

simulation_app.close()

"""Launch the AR4 mk5 arm in Isaac Lab, drive its joints through a test
sequence, and record joint data to HDF5.

Requires the USD asset to already exist (run build_asset.py first).

.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/drive_joints_demo.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the AR4 mk5 arm through a joint test sequence.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=600, help="Number of steps to run before exiting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedEnv  # noqa: E402

from tasks.ar4.env_cfg import Ar4EnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _raise_window_in_background(title_substr: str, duration_s: float) -> None:
    """Best-effort: keep the sim window raised on desktops that don't auto-focus new windows."""
    try:
        subprocess.Popen(
            ["python3", os.path.join(SCRIPT_DIR, "_raise_window.py"), title_substr, str(duration_s)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass  # not fatal if this desktop doesn't support it


def main() -> None:
    env_cfg = Ar4EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.recorders.dataset_export_dir_path = LOG_DIR
    env_cfg.recorders.dataset_filename = "drive_joints_demo"

    env = ManagerBasedEnv(cfg=env_cfg)

    run_time_s = args_cli.steps * env.step_dt + 10.0
    _raise_window_in_background("Isaac Sim Python", run_time_s)

    num_joints = len(ARM_JOINT_NAMES)
    amplitude = 0.3  # radians
    period_steps = 240

    # Everything (including both resets) stays inside one inference_mode
    # context. Rigid-object buffers (e.g. root_link_pose_w, for the objects
    # in the scene) are lazily allocated on first use rather than at env
    # construction; allocating them inside inference_mode and then
    # in-place-writing to them from outside it (e.g. a later reset()) raises
    # "Inplace update to inference tensor outside InferenceMode".
    with torch.inference_mode():
        env.reset()
        for step in range(args_cli.steps):
            step_start = time.time()
            phase = 2 * math.pi * (step % period_steps) / period_steps
            # Extra column is the gripper's binary open/close action; keep it open
            # throughout since this script only exercises the arm joints.
            target = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
            for j in range(num_joints):
                target[:, j] = amplitude * math.sin(phase + j * (math.pi / num_joints))
            target[:, num_joints] = 1.0
            obs, _ = env.step(target)
            if step % 60 == 0:
                joint_pos = obs["policy"][0, :num_joints].tolist()
                print(f"[Step {step:04d}] joint_pos: {[round(p, 3) for p in joint_pos]}")
            # Pace to real time so the motion is actually watchable in the GUI.
            sleep_time = env.step_dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Trigger a final reset so the recorder manager exports the episode.
        env.reset()

    print("Done. Holding window open for a few seconds before closing...")
    time.sleep(5.0)
    env.close()
    print(f"Joint data recorded to: {LOG_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()

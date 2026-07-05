"""Verify the AR4 mk5 gripper and object scene: reach a placed cube, close the
gripper on it, lift it, hold, then release.

The joint waypoints below are hardcoded (not solved live) - computed offline
via the AR4Kinematics IK solver (src/robot/src/ar4_mk5_kinematics.py) against
the cube's known fixed position from tasks/ar4/objects_cfg.py
(0.20, 0.28, 0.009), targeting a straight-down gripper approach with an
estimated ~0.09m offset from the ee_link frame to the jaw pinch point. That
offset is a rough estimate (the AR4 gripper's exact TCP length wasn't
measured) - if the gripper doesn't land on the cube, this is the first
number to adjust.

The robot's base is rotated 180 deg about world Z (see robot_cfg.py), so the
IK solver's frame_0 (the robot's own base frame) is rotated 180 deg from
world. World-frame targets are converted to base-frame by negating X and Y
before solving (equivalent to applying the inverse of that rotation, which
is self-inverse for a 180 deg turn).

Requires the USD assets to already exist (run build_asset.py first).

.. code-block:: bash

    ./isaaclab.sh -p scripts/grasp_demo.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reach, grasp, and lift the cube with the AR4 mk5 gripper.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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

# Joint targets (radians), solved offline for a straight-down approach to the
# cube at (0.20, 0.28, 0.009).
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PRE_GRASP_Q = [-2.1910457777674273, 0.786924864790331, 2.2832205904522227, 0.0, -1.499346402975541, -2.1910459031084772]
GRASP_Q = [-2.1910458128255588, 0.4814822358369837, 2.1198409433682897, 0.0, -1.0305259069738246, -2.191045812824039]

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# (duration_steps, arm_target, gripper_command)
PHASES = [
    (120, HOME_Q, GRIPPER_OPEN),  # hold at home so the scene is visible before moving
    (180, PRE_GRASP_Q, GRIPPER_OPEN),  # move above the cube
    (90, GRASP_Q, GRIPPER_OPEN),  # descend around the cube
    (60, GRASP_Q, GRIPPER_CLOSE),  # close the gripper
    (90, PRE_GRASP_Q, GRIPPER_CLOSE),  # lift
    (120, PRE_GRASP_Q, GRIPPER_CLOSE),  # hold lifted so the grasp is visible
    (60, PRE_GRASP_Q, GRIPPER_OPEN),  # release
    (180, HOME_Q, GRIPPER_OPEN),  # return home
]


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
    env_cfg.sim.device = args_cli.device
    env_cfg.recorders.dataset_export_dir_path = LOG_DIR
    env_cfg.recorders.dataset_filename = "grasp_demo"

    env = ManagerBasedEnv(cfg=env_cfg)

    total_steps = sum(duration for duration, _, _ in PHASES)
    run_time_s = total_steps * env.step_dt + 10.0
    _raise_window_in_background("Isaac Sim Python", run_time_s)

    num_joints = len(ARM_JOINT_NAMES)

    # Everything (including both resets) stays inside one inference_mode
    # context. Rigid-object buffers (e.g. root_link_pose_w) are lazily
    # allocated on first use rather than at env construction; allocating them
    # inside inference_mode and then in-place-writing to them from outside
    # it (e.g. a later reset()) raises "Inplace update to inference tensor
    # outside InferenceMode".
    with torch.inference_mode():
        env.reset()
        prev_q = HOME_Q
        for duration, target_q, gripper_cmd in PHASES:
            for i in range(duration):
                step_start = time.time()
                alpha = (i + 1) / duration
                q = [prev + alpha * (target - prev) for prev, target in zip(prev_q, target_q)]

                action = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
                for j in range(num_joints):
                    action[:, j] = q[j]
                action[:, num_joints] = gripper_cmd
                env.step(action)

                sleep_time = env.step_dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            prev_q = target_q

        # Trigger a final reset so the recorder manager exports the episode.
        env.reset()

    print("Done. Holding window open for a few seconds before closing...")
    time.sleep(5.0)
    env.close()
    print(f"Joint data recorded to: {LOG_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()

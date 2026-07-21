# scripts/_verify_gripper_mirror_fix.py
"""One-off, non-permanent verification script (ar4-franka-fixes-transfer
plan, Task 5 - controller-authorized gripper-mirror-sign bug fix,
2026-07-21): headless, direct-joint-control check that gripper_jaw2_joint
now tracks gripper_jaw1_joint's MIRRORED (negated) commanded position,
rather than the identical value the pre-fix code used to command.

Drives PhysX directly (write_data_to_sim + sim.step), same pattern as
scripts/interactive_joint_control.py, using the now-fixed
GRIPPER_OPEN_COMMAND_EXPR/GRIPPER_CLOSED_COMMAND_EXPR dicts
(tasks/ar4/robot_cfg.py) instead of a single shared scalar target for both
joints. Prints jaw1/jaw2 settled positions after reset (open), after
commanding close, and after commanding open again - real, direct evidence
(not a shaped metric or an eyeballed video) that jaw2 mirrors jaw1 rather
than overshooting/hitting its own hard limit.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_gripper_mirror_fix.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Verify gripper_jaw2_joint mirror-sign fix.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import (  # noqa: E402
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
)


def _settle(env, robot, gripper_cfg, target_expr, n_steps=60, print_every=10, label=""):
    target = torch.tensor(
        [[target_expr[name] for name in GRIPPER_JOINT_NAMES]], device=env.device
    )
    for i in range(n_steps):
        robot.set_joint_position_target(target, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        if i % print_every == 0 or i == n_steps - 1:
            pos = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
            print(f"  [{label} step {i:3d}] jaw1={pos[0]:+.5f}  jaw2={pos[1]:+.5f}  target={target.cpu().tolist()}")
    return robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)

    with torch.inference_mode():
        env.reset()

    # State 1: fresh reset - should already reflect the fixed init_state
    # (jaw1=+GRIPPER_OPEN_POS, jaw2=-GRIPPER_OPEN_POS).
    reset_pos = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()

    with torch.inference_mode():
        closed_pos = _settle(env, robot, gripper_cfg, GRIPPER_CLOSED_COMMAND_EXPR, label="CLOSE")
        open_pos = _settle(env, robot, gripper_cfg, GRIPPER_OPEN_COMMAND_EXPR, label="OPEN")

    print("=" * 70)
    print(f"GRIPPER_JOINT_NAMES = {GRIPPER_JOINT_NAMES}")
    print(f"GRIPPER_OPEN_COMMAND_EXPR   = {GRIPPER_OPEN_COMMAND_EXPR}")
    print(f"GRIPPER_CLOSED_COMMAND_EXPR = {GRIPPER_CLOSED_COMMAND_EXPR}")
    print("-" * 70)
    print(f"[reset, fixed init_state]   jaw1={reset_pos[0]:+.5f}  jaw2={reset_pos[1]:+.5f}  "
          f"jaw1+jaw2={reset_pos[0] + reset_pos[1]:+.5f}")
    print(f"[commanded CLOSED, settled] jaw1={closed_pos[0]:+.5f}  jaw2={closed_pos[1]:+.5f}  "
          f"jaw1+jaw2={closed_pos[0] + closed_pos[1]:+.5f}")
    print(f"[commanded OPEN, settled]   jaw1={open_pos[0]:+.5f}  jaw2={open_pos[1]:+.5f}  "
          f"jaw1+jaw2={open_pos[0] + open_pos[1]:+.5f}")
    print("-" * 70)
    mirror_ok = abs(open_pos[0] + open_pos[1]) < 0.002 and abs(closed_pos[0] + closed_pos[1]) < 0.002
    print(f"MIRROR CHECK (jaw1 ~= -jaw2 in both states, sum ~= 0): {'PASS' if mirror_ok else 'FAIL'}")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

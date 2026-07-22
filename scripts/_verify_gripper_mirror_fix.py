# scripts/_verify_gripper_mirror_fix.py
"""One-off, non-permanent verification script (ar4-franka-fixes-transfer
plan, Task 5 - controller-authorized gripper-mirror-sign bug fix,
2026-07-21; extended 2026-07-21 "later" pass to verify the follow-up
PhysxMimicJointAPI-removal fix): direct-joint-control check that
gripper_jaw2_joint now tracks gripper_jaw1_joint's MIRRORED (negated)
commanded position under real PD-actuator dynamics, rather than either
the identical-value pre-fix bug or the "pinned at a hard limit regardless
of target" behavior found in the first live dynamic rollout after the
sign fix alone (see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
2026-07-21 "later" UPDATE). That rollout implicated a physics-level
tug-of-war between gripper_jaw2_joint's PhysxMimicJointAPI spring
constraint and its independent ImplicitActuatorCfg PD actuator; the fix
(scripts/build_asset.py's new _remove_gripper_jaw2_mimic_constraint)
strips the mimic constraint entirely at asset-build time. This script's
job is to confirm, with real dynamic data (not just a static USD
inspection), that jaw2 now tracks its own PD target with a normal
convergence curve like jaw1's.

Drives PhysX directly (write_data_to_sim + sim.step), same pattern as
scripts/interactive_joint_control.py, using the
GRIPPER_OPEN_COMMAND_EXPR/GRIPPER_CLOSED_COMMAND_EXPR dicts
(tasks/ar4/robot_cfg.py) instead of a single shared scalar target for both
joints. Prints jaw1/jaw2 settled positions after reset (open), after
commanding close, and after commanding open again - real, direct evidence
(not a shaped metric or an eyeballed video) that jaw2 mirrors jaw1 rather
than overshooting/hitting its own hard limit.

Runs non-headless per this repo's standing convention (CLAUDE.md
"Environment conventions": don't set headless for Isaac-Sim-touching
scripts while the user wants to watch) - requires DISPLAY=:1 in the
launching environment.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_gripper_mirror_fix.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Verify gripper_jaw2_joint mirror-sign fix.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

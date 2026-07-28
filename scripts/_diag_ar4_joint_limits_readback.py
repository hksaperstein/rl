"""Tiny, fast diagnostic (2026-07-28, ar4-joint-tracking-closed-loop-fix
task follow-up): read back the REAL joint position limits baked into the
built AR4 USD asset's physics, via `robot.data.joint_pos_limits` /
`robot.data.soft_joint_pos_limits`, and compare them against
`scripts/ar4_graspable_workspace.py`'s own assumed `JOINT_LIMITS_DEG`
table (currently `joint_2: (-42.0, 90.0)`).

Motivation: `scripts/ar4_tracking_fix_confirm.py`'s stiffness-sweep/
closed-loop-primitive data showed joint_2's tracking error stuck at
~3.2-3.3deg REGARDLESS of a 20x stiffness increase (4000->80000), a 10x
effort_limit_sim increase (20->200), or a closed-loop integral correction
that grew the commanded target by +26deg without ever reducing the error -
while `applied_torque` for joint_2 stayed a roughly CONSTANT ~2.3Nm across
every one of those regimes (not scaling with stiffness the way a real PD
spring's torque=stiffness*error would predict a ~20x change). This is the
signature of a genuine hard position-limit constraint, not a finite-gain
PD droop - this script checks the actual baked-in limit value directly
instead of inferring it.

No cube, no phases, no camera - just construct the env and print the
limits immediately (populated at articulation init, before any reset/step
is even needed).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/_diag_ar4_joint_limits_readback.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Read back AR4's real baked-in arm joint position limits.")
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
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

# scripts/ar4_graspable_workspace.py's own ASSUMED table, copied verbatim
# for direct comparison against the REAL baked-in physics limits below.
ASSUMED_JOINT_LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-180.0, 180.0),
}


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)

    with torch.inference_mode():
        env.reset()

        hard_limits = robot.data.joint_pos_limits[0, arm_cfg.joint_ids].tolist()
        soft_limits = robot.data.soft_joint_pos_limits[0, arm_cfg.joint_ids].tolist()
        default_q = robot.data.default_joint_pos[0, arm_cfg.joint_ids].tolist()

        print("\n" + "=" * 70)
        print("REAL BAKED-IN ARM JOINT LIMITS vs. ASSUMED (ar4_graspable_workspace.py)")
        print("=" * 70)
        for i, name in enumerate(ARM_JOINT_NAMES):
            hard_lo_deg = math.degrees(hard_limits[i][0])
            hard_hi_deg = math.degrees(hard_limits[i][1])
            soft_lo_deg = math.degrees(soft_limits[i][0])
            soft_hi_deg = math.degrees(soft_limits[i][1])
            assumed_lo, assumed_hi = ASSUMED_JOINT_LIMITS_DEG[name]
            mismatch = (abs(hard_lo_deg - assumed_lo) > 0.5) or (abs(hard_hi_deg - assumed_hi) > 0.5)
            print(
                f"  {name}: REAL hard=[{hard_lo_deg:8.3f}, {hard_hi_deg:8.3f}]deg  "
                f"REAL soft=[{soft_lo_deg:8.3f}, {soft_hi_deg:8.3f}]deg  "
                f"ASSUMED=[{assumed_lo:8.3f}, {assumed_hi:8.3f}]deg  "
                f"default_q={math.degrees(default_q[i]):7.3f}deg  "
                f"{'*** MISMATCH ***' if mismatch else 'match'}"
            )
        print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

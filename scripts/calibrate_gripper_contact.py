"""Calibrate the AR4 gripper's ContactSensor-based grasp reward
(tasks/ar4/mdp.py's contact_grasp_bonus) against a real scripted grasp,
before spending a full training run on an untested force_threshold.

Reuses scripts/grasp_demo.py's already-solved IK waypoints verbatim - those
were computed for the cube's fixed position (0.20, 0.28, 0.009); this script
relocates the sphere to that exact position (and disables its usual random
reset jitter) so the same waypoints land on it, rather than re-deriving IK
for the sphere's own default (mirrored) spawn position.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/calibrate_gripper_contact.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate the AR4 gripper ContactSensor against a real sphere grasp.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.mdp import contact_grasp_bonus  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

# Verbatim from scripts/grasp_demo.py.
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PRE_GRASP_Q = [-2.1910457777674273, 0.786924864790331, 2.2832205904522227, 0.0, -1.499346402975541, -2.1910459031084772]
GRASP_Q = [-2.1910458128255588, 0.4814822358369837, 2.1198409433682897, 0.0, -1.0305259069738246, -2.191045812824039]

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# (duration_steps, arm_target, gripper_command, label)
PHASES = [
    (60, HOME_Q, GRIPPER_OPEN, "home"),
    (180, PRE_GRASP_Q, GRIPPER_OPEN, "approach"),
    (90, GRASP_Q, GRIPPER_OPEN, "descend"),
    (60, GRASP_Q, GRIPPER_CLOSE, "close"),
    (90, PRE_GRASP_Q, GRIPPER_CLOSE, "lift"),
    (120, PRE_GRASP_Q, GRIPPER_CLOSE, "hold"),
]

# Must match tasks/ar4/pickplace_env_cfg.py's grasp_contact RewTerm params.
FORCE_THRESHOLD = 0.05


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    # Relocate the sphere onto the cube's exact, pre-solved grasp position.
    env_cfg.scene.sphere.init_state.pos = (0.20, 0.28, 0.009)
    # Disable this task's usual +-2cm reset jitter for the sphere so it lands
    # exactly where the reused waypoints expect it.
    env_cfg.events.reset_sphere_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }
    total_steps = sum(duration for duration, _, _, _ in PHASES)
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = total_steps * step_dt + 5.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_joints = len(ARM_JOINT_NAMES)
    jaw1_cfg = SceneEntityCfg("gripper_jaw1_contact")
    jaw2_cfg = SceneEntityCfg("gripper_jaw2_contact")

    home_forces: list[list[float]] = []
    hold_forces: list[list[float]] = []
    hold_rewards: list[float] = []

    with torch.inference_mode():
        env.reset()
        prev_q = HOME_Q
        for duration, target_q, gripper_cmd, label in PHASES:
            for i in range(duration):
                alpha = (i + 1) / duration
                q = [prev + alpha * (target - prev) for prev, target in zip(prev_q, target_q)]

                action = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
                for j in range(num_joints):
                    action[:, j] = q[j]
                action[:, num_joints] = gripper_cmd
                env.step(action)

                jaw1_sensor = env.scene["gripper_jaw1_contact"]
                jaw2_sensor = env.scene["gripper_jaw2_contact"]
                jaw1_norm = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(1)
                jaw2_norm = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(1)
                force_norm = [jaw1_norm.item(), jaw2_norm.item()]
                reward = contact_grasp_bonus(env, FORCE_THRESHOLD, jaw1_cfg, jaw2_cfg)

                if label == "hold":
                    hold_forces.append(force_norm)
                    hold_rewards.append(reward.item())
                elif label == "home":
                    home_forces.append(force_norm)

            prev_q = target_q
            print(f"[phase done] {label}: last force_norm={force_norm}, reward={reward.item()}")

    print("\n=== Calibration summary ===")
    home_min = min(min(f) for f in home_forces)
    home_max = max(max(f) for f in home_forces)
    hold_min = min(min(f) for f in hold_forces)
    hold_max = max(max(f) for f in hold_forces)
    hold_success = sum(r == 1.0 for r in hold_rewards)
    print(f"home (open, far from sphere) force_norm: min={home_min:.4f}, max={home_max:.4f} N (expect ~0.0)")
    print(f"hold (closed, lifted)        force_norm: min={hold_min:.4f}, max={hold_max:.4f} N")
    print(
        f"hold reward==1.0 fraction: {hold_success}/{len(hold_rewards)} "
        f"(force_threshold={FORCE_THRESHOLD})"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Instrumented rollout of Experiment 23's trained checkpoint
(Ar4PickPlaceWarmResidualEnvCfg, model_1499.pt) - does antipodal contact
happen under the warm-started residual action, using the same
antipodal_grasp_bonus/genuine_grasp_and_lift computation this repo's own
mdp.py uses (tasks/ar4/mdp.py), reproduced inline exactly as Experiments
20/21/22's own contact diagnostics already do:
  - height_ok:          cube world z > minimal_height (0.03)
  - both_magnitude_ok:  jaw1_force_mag > force_threshold (0.05) AND
                         jaw2_force_mag > force_threshold (0.05)
  - antipodal_ok:       cos(angle between jaw1_force_dir, jaw2_force_dir)
                         < antipodal_cos_threshold (-0.7071)
  - grasp_ok:           both_magnitude_ok AND antipodal_ok
  - gate_fires:         height_ok AND grasp_ok

Also logs residual_authority (expected 1.0 throughout, since training is
complete and _step_count is already far past warmup_steps by the time
this checkpoint was saved) and gripper joint positions.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_contact_diagnostic.py \
        --checkpoint <path> --episodes 3
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 23 antipodal-contact diagnostic.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=3, help="Number of full episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4.pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402

FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
MINIMAL_HEIGHT = 0.03


def main() -> None:
    env_cfg = Ar4PickPlaceWarmResidualEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    arm_action_term = env.action_manager.get_term("arm_action")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    cube = env.scene["cube"]
    ee_frame = env.scene["ee_frame"]
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]
    robot = env.scene["robot"]

    gripper_joint_ids, gripper_joint_names = robot.find_joints(["gripper_jaw1_joint", "gripper_jaw2_joint"])
    print(f"[SETUP] gripper joint ids={gripper_joint_ids} names={gripper_joint_names}")
    print(
        f"[SETUP] thresholds: force_threshold={FORCE_THRESHOLD} "
        f"antipodal_cos_threshold={ANTIPODAL_COS_THRESHOLD} minimal_height={MINIMAL_HEIGHT} "
        f"warmup_steps={arm_action_term.cfg.warmup_steps}"
    )

    stats = {
        "total_steps": 0,
        "height_ok_steps": 0,
        "both_magnitude_ok_steps": 0,
        "antipodal_ok_steps": 0,
        "grasp_ok_steps": 0,
        "gate_fires_steps": 0,
        "max_jaw1_force": 0.0,
        "max_jaw2_force": 0.0,
        "max_cube_z": -1.0,
        "min_residual_authority": 2.0,
    }

    obs = wrapped.get_observations()
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            print(f"[EPISODE {episode} START]")
            for step in range(250):
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)

                cube_pos = cube.data.root_pos_w[0]
                cube_z = cube_pos[2].item()

                jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec).item()
                jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec).item()

                jaw1_dir = jaw1_force_vec / (jaw1_force_mag + 1e-8)
                jaw2_dir = jaw2_force_vec / (jaw2_force_mag + 1e-8)
                cos_angle = torch.sum(jaw1_dir * jaw2_dir).item()

                height_ok = cube_z > MINIMAL_HEIGHT
                both_magnitude_ok = (jaw1_force_mag > FORCE_THRESHOLD) and (jaw2_force_mag > FORCE_THRESHOLD)
                antipodal_ok = cos_angle < ANTIPODAL_COS_THRESHOLD
                grasp_ok = both_magnitude_ok and antipodal_ok
                gate_fires = height_ok and grasp_ok

                gripper_joint_pos = robot.data.joint_pos[0, gripper_joint_ids].tolist()
                residual_authority = min(1.0, arm_action_term._step_count / arm_action_term.cfg.warmup_steps)

                ee_quat_w = ee_frame.data.target_quat_w[0, 0, :]
                approach_dir = quat_apply(ee_quat_w.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=env.device))[0]
                orientation_dot = torch.dot(approach_dir, torch.tensor([0.0, 0.0, -1.0], device=env.device)).item()

                stats["total_steps"] += 1
                stats["height_ok_steps"] += int(height_ok)
                stats["both_magnitude_ok_steps"] += int(both_magnitude_ok)
                stats["antipodal_ok_steps"] += int(antipodal_ok)
                stats["grasp_ok_steps"] += int(grasp_ok)
                stats["gate_fires_steps"] += int(gate_fires)
                stats["max_jaw1_force"] = max(stats["max_jaw1_force"], jaw1_force_mag)
                stats["max_jaw2_force"] = max(stats["max_jaw2_force"], jaw2_force_mag)
                stats["max_cube_z"] = max(stats["max_cube_z"], cube_z)
                stats["min_residual_authority"] = min(stats["min_residual_authority"], residual_authority)

                print(
                    f"[EP {episode} STEP {step:3d}] cube_z={cube_z:.4f} height_ok={int(height_ok)} "
                    f"jaw1_force={jaw1_force_mag:.5f} jaw2_force={jaw2_force_mag:.5f} "
                    f"both_mag_ok={int(both_magnitude_ok)} cos_angle={cos_angle:.4f} antipodal_ok={int(antipodal_ok)} "
                    f"grasp_ok={int(grasp_ok)} GATE_FIRES={int(gate_fires)} "
                    f"jaw_joint_pos={gripper_joint_pos[0]:.5f}/{gripper_joint_pos[1]:.5f} "
                    f"orientation_dot={orientation_dot:.4f} residual_authority={residual_authority:.4f}"
                )

                if bool(dones[0]):
                    print(f"[EP {episode} STEP {step}] episode done (early termination), stopping episode")
                    break
            print(f"[EPISODE {episode} END]")

    print("[SUMMARY] " + " ".join(f"{k}={v}" for k, v in stats.items()))
    print("[DIAGNOSTIC COMPLETE]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

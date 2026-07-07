"""Experiment 19 Task 2: instrumented rollout verifying the mimic-joint
fix (Task 1) - do gripper_jaw1_joint and gripper_jaw2_joint now track
each other under contact load, not just at rest?

Adapted from the Experiment 17 Task 6 diagnostic pattern
(exp17_grasp_gate_diagnostic.py): loads the Experiment 18 checkpoint
(the same trained policy, frozen weights) and rolls it out against the
newly-rebuilt USD asset (Task 1's fix), logging both jaw joint positions
and their divergence every step, with running stats restricted to steps
with measurable jaw contact force (not just at rest, where both jaws
naturally agree regardless of whether the fix works).

Baseline to compare against: Experiment 17 Task 6 found gripper_jaw2_joint
drifting to 0.0168 while gripper_jaw1_joint stayed exactly at 0.0140 under
contact load - a 0.0028m (20% of the 0.014m travel range) divergence.
PASS threshold: max divergence during contact under 0.0014m (10% of the
travel range) - a clear, substantial improvement, not just noise.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/mimic_joint_verify.py \
        --checkpoint /home/saps/projects/rl/logs/train/2026-07-07_16-38-01/model_1499.pt --episodes 3
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 19 mimic-joint fix verification.")
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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_pregrasp_env_cfg import Ar4PickPlacePregraspEnvCfg  # noqa: E402

# Task 1's fix acceptance threshold: 10% of the 0.014m gripper travel
# range, well below Experiment 17 Task 6's measured 0.0028m (20%) divergence.
PASS_THRESHOLD_M = 0.0014
# Any force reading above this is treated as "real contact," not float noise
# at rest (Experiment 17 Task 6 found contact forces of 7-20N when real
# contact occurred, vs. exactly 0.00000 at rest).
CONTACT_FORCE_EPSILON = 1e-4


def main() -> None:
    env_cfg = Ar4PickPlacePregraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    robot = env.scene["robot"]
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]

    gripper_joint_ids, gripper_joint_names = robot.find_joints(["gripper_jaw1_joint", "gripper_jaw2_joint"])
    print(f"[SETUP] gripper joint ids={gripper_joint_ids} names={gripper_joint_names}")
    print(f"[SETUP] pass_threshold_m={PASS_THRESHOLD_M} contact_force_epsilon={CONTACT_FORCE_EPSILON}")

    stats = {
        "total_steps": 0,
        "contact_steps": 0,
        "max_jaw_pos_diff_at_rest": 0.0,
        "max_jaw_pos_diff_during_contact": 0.0,
        "sum_jaw_pos_diff_during_contact": 0.0,
        "max_jaw1_force": 0.0,
        "max_jaw2_force": 0.0,
    }

    obs = wrapped.get_observations()
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            print(f"[EPISODE {episode} START]", flush=True)
            try:
                for step in range(250):
                    actions = policy(obs)
                    obs, _, dones, _ = wrapped.step(actions)

                    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(1, 3)[0]
                    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(1, 3)[0]
                    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec).item()
                    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec).item()

                    gripper_joint_pos = robot.data.joint_pos[0, gripper_joint_ids].tolist()
                    jaw_pos_diff = abs(gripper_joint_pos[0] - gripper_joint_pos[1])

                    in_contact = (jaw1_force_mag > CONTACT_FORCE_EPSILON) or (jaw2_force_mag > CONTACT_FORCE_EPSILON)

                    stats["total_steps"] += 1
                    stats["max_jaw1_force"] = max(stats["max_jaw1_force"], jaw1_force_mag)
                    stats["max_jaw2_force"] = max(stats["max_jaw2_force"], jaw2_force_mag)
                    if in_contact:
                        stats["contact_steps"] += 1
                        stats["max_jaw_pos_diff_during_contact"] = max(
                            stats["max_jaw_pos_diff_during_contact"], jaw_pos_diff
                        )
                        stats["sum_jaw_pos_diff_during_contact"] += jaw_pos_diff
                    else:
                        stats["max_jaw_pos_diff_at_rest"] = max(stats["max_jaw_pos_diff_at_rest"], jaw_pos_diff)

                    print(
                        f"[EP {episode} STEP {step:3d}] jaw1_force={jaw1_force_mag:.5f} jaw2_force={jaw2_force_mag:.5f} "
                        f"in_contact={int(in_contact)} jaw1_pos={gripper_joint_pos[0]:.5f} "
                        f"jaw2_pos={gripper_joint_pos[1]:.5f} jaw_pos_diff={jaw_pos_diff:.5f}",
                        flush=True
                    )

                    if bool(dones[0]):
                        print(f"[EP {episode} STEP {step}] episode done (early termination), stopping episode", flush=True)
                        break
            except Exception as e:
                print(f"[ERROR] Exception during episode {episode}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            print(f"[EPISODE {episode} END]", flush=True)

    try:
        print("[STATS_CALC] Starting statistics calculation", flush=True)
        mean_diff_during_contact = (
            stats["sum_jaw_pos_diff_during_contact"] / stats["contact_steps"] if stats["contact_steps"] > 0 else None
        )
        result = "PASS" if (
            stats["contact_steps"] > 0 and stats["max_jaw_pos_diff_during_contact"] < PASS_THRESHOLD_M
        ) else "FAIL"

        print(
            "[SUMMARY] "
            f"total_steps={stats['total_steps']} contact_steps={stats['contact_steps']} "
            f"max_jaw_pos_diff_at_rest={stats['max_jaw_pos_diff_at_rest']:.5f} "
            f"max_jaw_pos_diff_during_contact={stats['max_jaw_pos_diff_during_contact']:.5f} "
            f"mean_jaw_pos_diff_during_contact={mean_diff_during_contact} "
            f"max_jaw1_force={stats['max_jaw1_force']:.5f} max_jaw2_force={stats['max_jaw2_force']:.5f}",
            flush=True
        )
        print(f"[RESULT] {result} (threshold={PASS_THRESHOLD_M}m, contact_steps={stats['contact_steps']})", flush=True)
        print("[DIAGNOSTIC COMPLETE]", flush=True)
    except Exception as e:
        print(f"[ERROR] Exception during stats calculation: {e}", flush=True)
        import traceback
        traceback.print_exc()

    try:
        print("[CLEANUP] Closing environment", flush=True)
        env.close()
        print("[CLEANUP] Environment closed", flush=True)
    except Exception as e:
        print(f"[ERROR] Exception during env close: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()

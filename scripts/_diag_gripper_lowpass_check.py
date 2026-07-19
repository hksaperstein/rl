"""Diagnostic (not a training/eval-suite script): directly check whether this
project's Franka gripper action formulation exhibits the Neunert et al.
(arXiv:2001.00449) "slow gripper acts as a low-pass filter on Gaussian
exploration noise" failure mode.

Dispatched from docs/superpowers/specs/research/2026-07-19-exploration-
reward-expansion-literature.md's flagged, unresolved prerequisite check
(section 5's "flagged, out-of-scope prerequisite check").

Config-reading already gives a strong prior: this project's gripper action
is `mdp.BinaryJointPositionActionCfg` (tasks/franka/lift_env_cfg.py,
inherited unchanged by every joint-die variant including the target-
selection-clutter Stage SO env). Isaac Lab's own `BinaryJointAction.
process_actions` (isaaclab/envs/mdp/actions/binary_joint_actions.py) maps
ANY raw action value via a hard SIGN THRESHOLD to the FULL open_command_expr
or close_command_expr joint-position target -- there is no proportional/
scaled mapping from raw action magnitude to a partial command the way a
continuous velocity/position-delta action space would have. This is
structurally different from Neunert et al.'s failure setup, where the
BASELINE (not their fix) controls the gripper in continuous VELOCITY mode
and small-magnitude, zero-mean Gaussian exploration noise gets attenuated
by slow finger actuator dynamics before ever producing enough integrated
motion to reach a meaningfully-closed position. This project's binary
threshold action, by construction, cannot suffer that exact mechanism: a
raw action of -1e-4 to closes it produces the SAME processed target
(full close_command_expr, panda_finger_.*: 0.0) as a raw action of -10.0.

This script gets the direct empirical check the config-reading finding
still needs per this project's verification standard: run the actual
Stage SO checkpoint for one full episode and record, every step:
  - the raw gripper action term (env.unwrapped.action_manager's
    "gripper_action" term's `.raw_actions`, i.e. exactly what the policy
    itself output for that action dimension, pre-threshold)
  - the PROCESSED command that term computed (`.processed_actions`,
    the joint-position target actually sent to the articulation)
  - the REALIZED physical joint position of the panda_finger joints
    (env.unwrapped.scene["robot"].data.joint_pos at the resolved finger
    joint indices) -- does the joint actually reach the commanded target,
    or does actuator dynamics (PD stiffness/damping/effort limits) prevent
    it within the episode?

Run via (desktop or cloud GPU, non-headless, under the flock lock):

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_gripper_lowpass_check.py \
        --checkpoint <path-to-model_1499.pt> --eval_target_shape d20 --num_envs 8"
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose gripper action low-pass-filtering (Neunert et al.).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--eval_target_shape", choices=["d12", "d20"], default="d20")
parser.add_argument("--num_steps", type=int, default=250, help="One full episode (5.0s @ decimation=2, dt=0.01 -> 250 steps).")
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Non-headless per standing project instruction (CLAUDE.md "Environment
# conventions": "don't set args_cli.headless = True / don't pass --headless
# for any Isaac-Sim-touching script" -- display available, user wants to
# watch). Left as whatever the launcher default/CLI flags resolve to; this
# script does not force headless=True anywhere.

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.dice_lift_joint_env_cfg import (  # noqa: E402
    FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg_PLAY_D12Target,
    FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg_PLAY_D20Target,
)

OUTPUT_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "diag_gripper_lowpass"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    env_cfg = (
        FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg_PLAY_D12Target()
        if args_cli.eval_target_shape == "d12"
        else FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg_PLAY_D20Target()
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)

    # Resolve panda_finger joint indices directly off the articulation, same
    # mechanism BinaryJointAction itself uses (Articulation.find_joints),
    # so the indices are guaranteed correct regardless of joint ordering.
    finger_joint_ids, finger_joint_names = env.scene["robot"].find_joints(["panda_finger.*"])
    print(f"[diag] resolved finger joint ids={finger_joint_ids} names={finger_joint_names}")

    gripper_term = env.action_manager.get_term("gripper_action")
    print(f"[diag] gripper_action term: {type(gripper_term).__name__}")
    print(f"[diag] open_command={gripper_term._open_command.tolist()} close_command={gripper_term._close_command.tolist()}")

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    num_envs = args_cli.num_envs
    num_steps = args_cli.num_steps
    num_fingers = len(finger_joint_ids)

    raw_gripper_action = torch.zeros((num_steps, num_envs))
    processed_gripper_cmd = torch.zeros((num_steps, num_envs, num_fingers))
    realized_finger_pos = torch.zeros((num_steps, num_envs, num_fingers))
    object_z = torch.zeros((num_steps, num_envs))

    obs = wrapped_env.get_observations()
    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            # gripper_action is the LAST action dim (arm_action is
            # JointPositionActionCfg over 7 panda_joint.* dims, registered
            # first in ActionsCfg; gripper_action, dim=1, registered second
            # -- isaaclab's ActionManager concatenates term action spaces in
            # registration order). Recorded BEFORE env.step() processes it,
            # i.e. this is exactly what the policy itself output.
            raw_gripper_action[step] = actions[:, -1].detach().cpu()

            obs, _, _, _ = wrapped_env.step(actions)

            # Read the term's own post-process_actions buffers directly,
            # rather than re-deriving them, so this matches EXACTLY what
            # was sent to the articulation this step.
            processed_gripper_cmd[step] = gripper_term.processed_actions.detach().cpu()
            realized_finger_pos[step] = env.scene["robot"].data.joint_pos[:, finger_joint_ids].detach().cpu()
            object_z[step] = env.scene["object"].data.root_pos_w[:, 2].detach().cpu()

    env.close()

    raw_np = raw_gripper_action.numpy()
    processed_np = processed_gripper_cmd.numpy()
    realized_np = realized_finger_pos.numpy()
    object_z_np = object_z.numpy()

    open_val = float(gripper_term._open_command[0])
    close_val = float(gripper_term._close_command[0])
    # "attempted close" = raw action sign is negative (BinaryJointAction's
    # own threshold rule for float actions: `binary_mask = actions < 0`).
    attempted_close_mask = raw_np < 0.0
    frac_steps_attempted_close_per_env = attempted_close_mask.mean(axis=0)  # (num_envs,)

    # "achieved close" = realized finger position within 20% of the way
    # from open_val to close_val, i.e. actually moved substantially toward
    # the commanded closed target (not just commanded closed).
    close_threshold = open_val - 0.2 * (open_val - close_val)  # 20% of the way closed
    achieved_close_mask = realized_np.mean(axis=2) <= close_threshold  # mean over fingers, (num_steps, num_envs)
    frac_steps_achieved_close_per_env = achieved_close_mask.mean(axis=0)

    # For steps where a close WAS commanded, how many steps later (if ever)
    # did the joint actually reach the close_threshold? Direct tracking-lag
    # check, per-env, using the first commanded-close run.
    tracking_lag_steps = {}
    for e in range(num_envs):
        close_cmd_steps = np.where(processed_np[:, e, :].mean(axis=1) <= close_threshold)[0]
        if len(close_cmd_steps) == 0:
            tracking_lag_steps[str(e)] = None
            continue
        first_cmd_step = int(close_cmd_steps[0])
        achieved_steps = np.where(achieved_close_mask[first_cmd_step:, e])[0]
        tracking_lag_steps[str(e)] = int(achieved_steps[0]) if len(achieved_steps) > 0 else None

    print("\n=== Gripper low-pass-filter diagnostic ===")
    print(f"open_command={open_val} close_command={close_val} close_threshold(20% closed)={close_threshold:.4f}")
    for e in range(num_envs):
        print(
            f"  env {e}: frac_steps_raw_action_negative(attempted close)="
            f"{frac_steps_attempted_close_per_env[e]:.3f} "
            f"frac_steps_joint_actually_>=20%_closed={frac_steps_achieved_close_per_env[e]:.3f} "
            f"tracking_lag_steps_after_first_close_cmd={tracking_lag_steps[str(e)]} "
            f"raw_action_min={raw_np[:, e].min():.4f} raw_action_max={raw_np[:, e].max():.4f}"
        )

    summary = {
        "checkpoint": args_cli.checkpoint,
        "eval_target_shape": args_cli.eval_target_shape,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "finger_joint_names": finger_joint_names,
        "open_command": open_val,
        "close_command": close_val,
        "close_threshold_20pct": float(close_threshold),
        "per_env": {
            str(e): {
                "frac_steps_raw_action_negative": float(frac_steps_attempted_close_per_env[e]),
                "frac_steps_joint_actually_20pct_closed": float(frac_steps_achieved_close_per_env[e]),
                "tracking_lag_steps_after_first_close_cmd": tracking_lag_steps[str(e)],
                "raw_action_min": float(raw_np[:, e].min()),
                "raw_action_max": float(raw_np[:, e].max()),
            }
            for e in range(num_envs)
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{args_cli.eval_target_shape}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    raw_path = os.path.join(OUTPUT_DIR, f"raw_arrays_{args_cli.eval_target_shape}.npz")
    np.savez(
        raw_path,
        raw_gripper_action=raw_np,
        processed_gripper_cmd=processed_np,
        realized_finger_pos=realized_np,
        object_z=object_z_np,
    )
    print(f"Raw arrays written to: {raw_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Diagnostic (not a training/eval-suite script): mechanism-level falsification
bar for H1 (docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-
discovery-design.md), Task 3 of docs/superpowers/plans/2026-07-19-exploration-
bonus-grasp-discovery-implementation.md.

Generalizes scripts/_diag_gripper_lowpass_check.py (NEVER modifies that file
in place - this is a new sibling) by additionally capturing per-step
end-to-effector-to-object distance and restricting the "did the policy ever
attempt closure" fraction to only the steps where the end-effector is actually
near the object. Unrestricted, the existing diagnostic can't distinguish "the
policy never closes near the object" from "the policy never closes at all,
including moments far away where closing is irrelevant" - this script closes
that gap.

Per the spec's own "Falsification bar" section 1 (design doc lines 341-343):
"near the object" = end-effector-to-object distance < 5cm (0.05m), matching
this design's own `std_gate` constant (also 0.05, tasks/franka/lift_env_cfg.py
`_EXPLORATION_BONUS_PARAMS`) - NOT independently chosen here. Computes, per
env, `frac_steps_raw_action_negative_near_object` = (# steps with raw
gripper action < 0 AND distance < 5cm) / (# steps with distance < 5cm),
i.e. restricted to the subset of steps where the env was near the object at
all - a step where the env is far from the object contributes to neither the
numerator nor the denominator.

Explicit edge case (per the implementation plan's Task 3, do not silently
default to 0.0): if an env never gets within 5cm of the object during the
whole episode (denominator == 0), this script reports
`frac_steps_raw_action_negative_near_object: null` for that env, not 0.0/an
error - "never got close enough to test the mechanism at all" is a
categorically different, and separately worth reporting, finding from a real
`0.000` ("got close, never attempted"). Conflating the two would misstate
H1's own mechanism-level result depending on which envs happen to fall into
which bucket (this is exactly why the spec/plan call this out explicitly).

Access patterns, both reused verbatim from already-proven-working call
sites, not re-derived:
  - raw gripper action: `env.action_manager.get_term("gripper_action").raw_actions`... in
    practice (matching _diag_gripper_lowpass_check.py's own proven pattern)
    captured as `actions[:, -1]`, the last dim of the policy's own pre-step()
    output tensor (gripper_action is registered second/last in ActionsCfg,
    so the ActionManager's concatenated action tensor puts it last) - NOT a
    post-step() `.raw_actions` attribute read, since the existing diagnostic
    already established the pre-step() capture as the correct, working
    pattern for "what the policy itself output."
  - end-effector / object position: `env.scene["ee_frame"].data.target_pos_w[..., 0, :]`
    vs. `env.scene["object"].data.root_pos_w[:, :3]`, identical to
    `tasks/franka/mdp.py`'s own `object_ee_distance` reward-term access
    pattern (mdp.py:87-98) - captured AFTER env.step() in the same step
    iteration, matching production reward-computation timing (the
    RewardManager computes gripper_closure_attempt_bonus off the
    post-physics-step scene state combined with the just-applied
    raw_actions, so this diagnostic's per-step (raw_action, distance) pairing
    mirrors exactly what the real reward term sees at that step).

Run via (desktop or cloud GPU, non-headless, under the flock lock):

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_gripper_closure_near_object_check.py \
        --variant joint-die-d8-big-exploration-bonus --checkpoint <path-to-model_1499.pt> --num_envs 8"
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

# Matches tasks/franka/lift_env_cfg.py's _EXPLORATION_BONUS_PARAMS["std_gate"]
# (0.05) and the design spec's own "Falsification bar" section 1 ("< 5cm,
# matching this design's own std_gate") - not an independently chosen value.
NEAR_OBJECT_THRESHOLD_M = 0.05

parser = argparse.ArgumentParser(
    description="Near-object-restricted gripper-closure-attempt mechanism diagnostic (H1 mechanism-level bar)."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument(
    "--variant",
    choices=["joint-die-d8-big-exploration-bonus"],
    default="joint-die-d8-big-exploration-bonus",
    help="Env-cfg variant to diagnose. Only the H1 exploration-bonus d8 env cfg is supported by this script.",
)
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
from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY  # noqa: E402

OUTPUT_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "diag_gripper_closure_near_object"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    env_cfg = FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)

    gripper_term = env.action_manager.get_term("gripper_action")
    print(f"[diag] gripper_action term: {type(gripper_term).__name__}")
    print(f"[diag] open_command={gripper_term._open_command.tolist()} close_command={gripper_term._close_command.tolist()}")
    print(f"[diag] near_object_threshold_m={NEAR_OBJECT_THRESHOLD_M} (matching std_gate, per the design spec)")

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    num_envs = args_cli.num_envs
    num_steps = args_cli.num_steps

    raw_gripper_action = torch.zeros((num_steps, num_envs))
    ee_object_distance = torch.zeros((num_steps, num_envs))
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
            # i.e. this is exactly what the policy itself output -- same
            # pattern as _diag_gripper_lowpass_check.py.
            raw_gripper_action[step] = actions[:, -1].detach().cpu()

            obs, _, _, _ = wrapped_env.step(actions)

            # Same access pattern as mdp.object_ee_distance (mdp.py:87-98),
            # captured post-step() so this pairs with the raw action applied
            # THIS step exactly the way the real reward term would see it at
            # reward-computation time.
            ee_frame = env.scene["ee_frame"]
            object_entity = env.scene["object"]
            ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
            object_pos_w = object_entity.data.root_pos_w[:, :3]
            dist = torch.linalg.norm(object_pos_w - ee_pos_w, dim=-1)
            ee_object_distance[step] = dist.detach().cpu()
            object_z[step] = object_entity.data.root_pos_w[:, 2].detach().cpu()

    env.close()

    raw_np = raw_gripper_action.numpy()
    dist_np = ee_object_distance.numpy()
    object_z_np = object_z.numpy()

    # "attempted close" = raw action sign is negative (BinaryJointAction's
    # own threshold rule for float actions: `binary_mask = actions < 0`),
    # identical to _diag_gripper_lowpass_check.py's own convention.
    attempted_close_mask = raw_np < 0.0  # (num_steps, num_envs)
    near_object_mask = dist_np < NEAR_OBJECT_THRESHOLD_M  # (num_steps, num_envs)
    near_and_attempted_mask = attempted_close_mask & near_object_mask

    frac_steps_attempted_close_per_env = attempted_close_mask.mean(axis=0)  # unrestricted, for reference/comparability
    near_object_step_count_per_env = near_object_mask.sum(axis=0)  # (num_envs,)

    per_env = {}
    print("\n=== Gripper-closure near-object mechanism diagnostic ===")
    for e in range(num_envs):
        near_count = int(near_object_step_count_per_env[e])
        if near_count == 0:
            frac_near_and_negative = None
            note = "env never got within 5cm of the object during the episode -- metric undefined (null), NOT 0.000."
        else:
            frac_near_and_negative = float(near_and_attempted_mask[:, e].sum() / near_count)
            note = None

        per_env[str(e)] = {
            "frac_steps_raw_action_negative_near_object": frac_near_and_negative,
            "near_object_step_count": near_count,
            "frac_steps_raw_action_negative_unrestricted": float(frac_steps_attempted_close_per_env[e]),
            "ee_object_distance_min": float(dist_np[:, e].min()),
            "ee_object_distance_max": float(dist_np[:, e].max()),
            "raw_action_min": float(raw_np[:, e].min()),
            "raw_action_max": float(raw_np[:, e].max()),
        }

        metric_str = "null" if frac_near_and_negative is None else f"{frac_near_and_negative:.3f}"
        print(
            f"  env {e}: frac_steps_raw_action_negative_near_object={metric_str} "
            f"near_object_step_count={near_count}/{num_steps} "
            f"frac_steps_raw_action_negative_unrestricted={frac_steps_attempted_close_per_env[e]:.3f} "
            f"ee_object_distance_min={dist_np[:, e].min():.4f}"
        )
        if note is not None:
            print(f"    NOTE: {note}")

    summary = {
        "checkpoint": args_cli.checkpoint,
        "variant": args_cli.variant,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "near_object_threshold_m": NEAR_OBJECT_THRESHOLD_M,
        "per_env": per_env,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    raw_path = os.path.join(OUTPUT_DIR, "raw_arrays.npz")
    np.savez(
        raw_path,
        raw_gripper_action=raw_np,
        ee_object_distance=dist_np,
        object_z=object_z_np,
    )
    print(f"Raw arrays written to: {raw_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

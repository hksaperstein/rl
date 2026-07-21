"""Diagnostic (not a training/eval-suite script): headless per-step rollout
instrumentation for root-causing WHY the antipodal grasp-quality reward
(`AntipodalGraspRewardsCfg.antipodal_grasp_quality`, `tasks/franka/mdp.py`'s
`antipodal_grasp_bonus`) regresses to exactly 0.0 under joint-space control
(`FrankaDieLiftJointD8BigAntipodalEnvCfg`, Condition A / H_joint) while
producing a real, sustained signal in 1 of 3 seeds under task-space control
(`FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`, Condition B / H_taskspace) -
see `kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s already-CLOSED
H_joint/H_taskspace result and its "Root cause investigation" follow-up
section (2026-07-20) this script was built to support.

Loads a trained checkpoint (any of the three conditions) and rolls out N envs for one
full episode, recording every step. Condition-relative support added per Task 2 of
`docs/superpowers/plans/2026-07-20-d8-relative-joint-action-implementation.md`:
  - both jaws' raw contact-force vectors (`panda_leftfinger_contact`/
    `panda_rightfinger_contact` ContactSensorCfg, the exact
    `force_matrix_w.view(N, 3)` reshape Task 1 of the antipodal-grasp-
    quality plan already empirically confirmed correct for this scene).
  - the antipodal condition's own two sub-conditions (`magnitude_ok`,
    `antipodal_ok`) plus the raw `cos_angle` between the two jaws' force
    directions - computed via the exact same pure-torch
    `antipodal_grasp_bonus_raw` function training itself uses
    (`tasks/franka/antipodal_grasp_reward.py`), reimplemented at the
    cos_angle level only far enough to expose the intermediate value the
    training-time function itself does not return (it only returns the
    final 0/1 bonus).
  - every `AntipodalGraspRewardsCfg` term's raw (pre-weight) value, computed
    by calling each term's own `mdp` function directly against the live env
    with its own exact `lift_env_cfg.py`-configured params - the same
    direct-call pattern this repo's own
    `scripts/smoke_test_graspgoal_ground_penalty.py` already established
    for reading a reward term's raw value outside the RewardManager's own
    internal weighted-sum bookkeeping.
  - the `ee_frame` FrameTransformer's own position/orientation
    (`target_pos_w`/`target_quat_w`), anchored at `panda_hand` - present in
    both conditions' scene cfg (`FrankaDieLiftContactSceneCfg` extends
    `FrankaLiftSceneCfg`, which already carries `ee_frame`), used both as a
    step-to-step position/orientation jitter proxy (candidate action-space-
    geometry/exploration-precision mechanism) and as the antipodal
    condition's own approach-pose context.
  - the object's root position (`root_pos_w`).
  - the policy's own raw action tensor for that step.

Saves ONE `.npz` per (variant, checkpoint) invocation - all per-step/
per-env arrays, shape `(num_steps, num_envs, ...)` - plus a compact JSON
summary of the derived aggregate stats (contact frequency, antipodal-
satisfying fraction, per-term mean reward, EE step-to-step position/
orientation jitter) written next to it. Cross-checkpoint/cross-variant
comparison happens OFFLINE afterward (plain numpy/json, no Isaac Sim
needed) - this script itself only collects data for one checkpoint at a
time (this repo's own established "one ManagerBasedRLEnv per process"
limitation - see `tasks/franka/dice_lift_joint_env_cfg.py`'s
`FrankaDieLiftJointD12D20MixedEnvCfg` docstring for the confirmed repro of
why a second env can't be built in the same process).

No `RecordVideo`/`--enable_cameras` needed (unlike
`scripts/franka_checkpoint_review.py`) - this is pure data collection,
headless-friendly by default.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/diag_antipodal_root_cause.py \
        --variant condition-a --checkpoint <path/to/model_N.pt> --num_envs 64 \
        --output_npz logs/diag_antipodal/condition-a_iterN.npz --headless"

(Cloud dispatch runs `--headless` per this project's own standing cloud
exception.)
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-step contact/reward/pose instrumentation for a trained antipodal checkpoint.")
parser.add_argument("--variant", choices=["condition-a", "condition-b", "condition-relative"], required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="Steps to roll out. Default: one full episode (max_episode_length - 1, avoiding the "
    "known auto-reset-teleport boundary sample - see franka_checkpoint_review.py's own "
    "analysis-window fix, commit 977a748, for the identical off-by-one this default matches).",
)
parser.add_argument("--output_npz", type=str, required=True, help="Path to write the per-step .npz data to.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

from tasks.franka import mdp  # noqa: E402
from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.antipodal_grasp_reward import antipodal_grasp_bonus_raw  # noqa: E402
from tasks.franka.dice_lift_joint_env_cfg import (  # noqa: E402
    FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg_PLAY,
    FrankaDieLiftJointD8BigAntipodalEnvCfg_PLAY,
    FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY,
)
from isaaclab.managers import SceneEntityCfg  # noqa: E402

# Fixed, non-tunable for this diagnostic - identical to AntipodalGraspRewardsCfg
# (tasks/franka/lift_env_cfg.py:362-397), read directly rather than hardcoded a
# second time as literals only for the antipodal term's own params (the one term
# whose exact threshold values matter bit-for-bit for this investigation); every
# other term's params are copied from RewardsCfg (lift_env_cfg.py:280-304) verbatim.
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.894427

# (name, func, params) - matches AntipodalGraspRewardsCfg's own term list exactly
# (RewardsCfg's 6 terms + antipodal_grasp_quality), called directly against the
# live env exactly as scripts/smoke_test_graspgoal_ground_penalty.py already
# established for reading a reward term's raw (pre-weight) value outside the
# RewardManager's own internal bookkeeping - both condition-a and condition-b use
# the SAME AntipodalGraspRewardsCfg (condition-b inherits it unchanged from
# condition-a, only re-asserting self.actions.arm_action - confirmed by direct
# read of dice_lift_joint_env_cfg.py), so one shared term list is correct for
# both variants, not two.
REWARD_TERMS = [
    ("reaching_object", mdp.object_ee_distance, {"std": 0.1}),
    ("lifting_object", mdp.object_is_lifted, {"minimal_height": 0.04}),
    ("object_goal_tracking", mdp.object_goal_distance, {"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"}),
    (
        "object_goal_tracking_fine_grained",
        mdp.object_goal_distance,
        {"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    ),
    ("action_rate", mdp.action_rate_l2, {}),
    ("joint_vel", mdp.joint_vel_l2, {"asset_cfg": SceneEntityCfg("robot")}),
    (
        "antipodal_grasp_quality",
        mdp.antipodal_grasp_bonus,
        {
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
            "jaw1_contact_cfg": SceneEntityCfg("panda_leftfinger_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("panda_rightfinger_contact"),
        },
    ),
]


def main() -> None:
    if args_cli.variant == "condition-a":
        env_cfg = FrankaDieLiftJointD8BigAntipodalEnvCfg_PLAY()
    elif args_cli.variant == "condition-b":
        env_cfg = FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg_PLAY()
    else:
        env_cfg = FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_envs = args_cli.num_envs

    episode_length_steps = int(env.max_episode_length)
    num_steps = args_cli.num_steps or (episode_length_steps - 1)
    print(f"[diag] variant={args_cli.variant} num_envs={num_envs} episode_length={episode_length_steps} num_steps={num_steps}")

    # Resolve SceneEntityCfg params against this env's own scene once, matching
    # scripts/smoke_test_graspgoal_ground_penalty.py's own established precedent
    # for calling an mdp reward function directly outside the RewardManager -
    # a no-op for every SceneEntityCfg used here (none carry name-pattern
    # joint_names/body_names needing index resolution), but done anyway so this
    # script never silently relies on an unresolved default without checking.
    for _, _, params in REWARD_TERMS:
        for v in params.values():
            if isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    action_dim = env.action_manager.total_action_dim

    # Per-step/per-env recorded arrays.
    jaw1_force = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    jaw2_force = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    magnitude_ok = np.zeros((num_steps, num_envs), dtype=bool)
    antipodal_ok = np.zeros((num_steps, num_envs), dtype=bool)
    cos_angle = np.zeros((num_steps, num_envs), dtype=np.float32)
    ee_pos_w = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    ee_quat_w = np.zeros((num_steps, num_envs, 4), dtype=np.float32)
    object_pos_w = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    actions_arr = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
    reward_terms = {name: np.zeros((num_steps, num_envs), dtype=np.float32) for name, _, _ in REWARD_TERMS}

    obs = wrapped_env.get_observations()
    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)

            jaw1_sensor = env.scene["panda_leftfinger_contact"]
            jaw2_sensor = env.scene["panda_rightfinger_contact"]
            jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(num_envs, 3)
            jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(num_envs, 3)

            jaw1_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
            jaw2_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
            both_mag_ok = (jaw1_mag > FORCE_THRESHOLD) & (jaw2_mag > FORCE_THRESHOLD)
            jaw1_dir = jaw1_force_vec / (jaw1_mag.unsqueeze(-1) + 1e-8)
            jaw2_dir = jaw2_force_vec / (jaw2_mag.unsqueeze(-1) + 1e-8)
            cos = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
            both_antipodal_ok = cos < ANTIPODAL_COS_THRESHOLD

            # Cross-check against the exact training-time function (must match
            # both_mag_ok & both_antipodal_ok exactly - a mismatch here would mean
            # this script's own recomputation has drifted from the real mechanism).
            bonus_check = antipodal_grasp_bonus_raw(jaw1_force_vec, jaw2_force_vec, FORCE_THRESHOLD, ANTIPODAL_COS_THRESHOLD)
            assert torch.equal(bonus_check.bool(), both_mag_ok & both_antipodal_ok), (
                "diag script's own antipodal recomputation diverged from antipodal_grasp_bonus_raw - do not trust this run's data"
            )

            ee_frame = env.scene["ee_frame"]
            object_entity = env.scene["object"]

            jaw1_force[step] = jaw1_force_vec.cpu().numpy()
            jaw2_force[step] = jaw2_force_vec.cpu().numpy()
            magnitude_ok[step] = both_mag_ok.cpu().numpy()
            antipodal_ok[step] = both_antipodal_ok.cpu().numpy()
            cos_angle[step] = cos.cpu().numpy()
            ee_pos_w[step] = ee_frame.data.target_pos_w[:, 0, :].cpu().numpy()
            ee_quat_w[step] = ee_frame.data.target_quat_w[:, 0, :].cpu().numpy()
            object_pos_w[step] = object_entity.data.root_pos_w.cpu().numpy()
            actions_arr[step] = actions.cpu().numpy()

            for name, func, params in REWARD_TERMS:
                reward_terms[name][step] = func(env, **params).detach().cpu().numpy()

    env.close()

    # --- Derived aggregate stats (also printed for a fast sanity check without
    # needing to load the .npz separately) ---
    contact_freq = float(magnitude_ok.mean())  # fraction of (step, env) with both jaws above force_threshold
    antipodal_freq = float((magnitude_ok & antipodal_ok).mean())  # fraction satisfying the FULL bonus condition
    # cos_angle distribution restricted to steps where contact is already happening
    # (magnitude_ok) - the geometric-precision-at-contact question this
    # investigation cares about, not diluted by the vast majority of no-contact steps.
    cos_at_contact = cos_angle[magnitude_ok] if magnitude_ok.any() else np.array([])

    ee_pos_step_diff = np.linalg.norm(np.diff(ee_pos_w, axis=0), axis=-1)  # (num_steps-1, num_envs)
    # quaternion angular step-to-step difference via |1 - |dot(q_t, q_t+1)||, small-angle safe
    q_dot = np.sum(ee_quat_w[:-1] * ee_quat_w[1:], axis=-1)
    ee_ang_step_diff = np.arccos(np.clip(np.abs(q_dot), 0.0, 1.0)) * 2.0  # radians

    summary = {
        "variant": args_cli.variant,
        "checkpoint": args_cli.checkpoint,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "contact_frequency": contact_freq,
        "antipodal_satisfying_frequency": antipodal_freq,
        "fraction_of_contact_steps_that_are_antipodal": (antipodal_freq / contact_freq) if contact_freq > 0 else None,
        "cos_angle_at_contact_mean": float(cos_at_contact.mean()) if cos_at_contact.size else None,
        "cos_angle_at_contact_std": float(cos_at_contact.std()) if cos_at_contact.size else None,
        "cos_angle_at_contact_min": float(cos_at_contact.min()) if cos_at_contact.size else None,
        "ee_pos_step_diff_mean_m": float(ee_pos_step_diff.mean()),
        "ee_pos_step_diff_std_m": float(ee_pos_step_diff.std()),
        "ee_ang_step_diff_mean_rad": float(ee_ang_step_diff.mean()),
        "ee_ang_step_diff_std_rad": float(ee_ang_step_diff.std()),
        "reward_term_means": {name: float(reward_terms[name].mean()) for name, _, _ in REWARD_TERMS},
    }

    print("\n=== diag_antipodal_root_cause summary ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_npz)), exist_ok=True)
    np.savez_compressed(
        args_cli.output_npz,
        jaw1_force=jaw1_force,
        jaw2_force=jaw2_force,
        magnitude_ok=magnitude_ok,
        antipodal_ok=antipodal_ok,
        cos_angle=cos_angle,
        ee_pos_w=ee_pos_w,
        ee_quat_w=ee_quat_w,
        object_pos_w=object_pos_w,
        actions=actions_arr,
        **{f"reward_{name}": reward_terms[name] for name, _, _ in REWARD_TERMS},
    )
    summary_path = os.path.splitext(args_cli.output_npz)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] npz written to: {args_cli.output_npz}")
    print(f"[diag] summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

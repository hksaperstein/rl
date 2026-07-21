# scripts/diag_ar4_antipodal_root_cause.py
"""Diagnostic (not a training/eval-suite script): headless per-step rollout
instrumentation for root-causing WHY AR4's own antipodal grasp-quality gate
(`tasks/ar4.mdp.antipodal_grasp_bonus`, already wired into
`Ar4PickPlaceGraspGoalEnvCfg`'s `RewardsCfg.grasp_goal_milestone_bonus`/
`TerminationsCfg.cube_reached_goal`/`ObservationsCfg.grasp_state`) never
converts into a real grasp+lift under Experiment 26's historical
absolute-joint-space action term (`cube_reached_goal 0.0000`), and whether
`RelativeJointPositionActionCfg` (Task 2's `Ar4PickPlaceGraspGoalRelativeEnvCfg`)
changes that - see `docs/superpowers/plans/2026-07-21-ar4-franka-fixes-
transfer-implementation.md` (Task 3) and its spec,
`docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`.

Adapts `scripts/diag_antipodal_root_cause.py`'s methodology (built for the
analogous Franka d8 investigation) to AR4's own scene/entity names and both
AR4 conditions, per this project's own "cross-check against the exact
training-time function, don't reimplement" discipline (Design decision 2 of
the spec: AR4's `antipodal_grasp_bonus` is already the correct, unmodified,
already-verified mechanism - `mu=1.0` -> `ANTIPODAL_COS_THRESHOLD=-0.7071`,
re-confirmed still physically correct, NOT Franka's refit `-0.894427`).

Per-step recorded arrays / cross-checks:
  - both jaws' raw contact-force vectors (`gripper_jaw1_contact`/
    `gripper_jaw2_contact` `ContactSensorCfg`, `force_matrix_w.view(N, 3)`
    - the same shape this scene's own reward/observation code already
      relies on).
  - `magnitude_ok`/`antipodal_ok`/`cos_angle`, recomputed directly at the
    per-step/per-env level (the training-time `antipodal_grasp_bonus`
    function only returns the final 0/1 bonus, not the intermediate
    magnitude/direction values this investigation needs), then
    cross-checked via `assert` against `ar4_mdp.antipodal_grasp_bonus`'s
    own returned tensor every single step - a mismatch here means this
    script's own recomputation has drifted from the real mechanism and the
    run's data should not be trusted.
  - whether each env has EVER achieved the full grasp+lift latch
    (`env._lifted`, the same stateful buffer `grasp_goal_milestone_bonus`/
    `cube_reached_goal_after_lift` both read) - the piece of evidence
    signature 3 (contact-and-antipodal-but-no-lift) needs, folded directly
    into this diagnostic's own output rather than requiring a second,
    separate pass over TensorBoard scalars just to tell signature 2 from
    signature 3.
  - the `ee_frame` FrameTransformer's `target_pos_w`/`target_quat_w`, the
    cube's `root_pos_w`, the policy's raw action tensor.
  - every `RewardsCfg` term's raw (pre-weight) value.

**Deliberate deviation from `diag_antipodal_root_cause.py`'s own direct-call
pattern for reading reward-term values - a real correctness issue found
while porting, fixed here rather than copied blindly**: Franka's script
reads every `AntipodalGraspRewardsCfg` term's raw value by calling the term's
own `mdp` function a SECOND time, directly, after `env.step()` already ran
it once via the RewardManager. This is safe for Franka's own reward terms
(all dense/stateless: `object_ee_distance`, `object_is_lifted`,
`object_goal_distance`, `action_rate_l2`, `joint_vel_l2`,
`antipodal_grasp_bonus` - none mutate `env` state depending on call count).
AR4's `RewardsCfg.grasp_goal_milestone_bonus` (`tasks/ar4/mdp.py:537`) is
NOT stateless: it is an undiscounted running-max bonus
(`env._grasp_goal_milestone_max = torch.maximum(env._grasp_goal_milestone_max,
raw); return env._grasp_goal_milestone_max - prev`). Calling it a second
time in the same step, after the RewardManager's own internal call already
updated the running max, would recompute an IDENTICAL `raw` (no time has
passed) against an ALREADY-UPDATED max, so the second call's returned diff
is always exactly 0.0 - silently corrupting this diagnostic's own reading of
the single most important reward term for this investigation into a
constant zero, regardless of what the policy actually earned that step.
Fix: read `env.reward_manager._step_reward[:, term_idx] / term_weight`
instead - the RewardManager's own already-computed per-step, per-term
weighted value (populated once, internally, during `env.step()`'s own
`RewardManager.compute()` call; see `isaaclab.managers.reward_manager.
RewardManager.compute`), divided back out by the term's configured
`weight` to recover the same "raw, pre-weight" semantics the Franka script
reports. This is strictly more faithful than a direct second call for EVERY
term (zero risk of the diagnostic's own recomputation ever drifting from
what the RewardManager actually used, not just for the one stateful term),
so it is used uniformly for all 5 `RewardsCfg` terms, not only the
stateful one.

No `RecordVideo`/`--enable_cameras` needed - pure data collection,
headless-friendly by default. Saves ONE `.npz` + summary JSON per
(variant, checkpoint) invocation, matching the Franka script's own schema
plus the `ever_lifted_frequency` addition above.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/diag_ar4_antipodal_root_cause.py \
        --variant condition-a --checkpoint <path/to/model_N.pt> --num_envs 64 \
        --output_npz logs/diag_ar4_antipodal/condition-a_iterN.npz --headless"

(Cloud dispatch runs `--headless` per this project's own standing cloud
exception.)
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-step contact/reward/pose instrumentation for a trained AR4 grasp-goal checkpoint.")
parser.add_argument(
    "--variant",
    choices=["condition-a", "condition-b"],
    required=True,
    help="condition-a: Ar4PickPlaceGraspGoalEnvCfg (Experiment 26, absolute joint-space). "
    "condition-b: Ar4PickPlaceGraspGoalRelativeEnvCfg (H_ar4_relative, relative/delta joint-space). "
    "Deliberately named to match the spec's own condition-a/condition-b naming rather than "
    "scripts/train.py's own flat --graspgoal/--graspgoalrelative boolean-flag convention - a "
    "judgment call scoped to this diagnostic script only (see Task 3 of the implementation plan).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="Steps to roll out. Default: one full episode (max_episode_length - 1, avoiding the "
    "known auto-reset-teleport boundary sample - see franka_checkpoint_review.py's own "
    "analysis-window fix, commit 977a748, for the identical off-by-one this default matches). "
    "NOT truncated by default despite AR4's 30s/1500-step episode being 6x Franka's own 5s/250 "
    "steps - the falsification bar's own trajectory-shape methodology depends on measuring real, "
    "full-episode contact-frequency behavior, not a partial window (see the implementation plan's "
    "Task 3 Files section).",
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
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import (  # noqa: E402
    ANTIPODAL_COS_THRESHOLD,
    FORCE_THRESHOLD,
    Ar4PickPlaceGraspGoalEnvCfg,
    Ar4PickPlaceGraspGoalRelativeEnvCfg,
)

# Read directly from the env-cfg module rather than duplicated as a second
# set of literals (an improvement on the Franka script's own literal copy -
# see that script's own comment acknowledging these must stay bit-for-bit
# in sync with RewardsCfg's real params; importing removes any possibility
# of drift). Re-verified physically correct for AR4's real mu=1.0 friction
# by the design spec's own Design decision 2 - no refit needed, unlike
# Franka's own mu=0.5-derived -0.894427.
assert FORCE_THRESHOLD == 0.05
assert ANTIPODAL_COS_THRESHOLD == -0.7071

# RewardsCfg's own term list, in the SAME order as pickplace_graspgoal_env_cfg.py's
# RewardsCfg class body (the order RewardManager._term_names/_step_reward columns
# follow, since it iterates the cfg's __dict__ in definition order) - used to look
# up each term's per-step weighted value directly from the RewardManager's own
# internal buffer (see this module's docstring for why, not via a second direct
# call to the term's own mdp function).
REWARD_TERM_NAMES = [
    "grasp_goal_milestone_bonus",
    "action_rate",
    "joint_vel",
    "arm_ground_contact_penalty",
    "slow_near_cube_bonus",
]


def main() -> None:
    if args_cli.variant == "condition-a":
        env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    else:
        env_cfg = Ar4PickPlaceGraspGoalRelativeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Both conditions use the SAME Ar4PickPlacePPORunnerCfg today - no
    # per-condition runner-cfg split exists yet (unlike Franka's own
    # condition-relative -> FrankaLiftRelativeJointPPORunnerCfg branch in
    # diag_antipodal_root_cause.py). Design note 4 of the implementation
    # plan flags this as an EXPECTED future one-line update: if Task 5's
    # real 1500-iteration Condition B run hits the critic-divergence
    # contingency (Design note 3) and a scoped
    # Ar4PickPlaceGraspGoalRelativePPORunnerCfg is added, this branch must
    # select it for --variant condition-b specifically (matching whatever
    # clip_actions the checkpoint was actually trained under) before Task 6
    # runs its real measurement sweep against those checkpoints.
    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_envs = args_cli.num_envs

    episode_length_steps = int(env.max_episode_length)
    num_steps = args_cli.num_steps or (episode_length_steps - 1)
    print(f"[diag] variant={args_cli.variant} num_envs={num_envs} episode_length={episode_length_steps} num_steps={num_steps}")

    jaw1_contact_cfg = SceneEntityCfg("gripper_jaw1_contact")
    jaw2_contact_cfg = SceneEntityCfg("gripper_jaw2_contact")
    jaw1_contact_cfg.resolve(env.scene)
    jaw2_contact_cfg.resolve(env.scene)

    # Resolve each RewardsCfg term's index into RewardManager._step_reward's
    # column axis, and its configured weight (to divide the weighted
    # per-step value back down to "raw, pre-weight" - matching the Franka
    # script's own reported semantics). Done once, up front - not per-step.
    reward_term_idx = {}
    reward_term_weight = {}
    for name in REWARD_TERM_NAMES:
        reward_term_idx[name] = env.reward_manager.active_terms.index(name)
        weight = env.reward_manager.get_term_cfg(name).weight
        assert weight != 0.0, f"reward term '{name}' has weight=0.0 - raw-value division would divide by zero"
        reward_term_weight[name] = weight

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
    ever_lifted = np.zeros((num_steps, num_envs), dtype=bool)
    ee_pos_w = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    ee_quat_w = np.zeros((num_steps, num_envs, 4), dtype=np.float32)
    cube_pos_w = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    actions_arr = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
    reward_terms = {name: np.zeros((num_steps, num_envs), dtype=np.float32) for name in REWARD_TERM_NAMES}

    obs = wrapped_env.get_observations()
    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)

            jaw1_sensor = env.scene["gripper_jaw1_contact"]
            jaw2_sensor = env.scene["gripper_jaw2_contact"]
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
            # this script's own recomputation has drifted from the real mechanism
            # this scene's reward/observation/termination chain actually uses).
            bonus_check = ar4_mdp.antipodal_grasp_bonus(
                env, FORCE_THRESHOLD, ANTIPODAL_COS_THRESHOLD, jaw1_contact_cfg, jaw2_contact_cfg
            )
            assert torch.equal(bonus_check.bool(), both_mag_ok & both_antipodal_ok), (
                "diag script's own antipodal recomputation diverged from ar4_mdp.antipodal_grasp_bonus - "
                "do not trust this run's data"
            )

            ee_frame = env.scene["ee_frame"]
            cube_entity = env.scene["cube"]

            jaw1_force[step] = jaw1_force_vec.cpu().numpy()
            jaw2_force[step] = jaw2_force_vec.cpu().numpy()
            magnitude_ok[step] = both_mag_ok.cpu().numpy()
            antipodal_ok[step] = both_antipodal_ok.cpu().numpy()
            cos_angle[step] = cos.cpu().numpy()
            # env._lifted is populated by _grasp_lift_state, called (at
            # least) by the grasp_goal_milestone_bonus reward term that the
            # RewardManager.compute() call inside wrapped_env.step() above
            # already ran this step - guaranteed to exist by this point.
            ever_lifted[step] = env._lifted.cpu().numpy()
            ee_pos_w[step] = ee_frame.data.target_pos_w[:, 0, :].cpu().numpy()
            ee_quat_w[step] = ee_frame.data.target_quat_w[:, 0, :].cpu().numpy()
            cube_pos_w[step] = cube_entity.data.root_pos_w.cpu().numpy()
            actions_arr[step] = actions.cpu().numpy()

            for name in REWARD_TERM_NAMES:
                idx = reward_term_idx[name]
                weight = reward_term_weight[name]
                reward_terms[name][step] = (env.reward_manager._step_reward[:, idx] / weight).cpu().numpy()

    env.close()

    # --- Derived aggregate stats (also printed for a fast sanity check without
    # needing to load the .npz separately) ---
    contact_freq = float(magnitude_ok.mean())  # fraction of (step, env) with both jaws above force_threshold
    antipodal_freq = float((magnitude_ok & antipodal_ok).mean())  # fraction satisfying the FULL bonus condition
    # cos_angle distribution restricted to steps where contact is already happening
    # (magnitude_ok) - the geometric-precision-at-contact question this
    # investigation cares about, not diluted by the vast majority of no-contact steps.
    cos_at_contact = cos_angle[magnitude_ok] if magnitude_ok.any() else np.array([])

    # ever_lifted is a monotonic per-env latch (env._lifted only ever goes
    # False->True, never resets mid-rollout - reset_grasp_goal_milestone only
    # runs on env reset, which doesn't happen mid-rollout here since num_steps
    # defaults to just under one full episode) - the fraction of envs that
    # EVER achieved a genuine antipodal grasp + lift during this rollout is
    # exactly ever_lifted's own final-step value, but computed via .any(axis=0)
    # for robustness against num_steps not covering the full episode.
    ever_lifted_frequency = float(ever_lifted.any(axis=0).mean()) if num_steps > 0 else 0.0

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
        "ever_lifted_frequency": ever_lifted_frequency,
        "cos_angle_at_contact_mean": float(cos_at_contact.mean()) if cos_at_contact.size else None,
        "cos_angle_at_contact_std": float(cos_at_contact.std()) if cos_at_contact.size else None,
        "cos_angle_at_contact_min": float(cos_at_contact.min()) if cos_at_contact.size else None,
        "ee_pos_step_diff_mean_m": float(ee_pos_step_diff.mean()),
        "ee_pos_step_diff_std_m": float(ee_pos_step_diff.std()),
        "ee_ang_step_diff_mean_rad": float(ee_ang_step_diff.mean()),
        "ee_ang_step_diff_std_rad": float(ee_ang_step_diff.std()),
        "reward_term_means": {name: float(reward_terms[name].mean()) for name in REWARD_TERM_NAMES},
    }

    print("\n=== diag_ar4_antipodal_root_cause summary ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_npz)), exist_ok=True)
    np.savez_compressed(
        args_cli.output_npz,
        jaw1_force=jaw1_force,
        jaw2_force=jaw2_force,
        magnitude_ok=magnitude_ok,
        antipodal_ok=antipodal_ok,
        cos_angle=cos_angle,
        ever_lifted=ever_lifted,
        ee_pos_w=ee_pos_w,
        ee_quat_w=ee_quat_w,
        cube_pos_w=cube_pos_w,
        actions=actions_arr,
        **{f"reward_{name}": reward_terms[name] for name in REWARD_TERM_NAMES},
    )
    summary_path = os.path.splitext(args_cli.output_npz)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] npz written to: {args_cli.output_npz}")
    print(f"[diag] summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

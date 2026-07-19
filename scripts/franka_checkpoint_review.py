"""Record a video of a trained Franka Panda lift checkpoint - a fresh,
from-scratch eval/demo entry point for tasks/franka/ (does NOT reuse/extend
tasks/ar4's own closeup-video scripts, e.g. scripts/graspgoal_closeup_video.py,
per the franka-panda-pivot's "everything new" instruction).

Simpler mechanism than the AR4-era closeup scripts: instead of a dedicated
per-env close-up FrameTransformer/camera sensor, this just uses
gym.wrappers.RecordVideo on the environment's own built-in viewport render
(render_mode="rgb_array") - the same mechanism scripts/train_franka.py's
own --video flag already uses - to capture one continuous video covering
all envs for one full episode. Good enough for "let's look at what this
checkpoint actually does" without needing a bespoke camera setup.

Supports two task variants via --variant (default ik-cube = original,
unchanged behavior):
  - ik-cube: stock relative-IK cube-lift (tasks/franka/lift_env_cfg.py).
  - joint-die: joint-space (no-IK) d20-die-lift
    (tasks/franka/dice_lift_joint_env_cfg.py, see
    docs/superpowers/plans/2026-07-11-joint-space-die-lift.md Task 4).
Both variants use the same FrankaLiftPPORunnerCfg (the experiment pins
everything except action space and object).

The built-in viewport render camera is repositioned (via the env cfg's
ViewerCfg) so env_0's whole arm (base to gripper) plus the table workspace
is in frame - not just the object region (standing user instruction on
video framing).

Also dumps an instrumented per-step object-height readout (env.unwrapped.
scene["object"].data.root_pos_w[:, 2]) to a .npy array + .json summary next
to the video, since a video alone can misrepresent contact/lift state
(Experiment 16 precedent: an apparent "lift" was actually the object wedged
against the wrist).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/franka_checkpoint_review.py \
        --checkpoint logs/train_franka/2026-07-09_22-05-51/model_800.pt --num_envs 8

    /home/saps/IsaacLab/isaaclab.sh -p scripts/franka_checkpoint_review.py \
        --variant joint-die --checkpoint logs/train_franka_jointdie/<ts>/model_1499.pt --num_envs 8
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of a trained Franka Panda lift checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs to render in the video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help=(
        "Video length in steps (default: two full episodes, 10s @ 50Hz - standing user rule 2026-07-12: "
        "eval videos must run LONGER than the event they document, with lead-in and aftermath on film)."
    ),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write the video to (default: logs/videos/franka_checkpoint_review/).",
)
parser.add_argument(
    "--variant",
    choices=[
        "ik-cube",
        "joint-die",
        "joint-cube",
        "joint-die-heavy",
        "joint-die-big",
        "joint-cube-baked",
        "joint-die-mixed",
        "joint-die-mid",
        "joint-die-d8-std",
        "joint-die-d10-std",
        "joint-die-d12-std",
        "joint-die-random-size",
        "joint-die-d8-big",
        "joint-die-d10-big",
        "joint-die-d12-big",
    ],
    default="ik-cube",
    help=(
        "ik-cube: the existing stock-recipe cube-lift with relative-IK actions (default, unchanged). "
        "joint-die: d20-die lift with direct joint-position actions (no IK) - see "
        "docs/superpowers/plans/2026-07-11-joint-space-die-lift.md. "
        "joint-cube: the plan's asset-vs-recipe fallback rung - same joint-position action space as "
        "joint-die, but with the recipe's own DexCube kept as the object (isolates the die asset "
        "as the variable). "
        "joint-die-heavy: asset-bisect rung 1 - the d20 at DexCube's measured 0.216kg mass "
        "(docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-die-big: asset-bisect rung 2 - the d20 scaled to DexCube's measured 48.0mm size, "
        "mass pinned at 0.216kg (docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-cube-baked: asset-bisect rung 3 - a flat-faced cube baked through this repo's own "
        "bake_die_asset.py pipeline at 48.0mm/0.216kg, isolating shape from pipeline provenance "
        "(docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-die-mixed: size-curriculum primary arm - per-env d20 size varied across "
        "{48.0,43.6,39.1,34.7,30.3}mm (deterministic round-robin), mass pinned 0.216kg "
        "(docs/superpowers/specs/2026-07-13-size-curriculum-design.md); _PLAY probe is a single "
        "all-30.3mm size. "
        "joint-die-mid: size-curriculum staged-anneal fallback, stage 2 - the d20 at 39.1mm, mass "
        "pinned at 0.216kg (docs/superpowers/specs/2026-07-13-size-curriculum-design.md Verdict section). "
        "joint-die-d8-std: multi-die specialist (Task 2) - physics-baked d8 die at its real standard "
        "~16mm size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-die-"
        "specialist-distillation.md); _PLAY probe is the same fixed size, 50 envs. "
        "joint-die-d10-std: multi-die specialist (Task 2) - physics-baked d10 die at its real standard "
        "~16mm face-to-face size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-"
        "die-specialist-distillation.md); _PLAY probe is the same fixed size, 50 envs. "
        "joint-die-d12-std: multi-die specialist (Task 2) - physics-baked d12 die at its real standard "
        "~18mm face-to-face size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-"
        "die-specialist-distillation.md); _PLAY probe is the same fixed size, 50 envs. "
        "joint-die-random-size: d20 size-DR + geometry-feature retry (Task 3) - per-env d20 size fixed at "
        "scene-spawn time via MultiAssetSpawnerCfg(random_choice=True) across {22.0,28.5,35.0,41.5,48.0}mm, "
        "mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-"
        "distillation.md); _PLAY probe is a single all-30.3mm-equivalent-scale UsdFileCfg, 50 envs "
        "(same pattern as joint-die-mixed's own _PLAY probe). "
        "joint-die-d8-big/d10-big/d12-big: 48mm-parity gate (Task 3.5) - physics-baked d8/d10/d12 die each "
        "scaled to its OWN freshly-derived 48.0mm-targeting scale (NOT the d20 Big rung's 0.001585 constant, "
        "which does not transfer across shapes), single undiluted 48mm population per shape/seed, mass pinned "
        "at 0.216kg - directly comparable to the asset-bisect's own cube (3/3) and d20 (1/3) 48mm baselines "
        "(docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md, "
        ".superpowers/sdd/task-3.5-brief.md); _PLAY probe is the same fixed size, 50 envs."
    ),
)
parser.add_argument(
    "--eval_scale",
    type=float,
    default=None,
    help=(
        "Override the resolved env cfg's scene.object.spawn.scale (isotropic, same value on all 3 axes) - "
        "lets a single-UsdFileCfg _PLAY variant be evaluated at a size other than its own hardcoded default, "
        "e.g. sweeping joint-die-random-size's 5 training sizes {0.000727,0.000941,0.001156,0.001370,0.001585} "
        "(22.0/28.5/35.0/41.5/48.0mm) one eval run at a time (Task 3, docs/superpowers/plans/2026-07-16-"
        "unified-multi-die-specialist-distillation.md). Only valid when scene.object.spawn is a single "
        "UsdFileCfg (every _PLAY class in this repo uses one); errors loudly otherwise."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video rendering

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.lift_env_cfg import FrankaLiftEnvCfg_PLAY  # noqa: E402

if args_cli.variant == "joint-die":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-cube":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeLiftJointEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-heavy":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-big":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointBigEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-cube-baked":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeBakedLiftJointEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-mixed":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointMixedEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-mid":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointMidEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d8-std":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8StandardEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d10-std":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD10StandardEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d12-std":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12StandardEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-random-size":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointRandomSizeEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d8-big":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d10-big":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD10BigEnvCfg_PLAY  # noqa: E402
elif args_cli.variant == "joint-die-d12-big":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12BigEnvCfg_PLAY  # noqa: E402

VIDEO_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "franka_checkpoint_review"
)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Stock lift recipe's own "lifted" height threshold (matches
# mdp.object_is_lifted's minimal_height param in both variants' RewardsCfg -
# tasks/franka/lift_env_cfg.py - not a new value invented for this script).
LIFT_HEIGHT_THRESHOLD_M = 0.04
# 0.5s sustained at decimation=2, sim.dt=0.01 -> 50 control steps/s.
SUSTAINED_LIFT_STEPS = 25
# THIRD AND FOURTH settle-detection bugs, found and fixed during the
# d20-big-geom undiluted-48mm gate task (2026-07-19, closing Task 3's
# dilution ambiguity). The prior mechanism (977a748: a forward scan for the
# first window of SETTLE_STABLE_WINDOW=15 consecutive steps with range <
# SETTLE_STABLE_EPS_M=5e-5m) turned out to be fundamentally fragile for
# this env cfg's fast, decisive grasps, in two successive ways found by
# direct per-step inspection of joint-die-big seed42/123/7's raw .npy:
# (a) 5e-5m (0.05mm) is tighter than genuine PhysX table-rest jitter ever
# gets here (measured 15-step window-range floor during true table-rest:
# ~0.0010-0.0024m across 24 env/seed combinations checked - 20-45x above
# 5e-5m, not a one-off). The forward scan then walked straight past the
# true (brief, ~10-20-step) table-rest phase and silently locked onto a
# much LATER, fully-static, arm-stopped-moving plateau near the episode's
# end instead (settle_step values of 70-230, not ~15-25) - with NO warning
# printed, since a window WAS found (just the wrong one), unlike the
# already-flagged settle_step=-1 fallback case.
# (b) Loosening the tolerance to 2e-3m (2mm) - chosen as a first attempt
# to sit above the observed rest-jitter ceiling - was STILL insufficient:
# a few envs' true-rest window-range floor measured as high as 0.0024m,
# marginally above 2e-3m, so the same silent-wrong-plateau failure still
# fired for a subset of envs. Root problem: a single global flatness
# epsilon cannot cleanly separate "resting" from "slow asymptotic approach
# to a held plateau" for this trajectory shape - the window-range shrinks
# smoothly as the object's carried position converges, so real motion and
# real rest overlap in range at some point along the curve regardless of
# where the epsilon is set.
# FIX: abandoned the flatness-window heuristic entirely in favor of a
# simpler, more robust, physically-grounded MIN over a fixed early window
# (EARLY_SETTLE_START:EARLY_SETTLE_END below) - the true table-rest height
# is *by construction* the global minimum of the trajectory in the window
# after free-fall begins and before any grasp-driven ascent could
# plausibly complete (ascent only moves the object UP from its rest
# height, never below it, so MIN is insensitive to exactly where within
# the window the low point falls, unlike a flatness test). Window bounds
# were set from direct measurement across all 24 env/seed combinations in
# this task's own data: genuine table-rest is reliably present within
# steps ~10-26 (spawn-to-table free-fall always completes by ~step 10,
# matching this project's own established ~24-25-step worst-case settle
# time with margin), and the earliest observed grasp-driven ascent onset
# is ~step 24-30 - EARLY_SETTLE_END=45 sits comfortably past the former
# and doesn't need to sit before the latter (MIN over a window that
# partially overlaps early ascent still correctly reports the window's
# true minimum). Verified directly: this MIN-based approach reproduces a
# clean, consistent ~0.0188-0.0192m resting_z for every genuinely-resting
# env across all 3 seeds (vs. the flatness approach's spurious
# 0.26-0.46m readings for the same envs) and correctly reads 0/8, 8/8,
# 8/8 for this task's seed 42/123/7 - see
# kb/wiki/experiments/unified-multi-die-specialist-distillation.md's
# "d20-big-geom" entry for the full raw-trajectory numbers this was
# derived from. Not re-validated against Task 3.5's own already-reported
# d8-big/d10-big/d12-big numbers (out of this task's scope - flagged to
# BACKLOG.md as a follow-up: those may also undercount true positives,
# since the old flatness approach was never reliable for ANY env cfg
# checked so far, not just this one).
EARLY_SETTLE_START = 10
EARLY_SETTLE_END = 45


def _max_consecutive_true(mask: list[bool]) -> int:
    """Longest run of consecutive True values in a 1D bool sequence."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    if args_cli.variant == "joint-die":
        env_cfg = FrankaDieLiftJointEnvCfg_PLAY()
    elif args_cli.variant == "joint-cube":
        env_cfg = FrankaCubeLiftJointEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-heavy":
        env_cfg = FrankaDieLiftJointHeavyEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-big":
        env_cfg = FrankaDieLiftJointBigEnvCfg_PLAY()
    elif args_cli.variant == "joint-cube-baked":
        env_cfg = FrankaCubeBakedLiftJointEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-mixed":
        env_cfg = FrankaDieLiftJointMixedEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-mid":
        env_cfg = FrankaDieLiftJointMidEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d8-std":
        env_cfg = FrankaDieLiftJointD8StandardEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d10-std":
        env_cfg = FrankaDieLiftJointD10StandardEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d12-std":
        env_cfg = FrankaDieLiftJointD12StandardEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-random-size":
        env_cfg = FrankaDieLiftJointRandomSizeEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d8-big":
        env_cfg = FrankaDieLiftJointD8BigEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d10-big":
        env_cfg = FrankaDieLiftJointD10BigEnvCfg_PLAY()
    elif args_cli.variant == "joint-die-d12-big":
        env_cfg = FrankaDieLiftJointD12BigEnvCfg_PLAY()
    else:
        env_cfg = FrankaLiftEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    if args_cli.eval_scale is not None:
        spawn_cfg = env_cfg.scene.object.spawn
        if not hasattr(spawn_cfg, "scale"):
            sys.exit(
                f"--eval_scale requires scene.object.spawn to be a single UsdFileCfg with a .scale attribute; "
                f"got {type(spawn_cfg).__name__} for --variant {args_cli.variant}."
            )
        s = args_cli.eval_scale
        spawn_cfg.scale = (s, s, s)

    # FULL-ARM VIDEO FRAMING (standing user instruction): reposition the
    # built-in viewport render camera (the one render_mode="rgb_array"
    # captures below) so env_0's whole Franka arm (base to gripper) plus the
    # table workspace is in frame, not just the object region. Robot base
    # sits at env_0's local origin (0,0,0); table/object sit at local
    # (0.5, 0, *) - see tasks/franka/lift_env_cfg.py's FrankaLiftSceneCfg.
    # origin_type="env" anchors eye/lookat to that env's own frame
    # (isaaclab/envs/common.py's ViewerCfg + envs/ui/viewport_camera_
    # controller.py's ViewportCameraController, read directly to confirm
    # this mechanism - a static "env"-anchored view, no per-step tracking
    # needed for a single fixed env_0 shot).
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.8, 1.8, 1.1)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.35)

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    checkpoint_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    video_kwargs = {
        "video_folder": VIDEO_DIR,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": f"franka_checkpoint_review_{args_cli.variant}_{checkpoint_name}",
        "disable_logger": True,
    }
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # INSTRUMENTED HEIGHT READOUT (Experiment 16 precedent - video alone can
    # misrepresent grasp/lift state). Read the object's world z directly
    # from the RigidObject data buffer every step, independent of any
    # reward/termination signal.
    num_envs = args_cli.num_envs
    height_history = torch.zeros((args_cli.video_length, num_envs))

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.video_length):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            object_z = env.unwrapped.scene["object"].data.root_pos_w[:, 2].detach().cpu()
            height_history[step] = object_z

    env.close()
    print(f"Checkpoint review video written to: {VIDEO_DIR}")

    # MEASUREMENT-ARTIFACT FIXES. Tasks 2/3 independently traced a naive
    # max()/argmax over the full multi-episode video_length window picking up
    # a one-step reset-teleport spike whenever video_length spans more than
    # one episode; a first fix (Task 3.5, commit 1ce90a4) restricted derived
    # stats to `env.unwrapped.max_episode_length` steps ("the first episode
    # only"). Re-verifying that fix against Task 3.5's own real d8-big raw
    # data (independent per-shape/per-seed re-derivation, per this task's own
    # dispatch instruction) found TWO further, still-live bugs in it - fixed
    # here, forward-only, same as the prior fix (does not touch any
    # already-synced GCS artifact or already-written report):
    #
    # (a) OFF-BY-ONE at the episode boundary: this env's vectorized
    # auto-reset convention (isaaclab's ManagerBasedRLEnv / gym vec-env step
    # semantics) returns the POST-reset observation on the very step where an
    # episode ends, not the terminal one - i.e. index
    # `max_episode_length - 1` (0-indexed) is already episode 2's first
    # frame, not episode 1's last. Confirmed directly: for a 250-step
    # episode, height_history[249] == height_history[0] byte-for-byte across
    # every env, and height_history[250] == height_history[1] - the previous
    # `analysis_end = min(episode_length_steps, video_length)` (250) included
    # this one contaminated sample. Fixed by subtracting 1.
    #
    # (b) SETTLE WINDOW TOO SHORT, then (c)/(d) two further flatness-
    # detection failures - see EARLY_SETTLE_START/EARLY_SETTLE_END's own
    # docstring above for the full history. Current approach: resting_z is
    # the MIN of each env's own trajectory over a fixed early window
    # (EARLY_SETTLE_START:EARLY_SETTLE_END) - physically grounded (a
    # grasp-driven ascent only moves the object UP from its true rest
    # height, so MIN is correct regardless of where within the window the
    # low point falls) and empirically verified against this task's own
    # 24 env/seed raw trajectories. Derived stats (gain/lifted_mask/
    # sustained_lift/max_z) are computed from EARLY_SETTLE_START onward -
    # the pre-EARLY_SETTLE_START free-fall segment is real motion but not
    # policy-driven, and leaving it in the max_z/gain window produces false
    # positives (the object's own spawn height, ~4cm above its rest height,
    # otherwise reads as a "gain" in its own right, independent of anything
    # the policy does).
    episode_length_steps = int(env.unwrapped.max_episode_length)
    analysis_end = min(episode_length_steps - 1, args_cli.video_length)
    if analysis_end < args_cli.video_length:
        print(
            f"[analysis window] restricting derived stats to the first episode only: steps "
            f"0-{analysis_end - 1} (episode_length={episode_length_steps} steps, one boundary-"
            f"contaminated sample excluded, recording={args_cli.video_length} steps)."
        )
    analysis_history = height_history[:analysis_end]

    resting_window_end = min(EARLY_SETTLE_END, analysis_end)
    resting_z = analysis_history[EARLY_SETTLE_START:resting_window_end].amin(dim=0)

    post_settle_start = min(EARLY_SETTLE_START, analysis_end)
    post_settle_history = analysis_history[post_settle_start:]

    gain = post_settle_history - resting_z.unsqueeze(0)
    max_gain = gain.max(dim=0).values
    max_z = post_settle_history.max(dim=0).values

    lifted_mask = gain >= LIFT_HEIGHT_THRESHOLD_M  # (post-settle steps, num_envs) bool

    summary = {}
    print("\n=== Instrumented height readout (per env) ===")
    print(f"lift threshold: {LIFT_HEIGHT_THRESHOLD_M} m above settled z, sustained >= {SUSTAINED_LIFT_STEPS} steps")
    for env_idx in range(num_envs):
        run_len = _max_consecutive_true(lifted_mask[:, env_idx].tolist())
        sustained = run_len >= SUSTAINED_LIFT_STEPS
        summary[str(env_idx)] = {
            "resting_z_m": float(resting_z[env_idx]),
            "resting_window": [EARLY_SETTLE_START, resting_window_end],
            "max_z_m": float(max_z[env_idx]),
            "max_height_gain_m": float(max_gain[env_idx]),
            "max_consecutive_lifted_steps": run_len,
            "sustained_lift": bool(sustained),
        }
        print(
            f"  env {env_idx}: resting_z={resting_z[env_idx]:.4f}m (min over steps "
            f"{EARLY_SETTLE_START}-{resting_window_end}) max_z={max_z[env_idx]:.4f}m "
            f"max_gain={max_gain[env_idx]:.4f}m max_consecutive_lifted_steps={run_len} "
            f"sustained_lift={sustained}"
        )
    n_sustained = sum(1 for v in summary.values() if v["sustained_lift"])
    print(f"envs with sustained lift: {n_sustained}/{num_envs}")

    heights_npy_path = os.path.join(VIDEO_DIR, f"heights_{args_cli.variant}_{checkpoint_name}.npy")
    np.save(heights_npy_path, height_history.numpy())
    summary_json_path = os.path.join(VIDEO_DIR, f"heights_{args_cli.variant}_{checkpoint_name}.json")
    with open(summary_json_path, "w") as f:
        json.dump(
            {
                "variant": args_cli.variant,
                "checkpoint": args_cli.checkpoint,
                "num_envs": num_envs,
                "video_length_steps": args_cli.video_length,
                "episode_length_steps": episode_length_steps,
                "analysis_window_steps": analysis_end,
                "post_settle_start_step": post_settle_start,
                "lift_height_threshold_m": LIFT_HEIGHT_THRESHOLD_M,
                "sustained_lift_steps": SUSTAINED_LIFT_STEPS,
                "envs_with_sustained_lift": n_sustained,
                "per_env": summary,
            },
            f,
            indent=2,
        )
    print(f"Height array written to: {heights_npy_path}")
    print(f"Height summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

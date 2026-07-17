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
# Steps used to estimate each env's settled/resting object height before any
# policy action has moved it (object is spawned already resting on the
# table via reset_object_position - see FrankaLiftSceneCfg's EventCfg).
SETTLE_WINDOW_STEPS = 10


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

    # MEASUREMENT-ARTIFACT FIX (Task 3.5, docs/superpowers/plans/2026-07-16-
    # unified-multi-die-specialist-distillation.md): Tasks 2 and 3 both
    # independently traced the SAME artifact in this script's derived stats
    # (task-2-report.md; task-3-report.md Section 7, which pinned the exact
    # mechanism) - a naive max()/argmax over the full multi-episode
    # video_length window picks up a one-step reset-teleport spike whenever
    # video_length spans more than one episode (this script's own default,
    # 500 steps, is exactly 2 episodes at this env's 250-step episode
    # length): at every episode-length boundary the object is
    # reset-teleported back to its RigidObjectCfg.InitialStateCfg spawn z
    # for one step, then free-falls and resettles identically to steps 0-4
    # of the recording - not a policy-driven event. Fixed here (3rd
    # occurrence, per this task's own dispatch instruction to stop working
    # around it) by restricting every derived stat (resting_z, gain, max_z,
    # lifted_mask, sustained_lift) to the FIRST episode only
    # (env.unwrapped.max_episode_length steps, read generically rather than
    # hardcoded, so this is correct for any variant/episode-length
    # combination, not just this env's own 250-step episode). The full raw
    # per-step array is still saved to the .npy completely unchanged -
    # nothing is lost, only the derived-stats window is narrowed. This only
    # changes this script's FORWARD behavior from this commit onward - it
    # does not touch or retroactively alter any already-synced GCS artifact
    # or already-written report from Task 2 or Task 3.
    episode_length_steps = int(env.unwrapped.max_episode_length)
    analysis_end = min(episode_length_steps, args_cli.video_length)
    if analysis_end < args_cli.video_length:
        print(
            f"[analysis window] restricting derived stats to the first episode only: steps "
            f"0-{analysis_end - 1} (episode_length={episode_length_steps} steps, "
            f"recording={args_cli.video_length} steps) to avoid the known episode-reset-boundary "
            f"artifact (see .superpowers/sdd/task-2-report.md, task-3-report.md Section 7)."
        )
    analysis_history = height_history[:analysis_end]

    # Per-env settled/resting z: min over the first SETTLE_WINDOW_STEPS steps
    # (the object is spawned already resting on the table per
    # reset_object_position, so this window just absorbs any initial
    # micro-settling rather than measuring a real drop).
    settle_end = min(SETTLE_WINDOW_STEPS, analysis_end)
    resting_z = analysis_history[:settle_end].min(dim=0).values
    gain = analysis_history - resting_z.unsqueeze(0)
    max_gain = gain.max(dim=0).values
    max_z = analysis_history.max(dim=0).values

    lifted_mask = gain >= LIFT_HEIGHT_THRESHOLD_M  # (analysis_end, num_envs) bool

    summary = {}
    print("\n=== Instrumented height readout (per env) ===")
    print(f"lift threshold: {LIFT_HEIGHT_THRESHOLD_M} m above settled z, sustained >= {SUSTAINED_LIFT_STEPS} steps")
    for env_idx in range(num_envs):
        run_len = _max_consecutive_true(lifted_mask[:, env_idx].tolist())
        sustained = run_len >= SUSTAINED_LIFT_STEPS
        summary[str(env_idx)] = {
            "resting_z_m": float(resting_z[env_idx]),
            "max_z_m": float(max_z[env_idx]),
            "max_height_gain_m": float(max_gain[env_idx]),
            "max_consecutive_lifted_steps": run_len,
            "sustained_lift": bool(sustained),
        }
        print(
            f"  env {env_idx}: resting_z={resting_z[env_idx]:.4f}m max_z={max_z[env_idx]:.4f}m "
            f"max_gain={max_gain[env_idx]:.4f}m max_consecutive_lifted_steps={run_len} sustained_lift={sustained}"
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
                "lift_height_threshold_m": LIFT_HEIGHT_THRESHOLD_M,
                "sustained_lift_steps": SUSTAINED_LIFT_STEPS,
                "settle_window_steps": settle_end,
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

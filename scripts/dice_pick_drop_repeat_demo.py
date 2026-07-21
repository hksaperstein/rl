"""Continuous multi-cycle pick -> lift -> scripted-release -> reset demo of
the FINISHED target-selection-clutter Stage D2 checkpoint
(kb/wiki/experiments/target-selection-clutter.md's FINAL VERDICT -
d12 8/8, d20 8/8 sustained-lift, confirmed zero wrong-die grasps -
checkpoint `model_5096.pt`). This is a DEPLOYMENT of that already-finished,
already-validated experiment - no training, no new reward/mechanism, so
CLAUDE.md's Tier-1 spec/plan gate does not apply (same category as the
earlier scripted-demo primitives work).

What this does, one continuous rollout / one continuous video:
  1. Loads the D2 checkpoint into the trained policy (rsl_rl OnPolicyRunner,
     same load path as scripts/franka_checkpoint_review.py).
  2. Builds `FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg` directly (the
     real training env cfg, NOT one of its `_PLAY_D12Target`/`_PLAY_D20Target`
     eval variants, which pin ALL envs to a single shape) with
     `scene.num_envs == --num_cycles` - one parallel env per demo cycle.
  3. Runs the trained policy on all envs simultaneously (a real vectorized
     rollout - every env gets its own live policy action every step, exactly
     like training/eval) and reuses franka_checkpoint_review.py's own
     sustained-lift detection constants/logic (COPIED, not imported - see
     "Detection logic" note below for why) to decide, live, per env, once
     that env's own commanded die has been reliably lifted.
  4. Once lifted, overrides ONLY that one env's own action row for the
     remainder of its episode: freezes arm_action at its last commanded
     joint targets (a simple "hold position" script, not a new policy) and
     forces gripper_action positive (BinaryJointPositionActionCfg's
     open_command_expr convention - `mdp.BinaryJointPositionActionCfg` maps
     any positive raw action to full open, confirmed directly against this
     project's own gripper-lowpass-filtering diagnostic, see kb's "ruled
     out" section). Every other env keeps receiving its own real policy
     action, unaffected.
  5. The episode's own existing time_out termination (5.0s @ 50Hz control =
     250 steps, `tasks/franka/lift_env_cfg.py`'s `TerminationsCfg`) then
     resets that env automatically (Isaac Lab's per-env vectorized
     auto-reset convention) - a real reset of distractor positions (fresh
     `reset_root_state_uniform` draw) exactly as D2 training's own
     `TargetSelectionEventCfg` already does every episode.
  6. The viewport camera (the one `gym.wrappers.RecordVideo`'s
     `render_mode="rgb_array"` captures - see
     `isaaclab.envs.ui.viewport_camera_controller.ViewportCameraController.
     set_view_env_index`, confirmed to support a LIVE mid-rollout env-index
     switch, not just a construction-time-only setting) moves to a new env
     index at each fixed 250-step boundary, so cycle N of the demo = env
     index N's own single episode. One continuous `RecordVideo` recording
     spans the entire num_cycles*250-step rollout - not one clip per cycle.

Target-shape "randomization" - a real constraint, a real judgment call:
`scene.object.spawn`'s shape assignment (`MultiAssetSpawnerCfg`, inherited
unchanged from `FrankaDieLiftJointD12D20MixedEnvCfg`) is fixed PER ENV at
scene-construction time, not re-drawable at each reset - so "the commanded
target randomized across both shapes" is implemented as one INDEPENDENT
random draw per env/cycle (env 0 -> cycle 1's shape, env 1 -> cycle 2's
shape, etc.), not a true re-roll of the SAME env's shape between its own
resets (not supported by this training env's own design, and out of this
deployment task's scope to change). This script deliberately does NOT
override `scene.object.spawn`'s `random_choice` to `True` to get a more
literal "random" draw: `die_shape_classes_per_env` (the OTHER config field
that drives the `shape_class`/`geometry_descriptor` POLICY OBSERVATION
terms, see `tasks/franka/mdp.py::object_shape_class_onehot`) is a config-
time-known, deterministic `env_index % len(die_shape_classes_per_env)`
formula, NOT read off the live spawner state - if the two were set to
different randomizations, the policy's own shape-identity observation
would silently mismatch the physically-spawned shape for some envs, a
subtle, hard-to-notice correctness bug. This script instead keeps
`random_choice=False` (the base class's own default, an exact,
deterministic env0=d12/env1=d20/env2=d12/... round-robin,
`_D12D20_MIXED_ASSETS_ORDER` in `tasks/franka/dice_lift_joint_env_cfg.py`) -
the EXACT mechanism this checkpoint was trained/evaluated under, so both
shapes are guaranteed to appear (in strict alternation) with zero risk of
an observation/physical-shape mismatch. Distractor shapes/positions DO use
`random_choice=True` (unchanged from training) and DO get a fresh position
draw every reset via `TargetSelectionEventCfg`, per the D2 env's own
established randomization.

Detection logic (reused, not re-derived, per this task's own instruction):
`scripts/franka_checkpoint_review.py`'s own
LIFT_HEIGHT_THRESHOLD_M/SUSTAINED_LIFT_STEPS/EARLY_SETTLE_START/
EARLY_SETTLE_END constants and `_max_consecutive_true` helper are copied
verbatim below rather than `import`ed - that script has top-level
`argparse.parse_args()`/`AppLauncher(...)` side effects at MODULE IMPORT
time (it is a standalone entry point, not a library), so importing it
directly into this script would re-parse this script's own CLI args
through its own parser and attempt to launch a SECOND AppLauncher/Isaac Sim
app in the same process - not safe. Copying the small, pure, already-
verified constants/helper is the practical equivalent of "reuse rather
than re-derive" here. Unlike that script (which does one post-hoc pass
over a full pre-recorded height array), this script runs the exact same
resting_z-then-gain-then-consecutive-run logic LIVE, per env, per step, so
it can act on the result mid-rollout (trigger the scripted release).

Live per-env episode-boundary tracking uses `env.unwrapped.episode_length_
buf[env_idx]` (Isaac Lab's own live per-env step-since-last-reset counter,
confirmed directly against a real IsaacLab source checkout,
`isaaclab/envs/manager_based_rl_env.py`) rather than a hand-rolled global-
step assumption - this self-corrects if a given cycle's own env resets
EARLY (e.g. a rare `object_dropping` termination, ~0.5-1.3% base rate per
kb's own Task 5/6 training-curve numbers) rather than silently
misinterpreting stale height data as if no reset had happened. The
FIXED, num_cycles-many 250-step camera-switch schedule is intentionally
NOT re-synced to this per-env signal (keeps total video length/schedule
simple and predictable); an early within-slot reset is logged and that
slot's own tracking window restarts from scratch, so the reported
per-cycle outcome always reflects whatever is actually on screen when
that cycle's fixed window ends.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_drop_repeat_demo.py \
        --checkpoint /path/to/model_5096.pt --num_cycles 10 --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Continuous multi-cycle pick->lift->scripted-release->reset demo of the target-selection-clutter "
    "Stage D2 checkpoint."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the D2 rsl_rl checkpoint (.pt) to load.")
parser.add_argument(
    "--num_cycles",
    type=int,
    default=10,
    help="Number of full pick->lift->drop->reset cycles to run (also sets scene.num_envs - one env per cycle, "
    "since target shape is fixed per env at scene-construction time; see module docstring).",
)
parser.add_argument(
    "--release_hold_steps",
    type=int,
    default=60,
    help="How many control steps (@ 50Hz) to hold the scripted open-gripper/frozen-arm release action for once "
    "sustained lift is confirmed, before just letting the episode run out to its own natural time_out reset "
    "(1.2s default - comfortably longer than a ~14-step free-fall from a ~40cm release height plus settle time).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write the video to (default: logs/videos/dice_pick_drop_repeat_demo/).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video rendering

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")
if args_cli.num_cycles < 1:
    sys.exit(f"--num_cycles must be >= 1, got {args_cli.num_cycles}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg  # noqa: E402

VIDEO_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "dice_pick_drop_repeat_demo"
)
os.makedirs(VIDEO_DIR, exist_ok=True)

# --- Detection constants, copied verbatim from scripts/franka_checkpoint_review.py
# (see that script's own extensive commit history/comments for the full derivation -
# NOT re-derived here, per this task's own "reuse rather than re-derive" instruction;
# see this module's own docstring "Detection logic" section for why this is a copy,
# not an `import`). ---
LIFT_HEIGHT_THRESHOLD_M = 0.04
SUSTAINED_LIFT_STEPS = 25
EARLY_SETTLE_START = 10
EARLY_SETTLE_END = 45

# Positive raw gripper action = OPEN (mdp.BinaryJointPositionActionCfg's
# open_command_expr convention) - confirmed directly against a live rollout in
# kb/wiki/experiments/target-selection-clutter.md's "Gripper actuator low-pass-
# filtering check" section (100% of a real checkpoint's own raw gripper actions were
# positive/open, 0% negative). Magnitude is irrelevant (BinaryJointPositionAction only
# reads the sign), a decisively large value is used just to make the override obvious
# in any raw-action debug printout.
GRIPPER_OPEN_ACTION = 5.0

# How far (m) the object must fall from its own height at the instant the scripted
# release engaged, before this script calls the release "confirmed" rather than just
# "commanded" - i.e. did the object actually separate and drop, not just get an open
# command sent to it. Chosen well below this checkpoint's own typical ~300-480mm
# max_height_gain (kb's Task 6 table) so a real release is never missed, but well
# above sensor/physics jitter.
RELEASE_CONFIRM_DROP_M = 0.05


def _max_consecutive_true(mask: list) -> int:
    """Longest run of consecutive True values in a 1D bool sequence. Copied verbatim
    from scripts/franka_checkpoint_review.py (see module docstring)."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    env_cfg = FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_cycles
    env_cfg.sim.device = args_cli.device
    # Deterministic eval, matching every existing _PLAY variant's own convention
    # (e.g. FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg_PLAY_D12Target).
    env_cfg.observations.policy.enable_corruption = False

    # Same full-arm-in-frame viewer framing as franka_checkpoint_review.py; env_index
    # is just the INITIAL value here - the main loop below moves the camera live via
    # viewport_camera_controller.set_view_env_index() at each cycle boundary.
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.8, 1.8, 1.1)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.35)

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    episode_length_steps = int(env.max_episode_length)
    num_cycles = args_cli.num_cycles
    total_steps = num_cycles * episode_length_steps
    print(
        f"[dice_pick_drop_repeat_demo] {num_cycles} cycles x {episode_length_steps} steps/episode = "
        f"{total_steps} total steps ({total_steps / 50.0:.1f}s @ 50Hz control rate)."
    )

    checkpoint_run_dir = os.path.basename(os.path.dirname(os.path.abspath(args_cli.checkpoint)))
    checkpoint_name = f"{checkpoint_run_dir}_{os.path.splitext(os.path.basename(args_cli.checkpoint))[0]}"
    video_kwargs = {
        "video_folder": VIDEO_DIR,
        "step_trigger": lambda step: step == 0,
        "video_length": total_steps,
        "name_prefix": f"dice_pick_drop_repeat_demo_{checkpoint_name}_{num_cycles}cycles",
        "disable_logger": True,
    }
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # load_optimizer=False: eval-only entry point, matches franka_checkpoint_review.py.
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    die_shape_classes_per_env = env.unwrapped.cfg.die_shape_classes_per_env  # ("d12", "d20")

    def shape_for_env(env_idx: int) -> str:
        return die_shape_classes_per_env[env_idx % len(die_shape_classes_per_env)]

    viewport = env.unwrapped.viewport_camera_controller

    cycle_results = []

    # --- per-active-cycle-slot tracking state (reset at every fixed 250-step camera
    # boundary AND at any early within-slot per-env reset detected via
    # episode_length_buf) ---
    cur_heights: list = []
    resting_z = None
    max_gain_this_cycle = 0.0
    lifted_confirmed = False
    lifted_local_step = None
    release_started = False
    frozen_arm_action = None
    peak_height_at_release = None
    current_shape = None
    prev_local_step = None

    def reset_cycle_tracking():
        nonlocal cur_heights, resting_z, max_gain_this_cycle, lifted_confirmed
        nonlocal lifted_local_step, release_started, frozen_arm_action, peak_height_at_release, prev_local_step
        cur_heights = []
        resting_z = None
        max_gain_this_cycle = 0.0
        lifted_confirmed = False
        lifted_local_step = None
        release_started = False
        frozen_arm_action = None
        peak_height_at_release = None
        prev_local_step = None

    def flush_cycle_result(cycle_num: int, watched_env: int, shape: str):
        h_final = cur_heights[-1] if cur_heights else None
        drop = None
        release_confirmed = False
        if release_started and peak_height_at_release is not None and h_final is not None:
            drop = peak_height_at_release - h_final
            release_confirmed = drop >= RELEASE_CONFIRM_DROP_M
        result = {
            "cycle": cycle_num,
            "env_index": watched_env,
            "commanded_shape": shape,
            "sustained_lift_detected": bool(lifted_confirmed),
            "lifted_local_step": lifted_local_step,
            "max_height_gain_m": float(max_gain_this_cycle),
            "release_engaged": bool(release_started),
            "height_at_release_m": peak_height_at_release,
            "final_height_m": h_final,
            "height_drop_after_release_m": drop,
            "release_confirmed": bool(release_confirmed),
        }
        cycle_results.append(result)
        print(
            f"  [cycle {cycle_num}/{num_cycles}] RESULT env={watched_env} shape={shape} "
            f"sustained_lift={result['sustained_lift_detected']} max_gain={result['max_height_gain_m']:.4f}m "
            f"release_engaged={result['release_engaged']} release_confirmed={result['release_confirmed']} "
            f"(drop={drop if drop is None else f'{drop:.4f}m'})"
        )

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(total_steps):
            fixed_slot = step // episode_length_steps
            fixed_slot_start = step == fixed_slot * episode_length_steps
            watched_env = fixed_slot

            if fixed_slot_start:
                if fixed_slot > 0:
                    flush_cycle_result(fixed_slot, fixed_slot - 1, current_shape)
                viewport.set_view_env_index(watched_env)
                current_shape = shape_for_env(watched_env)
                print(f"=== Cycle {fixed_slot + 1}/{num_cycles}: env {watched_env}, commanded target = {current_shape} ===")
                reset_cycle_tracking()

            actions = policy(obs)

            if lifted_confirmed:
                if not release_started:
                    frozen_arm_action = actions[watched_env, :7].clone()
                    peak_height_at_release = cur_heights[-1] if cur_heights else None
                    release_started = True
                    print(
                        f"  [cycle {fixed_slot + 1}] sustained lift confirmed at local_step={lifted_local_step} "
                        f"(max_gain so far {max_gain_this_cycle:.4f}m) -> scripted release engaged "
                        f"(gripper open, arm held at last commanded target)"
                    )
                actions[watched_env, :7] = frozen_arm_action
                actions[watched_env, 7] = GRIPPER_OPEN_ACTION

            obs, _, _, _ = env.step(actions)

            local_step = int(env.unwrapped.episode_length_buf[watched_env].item())
            # BUG FOUND AND FIXED (2026-07-21, first real cloud run): the env's natural
            # end-of-episode auto-reset ALSO wraps episode_length_buf 249->0, at the LAST
            # step of the current fixed slot (local_step==episode_length_steps-1 just
            # BEFORE the wrap) - not at the next slot's own fixed_slot_start iteration
            # (that happens one global step later). The original condition here
            # (`not fixed_slot_start`) treated this expected, once-per-cycle wrap as an
            # anomalous EARLY reset and called reset_cycle_tracking() right before
            # flush_cycle_result() read that state on the next iteration - silently
            # discarding a correctly-detected sustained_lift_confirmed=True on every
            # single cycle (confirmed directly: the very first real 10-cycle cloud run
            # printed "sustained lift confirmed" for 9/10 cycles, immediately followed by
            # this wrap-detection firing and every cycle still being reported as
            # sustained_lift=False/max_gain=0.0000 - the underlying rollout was working,
            # only this reporting logic was wrong). Fix: only treat a buf decrease as a
            # genuine early/anomalous reset when it did NOT come from the natural final
            # step of the slot (prev_local_step == episode_length_steps - 1).
            is_natural_episode_boundary = prev_local_step == episode_length_steps - 1
            if prev_local_step is not None and local_step < prev_local_step and not is_natural_episode_boundary:
                print(
                    f"  [cycle {fixed_slot + 1}] NOTE: env {watched_env}'s episode_length_buf wrapped "
                    f"{prev_local_step}->{local_step} mid-slot (an early per-env reset, e.g. object_dropping "
                    f"termination, fired before this slot's own fixed 250-step boundary) - restarting this "
                    f"cycle's own tracking window from scratch."
                )
                reset_cycle_tracking()
            prev_local_step = local_step

            h = float(env.unwrapped.scene["object"].data.root_pos_w[watched_env, 2].detach().cpu())
            cur_heights.append(h)

            if not lifted_confirmed and len(cur_heights) >= EARLY_SETTLE_END:
                if resting_z is None:
                    resting_z = min(cur_heights[EARLY_SETTLE_START:EARLY_SETTLE_END])
                gain = h - resting_z
                max_gain_this_cycle = max(max_gain_this_cycle, gain)
                post_settle = cur_heights[EARLY_SETTLE_START:]
                lifted_mask = [(x - resting_z) >= LIFT_HEIGHT_THRESHOLD_M for x in post_settle]
                if _max_consecutive_true(lifted_mask) >= SUSTAINED_LIFT_STEPS:
                    lifted_confirmed = True
                    lifted_local_step = local_step

        # Flush the final cycle - no further fixed_slot_start iteration exists to do it.
        flush_cycle_result(num_cycles, num_cycles - 1, current_shape)

    # Write the per-cycle JSON summary and print the report BEFORE env.close() - this
    # project has repeatedly hit a real, documented Isaac Sim gotcha (CLAUDE.md's
    # "Known gap: a hung process still holds the lock") where the app hangs during its
    # own Kit/extension shutdown teardown AFTER the script's actual work is already
    # done. All the real rollout data is already final at this point (the for-step loop
    # above has completed); writing it out now means a teardown hang doesn't lose it.
    n_sustained = sum(1 for r in cycle_results if r["sustained_lift_detected"])
    n_release_confirmed = sum(1 for r in cycle_results if r["release_confirmed"])
    print(f"\n=== Summary: {n_sustained}/{num_cycles} cycles reached sustained lift, "
          f"{n_release_confirmed}/{num_cycles} cycles confirmed a real post-lift release/drop ===")
    for r in cycle_results:
        print(f"  cycle {r['cycle']}: shape={r['commanded_shape']} sustained_lift={r['sustained_lift_detected']} "
              f"release_confirmed={r['release_confirmed']} max_gain={r['max_height_gain_m']:.4f}m")

    summary_json_path = os.path.join(VIDEO_DIR, f"summary_{checkpoint_name}_{num_cycles}cycles.json")
    with open(summary_json_path, "w") as f:
        json.dump(
            {
                "checkpoint": args_cli.checkpoint,
                "num_cycles": num_cycles,
                "episode_length_steps": episode_length_steps,
                "total_steps": total_steps,
                "lift_height_threshold_m": LIFT_HEIGHT_THRESHOLD_M,
                "sustained_lift_steps": SUSTAINED_LIFT_STEPS,
                "release_confirm_drop_m": RELEASE_CONFIRM_DROP_M,
                "cycles_with_sustained_lift": n_sustained,
                "cycles_with_release_confirmed": n_release_confirmed,
                "per_cycle": cycle_results,
            },
            f,
            indent=2,
        )
    print(f"Per-cycle summary written to: {summary_json_path}")

    env.close()
    print(f"Demo video written to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()

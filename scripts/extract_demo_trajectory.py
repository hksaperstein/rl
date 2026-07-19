# scripts/extract_demo_trajectory.py
"""Task 1 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
implementation.md: captures a scripted-grasp DEMONSTRATION TRAJECTORY for
d8/d10 at the 48mm-parity scale H1 actually trains at, by running Task 0's
own re-verified `dice_pick_demo.py` scripted DiffIK grasp mechanism and
logging, every physics step, the desired joint-position target just issued
to `panda_joint.*` and the currently-commanded gripper joint-position
target - the raw material Task 2's BC-pretrain replay driver converts (via
`tasks/franka/demo_action_mapping.py`) into (observation, action) pairs in
the real RL env's own action space.

New sibling script - NEVER modifies `scripts/dice_pick_demo.py` in place
(this repo's own established reuse convention). Imports that file's
`spawn_scene_and_settle`/`run_detector_subprocess`/`select_target_detection`/
`run_pick_sequence` UNCHANGED, plus Task 0's own
`scripts/_diag_d8d10_48mm_grasp_reverify.py` `override_die_scale`/
`measure_settled_rest_height`/`measure_settled_position_m`/
`recapture_camera_frame`/`_SCALE_48MM` (reused, NOT duplicated - this
module's own capture flow mirrors that diagnostic's `run_shape_reverify`
line-for-line, substituting the diagnostic's JSON-verdict-only output for
this module's own additional per-step trajectory logging + `.pt` save).

**Import-time side-effect note, --gt-xy-bypass, one-shape-per-process,
--headless discipline**: all IDENTICAL to
`scripts/_diag_d8d10_48mm_grasp_reverify.py`'s own module docstring - see
that file for the full explanation of why (sys.argv/argv[0] swap around the
`dice_pick_demo` import; `--gt-xy-bypass` reusing that file's own d4-rung-1-
precedented ground-truth XY-bypass mechanism, needed for d10 at 48mm scale
per Task 0's own finding that d10 detection fails at this scale; one shape
AND one seed per process invocation, since this Isaac Lab installation
cannot hold two live `InteractiveScene`s in one process; never pass
--headless, a display exists and the user wants to watch).

**Per-shape bypass usage (Task 0's own already-established, per-shape
finding - reused here, not re-decided)**: Task 0 found d8 PASSes WITHOUT
`--gt-xy-bypass` (242.6mm z-gain, detector finds d8 fine at 48mm) and d10
needs `--gt-xy-bypass` to even reach the grasp mechanism (detector finds
ZERO d10 detections at 48mm scale - a perception gap, not a grasp-mechanism
failure). This script's own Step 7 real captures therefore run d8 WITHOUT
the flag and d10 WITH it, for all 5 seeds each - the same per-shape
configuration Task 0 already validated PASSes the underlying grasp
mechanism, rather than re-deciding this question. `--gt-xy-bypass` only
changes where `target_xy` is SOURCED from (ground truth vs. detector); it
does not change the pick sequence's own joint-position/gripper-target
control flow at all, so it has no bearing on this capture's own logged
trajectory data quality once a valid `target_xy` is in hand.

Output: `data/franka_demo_trajectories/{shape}/seed{N}.pt` (`torch.save` of
a plain dict - `data/` is gitignored per this repo's public-repo-since-
2026-07-13 no-datasets convention, so these files are never committed, only
the code that produces/consumes them). Dict keys:
  - "shape", "seed": str, int - which capture this is.
  - "gt_xy_bypass": bool - whether target_xy was ground-truth-sourced.
  - "arm_joint_pos_target": `(num_steps, 7)` float32 tensor - the desired
    absolute joint-position target commanded to `panda_joint.*` at every
    physics step of the pick sequence (stage 0's joint-space prep AND every
    `_step_toward` call in stages 1-4), read back from
    `robot.data.joint_pos_target` (the exact tensor
    `Articulation.set_joint_position_target` writes into,
    `isaaclab/assets/articulation/articulation.py:1079`) immediately after
    each step's `on_step` callback fires - NOT independently re-derived,
    so this is guaranteed byte-identical to what `dice_pick_demo.py`'s own
    control loop actually commanded that step.
  - "gripper_target": `(num_steps, 2)` float32 tensor - the same read-back,
    for the 2 `panda_finger_.*` joints (always `dice_pick_demo.py`'s own
    `open_target`=0.04 or `close_target`=0.0 per row).
  - "default_joint_pos": `(7,)` float32 tensor - THIS capture scene's own
    live-read default joint pos, informational/diagnostic only. Task 2's
    replay driver must read `default_joint_pos` LIVE from the actual target
    RL env it replays into (a different scene/asset instance), not from
    this field - flagged here so that is never silently assumed
    interchangeable.
  - "waypoint_status", "pick_sequence_error", "verdict_table", "pass": same
    shape/meaning as `_diag_d8d10_48mm_grasp_reverify.py`'s own verdict
    JSON fields - a capture whose "pass" is False is NOT a valid
    demonstration (Step 7 below drops and replaces any such seed, per the
    plan's own instruction).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/extract_demo_trajectory.py --shape d8 --seed 42"

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/extract_demo_trajectory.py --shape d10 --seed 42 --gt-xy-bypass"
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "extract_demo_trajectory")
TRAJECTORY_DIR = os.path.join(REPO_ROOT, "data", "franka_demo_trajectories")

# --- Parse THIS script's own CLI args from the REAL argv, before
# dice_pick_demo.py's own module-level argparse.parse_args() call runs (see
# module docstring's "Import-time side-effect note") - `parse_known_args` so
# AppLauncher's own extra flags don't trip an "unrecognized arguments" error
# here. ---
_parser = argparse.ArgumentParser(
    description="Capture a demonstration trajectory (Task 1) via dice_pick_demo.py's scripted grasp at 48mm scale."
)
_parser.add_argument("--shape", type=str, choices=["d8", "d10"], required=True, help="Which die to capture this run.")
_parser.add_argument(
    "--seed", type=int, required=True, help="Seed for the randomized 5-die layout - dice_pick_demo.py's own --seed semantics."
)
_parser.add_argument(
    "--gt-xy-bypass",
    action="store_true",
    default=False,
    help=(
        "Reuses dice_pick_demo.py's own --gt-xy-bypass mechanism (via "
        "scripts/_diag_d8d10_48mm_grasp_reverify.py's identical flag) - see this module's own "
        "docstring's 'Per-shape bypass usage' section for which shape needs this and why."
    ),
)
own_args, _unknown_args = _parser.parse_known_args()

# --- Import dice_pick_demo.py's machinery (see module docstring's "Import-time
# side-effect note" - sys.argv, INCLUDING argv[0], must be swapped around this
# import, exactly matching _diag_d8d10_48mm_grasp_reverify.py's own established
# pattern). ---
_real_argv = sys.argv
_dice_pick_demo_path = os.path.join(SCRIPTS_DIR, "dice_pick_demo.py")
sys.argv = [_dice_pick_demo_path, "--gate", "g"]
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import dice_pick_demo as dpd  # noqa: E402

sys.argv = _real_argv

# Task 0's own already-validated 48mm-scale re-verification machinery -
# imported, not duplicated (this module's own docstring).
import _diag_d8d10_48mm_grasp_reverify as diag48  # noqa: E402

sys.argv = _real_argv  # diag48's own import-time argv swap (mirrors dpd's) - restore again for safety

from isaaclab.managers import SceneEntityCfg  # noqa: E402
import torch  # noqa: E402


def capture_trajectory(shape: str, seed: int, gt_xy_bypass: bool) -> dict:
    """Mirrors `_diag_d8d10_48mm_grasp_reverify.py`'s own `run_shape_reverify`
    flow (spawn/settle -> rescale target die + re-settle -> recapture camera
    frame -> detector subprocess -> target select (bypass-aware) ->
    run_pick_sequence WITH a step-logging on_step hook -> verdict), adding
    this module's own per-step joint-position/gripper-target trajectory
    logging and returning a dict shaped for `torch.save`, in place of that
    diagnostic's JSON-only verdict output."""
    out_dir = os.path.join(OUT_DIR, shape)
    os.makedirs(out_dir, exist_ok=True)

    sim, scene, positions, results = dpd.spawn_scene_and_settle(out_dir, seed)

    scale = diag48._SCALE_48MM[shape]
    diag48.override_die_scale(sim, scene, shape, scale)

    # Same stale-camera-frame bug fix Task 0 found and fixed - must run
    # before the detector subprocess and before the rest-height measurement.
    diag48.recapture_camera_frame(sim, scene, out_dir)

    measured_rest_height_m = diag48.measure_settled_rest_height(scene, shape)
    print(
        f"[CAPTURE] {shape} seed={seed}: measured settled rest height = {measured_rest_height_m * 1000:.2f}mm "
        f"(48mm-parity scale={scale})"
    )

    detection_output = dpd.run_detector_subprocess(out_dir)
    detections = detection_output["detections"]
    print(f"[CAPTURE] perception subprocess returned {len(detections)} detections")

    # Ground-truth XY-bypass - identical mechanism/precedent to
    # _diag_d8d10_48mm_grasp_reverify.py's own (see that file's own
    # --gt-xy-bypass argparse help for the full citation chain back to the
    # d4 rung-1 precedent).
    target_det = None
    det_x = det_y = det_z = None
    try:
        target_det = dpd.select_target_detection(detections, shape)
        det_x, det_y, det_z = target_det["world_pos"]
        print(
            f"[CAPTURE] target detection for '{shape}': class={target_det['class']} "
            f"conf={target_det['confidence']:.3f} world_pos=({det_x:.4f}, {det_y:.4f}, {det_z:.4f})"
        )
    except RuntimeError as e:
        if not gt_xy_bypass:
            raise
        print(f"[CAPTURE] detector FAILED to find '{shape}' ({e}) - continuing because --gt-xy-bypass is active")

    gt_x, gt_y, _gt_z = diag48.measure_settled_position_m(scene, shape)
    if gt_xy_bypass:
        target_xy = (gt_x, gt_y)
        print(f"[CAPTURE] target_xy SOURCED FROM GROUND TRUTH (bypass active): ({target_xy[0]:.4f}, {target_xy[1]:.4f})")
    else:
        target_xy = (float(det_x), float(det_y))
        print(f"[CAPTURE] target_xy sourced from DETECTOR: ({target_xy[0]:.4f}, {target_xy[1]:.4f})")

    # --- Trajectory logging setup. Re-resolves the SAME two SceneEntityCfg
    # queries run_pick_sequence itself resolves internally (robot arm
    # joints, gripper finger joints) - a fresh, independent resolution
    # against the same live `scene`, not a hack reaching into
    # run_pick_sequence's own internals. `on_step` (passed into
    # run_pick_sequence, the same extension point Gate V's own video-frame
    # capture hook already uses) reads `robot.data.joint_pos_target` AFTER
    # each step - the exact tensor `Articulation.set_joint_position_target`
    # writes into (isaaclab/assets/articulation/articulation.py:1079,
    # confirmed by direct source read this task) - so every logged row is
    # guaranteed byte-identical to what run_pick_sequence's own
    # _step_toward/_joint_space_prep actually commanded that step, without
    # needing read access to either function's own local joint_pos_des
    # variable. ---
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
    robot_entity_cfg.resolve(scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    gripper_cfg.resolve(scene)

    arm_targets: list = []
    gripper_targets: list = []

    def _on_step() -> None:
        joint_pos_target = robot.data.joint_pos_target
        arm_targets.append(joint_pos_target[0, robot_entity_cfg.joint_ids].clone().cpu())
        gripper_targets.append(joint_pos_target[0, gripper_cfg.joint_ids].clone().cpu())

    pick_sequence_error = None
    try:
        waypoint_status = dpd.run_pick_sequence(
            sim, scene, target_xy, measured_rest_height_m, shape, on_step=_on_step, results=results
        )
    except dpd._StageTimeoutError as e:
        pick_sequence_error = str(e)
        waypoint_status = {"error": pick_sequence_error}
        print(f"[CAPTURE] *** pick sequence FAILED for {shape} seed={seed}: stage timeout - {pick_sequence_error} ***")

    verdict_table, all_ok = dpd._compute_verdict_table(scene, results, shape)
    passed = bool(all_ok) and pick_sequence_error is None
    print(f"[CAPTURE] {shape} seed={seed}: {'PASS' if passed else 'FAIL'} (waypoints={waypoint_status})")

    num_steps = len(arm_targets)
    if num_steps == 0:
        arm_traj = torch.zeros((0, len(robot_entity_cfg.joint_ids)), dtype=torch.float32)
        gripper_traj = torch.zeros((0, len(gripper_cfg.joint_ids)), dtype=torch.float32)
    else:
        arm_traj = torch.stack(arm_targets).float()
        gripper_traj = torch.stack(gripper_targets).float()

    default_joint_pos = robot.data.default_joint_pos[0, robot_entity_cfg.joint_ids].clone().cpu().float()

    return {
        "shape": shape,
        "seed": seed,
        "gt_xy_bypass": bool(gt_xy_bypass),
        "arm_joint_pos_target": arm_traj,
        "gripper_target": gripper_traj,
        "default_joint_pos": default_joint_pos,
        "waypoint_status": waypoint_status,
        "pick_sequence_error": pick_sequence_error,
        "verdict_table": verdict_table,
        "pass": passed,
        "num_steps": num_steps,
    }


def main() -> None:
    if own_args.shape not in dpd.DIE_TYPES:
        raise RuntimeError(f"'{own_args.shape}' is not one of the physical dice in this scene: {dpd.DIE_TYPES}")

    trajectory = capture_trajectory(own_args.shape, own_args.seed, own_args.gt_xy_bypass)

    shape_dir = os.path.join(TRAJECTORY_DIR, own_args.shape)
    os.makedirs(shape_dir, exist_ok=True)
    out_path = os.path.join(shape_dir, f"seed{own_args.seed}.pt")
    torch.save(trajectory, out_path)
    print(
        f"\n[TASK1 CAPTURE] {own_args.shape} seed={own_args.seed}: "
        f"{'PASS' if trajectory['pass'] else 'FAIL'} - {trajectory['num_steps']} steps logged -> {out_path}"
    )

    # A verdict summary alongside the .pt (not part of the gitignored data/
    # dataset itself - written to outputs/, mirroring
    # _diag_d8d10_48mm_grasp_reverify.py's own verdict-JSON convention) for
    # quick human inspection without loading the tensor file.
    summary_path = os.path.join(OUT_DIR, own_args.shape, f"summary_seed{own_args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "shape": trajectory["shape"],
                "seed": trajectory["seed"],
                "gt_xy_bypass": trajectory["gt_xy_bypass"],
                "pass": trajectory["pass"],
                "num_steps": trajectory["num_steps"],
                "pick_sequence_error": trajectory["pick_sequence_error"],
                "waypoint_status": trajectory["waypoint_status"],
                "verdict_table": trajectory["verdict_table"],
                "saved_to": out_path,
            },
            f,
            indent=2,
        )
    print(f"[TASK1 CAPTURE] saved summary: {summary_path}")

    if not trajectory["pass"]:
        raise RuntimeError(
            f"capture FAILED for shape={own_args.shape} seed={own_args.seed} - not a usable demonstration "
            f"(see waypoint_status/pick_sequence_error above). Per the plan's own Step 7 instruction: drop this "
            f"seed and capture a replacement seed instead - do not pool a failed-grasp trajectory."
        )


if __name__ == "__main__":
    # try/except/finally so simulation_app.close() ALWAYS runs, even if
    # main() raises - same Kit-teardown-hang avoidance reasoning as
    # dice_pick_demo.py's own __main__ guard and
    # _diag_d8d10_48mm_grasp_reverify.py's own identical guard (both cited
    # in this repo's CLAUDE.md documented failure mode).
    try:
        main()
        print("[DONE] holding window briefly before close...")
        time.sleep(3.0)
    except BaseException:
        import traceback

        print("[ERROR] exception in main(), full traceback follows:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        dpd.simulation_app.close()

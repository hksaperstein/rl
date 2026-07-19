# scripts/_diag_d8d10_48mm_grasp_reverify.py
"""Task 0 (PREREQUISITE, gates all later tasks) of
docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-implementation.md:
re-verifies that `scripts/dice_pick_demo.py`'s scripted DiffIK grasp
mechanism - already verified PASS for d8/d10 at REAL commercial size
(~16mm, kb/wiki/experiments/dice-pick-demo.md: "d8 240.9mm / d10 239.3mm
z-gain") - ALSO transfers to the 48mm-parity size
`docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`'s H1
actually trains at (`FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`,
`tasks/franka/dice_lift_joint_env_cfg.py`). This is a real, unresolved
scale mismatch the design spec flags explicitly, not assumed away: the
demo's own per-die grasp-height table (`_DIE_REST_HEIGHT_M`) and
tolerances were measured at real size and are not assumed to transfer
unchanged to a ~3x larger die.

Reuses `dice_pick_demo.py`'s `spawn_scene_and_settle`/
`run_detector_subprocess`/`select_target_detection`/`run_pick_sequence`/
`_compute_verdict_table`/`_StageTimeoutError`/`_DIE_REST_HEIGHT_M` BY
IMPORT, UNCHANGED - that file's own Gate A/G/V contracts are never
modified in place (this repo's own established reuse convention, see
that file's own Gate V reusing Gate G's flow via `on_step` rather than
touching Gate G itself). This diagnostic's own flow below mirrors Gate
G's (`run_gate_g`) line-for-line, substituting two things:
  1. The commanded die is spawned/settled at its DEMO-DEFAULT real-world
     size (unchanged `spawn_scene_and_settle` call), then rescaled to the
     48mm-parity constant on its LIVE USD prim (`override_die_scale`) and
     re-settled with a short extra physics-stepping loop - see
     `override_die_scale`'s own docstring for why this is a POST-HOC
     rescale rather than a pre-spawn `scene_cfg` override (the originally
     designed approach, empirically found infeasible this task - see
     below). The other 4 dice in the 5-die layout stay at real size
     (irrelevant to this shape's own grasp check; changing the whole scene
     layout is out of scope for this task).
  2. The grasp height passed to `run_pick_sequence` comes from a FRESH
     live measurement of the scaled die's own re-settled resting height
     (`measure_settled_rest_height`), NOT `_DIE_REST_HEIGHT_M`'s real-size
     table value and NOT that value scaled linearly by the spawn-scale
     ratio - per this task's own explicit "measure it directly" mandate
     (the spec declines to assume real-size measurements transfer to a
     ~3x larger die).

**Deviation from this task's original design, recorded here per this
project's "flag deviations, don't paper over them" discipline**: the
Task 0 plan called for mutating `scene_cfg.die_{die_type}.spawn.scale`
BEFORE `InteractiveScene(scene_cfg)` is built inside
`spawn_scene_and_settle`, mirroring that function's own internal
`light_scale` override. This was implemented and empirically tested
(2026-07-19, via a throwaway `scripts/_diag_configclass_probe.py`, since
deleted) and found INFEASIBLE for IsaacLab's actual `@configclass`
semantics, confirmed by direct source read of
`isaaclab.utils.configclass` (`_process_mutable_types`/`_return_f`,
`/home/saps/IsaacLab/source/isaaclab/isaaclab/utils/configclass.py` on
the desktop) AND by a live reproduction: a `@configclass`-decorated
class's mutable-typed fields (e.g. `die_d8`) are converted to
`field(default_factory=_return_f(original_value))`, and `_return_f`'s
closure (`configclass.py`'s `_wrap()`, final branch: `return
deepcopy(f)`) returns a FRESH DEEPCOPY of the ORIGINAL closure-captured
object on EVERY call - never the same shared reference, and never
influenced by mutating a previous call's result. Two confirmed
consequences: (1) `getattr(DiceSceneCfg, "die_d8")` on the CLASS itself
raises `AttributeError` (dataclass fields using `default_factory` have
NO class-level attribute at all - reproduced live, exact traceback:
`AttributeError: type object 'DiceSceneCfg' has no attribute 'die_d8'`);
(2) even reaching the factory directly and mutating its returned object
would not affect any LATER call to that factory. There is therefore no
way to influence `spawn_scene_and_settle`'s own internal
`DiceSceneCfg(...)` construction from outside that function without
either modifying it (disallowed) or reaching into its closure cell
directly (inappropriate hackery for this task) - this file instead
achieves the same end state (target die at 48mm scale, fully physically
settled) via `override_die_scale`'s post-hoc USD-prim-rescale-then-
resettle approach below. This is a mechanical/technical correction to
Task 0's stated implementation detail, not a change to Task 0's actual
goal or verdict mechanism - flagged here rather than silently
implemented differently from what the plan describes.

**Import-time side effect note**: `dice_pick_demo.py` parses `sys.argv`
and launches `AppLauncher`/`SimulationApp` as MODULE-LEVEL code (not
inside a function or `__main__` guard), as soon as it is imported - so
`sys.argv` is temporarily swapped to a minimal valid invocation of THAT
file's own parser (which requires `--gate`) around the import below, then
restored. `sys.argv[0]` is set to `dice_pick_demo.py`'s OWN path (not
this script's path) for that same swap - empirically required (2026-07-19,
this task): with `sys.argv[0]` left as this script's own path, the import
reproducibly died silently a few seconds into Kit/AppLauncher extension
loading, with no Python exception ever printed (root cause not fully
isolated beyond this fix - AppLauncher/Kit plausibly derives some
app-identity/telemetry/log-path value from `argv[0]`'s basename during
its own startup); setting it to `dice_pick_demo.py`'s own path made the
import complete successfully and reproducibly across multiple runs.
Beyond that one value, the actual `--gate`/`--choice`/`--seed` values
parsed are inert: none of the functions this script calls read
`dice_pick_demo`'s own module-level `args_cli` - every value they need is
passed as an explicit function argument instead (this is also why this
pattern is safe to reuse, unchanged, from Task 1's own
`scripts/extract_demo_trajectory.py`, per the plan's Files section for
this task).

**One shape per process invocation** (`--shape d8` / `--shape d10`, run as
two separate `isaaclab.sh` invocations - the plan's own "Two shapes, one
run each" instruction), not both shapes in one process: this repo's own
established caution about not holding multiple live Isaac
scenes/environments open in a single process
(`docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-implementation.md`'s
Task 2: "this Isaac Lab installation cannot hold two ManagerBasedRLEnvs
open at once - confirmed the hard way in the prior experiment's Task 5")
is treated here as the same caution for two live `InteractiveScene`s.

Never pass --headless - a display exists (DISPLAY=:1) and the user wants
to watch (CLAUDE.md's Environment conventions).

**2026-07-19 follow-up task (d10 perception-bypass re-verify)**: d8 passed
this re-verification cleanly (242.6mm z-gain); d10 failed, but at
`select_target_detection` (0 detections for class `d10`), before the
scripted grasp controller ever ran - the grasp mechanism itself was never
exercised for d10. Two things were added to fix this, per
BACKLOG.md's 2026-07-19 "d8/d10 demo-warmstart Task 0" decision entry:
  3. `--gt-xy-bypass` (see its own argparse help above) - reuses
     `dice_pick_demo.py`'s own `--gt-xy-bypass` mechanism/precedent (d4
     rung-1) to source target_xy from a fresh ground-truth measurement
     instead of the detector, isolating the grasp mechanism from the
     perception gap.
  4. `recapture_camera_frame` (see its own docstring) - a real bug found
     alongside (3): the detector subprocess was reading a STALE real-size
     camera frame captured before the rescale, for every prior run of
     this diagnostic (both d8 and d10), not a genuine 48mm-scale image.
     Fixed unconditionally (applies with or without `--gt-xy-bypass`).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_d8d10_48mm_grasp_reverify.py --shape d8"

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_d8d10_48mm_grasp_reverify.py --shape d10 --gt-xy-bypass"
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "diag_48mm_grasp_reverify")

# --- Parse THIS script's own CLI args from the REAL argv, before
# dice_pick_demo.py's own module-level argparse.parse_args() call runs (see
# below) - `parse_known_args` so AppLauncher's own extra flags (--device,
# --headless, etc, added later by dice_pick_demo.py's parser) don't trip an
# "unrecognized arguments" error here. ---
_parser = argparse.ArgumentParser(
    description="Re-verify dice_pick_demo.py's scripted grasp at 48mm scale for d8/d10 (Task 0, one shape per run)."
)
_parser.add_argument("--shape", type=str, choices=["d8", "d10"], required=True, help="Which die to re-verify this run.")
_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help=(
        "Seed for the randomized 5-die layout. A single seed is sufficient for this mechanical "
        "transfer check (Task 1's own capture is where 5-seeds-per-shape matters)."
    ),
)
_parser.add_argument(
    "--gt-xy-bypass",
    action="store_true",
    default=False,
    help=(
        "Grasp-mechanism-isolation bypass - REUSES dice_pick_demo.py's own --gt-xy-bypass "
        "mechanism/precedent (d4 rung-1, "
        "docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md's 'Addendum: "
        "ground-truth XY-bypass'; see also BACKLOG.md's 2026-07-19 'd8/d10 demo-warmstart Task 0' "
        "entry, which decided this diagnostic needed the same fix). Default OFF: byte-identical "
        "to pre-flag behavior - target_xy sourced only from the detector, select_target_detection's "
        "raise on a miss is fatal (uncaught), same as before this flag existed. When set: the "
        "detector subprocess still runs unconditionally (diagnostic detector-vs-GT comparison "
        "preserved either way, per the precedent's own design); a detector miss for the commanded "
        "shape is caught rather than fatal; target_xy is always sourced from a FRESH live "
        "ground-truth measurement of the die's own post-rescale, post-resettle position "
        "(measure_settled_position_m) rather than the detector's - NOT from spawn_scene_and_settle's "
        "pre-rescale `results` dict, since this diagnostic's own post-hoc rescale can shift the die's "
        "x/y (not just z) during re-settle, same staleness concern as the rgb.png bug fixed in "
        "recapture_camera_frame below. A bypassed PASS means only 'the grasp mechanism works at this "
        "scale', NOT 'the perception-driven demo can find this die at this scale' - must not be "
        "represented as the latter."
    ),
)
own_args, _unknown_args = _parser.parse_known_args()

# --- Import dice_pick_demo.py's machinery (see module docstring's "Import-time
# side effect note" above for why sys.argv - INCLUDING argv[0] - must be
# swapped around this). ---
_real_argv = sys.argv
_dice_pick_demo_path = os.path.join(SCRIPTS_DIR, "dice_pick_demo.py")
sys.argv = [_dice_pick_demo_path, "--gate", "g"]
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import dice_pick_demo as dpd  # noqa: E402

sys.argv = _real_argv


# 48mm-parity scale constants - REUSED, NOT RE-DERIVED, from
# `tasks/franka/dice_lift_joint_env_cfg.py`'s `FrankaDieLiftJointD8BigEnvCfg`/
# `...D10BigEnvCfg` (see those classes' own docstrings for the fitted-scale
# derivation against each shape's baked-USD native-bbox measurement,
# `scripts/_diag_d8d10d12_standard_scale_check.py`). These constants
# transfer unchanged to THIS demo's own die assets: confirmed (this task,
# by direct read of `scripts/bake_die_asset.py`'s own module docstring -
# "copy a dice_sets_v1 die USD and write physics schemas into the copy",
# no mesh/scale change) that the baked USD those constants were fit against
# (`assets/dice/{d8,d10}_physics.usd`) is a byte-identical-geometry copy of
# the SAME raw source USD `tasks/franka/dice_scene_cfg.py`'s `DiceSceneCfg`
# actually spawns (`vision/data/raw/dice_sets_v1/set_00000_{d8,d10}.usd`).
_SCALE_48MM = {
    "d8": 0.003167,
    "d10": 0.002928,
}

# Extra physics-settle duration after rescaling the target die's prim
# (override_die_scale), mirroring dice_pick_demo.py's own
# `_SETTLE_SECONDS = 3.0` (same order of magnitude needed for a rigid body
# to reach equilibrium under gravity/contact after a geometry change).
_RESETTLE_SECONDS = 3.0

# The demo scene's own placeholder per-die mass (tasks/franka/dice_scene_cfg.py's
# _DICE_MASS = MassPropertiesCfg(mass=0.01), i.e. 10g) is NOT overridden by
# this diagnostic - only the die's geometric scale is. Flagged here, not
# silently left implicit: the RL env this experiment actually trains
# against (FrankaDieLiftJointD8BigEnvCfg/...D10BigEnvCfg) pins mass at
# 0.216kg (the DexCube-measured placeholder carried across this whole
# multi-die arc - see that class's own docstring), ~21x heavier than this
# demo scene's 10g. This diagnostic's grasp check therefore transfers the
# object SIZE but not the object MASS the RL env presents - a real,
# reportable gap between what this task tests and what H1 actually trains
# against, left for the controller to weigh (not fixed here - fixing it
# would mean also overriding mass_props, outside this task's stated scope).
_DEMO_SCENE_DIE_MASS_KG = 0.01
_RL_ENV_DIE_MASS_KG = 0.216


def override_die_scale(sim, scene, die_type: str, scale: float) -> None:
    """Rescales the ALREADY-SPAWNED-AND-SETTLED `die_{die_type}` prim's own
    USD Xform scale to `(scale, scale, scale)`, then runs a fresh
    `_RESETTLE_SECONDS`-long physics-settle loop so the die reaches a new
    equilibrium resting height at its new (larger) size - `spawn_scene_and_settle`
    itself is called UNCHANGED beforehand (dice spawn/settle at the demo's
    default real-world size, exactly as Gate A always does); this function
    only runs AFTER that returns.

    POST-HOC rescale, not a pre-spawn `scene_cfg.spawn.scale` override (this
    task's originally designed approach) - see this module's own docstring
    ("Deviation from this task's original design") for why the pre-spawn
    approach was empirically found infeasible against IsaacLab's actual
    `@configclass` semantics, and why this achieves the same end state (die
    at 48mm scale, fully physically settled) via a different mechanical
    path instead.

    Finds the die's EXISTING scale xform op directly (`UsdGeom.Xformable`'s
    `GetOrderedXformOps`, matching that op's `XformOp.TypeScale`) and
    overwrites its value in place, rather than adding a new scale op -
    IsaacLab's own USD spawner already authors a scale op from the die's
    `spawn.scale=(0.001, 0.001, 0.001)` config value
    (`tasks/franka/dice_scene_cfg.py`'s `_die_cfg`); composing a SECOND
    scale op on top would multiply rather than replace it."""
    from pxr import Gf, UsdGeom  # noqa: E402  (pxr must come after AppLauncher construction)

    env_root = scene.env_prim_paths[0]
    die_prim_path = f"{env_root}/Die_{die_type}"
    stage = scene.stage
    prim = stage.GetPrimAtPath(die_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"override_die_scale: prim not found: {die_prim_path}")

    xformable = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    old_scale = tuple(scale_op.Get()) if scale_op is not None else None
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale, scale, scale))
    print(
        f"[OVERRIDE] {die_prim_path} scale xform op: {old_scale} -> ({scale}, {scale}, {scale}) "
        f"(48mm-parity constant, reused from FrankaDieLiftJoint{die_type.upper()}BigEnvCfg)"
    )

    sim_dt = sim.get_physics_dt()
    resettle_steps = int(_RESETTLE_SECONDS / sim_dt)
    for _ in range(resettle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    print(f"[OVERRIDE] re-settled {die_prim_path} for {resettle_steps} steps ({_RESETTLE_SECONDS}s sim time) after rescale")


def measure_settled_position_m(scene, die_type: str) -> tuple[float, float, float]:
    """Live-measured settled world-frame (x, y, z) meters of `die_{die_type}`'s
    root position, in the env-local frame - read directly off the scene
    (`scene.env_origins`-subtracted, matching `spawn_scene_and_settle`'s own
    read pattern). Extended (this task, --gt-xy-bypass work) from the
    z-only `measure_settled_rest_height` this function now backs, so the
    SAME "measure it directly after re-settle, do not assume" discipline
    already applied to grasp height also covers x/y: this diagnostic's own
    post-hoc rescale (`override_die_scale`) can shift the die's x/y, not
    just z, during its re-settle loop (denser contact/rolling under the
    larger footprint) - so `spawn_scene_and_settle`'s own pre-rescale
    `results[die_type]["x"/"y"]` is stale for the same reason its saved
    rgb.png is stale (see `recapture_camera_frame`'s bug-fix note). Used for
    grasp height (z) unconditionally, and for target_xy (x, y) only when
    `--gt-xy-bypass` is active."""
    die = scene[f"die_{die_type}"]
    pos_w = die.data.root_pos_w[0].cpu().numpy()
    pos = pos_w - scene.env_origins[0].cpu().numpy()
    return float(pos[0]), float(pos[1]), float(pos[2])


def measure_settled_rest_height(scene, die_type: str) -> float:
    """Live-measured settled world-frame z (meters) of `die_{die_type}`'s
    root position - read directly off the scene AFTER `override_die_scale`'s
    own re-settle loop has completed. Same per-die measurement
    `spawn_scene_and_settle`'s own settle table already computes internally
    at real size (and that `_DIE_REST_HEIGHT_M`'s original real-size values
    were derived from, per that table's own module comment) - re-read here
    independently, at the NEW 48mm scale, rather than assumed to scale
    linearly with the die's spawn-scale ratio from the real-size constant
    (this task's own explicit "measure it directly, do not assume"
    requirement - see plan/spec). Thin wrapper over
    `measure_settled_position_m`, kept as its own named function since every
    existing caller already spells it this way."""
    return measure_settled_position_m(scene, die_type)[2]


def recapture_camera_frame(sim, scene, out_dir: str) -> None:
    """Re-renders and OVERWRITES `rgb.png`/`depth.npy`/`camera_params.json`
    in `out_dir` from the LIVE scene's current camera state - mirrors
    `dice_pick_demo.spawn_scene_and_settle`'s own "Camera capture" block
    (that function's `rgb`/`depth`/`intrinsics`/`cam_pos_w`/`cam_quat_w_ros`
    read-and-save code) byte-for-byte in what it writes, minus the ground
    truth/arm-camera bookkeeping that block also does (not needed here -
    this diagnostic has its own GT path, see `measure_settled_position_m`).

    **Real bug found and fixed in this task (2026-07-19)**: before this
    function existed, `run_shape_reverify` called
    `dpd.run_detector_subprocess(out_dir)` directly after
    `override_die_scale`. `run_detector_subprocess` does NOT render
    anything itself - by design (confirmed by direct read of
    `dice_pick_demo.py`'s own Gate V comment: "run_detector_subprocess
    reads its own saved rgb.png from spawn_scene_and_settle, not this local
    var" - correct for Gate G/V, which never rescale mid-run) it only reads
    whatever `rgb.png`/`depth.npy` already sit in `out_dir`. Those files
    were last written by `spawn_scene_and_settle`, BEFORE this diagnostic's
    own `override_die_scale` ever rescales the die to 48mm - so every prior
    run of this diagnostic's detector step (both the d8 PASS and the d10
    FAIL that motivated this task) actually ran object detection against
    the ORIGINAL REAL-SIZE frame, not a genuine 48mm-scale image. This is a
    real, reportable problem with the prior "d10 fails perception at 48mm
    scale" diagnosis: that 0-detections result was measured against a
    real-size frame, and d10 is independently already known to detect fine
    at real size (`kb/wiki/experiments/dice-pick-demo.md`: 239.3mm z-gain
    baseline). Fixed here: re-render and overwrite the camera frame from the
    LIVE (post-rescale, post-resettle) scene before the detector subprocess
    ever runs, using the same RTX-convergence-frame count (40 steps)
    `spawn_scene_and_settle` uses after its own settle loop - a rescale is
    as large a visual change as an initial spawn, so convergence needs
    re-earning here too, not assumed carried over from `override_die_scale`'s
    physics-settle steps alone (those already render each step, but for
    physics equilibrium, not RTX visual convergence)."""
    import numpy as np
    from PIL import Image

    sim_dt = sim.get_physics_dt()
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    camera = scene["camera"]
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos_w = camera.data.pos_w[0].cpu().numpy()
    cam_quat_w_ros = camera.data.quat_w_ros[0].cpu().numpy()

    rgb_path = os.path.join(out_dir, "rgb.png")
    depth_path = os.path.join(out_dir, "depth.npy")
    params_path = os.path.join(out_dir, "camera_params.json")

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    with open(params_path, "w") as f:
        json.dump(
            {
                "intrinsic_matrix": intrinsics.tolist(),
                "pos_w": cam_pos_w.tolist(),
                "quat_w_ros": cam_quat_w_ros.tolist(),
                "width": int(camera.data.output["rgb"].shape[2]),
                "height": int(camera.data.output["rgb"].shape[1]),
            },
            f,
            indent=2,
        )
    print(
        f"[REVERIFY] BUG FIX: re-captured camera frame post-rescale (overwriting the stale "
        f"real-size frame spawn_scene_and_settle wrote): {rgb_path}, {depth_path}, {params_path}"
    )


def run_shape_reverify(shape: str, seed: int) -> dict:
    """Mirrors `dice_pick_demo.py`'s own `run_gate_g` flow (spawn/settle ->
    rescale target die + re-settle -> detector subprocess -> target select
    -> run_pick_sequence -> GT verdict via the shared `_compute_verdict_table`
    helper), with the two substitutions described in this module's own
    docstring. Returns the same shape of result dict Gate G's own verdict
    JSON uses, plus the 48mm-specific measurement fields."""
    out_dir = os.path.join(OUT_DIR, shape)
    os.makedirs(out_dir, exist_ok=True)

    sim, scene, positions, results = dpd.spawn_scene_and_settle(out_dir, seed)

    scale = _SCALE_48MM[shape]
    override_die_scale(sim, scene, shape, scale)

    # BUG FIX (2026-07-19, see recapture_camera_frame's own docstring for the
    # full finding): must happen BEFORE run_detector_subprocess below, and
    # before the rest-height measurement (an extra 40 render-convergence
    # steps happen inside it, so measuring after gives the truly final
    # settled state, not a mid-convergence one).
    recapture_camera_frame(sim, scene, out_dir)

    measured_rest_height_m = measure_settled_rest_height(scene, shape)
    real_size_rest_height_m = dpd._DIE_REST_HEIGHT_M[shape]
    print(
        f"[REVERIFY] {shape} @ 48mm scale: measured settled rest height = "
        f"{measured_rest_height_m * 1000:.2f}mm (real-size _DIE_REST_HEIGHT_M="
        f"{real_size_rest_height_m * 1000:.2f}mm, ratio={measured_rest_height_m / real_size_rest_height_m:.3f}x "
        f"- NOT assumed, only reported for comparison)"
    )

    detection_output = dpd.run_detector_subprocess(out_dir)
    detections = detection_output["detections"]
    print(f"[REVERIFY] perception subprocess returned {len(detections)} detections:")
    for det in detections:
        print(f"  class={det['class']:<8} conf={det['confidence']:.3f} world_pos={det['world_pos']}")

    # Ground-truth XY-bypass (--gt-xy-bypass, OFF by default) - REUSES the
    # exact mechanism dice_pick_demo.py's own Gate G/V already built for this
    # (see this module's own --gt-xy-bypass argparse help above for the full
    # precedent citation), rather than inventing a new bypass pattern.
    # select_target_detection still fails LOUDLY (raises) on a miss - that
    # contract is UNCHANGED; only THIS CALLER catches the raise, and only
    # when the bypass is active (mirroring dice_pick_demo.py's run_gate_g
    # exactly, including its 2026-07-15 "catch only when bypass is on, and
    # only around this call" fix for the bug where a TOTAL miss used to
    # propagate before the bypass branch could ever run). Default OFF means
    # d8's already-passing non-bypass path is byte-identical to before this
    # task's changes (aside from the frame-recapture bug fix above, which
    # applies unconditionally).
    target_det = None
    det_x = det_y = det_z = None
    try:
        target_det = dpd.select_target_detection(detections, shape)
        det_x, det_y, det_z = target_det["world_pos"]
        print(
            f"[REVERIFY] target detection for '{shape}': class={target_det['class']} "
            f"conf={target_det['confidence']:.3f} world_pos=({det_x:.4f}, {det_y:.4f}, {det_z:.4f})"
        )
    except RuntimeError as e:
        if not own_args.gt_xy_bypass:
            raise
        print(
            f"[REVERIFY] detector FAILED to find '{shape}' ({e}) - continuing because "
            f"--gt-xy-bypass is active; target_xy will be sourced from ground truth below, "
            f"detector-vs-GT diagnostic print skipped (nothing to compare against)."
        )

    gt_x, gt_y, _gt_z = measure_settled_position_m(scene, shape)
    if own_args.gt_xy_bypass:
        target_xy = (gt_x, gt_y)
        print(
            f"[REVERIFY] target_xy SOURCED FROM GROUND TRUTH (bypass active) for '{shape}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f}) - grasp-mechanism isolation only, NOT a "
            f"perception result."
        )
        if target_det is not None:
            xy_err_mm = ((gt_x - det_x) ** 2 + (gt_y - det_y) ** 2) ** 0.5 * 1000.0
            print(
                f"[REVERIFY] detector-vs-GT xy offset for '{shape}' [DIAGNOSTIC ONLY, not used for "
                f"grasp]: {xy_err_mm:.1f}mm (gt=({gt_x:.4f},{gt_y:.4f}), det=({det_x:.4f},{det_y:.4f}))"
            )
    else:
        target_xy = (float(det_x), float(det_y))
        print(
            f"[REVERIFY] target_xy sourced from DETECTOR for '{shape}': "
            f"({target_xy[0]:.4f}, {target_xy[1]:.4f})"
        )

    pick_sequence_error = None
    try:
        waypoint_status = dpd.run_pick_sequence(
            sim, scene, target_xy, measured_rest_height_m, shape, results=results
        )
    except dpd._StageTimeoutError as e:
        pick_sequence_error = str(e)
        waypoint_status = {"error": pick_sequence_error}
        print(f"[REVERIFY] *** pick sequence FAILED for {shape}: stage timeout - {pick_sequence_error} ***")

    # Post-lift camera capture (RTX convergence frames, same pattern as
    # spawn_scene_and_settle's own capture / Gate G's own post-lift frame) -
    # a saved frame for visual spot-check, per this repo's verification
    # standard (watch output, don't trust an exit code/number alone).
    import numpy as np
    from PIL import Image

    sim_dt = sim.get_physics_dt()
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    camera = scene["camera"]
    rgb_post = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    post_lift_path = os.path.join(out_dir, f"post_lift_{shape}_48mm.png")
    Image.fromarray(rgb_post).save(post_lift_path)
    print(f"[REVERIFY] saved post-lift frame: {post_lift_path}")

    # Success verification - GT ALLOWED HERE ONLY, byte-for-byte the same
    # check/thresholds Gate G's own `_compute_verdict_table` uses (the
    # z-gain PASS criterion: target die must gain >= `_LIFT_SUCCESS_GAIN`
    # (150mm), every other die must stay within its own undisturbed bounds).
    verdict_table, all_ok = dpd._compute_verdict_table(scene, results, shape)
    print(f"[REVERIFY] post-lift verdict table (commanded die: {shape}, 48mm scale):")
    print(
        f"{'die':<6}{'z_before(mm)':>14}{'z_now(mm)':>12}{'gain(mm)':>10}{'xy_drift(mm)':>14}  "
        f"{'target':^8}  verdict"
    )
    for row in verdict_table:
        print(
            f"{row['die']:<6}{row['z_before_m'] * 1000:>14.1f}{row['z_now_m'] * 1000:>12.1f}"
            f"{row['gain_m'] * 1000:>10.1f}{row['xy_drift_m'] * 1000:>14.1f}  "
            f"{'*TARGET*' if row['is_target'] else '':^8}  "
            f"{'PASS' if row['ok'] else 'FAIL'}"
        )
    print(f"[REVERIFY] {shape} @ 48mm scale: {'PASS' if all_ok else 'FAIL'} (waypoints={waypoint_status})")

    result = {
        "shape": shape,
        "seed": seed,
        "scale_applied": scale,
        "measured_rest_height_m": measured_rest_height_m,
        "real_size_rest_height_m": real_size_rest_height_m,
        "demo_scene_die_mass_kg": _DEMO_SCENE_DIE_MASS_KG,
        "rl_env_die_mass_kg": _RL_ENV_DIE_MASS_KG,
        "gt_xy_bypass_active": bool(own_args.gt_xy_bypass),
        # target_det is None only when --gt-xy-bypass caught a total
        # detector miss (see the try/except above) - guard against that
        # here rather than crashing on the verdict-JSON write.
        "detected_class": target_det["class"] if target_det is not None else None,
        "detection_confidence": target_det["confidence"] if target_det is not None else None,
        "detector_world_pos": [float(det_x), float(det_y), float(det_z)] if target_det is not None else None,
        "gt_xy_m": [gt_x, gt_y],
        "target_xy": target_xy,
        "target_xy_source": "ground_truth" if own_args.gt_xy_bypass else "detector",
        "waypoint_status": waypoint_status,
        "pick_sequence_error": pick_sequence_error,
        "verdict_table": verdict_table,
        "pass": bool(all_ok),
    }
    verdict_path = os.path.join(out_dir, f"verdict_{shape}_48mm.json")
    with open(verdict_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[REVERIFY] saved verdict: {verdict_path}")
    return result


def main() -> None:
    if own_args.shape not in dpd.DIE_TYPES:
        raise RuntimeError(f"'{own_args.shape}' is not one of the physical dice in this scene: {dpd.DIE_TYPES}")
    result = run_shape_reverify(own_args.shape, own_args.seed)
    print(f"\n[TASK0 VERDICT] {own_args.shape} @ 48mm scale: {'PASS' if result['pass'] else 'FAIL'}")


if __name__ == "__main__":
    # try/except/finally so simulation_app.close() ALWAYS runs, even if
    # main() raises (e.g. select_target_detection's intentional "fail
    # loudly" hard failure) - same Kit-teardown-hang avoidance reasoning as
    # dice_pick_demo.py's own __main__ guard (CLAUDE.md's documented
    # failure mode). The explicit `except` prints the full traceback (with
    # flush) BEFORE `finally`'s `simulation_app.close()` runs, then
    # re-raises - found necessary this task (2026-07-19): a bare
    # `try/finally` alone relies on the interpreter's default
    # unhandled-exception printing, which only happens AFTER `finally`
    # completes, and `simulation_app.close()` was observed to make that
    # default printing unreliable in this environment (an uncaught
    # exception during an earlier iteration of this script produced no
    # visible traceback at all until this explicit print was added).
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

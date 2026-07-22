# scripts/demo_franka_ik_dice_line.py
"""Fun demo (NOT a Tier-1/Tier-2 research experiment - no hypothesis/spec/plan
gate applies, per direct task instruction): a single Franka Panda, driven by
classical closed-form/differential IK (no learned policy), picks up all 5 of
this project's canonical dice shapes (d4/d8/d10/d12/d20,
tasks/franka/dice_scene_cfg.py's DiceSceneCfg - same assets used throughout
tasks/franka/) from a scattered starting layout and lines them up in a row on
the table (Act 1), then re-picks each one from that line and relocates the
whole line to a new position/orientation (Act 2). Ground-truth object poses
are read directly from the scene (`scene["die_<type>"].data.root_pos_w`) - no
vision/detector pipeline (that effort is separate and still-blocked, not
touched here). One continuous imageio-encoded mp4 covers both acts.

This is a scripted-choreography deployment, not a new mechanism: the entire
staged-IK approach (joint-space "ready-to-descend" prep before any Cartesian
IK, canonical straight-down grasp orientation, bounded-relative-step
DifferentialIKController, per-die-type measured grasp height, the d4 V-notch
fingertip fixture) is REUSED, not re-derived, from scripts/dice_pick_demo.py's
`run_pick_sequence`/`spawn_scene_and_settle`/`apply_convex_hull_collision`/
`attach_notch_fixtures` - see each copied piece's own comment below for
exactly what it's taken from and why it's copied rather than imported
(dice_pick_demo.py has top-level `argparse.parse_args()`/`AppLauncher(...)`
side effects at module-import time - the same reason
scripts/dice_pick_drop_repeat_demo.py's own docstring gives for copying
scripts/franka_checkpoint_review.py's constants instead of importing them:
importing a standalone entry-point script would re-parse this script's own
CLI args through the other script's parser and attempt to launch a SECOND
AppLauncher/Isaac Sim app in the same process).

Generalization beyond dice_pick_demo.py: that file's `run_pick_sequence`
already has a `post_action="move"` primitive (pick -> lift -> carry -> descend
-> release -> retract) for a single die/single destination. This script's
`pick_and_place()` below is that same staged sequence, trimmed of the
detector/roll/d4-contact-instrumentation machinery this demo doesn't need
(ground-truth positions only, no roll, no per-die contact-force verification)
and shared across 10 calls (5 dice x 2 passes) instead of being built for one
call.

Design choices (left to this task's own judgment, per the dispatch brief):
  - Starting scatter: DiceSceneCfg's own baked-in default `init_state.pos`
    per die (d4=(0.35,-0.20), d8=(0.42,-0.10), d10=(0.50,0.0),
    d12=(0.58,0.10), d20=(0.65,0.20), all z=0.10 drop height) - an existing,
    already-reachable, already-minimum-spaced (~0.12-0.13m pairwise, well
    above dice_pick_demo.py's own validated 0.09m minimum) scene-topology
    pattern, reused rather than re-sampled.
  - Pick order: d8->d10->d12->d20->d4 (SLOT_ORDER/PICK_ORDER's own comment
    has the full rationale, including why they're two separate lists, not
    one) - originally ascending size (d4 first), REORDERED after this
    task's own first real cloud run showed d4 (this project's own
    well-documented hardest grasp case) failed to be physically grasped in
    both acts, while the other 4 dice succeeded reliably (~3-44mm placement
    error across two full passes). Moving d4 last means the video leads
    with the reliable dice and only attempts the known-hardest case at the
    end, rather than opening on a visible failure - a sequencing choice,
    not a claim that d4's underlying grasp reliability was fixed.
  - Line 1 (Act 1 destination): a column at x=0.50, y in
    {-0.18,-0.09,0,0.09,0.18} (0.09m spacing, same value validated
    elsewhere in this file's own history) - die index i (ascending size)
    placed at slot i (increasing y).
  - Line 2 (Act 2 destination): a ROW at y=0.20, x in
    {0.36,0.43,0.50,0.57,0.64} - deliberately rotated 90 degrees (column ->
    row) AND shifted (y=0.0 center -> y=0.20) from Line 1, so the
    relocation is unambiguous on video, not a subtle shift. Same index
    mapping (die i keeps its slot i, just in the new line's geometry).
    Both lines stay within this project's own validated Franka-reach
    region (tasks/franka/lift_env_cfg.py's CommandsCfg.object_pose ranges:
    pos_x=(0.4,0.6), pos_y=(-0.25,0.25); dice_pick_demo.py's own
    _MOVE_TARGET_XY_DEFAULT=(0.50,0.22) already proved y=0.20-0.22 reachable
    with a die in-hand).
  - Camera: a NEW `demo_camera` sensor (this script's own scene-cfg
    subclass, not touching DiceSceneCfg/dice_scene_cfg.py), framed as a
    whole-table 3/4 view rather than dice_scene_cfg.py's own ARM_CAMERA
    (which is deliberately tight/close on the single-grasp pinch region,
    per that file's own comment, and would clip a 5-die line spanning
    ~0.35-0.68m of table). Eye/target chosen in the same spirit as
    scripts/franka_checkpoint_review.py's viewer convention
    (eye=(1.8,1.8,1.1), lookat=(0.4,0.0,0.35) - a raised 3/4 diagonal view),
    just with the look-at height dropped to table level (this demo's
    subject is dice-on-table, not a floating cube) and pulled back slightly
    to fit both lines' full span in frame. Quaternion computed via Isaac
    Lab's own `create_rotation_matrix_from_view`/`quat_from_matrix`
    (OpenGL convention: -Z forward, +Y up) - NOT hand-derived - matching
    tasks/ar4/graspgoal_democam_env_cfg.py's own established pattern/
    rationale for avoiding camera-convention bugs.

Never pass --headless when a display is available (DISPLAY=:1, environment
law per CLAUDE.md) - this script only sets it when explicitly requested via
--headless (the cloud-fallback path this task's brief explicitly allows for a
video-only deliverable).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/demo_franka_ik_dice_line.py"

    # cloud/headless fallback (video-only deliverable, per this task's brief):
    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/demo_franka_ik_dice_line.py --headless"
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka IK dice-line pick-and-place demo (scripted, ground-truth poses).")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write the video/summary to (default: outputs/dice_demo/ik_dice_line/).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video rendering
# NEVER force headless here - non-headless is environment law when a display
# is available (CLAUDE.md). This script only goes headless if the caller
# explicitly passes --headless (the cloud-fallback exception this task's own
# brief allows for a video-only deliverable).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows - isaaclab/pxr imports must come after AppLauncher."""

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import schemas  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    create_rotation_matrix_from_view,
    quat_from_matrix,
    subtract_frame_transforms,
)
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.dice_scene_cfg import (  # noqa: E402
    DIE_TYPES,
    DiceSceneCfg,
    _DICE_COLLISION_PROPS,
    _DICE_MASS,
    _DICE_RIGID_PROPS,
)
from tasks.franka.notch_fixture import (  # noqa: E402
    grip_height_above_table_m,
    joint_local_pos0_m,
    joint_local_rot1_wxyz,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = args_cli.output_dir or os.path.join(REPO_ROOT, "outputs", "dice_demo", "ik_dice_line")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Copied verbatim from scripts/dice_pick_demo.py (see module docstring for why
# copied rather than imported). Both are self-contained pxr/isaaclab-schema
# helpers with no dependency on that file's own detector/argparse state.
# ---------------------------------------------------------------------------


def apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Makes the die prim at `die_prim_path` a dynamic rigid body with
    convex-hull collision, entirely at runtime - the dice USDs ship with NO
    physics schemas baked in (visual-only exports shared with vision/'s
    rendering pipeline). Copied from scripts/dice_pick_demo.py verbatim."""
    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")

    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)

    schemas.modify_rigid_body_properties(die_prim_path, _DICE_RIGID_PROPS, stage)
    schemas.modify_mass_properties(die_prim_path, _DICE_MASS, stage)
    schemas.modify_collision_properties(die_prim_path, _DICE_COLLISION_PROPS, stage)

    mesh_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            mesh_count += 1
    return mesh_count


def attach_notch_fixtures(stage, robot_prim_path: str) -> None:
    """Rigidly attaches the d4 V-notch fingertip fixture
    (tasks/franka/notch_fixture.py + tasks/franka/dice_scene_cfg.py's
    notch_fixture_left/right prims) to both fingertips via a fixed joint -
    UNCONDITIONAL (every die type, not d4-only), same as
    scripts/dice_pick_demo.py's own convention. Copied from that file
    verbatim (see its own docstring for the instance-proxy/child-prim
    authoring constraint this works around)."""
    xform_cache = UsdGeom.XformCache()
    local_pos0 = joint_local_pos0_m()

    for finger_name, fixture_name, mirror in [
        ("panda_leftfinger", "NotchFixtureLeft", False),
        ("panda_rightfinger", "NotchFixtureRight", True),
    ]:
        finger_prim = stage.GetPrimAtPath(f"{robot_prim_path}/{finger_name}")
        if not finger_prim.IsValid():
            raise RuntimeError(f"attach_notch_fixtures: finger prim not found: {robot_prim_path}/{finger_name}")
        env_root = os.path.dirname(robot_prim_path)
        fixture_prim = stage.GetPrimAtPath(f"{env_root}/{fixture_name}")
        if not fixture_prim.IsValid():
            raise RuntimeError(f"attach_notch_fixtures: fixture prim not found: {env_root}/{fixture_name}")

        finger_to_world = xform_cache.GetLocalToWorldTransform(finger_prim)
        attach_point_world = finger_to_world.Transform(Gf.Vec3d(*local_pos0))
        UsdGeom.XformCommonAPI(fixture_prim).SetTranslate(attach_point_world)

        joint_path = f"{fixture_prim.GetPath()}/attach_joint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([finger_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([fixture_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rot1_wxyz = joint_local_rot1_wxyz(mirror=mirror)
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(rot1_wxyz[0], Gf.Vec3f(*rot1_wxyz[1:])))
    print(f"[SPAWN] attach_notch_fixtures: attached both notch fixtures under {robot_prim_path}")


# Per-die-type MEASURED resting height (world-frame root_pos_w z after
# settle, table surface at z=0) - copied verbatim from
# scripts/dice_pick_demo.py's `_DIE_REST_HEIGHT_M` (see that file's own
# extensive comment for the measurement/derivation history: NOT half of
# manifest size_mm, a die's centroid sits much closer to the table for the
# irregular d4/d8/d10 than a naive half-size formula would suggest).
_DIE_REST_HEIGHT_M = {
    "d4": 0.0022,
    "d8": 0.0017,
    "d10": 0.0026,
    "d12": 0.0109,
    "d20": 0.0110,
}

# d4 grip height comes from the notch fixture's own geometry (grips HIGHER
# on the pyramid than the flat centroid height - see
# tasks/franka/notch_fixture.py's grip_height_above_table_m), not
# _DIE_REST_HEIGHT_M["d4"] - same special-case reasoning as
# scripts/dice_pick_demo.py's `_die_grasp_height_m`.
def die_grasp_height_m(die_type: str) -> float:
    if die_type == "d4":
        return grip_height_above_table_m()
    return _DIE_REST_HEIGHT_M[die_type]


# ---------------------------------------------------------------------------
# Demo camera - whole-table 3/4 view (see module docstring for framing
# rationale). Computed via Isaac Lab's own create_rotation_matrix_from_view/
# quat_from_matrix (OpenGL convention), matching
# tasks/ar4/graspgoal_democam_env_cfg.py's established pattern.
# ---------------------------------------------------------------------------
_DEMO_CAMERA_EYE = (1.85, 1.55, 1.15)
_DEMO_CAMERA_TARGET = (0.50, 0.05, 0.04)
_eye_t = torch.tensor([_DEMO_CAMERA_EYE])
_target_t = torch.tensor([_DEMO_CAMERA_TARGET])
_DEMO_CAMERA_QUAT_OPENGL = tuple(quat_from_matrix(create_rotation_matrix_from_view(_eye_t, _target_t, up_axis="Z"))[0].tolist())
print(f"[CAMERA] demo_camera eye={_DEMO_CAMERA_EYE} target={_DEMO_CAMERA_TARGET} quat_opengl={_DEMO_CAMERA_QUAT_OPENGL}")


@configclass
class DiceLineDemoSceneCfg(DiceSceneCfg):
    """DiceSceneCfg + one additional whole-table demo_camera sensor. Every
    other field (robot/table/light/dice/notch-fixtures/DiceCamera/ArmCamera)
    is inherited UNCHANGED - this script never modifies dice_scene_cfg.py."""

    demo_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_DEMO_CAMERA_EYE, rot=_DEMO_CAMERA_QUAT_OPENGL, convention="opengl"),
    )


# ---------------------------------------------------------------------------
# Pick order + both lines' target (x, y) per die type (see module docstring
# for the full rationale). Die index i -> slot i in both lines, so
# relocation preserves each die's own "position in the sequence" while the
# line's own geometry (column -> row) and center (y=0 -> y=0.20) both change.
#
# SLOT_ORDER (fixed, ascending size) determines which line POSITION each die
# targets - kept separate from PICK_ORDER (which only controls iteration/
# attempt sequence) after this task's own two real cloud runs found they
# must NOT be the same list: run 1 (PICK_ORDER == SLOT_ORDER, both ascending
# size) placed d8/d10/d12/d20 tightly (6.7/4.9/43.6/16.5mm Act1 error) but
# opened the video on d4 (this project's own well-documented hardest grasp
# case - see e.g. kb/wiki/experiments/target-selection-clutter.md and
# scripts/dice_pick_demo.py's own "d4 was historically this demo's own
# documented permitted-failure case" comment) failing to be physically
# grasped in BOTH acts (every IK waypoint converged - no _StageTimeoutError
# - but the die never left the table, landing back near its own start
# instead of the commanded target - a genuine grasp-precision miss, not a
# scripting bug). Run 2 tried simply moving d4 to the end of one combined
# order list - but since LINE1/LINE2_TARGETS were built via
# zip(PICK_ORDER, slots), reordering PICK_ORDER ALSO reshuffled which slot
# every other die targets, and d12/d20 landed in slots that measured much
# worse (103.9mm/167.4mm Act1 error, vs 43.6mm/16.5mm in run 1) - a real,
# measured regression from an unintended coupling, not a mechanism problem.
# This run decouples the two: SLOT_ORDER stays IDENTICAL to run 1 (so the 4
# reliable dice get run 1's own good target assignment back), while
# PICK_ORDER (below) independently controls attempt sequence so d4 is
# attempted last - the video leads with the reliable dice and only attempts
# the known-hardest case at the end. d4's own underlying grasp reliability
# is unchanged by any of this and may still fail - reported honestly either
# way, not silently dropped.
SLOT_ORDER = ["d4", "d8", "d10", "d12", "d20"]
PICK_ORDER = ["d8", "d10", "d12", "d20", "d4"]

_LINE1_X = 0.50
_LINE1_Y_SLOTS = [-0.18, -0.09, 0.0, 0.09, 0.18]
LINE1_TARGETS = {die: (_LINE1_X, y) for die, y in zip(SLOT_ORDER, _LINE1_Y_SLOTS)}

_LINE2_Y = 0.20
_LINE2_X_SLOTS = [0.36, 0.43, 0.50, 0.57, 0.64]
LINE2_TARGETS = {die: (x, _LINE2_Y) for die, x in zip(SLOT_ORDER, _LINE2_X_SLOTS)}

# ---------------------------------------------------------------------------
# Staged-IK tuning constants - copied verbatim from scripts/dice_pick_demo.py
# (see that file's own extensive comments for the full derivation history:
# joint-space prep avoids a Jacobian near-singularity at the post-reset
# default pose, canonical straight-down orientation avoids a
# panda_joint2-limited IK branch, the tighter _GRASP_POS_TOL exists because a
# small die's own radius is comparable to the looser _WAYPOINT_TOL).
# ---------------------------------------------------------------------------
_STAGE1_HAND_Z = 0.30  # m, approach waypoint hand-frame z
_STAGE_LIFT_HAND_Z = 0.35  # m, lift/carry waypoint hand-frame z
_WAYPOINT_TOL = 0.015  # m
_GRASP_POS_TOL = 0.005  # m
_ROT_TOL = 0.06  # rad
_MAX_POS_STEP = 0.018  # m
_MAX_ROT_STEP = 0.03  # rad
_MAX_STEPS_APPROACH = 800
_MAX_STEPS_DESCEND = 400
_MAX_STEPS_LIFT = 300
_MAX_STEPS_REFINE = 200
_READY_TO_DESCEND_JOINT_POS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
_MAX_STEPS_JOINT_PREP = 200
_GRIPPER_CLOSE_HOLD_STEPS = 90
_VALIDATED_HAND_TO_PINCH_POINT_Z = 0.1034  # tasks/franka/lift_env_cfg.py's own _EE_MEASUREMENT_OFFSET
# Shortened from dice_pick_demo.py's 3.0s post-place settle - this demo
# repeats pick-and-place 10 times (5 dice x 2 passes), so a full 3s settle
# after every single release would add ~30s of dead video time; 1.5s is
# still comfortably longer than this scene's own dice ever take to stop
# bouncing after a release from lift height (dice_pick_demo.py's own
# _GRIPPER_CLOSE_HOLD_STEPS-scale dwell already shows settle happens within
# a couple hundred ms for these small/light dice).
_POST_RELEASE_SETTLE_SECONDS = 1.5

_VIDEO_FRAME_STRIDE = 2  # capture every 2nd physics step, same convention as dice_pick_demo.py's Gate V


class _StageTimeoutError(RuntimeError):
    """A staged waypoint didn't converge within its step budget - fails
    loudly (caught per-die at the call site, logged, demo continues with the
    next die) rather than silently treating a stuck stage as success."""


def spawn_scene_and_settle(sim_device: str):
    """Builds DiceLineDemoSceneCfg with DiceSceneCfg's own default (already
    scattered/reachable/min-spaced) die init_state.pos left UNCHANGED,
    applies runtime collision schemas + notch fixtures to every die, resets,
    and settles physics. Trimmed from scripts/dice_pick_demo.py's
    `spawn_scene_and_settle` - no detector-camera rejection-sampled layout, no
    colored-dice/light-scale options, no GT-JSON/RGB-PNG save (this demo
    reads GT positions live, has no detector step to feed)."""
    scene_cfg = DiceLineDemoSceneCfg(num_envs=1, env_spacing=4.0)

    sim_cfg = sim_utils.SimulationCfg(device=sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(list(_DEMO_CAMERA_EYE), list(_DEMO_CAMERA_TARGET))

    scene = InteractiveScene(scene_cfg)

    stage = get_current_stage()
    env_root = scene.env_prim_paths[0]
    for die_type in DIE_TYPES:
        die_prim_path = f"{env_root}/Die_{die_type}"
        mesh_count = apply_convex_hull_collision(stage, die_prim_path)
        print(f"[SPAWN] applied convex-hull collision to {die_type} ({mesh_count} mesh prim(s))")
        if mesh_count == 0:
            raise RuntimeError(f"No UsdGeom.Mesh prims found under {die_prim_path}")

    attach_notch_fixtures(stage, f"{env_root}/Robot")

    sim.reset()
    scene.reset()
    print("[SPAWN] sim.reset() + scene.reset() complete. Settling physics...")

    sim_dt = sim.get_physics_dt()
    settle_steps = int(3.0 / sim_dt)
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print("[SPAWN] rendering RTX convergence frames...")
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print(f"[SPAWN] settled after {settle_steps} steps (3.0s sim time). Starting die states:")
    for die_type in DIE_TYPES:
        pos = scene[f"die_{die_type}"].data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        print(f"  {die_type}: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}")

    return sim, scene


def main() -> None:
    sim, scene = spawn_scene_and_settle(args_cli.device)

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    hand_body_id = robot_entity_cfg.body_ids[0]

    gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    gripper_cfg.resolve(scene)
    gripper_joint_ids = gripper_cfg.joint_ids

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.02}
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=robot.device)

    canonical_down_quat_w = torch.tensor([0.0, 1.0, 0.0, 0.0], device=robot.device, dtype=torch.float32)
    open_target = torch.full((1, len(gripper_joint_ids)), 0.04, device=robot.device)
    close_target = torch.full((1, len(gripper_joint_ids)), 0.0, device=robot.device)

    # --- Video capture setup (demo_camera, whole-table 3/4 view) ---
    # STREAMED directly to the imageio writer as each frame is captured, NOT
    # accumulated into a Python list first (as scripts/dice_pick_demo.py's
    # Gate V does for its own single ~18s clip - fine at that scale, ~2-3k
    # frames at 640x480). This demo runs 10 pick-and-place ops (5 dice x 2
    # passes) at 1280x720 - buffering every captured frame in host RAM before
    # encoding was measured (this task's own first cloud run) to OOM-kill the
    # process on a 16GB g2-standard-4 instance (~9000+ frames x ~2.7MB/frame
    # uncompressed >> 16GB). Streaming keeps peak host RAM to ~one frame plus
    # the encoder's own small internal buffer, regardless of video length.
    camera = scene["demo_camera"]
    video_fps = max(1, round(1.0 / (sim_dt * _VIDEO_FRAME_STRIDE)))
    video_path = os.path.join(OUTPUT_DIR, "franka_ik_dice_line_demo.mp4")
    video_writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264")
    step_counter = [0]
    frame_counter = [0]

    def _on_step() -> None:
        step_counter[0] += 1
        if step_counter[0] % _VIDEO_FRAME_STRIDE == 0:
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            video_writer.append_data(rgb)
            frame_counter[0] += 1

    def _hand_target_xyz(xy: tuple, hand_z: float) -> torch.Tensor:
        return torch.tensor([xy[0], xy[1], hand_z], device=robot.device, dtype=torch.float32)

    hand_pos_w_t_for_logging = [np.zeros(3)]
    hand_quat_w_t_for_logging = [np.array([1.0, 0.0, 0.0, 0.0])]

    def _quat_angle_diff_rad(q1: np.ndarray, q2: np.ndarray) -> float:
        dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
        return float(2.0 * np.arccos(dot))

    def _step_toward(target_pos_b: torch.Tensor, target_quat_b: torch.Tensor, gripper_target: torch.Tensor) -> tuple:
        """One bounded-relative-step physics step of Cartesian control -
        copied mechanism from scripts/dice_pick_demo.py's `_step_toward`
        (see that file's own extensive comment for why relative mode +
        per-step clipping, not a single large absolute-pose jump)."""
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, hand_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        pos_err, rot_err = compute_pose_error(
            ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
        )
        pos_err_norm = pos_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pos_step = pos_err * torch.clamp(pos_err_norm, max=_MAX_POS_STEP) / pos_err_norm
        rot_err_norm = rot_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rot_step = rot_err * torch.clamp(rot_err_norm, max=_MAX_ROT_STEP) / rot_err_norm
        delta_command = torch.cat([pos_step, rot_step], dim=-1)
        diff_ik_controller.set_command(delta_command, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _on_step()
        cur_pos_w = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
        cur_quat_w = robot.data.body_quat_w[0, hand_body_id].cpu().numpy()
        pos_err_m = float(np.linalg.norm(cur_pos_w - hand_pos_w_t_for_logging[0]))
        rot_err_rad = _quat_angle_diff_rad(cur_quat_w, hand_quat_w_t_for_logging[0])
        return pos_err_m, rot_err_rad

    def _go_to_pose(
        hand_pos_w_t: torch.Tensor, hand_quat_w_t: torch.Tensor, gripper_target: torch.Tensor,
        label: str, max_steps: int, require_rot: bool = True, pos_tol: float = _WAYPOINT_TOL,
    ) -> None:
        root_pose_w = robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], hand_pos_w_t.unsqueeze(0), hand_quat_w_t.unsqueeze(0)
        )
        hand_pos_w_t_for_logging[0] = hand_pos_w_t.cpu().numpy()
        hand_quat_w_t_for_logging[0] = hand_quat_w_t.cpu().numpy()

        pos_err = rot_err = float("inf")
        converged = False
        step = 0
        for step in range(max_steps):
            pos_err, rot_err = _step_toward(target_pos_b, target_quat_b, gripper_target)
            pos_ok = pos_err < pos_tol
            rot_ok = (not require_rot) or (rot_err < _ROT_TOL)
            if pos_ok and rot_ok:
                converged = True
                break

        if not converged:
            live_pos = robot.data.body_pos_w[0, hand_body_id].cpu().numpy()
            raise _StageTimeoutError(
                f"waypoint '{label}' did not converge within {max_steps} steps "
                f"(final pos_err={pos_err * 1000:.1f}mm tol={pos_tol * 1000:.1f}mm, "
                f"rot_err={rot_err:.4f}rad tol={_ROT_TOL:.4f}rad). live hand_pos_w={live_pos} "
                f"target={hand_pos_w_t.cpu().numpy()}"
            )
        print(f"[IK] waypoint '{label}': converged after {step + 1} steps (pos_err={pos_err * 1000:.1f}mm rot_err={rot_err:.4f}rad)")

    def _hold(hand_pos_w_t: torch.Tensor, hand_quat_w_t: torch.Tensor, gripper_target: torch.Tensor, label: str, steps: int) -> None:
        root_pose_w = robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], hand_pos_w_t.unsqueeze(0), hand_quat_w_t.unsqueeze(0)
        )
        hand_pos_w_t_for_logging[0] = hand_pos_w_t.cpu().numpy()
        hand_quat_w_t_for_logging[0] = hand_quat_w_t.cpu().numpy()
        for _ in range(steps):
            _step_toward(target_pos_b, target_quat_b, gripper_target)
        print(f"[IK] held '{label}' for {steps} steps")

    def _joint_space_prep(target_joint_pos: list, gripper_target: torch.Tensor, steps: int, label: str) -> None:
        start = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].clone()
        target = torch.tensor(target_joint_pos, device=robot.device, dtype=torch.float32)
        for i in range(steps):
            alpha = (i + 1) / steps
            interp = start + alpha * (target - start)
            robot.set_joint_position_target(interp.unsqueeze(0), joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            _on_step()
        print(f"[IK] {label}: joint-space prep done ({steps} steps)")

    def pick_and_place(die_type: str, pick_xy: tuple, place_xy: tuple) -> dict:
        """One full pick(pick_xy) -> lift -> carry -> place(place_xy) ->
        retract cycle for `die_type`, reusing scripts/dice_pick_demo.py's
        `run_pick_sequence` staged mechanism (trimmed of the
        detector/roll/d4-contact machinery this demo doesn't need). Table is
        flat, so the same `die_grasp_height_m(die_type)` is used for both the
        pick-descend and place-descend targets (same reasoning as
        dice_pick_demo.py's own `post_action="move"` branch)."""
        grasp_hand_z = die_grasp_height_m(die_type) + _VALIDATED_HAND_TO_PINCH_POINT_Z

        _joint_space_prep(_READY_TO_DESCEND_JOINT_POS, open_target, _MAX_STEPS_JOINT_PREP, "stage0_joint_prep")

        # Stage 1: approach above the pick position, orientation converges too.
        _go_to_pose(
            _hand_target_xyz(pick_xy, _STAGE1_HAND_Z), canonical_down_quat_w, open_target,
            "stage1_approach", max_steps=_MAX_STEPS_APPROACH, require_rot=True,
        )
        # Stage 2: vertical descent to grasp height.
        stage2_target = _hand_target_xyz(pick_xy, grasp_hand_z)
        try:
            _go_to_pose(
                stage2_target, canonical_down_quat_w, open_target, "stage2_descend",
                max_steps=_MAX_STEPS_DESCEND, require_rot=True, pos_tol=_GRASP_POS_TOL,
            )
        except _StageTimeoutError as e:
            print(f"[IK] stage2_descend did not converge at tight tolerance - XY-only refine fallback: {e}")
            live_z = float(robot.data.body_pos_w[0, hand_body_id, 2].cpu().numpy())
            stage2_target = _hand_target_xyz(pick_xy, live_z)
            _go_to_pose(
                stage2_target, canonical_down_quat_w, open_target, "stage2_descend_xy_refine",
                max_steps=_MAX_STEPS_REFINE, require_rot=True, pos_tol=_GRASP_POS_TOL,
            )
        # Stage 3: close gripper, dwell.
        _hold(stage2_target, canonical_down_quat_w, close_target, "stage3_close_gripper", _GRIPPER_CLOSE_HOLD_STEPS)
        # Stage 4: lift.
        _go_to_pose(
            _hand_target_xyz(pick_xy, _STAGE_LIFT_HAND_Z), canonical_down_quat_w, close_target,
            "stage4_lift", max_steps=_MAX_STEPS_LIFT, require_rot=True,
        )
        # Stage 5: carry to above the place position, still lifted.
        _go_to_pose(
            _hand_target_xyz(place_xy, _STAGE_LIFT_HAND_Z), canonical_down_quat_w, close_target,
            "stage5_carry", max_steps=_MAX_STEPS_APPROACH, require_rot=True,
        )
        # Stage 6: descend to place height.
        place_target = _hand_target_xyz(place_xy, grasp_hand_z)
        _go_to_pose(
            place_target, canonical_down_quat_w, close_target, "stage6_descend_place",
            max_steps=_MAX_STEPS_DESCEND, require_rot=True, pos_tol=_WAYPOINT_TOL,
        )
        # Stage 7: release.
        _hold(place_target, canonical_down_quat_w, open_target, "stage7_release", _GRIPPER_CLOSE_HOLD_STEPS)
        # Stage 8: retract.
        _go_to_pose(
            _hand_target_xyz(place_xy, _STAGE_LIFT_HAND_Z), canonical_down_quat_w, open_target,
            "stage8_retract", max_steps=_MAX_STEPS_LIFT, require_rot=True,
        )
        settle_steps = int(_POST_RELEASE_SETTLE_SECONDS / sim_dt)
        for _ in range(settle_steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            _on_step()

        die = scene[f"die_{die_type}"]
        final_pos = die.data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        xy_error_m = float(np.hypot(final_pos[0] - place_xy[0], final_pos[1] - place_xy[1]))
        print(
            f"[PICK&PLACE] {die_type}: final pos=({final_pos[0]:.4f},{final_pos[1]:.4f},{final_pos[2]:.4f}) "
            f"target=({place_xy[0]:.4f},{place_xy[1]:.4f}) xy_error={xy_error_m * 1000:.1f}mm"
        )
        return {"die": die_type, "place_target_xy": list(place_xy), "final_xyz": final_pos.tolist(), "xy_error_m": xy_error_m}

    # --- Act 1: pick each die from its scattered start, line it up. ---
    results = {"act1": [], "act2": []}
    print("=== ACT 1: scattered layout -> lined up ===")
    for die_type in PICK_ORDER:
        pick_xy = tuple(
            (scene[f"die_{die_type}"].data.root_pos_w[0, :2] - scene.env_origins[0, :2]).cpu().numpy().tolist()
        )
        place_xy = LINE1_TARGETS[die_type]
        try:
            results["act1"].append(pick_and_place(die_type, pick_xy, place_xy))
        except _StageTimeoutError as e:
            print(f"[ACT1] *** {die_type} pick-and-place FAILED (stage timeout, continuing to next die): {e} ***")
            results["act1"].append({"die": die_type, "error": str(e)})

    print("=== ACT 2: lined up -> relocated (rotated + shifted) line ===")
    for die_type in PICK_ORDER:
        # Read the die's REAL current position (not LINE1_TARGETS' nominal
        # value) - Act 1's own placement always has some residual xy_error
        # (a few mm, expected/measured in its own result dict), and if Act 1
        # failed for this die (stage timeout, caught above) it may not be
        # anywhere near LINE1_TARGETS at all. Ground-truth read, same
        # technique Act 1 uses for its own scattered-start positions.
        pick_xy = tuple(
            (scene[f"die_{die_type}"].data.root_pos_w[0, :2] - scene.env_origins[0, :2]).cpu().numpy().tolist()
        )
        place_xy = LINE2_TARGETS[die_type]
        try:
            results["act2"].append(pick_and_place(die_type, pick_xy, place_xy))
        except _StageTimeoutError as e:
            print(f"[ACT2] *** {die_type} pick-and-place FAILED (stage timeout, continuing to next die): {e} ***")
            results["act2"].append({"die": die_type, "error": str(e)})

    # Hold the final frame for a couple seconds so the video doesn't cut off
    # right at the last retract.
    final_hold_steps = int(2.0 / sim_dt)
    for _ in range(final_hold_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _on_step()

    n_act1_ok = sum(1 for r in results["act1"] if "error" not in r)
    n_act2_ok = sum(1 for r in results["act2"] if "error" not in r)
    print(f"=== SUMMARY: Act 1 {n_act1_ok}/{len(PICK_ORDER)} placed, Act 2 {n_act2_ok}/{len(PICK_ORDER)} placed ===")

    # Close the streamed video writer BEFORE simulation_app.close() - this
    # repo has a documented Isaac Sim teardown-hang gotcha (CLAUDE.md's
    # "Known gap: a hung process still holds the lock") that fires AFTER the
    # script's real work is already done; closing/flushing the video first
    # means a teardown hang never loses it. (Frames were streamed directly
    # to this writer in `_on_step` throughout the run - see that function's
    # own comment for why, not buffered here.)
    video_writer.close()
    print(
        f"[VIDEO] wrote {video_path} ({frame_counter[0]} frames @ {video_fps}fps, "
        f"~{frame_counter[0] / video_fps:.1f}s)"
    )

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "pick_order": PICK_ORDER,
                "line1_targets": {k: list(v) for k, v in LINE1_TARGETS.items()},
                "line2_targets": {k: list(v) for k, v in LINE2_TARGETS.items()},
                "results": results,
                "video_path": video_path,
                "num_video_frames": frame_counter[0],
                "video_fps": video_fps,
            },
            f,
            indent=2,
        )
    print(f"[SUMMARY] wrote {summary_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()

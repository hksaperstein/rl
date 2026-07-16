# tasks/franka/dice_scene_cfg.py
"""Scene definition for the dice-pick convergence demo (see CLAUDE.md's active
top-priority goal): Franka Panda + table + five dice (d4, d8, d10, d12, d20,
one each, from vision/data/raw/dice_sets_v1/set_00000_<type>.usd - the one set
whose manifests show empty mesh_quality_warnings for every type checked) +
an angled RGB-D perception camera.

Reuses the table/robot/plane/light layout from tasks/franka/lift_env_cfg.py's
FrankaLiftSceneCfg verbatim (not imported, to keep this module independent
of the RL lift task - this is a plain-InteractiveScene scripted demo, not a
ManagerBasedRLEnv) - same reasoning as lift_env_cfg.py's own docstring: no
task-specific retuning of the known-good Franka+table recipe.

Dice USDs are visual-only exports (no physics schemas). RigidObjectCfg's
rigid_props, mass_props, and collision_props all silently no-op: they only
*modify* existing schemas and return False if none exist (confirmed by reading
isaaclab/sim/schemas/schemas.py). With schema-less USDs, the configured
_DICE_RIGID_PROPS, _DICE_MASS, _DICE_COLLISION_PROPS are never applied.

The demo script's `apply_convex_hull_collision` helper (scripts/dice_pick_demo.py)
fixes this by: (1) applying bare schemas via pxr (UsdPhysics.RigidBodyAPI,
UsdPhysics.MassAPI, UsdPhysics.CollisionAPI + MeshCollisionAPI on each mesh
prim), then (2) re-applying the tuned properties via isaaclab's
modify_rigid_body_properties/modify_mass_properties/modify_collision_properties
helpers, which now work because the schemas exist. This pattern matches
scripts/build_asset.py's wedge-asset technique, just applied at runtime
instead of offline (these dice USDs are shared with vision/'s rendering
pipeline, so baking physics into the source files is deliberately avoided).

Lighting: DomeLightCfg alone doesn't adequately illuminate camera sensor
renders; DistantLightCfg was added to the scene to ensure the perception
camera captures a properly lit frame for detector inference.

d4 rung-1 V-notch fingertip fixture (2026-07-15, see
docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md): two new
`notch_fixture_left`/`notch_fixture_right` scene prims, authored offline by
scripts/build_notch_fixture_asset.py from tasks/franka/notch_fixture.py's
pure geometry, attached to the Franka's two fingertips via a fixed joint
created at runtime by scripts/dice_pick_demo.py's `attach_notch_fixtures` -
UNCONDITIONAL (both fingertips, every die type, not a d4-only branch) per
the spec's own North Star call. The stock Franka fingertip mesh/collision
geometry itself is never touched (only new sibling prims + a joint are
added) - see that spec's "byte-identity" regression-guard requirement.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# d4 edge-grasp rung-0 Task 2 contact instrumentation (2026-07-13, see
# docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md's Desk-check
# corrections and .superpowers/sdd/task-d4-rung0-tasks01-report.md's "What
# Task 2 needs going in" - a real contact-force/point reading is needed to
# confirm which phi-regime (near-0 vs the discontinuous 54.7deg case) the
# real grasp lands in, not just the lateral-ejection proxy metric). PhysX
# activates PhysxContactReportAPI per-body at USD-spawn time for the WHOLE
# robot, not selectively per-body - copy()-then-mutate is the SAME idiom
# isaaclab_assets' own franka.py uses to derive FRANKA_PANDA_HIGH_PD_CFG from
# FRANKA_PANDA_CFG (e.g. its own `.spawn.rigid_props.disable_gravity = True`
# line), not a new pattern introduced here.
_FRANKA_ROBOT_CFG_WITH_CONTACT = FRANKA_PANDA_HIGH_PD_CFG.copy()
_FRANKA_ROBOT_CFG_WITH_CONTACT.spawn.activate_contact_sensors = True

# The five die types the commanded-pick demo distinguishes between (d100/d10_pct
# is an alias of d10, not a separate physical die in this scene - see Gate G's
# --choice alias handling in scripts/dice_pick_demo.py).
DIE_TYPES = ["d4", "d8", "d10", "d12", "d20"]

_DICE_SET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vision",
    "data",
    "raw",
    "dice_sets_v1",
)
_DICE_SET_NAME = "set_00000"  # every DIE_TYPES manifest for this set has empty mesh_quality_warnings

# d4 rung-1 V-notch fingertip fixture (2026-07-15, see
# docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md and
# .superpowers/sdd/task-1-brief.md) - authored offline by
# scripts/build_notch_fixture_asset.py (plain pxr, no sim launch/GPU) via
# tasks/franka/notch_fixture.py's pure geometry. UNCONDITIONAL per the
# spec's own "North Star call": attached to BOTH fingertips for every die
# type, not gated on `choice == "d4"` - see `notch_fixture_left`/
# `notch_fixture_right` below and scripts/dice_pick_demo.py's
# `attach_notch_fixtures`.
_NOTCH_FIXTURE_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "shapes", "notch_fixture.usd",
)

# Small-object rigid props: light solver iteration bump (small/light bodies
# are prone to jitter/tunneling at default iteration counts - same rationale
# as lift_env_cfg.py's DexCube rigid_props), disable_gravity=False so they
# actually fall/settle onto the table under gravity.
_DICE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
# ~5-15g real-dice-scale mass (a plastic d20 this size is roughly this order
# of magnitude) - not measured/verified against a real dice mass reference,
# a reasonable placeholder for grasp dynamics, not a precision-critical value.
_DICE_MASS = MassPropertiesCfg(mass=0.01)
_DICE_COLLISION_PROPS = CollisionPropertiesCfg(collision_enabled=True)


def _die_usd_path(die_type: str) -> str:
    path = os.path.join(_DICE_SET_DIR, f"{_DICE_SET_NAME}_{die_type}.usd")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Die USD not found: {path}")
    return path


def _notch_fixture_usd_path() -> str:
    if not os.path.isfile(_NOTCH_FIXTURE_USD_PATH):
        raise FileNotFoundError(
            f"Notch fixture USD not found: {_NOTCH_FIXTURE_USD_PATH} - run "
            "scripts/build_notch_fixture_asset.py once per machine/checkout first "
            "(plain pxr, no sim launch/GPU - see that script's own docstring for "
            "the exact invocation)."
        )
    return _NOTCH_FIXTURE_USD_PATH


def _notch_fixture_cfg(prim_name: str, init_pos: tuple[float, float, float]) -> AssetBaseCfg:
    """One notch-fixture instance (left or right) - a plain AssetBaseCfg
    (matching table/plane/light's own "no data buffers needed" pattern,
    not RigidObjectCfg - nothing in this demo ever reads
    `scene["notch_fixture_*"].data`). `init_pos` only needs to be roughly
    near the default gripper pose for numerical stability at the FIRST
    physics step - scripts/dice_pick_demo.py's `attach_notch_fixtures`
    overwrites this with an exactly-measured world position (from the live
    finger prim's own transform) before `sim.reset()` ever runs, and the
    fixed joint it creates there is what actually determines the fixture's
    held pose from then on, not this placeholder."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=init_pos),
        spawn=UsdFileCfg(usd_path=_notch_fixture_usd_path(), activate_contact_sensors=True),
    )


def _die_cfg(die_type: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    # The dice USDs (vision/data/raw/dice_sets_v1/set_00000_<type>.usd) are
    # authored in millimeters-as-meters (mm geometry, metersPerUnit=1.0), not
    # standard meters. vision/scripts/render_detection_dataset.py:125 applies
    # MM_TO_M = 0.001 uniform scaling when importing these same dice for the
    # detector training renders (confirmed match: the manifest .json records
    # each die's real size_mm; our measured 17.32mm for d4 matches the ~17.3
    # extent in stage units). Use that same uniform 0.001 scale factor here,
    # preserving the relative per-die size distribution the detector was
    # trained on (the per-die size variation is intentional datagen
    # randomization, not to be compensated per-die).
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Die_" + die_type,
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=_die_usd_path(die_type),
            scale=(0.001, 0.001, 0.001),
            rigid_props=_DICE_RIGID_PROPS,
            mass_props=_DICE_MASS,
            collision_props=_DICE_COLLISION_PROPS,
        ),
    )


# Camera placed to look down at the dice-spread region of the table at an
# elevation angle inside the training-data distribution used by the
# vision/ detector (vision/scripts/render_detection_dataset.py samples
# elevation Uniform(15deg, 65deg), distance 0.15-0.35m per die in its
# non-closeup mode) - chosen close enough to keep the dice at a
# recognizable apparent size for the detector, while still fitting the
# whole ~0.3m dice-spread region in frame (a compromise against the
# training distribution's much closer single/few-die framing - flagged as
# a documented domain-gap risk for Gate P to measure empirically, not
# assumed safe). Offset along -Y (in front of the table, not directly above
# the arm's own reach path) at 50deg elevation, 0.55m from the dice-region
# center (0.5, 0.0, 0.03) - position/quat solved numerically (look-at,
# CameraCfg "world" convention: local +X forward, +Z up) and verified by
# rotating the local forward axis by the quat and confirming it matches the
# analytic camera-to-target direction (see plan notes); not hand-derived
# on paper only, given this repo's own prior camera-orientation bugs.
DICE_CAMERA_POS = (0.5, -0.35353319, 0.45132444)
DICE_CAMERA_QUAT_WORLD = (0.64085638, -0.29883624, 0.29883624, 0.64085638)

# Second, separate camera (2026-07-11 colored-dice repeat task's Franka
# material check) dedicated to a whole-arm view - DiceCamera's own framing
# is pinned to the detector's training distribution (tight on the
# dice-spread region) and must NOT be reframed for this purpose (existing
# gates depend on it staying exactly as calibrated). Position/quat solved
# and verified the same way as DICE_CAMERA_POS/QUAT above (rotate local +X
# by the quat, confirm it matches the analytic camera->target unit vector;
# also checked local +Z has a strongly positive world-z component, i.e. not
# upside down) - not hand-derived on paper only, per this repo's own history
# of camera-convention bugs.
#
# RE-AIMED 2026-07-15 (Gate V video-camera task): the original pose sat at
# roughly equal -X/-Y offset from the robot, i.e. its camera->target view
# direction had nearly equal X and Y world components. That mattered because
# the Franka's gripper fingers (panda_finger_joint1/2) are prismatic joints
# with LOCAL axis (0,1,0) on panda_hand (see the URDF,
# franka_description/robots/panda_arm_hand.urdf); this demo's canonical
# straight-down grasp quat (0,1,0,0 wxyz, used for every die type - see
# scripts/dice_pick_demo.py's canonical_down_quat_w) rotates panda_hand 180
# degrees about world X, which maps local Y to world -Y unchanged in axis -
# so the fingers ALWAYS open/close along the world Y axis, for every grasp
# in this demo regardless of target xy. A camera whose view direction has a
# large Y component is therefore looking nearly down the pinch axis itself:
# the two fingers (and the die between them) project onto nearly the same
# image pixels and occlude each other - exactly the "side profile" video
# complaint this re-aim fixes. The fix is to make the view direction
# X-dominant / Y-minimal instead: with Y close to constant across the shot,
# the fingers separate along the image's horizontal axis and their closing
# motion (and the die between them) reads clearly. New pose sits behind the
# robot base along -X (opposite side from the dice spread, out of the arm's
# own reach path) with only a small Y offset, elevated and aimed down at a
# point centered over the dice-spread/grasp region (x=0.5, y=0.0, z=0.15 -
# roughly waist height, between the table surface where the pinch happens
# and the arm's own elevated resting height) so the full kinematic chain
# (base to gripper) and the table are both still in frame alongside the
# clearer pinch angle. Verified analytically (rotate local +X by the quat,
# confirm it matches the numeric camera->target unit vector computed at
# pose-selection time; confirmed local +Z has a strongly positive world-z
# component) AND empirically (rendered a still frame from this exact pose
# via `--gate a`, inspected the PNG directly - see
# outputs/dice_demo/gate_a/arm_camera_rgb.png from that verification run).
#
# TIGHTENED 2026-07-15 (same-day follow-up): the first re-aim above fixed
# the occlusion problem (pinch axis no longer points at the camera) but was
# too far/high (1.8m out, 0.75m up, aimed at z=0.15) - correct in direction
# but the grasp region ended up a small corner of frame, dominated by the
# arm's own tall vertical reach (confirmed by inspecting the actual
# rendered Gate V video frame-by-frame, not just the earlier still: at
# closure the die+fingers occupied only a few percent of frame width).
# Moved much closer (~0.7m, was ~1.8m) and lower (0.30m, was 0.75m) with
# the aim point dropped to table/grasp height (z=0.08, was 0.15) so the
# grip fills a legible fraction of the frame - this trades away some of
# the full kinematic chain's upper links for what the video is actually
# for (seeing the grip occur), per direct user instruction. Same
# analytical derivation method as above (local +X -> normalized
# camera->target vector, local +Z stays strongly positive/not upside
# down, local +Y - the image "right" axis - checked to point predominantly
# along world +Y so the fingers' Y-axis closing motion reads as
# horizontal, not toward/away from camera).
ARM_CAMERA_POS = (-0.2, -0.15, 0.30)
ARM_CAMERA_QUAT_WORLD = (0.98617192, -0.007231, 0.15940998, 0.04473376)


@configclass
class DiceSceneCfg(InteractiveSceneCfg):
    """Table + Franka Panda (high-PD) + five dice (d4/d8/d10/d12/d20) + an
    angled RGB-D perception camera. Dice `init_state.pos` values here are
    placeholders overwritten by the demo script's own randomized,
    minimum-spacing table layout before `sim.reset()`."""

    robot = _FRANKA_ROBOT_CFG_WITH_CONTACT.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Isaac Sim's ACTUAL default-stage light rig (2026-07-13, direct user
    # directive "use default light rig for all work"), values transcribed
    # verbatim from the installed omni.kit.stage_templates default_stage.py
    # template: an HDR "Sky" dome (CarLight HDR, intensity 1 x exposure 9,
    # 6250K) + a soft "sun" DistantLight (angle 2.5, intensity 1 x
    # exposure 10, 7250K). The HDR is vendored into tasks/franka/assets/
    # (the extscache path is Isaac-version-fragile). Replaces the earlier
    # flat grey DomeLight(3000) which rendered dark, and the removed
    # DistantLight(3000) stage light which blew the scene out.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1.0,
            exposure=10.0,  # +1 stop over the Kit template (its rig targets the default light-grey ground; our table albedo is dark - measured 63 vs target ~130 mean)
            enable_color_temperature=True,
            color_temperature=6250.0,
            texture_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "CarLight_512x256.hdr"),
        ),
    )

    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(
            angle=2.5,
            intensity=1.0,
            exposure=11.0,  # +1 stop over template, same rationale as the dome
            enable_color_temperature=True,
            color_temperature=7250.0,
        ),
    )

    die_d4: RigidObjectCfg = _die_cfg("d4", (0.35, -0.20, 0.10))
    die_d8: RigidObjectCfg = _die_cfg("d8", (0.42, -0.10, 0.10))
    die_d10: RigidObjectCfg = _die_cfg("d10", (0.50, 0.0, 0.10))
    die_d12: RigidObjectCfg = _die_cfg("d12", (0.58, 0.10, 0.10))
    die_d20: RigidObjectCfg = _die_cfg("d20", (0.65, 0.20, 0.10))

    # d4 rung-1 V-notch fingertip fixture (2026-07-15) - UNCONDITIONAL, both
    # fingertips, every die type (spec's North Star call; see
    # `_NOTCH_FIXTURE_USD_PATH`'s comment above). Declared as plain top-level
    # scene prims (siblings of Die_*/Robot/Table, NOT nested under `Robot`'s
    # own prim tree) specifically so attaching them never requires authoring
    # anything on the Franka articulation's own (partly instanceable) prim
    # hierarchy - Task 0's own finding was that a new child prim under
    # `panda_leftfinger`/`panda_rightfinger` is not straightforwardly
    # authorable (instanceable prims), which is why the design is "new
    # sibling rigid body + fixed joint", not "new child mesh under the
    # existing finger". Declared BEFORE the ContactSensorCfg fields below
    # purely for readability (grouping "what exists" ahead of "what reads
    # it") - NOT a correctness requirement: `ContactSensor.__init__` never
    # touches the stage, and its actual prim/PhysxContactReportAPI lookup
    # happens lazily in `_initialize_impl`, deferred to a play-event
    # callback that runs after every scene field (regardless of declaration
    # order) has already been spawned. See instead the module-level
    # `activate_contact_sensors=True` fix on `_notch_fixture_cfg`'s
    # `UsdFileCfg` (2026-07-15 review finding) for what actually determines
    # whether these contact sensors find a body to attach to.
    notch_fixture_left: AssetBaseCfg = _notch_fixture_cfg("NotchFixtureLeft", (0.13, -0.03, 0.85))
    notch_fixture_right: AssetBaseCfg = _notch_fixture_cfg("NotchFixtureRight", (0.13, 0.03, 0.85))

    # d4-only contact instrumentation (2026-07-13, Task 2; RETARGETED
    # 2026-07-15 rung-1 Task 1 - see module-level comment above
    # `_FRANKA_ROBOT_CFG_WITH_CONTACT` for the original rationale).
    # prim_path now points at the notch fixture prims, NOT the bare
    # panda_leftfinger/panda_rightfinger body prims: with the fixture
    # rigidly fixed-jointed on as a SEPARATE PhysX rigid body (Task 0's
    # finding - a fixed joint constrains two distinct bodies, it does not
    # merge their collision geometry into one), the die's actual contact
    # now registers against the FIXTURE's own collision shapes, not the
    # finger body's - a ContactSensorCfg still targeting
    # `.../Robot/panda_leftfinger` would see zero force even during a
    # genuine notch-facet grip. filter_prim_paths_expr targets ONLY Die_d4,
    # not a wildcard over all 5 dice - these sensors report zero/empty data
    # for the whole scene unless something specifically touches Die_d4, and
    # scripts/dice_pick_demo.py's non-d4 code path never reads
    # `scene["d4_*_contact"]` at all, so this is additive/inert for the
    # d8/d10/d12/d20 runs (same "spawn universally, read only on the
    # relevant branch" pattern the die_d4 RigidObjectCfg itself already
    # uses). Two separate single-body sensors (not one two-body sensor)
    # because PhysX requires the filter match count to equal the sensor
    # body count - see tasks/ar4/pickplace_env_cfg.py's
    # gripper_jaw1_contact/gripper_jaw2_contact, the same pattern already
    # validated in this repo.
    d4_leftfinger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/NotchFixtureLeft",
        update_period=0.0,
        history_length=0,
        track_contact_points=True,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Die_d4"],
    )
    d4_rightfinger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/NotchFixtureRight",
        update_period=0.0,
        history_length=0,
        track_contact_points=True,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Die_d4"],
    )

    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DiceCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=DICE_CAMERA_POS, rot=DICE_CAMERA_QUAT_WORLD, convention="world"),
    )

    # Diagnostic-only whole-arm 3/4 view (2026-07-11 colored-dice repeat
    # task's Franka material check) - see ARM_CAMERA_POS/QUAT_WORLD's comment
    # above for why this is a SEPARATE camera from DiceCamera. Not used by
    # any gate's pass/fail logic; captured purely for visual inspection of
    # the robot's material appearance.
    arm_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ArmCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 3.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=ARM_CAMERA_POS, rot=ARM_CAMERA_QUAT_WORLD, convention="world"),
    )

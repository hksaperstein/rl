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
collision_props only applies UsdPhysics.CollisionAPI to the prim path given -
it does NOT set UsdPhysics.MeshCollisionAPI's approximation, and USD's own
default for an un-set MeshCollisionAPI is "none" (exact triangle mesh), which
PhysX rejects for *dynamic* (non-kinematic) rigid bodies (only static/
kinematic colliders may use a triangle-mesh approximation). Left at that
default, the dice would either fail to cook a collider at all (falling
through the table) or cook an unpredictable fallback - so
`apply_convex_hull_collision` (called by the demo script, after scene
construction, before `sim.reset()`) walks each die's prim subtree, finds
every UsdGeom.Mesh prim, and explicitly applies UsdPhysics.CollisionAPI +
UsdPhysics.MeshCollisionAPI(approximation="convexHull") directly - the same
technique scripts/build_asset.py used for the AR4-era wedge asset, just
applied at runtime instead of as an offline asset-authoring step (these dice
USDs are shared with vision/'s rendering pipeline, so baking physics into the
source files themselves is deliberately avoided).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

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


@configclass
class DiceSceneCfg(InteractiveSceneCfg):
    """Table + Franka Panda (high-PD) + five dice (d4/d8/d10/d12/d20) + an
    angled RGB-D perception camera. Dice `init_state.pos` values here are
    placeholders overwritten by the demo script's own randomized,
    minimum-spacing table layout before `sim.reset()`."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    die_d4: RigidObjectCfg = _die_cfg("d4", (0.35, -0.20, 0.10))
    die_d8: RigidObjectCfg = _die_cfg("d8", (0.42, -0.10, 0.10))
    die_d10: RigidObjectCfg = _die_cfg("d10", (0.50, 0.0, 0.10))
    die_d12: RigidObjectCfg = _die_cfg("d12", (0.58, 0.10, 0.10))
    die_d20: RigidObjectCfg = _die_cfg("d20", (0.65, 0.20, 0.10))

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

"""RigidObjectCfg definitions for the small graspable objects in the AR4 scene.

Four objects sized to fit the gripper's ~28mm max aperture, laid out two per
side of the arm base. Fixed initial poses only - no reset/randomization logic
(that belongs to the follow-on task/RL sub-project).

Grip friction is set once, scene-wide, via Ar4EnvCfg's sim.physics_material
(UsdFileCfg has no per-spawn physics_material field like the procedural
shape/mesh spawners do, so a shared scene default is used instead).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WEDGE_USD_PATH = os.path.join(_RL_ROOT, "assets", "shapes", "wedge.usd")

_MASS = sim_utils.MassPropertiesCfg(mass=0.01)
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(collision_enabled=True)


def _check_wedge_usd_exists() -> str:
    if not os.path.isfile(_WEDGE_USD_PATH):
        sys.exit(
            f"Wedge USD asset not found (missing {_WEDGE_USD_PATH}).\n"
            "Run rl/scripts/build_asset.py first."
        )
    return _WEDGE_USD_PATH


CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.009)),
    spawn=sim_utils.CuboidCfg(
        size=(0.018, 0.018, 0.018),
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS,
        collision_props=_COLLISION_PROPS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
)

RECT_PRISM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/RectPrism",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.34, 0.015)),
    spawn=sim_utils.CuboidCfg(
        size=(0.016, 0.016, 0.030),
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS,
        collision_props=_COLLISION_PROPS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
    ),
)

SPHERE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Sphere",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.20, 0.28, 0.009)),
    spawn=sim_utils.SphereCfg(
        radius=0.009,
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS,
        collision_props=_COLLISION_PROPS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
    ),
)

WEDGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Wedge",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.20, 0.34, 0.009)),
    spawn=sim_utils.UsdFileCfg(
        usd_path=_check_wedge_usd_exists(),
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS,
        collision_props=_COLLISION_PROPS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.1)),
    ),
)

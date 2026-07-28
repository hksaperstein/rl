"""RigidObjectCfg definitions for the small graspable objects in the AR4 scene.

Four objects sized to fit the gripper's ~28mm max aperture, laid out two per
side of the arm base. Fixed initial poses only - no reset/randomization logic
(that belongs to the follow-on task/RL sub-project).

Grip friction, for objects that don't set their own `physics_material`
(RECT_PRISM_CFG/SPHERE_CFG/WEDGE_CFG below), is inherited from Ar4EnvCfg's
scene-wide `sim.physics_material` default (`static_friction=1.0,
dynamic_friction=1.0` in every `tasks/ar4/*_env_cfg.py`'s own
`__post_init__` - confirmed via Isaac Lab's own
`SimulationCfg.physics_material` docstring: "the physics engine defaults to
this material for all rigid body prims that do not have any physics
material specified on them"; `UsdFileCfg` - used by WEDGE_CFG - has no
per-spawn `physics_material` field, so a shared scene default was the only
option for it).

CUBE_CFG (2026-07-28, ar4-grasp-trivial-friction-check task, direct user
decision) now sets its OWN explicit `physics_material` instead of silently
inheriting the scene default - not because the scene default was found to
be wrong (it was independently confirmed correct: `mu=1.0` scene-wide,
applying automatically to both this cube and the gripper's own jaw links
since neither had a per-body override, `friction_combine_mode="average"`
by Isaac Lab's own `RigidBodyMaterialCfg` default, so no `min`-combine
capping risk either - and 1.0 was already HIGHER than Franka's own
dice-grasp setup, which sets no override at all and runs on Isaac Lab's
library default of 0.5/0.5 and grips successfully), but per direct user
instruction to make a real graspable-material friction value an explicit,
visible property of the object itself rather than an implicit scene-wide
side effect. Value chosen: static=0.8/dynamic=0.7, the user's own specified
"wood/plastic/resin" realistic range (0.7-0.9 static / 0.6-0.8 dynamic) -
deliberately LOWER than the pre-existing 1.0/1.0 scene default despite that
default already being sufficient, precisely so this cube's own friction is
governed by its own explicit, physically-labeled material rather than by
happening to match whatever the scene-wide default is. Combined against
the gripper's own unchanged 1.0/1.0 scene-default material (average
combine mode: effective ~0.9 static / 0.85 dynamic) this remains solidly
in the graspable range.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WEDGE_USD_PATH = os.path.join(_RL_ROOT, "assets", "shapes", "wedge.usd")

_MASS = sim_utils.MassPropertiesCfg(mass=0.01)
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(collision_enabled=True)

# Explicit, realistic wood/plastic/resin-range graspable-object friction
# (2026-07-28, ar4-grasp-trivial-friction-check task, direct user decision -
# see CUBE_CFG's own docstring above for the full rationale/derivation).
# friction_combine_mode="average" is Isaac Lab's own RigidBodyMaterialCfg
# default, set explicitly here (not left implicit) so this is not
# accidentally re-defaulted to "min" later, which would let a low-friction
# material on either side of a contact cap the effective friction.
CUBE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.7,
    friction_combine_mode="average",
    restitution_combine_mode="average",
)


def _check_wedge_usd_exists() -> str:
    if not os.path.isfile(_WEDGE_USD_PATH):
        sys.exit(
            f"Wedge USD asset not found (missing {_WEDGE_USD_PATH}).\n"
            "Run scripts/build_asset.py first."
        )
    return _WEDGE_USD_PATH


CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    # 2026-07-24 (ar4-cube-size-increase task, direct user decision): bumped
    # from 12mm to 20mm, then to 15mm (also 2026-07-24, mirrors the same
    # decision process). Rationale for the original 12mm->20mm bump: the
    # ~9-10mm classical-IK positioning residual (see
    # kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md) means only
    # one gripper jaw reliably reaches the cube; a bigger cube gives both
    # jaws more surface area to reach despite the same residual. Also
    # directly addressed a separate, standing visual-confirmation gap
    # (existing cameras don't resolve the 12mm cube clearly at render
    # distance - ROADMAP.md's 2026-07-24 ar4-jaw-contact-sensor-hypothesis
    # entry). Gripper max aperture is ~28mm (see module docstring), so 15mm
    # still fits with 6.5mm clearance per side (looser than 20mm's 4mm,
    # tighter than the original 12mm's 8mm, but physically valid).
    # init_state pos z is half the new size (0.0075) so the cube still
    # rests exactly on the table.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.0075)),
    spawn=sim_utils.CuboidCfg(
        size=(0.015, 0.015, 0.015),
        rigid_props=_RIGID_PROPS,
        mass_props=_MASS,
        collision_props=_COLLISION_PROPS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        physics_material=CUBE_PHYSICS_MATERIAL,
    ),
)

PEDESTAL_HEIGHT_M = 0.040
"""Static raised-work-surface height (2026-07-28, ar4-pedestal-ground-
clearance-fix task) - MUST match scripts/ar4_graspable_workspace.py's own
PEDESTAL_HEIGHT_M (that module's own comment has the full derivation/
provenance: the graspable workspace with a cube resting directly on the
ground (z=0) was found to be genuinely EMPTY - even in the best case found
anywhere in an 8M-sample FK sweep, the gripper's real fingertip would sit
3.55mm BELOW the ground plane, a real geometric floor collision the
gripper's own ~18.5mm finger length causes for ANY near-vertical grasp of a
cube this size resting flush on the ground. Raising the cube 40mm off the
ground (mirroring real pick-and-place - objects on a raised work surface,
not on the floor at the robot's own base) fixes this: a cheap FK probe
across candidate heights found the achievable ground-clearance ceiling
rises almost exactly 1:1 with pedestal height, and 40mm gives a real
fingertip-clearance ceiling of +35.77mm - over 3x the existing
GROUND_CLEARANCE_MIN_M filter's own already-conservative required margin,
comfortably inside this task's own dispatch brief's suggested 40-60mm
band. Duplicated here (not imported) since ar4_graspable_workspace.py
is a plain-numpy/no-isaaclab-import module by design (importable before an
AppLauncher exists - see its own module docstring) and this file requires
isaaclab.sim already."""


def make_pedestal_cfg(
    prim_path: str,
    center_xy: tuple[float, float],
    height_m: float = PEDESTAL_HEIGHT_M,
    footprint_xy_m: tuple[float, float] = (0.10, 0.10),
) -> AssetBaseCfg:
    """A static (non-rigid-body) raised platform the cube rests on top of.

    Plain AssetBaseCfg + CuboidCfg spawn with collision but NO rigid_props -
    matching this repo's existing static-prop precedent (e.g.
    tasks/franka/dice_scene_cfg.py's own `table`/`_notch_fixture_cfg`, and
    this scene's own ground plane) rather than a RigidObjectCfg: nothing
    needs to read this prim's own pose/velocity data at runtime (it never
    moves), so the extra RigidObject data-buffer machinery would be pure
    overhead, and a prim with CollisionAPI but no RigidBodyAPI is treated by
    PhysX as an ordinary static collider - exactly what a fixed pedestal is.

    ``footprint_xy_m`` ((10cm, 10cm) square by default, but callers may pass
    a non-square rectangle - see this task's own closing-grasp scene, which
    needs an elongated platform to cover 3 validation points spread ~18cm
    apart in X but only ~2cm apart in Y) is deliberately generous relative to
    the 15mm cube and the gripper's own ~28mm max aperture - comfortably
    covers a modest local cluster of nearby candidate grasp points (so the
    SAME static pedestal can underlie multiple validation points without
    needing to be repositioned, since a static AssetBaseCfg prim cannot be
    teleported at runtime the way the RigidObject cube can) while staying far
    smaller than the reach radius (~0.3-0.4m) separating it from the robot's
    own base, so it cannot obstruct the arm's approach path."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(center_xy[0], center_xy[1], height_m / 2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(footprint_xy_m[0], footprint_xy_m[1], height_m),
            collision_props=_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
        ),
    )


# Centered on the centroid of the 3 pedestal-height-corrected validation
# points found by scripts/_ar4_pedestal_select_grasp_points.py this task
# (Q0_bearing90 cube_xy=(-0.0327, 0.3737), Q1_bearing104=(-0.1157, 0.3568),
# Q2_bearing76=(0.0634, 0.3752)) - x range 0.1791m (Q1 to Q2), y range
# 0.0184m, so the platform is a wide-but-shallow strip (0.30 x 0.14m, ~5-6cm
# margin beyond each point's own extent on every side) rather than square,
# sized to underlie all 3 without repositioning (see make_pedestal_cfg's own
# docstring for why a single static pedestal needs this).
GRASP_PEDESTAL_CENTER_XY = (-0.0262, 0.3660)
GRASP_PEDESTAL_FOOTPRINT_XY_M = (0.30, 0.14)

PEDESTAL_CFG = make_pedestal_cfg(
    prim_path="{ENV_REGEX_NS}/Pedestal",
    center_xy=GRASP_PEDESTAL_CENTER_XY,
    footprint_xy_m=GRASP_PEDESTAL_FOOTPRINT_XY_M,
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

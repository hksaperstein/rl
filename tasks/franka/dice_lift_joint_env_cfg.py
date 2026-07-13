# tasks/franka/dice_lift_joint_env_cfg.py
"""Joint-space (no-IK) d20-die-lift variant of the Franka lift task.

Subclasses tasks/franka/lift_env_cfg.py's FrankaLiftEnvCfg and overrides
exactly two things (the experiment's two variables, per
docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md):

1. arm_action: DifferentialInverseKinematicsActionCfg (task-space IK) ->
   JointPositionActionCfg with scale=0.5, use_default_offset=True - the
   exact values of Isaac Lab's own validated joint_pos lift variant
   (isaaclab_tasks/.../lift/config/franka/joint_pos_env_cfg.py:34-36,
   read directly), which is the only lift variant Isaac Lab ships RL
   agent configs for (see the research doc). No IK anywhere.
2. object: DexCube -> physics-baked d20 die (assets/dice/d20_physics.usd,
   Task 1 of the plan; default prim 'Object' so the stock recipe's
   SceneEntityCfg("object", body_names="Object") terms match unchanged),
   spawn-time scale 0.001 (mm-as-m source units, dice-demo convention),
   same solver-iteration rigid props as the DexCube recipe.

Everything else (rewards, observations, commands, events, terminations,
curriculum, episode length, PPO cfg) inherits byte-identical from
FrankaLiftEnvCfg. Import only after an AppLauncher exists.
"""

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass

from . import mdp
from .lift_env_cfg import FrankaLiftEnvCfg

_D20_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "d20_physics.usd",
)

_CUBE48_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "cube48_physics.usd",
)

# Extracted from FrankaDieLiftJointEnvCfg's original inline RigidBodyPropertiesCfg literal
# (single source of truth for both the base class and FrankaDieLiftJointMixedEnvCfg's
# per-asset entries, task-1-brief.md Step 2 note) - value-identical to the prior inline block.
_D20_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


@configclass
class FrankaDieLiftJointEnvCfg(FrankaLiftEnvCfg):
    """d20 lift with direct joint-position arm actions (no IK)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Variable 1: joint-space arm action (exact Isaac Lab joint_pos values).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # gripper_action inherited unchanged (BinaryJointPositionActionCfg).

        # Variable 2: the d20 die replaces the DexCube.
        if not os.path.isfile(_D20_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py: {_D20_USD}")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=_D20_USD,
                scale=(0.001, 0.001, 0.001),
                rigid_props=_D20_RIGID_PROPS,
            ),
        )


@configclass
class FrankaDieLiftJointEnvCfg_PLAY(FrankaDieLiftJointEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftJointEnvCfg(FrankaLiftEnvCfg):
    """Fallback rung (spec's verdict protocol, fired 2026-07-12 after the
    d20 run FAILED the position_error criterion): joint-space arm action
    with the recipe's own DexCube kept as the object, to isolate
    asset-vs-recipe. Applies ONLY the die variant's Variable 1 (identical
    JointPositionActionCfg values); the object is inherited byte-identical
    from FrankaLiftEnvCfg rather than swapped back, so the only diff vs
    the validated ik-cube baseline is the action space, and the only diff
    vs FrankaDieLiftJointEnvCfg is the object."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )


@configclass
class FrankaCubeLiftJointEnvCfg_PLAY(FrankaCubeLiftJointEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointHeavyEnvCfg(FrankaDieLiftJointEnvCfg):
    """Asset-bisect rung 1 (docs/superpowers/specs/2026-07-12-asset-bisect-design.md):
    the d20 with its mass raised 0.0100kg -> 0.216kg (DexCube's measured
    live PhysX mass, scripts/_diag_object_mass_check.py 2026-07-12).
    Shape, 30.3mm size, friction, and the whole joint-space config stay
    pinned - mass is this rung's ONLY variable."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=0.216)


@configclass
class FrankaDieLiftJointHeavyEnvCfg_PLAY(FrankaDieLiftJointHeavyEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointBigEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Asset-bisect rung 2: d20 scaled 30.3mm -> 48.0mm (DexCube's
    measured effective size) with mass PINNED at 0.216kg by the inherited
    mass_props override - size is this rung's ONLY new variable (letting
    mass scale with volume would silently reintroduce rung 1's variable,
    per the spec)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.scale = (0.001585, 0.001585, 0.001585)


@configclass
class FrankaDieLiftJointBigEnvCfg_PLAY(FrankaDieLiftJointBigEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeBakedLiftJointEnvCfg(FrankaDieLiftJointEnvCfg):
    """Asset-bisect rung 3 (docs/superpowers/specs/2026-07-12-asset-bisect-
    design.md): a flat-faced cube baked through this repo's OWN
    bake_die_asset.py pipeline (scripts/bake_die_asset.py --shape cube
    --size 48.0 --mass 0.216 -> assets/dice/cube48_physics.usd), at
    48.0mm/0.216kg - the same size and mass rung 2 already pinned. This
    isolates shape (rounded d20 vs flat-faced cube) from pipeline
    provenance: if this cube trains reliably where the same-size/mass d20
    (rung 2) did not, the die's own rolling geometry is implicated rather
    than anything about the bake pipeline itself (a genuine DexCube, not
    of our own provenance, is rung 4's separate control). Only usd_path
    and mass_props are overridden; scale stays inherited at 0.001 (48
    stage units -> 48mm, same mm-as-m convention as every other baked
    asset in this file)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_CUBE48_USD):
            raise FileNotFoundError(
                f"baked cube48 asset missing - run scripts/bake_die_asset.py --shape cube --size 48.0 "
                f"--mass 0.216: {_CUBE48_USD}"
            )
        self.scene.object.spawn.usd_path = _CUBE48_USD
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=0.216)


@configclass
class FrankaCubeBakedLiftJointEnvCfg_PLAY(FrankaCubeBakedLiftJointEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointMixedEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Size-curriculum primary arm (docs/superpowers/specs/2026-07-13-size-
    curriculum-design.md): per-env die size varied across {48.0, 43.6,
    39.1, 34.7, 30.3}mm (deterministic round-robin via MultiAssetSpawnerCfg
    with random_choice=False - confirmed by direct source read of
    isaaclab/sim/spawners/wrappers/wrappers.py:spawn_multi_asset,
    task-1-report.md - proto_prim_paths[index % len(proto_prim_paths)],
    giving an exact ~819-envs-per-size split over 4096 envs), mass pinned
    0.216kg on every size (rung 1/2's already-pinned value, not a new
    variable here). Everything else inherits from the heavy variant
    unchanged.

    scene.replicate_physics = False (controller decision, task-1-report.md
    NEEDS_CONTEXT finding): InteractiveSceneCfg's replicate_physics defaults
    True, and when True InteractiveScene.clone_environments(copy_from_source=
    False) clones every env from env_0's own content AFTER spawn_multi_asset
    has already authored heterogeneous per-env assets - silently discarding
    the round-robin variation before it reaches the live PhysX stage (live-
    verified: 16 envs all read the SAME 48.0mm scale with this unset). Isaac
    Lab's own only other MultiAssetSpawnerCfg consumer,
    isaaclab_tasks/manager_based/manipulation/dexsuite/dexsuite_env_cfg.py:395,
    sets this same flag for the same reason. Scoped to this class only -
    every other variant in this file keeps the InteractiveSceneCfg default
    (True), unaffected."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.replicate_physics = False
        _scales = (0.001585, 0.001440, 0.001291, 0.001146, 0.001000)
        self.scene.object.spawn = MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    usd_path=_D20_USD,
                    scale=(s, s, s),
                    rigid_props=_D20_RIGID_PROPS,
                    mass_props=MassPropertiesCfg(mass=0.216),
                )
                for s in _scales
            ],
            random_choice=False,
        )


@configclass
class FrankaDieLiftJointMixedEnvCfg_PLAY(FrankaDieLiftJointMixedEnvCfg):
    """All-30.3mm eval probe (the spec's verdict measurement) - a single
    UsdFileCfg (not a MultiAssetSpawnerCfg), matching FrankaDieLiftJointEnvCfg
    at its original size/mass."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn = UsdFileCfg(
            usd_path=_D20_USD,
            scale=(0.001, 0.001, 0.001),
            rigid_props=_D20_RIGID_PROPS,
            mass_props=MassPropertiesCfg(mass=0.216),
        )
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

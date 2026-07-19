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

# Task 0 of docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
# distillation.md: d8/d10/d12 physics assets baked via scripts/bake_die_asset.py
# --die {d8,d10,d12} (no code changes to that script - it already supports
# all 5 die choices).
_D8_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "d8_physics.usd",
)

_D10_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "d10_physics.usd",
)

_D12_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "d12_physics.usd",
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

# Task 5 (FrankaDieLiftJointD12D20MixedEnvCfg below): the exact assets-list
# order/length its MultiAssetSpawnerCfg is built with - a plain module-level
# constant (not a class attribute) so it can't be mistaken for a configclass/
# dataclass field by isaaclab.utils.configclass's field-processing.
_D12D20_MIXED_ASSETS_ORDER = ("d12", "d20")


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
        # Task 1 (docs/superpowers/plans/2026-07-16-unified-multi-die-
        # specialist-distillation.md): explicit even though "d20" is also
        # FrankaLiftEnvCfg's own inherited default - every die-bearing env
        # cfg in this file sets this explicitly rather than relying on the
        # base class's cube-recipe fallback value.
        self.die_shape_class = "d20"


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
class FrankaDieLiftJointStandardEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Corrected standard-size rung (2026-07-15 ROADMAP entry): a real
    *standard* commercial d20 measures ~20-22mm, not 30.3mm - 30.3mm is
    itself a real, commonly-sold "jumbo" d20 size (e.g. Twenty Sided
    Store's "Jumbo Dice 30mm D20"), not a mistake, but every rung above
    (Heavy/Big/Mid/Mixed) anneals *toward* 30.3mm as if it were the true
    target. This class does not correct or reinterpret any of that prior
    history (the asset-bisect/size-curriculum verdicts at 30.3mm/48.0mm/
    etc. stand as originally reported) - it only adds the forward-facing
    22mm target for future d20 grasp work. Mass stays pinned at 0.216kg
    via the inherited FrankaDieLiftJointHeavyEnvCfg mass_props override
    (asset-bisect rung 1 finding: light objects cause PhysX depenetration-
    impulse chaos - not this change's variable); size is the only new
    variable here.

    Scale derivation: this file's four non-base rungs give scale-per-mm
    ratios of 0.001585/48.0=3.302083e-5, 0.001440/43.6=3.302752e-5,
    0.001291/39.1=3.301790e-5, 0.001146/34.7=3.302594e-5 (average
    3.302305e-5/mm - display sizes are rounded to 1 decimal but the scale
    constants themselves are precise to 6 decimals, so the ratio is fit
    from the constants, not the rounded mm labels). 22mm * 3.302305e-5 =
    7.265071e-4, rounded to this file's 6-decimal convention: 0.000727.
    Live-verified via scripts/_diag_d20_standard_scale_check.py at this
    scale (see ROADMAP.md's 2026-07-15 entry for the measured bbox)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.scale = (0.000727, 0.000727, 0.000727)


@configclass
class FrankaDieLiftJointStandardEnvCfg_PLAY(FrankaDieLiftJointStandardEnvCfg):
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


@configclass
class FrankaDieLiftJointMidEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Staged-anneal fallback, stage 2 (docs/superpowers/specs/2026-07-13-
    size-curriculum-design.md Verdict section - fired after the mixed-size
    primary arm was FALSIFIED 0/3): d20 scaled 30.3mm -> 39.1mm, the
    curriculum's middle rung between stage 1 (48.0mm, the existing
    `joint-die-big` variant) and stage 3 (30.3mm, the existing
    `joint-die-heavy` variant). Mass stays pinned at 0.216kg via the
    inherited mass_props override, same pattern as FrankaDieLiftJointBigEnvCfg
    - size is this rung's only new variable relative to its parent. Trained
    by resuming each seed's stage-1 checkpoint via train_franka.py's existing
    --checkpoint flag; not a from-scratch run."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.scale = (0.001291, 0.001291, 0.001291)


@configclass
class FrankaDieLiftJointMidEnvCfg_PLAY(FrankaDieLiftJointMidEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointRandomSizeEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d20 size-domain-randomization + geometry-feature retry (Task 3,
    docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
    distillation.md): retries the already-falsified
    FrankaDieLiftJointMixedEnvCfg's 0/3 size-curriculum result
    (docs/superpowers/specs/2026-07-13-size-curriculum-design.md verdict,
    ROADMAP.md 2026-07-13) with exactly one new ingredient - Task 1's
    object_shape_class_onehot/object_geometry_descriptor observation terms,
    already present on every FrankaLiftEnvCfg subclass via die_shape_class
    (here "d20", inherited unchanged from FrankaDieLiftJointEnvCfg, nothing
    reimplemented) - to test whether geometry-descriptor conditioning lets
    the same per-env size-population-dilution mechanism that failed without
    it succeed instead.

    MECHANISM CORRECTION (task-3-brief.md Step 1, confirmed by direct read
    of isaaclab/sim/spawners/wrappers/wrappers.py::spawn_multi_asset,
    task-3-report.md): random_choice=True does NOT resample per-episode-
    reset. Exactly like FrankaDieLiftJointMixedEnvCfg's own
    random_choice=False, spawn_multi_asset assigns each environment ONE
    size, ONCE, at scene-spawn time - random_choice=True only changes the
    *assignment pattern* from deterministic round-robin
    (proto_prim_paths[index % len(proto_prim_paths)]) to per-env-index
    random.choice(proto_prim_paths), not from "fixed" to "resampled". Every
    env keeps the SAME assigned size for the entire training run under
    either setting. Mechanically this class is IDENTICAL to
    FrankaDieLiftJointMixedEnvCfg except for (a) random vs. round-robin
    per-env assignment pattern (chosen here per the brief's explicit
    instruction, to also diversify which size lands on which env index
    across the 3 training seeds, unlike Mixed's seed-invariant round-robin)
    and (b) the geometry-descriptor observation conditioning, which is the
    actual new variable under test. The class name ("RandomSize") names the
    random_choice=True *assignment pattern*, not a resampling claim - it
    must not be read as implying per-episode size variation, which
    spawn_multi_asset does not support in either mode.

    Size range: 22.0mm (FrankaDieLiftJointStandardEnvCfg's already-verified
    0.000727 scale - this repo's forward-facing real-standard-d20-size
    target) to 48.0mm (FrankaDieLiftJointBigEnvCfg's already-verified
    0.001585 scale - the asset-bisect anchor where d20 achieved its known
    1/3 discovery baseline) - both endpoints' scale constants reused
    directly from those two classes, not re-derived. 5 discrete sizes
    (matching FrankaDieLiftJointMixedEnvCfg's own 5-point precedent), evenly
    spaced in mm using FrankaDieLiftJointStandardEnvCfg's own fitted
    scale-per-mm ratio for the d20 mesh (3.302305e-5/mm): 22.0mm (0.000727),
    28.5mm (0.000941), 35.0mm (0.001156), 41.5mm (0.001370), 48.0mm
    (0.001585). Mass pinned at 0.216kg on every size (the same placeholder
    used across every rung in this file, not a new variable here).

    scene.replicate_physics = False (same reason as
    FrankaDieLiftJointMixedEnvCfg: InteractiveSceneCfg's default True clones
    every env from env_0's own content AFTER spawn_multi_asset has already
    authored heterogeneous per-env assets, silently discarding the per-env
    size variation before it reaches the live PhysX stage)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.replicate_physics = False
        _scales = (0.001585, 0.001370, 0.001156, 0.000941, 0.000727)
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
            random_choice=True,
        )


@configclass
class FrankaDieLiftJointRandomSizeEnvCfg_PLAY(FrankaDieLiftJointRandomSizeEnvCfg):
    """All-30.3mm-equivalent-scale eval probe - a single UsdFileCfg (not a
    MultiAssetSpawnerCfg), matching FrankaDieLiftJointEnvCfg's original
    size/mass, same pattern as FrankaDieLiftJointMixedEnvCfg_PLAY. Instrumented
    eval (Step 4) additionally sweeps the full size range via separate
    ad hoc scale overrides on top of this class - see task-3-report.md."""

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


@configclass
class FrankaDieLiftJointD8StandardEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d8 specialist env cfg (Task 0, docs/superpowers/plans/2026-07-16-
    unified-multi-die-specialist-distillation.md): physics-baked d8 die
    (assets/dice/d8_physics.usd) at its real standard commercial size,
    ~16mm (Task 0 Step 1 web research - convergent consensus of multiple
    independent tabletop-gaming retailer/blog size guides, not one single
    clean citation the way d20's "Twenty Sided Store Jumbo 30mm D20"
    listing was; see .superpowers/sdd/task-0-report.md for exact sources).
    Mass stays pinned at 0.216kg via the inherited
    FrankaDieLiftJointHeavyEnvCfg mass_props override - this is DexCube's
    own measured live PhysX mass (see that class's own docstring), NOT a
    real d8-density estimate. No established method exists yet to derive a
    real per-shape mass for d8/d10/d12 analogously to d20 (d20's own
    0.216kg was never itself a real-world-density derivation - see
    task-0-report.md's concern section for the full explanation); this is
    a carried-over placeholder pending real per-shape mass research, not a
    considered value for this shape specifically.

    Scale derivation: unlike FrankaDieLiftJointStandardEnvCfg (which reused
    this file's existing d20-rung scale constants' fitted scale-per-mm
    ratio, valid only for the d20 mesh), d8's own native (unscaled) baked
    mesh bbox is a different size than d20's, so this scale is freshly fit
    from a direct measurement of d8_physics.usd
    (scripts/_diag_d8d10d12_standard_scale_check.py): native_max_dim =
    15.1544 stage units (isotropic - d8's bbox measures 15.1544 on all
    three axes), target 16.0mm -> scale = 16.0 / (15.1544 * 1000) =
    1.055799e-3, rounded to this file's 6-decimal convention: 0.001056.
    Live-verified: measured effective max dim at this scale is 16.003mm
    (delta +0.003mm), within the 0.3mm tolerance used throughout this
    file."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D8_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d8: {_D8_USD}")
        self.scene.object.spawn.usd_path = _D8_USD
        self.scene.object.spawn.scale = (0.001056, 0.001056, 0.001056)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d8"


@configclass
class FrankaDieLiftJointD8StandardEnvCfg_PLAY(FrankaDieLiftJointD8StandardEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD10StandardEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d10 specialist env cfg (Task 0, docs/superpowers/plans/2026-07-16-
    unified-multi-die-specialist-distillation.md): physics-baked d10 die
    (assets/dice/d10_physics.usd) at its real standard commercial size,
    ~16mm face-to-face (Task 0 Step 1 web research; see
    .superpowers/sdd/task-0-report.md for exact sources - one source
    explicitly states the face-to-face convention, others report a
    convergent ~16mm figure without stating their measurement axis; a
    ~23mm point-to-point figure for the same 16mm-class d10 could not be
    pinned to a citable source and was not used). Mass stays pinned at
    0.216kg via the inherited FrankaDieLiftJointHeavyEnvCfg mass_props
    override - the same DexCube-measured carried-over placeholder value
    used across every shape in this file, not a real d10-density estimate
    (see FrankaDieLiftJointD8StandardEnvCfg's docstring and
    task-0-report.md's concern section for why no real per-shape mass
    derivation exists yet).

    Scale derivation: d10's native (unscaled) baked mesh bbox is
    ANISOTROPIC (a pentagonal trapezohedron is not a shape with equal
    extent on all three axes, unlike d8/d12's near-regular measured boxes)
    - measured directly via scripts/_diag_d8d10d12_standard_scale_check.py:
    native bbox = 16.3931 x 15.7156 x 14.9345 stage units, native_max_dim =
    16.3931 (the longest axis). scale = 16.0 / (16.3931 * 1000) =
    9.760209e-4, rounded to this file's 6-decimal convention: 0.000976.
    Live-verified: measured effective dims at this scale are 16.000 x
    15.338 x 14.576mm - the longest axis lands at 16.000mm (delta
    -0.000mm), within the 0.3mm tolerance. The other two axes are smaller
    by construction (uniform scale applied to an anisotropic native bbox);
    this matches the file's existing convention of fitting scale against
    the single longest/max dimension (see FrankaDieLiftJointStandardEnvCfg
    and _diag_d20_standard_scale_check.py, which both use "max dim" as the
    controlling measurement)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D10_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d10: {_D10_USD}")
        self.scene.object.spawn.usd_path = _D10_USD
        self.scene.object.spawn.scale = (0.000976, 0.000976, 0.000976)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d10"


@configclass
class FrankaDieLiftJointD10StandardEnvCfg_PLAY(FrankaDieLiftJointD10StandardEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD12StandardEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d12 specialist env cfg (Task 0, docs/superpowers/plans/2026-07-16-
    unified-multi-die-specialist-distillation.md): physics-baked d12 die
    (assets/dice/d12_physics.usd) at its real standard commercial size,
    ~18mm face-to-face (Task 0 Step 1 web research: multiple independent
    retailer/blog size guides converge on ~18mm, cross-checked against a
    real Chessex product listing itself named "Planets Blue/white 18mm
    d12" - Chessex's own face-to-face-labeled product naming convention;
    see .superpowers/sdd/task-0-report.md for exact sources). Mass stays
    pinned at 0.216kg via the inherited FrankaDieLiftJointHeavyEnvCfg
    mass_props override - the same DexCube-measured carried-over
    placeholder value used across every shape in this file, not a real
    d12-density estimate (see FrankaDieLiftJointD8StandardEnvCfg's
    docstring and task-0-report.md's concern section for why no real
    per-shape mass derivation exists yet).

    Scale derivation: d12's native (unscaled) baked mesh bbox measured
    directly via scripts/_diag_d8d10d12_standard_scale_check.py:
    native_max_dim = 32.5160 stage units (isotropic - d12's bbox measures
    32.5160 on all three axes). scale = 18.0 / (32.5160 * 1000) =
    5.535732e-4, rounded to this file's 6-decimal convention: 0.000554.
    Live-verified: measured effective max dim at this scale is 18.014mm
    (delta +0.014mm), within the 0.3mm tolerance used throughout this
    file."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D12_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d12: {_D12_USD}")
        self.scene.object.spawn.usd_path = _D12_USD
        self.scene.object.spawn.scale = (0.000554, 0.000554, 0.000554)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d12"


@configclass
class FrankaDieLiftJointD12StandardEnvCfg_PLAY(FrankaDieLiftJointD12StandardEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD8BigEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d8 48mm-parity specialist env cfg (Task 3.5, docs/superpowers/plans/
    2026-07-16-unified-multi-die-specialist-distillation.md, gate inserted
    2026-07-16 before Task 4): physics-baked d8 die (assets/dice/
    d8_physics.usd) scaled to the asset-bisect's own 48.0mm anchor size -
    the SAME size/mass point where the asset-bisect ladder got real
    discovery for the cube (3/3) and d20 (1/3), evaluated here as a
    single, undiluted 48mm population, exactly like both of those runs
    (not mixed with any other size, unlike Task 2's own ~16mm-real-size
    d8 run and Task 3's d20 size-DR sweep). Mass stays pinned at 0.216kg
    via the inherited FrankaDieLiftJointHeavyEnvCfg mass_props override -
    the same DexCube-measured carried-over placeholder value used across
    every shape/size in this file, not a real d8-density estimate (see
    FrankaDieLiftJointD8StandardEnvCfg's docstring for why no real
    per-shape mass derivation exists yet).

    Scale derivation (per this task's own explicit warning: d20's
    FrankaDieLiftJointBigEnvCfg 0.001585 constant does NOT transfer here -
    d8's native mesh bbox is a different absolute size than d20's, so the
    scale that hits 48.0mm for d20 does not hit 48.0mm for d8). Freshly
    fit from Task 0's own direct measurement of d8_physics.usd
    (scripts/_diag_d8d10d12_standard_scale_check.py, same measurement
    FrankaDieLiftJointD8StandardEnvCfg's own docstring cites):
    native_max_dim = 15.1544 stage units (isotropic - d8's bbox measures
    15.1544 on all three axes). scale = 48.0 / (15.1544 * 1000) =
    3.167397e-3, rounded to this file's 6-decimal convention: 0.003167.
    Effective max dim at this scale = 47.994mm (delta -0.006mm from the
    48.0mm target), well within the 0.3mm tolerance used throughout this
    file."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D8_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d8: {_D8_USD}")
        self.scene.object.spawn.usd_path = _D8_USD
        self.scene.object.spawn.scale = (0.003167, 0.003167, 0.003167)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d8"


@configclass
class FrankaDieLiftJointD8BigEnvCfg_PLAY(FrankaDieLiftJointD8BigEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD10BigEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d10 48mm-parity specialist env cfg (Task 3.5, docs/superpowers/plans/
    2026-07-16-unified-multi-die-specialist-distillation.md, gate inserted
    2026-07-16 before Task 4): physics-baked d10 die (assets/dice/
    d10_physics.usd) scaled to the asset-bisect's own 48.0mm anchor size -
    the SAME size/mass point where the asset-bisect ladder got real
    discovery for the cube (3/3) and d20 (1/3), evaluated here as a
    single, undiluted 48mm population, exactly like both of those runs.
    Mass stays pinned at 0.216kg via the inherited
    FrankaDieLiftJointHeavyEnvCfg mass_props override - the same
    DexCube-measured carried-over placeholder value used across every
    shape/size in this file, not a real d10-density estimate (see
    FrankaDieLiftJointD8StandardEnvCfg's docstring and
    FrankaDieLiftJointD10StandardEnvCfg's docstring for why no real
    per-shape mass derivation exists yet).

    Scale derivation (per this task's own explicit warning: d20's
    FrankaDieLiftJointBigEnvCfg 0.001585 constant does NOT transfer here -
    d10's native mesh bbox is a different absolute size than d20's).
    Freshly fit from Task 0's own direct measurement of d10_physics.usd
    (scripts/_diag_d8d10d12_standard_scale_check.py, same measurement
    FrankaDieLiftJointD10StandardEnvCfg's own docstring cites): native
    bbox = 16.3931 x 15.7156 x 14.9345 stage units (ANISOTROPIC - a
    pentagonal trapezohedron, not equal-extent on all three axes),
    native_max_dim = 16.3931 (the longest axis). scale = 48.0 /
    (16.3931 * 1000) = 2.928061e-3, rounded to this file's 6-decimal
    convention: 0.002928. Effective dims at this scale: 47.999 x 46.015 x
    43.728mm - the longest axis lands at 47.999mm (delta -0.001mm from
    the 48.0mm target), within the 0.3mm tolerance. The other two axes
    are smaller by construction (uniform scale applied to an anisotropic
    native bbox), matching this file's existing convention of fitting
    scale against the single longest/max dimension (see
    FrankaDieLiftJointD10StandardEnvCfg and
    _diag_d8d10d12_standard_scale_check.py, which both use "max dim" as
    the controlling measurement)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D10_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d10: {_D10_USD}")
        self.scene.object.spawn.usd_path = _D10_USD
        self.scene.object.spawn.scale = (0.002928, 0.002928, 0.002928)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d10"


@configclass
class FrankaDieLiftJointD10BigEnvCfg_PLAY(FrankaDieLiftJointD10BigEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD12BigEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """d12 48mm-parity specialist env cfg (Task 3.5, docs/superpowers/plans/
    2026-07-16-unified-multi-die-specialist-distillation.md, gate inserted
    2026-07-16 before Task 4): physics-baked d12 die (assets/dice/
    d12_physics.usd) scaled to the asset-bisect's own 48.0mm anchor size -
    the SAME size/mass point where the asset-bisect ladder got real
    discovery for the cube (3/3) and d20 (1/3), evaluated here as a
    single, undiluted 48mm population, exactly like both of those runs.
    Mass stays pinned at 0.216kg via the inherited
    FrankaDieLiftJointHeavyEnvCfg mass_props override - the same
    DexCube-measured carried-over placeholder value used across every
    shape/size in this file, not a real d12-density estimate (see
    FrankaDieLiftJointD8StandardEnvCfg's docstring and
    FrankaDieLiftJointD12StandardEnvCfg's docstring for why no real
    per-shape mass derivation exists yet).

    Scale derivation (per this task's own explicit warning: d20's
    FrankaDieLiftJointBigEnvCfg 0.001585 constant does NOT transfer here -
    d12's native mesh bbox is a different absolute size than d20's).
    Freshly fit from Task 0's own direct measurement of d12_physics.usd
    (scripts/_diag_d8d10d12_standard_scale_check.py, same measurement
    FrankaDieLiftJointD12StandardEnvCfg's own docstring cites):
    native_max_dim = 32.5160 stage units (isotropic - d12's bbox measures
    32.5160 on all three axes). scale = 48.0 / (32.5160 * 1000) =
    1.476196e-3, rounded to this file's 6-decimal convention: 0.001476.
    Effective max dim at this scale = 47.994mm (delta -0.006mm from the
    48.0mm target), well within the 0.3mm tolerance used throughout this
    file."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D12_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d12: {_D12_USD}")
        self.scene.object.spawn.usd_path = _D12_USD
        self.scene.object.spawn.scale = (0.001476, 0.001476, 0.001476)
        # Task 1: this env cfg's shape-class constant (mdp.object_shape_class_onehot/
        # object_geometry_descriptor) - overrides the d20 inherited from
        # FrankaDieLiftJointHeavyEnvCfg.
        self.die_shape_class = "d12"


@configclass
class FrankaDieLiftJointD12BigEnvCfg_PLAY(FrankaDieLiftJointD12BigEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaDieLiftJointD12D20MixedEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Task 5 mixed-population env (docs/superpowers/plans/2026-07-16-
    unified-multi-die-specialist-distillation.md Task 5; BACKLOG.md's
    2026-07-19 controller decision "(b) single mixed-population env" -
    read that entry for the full architectural rationale). Replaces the
    original two-envs-side-by-side distillation design
    (`tasks/franka/distillation.py`'s module docstring), which hit a real
    Isaac Lab limitation: a second `ManagerBasedRLEnv` cannot be constructed
    in-process after a first one is built, either simultaneously
    (`RuntimeError: Simulation context already exists`) or sequentially
    after `.close()` (hangs indefinitely - independently confirmed via an
    isolated minimal repro, BACKLOG.md's BLOCKED entry). This class instead
    splits `num_envs` between d12 and d20 within ONE `ManagerBasedRLEnv`.

    Mechanism: `MultiAssetSpawnerCfg(assets_cfg=[d12_cfg, d20_cfg],
    random_choice=False)` - the exact same deterministic-round-robin,
    per-env-fixed-at-spawn-time mechanism `FrankaDieLiftJointMixedEnvCfg`
    already uses for per-env SIZE (not shape) variation, reused here for
    per-env SHAPE variation instead. Confirmed by a fresh direct source
    read for this task (not just trusting that class's own docstring
    citation), `isaaclab/sim/spawners/wrappers/wrappers.py::
    spawn_multi_asset`:

        proto_path = proto_prim_paths[index % len(proto_prim_paths)]

    where `index` enumerates `prim_paths` (each env's own `.../Object` prim,
    resolved via `find_matching_prim_paths` and iterated in ascending
    env-index order) and `proto_prim_paths` preserves `cfg.assets_cfg`'s own
    list order. So with `assets_cfg=[d12_cfg, d20_cfg]`
    (`_D12D20_MIXED_ASSETS_ORDER`, this module's own top-level constant):
    env 0 -> d12, env 1 -> d20, env 2 -> d12, env 3 -> d20, ... - an exact,
    deterministic 50/50 split (up to the +/-1 parity remainder for an odd
    `num_envs`), matching `FrankaDieLiftJointMixedEnvCfg`'s own already-
    verified ~819-envs-per-size 5-way split at 4096 envs. `self.
    die_shape_classes_per_env = _D12D20_MIXED_ASSETS_ORDER` (set below) tells
    `tasks/franka/mdp.py`'s `object_shape_class_onehot`/
    `object_geometry_descriptor` to replicate this exact `index % len(...)`
    formula themselves (no live USD/spawner-state query needed - the
    controller decision's own de-risking argument) instead of the single-
    shape broadcast every other env cfg in this file uses.

    Both shapes at their own already-verified 48mm-parity scale/mass -
    reused directly from `FrankaDieLiftJointD12BigEnvCfg` (d12: scale
    0.001476) and `FrankaDieLiftJointBigEnvCfg` (d20: scale 0.001585), not
    re-derived; mass pinned at 0.216kg on both (this file's existing
    carried-over placeholder, same as every other rung).

    scene.replicate_physics = False (same reason as
    `FrankaDieLiftJointMixedEnvCfg`/`FrankaDieLiftJointRandomSizeEnvCfg`:
    `InteractiveSceneCfg`'s default True clones every env from env_0's own
    content AFTER `spawn_multi_asset` has already authored heterogeneous
    per-env assets, silently discarding the per-env shape variation before
    it reaches the live PhysX stage)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not os.path.isfile(_D12_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py --die d12: {_D12_USD}")
        # _D20_USD is already existence-checked by FrankaDieLiftJointEnvCfg's
        # own __post_init__, in the super() chain above.
        self.scene.replicate_physics = False
        self.die_shape_classes_per_env = _D12D20_MIXED_ASSETS_ORDER
        self.scene.object.spawn = MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    usd_path=_D12_USD,
                    scale=(0.001476, 0.001476, 0.001476),
                    rigid_props=_D20_RIGID_PROPS,
                    mass_props=MassPropertiesCfg(mass=0.216),
                ),
                UsdFileCfg(
                    usd_path=_D20_USD,
                    scale=(0.001585, 0.001585, 0.001585),
                    rigid_props=_D20_RIGID_PROPS,
                    mass_props=MassPropertiesCfg(mass=0.216),
                ),
            ],
            random_choice=False,
        )

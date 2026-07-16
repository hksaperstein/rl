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

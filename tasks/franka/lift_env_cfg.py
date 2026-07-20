# tasks/franka/lift_env_cfg.py
"""Franka Emika Panda cube-lift task: a from-scratch reproduction of Isaac
Lab's own official Franka cube-lift recipe, written fresh for the
franka-panda-pivot (see CLAUDE.md's "Platform pivot (2026-07-09)" section -
Franka replaces the AR4 as this project's primary arm, on a genuinely new
`tasks/franka/` package, no imports from `tasks/ar4/` or from Isaac Lab's
own `isaaclab_tasks.manager_based.manipulation.lift` task module).

Deliberately reproduces the STOCK recipe's reward shape (reaching_object /
lifting_object / object_goal_tracking(_fine_grained) + small action-rate/
joint-vel penalties), episode length (5.0s, dt=0.01, decimation=2), and
scene (Franka + table + DexCube + randomized UniformPoseCommand goal) -
see isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py and
.../config/franka/joint_pos_env_cfg.py (read directly, not imported) - so
the first empirical test on this new asset is "does Franka + the stock
recipe, in this repo's own fresh files, converge as published", not
confounded by a different reward/task design at the same time.

Action space is relative differential-IK (task-space/Cartesian), not the
plain joint-position variant, matching
isaaclab_tasks/.../lift/config/franka/ik_rel_env_cfg.py (also read
directly) - this project's own AR4-era Experiment 11 found task-space/
Cartesian action critical for reliable grasp, and CLAUDE.md's North Star
favors task-space action formulations for generalization, so the IK-rel
variant (itself an official, pre-validated Isaac Lab recipe, not a novel
mechanism) is used as this file's design basis rather than the plain
joint-position default.

All reward/observation/termination math is reimplemented from scratch in
tasks/franka/mdp.py and tasks/franka/lift_reward.py - not imported from
isaaclab_tasks' own lift task module. Only Isaac Lab's own official,
pre-built library/asset code (isaaclab_assets.robots.franka, the generic
isaaclab.envs.mdp manager-term library) is reused directly, per CLAUDE.md's
explicit exception for Isaac Lab's own installed package (as opposed to
this repo's own AR4-era code).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from . import mdp

# EE frame measurement point: panda_link0 -> panda_hand, offset 0.1034m along
# panda_hand's local +Z to the fingertip pinch point - the same
# officially-validated offset isaaclab_tasks' own franka/joint_pos_env_cfg.py
# uses (no recalibration needed here, unlike this project's AR4-era
# scripts/build_asset.py + multi-round EE-offset verification work). This is
# the point object_ee_distance/object_goal_distance measure from - distinct
# from the IK controller's own body_offset below, which is the point the
# differential-IK *controller* tracks a commanded pose to.
_EE_MEASUREMENT_OFFSET = (0.0, 0.0, 0.1034)

# IK control target offset: panda_hand -> the point the differential-IK
# controller drives to the commanded relative pose. Matches
# isaaclab_tasks/.../lift/config/franka/ik_rel_env_cfg.py's own
# DifferentialInverseKinematicsActionCfg.OffsetCfg exactly (0.107, not
# 0.1034 - two different official reference points for two different
# roles: sensing vs. control - both are stock, not a typo).
_IK_BODY_OFFSET = (0.0, 0.0, 0.107)


@configclass
class FrankaLiftSceneCfg(InteractiveSceneCfg):
    """Table + Franka Panda (high-PD variant, for tighter IK tracking) + a
    DexCube rigid-object target + an end-effector FrameTransformer sensor."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # Isaac Sim default-stage light rig (2026-07-13, direct user directive
    # "use default light rig for all work") — values verbatim from the
    # installed omni.kit.stage_templates default_stage.py template; HDR
    # vendored in tasks/franka/assets/. Visualization-only for this
    # camera-less RL env: zero physics/observation/reward effect.
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

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_MEASUREMENT_OFFSET),
            ),
        ],
    )


@configclass
class ActionsCfg:
    """Relative differential-IK arm action (task-space/Cartesian) + binary gripper action."""

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_IK_BODY_OFFSET),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class CommandsCfg:
    """Randomized goal pose for the cube, resampled every 5s (matches the stock recipe)."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        # debug_vis off (2026-07-13, direct user directive): the goal-pose
        # command drew one RGB axis triad per env — a forest of gizmos at
        # 4096 envs. Visualization-only; no physics/reward effect.
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        # Task 1 of docs/superpowers/plans/2026-07-16-unified-multi-die-
        # specialist-distillation.md: shape-class one-hot (4 dims) +
        # geometry-descriptor (1 dim, Wadell sphericity) - both per-env-cfg
        # broadcast constants driven by self.die_shape_class below (see
        # mdp.object_shape_class_onehot/object_geometry_descriptor and
        # tasks/franka/shape_observations.py for the pure math). Grows the
        # observation space by exactly 4 + 1 = 5 dims.
        shape_class = ObsTerm(func=mdp.object_shape_class_onehot)
        geometry_descriptor = ObsTerm(func=mdp.object_geometry_descriptor)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TargetSelectionObservationsCfg(ObservationsCfg):
    """`ObservationsCfg` + one new, additive term,
    `distractor_distance_summary` (Task 2, docs/superpowers/plans/2026-07-19-
    target-selection-clutter-implementation.md; spec:
    docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md) -
    DexSinGrasp's own `d_t^S` mechanism (arXiv:2504.04516 §III-A Eq. 1), a
    fixed-size K=2 hard-zero-padded target-to-distractor distance vector
    (see mdp.distractor_distance_summary and
    tasks/franka/distractor_observations.py for the pure math). Grows the
    observation space by exactly 2 dims (41 -> 43) for whichever env cfg
    uses this class.

    Deliberately a NEW class, NOT a term added directly to the shared base
    `ObservationsCfg.PolicyCfg` above: unlike `shape_class`/
    `geometry_descriptor` (safe to broadcast unconditionally, since every
    env cfg already has a `die_shape_class` constant),
    `distractor_distance_summary` requires `env.scene["distractor_1"]`/
    `env.scene["distractor_2"]` to actually exist as scene entities - only
    dice_lift_joint_env_cfg.py's new target-selection curriculum-stage env
    cfgs (FrankaDieLiftTargetSelectionSceneCfg, Task 1) have those. Adding
    this term to the shared base class would `KeyError` on every other,
    single-die env cfg already in this codebase (`ik-cube`, `joint-die`,
    `joint-die-d12-d20-mixed`, etc.) the moment its observation manager ran.
    This class is genuinely additive to the *new* clutter env cfgs only -
    every other env cfg in this repo keeps using the original
    `ObservationsCfg` unchanged."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        distractor_distance_summary = ObsTerm(func=mdp.distractor_distance_summary)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: put the whole scene back to default, then jitter the cube's start pose."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Stock reward shape: dense reach + binary lift (NOT antipodal/contact-force-gated - a
    deliberate, known difference from this project's own AR4-era grasp-verification gate, see
    module docstring) + lift-gated goal-tracking (coarse + fine) + small action-rate/joint-vel
    penalties."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


# Shared params for both new exploration-bonus reward terms (Task 2,
# docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-discovery-
# implementation.md; spec: docs/superpowers/specs/2026-07-19-exploration-
# bonus-grasp-discovery-design.md). Defined ONCE, referenced by both
# RewTerm.params below - never copy-pasted - so term 1's own F_t and term
# 2's own redundant recomputation of F_t (needed to compute Correction_t,
# see mdp.GripperClosureAttemptBonusCorrection) can never silently diverge
# on these three values. Implementer-set starting values (spec's own
# "Global constraints": NOT tuned by this experiment - a Tier 2 hillclimb
# candidate once/if this mechanism is validated).
_EXPLORATION_BONUS_PARAMS = {"w_attempt": 1.0, "k": 1.0, "std_gate": 0.05}
# MUST equal FrankaLiftPPORunnerCfg.algorithm.gamma exactly
# (tasks/franka/agents/rsl_rl_ppo_cfg.py:50, `gamma=0.98`) - GRM's Theorem 1
# is a telescoping-sum identity over the SAME discount factor the agent's
# own returns/advantages are computed with; a mismatched gamma here breaks
# the policy-invariance guarantee this mechanism exists to provide, silently
# (plan's own "Global Constraints").
_PPO_GAMMA = 0.98


@configclass
class ExplorationBonusRewardsCfg(RewardsCfg):
    """`RewardsCfg` + two new, additive reward terms implementing GRM D=1
    (Forbes et al., arXiv:2410.12197/2505.12611), a formally policy-invariant
    action-dependent exploration bonus for gripper-closure attempts near the
    object (Task 2 of the implementation plan above; H1 of the design spec).
    Deliberately a NEW class, NOT terms added directly to the shared base
    `RewardsCfg` - mirrors `TargetSelectionObservationsCfg`'s own established
    "new subclass, base untouched" precedent immediately above, so the
    concurrently-running d8/d10 demo-warmstart plan's own use of the plain,
    unmodified `RewardsCfg`/`FrankaDieLiftJointD8BigEnvCfg` stays completely
    unaffected (plan's own "Global Constraints").

    `gripper_closure_attempt_bonus` (term 1, mdp.py) is the raw, stateless
    F_t. `gripper_closure_attempt_bonus_correction` (term 2, mdp.py's
    `GripperClosureAttemptBonusCorrection`, the ONE new stateful mechanism
    this plan introduces) is NOT the spec's own F'_t verbatim - it is
    Correction_t := F'_t - F_t, so that Term1 + Term2 == F'_t once both are
    summed by the RewardManager (see exploration_bonus_reward.py's own module
    docstring for the full double-counting-avoidance derivation). Both
    weights are 1.0 (the correction term's own "weight" is a manager-level
    multiplier on Correction_t, not a second application of w_attempt -
    w_attempt already lives inside _EXPLORATION_BONUS_PARAMS, consumed by
    both terms identically)."""

    gripper_closure_attempt_bonus = RewTerm(func=mdp.gripper_closure_attempt_bonus, params=_EXPLORATION_BONUS_PARAMS, weight=1.0)

    gripper_closure_attempt_bonus_correction = RewTerm(
        func=mdp.GripperClosureAttemptBonusCorrection,
        params={**_EXPLORATION_BONUS_PARAMS, "gamma": _PPO_GAMMA},
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """time_out + object_dropping only - no success-based early termination (matches the stock
    recipe; 100% time-out is the expected steady state here, not evidence of failure)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Ramps the action-rate/joint-vel penalty weights up after 10k steps, matching the stock recipe."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class FrankaLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Franka Panda cube-lift task: relative-IK arm action, stock reward shape/episode length."""

    scene: FrankaLiftSceneCfg = FrankaLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Per-env-cfg constant (Task 1, docs/superpowers/plans/2026-07-16-
    # unified-multi-die-specialist-distillation.md): which of {d8, d10, d12,
    # d20} this training run's object is, read by
    # mdp.object_shape_class_onehot/object_geometry_descriptor and broadcast
    # identically to every parallel env - NOT a per-environment-varying
    # value (see tasks/franka/shape_observations.py's module docstring for
    # the scope rationale). This base class's own object is the stock
    # DexCube, not a die at all, so this default is an arbitrary but
    # historically-dominant fallback (d20 was this project's first/primary
    # die shape); die-specialist subclasses in dice_lift_joint_env_cfg.py
    # override this in their own __post_init__.
    die_shape_class: str = "d20"

    # Per-env-cfg constant (Task 5, BACKLOG.md's 2026-07-19 controller
    # decision "(b) single mixed-population env"): for a MIXED-shape env cfg
    # only (FrankaDieLiftJointD12D20MixedEnvCfg), the ordered tuple of shape
    # classes its MultiAssetSpawnerCfg(assets_cfg=[...], random_choice=False)
    # was built with - e.g. ("d12", "d20"). When set (non-None),
    # mdp.object_shape_class_onehot/object_geometry_descriptor compute EACH
    # env's own shape class as `env_index % len(die_shape_classes_per_env)`
    # (mirroring the live spawner's own deterministic round-robin formula,
    # see tasks/franka/shape_observations.py's module docstring) instead of
    # broadcasting the single `die_shape_class` constant above to every row.
    # Default None for every other env cfg in this file - this is additive,
    # the single-shape broadcast path (`die_shape_class`) is unaffected.
    die_shape_classes_per_env: tuple[str, ...] | None = None

    # Per-env-cfg constant (Task 1, docs/superpowers/plans/2026-07-19-
    # target-selection-clutter-implementation.md): how many of the two
    # `distractor_1`/`distractor_2` scene slots (added by
    # dice_lift_joint_env_cfg.py's FrankaDieLiftTargetSelectionSceneCfg) are
    # REAL/active for this env cfg's own curriculum stage, vs. parked
    # (off-workspace, degenerate zero-width reset range - see
    # TargetSelectionEventCfg). Read by mdp.distractor_distance_summary
    # (Task 2) to decide which of its two K=2 output columns are real
    # distances vs. hard-zeroed - NOT a per-environment-varying value, same
    # per-env-cfg-constant convention as die_shape_class/
    # die_shape_classes_per_env above. Default 0 so every existing env cfg
    # in this repo (none of which have distractor_1/distractor_2 scene
    # entities at all) is completely unaffected; only the 3 new
    # target-selection curriculum-stage env cfgs (SO=0, D1=1, D2=2) in
    # dice_lift_joint_env_cfg.py override this.
    active_distractor_count: int = 0

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class FrankaLiftEnvCfg_PLAY(FrankaLiftEnvCfg):
    """Smaller, non-randomized-observation variant for interactive play/eval."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

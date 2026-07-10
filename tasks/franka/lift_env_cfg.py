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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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
        debug_vis=True,
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

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

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

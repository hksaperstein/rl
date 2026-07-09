# tasks/ar4/pickplace_graspgoal_env_cfg.py
"""Grasp/lift/goal variant of the AR4 pick-and-place task (Experiment
26): reintroduces the gripper after Experiment 25 removed it. Composes
Experiment 21's proximity-gated gripper and Experiment 17's antipodal
grasp gate, with a 30s episode and a 4-stage extension of Experiment
25's validated monotonic reward mechanism.

Originally also planned to reuse Experiment 22's mirroring mechanism
(corrected for its own identified reactive-lag bug); final whole-branch
review found that fix is an unconditional no-op under this action space
(BinaryJointAction assigns both jaws the same commanded value already -
see ActionsCfg's own docstring below for the full trace) and retired it
rather than attempt a fourth jaw-synchronization-specific fix. See
docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md.

Additive/parallel to pickplace_touchgoal_env_cfg.py: deliberately does
NOT modify that file, env_cfg.py, objects_cfg.py, or mdp.py's Experiment
25 functions.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has
been created.
"""

import math

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .actions import ProximityGatedBinaryJointPositionActionCfg
from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import (
    ARM_JOINT_NAMES,
    AR4_MK5_CFG,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_POS,
)

# Same fixed cube spawn as Experiment 25; goal is now where the CUBE must
# end up (carried there), not an end-effector waypoint - same offset
# value, different physical meaning.
GOAL_OFFSET = (-0.40, 0.0, 0.144)

REACH_DIST_NORM = 0.3
LIFT_MINIMAL_HEIGHT = 0.03
# NOTE: `lifted` latches at LIFT_MINIMAL_HEIGHT (0.03), not this value, so
# the 0.50-0.75 reward segment this normalizes is only ever traversed to
# ~0.575 in practice before the formula jumps to the goal segment (whose
# 3D distance-to-goal term subsumes the remaining lift height anyway) -
# this constant does not meaningfully shape mid-lift behavior, flagged by
# final whole-branch review, not a bug.
LIFT_TARGET_HEIGHT = 0.10
GOAL_TOLERANCE = 0.02
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
PROXIMITY_THRESHOLD = 0.05
# Derived, not hardcoded - matches Experiment 25's final-review lesson on
# geometry constants silently drifting from what they measure.
CUBE_TO_GOAL_DIST = math.sqrt(GOAL_OFFSET[0] ** 2 + GOAL_OFFSET[1] ** 2 + GOAL_OFFSET[2] ** 2)


@configclass
class ActionsCfg:
    """Arm (Experiment 25's proven scale=0.5) + gripper (proximity-gated
    only - no jaw-mirroring, see note below).

    Final whole-branch review of this experiment found MirroredGripperActionCfg
    (jaw2 tracks jaw1's target) is an unconditional no-op here: BinaryJointAction
    already assigns both jaws the IDENTICAL commanded value (open_command_expr/
    close_command_expr use the same constant for both joint names), so there is
    nothing for jaw2 to diverge from at the command level in the first place -
    the actual jaw asymmetry happens at the physics/actuator level under contact,
    which no command-level mirroring can address. This is the third jaw-
    synchronization-specific fix attempt to prove ineffective (Experiment 19's
    PhysxMimicJointAPI made things worse; Experiment 22's original mirror-actual
    had a reactive-lag bug; this pass's lag fix, mirror-commanded, is a true
    no-op) - retired as a lever per this project's own "3 failed fixes means
    question the architecture, not patch again" discipline, not attempted a
    fourth time. Uses plain ProximityGatedBinaryJointPositionActionCfg
    (Experiment 21's validated fix, unaffected by this finding) instead."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = ProximityGatedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=PROXIMITY_THRESHOLD,
    )


@configclass
class Ar4PickPlaceGraspGoalSceneCfg(InteractiveSceneCfg):
    """AR4 arm + gripper + a single fixed-position cube. Re-adds the
    gripper-jaw contact sensors Experiment 25's touch-goal lineage
    dropped (required for antipodal_grasp_bonus)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.006)),
        spawn=CUBE_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            )
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
        ],
    )
    gripper_jaw1_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class ObservationsCfg:
    """Arm + gripper joint state (unrestricted joint_names - gripper is
    actuated again), cube position, goal position, grasp/lift latch
    state, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        goal_position = ObsTerm(
            func=ar4_mdp.cube_goal_position_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot")}
        )
        grasp_state = ObsTerm(
            func=ar4_mdp.grasp_state_observation,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
                "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
                "force_threshold": FORCE_THRESHOLD,
                "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
                "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: whole scene back to default, then snapshot the cube
    goal position, then zero the milestone/latch buffers."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_cube_goal_position = EventTerm(
        func=ar4_mdp.set_cube_goal_position,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
    )

    reset_grasp_goal_milestone = EventTerm(func=ar4_mdp.reset_grasp_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Success (genuine grasp+lift occurred, then the cube reached the
    goal) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.cube_reached_goal_after_lift,
        params={
            "threshold": GOAL_TOLERANCE,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
        },
    )


@configclass
class RewardsCfg:
    """Four-stage gated running-max milestone bonus: reach, grasp
    (antipodal gate), lift, goal - no ungated additive sum (Experiment
    16's wedging exploit), no separately-weighted independent terms
    (Experiment 17/18's discoverability gap)."""

    grasp_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.grasp_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_dist_norm": REACH_DIST_NORM,
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "lift_target_height": LIFT_TARGET_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
            "cube_to_goal_dist": CUBE_TO_GOAL_DIST,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceGraspGoalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 grasp/lift/goal task (Experiment 26): reach, grasp, lift, and
    carry the cube to a fixed goal point. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run."""

    scene: Ar4PickPlaceGraspGoalSceneCfg = Ar4PickPlaceGraspGoalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

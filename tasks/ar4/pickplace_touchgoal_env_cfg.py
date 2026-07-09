"""Touch-goal variant of the AR4 pick-and-place task (Experiment 25): the
arm's only job is to touch the top of a fixed cube, then reach a fixed
goal point. No grasp, no lift, no gripper action - see
docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md.

Additive/parallel to pickplace_mirror_env_cfg.py: deliberately does NOT
touch that file, env_cfg.py, or objects_cfg.py.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import ARM_JOINT_NAMES, AR4_MK5_CFG

# NOTE (plan self-review, 2026-07-09): JointPositionActionCfg comes from
# isaaclab.envs.mdp (aliased isaaclab_mdp above), NOT from the
# isaaclab_tasks lift-task mdp module (aliased plain `mdp` above, used
# for everything else in this file: joint_pos_rel, object_position_in_
# robot_root_frame, last_action, reset_scene_to_default, time_out,
# action_rate_l2, joint_vel_l2) - exactly matching
# pickplace_mirror_env_cfg.py's own three-alias import convention
# (isaaclab_mdp / mdp / ar4_mdp). Using the wrong one of the two `mdp`
# aliases for ActionsCfg was caught here before implementation, not
# after a failed smoke test.

# Fixed goal point, expressed as an offset from the cube's own (also
# fixed) spawn position so per-env world placement stays correct without
# a separate randomization buffer: cube world (0.20, 0.28, 0.006) + this
# offset = world (-0.20, 0.28, 0.15) - mirrored across the cube in X,
# elevated clear of the ground plane.
GOAL_OFFSET = (-0.40, 0.0, 0.144)

CUBE_HALF_SIZE = 0.006  # meters (12mm cube, tasks/ar4/objects_cfg.py)
TOUCH_STD = 0.05
TOUCH_TOLERANCE = 0.02
TOUCH_TO_GOAL_DIST = math.sqrt(GOAL_OFFSET[0] ** 2 + GOAL_OFFSET[1] ** 2 + (GOAL_OFFSET[2] - CUBE_HALF_SIZE) ** 2)
GOAL_TOLERANCE = 0.02


@configclass
class ActionsCfg:
    """Arm-only action space - no gripper action term at all (Experiment
    25 removes grasp/lift entirely, so the gripper serves no purpose
    here; the gripper joints stay physically present but unactuated)."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)


@configclass
class Ar4PickPlaceTouchGoalSceneCfg(InteractiveSceneCfg):
    """AR4 arm + a single fixed-position cube, no rect_prism/wedge/sphere,
    no gripper contact sensors (no grasp signal needed for this task)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.006)),
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


@configclass
class ObservationsCfg:
    """Observation specifications: arm joint state (gripper joints
    excluded - they're unactuated in this task), cube position, fixed
    goal position, last action (6-dim, arm-only)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
        )
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        goal_position = ObsTerm(
            func=ar4_mdp.touch_goal_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: whole scene back to default, then zero the
    touch-goal milestone buffer and touched-cube latch."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_touch_goal_position = EventTerm(
        func=ar4_mdp.set_touch_goal_position,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
    )

    reset_touch_goal_milestone = EventTerm(func=ar4_mdp.reset_touch_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Success (end-effector touched the cube, then reached the goal)
    ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=ar4_mdp.touch_then_goal_reached,
        params={
            "threshold": GOAL_TOLERANCE,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_half_size": CUBE_HALF_SIZE,
            "touch_tolerance": TOUCH_TOLERANCE,
        },
    )


@configclass
class RewardsCfg:
    """Two-stage gated running-max milestone bonus: touch the cube top,
    then reach the goal - no grasp/lift terms at all."""

    touch_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.touch_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_half_size": CUBE_HALF_SIZE,
            "touch_std": TOUCH_STD,
            "touch_tolerance": TOUCH_TOLERANCE,
            "touch_to_goal_dist": TOUCH_TO_GOAL_DIST,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
    )


@configclass
class Ar4PickPlaceTouchGoalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 touch-then-goal task (Experiment 25): touch the cube's top,
    then reach a fixed goal point. No grasp, no lift, arm-only action
    space. num_envs=4096 default, matching the mirror task's training
    scale - scripts/train.py's --num_envs flag overrides this per-run."""

    scene: Ar4PickPlaceTouchGoalSceneCfg = Ar4PickPlaceTouchGoalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        # 5.0s (copied from pickplace_mirror_env_cfg.py's single-object lift
        # task) gave the arm too little time for a two-stage sequential task:
        # the first training run (5.0s) showed episodes running to the full
        # 250-step timeout almost every time, with goal_reached declining
        # rather than converging. Isaac Lab's own reference manipulation
        # tasks scale episode length with task structure, not object count -
        # Reach (single target): 12.0s; Lift (reach+grasp+lift, one object):
        # 5.0s; Cabinet (reach+grasp+open): 8.0s; Stack (reach+grasp+lift+
        # move+place, sequential multi-stage - the closest structural analog
        # to touch-then-goal): 30.0s, 6x Lift's episode length. 20.0s sits
        # inside that established range for a two-stage sequential task.
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

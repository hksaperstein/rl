# tasks/ar4/pickplace_provenrecipe_env_cfg.py
"""From-scratch replication of two proven, independently-published
Isaac-ecosystem manipulation reward recipes (Experiment 16): Isaac Lab's
own Franka Cube Lift task
(isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py,
mdp/rewards.py, config/franka/joint_pos_env_cfg.py) and IsaacGymEnvs'
FrankaCubeStack task (isaacgymenvs/tasks/franka_cube_stack.py) - both read
directly from source. Unlike every prior experiment this session, this
env cfg has NO standalone grasp-quality reward (no antipodal/contact
sensing), a PLAIN BINARY per-step lift reward (not a milestone/running-max
bonus), goal-tracking reward MULTIPLICATIVELY GATED on the lift condition,
and plain joint-space action (not task-space/IK) - see
docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md
for the full hypothesis and cited research.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_taskspace_env_cfg.py, pickplace_residual_env_cfg.py,
pickplace_reachskip_env_cfg.py, or pickplace_baseproximity_env_cfg.py.
Reuses Ar4PickPlaceMirrorSceneCfg and Ar4PickPlacePPORunnerCfg directly.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class ActionsCfg:
    """Plain joint-space action, matching both proven references exactly
    (neither uses task-space/IK-driven control). scale=0.5 matches
    pickplace_mirror_env_cfg.py's own ActionsCfg, which already cites the
    same Franka lift-task precedent for this value."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    """Identical structure to every prior experiment's ObservationsCfg -
    this axis was already well-aligned with the proven references, no
    change needed."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        target_object_position = ObsTerm(
            func=ar4_mdp.mirrored_target_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events, in registration order - deliberately simpler than
    every prior experiment's EventCfg: NO compute_path_waypoints, since
    this design has no waypoint/milestone system.
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the
       mirrored goal into env._target_pos_w."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_y_range": (0.10, 0.45),
            "goal_z_range": (0.0, 0.02),
        },
    )


@configclass
class TerminationsCfg:
    """Success (cube at the mirrored goal) ends the episode early;
    otherwise a fixed timeout - this repo's own established success
    definition, kept since the proven references' tabletop-drop
    termination doesn't apply to this repo's ground-level scene."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Six terms, matching the proven references' count and structure
    exactly. reaching_object/lifting_object are REUSED DIRECTLY from
    isaaclab_tasks.manager_based.manipulation.lift.mdp - not
    reimplemented, since both are already fully generic (parameterized by
    object_cfg/ee_frame_cfg, no Franka-specific assumptions).
    object_goal_tracking/object_goal_tracking_fine_grained use the new
    mirrored_goal_distance_gated (Task 1) - the SAME gating formula as
    the reference's object_goal_distance, adapted only to this repo's
    goal-storage mechanism. NO standalone grasp-quality reward (no
    antipodal/contact-force term) - grasp is purely instrumental, matching
    both references. See
    docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    object_goal_tracking = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_gated,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_gated,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """New to this repo: Isaac Lab's curriculum manager, replicating the
    proven reference's regularization-weight curriculum exactly
    (mdp.modify_reward_weight is a framework-provided function, already
    available via this file's `from isaaclab_tasks.manager_based.manipulation.lift
    import mdp` - that module re-exports isaaclab.envs.mdp's contents)."""

    action_rate_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class Ar4PickPlaceProvenRecipeEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 proven-recipe task (Experiment 16): from-scratch replication of
    Isaac Lab's Franka Cube Lift and IsaacGymEnvs' FrankaCubeStack reward
    structure and action space on the AR4+cube scene. num_envs=4096
    default - scripts/train.py's --num_envs flag overrides this per-run
    same as every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

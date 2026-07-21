# tasks/ar4/pickplace_pregrasp_env_cfg.py
"""Dense pre-grasp-readiness shaping variant (Experiment 18): identical to
Experiment 17's Ar4PickPlaceGraspGatedEnvCfg
(tasks/ar4/pickplace_graspgated_env_cfg.py) in every respect except one
new reward term, pregrasp_readiness (proximity x gripper-closedness).
Experiment 17's own instrumented investigation (Task 6,
.superpowers/sdd/task-6-report.md) found the trained policy exploring
"get close to the object" and "close the gripper" independently but
never combining them - the binary antipodal grasp gate
(lifting_object_grasp_gated/mirrored_goal_distance_grasp_gated, both
REUSED UNCHANGED here) never fired even once across 1500 iterations
because the compound behavior it requires was never discovered. This
experiment adds a dense stepping-stone signal toward that compound
behavior without touching the gate itself. See
docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_taskspace_env_cfg.py, pickplace_residual_env_cfg.py,
pickplace_reachskip_env_cfg.py, pickplace_baseproximity_env_cfg.py,
pickplace_provenrecipe_env_cfg.py, or pickplace_graspgated_env_cfg.py.
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
from .robot_cfg import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
    GRIPPER_OPEN_POS,
)


@configclass
class ActionsCfg:
    """Plain joint-space action, identical to Experiment 17's
    ActionsCfg."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSED_COMMAND_EXPR,
    )


@configclass
class ObservationsCfg:
    """Identical to Experiment 17's ObservationsCfg."""

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
    """Identical to Experiment 17's EventCfg."""

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
    """Identical to Experiment 17's TerminationsCfg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Seven terms: the six from Experiment 17, unchanged (including the
    binary grasp gate on lifting_object/object_goal_tracking - reused
    completely unmodified), plus one new dense term, pregrasp_readiness
    (Task 1). See
    docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    pregrasp_readiness = RewTerm(
        func=ar4_mdp.pregrasp_readiness_bonus,
        weight=2.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_joint_names": GRIPPER_JOINT_NAMES,
            "open_pos": GRIPPER_OPEN_POS,
            "closed_pos": GRIPPER_CLOSED_POS,
        },
    )

    lifting_object = RewTerm(
        func=ar4_mdp.lifting_object_grasp_gated,
        weight=15.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "minimal_height": 0.03,
        },
    )

    object_goal_tracking = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """Identical to Experiment 17's CurriculumCfg."""

    action_rate_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class Ar4PickPlacePregraspEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 pre-grasp-readiness task (Experiment 18): identical to
    Experiment 17 plus one new dense shaping term rewarding the
    combination of proximity and gripper closure. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

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

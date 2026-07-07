# tasks/ar4/pickplace_orientationbias_env_cfg.py
"""Soft orientation-alignment bias variant (Experiment 20, revised):
identical to Experiment 18's Ar4PickPlacePregraspEnvCfg
(tasks/ar4/pickplace_pregrasp_env_cfg.py) in every respect - same
plain joint-space action, same scene, same antipodal grasp gate, same
pregrasp_readiness term - plus one new dense reward term,
orientation_alignment_bonus, rewarding the policy for keeping the
gripper's approach axis close to vertical (top-down).

Experiment 20 originally tried a HARD action-space constraint (a custom
absolute-pose differential-IK action term locking orientation exactly);
independent instrumented verification found that mechanism structurally
unstable (real end-effector drift of 75-99 degrees off target within a
single episode, across three independently-tried fixes). This file
instead layers a SOFT reward bias onto the already-proven joint-space
action, testing the identical underlying hypothesis (reduce the
policy's orientation-discovery burden) without the IK-stability problem
class. See
docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md's
"Revision" section.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_pregrasp_env_cfg.py, pickplace_graspgated_env_cfg.py,
pickplace_verticallock_env_cfg.py, tasks/ar4/actions.py, or any other
existing env cfg. Reuses Ar4PickPlaceMirrorSceneCfg,
Ar4PickPlacePPORunnerCfg, and Experiment 18's ActionsCfg/
ObservationsCfg/EventCfg/TerminationsCfg/CurriculumCfg directly.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_pregrasp_env_cfg import (
    ActionsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from .robot_cfg import GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class RewardsCfg:
    """The seven terms from Experiment 18, unchanged (including the
    binary grasp gate on lifting_object/object_goal_tracking and the
    pregrasp_readiness dense shaping term - both reused completely
    unmodified), plus one new dense term, orientation_alignment_bonus.
    See
    docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md."""

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

    orientation_alignment = RewTerm(
        func=ar4_mdp.orientation_alignment_bonus,
        weight=2.0,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
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
class Ar4PickPlaceOrientationBiasEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 soft-orientation-bias task (Experiment 20, revised): Experiment
    18's exact action/observation/event/termination/curriculum
    configuration, plus one new dense reward term biasing the policy
    toward a vertical (top-down) approach without hard-constraining the
    action space. num_envs=4096 default - scripts/train.py's --num_envs
    flag overrides this per-run same as every other env cfg in this
    repo."""

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

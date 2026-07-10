# tasks/ar4/pickplace_hierarchical_env_cfg.py
"""Hierarchical-decomposition research prototype (not yet a numbered
production experiment - a feasibility check for a Senior research thread):
tests whether training a SEPARATE policy for grasp/lift/carry, starting
each episode from a REAL physically-simulated post-reach state harvested
from the already-trained Experiment 25 touch-goal reach policy (see
scripts/harvest_reach_handoff_states.py, tasks/ar4/mdp.py's
reset_arm_from_handoff_bank), makes genuine antipodal grasp discoverable
where Experiment 26's flat single-policy reach+grasp+lift+goal reward
never discovered it in 1500 iterations.

Differs from Experiment 14's reset_arm_to_pregrasp_pose in two ways that
matter for this hypothesis: (1) the reset state comes from an actual
rollout of a validated, already-converged closed-loop policy, never a
one-shot analytic IK solve (Experiment 14's own root-cause hypothesis for
its new "folds toward its own base" failure mode was exactly this kind of
IK elbow-up/down ambiguity landing on an awkward configuration); (2) the
reward this env cfg uses (grasp_lift_goal_milestone_bonus,
tasks/ar4/grasp_only_reward.py) has NO reach stage at all and is trained
completely separately from any reach-stage reward, rather than reusing
Experiment 14's unchanged flat reward under a new starting-state
distribution.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_graspgoal_env_cfg.py, pickplace_touchgoal_env_cfg.py, env_cfg.py,
objects_cfg.py, or mdp.py's existing functions (only new functions were
appended to mdp.py). Reuses Ar4PickPlaceGraspGoalSceneCfg and ActionsCfg
directly from pickplace_graspgoal_env_cfg.py (identical scene/action
space - only the reward and one new reset event differ).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import math

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_graspgoal_env_cfg import (
    ANTIPODAL_COS_THRESHOLD,
    ARM_GROUND_CONTACT_FORCE_THRESHOLD,
    CUBE_TO_GOAL_DIST,
    FORCE_THRESHOLD,
    GOAL_OFFSET,
    GOAL_TOLERANCE,
    LIFT_MINIMAL_HEIGHT,
    LIFT_TARGET_HEIGHT,
    SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD,
    SLOW_NEAR_CUBE_SPEED_CAP,
    ActionsCfg,
    Ar4PickPlaceGraspGoalSceneCfg,
)

# Path to the real reach-converged handoff-state bank, produced by
# scripts/harvest_reach_handoff_states.py. Must exist before this env cfg
# is used to train/step - reset_arm_from_handoff_bank loads it lazily on
# first reset.
HANDOFF_BANK_PATH = "logs/reach_handoff_states.pt"


@configclass
class ObservationsCfg:
    """Identical to pickplace_graspgoal_env_cfg.py's ObservationsCfg - same
    observation surface, since the action space and scene are unchanged."""

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
    """Reset events, in registration order:
    1. reset_all - whole scene back to default (arm home pose, gripper
       open, cube to its fixed spawn).
    2. set_cube_goal_position - reads the (fixed) cube position, sets the
       fixed goal the cube must be carried to.
    3. reset_arm_from_handoff_bank (NEW) - overwrites the arm's joint
       state with a randomly-sampled REAL post-reach state from the
       harvested bank (must run after reset_all, which would otherwise
       clobber it).
    4. reset_grasp_lift_goal_milestone - zero the running-max buffer and
       grasped/lifted latches."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_cube_goal_position = EventTerm(
        func=ar4_mdp.set_cube_goal_position,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
    )

    reset_arm_from_handoff_bank = EventTerm(
        func=ar4_mdp.reset_arm_from_handoff_bank,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]),
            "bank_path": HANDOFF_BANK_PATH,
        },
    )

    reset_grasp_lift_goal_milestone = EventTerm(func=ar4_mdp.reset_grasp_lift_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Identical success/safety terminations to pickplace_graspgoal_env_cfg.py."""

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

    arm_ground_contact = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={
            "threshold": ARM_GROUND_CONTACT_FORCE_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("arm_ground_contact"),
        },
    )


@configclass
class RewardsCfg:
    """Grasp-only 3-stage running-max milestone bonus (no reach stage at
    all - see tasks/ar4/grasp_only_reward.py), plus the same safety/
    settling terms Experiment 26 used (ground-contact penalty, slow-near-
    cube bonus, action-rate/joint-vel penalties)."""

    grasp_lift_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.grasp_lift_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "lift_target_height": LIFT_TARGET_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
            "cube_to_goal_dist": CUBE_TO_GOAL_DIST,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})

    arm_ground_contact_penalty = RewTerm(
        func=ar4_mdp.arm_ground_contact_penalty,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("arm_ground_contact"),
            "threshold": ARM_GROUND_CONTACT_FORCE_THRESHOLD,
        },
    )

    slow_near_cube_bonus = RewTerm(
        func=ar4_mdp.slow_near_cube_bonus,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "reach_dist_threshold": SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD,
            "speed_cap": SLOW_NEAR_CUBE_SPEED_CAP,
        },
    )


@configclass
class Ar4PickPlaceHierarchicalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 hierarchical-decomposition prototype: grasp/lift/carry-only
    sub-policy, trained on episodes that start from a real, physically-
    simulated post-reach state (see HANDOFF_BANK_PATH). num_envs=4096
    default - scripts/train.py's --num_envs flag overrides this per-run."""

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

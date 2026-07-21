"""Reward-shaping variant of the AR4 mirror-goal cube task (Experiment 15):
identical scene/action/observations/events/terminations as
pickplace_taskspace_env_cfg.py (Experiment 12's clean, non-regressed
baseline - NOT Experiment 13's residual action or Experiment 14's
reach-skip reset), with only RewardsCfg changed: the existing but
never-activated ground_penalty function wired in, a new
base_proximity_penalty function (cube x/y distance to the robot's own
base), and antipodal_grasp_bonus's weight raised (with a matched
stillness_penalty raise preserving the already-verified anti-freeze
reward-rate margin). See
docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_ik_guided_env_cfg.py, pickplace_taskspace_env_cfg.py,
pickplace_residual_env_cfg.py, or pickplace_reachskip_env_cfg.py. Reuses
Ar4PickPlaceMirrorSceneCfg and Ar4PickPlaceTaskspacePPORunnerCfg directly -
only RewardsCfg differs from pickplace_taskspace_env_cfg.py.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
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
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg
from .robot_cfg import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
    GRIPPER_OPEN_POS,
)

# Same values as pickplace_taskspace_env_cfg.py's EventCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ActionsCfg:
    """Identical to pickplace_taskspace_env_cfg.py's ActionsCfg - the
    plain (non-residual) DifferentialInverseKinematicsActionCfg."""

    arm_action = isaaclab_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=isaaclab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
    )
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSED_COMMAND_EXPR,
    )


@configclass
class ObservationsCfg:
    """Identical to pickplace_taskspace_env_cfg.py's ObservationsCfg."""

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
    """Identical to pickplace_taskspace_env_cfg.py's EventCfg (no reset
    event changes in this experiment - reward function only):
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the cube's position and the
       goal, computes the 5-waypoint path, and resets path-progress state
       (including env._ik_milestone_max, reused by path_proximity_bonus)."""

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

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    """Success (cube at the mirrored goal) ends the episode early;
    otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Experiment 15: adds ground_penalty (existing function, never
    previously wired into any RewardsCfg) and base_proximity_penalty (new
    function, Task 1) on top of Experiment 12's exact RewardsCfg, and
    raises antipodal_grasp_bonus's weight 3.0 -> 4.0 with a matched
    stillness_penalty raise 5.0 -> 6.0 (preserves the exact -2.0/step net
    margin Experiment 12 verified for the grasped-and-stagnant state - see
    the design spec's section 3 for the full arithmetic).
    path_proximity_bonus/gripper_schedule_bonus/action_rate/joint_vel are
    unchanged from pickplace_taskspace_env_cfg.py. See
    docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md."""

    path_proximity_bonus = RewTerm(
        func=ar4_mdp.path_proximity_bonus,
        weight=25.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "proximity_std": 0.1,
            "advance_tolerance": 0.03,
        },
    )

    gripper_schedule_bonus = RewTerm(
        func=ar4_mdp.gripper_schedule_bonus,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_joint_names": GRIPPER_JOINT_NAMES,
            "open_pos": GRIPPER_OPEN_POS,
            "closed_pos": GRIPPER_CLOSED_POS,
        },
    )

    # weight raised 3.0 -> 4.0 (Experiment 15, direct user request: "higher
    # reward for the cube being in the grasp position"). stillness_penalty
    # below is raised in matched proportion to preserve the exact -2.0/step
    # net margin Experiment 12 verified for the grasped-and-stagnant state.
    antipodal_grasp_bonus = RewTerm(
        func=ar4_mdp.antipodal_grasp_bonus,
        weight=4.0,
        params={
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    # weight raised 5.0 -> 6.0 (Experiment 15): 4.0 - 6.0 = -2.0/step net,
    # identical margin to Experiment 12's 3.0 - 5.0 = -2.0/step. See
    # docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md.
    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=6.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )

    # New (Experiment 15): existing function, never previously wired into
    # any RewardsCfg. Direct user request: "negative reward for contacting
    # the ground." weight=0.1 kept small since this fires almost every
    # step until any lift happens (unlike the running-max milestone
    # bonuses) - see the design spec's section 1 for the per-episode
    # magnitude reasoning.
    ground_penalty = RewTerm(
        func=ar4_mdp.ground_penalty,
        weight=0.1,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ground_height_threshold": 0.015,
        },
    )

    # New (Experiment 15): new function (Task 1). Direct user request:
    # "negative reward for the cube contacting the base of the robot" -
    # explicitly a new function, distinct from ground_penalty (x/y
    # proximity to the robot's own base, not z-height). weight=0.1, same
    # magnitude reasoning as ground_penalty.
    base_proximity_penalty = RewTerm(
        func=ar4_mdp.base_proximity_penalty,
        weight=0.1,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "base_xy_threshold": 0.08,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceBaseProximityEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 reward-shaping task (Experiment 15): same scene/action/events as
    Experiment 12's clean baseline, with ground_penalty and
    base_proximity_penalty newly wired into the reward, and
    antipodal_grasp_bonus/stillness_penalty weights raised in matched
    proportion. num_envs=4096 default (a real training-scale run) -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

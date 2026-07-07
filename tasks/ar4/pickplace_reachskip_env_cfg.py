# tasks/ar4/pickplace_reachskip_env_cfg.py
"""Reach-skip curriculum variant of the AR4 mirror-goal cube task
(Experiment 14): identical scene/action/reward as
pickplace_taskspace_env_cfg.py (Experiment 12's clean, non-regressed
baseline - NOT Experiment 13's unresolved residual-action regression),
but the arm starts each episode already at the pregrasp pose for that
episode's randomized cube position (via a one-shot IK reset event)
instead of a fixed home pose. Removes the reach sub-problem - already
reliably solved across Experiments 11-13 - so the full step budget goes
toward the sub-problem those experiments never solved:
grasp->lift->carry->place. See
docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_ik_guided_env_cfg.py, pickplace_taskspace_env_cfg.py, or
pickplace_residual_env_cfg.py. Reuses Ar4PickPlaceMirrorSceneCfg and
Ar4PickPlaceTaskspacePPORunnerCfg directly - only the EventCfg (new
reset_arm_to_pregrasp_pose step) differs from pickplace_taskspace_env_cfg.py.

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
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

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
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
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
    """Reset events, in registration order - ONE new step relative to
    pickplace_taskspace_env_cfg.py's EventCfg:
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. reset_arm_to_pregrasp_pose (NEW) - one-shot IK teleport of the arm
       to the pregrasp pose for THIS episode's now-randomized cube
       position. Must run after reset_cube_position (needs the cube's
       real position) and before compute_path_waypoints.
    4. randomize_goal - reads the cube's position, sets the mirrored goal.
    5. compute_path_waypoints - unchanged, still computes the full
       5-waypoint path; the arm now starts at/near waypoint 0."""

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

    reset_arm_to_pregrasp_pose = EventTerm(
        func=ar4_mdp.reset_arm_to_pregrasp_pose,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "pregrasp_hover": _PREGRASP_HOVER,
            "gripper_tool_offset": _EE_OFFSET,
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
    """Identical weights to pickplace_taskspace_env_cfg.py's RewardsCfg
    (post-Experiment-12: stillness_penalty weight 5.0) - this experiment
    isolates the starting-state variable alone, reward function unchanged."""

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

    antipodal_grasp_bonus = RewTerm(
        func=ar4_mdp.antipodal_grasp_bonus,
        weight=3.0,
        params={
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceReachskipEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 reach-skip curriculum task: same scene/action/reward as
    Experiment 12's clean baseline, but the arm starts each episode
    already at the pregrasp pose for that episode's cube position,
    instead of a fixed home pose. num_envs=4096 default (a real
    training-scale run) - scripts/train.py's --num_envs flag overrides
    this per-run same as every other env cfg in this repo."""

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

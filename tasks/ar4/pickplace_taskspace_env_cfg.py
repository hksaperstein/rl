# tasks/ar4/pickplace_taskspace_env_cfg.py
"""Task-space IK-driven action variant of the AR4 mirror-goal cube task
(Experiment 11): identical scene/spawn-randomization/mirrored-goal/reward
fixes as pickplace_ik_guided_env_cfg.py, but the arm's action space is
replaced with Isaac Lab's built-in DifferentialInverseKinematicsActionCfg
(a Cartesian end-effector delta each step, converted to joint targets by
a live differential-IK controller inside the control loop) instead of
JointPositionActionCfg (direct joint-angle deltas). See
docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
or pickplace_ik_guided_env_cfg.py. Reuses Ar4PickPlaceMirrorSceneCfg
directly (same cube scene, same contact sensors, same ee_frame) - only the
action space and the path-tracking reward's internals differ from
pickplace_ik_guided_env_cfg.py.

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
from .agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_ik_guided_env_cfg.py's EventCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ActionsCfg:
    """Task-space action specifications: the arm is controlled via
    incremental Cartesian position commands converted to joint targets by
    a live differential-IK solver (Isaac Lab's built-in
    DifferentialInverseKinematicsActionCfg), rather than direct
    joint-position targets (contrast pickplace_mirror_env_cfg.py's
    ActionsCfg, used by every prior experiment this session).

    - command_type="position": 3D Cartesian only, not full 6-DOF pose -
      orientation isn't critical for this task's fixed top-down approach
      geometry, keeping the action space small like every prior experiment.
    - controller.use_relative_mode=True: DifferentialInverseKinematicsAction
      recomputes the current end-effector pose every step
      (process_actions -> self._compute_frame_pose()) and passes it to
      the controller's set_command() alongside the scaled raw action; with
      use_relative_mode=True the controller adds the scaled action to that
      CURRENT pose each step, i.e. the policy outputs incremental Cartesian
      deltas, not absolute positions - confirmed via
      isaaclab/envs/mdp/actions/task_space_actions.py's process_actions
      and isaaclab/controllers/differential_ik.py's set_command. There is
      only one use_relative_mode flag in this action term and it lives on
      the controller cfg (DifferentialInverseKinematicsActionCfg itself
      has no separate relative-mode field).
    - body_name="link_6", body_offset=OffsetCfg(pos=_EE_OFFSET): reuses the
      already-measured-and-verified 0.036m gripper-pinch-point offset (the
      same constant used throughout this session's ee_frame
      FrameTransformerCfg), so the controlled point is the actual gripper
      tip, not link_6 itself.
    - scale=0.05: 5cm maximum Cartesian step per unit of policy output,
      reasoned from the workspace's ~0.3-0.5m scale.
    - ik_method="dls": matches the damped-least-squares choice already
      validated in Experiments 8-10's reward-side IK usage
      (ik_guided_path_bonus).

    DifferentialInverseKinematicsAction (isaaclab/envs/mdp/actions/
    task_space_actions.py) already handles the fixed-base Jacobian
    indexing internally (self._asset.is_fixed_base branch) - the exact
    same logic ik_guided_path_bonus had to hand-implement for its
    reward-only IK usage. No new indexing code needed here.
    """

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
    """Observation specifications for the MDP - identical to
    pickplace_ik_guided_env_cfg.py's ObservationsCfg (same scene, same
    goal mechanism); the action term's internals changed but these
    observation functions read joint/object state directly, not the
    action term, so they remain valid unmodified. last_action will now
    report the 3D Cartesian delta + 2D binary gripper command instead of
    the previous 6D joint-delta + 2D gripper command - a smaller
    observation vector, no code change needed since it's already generic."""

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
    """Reset events, in registration order - identical to
    pickplace_ik_guided_env_cfg.py's EventCfg:
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
    """path_proximity_bonus replaces ik_guided_path_bonus (drops the
    now-redundant IK-action-matching sub-signal, since IK is now part of
    the control loop itself - see path_proximity_bonus's docstring).
    antipodal_grasp_bonus (Experiment 10's physics-corrected -0.7071
    threshold), gripper_schedule_bonus, action_rate, and joint_vel carry
    over unchanged from pickplace_ik_guided_env_cfg.py. stillness_penalty's
    weight is raised from 2.0 to 5.0 (Experiment 12): Experiment 11 showed
    grasp-and-freeze nets +1.0/step (antipodal_grasp_bonus's +3.0 minus the
    old stillness_penalty's -2.0 once its patience window elapses) - this
    flips that to -2.0/step. See
    docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md."""

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

    # weight raised 2.0 -> 5.0 (Experiment 12): with antipodal_grasp_bonus's
    # weight at 3.0, the old weight=2.0 only closed the reward-rate gap to
    # +3.0 - 2.0 = +1.0/step net POSITIVE for holding a grasp without
    # progress once patience_steps elapses - it never flipped the sign.
    # 5.0 makes it 3.0 - 5.0 = -2.0/step net negative. See
    # docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md.
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
class Ar4PickPlaceTaskspaceEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 task-space IK-driven-action task: same scene/spawn/goal as the
    mirror and ik_guided tasks, but the arm's action space is Cartesian
    deltas converted to joint targets by a live differential-IK
    controller in the control loop, instead of direct joint-position
    deltas. num_envs=4096 default (a real training-scale run) -
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


@configclass
class Ar4PickPlaceTaskspacePPORunnerCfg(Ar4PickPlacePPORunnerCfg):
    """Identical to Ar4PickPlacePPORunnerCfg except clip_actions=5.0 (~3.4x
    the observed Mean action noise std of ~1.46), added specifically for the
    task-space IK-driven action term after a real, confirmed critic
    (Mean value_function loss) divergence starting at iteration 67/1500 of
    the first full training run under this action space - never seen in any
    joint-space experiment using the unmodified Ar4PickPlacePPORunnerCfg. An
    outlier raw action, previously harmless under JointPositionActionCfg
    (which just saturates at joint limits), can drive the new differential-
    IK action term's solve into a discontinuous joint-space jump; bounding
    the raw action before it reaches the IK controller is a standard,
    minimal defensive measure (RSL_RL's own built-in clip_actions mechanism,
    applied in RslRlVecEnvWrapper.step() before scale/IK). See
    docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md's
    "Controller correction" section for the full trace."""

    clip_actions: float = 5.0

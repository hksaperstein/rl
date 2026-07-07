# tasks/ar4/pickplace_ik_guided_env_cfg.py
"""Classical-IK-guided variant of the AR4 mirror-goal pick-and-place task:
same scene/spawn-randomization/mirrored-goal as pickplace_mirror_env_cfg.py
(cube-only, default 18mm size, spawn randomized across the full
workspace, goal on the opposite side of the robot), but the staged reward
is replaced by a classical-IK-guided path: 5 geometric Cartesian waypoints
(pre-grasp, grasp, lift, transit, place) plus a live, per-step comparison
between the policy's actual joint configuration and what
isaaclab.controllers.DifferentialIKController suggests toward the current
waypoint. See
docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md.

Additive/parallel to pickplace_mirror_env_cfg.py: does NOT modify that
file, env_cfg.py, objects_cfg.py, or pickplace_env_cfg.py. Reuses
Ar4PickPlaceMirrorSceneCfg and the mirrored-goal observation/termination
functions directly (unchanged) - only the scene's identity and the goal
mechanism are shared, not the reward.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

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
from .pickplace_mirror_env_cfg import ActionsCfg, Ar4PickPlaceMirrorSceneCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_mirror_env_cfg.py's EventCfg/RewardsCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP - identical to
    pickplace_mirror_env_cfg.py's ObservationsCfg (same scene, same goal
    mechanism), duplicated here rather than imported since this file's
    RewardsCfg/EventCfg differ enough that keeping all manager configs
    together in one file is clearer for this new task."""

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
    """Reset events, in registration order:
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the cube's position and the
       goal, computes the 5-waypoint path, and resets path-progress state."""

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
    """Classical-IK-guided path reward: replaces the old staged
    reach/grasp/lift/goal signal with waypoint-sequenced Cartesian
    proximity + live IK-action-matching (ik_guided_path_bonus), plus a
    gripper-open/closed timing bonus. contact_grasp_bonus and
    stillness_penalty are reused unchanged from the mirror-scene task as
    standalone additive terms (no longer folded inside a staged signal)."""

    ik_guided_path_bonus = RewTerm(
        func=ar4_mdp.ik_guided_path_bonus,
        weight=25.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "proximity_std": 0.1,
            "advance_tolerance": 0.03,
            "ik_joint_std": 0.5,
            "gripper_tool_offset": (0.0, 0.0, 0.036),
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

    # Replaces contact_grasp_bonus (magnitude-only, ungated) after two
    # independent research passes found real problems with it: (1) it
    # paid out every step regardless of lift progress, a ~9:1 reward-rate
    # advantage for freezing after grasp vs. stillness_penalty's net -2,
    # confirmed in real training data as a ~118:1 ratio; (2) it checked
    # force magnitude only, not direction, missing the classical
    # antipodal/force-closure condition. Weight reduced 20.0 -> 3.0 to
    # close the reward-rate gap directly, on top of the stricter
    # antipodal check. See
    # docs/superpowers/specs/research/2026-07-06-rl-manipulation-senior-b.md
    # and 2026-07-06-classical-manipulation-senior-a.md.
    #
    # antipodal_cos_threshold: -0.7071 (not -0.85) is physics-derived from
    # the scene's actual static_friction=dynamic_friction=1.0 (confirmed in
    # __post_init__'s sim.physics_material). The classical friction-cone
    # half-angle is arctan(mu); for mu=1.0 that's 45°, giving a correct
    # antipodal cosine threshold of cos(180° - 45°) = -0.7071. The previous
    # -0.85 assumed ~30° friction-cone half-angle, causing the antipodal
    # condition to fire ~1800x less often than magnitude-only checks (verified
    # empirically in Experiment 9).
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
        weight=2.0,
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
class Ar4PickPlaceIkGuidedEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 classical-IK-guided task: same scene/spawn/goal as the mirror
    task with cube (default 18mm size), but reach/grasp/lift/carry is shaped by a
    live classical-IK path-tracking reward instead of ad hoc end-state distances.
    num_envs=4096 default (a real training-scale run) -
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

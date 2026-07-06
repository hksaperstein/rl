# tasks/ar4/pickplace_ik_guided_env_cfg.py
"""Classical-IK-guided variant of the AR4 mirror-goal pick-and-place task:
same scene/spawn-randomization/mirrored-goal as pickplace_mirror_env_cfg.py
(sphere-only, shrunk to 12mm diameter, spawn randomized across the full
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
from .env_cfg import ActionsCfg
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg

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
        sphere_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("sphere")}
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
    2. reset_sphere_position - randomize the sphere across the full workspace.
    3. randomize_goal - reads the sphere's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the sphere's position and the
       goal, computes the 5-waypoint path, and resets path-progress state."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": (0.10, 0.45),
            "goal_z_range": (0.0, 0.02),
        },
    )

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    """Success (sphere at the mirrored goal) ends the episode early;
    otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    sphere_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("sphere")},
    )

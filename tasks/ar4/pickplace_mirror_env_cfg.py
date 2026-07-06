# tasks/ar4/pickplace_mirror_env_cfg.py
"""Mirror-goal variant of the AR4 pick-and-place task: only the sphere is
present in the scene (no cube/rect_prism/wedge), its spawn is randomized
across the full workspace, and the goal is always on the opposite side of
the robot from wherever it spawned. See
docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.

Additive/parallel to pickplace_env_cfg.py: deliberately does NOT touch
`Ar4SceneCfg` (env_cfg.py) or `objects_cfg.py`, since interactive_demo.py,
perception/tests, and other scripts depend on all four objects existing
there - same convention as pickplace_single_object_env_cfg.py. Also does
NOT reuse pickplace_env_cfg.py's CommandsCfg/RewardsCfg/ObservationsCfg/
TerminationsCfg - this task replaces the CommandManager-based goal with a
stateful per-env buffer (env._target_pos_w) that can express "goal is a
function of the sphere's own random spawn", which UniformPoseCommandCfg
cannot.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

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
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .env_cfg import ActionsCfg
from .objects_cfg import SPHERE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import AR4_MK5_CFG

# Env-local (robot-relative) workspace bounds this task randomizes the
# sphere's spawn and the goal's y-coordinate within. Defined independently
# of pickplace_env_cfg.py's WORKSPACE_BOUNDS (that constant is documented
# as being for the interactive demo/perception entry points specifically),
# even though the values currently match.
_WORKSPACE_X = (-0.30, 0.30)
_WORKSPACE_Y = (0.10, 0.45)
_GOAL_Z = (0.0, 0.02)


@configclass
class Ar4PickPlaceMirrorSceneCfg(InteractiveSceneCfg):
    """AR4 gripper + a single sphere (no cube/rect_prism/wedge), plus the
    same end-effector FrameTransformer and gripper-to-sphere ContactSensors
    as Ar4PickPlaceSceneCfg (pickplace_env_cfg.py) - copied, not imported,
    since the base scene class differs (no cube/rect_prism/wedge fields)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Recentered to the workspace midpoint (local x=0.0, y=0.275) so
    # reset_sphere_position's pose_range in EventCfg below can cover the
    # full _WORKSPACE_X/_WORKSPACE_Y range symmetrically - SPHERE_CFG
    # itself (objects_cfg.py) is unchanged; .replace() returns a new cfg.
    sphere: RigidObjectCfg = SPHERE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.275, 0.009))
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
    gripper_jaw1_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

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
    """Reset events, in registration order (Isaac Lab's EventManager runs
    same-mode terms in registration order - later terms may depend on
    earlier ones' output within the same reset):
    1. reset_all - whole scene back to default.
    2. reset_sphere_position - randomize the sphere across the full
       workspace (reuses the existing, proven reset_root_state_uniform,
       just with a wider pose_range than pickplace_env_cfg.py's ±2cm
       jitter).
    3. randomize_goal - reads the sphere's now-updated position, sets the
       mirrored goal into env._target_pos_w.
    4. reset_lift_milestone / reset_stillness_buffers - zero the new
       reward terms' stateful buffers, so a new episode starts with no
       carried-over progress or stale stillness reference."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": _WORKSPACE_X, "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": _WORKSPACE_Y,
            "goal_z_range": _GOAL_Z,
        },
    )

    reset_lift_milestone = EventTerm(func=ar4_mdp.reset_lift_milestone, mode="reset")

    reset_stillness_buffers = EventTerm(
        func=ar4_mdp.reset_stillness_buffers,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("sphere")},
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


@configclass
class RewardsCfg:
    """Corrected undiscounted staged milestone bonus (see
    staged_milestone_bonus's docstring for the bug this fixes in
    staged_potential_progress) plus a grasp-gated stillness penalty."""

    staged_milestone_bonus = RewTerm(
        func=ar4_mdp.staged_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("sphere"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_std": 0.1,
            "force_threshold": 0.05,
            "lift_minimal_height": 0.03,
            "goal_std": 0.3,
        },
    )

    # weight is POSITIVE, not negative: stillness_penalty's own return value
    # is already the signed penalty (-1.0 when triggered, 0.0 otherwise).
    # RewardManager.compute() multiplies func(...) * weight * dt
    # (reward_manager.py:149) - a negative weight here would multiply two
    # negatives into a POSITIVE reward for the exact stay-still-after-grasp
    # behavior this term exists to punish. Confirmed as a real bug via a
    # completed training run: Episode_Reward/stillness_penalty grew to
    # +1.3 (should be <= 0 always). See ROADMAP.md.
    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("sphere"),
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
class Ar4PickPlaceMirrorEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 mirror-goal task: pick up the sphere (randomized spawn across
    the full workspace) and place it on the opposite side of the robot.
    num_envs=4096 default (a real training-scale run, not the smaller
    num_envs=16 used by the single-object camera-training precedent) -
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

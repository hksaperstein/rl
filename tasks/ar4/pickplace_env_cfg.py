# tasks/ar4/pickplace_env_cfg.py
"""Pick-and-place RL task for the AR4 mk5 arm: pick up the cube and place it
near the sphere/wedge on the other side of the workspace.

Reward, observation, and termination logic is reused directly from Isaac
Lab's built-in Franka lift-task mdp module (generic, parametrized by
SceneEntityCfg - not Franka-specific despite the import path) rather than
reimplemented here. See docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md.

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
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .env_cfg import ActionsCfg, Ar4SceneCfg

# Empirically-tuned offset (m) from the link_6 frame to the gripper's jaw
# pinch point along link_6's local +Z axis (ee_link sits at this same frame
# with an identity transform, but isn't itself a rigid body, so link_6 is
# used directly) - same value used for the scripted IK reach in grasp_demo.py.
_EE_OFFSET = (0.0, 0.0, 0.09)


@configclass
class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
    """AR4 gripper+objects scene, plus an end-effector FrameTransformer sensor."""

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


@configclass
class CommandsCfg:
    """The cube's randomized target placement, in the robot's root frame.

    The robot base is rotated 180 deg about world Z (see robot_cfg.py), so
    a world-frame target near the sphere/wedge row (world x=-0.20,
    y in [0.28, 0.34]) becomes (x=+0.20, y in [-0.34, -0.28]) in the
    robot's own root frame (negate x and y - see grasp_demo.py's docstring
    for the same transform).
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.18, 0.22),
            pos_y=(-0.34, -0.28),
            pos_z=(0.0, 0.02),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: put the whole scene back to default, then jitter the cube's start pose."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    """Dense, staged reward: reach, lift, coarse + fine-grained goal tracking, and small
    action penalties. There is no separate sparse success-bonus term - success is signaled
    via the `cube_reached_goal` termination combined with the fine-grained goal-tracking
    reward, which increasingly rewards precise placement as the cube nears the target."""

    reaching_cube = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    lifting_cube = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")}, weight=15.0
    )

    cube_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=16.0,
    )

    cube_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Success (cube at target) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class Ar4PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 pick-and-place task: pick up the cube, place it near the sphere/wedge."""

    scene: Ar4PickPlaceSceneCfg = Ar4PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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


# World-frame constants shared by perception-consuming entry points
# (eval_loop.py --perception, interactive_demo.py, perception_calibration.py).
# Ground plane is world Z=0 (Ar4SceneCfg's GroundPlaneCfg has no pose override).
GROUND_Z = 0.0

# A generous box around the objects' known spawn region (x:[-0.2,0.2], y:[0.28,0.34]
# in objects_cfg.py) - wide enough for the interactive demo to tolerate the cube
# being dragged well outside its training-time randomization range, while still
# rejecting positions the policy was never trained anywhere near.
WORKSPACE_BOUNDS = {"x": (-0.30, 0.30), "y": (0.10, 0.45), "z": (0.0, 0.05)}

# Mounted above the workspace center (x=0, y=0.31 - the midpoint of the object
# layout), 0.55m above the ground plane, looking straight down. Under CameraCfg's
# "world" offset convention (forward +X, up +Z), a +90deg rotation about Y maps
# local forward (+X) to world -Z (straight down): quat (cos45, 0, sin45, 0).
# Verified empirically by Task 7 Step 2's smoke test, not just derived on paper.
_PERCEPTION_CAMERA_POS = (0.0, 0.31, 0.55)
_PERCEPTION_CAMERA_QUAT_WORLD = (0.70710678, 0.0, 0.70710678, 0.0)


@configclass
class Ar4PickPlacePerceptionSceneCfg(Ar4PickPlaceSceneCfg):
    """Pick-and-place scene plus a static top-down RGB-D perception camera.

    Only used by eval/demo entry points - never by train.py, since camera
    rendering isn't free even when unused and training stays on privileged
    simulation state (see docs/superpowers/specs/2026-07-04-ar4-perception-integration-design.md).
    """

    perception_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PerceptionCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40.0, clipping_range=(0.2, 1.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_PERCEPTION_CAMERA_POS, rot=_PERCEPTION_CAMERA_QUAT_WORLD, convention="world"),
    )


@configclass
class Ar4PickPlacePerceptionEnvCfg(Ar4PickPlaceEnvCfg):
    """Ar4PickPlaceEnvCfg with the perception camera enabled, num_envs=1
    (eval/demo run one environment at a time in the GUI)."""

    scene: Ar4PickPlacePerceptionSceneCfg = Ar4PickPlacePerceptionSceneCfg(num_envs=1, env_spacing=2.5)


@configclass
class DemoEventCfg:
    """Only resets the robot's joints on episode end - the cube (and other
    objects) are left exactly where they physically are, since the interactive
    demo relies on the user placing the cube by hand rather than training-time
    randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Ar4PickPlaceDemoEnvCfg(Ar4PickPlacePerceptionEnvCfg):
    """Perception-enabled pick-and-place env for the interactive demo."""

    events: DemoEventCfg = DemoEventCfg()

# tasks/ar4/grasp_verify_env_cfg.py
"""Bare (non-RL) environment config for grasp verification via contact sensors.
Reuses Ar4PickPlaceMirrorSceneCfg's scene (which includes gripper contact
sensors and the cube), with plain joint-space actions and minimal observations.
Adds a camera for video recording.

This is used by scripts/grasp_demo.py to verify antipodal grasping using
contact force measurements. See docs/superpowers/specs/ for the grasp_demo
design spec.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_env_cfg import _PERCEPTION_CAMERA_POS, _PERCEPTION_CAMERA_QUAT_WORLD
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
    """Action specifications: raw joint position targets for the 6 arm joints,
    plus a binary open/close command for the gripper's two jaw joints.
    Identical to env_cfg.py's Ar4EnvCfg.ActionsCfg."""

    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=1.0)
    gripper_position = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSED_COMMAND_EXPR,
    )


@configclass
class ObservationsCfg:
    """Observation specifications: joint position and velocity.
    Identical to env_cfg.py's Ar4EnvCfg.ObservationsCfg."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


_DEMO_CAMERA_POS = (-1.0, 1.0, 0.8)
_DEMO_CAMERA_QUAT_OPENGL = (-0.27031026, -0.16411083, 0.49233248, 0.81093078)
"""Wide 3/4-view camera, distinct from _PERCEPTION_CAMERA_POS's tight
close-up framing (tuned for object detection, not for watching a wider
arm motion). Eye (-1.0, 1.0, 0.8) looking at (-0.25, 0.0, 0.15) in world
frame - computed via the standard OpenGL look-at convention (forward=-Z,
up=+Y, matching isaaclab.utils.math.create_rotation_matrix_from_view's
own convention) rather than hand-derived, to avoid convention errors.
World-frame target chosen to center on the square-path demo's workspace
region (robot-frame x=0.17-0.33 maps to world x=-0.17..-0.33 given the
robot's 180deg base yaw - see tasks/ar4/robot_cfg.py's InitialStateCfg)."""

_CLOSEUP_CAMERA_POS = (0.15, 0.36, 0.10)
_CLOSEUP_CAMERA_QUAT_OPENGL = (1.0, 0.0, 0.0, 0.0)
"""Placeholder eye/orientation only - a genuinely useful close-up on the
gripper-fingertip/cube contact region depends on the arm's own live-solved
GRASP_Q waypoint (reach/tilt-dependent, not fixed), so these values are
overwritten at runtime via closeup_camera.set_world_poses() once that
waypoint's real jaw1/jaw2/cube world positions are measured - see
scripts/grasp_demo_v2.py's ``--closeup-camera``/``_compute_closeup_camera``
(2026-07-24, ar4-closeup-grasp-video task), which follows the same
live-measure-then-reposition pattern already established by
scripts/_record_jaw_fix_open_close_cycle.py's own camera-tuning history
(see that script's own comments for what worked/produced black frames)."""

_ELBOW_CONTEXT_CAMERA_POS = (0.15, 0.36, 0.10)
_ELBOW_CONTEXT_CAMERA_QUAT_OPENGL = (1.0, 0.0, 0.0, 0.0)
"""Placeholder only - overwritten at runtime, same pattern as
_CLOSEUP_CAMERA_POS/_CLOSEUP_CAMERA_QUAT_OPENGL above. A FOURTH camera
(2026-07-24, ar4-axis-align-ik task, coordinator-directed mid-task
correction), distinct from closeup_camera's tight gripper/cube-only
framing: this one is deliberately WIDE enough to keep the elbow
(``link_3``, the joint whose own hard limit this whole investigation's
current focus - see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md
- is centered on), forearm, wrist, and gripper/cube ALL visible together in
one frame, so the elbow's own live behavior during a grasp attempt is
directly watchable, not just its downstream effect on the gripper. Repositioned
via ``elbow_context_camera.set_world_poses()`` from a LIVE measurement of
link_3's and the gripper/cube's real world positions at the settled GRASP_Q
pose - see ``scripts/grasp_demo_v2.py``'s ``--elbow-camera``/
``_compute_elbow_context_camera`` and the ``isaac-sim-video-capture``
skill's "Deriving a new camera position" section (live-measure-the-subject,
never guess coordinates)."""


@configclass
class Ar4GraspVerifySceneCfg(Ar4PickPlaceMirrorSceneCfg):
    """Ar4PickPlaceMirrorSceneCfg extended with cameras for video recording."""

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

    demo_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=40.0, clipping_range=(0.1, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_DEMO_CAMERA_POS, rot=_DEMO_CAMERA_QUAT_OPENGL, convention="opengl"),
    )

    # 2026-07-24 (ar4-closeup-grasp-video task): a THIRD camera, distinct
    # from both perception_camera (tuned for object detection, tight but
    # fixed to the resting/vertical pose) and demo_camera (wide 3/4 view,
    # never resolves the (then-)12mm cube clearly per this project's own
    # standing finding - kb/wiki/concepts/ar4-vs-franka-root-cause-
    # comparison.md's 2026-07-24 ar4-jaw-contact-sensor-hypothesis UPDATE
    # (cube bumped 12mm->20mm->15mm, both changes 2026-07-24,
    # ar4-cube-size-increase task, partly to address this same visibility
    # gap). Always present
    # in the scene (harmless/unused when not recorded) so
    # scripts/grasp_demo_v2.py's --closeup-camera flag can reposition and
    # record it without a scene-cfg-level conditional. Narrower aperture
    # (20.955mm, the PinholeCameraCfg default - left unset here, same as
    # _record_jaw_fix_open_close_cycle.py's own close-up camera) combined
    # with a longer focal_length than demo_camera's wide-view 18mm gives a
    # tighter FOV appropriate for resolving the cube's contact with the
    # jaw fingertips at close range.
    closeup_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CloseupCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=40.0, focus_distance=400.0, clipping_range=(0.02, 1.0)),
        offset=CameraCfg.OffsetCfg(pos=_CLOSEUP_CAMERA_POS, rot=_CLOSEUP_CAMERA_QUAT_OPENGL, convention="opengl"),
    )

    # 2026-07-24 (ar4-axis-align-ik task, coordinator-directed mid-task
    # correction): a FOURTH camera, wide enough to keep link_3 (elbow),
    # forearm, wrist, and gripper/cube all visible together - see
    # _ELBOW_CONTEXT_CAMERA_POS's own docstring above for why this is
    # distinct from closeup_camera's tight gripper-only framing. Always
    # present in the scene (harmless/unused when not recorded), same
    # convention as closeup_camera. Wider clipping-range far bound than
    # closeup_camera (0.02-1.0) since this camera sits further back to fit
    # the whole elbow-to-gripper span in frame, not just the fingertip
    # region.
    elbow_context_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ElbowContextCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, clipping_range=(0.02, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=_ELBOW_CONTEXT_CAMERA_POS, rot=_ELBOW_CONTEXT_CAMERA_QUAT_OPENGL, convention="opengl"
        ),
    )


@configclass
class Ar4GraspVerifyEnvCfg(ManagerBasedEnvCfg):
    """Bare (non-RL) env cfg for grasp verification: reuses
    Ar4PickPlaceMirrorSceneCfg (which includes gripper contact sensors and
    the cube) with plain joint-space actions and minimal observations.
    Adds a camera for video recording.
    num_envs=1 (single verification instance)."""

    scene: Ar4GraspVerifySceneCfg = Ar4GraspVerifySceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 2
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        # Scene-wide default so the small objects grip reliably (default friction is too low).
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

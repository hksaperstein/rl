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
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class ActionsCfg:
    """Action specifications: raw joint position targets for the 6 arm joints,
    plus a binary open/close command for the gripper's two jaw joints.
    Identical to env_cfg.py's Ar4EnvCfg.ActionsCfg."""

    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=1.0)
    gripper_position = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
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


@configclass
class Ar4GraspVerifySceneCfg(Ar4PickPlaceMirrorSceneCfg):
    """Ar4PickPlaceMirrorSceneCfg extended with a camera for video recording."""

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
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        # Scene-wide default so the small objects grip reliably (default friction is too low).
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

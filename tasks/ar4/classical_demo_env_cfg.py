# tasks/ar4/classical_demo_env_cfg.py
"""Scene + env cfg for the classical (zero-RL) camera-perception-driven
pick-and-place demo (scripts/classical_pickplace_demo.py). See
docs/superpowers/specs/2026-07-07-ar4-classical-perception-pickplace-demo-design.md.

Additive/parallel to every other tasks/ar4/*.py file: does NOT modify
env_cfg.py, objects_cfg.py, or any pickplace_*.py file. Closely modeled on
Ar4PickPlaceSingleObjectSceneCfg (pickplace_single_object_env_cfg.py),
swapping SPHERE_CFG for CUBE_CFG to match this repo's current single-cube
scope, and on Ar4EnvCfg (env_cfg.py) for the bare ManagerBasedEnvCfg
pattern (no reward/termination managers - this is a scripted demo, not an
RL task).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET, _PERCEPTION_CAMERA_POS, _PERCEPTION_CAMERA_QUAT_WORLD
from .robot_cfg import AR4_MK5_CFG, ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class Ar4ClassicalDemoSceneCfg(InteractiveSceneCfg):
    """AR4 gripper + a single cube, plus the end-effector FrameTransformer
    sensor and the top-down RGB-D perception camera - both always on, since
    this scene exists specifically for the live-perception classical demo.

    Deliberately duplicates (rather than subclasses) Ar4SceneCfg/
    Ar4PickPlaceSceneCfg/Ar4PickPlaceSingleObjectSceneCfg, since all three
    include either the full four-object set or the sphere instead of the
    cube - same rationale as Ar4PickPlaceSingleObjectSceneCfg's own
    docstring.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG

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
class ActionsCfg:
    """Task-space Cartesian action (arm) + binary gripper command - the
    same plain (non-residual) DifferentialInverseKinematicsActionCfg every
    task-space env cfg in this repo uses. This demo has no policy: the
    script itself commands the base pursuit step directly as the raw
    action each step (see scripts/classical_pickplace_demo.py), so there
    is nothing to add a residual on top of."""

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
    """Minimal observation group (joint pos/vel only) - required by
    ManagerBasedEnvCfg's manager machinery, matching Ar4EnvCfg's own
    minimal ObservationsCfg (env_cfg.py). Not used by the demo script's
    own control logic, which reads camera/ee_frame/robot state directly."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ar4ClassicalDemoEnvCfg(ManagerBasedEnvCfg):
    """Bare (non-RL) env cfg for the classical perception-driven demo - no
    reward/termination managers, matching Ar4EnvCfg's (env_cfg.py)
    precedent for scripted demos. num_envs=1 (single demo instance,
    matching interactive_demo.py's convention)."""

    scene: Ar4ClassicalDemoSceneCfg = Ar4ClassicalDemoSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

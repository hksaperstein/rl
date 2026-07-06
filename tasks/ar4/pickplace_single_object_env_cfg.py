# tasks/ar4/pickplace_single_object_env_cfg.py
"""Single-object variant of the AR4 pick-and-place task: only the sphere is
present in the scene (no cube/rect_prism/wedge), and the perception camera is
always enabled - this env cfg is built for the camera-observed training
experiment in docs/superpowers/specs/2026-07-05-ar4-single-object-camera-training-design.md.

Additive/parallel to pickplace_env_cfg.py: deliberately does NOT touch
`Ar4SceneCfg` (env_cfg.py) or `objects_cfg.py`, since interactive_demo.py,
perception/tests, and other scripts depend on all four objects existing
there. Reuses the four-object task's reward/termination/command/action/
observation manager configs unchanged (they're already parametrized purely
by SceneEntityCfg("sphere")/("ee_frame") and don't reference the other
objects), swapping in only a smaller scene.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from .objects_cfg import SPHERE_CFG
from .pickplace_env_cfg import (
    _EE_OFFSET,
    _PERCEPTION_CAMERA_POS,
    _PERCEPTION_CAMERA_QUAT_WORLD,
    Ar4PickPlaceEnvCfg,
)
from .robot_cfg import AR4_MK5_CFG


@configclass
class Ar4PickPlaceSingleObjectSceneCfg(InteractiveSceneCfg):
    """AR4 gripper + sphere only (no cube/rect_prism/wedge), plus the
    end-effector FrameTransformer sensor and the top-down RGB-D perception
    camera - both always on, since this scene exists specifically for the
    camera-observed training experiment.

    Deliberately duplicates (rather than subclasses) `Ar4SceneCfg`/
    `Ar4PickPlaceSceneCfg`/`Ar4PickPlacePerceptionSceneCfg` from env_cfg.py/
    pickplace_env_cfg.py, since all three include cube/rect_prism/wedge.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sphere = SPHERE_CFG

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
class Ar4PickPlaceSingleObjectEnvCfg(Ar4PickPlaceEnvCfg):
    """Ar4PickPlaceEnvCfg with the single-object scene (sphere only) and the
    perception camera always enabled, for camera-observed training. Small
    `num_envs` default - see the design doc's discussion of the perception
    pipeline's CPU/numpy cost; scripts/train.py's --num_envs flag overrides
    this per-run."""

    scene: Ar4PickPlaceSingleObjectSceneCfg = Ar4PickPlaceSingleObjectSceneCfg(num_envs=16, env_spacing=2.5)

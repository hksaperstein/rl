# tasks/ar4/pedestal_grasp_camera_env_cfg.py
"""Camera-enabled variant of Ar4PickPlaceGraspGoalEnvCfg (2026-07-28,
ar4-grasp-trivial-friction-check task) - adds the two already-proven
diagnostic cameras (closeup_camera/elbow_context_camera) from
grasp_verify_env_cfg.py's own established pattern onto the PEDESTAL scene
(Ar4PickPlaceGraspGoalSceneCfg, which carries the pedestal + the
now-realistic-friction cube - see objects_cfg.py's CUBE_PHYSICS_MATERIAL),
instead of the ground-level Ar4PickPlaceMirrorSceneCfg that
grasp_verify_env_cfg.py itself uses.

Purely additive: does not modify pickplace_graspgoal_env_cfg.py,
grasp_verify_env_cfg.py, or objects_cfg.py. CameraCfg field values
(lens/clipping-range) are copied verbatim from grasp_verify_env_cfg.py's
own closeup_camera/elbow_context_camera - only the eye/target pose is a
placeholder, overwritten at runtime via set_world_poses() from a LIVE
measurement of jaw1/jaw2/cube/elbow world positions, same
live-measure-then-reposition convention as
scripts/grasp_demo_v2.py's _compute_closeup_camera/_compute_elbow_context_camera
(reused directly by the script that imports this module - see
scripts/ar4_pedestal_grasp_trivial_check.py).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg, Ar4PickPlaceGraspGoalSceneCfg

# Placeholder eye/target only - both cameras are repositioned at runtime
# from live-measured jaw1/jaw2/cube/elbow world positions (see this
# module's own docstring). Kept as a harmless, valid default so
# CameraCfg.OffsetCfg construction has something to start from.
_PLACEHOLDER_POS = (0.15, 0.36, 0.10)
_PLACEHOLDER_QUAT_OPENGL = (1.0, 0.0, 0.0, 0.0)


@configclass
class Ar4PedestalGraspCameraSceneCfg(Ar4PickPlaceGraspGoalSceneCfg):
    """Ar4PickPlaceGraspGoalSceneCfg (pedestal + realistic-friction cube)
    extended with a tight gripper/cube close-up camera and a wider
    elbow-inclusive context camera - identical CameraCfg values to
    grasp_verify_env_cfg.py's own closeup_camera/elbow_context_camera."""

    closeup_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CloseupCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=40.0, focus_distance=400.0, clipping_range=(0.02, 1.0)),
        offset=CameraCfg.OffsetCfg(pos=_PLACEHOLDER_POS, rot=_PLACEHOLDER_QUAT_OPENGL, convention="opengl"),
    )

    elbow_context_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ElbowContextCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, clipping_range=(0.02, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=_PLACEHOLDER_POS, rot=_PLACEHOLDER_QUAT_OPENGL, convention="opengl"),
    )


@configclass
class Ar4PedestalGraspCameraEnvCfg(Ar4PickPlaceGraspGoalEnvCfg):
    """Ar4PickPlaceGraspGoalEnvCfg with num_envs=1 and the two diagnostic
    cameras above, for a single-env scripted grasp+lift attempt with real
    close-up + elbow-inclusive frame/video capture."""

    scene: Ar4PedestalGraspCameraSceneCfg = Ar4PedestalGraspCameraSceneCfg(num_envs=1, env_spacing=5.0)

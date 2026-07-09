# tasks/ar4/graspgoal_democam_env_cfg.py
"""Close-up demo-camera variant of Ar4PickPlaceGraspGoalEnvCfg (Experiment 26
post-hoc visual verification only, see
docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md).
Adds a single demo_camera to the scene for individual-env close-up video,
following tasks/ar4/touchgoal_democam_env_cfg.py's exact pattern (a plain
scene-cfg subclass adding a CameraCfg field, env-cfg subclass pointing at the
new scene cfg). Everything else (observations/actions/rewards/terminations/
events) is inherited unchanged from Ar4PickPlaceGraspGoalEnvCfg.

Deliberately does NOT modify pickplace_graspgoal_env_cfg.py, mdp.py, or
grasp_goal_reward.py - additive/parallel only, same convention as that file's
own module docstring.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg, Ar4PickPlaceGraspGoalSceneCfg

# Eye (0.0, 0.6, 0.2) looking at (0.0, 0.0, 0.15) in world frame - a front,
# head-on view of the robot itself (not the wider workspace), ~0.6m (~2ft)
# away. CORRECTED SIDE: the robot's front faces +Y (toward the cube/goal
# workspace, established empirically from the graspgoal task's cube/goal
# world positions - both at y=0.28 while the robot base sits at y=0, per
# tasks/ar4/pickplace_graspgoal_env_cfg.py), so to see its FRONT the camera
# must be on the +Y side, beyond the robot, looking back toward -Y - a
# camera at -Y looking toward +Y (the first attempt) sees the robot's BACK
# (confirmed directly: that render showed a rear access panel/vents, not a
# face). Height/target unchanged from the prior fix (z=0.2/0.15 - the very
# first attempt at z=0.3/0.3 only caught a sliver of the gripper, since most
# of the arm's resting bulk sits lower than shoulder height). Quaternion
# computed via Isaac Lab's own create_rotation_matrix_from_view/
# quat_from_matrix (OpenGL: -Z forward, +Y up), not hand-derived, matching
# touchgoal_democam_env_cfg.py's own convention and rationale for avoiding
# convention errors.
_DEMO_CAMERA_POS = (0.0, 0.6, 0.2)
_DEMO_CAMERA_QUAT_OPENGL = (0.7358822822570801, -0.6771095395088196, -0.0, -0.0)

# Dark backdrop wall, positioned behind the robot from the camera's new
# vantage (beyond the robot on the -Y side) - eliminates the washed-out
# empty-background problem at its root (no sky texture behind the dome
# light means anything not intersecting real geometry renders as flat
# bright gray) rather than only fighting it via light intensity.
_BACKDROP_POS = (0.0, -1.0, 0.75)
_BACKDROP_SIZE = (2.0, 0.05, 1.6)


@configclass
class Ar4GraspGoalDemoSceneCfg(Ar4PickPlaceGraspGoalSceneCfg):
    """Ar4PickPlaceGraspGoalSceneCfg extended with a close-up demo camera and
    a dark backdrop wall. Overrides the inherited light (much dimmer than the
    base scene's intensity=2000 DomeLightCfg, and dimmer than the touchgoal
    precedent's 800 - this camera's first render at 800 was still badly
    overexposed: not literal fog/atmosphere (Isaac Lab's RenderCfg has no
    such setting), but the dome light illuminating empty background
    uniformly bright with no sky texture behind it) and widens env_spacing
    (isolates envs further apart so a neighboring env's arm doesn't bleed
    into frame)."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=150.0, color=(0.75, 0.75, 0.75)),
    )

    backdrop: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Backdrop",
        spawn=sim_utils.CuboidCfg(
            size=_BACKDROP_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BACKDROP_POS),
    )

    demo_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40.0, clipping_range=(0.05, 3.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_DEMO_CAMERA_POS, rot=_DEMO_CAMERA_QUAT_OPENGL, convention="opengl"),
    )


@configclass
class Ar4GraspGoalDemoEnvCfg(Ar4PickPlaceGraspGoalEnvCfg):
    """Ar4PickPlaceGraspGoalEnvCfg with a small num_envs and a close-up demo
    camera, for recording individual-env verification video instead of the
    wide multi-env training grid camera. env_spacing widened 2.5 -> 5.0 to
    keep neighboring envs out of the demo camera's frame."""

    scene: Ar4GraspGoalDemoSceneCfg = Ar4GraspGoalDemoSceneCfg(num_envs=4, env_spacing=5.0)

# tasks/ar4/touchgoal_democam_env_cfg.py
"""Close-up demo-camera variant of Ar4PickPlaceTouchGoalEnvCfg (Experiment 25
post-hoc visual verification only, see
docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md).
Adds a single demo_camera to the scene for individual-env close-up video,
following tasks/ar4/grasp_verify_env_cfg.py's demo_camera pattern (a plain
scene-cfg subclass adding a CameraCfg field, env-cfg subclass pointing at the
new scene cfg). Everything else (observations/actions/rewards/terminations/
events) is inherited unchanged from Ar4PickPlaceTouchGoalEnvCfg.

Deliberately does NOT modify pickplace_touchgoal_env_cfg.py, mdp.py, or
touch_goal_reward.py - additive/parallel only, same convention as that file's
own module docstring.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg, Ar4PickPlaceTouchGoalSceneCfg

# Eye (0.30, 0.95, 0.75) looking at (0.0, 0.28, 0.078) in world frame - the
# midpoint between the cube's touch point (0.20, 0.28, 0.006+~0) and the fixed
# goal point (-0.20, 0.28, 0.15), ~0.995m away (vs. the original ~0.606m) -
# higher and further back than the first attempt, which was low and close
# enough to catch an overexposed horizon/sky and a neighboring env's arm in
# frame. Quaternion computed via Isaac Lab's own
# create_rotation_matrix_from_view/quat_from_matrix (OpenGL: -Z forward, +Y
# up), not hand-derived (scripts/_compute_camera_quat.py), matching
# grasp_verify_env_cfg.py's own convention and rationale for avoiding
# convention errors.
_DEMO_CAMERA_POS = (0.30, 0.95, 0.75)
_DEMO_CAMERA_QUAT_OPENGL = (0.912114143371582, -0.3530109226703644, 0.19435302913188934, 0.07521946728229523)


@configclass
class Ar4TouchGoalDemoSceneCfg(Ar4PickPlaceTouchGoalSceneCfg):
    """Ar4PickPlaceTouchGoalSceneCfg extended with a close-up demo camera.
    Overrides the inherited light (dimmer, since the base scene's
    intensity=2000 DomeLightCfg overexposed this camera's closer framing)
    and widens env_spacing (isolates envs further apart so a neighboring
    env's arm doesn't bleed into frame at this camera's field of view)."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75)),
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
class Ar4TouchGoalDemoEnvCfg(Ar4PickPlaceTouchGoalEnvCfg):
    """Ar4PickPlaceTouchGoalEnvCfg with a small num_envs and a close-up demo
    camera, for recording individual-env verification video instead of the
    wide multi-env training grid camera. env_spacing widened 2.5 -> 5.0 to
    keep neighboring envs out of the demo camera's frame."""

    scene: Ar4TouchGoalDemoSceneCfg = Ar4TouchGoalDemoSceneCfg(num_envs=4, env_spacing=5.0)

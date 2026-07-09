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
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg, Ar4PickPlaceTouchGoalSceneCfg

# Eye (0.45, 0.59, 0.34) looking at (0.0, 0.28, 0.078) in world frame - the
# midpoint between the cube's touch point (0.20, 0.28, 0.006+~0) and the fixed
# goal point (-0.20, 0.28, 0.15), ~0.606m away, a 3/4-elevated close-up view
# of the touch-to-goal traverse. Quaternion computed via Isaac Lab's own
# create_rotation_matrix_from_view/quat_from_matrix (OpenGL: -Z forward, +Y
# up), not hand-derived, matching grasp_verify_env_cfg.py's own convention and
# rationale for avoiding convention errors.
_DEMO_CAMERA_POS = (0.45, 0.59, 0.34)
_DEMO_CAMERA_QUAT_OPENGL = (0.3936, 0.2478, 0.4716, 0.7492)


@configclass
class Ar4TouchGoalDemoSceneCfg(Ar4PickPlaceTouchGoalSceneCfg):
    """Ar4PickPlaceTouchGoalSceneCfg extended with a close-up demo camera."""

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
    wide multi-env training grid camera."""

    scene: Ar4TouchGoalDemoSceneCfg = Ar4TouchGoalDemoSceneCfg(num_envs=4, env_spacing=2.5)

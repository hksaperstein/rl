# tasks/ar4/pickplace_warmresidual_env_cfg.py
"""Warm-started residual-RL variant (Experiment 23): identical to
Experiment 22's Ar4PickPlaceJawMirrorEnvCfg
(tasks/ar4/pickplace_jawmirror_env_cfg.py) in reward/observation/
termination/curriculum configuration and gripper action - only the ARM
action term changes, from plain joint-space position control to
WarmStartedResidualDifferentialIKActionCfg (a classical waypoint-pursuit
base controller plus a warm-started RL residual on top).

Revisits Experiment 13's residual-RL-over-classical-controller paradigm
(tasks/ar4/residual_ik_action.py, tasks/ar4/pickplace_residual_env_cfg.py)
with the specific literature-grounded fix (Johannink et al. 2019's
warm-start technique) Experiment 13's own ROADMAP entry diagnosed as
missing but never implemented. See
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_jawmirror_env_cfg.py, pickplace_orientationbias_env_cfg.py,
pickplace_residual_env_cfg.py, tasks/ar4/residual_ik_action.py, or any
other existing env cfg/action file. Reuses Ar4PickPlaceMirrorSceneCfg,
Ar4PickPlaceTaskspacePPORunnerCfg, and Experiment 22's RewardsCfg/
ObservationsCfg/TerminationsCfg/CurriculumCfg directly (imported via
Experiment 20's pickplace_orientationbias_env_cfg.py, the same reuse
chain pickplace_jawmirror_env_cfg.py already uses).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp as ar4_mdp
from .actions import MirroredGripperActionCfg, WarmStartedResidualDifferentialIKActionCfg
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_orientationbias_env_cfg import (
    CurriculumCfg,
    EventCfg as _BaseEventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg  # noqa: F401 (re-exported for scripts/train.py's PPO-cfg selection)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_residual_env_cfg.py's/pickplace_taskspace_env_cfg.py's
# EventCfg reuse - the waypoint geometry itself is unchanged, only the
# action term consuming it is new.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10

# Same value as Experiment 21/22 - unchanged, this experiment isolates
# the arm action term as the only new variable.
_PROXIMITY_THRESHOLD = 0.05

# 50 iterations x num_steps_per_env=24 (tasks/ar4/agents/rsl_rl_ppo_cfg.py)
# = 1200 env steps, ~3.3% of the full 1500-iteration/36,000-step run.
_WARMUP_STEPS = 1200
_ADVANCE_TOLERANCE = 0.03
_BASE_MAX_STEP = 0.05


@configclass
class ActionsCfg:
    """Arm action replaced with WarmStartedResidualDifferentialIKActionCfg
    (classical waypoint pursuit + warm-started RL residual). Gripper
    action unchanged from Experiment 22: MirroredGripperActionCfg (jaw2
    mirrors jaw1's actual position, gated open unless the cube is within
    _PROXIMITY_THRESHOLD)."""

    arm_action = WarmStartedResidualDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=isaaclab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        base_max_step=_BASE_MAX_STEP,
        advance_tolerance=_ADVANCE_TOLERANCE,
        warmup_steps=_WARMUP_STEPS,
    )
    gripper_position = MirroredGripperActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=_PROXIMITY_THRESHOLD,
    )


@configclass
class EventCfg(_BaseEventCfg):
    """Experiment 22's exact reset_all/reset_cube_position/randomize_goal
    events (inherited unchanged from Experiment 20's EventCfg, via
    pickplace_orientationbias_env_cfg.py), plus one new event:
    compute_path_waypoints, registered LAST (after cube-position reset
    and goal randomization, per that function's own documented ordering
    requirement - it reads both env._target_pos_w and the cube's
    now-updated position)."""

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class Ar4PickPlaceWarmResidualEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 warm-started residual-RL task (Experiment 23): Experiment 22's
    exact reward/observation/termination/curriculum configuration and
    gripper mechanism, with the arm's action replaced by a classical
    waypoint-pursuit base controller plus a warm-started RL residual.
    num_envs=4096 default - scripts/train.py's --num_envs flag overrides
    this per-run same as every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)

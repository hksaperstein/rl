# tasks/ar4/pickplace_verticallock_env_cfg.py
"""IK-constrained vertical/top-down approach orientation variant
(Experiment 20): identical reward/observation/event/termination/
curriculum configuration to Experiment 18's Ar4PickPlacePregraspEnvCfg
(tasks/ar4/pickplace_pregrasp_env_cfg.py, imported and reused directly,
not copied), with one changed variable - the arm's action space is
replaced with VerticalLockDifferentialIKActionCfg (tasks/ar4/actions.py),
which locks the end-effector's orientation to a fixed, always-vertical
target every step, leaving only 3D Cartesian position under policy
control. See
docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md.

Additive/parallel to every other pickplace_*.py file: does not modify
pickplace_pregrasp_env_cfg.py, pickplace_graspgated_env_cfg.py, or any
other existing env cfg. Reuses Ar4PickPlaceMirrorSceneCfg and
Ar4PickPlaceTaskspacePPORunnerCfg directly (task-space action, same
agent cfg family as Experiment 11's taskspace variant).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass

from .actions import VerticalLockDifferentialIKActionCfg
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_pregrasp_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS
import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils

# Determined empirically (Task 2, Step 1): the quaternion (w, x, y, z)
# that points the gripper's approach axis straight down (world -Z).
# Measured via sweeping joint_5 and identifying the configuration where
# the gripper's +Z local axis aligns with world -Z direction.
_FIXED_DOWNWARD_QUAT = (-0.007073760032653809, 0.006892431993037462, -0.9996296167373657, -0.02535712718963623)


@configclass
class ActionsCfg:
    """Task-space action: 3D Cartesian position under policy control,
    orientation locked to _FIXED_DOWNWARD_QUAT every step. Gripper action
    unchanged from every prior experiment."""

    arm_action = VerticalLockDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=VerticalLockDifferentialIKActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        fixed_orientation=_FIXED_DOWNWARD_QUAT,
    )
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class Ar4PickPlaceVerticalLockEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 vertical-orientation-lock task (Experiment 20): Experiment
    18's exact reward/observation/event/termination/curriculum
    configuration, with the arm's action space replaced by a fixed-
    orientation task-space IK action. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

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

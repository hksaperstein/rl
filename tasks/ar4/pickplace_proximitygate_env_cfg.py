# tasks/ar4/pickplace_proximitygate_env_cfg.py
"""Proximity-gated gripper closing variant (Experiment 21): identical to
Experiment 20's Ar4PickPlaceOrientationBiasEnvCfg
(tasks/ar4/pickplace_orientationbias_env_cfg.py) in every respect - same
reward set (including orientation_alignment_bonus and
pregrasp_readiness), same scene, same antipodal grasp gate - except the
gripper action term is replaced with
ProximityGatedBinaryJointPositionActionCfg (tasks/ar4/actions.py),
which forces the gripper open regardless of the policy's own command
unless the cube is within proximity_threshold of the end-effector.

Directly motivated by Experiment 20's own follow-up instrumented
diagnostic: across 750 rollout steps of the trained checkpoint,
gripper_jaw1_joint's contact sensor registered zero force at every
step while gripper_jaw2_joint registered contact intermittently - an
asymmetric single-jaw-contact failure. This experiment tests whether
premature/imprecise closing (allowed at any distance in every prior
experiment) explains that asymmetry. See
docs/superpowers/specs/2026-07-07-ar4-experiment21-proximity-gated-gripper-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_orientationbias_env_cfg.py, pickplace_pregrasp_env_cfg.py, or
any other existing env cfg. Reuses Ar4PickPlaceMirrorSceneCfg,
Ar4PickPlacePPORunnerCfg, and Experiment 20's RewardsCfg/
ObservationsCfg/EventCfg/TerminationsCfg/CurriculumCfg directly.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .actions import ProximityGatedBinaryJointPositionActionCfg
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_orientationbias_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Comfortably larger than the cube's own size (0.018m per side, 0.009m
# half-extent) and the _EE_OFFSET pinch-point geometry, giving the
# policy room to initiate closing just before full contact rather than
# requiring exact contact first (which would make closing impossible to
# ever trigger). Matches the general "close" proximity scale already
# used elsewhere in this repo (reaching_object's std=0.1).
_PROXIMITY_THRESHOLD = 0.05


@configclass
class ActionsCfg:
    """Plain joint-space arm action, identical to Experiment 20. Gripper
    action replaced with ProximityGatedBinaryJointPositionActionCfg -
    forced open unless the cube is within _PROXIMITY_THRESHOLD of the
    end-effector."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = ProximityGatedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=_PROXIMITY_THRESHOLD,
    )


@configclass
class Ar4PickPlaceProximityGateEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 proximity-gated-gripper task (Experiment 21): Experiment 20's
    exact reward/observation/event/termination/curriculum configuration,
    with the gripper action hard-gated to stay open until the cube is
    close. num_envs=4096 default - scripts/train.py's --num_envs flag
    overrides this per-run same as every other env cfg in this repo."""

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

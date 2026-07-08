# tasks/ar4/pickplace_jawmirror_env_cfg.py
"""Software jaw-mirroring variant (Experiment 22): identical to
Experiment 21's Ar4PickPlaceProximityGateEnvCfg
(tasks/ar4/pickplace_proximitygate_env_cfg.py) in every respect - same
reward set, same scene, same antipodal grasp gate, same proximity-gated
closing - except the gripper action term adds jaw2-follows-jaw1
software mirroring on top of the existing proximity gate.

Directly motivated by Experiment 21's own instrumented contact
diagnostic: both jaws now genuinely contact the cube with real force
(jaw1: 6.73N, jaw2: 27.44N) but never simultaneously
(both_magnitude_ok_steps=0/750). This experiment tests whether
commanding gripper_jaw2_joint's target to continuously track
gripper_jaw1_joint's actual measured position - a software control-loop
reference, not the PhysX-level constraint Experiment 19 already ruled
out - achieves the simultaneity that timing alone (Experiment 21) did
not. See
docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_proximitygate_env_cfg.py, pickplace_orientationbias_env_cfg.py,
tasks/ar4/robot_cfg.py, scripts/build_asset.py, or any other existing
env cfg/asset file. Reuses Ar4PickPlaceMirrorSceneCfg,
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

from .actions import MirroredGripperActionCfg
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_orientationbias_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same value as Experiment 21 - unchanged, this experiment isolates
# jaw-mirroring as the only new variable.
_PROXIMITY_THRESHOLD = 0.05


@configclass
class ActionsCfg:
    """Plain joint-space arm action, identical to Experiment 21. Gripper
    action replaced with MirroredGripperActionCfg - keeps the proximity
    gate (forced open unless the cube is within _PROXIMITY_THRESHOLD)
    and adds jaw2-tracks-jaw1's-actual-position software mirroring on
    top."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
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
class Ar4PickPlaceJawMirrorEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 jaw-mirroring task (Experiment 22): Experiment 21's exact
    reward/observation/event/termination/curriculum configuration and
    proximity-gated closing, with jaw2's target additionally mirroring
    jaw1's actual measured position every step. num_envs=4096 default -
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

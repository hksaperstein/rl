"""Sim-foundation environment for the AR4 mk5 arm, its gripper, and the
graspable objects in the scene.

Launch, read joint state, drive joints, record joint data. Deliberately has
no reward or termination logic - that belongs to the task/scenario layer
built on top of this foundation.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .objects_cfg import CUBE_CFG, RECT_PRISM_CFG, SPHERE_CFG, WEDGE_CFG
from .robot_cfg import (
    AR4_MK5_CFG,
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
    GRIPPER_OPEN_POS,
)


@configclass
class Ar4SceneCfg(InteractiveSceneCfg):
    """AR4 mk5 arm + gripper and four graspable objects, on a ground plane."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG
    rect_prism: RigidObjectCfg = RECT_PRISM_CFG
    sphere: RigidObjectCfg = SPHERE_CFG
    wedge: RigidObjectCfg = WEDGE_CFG


@configclass
class ActionsCfg:
    """Action specifications: raw joint position targets for the 6 arm joints,
    plus a binary open/close command for the gripper's two jaw joints."""

    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=1.0)
    gripper_position = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSED_COMMAND_EXPR,
    )


@configclass
class ObservationsCfg:
    """Observation specifications: joint position and velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ar4EnvCfg(ManagerBasedEnvCfg):
    """AR4 sim foundation: launch, drive joints, read joint state, record data."""

    scene: Ar4SceneCfg = Ar4SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 2
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        # Scene-wide default so the small objects grip reliably (default friction is too low).
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

"""ArticulationCfg for the AR4 mk5 arm, including the parallel-jaw gripper.

Points at the USD asset produced by scripts/build_asset.py. Import this
module only after an Isaac Sim/Isaac Lab AppLauncher has been created.
"""

import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_USD_MANIFEST = os.path.join(_RL_ROOT, "assets", "ar4_mk5", "usd_path.txt")

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_JOINT_NAMES = ["gripper_jaw1_joint", "gripper_jaw2_joint"]
GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0


def _resolve_usd_path() -> str:
    if not os.path.isfile(_USD_MANIFEST):
        sys.exit(
            f"AR4 USD asset not found (missing {_USD_MANIFEST}).\n"
            "Run scripts/build_asset.py first."
        )
    with open(_USD_MANIFEST) as f:
        usd_path = f.read().strip()
    if not os.path.isfile(usd_path):
        sys.exit(f"AR4 USD asset manifest points at a missing file: {usd_path}\nRun scripts/build_asset.py again.")
    return usd_path


AR4_MK5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Base rotated 180 deg about the vertical (Z) axis so the arm's
        # natural reach faces the object layout. Quaternion order is (w, x, y, z).
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            **{name: 0.0 for name in ARM_JOINT_NAMES},
            **{name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit_sim=20.0,
            velocity_limit_sim=3.0,
            stiffness=40.0,
            damping=4.0,
            armature=1e-3,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_jaw[12]_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=50.0,
            armature=1e-3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the AR4 mk5 arm, including the parallel-jaw gripper."""

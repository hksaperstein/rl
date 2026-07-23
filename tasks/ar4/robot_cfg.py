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
# UPDATE 2026-07-23: the -1.0-gearing convention below (2026-07-21) was
# ITSELF WRONG, empirically disproven by a direct live measurement
# (scripts/_sweep_jaw2_symmetry.py, see kb/wiki/concepts/
# ar4-vs-franka-root-cause-comparison.md's 2026-07-23 UPDATE for the full
# writeup). That script held jaw1 fixed at its OPEN target and swept
# jaw2's commanded joint value directly, reading back both jaws' actual
# world-frame body positions: jaw2's world-X position is exactly
# -1.0 * (jaw2's own commanded joint value) - i.e. jaw2's local-to-world
# mapping ALREADY contains a -1 factor (from its 180-degree-rotated joint
# frame, see robot_cfg.py history / the 2026-07-21 asset debugging doc).
# Commanding jaw2 to -1.0 * jaw1's value (this file's old convention)
# therefore DOUBLE-NEGATES and drives jaw2 to the SAME world point as
# jaw1 (measured separation 0.00001m - confirmed live, both jaws
# collapsing onto one point instead of spreading into a pincer). The
# correct command, confirmed by the same sweep (q2=+0.014 -> a clean
# 0.02800m/28mm separation, the full intended open aperture) is the SAME
# signed value for both joints. Original comment (2026-07-21, now
# superseded) is kept below for history:
#
# gripper_jaw2_joint's PhysxMimicJointAPI mirrors gripper_jaw1_joint with
# gearing=-1.0, offset=0.0 (confirmed by direct USD inspection, commit
# 64ab5cc / docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md):
# jaw2's physically-correct commanded position is always -1.0 * jaw1's
# commanded position, NOT the same signed value. Every open/close command
# dict across tasks/ar4/*.py (and this file's own init_state, below)
# previously assigned BOTH joints the IDENTICAL constant - a real,
# load-bearing bug discovered 2026-07-21 (ar4-franka-fixes-transfer plan,
# Task 5) when the newly-corrected, mimic-consistent jaw2 hard limits from
# 64ab5cc ([-0.014, 0.000]) correctly rejected the old +0.014 default that
# the PRE-64ab5cc (broken) jaw2 limits ([-0.0028, 0.0168]) had silently
# tolerated for as long as this constant has existed. Fixed at this single
# shared source so every call site inherits the mirrored value automatically
# instead of repeating (and risking re-introducing) the same sign bug per
# file. Controller-authorized cross-experiment fix (2026-07-21) - see
# kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md for the full
# writeup of why this may be a genuine root-cause candidate for AR4's own
# long-standing jaw-asymmetry problem (Experiments 17-22's three failed
# jaw-mimic fix attempts never diagnosed this specific sign error).
#
# scripts/build_asset.py's jaw2 hard-limit derivation was corrected to
# match (gearing flipped from -1.0 to +1.0 there too, see that file) -
# jaw2's own hard physics:lowerLimit/upperLimit now mirror jaw1's own
# [0.000, 0.014] directly, not negated, so this corrected command range
# is actually reachable rather than silently clamped.
GRIPPER_OPEN_COMMAND_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_COMMAND_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}


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
            **GRIPPER_OPEN_COMMAND_EXPR,
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

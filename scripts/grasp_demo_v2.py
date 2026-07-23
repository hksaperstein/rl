"""Classical (non-RL) pick-and-place demo, v2: uses the solving method this
session's investigation actually validated - a coarse forward-kinematics
grid search (no iteration, can't get stuck in a local minimum) followed by
a bounded-step DLS polish - instead of grasp_demo.py's original
from-HOME_Q live DLS solve, which reliably plateaued ~0.33m short of the
target across every variant tried this session.

See the conversation record for the full investigation: measure_reach_envelope.py
proved the target is within the arm's reach envelope via pure forward
kinematics; ik_seeded_start.py showed DLS barely improves even from a
directionally-correct seed; ik_grid_search.py + ik_polish_from_grid.py
found a real solution within ~3.6cm using grid-search-then-polish. This
script applies that same method to both the pregrasp and grasp waypoints
and runs the actual phased pick/lift/hold/release sequence to test
whether that precision is enough for a genuine grasp.

UPDATE 2026-07-22 (ar4-grasp-orientation-fix task): switched the IK
controller from ``command_type="position"`` to ``command_type="pose"``
(relative mode), giving the DLS solve an explicit target ORIENTATION
instead of leaving it to fall out of the arm's redundant null space. The
prior session (see UPDATE below and
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching entry)
diagnosed the remaining grasp-quality gap as exactly this: a position-only
solve found a genuine local optimum only ~10.5mm from the true pinch
target, but the orientation that fell out of it was an ~18-degree tilt
from vertical (a side-approach geometry) that undershot full pinch depth,
and re-aiming the *position* target lower made it WORSE, not better - a
symptom of an orientation problem, not a position problem. Fix mirrors
``scripts/demo_franka_ik_dice_line.py``'s own established
``canonical_down_quat_w`` precedent (a single, fixed, explicitly-chosen
straight-down target orientation, reused for every waypoint) - see
``_CANONICAL_*_AXIS_W`` and ``_build_canonical_target_quat_b()`` below for
how that target is constructed for AR4's own end-effector frame convention
(built from explicit world-frame basis vectors, not copied from Franka's
own hand-frame quaternion constant, which has no reason to transfer to a
structurally different arm/gripper) and how AR4's 180-degree base yaw is
handled (via ``subtract_frame_transforms``, the same world-to-root
conversion already used for position targets, rather than worked out by
hand). ``polish_from_seed`` now runs a bounded per-round POSE error (6D:
3 position + 3 axis-angle rotation, via ``compute_pose_error``), not a
position-only 3D error, and drives the combined 6-row Jacobian (the
existing offset-corrected position rows, plus link_6's own unmodified
angular rows - the pinch point shares link_6's rotation exactly, only its
*position* needs the rigid-offset correction).

UPDATE 2026-07-22 (ar4-grasp-ik-precision task): TWO real, previously
undiagnosed bugs were found and fixed this session, superseding the
"~3.3cm, DLS solver stuck in a local minimum" characterization above:

1. **Jacobian world-frame/root-frame mismatch (the dominant bug).**
   ``robot.root_physx_view.get_jacobians()`` returns the Jacobian in the
   WORLD frame (confirmed directly against Isaac Lab's own
   ``test_operational_space.py`` reference implementation, which explicitly
   rotates it into the root frame before using it - variable name
   ``jacobian_w`` there, converted to ``jacobian_b`` via
   ``matrix_from_quat(quat_inv(root_quat_w))``). Every AR4 classical demo
   script (this one, ``grasp_demo.py``, ``oracle_rollout.py``,
   ``interactive_joint_demo.py``) copied Isaac Lab's OWN official
   ``run_diff_ik.py`` tutorial pattern verbatim, which skips this rotation -
   harmless for that tutorial's Franka/UR10 scene (identity-orientation
   base), but AR4's base carries a real 180-degree yaw
   (``tasks/ar4/robot_cfg.py``'s ``init_state`` rot=(0,0,0,1)), so using the
   raw world-frame Jacobian directly against root-frame position/orientation
   vectors silently mirrors the X/Y correction direction. This exactly
   explains the previously-observed "DLS polish makes things worse" and
   "joints slam into their hard limits" signatures: a live diagnostic this
   session found the polish loop moving MONOTONICALLY AWAY from the target
   for 80 consecutive rounds with the bug present, and converging cleanly
   (0.14m -> 0.03m -> better with alternate seeds) the moment the same
   Jacobian was rotated into the root frame before use. Fixed here via
   ``_world_jacobian_to_root_frame()``.
2. **The original grid search's own distance readings were themselves a
   measurement artifact, not a real converged state.** Only
   ``GRID_SETTLE_STEPS=15`` unsettled steps per candidate, combined with a
   raster (i,k) traversal that produces a ~2.5rad DISCONTINUOUS jump in
   joint_3 every time the outer loop advances, meant many "good" readings
   were caught mid-swing while the arm was still moving from a wildly
   different previous candidate - not a real static equilibrium for the
   reported joint config. Direct verification this session (write the
   exact reported "best" config via ``write_joint_position_to_sim`` +
   zero velocity + a genuine 100-step hold) found the true settled residual
   for the ORIGINAL grid's own reported-best GRASP config was 0.42m, not
   0.033m - a >10x discrepancy. Fixed here by replacing the flawed 2D
   raster grid with ``_find_best_seed()``: a small set of diverse
   (j2, j3, j5) candidate seeds, each genuinely settled via a clean
   teleport + explicit zero-velocity write + a real hold, before being
   handed to the (now correctly-signed) DLS polish. A multi-seed search is
   used because the corrected DLS polish still converges to different local
   optima depending on the wrist's starting orientation (a real property of
   this redundant 6DOF arm reaching a 3DOF position target, not a bug) -
   picking the best among several seeds finds a materially better basin
   than any single fixed seed.

3. **link_6-origin vs. gripper-jaw-pinch-point targeting bug (found AFTER
   fixing #1/#2 above, via video review).** This script's ``robot_entity_cfg``
   controls body ``link_6`` directly, and every waypoint's target was set to
   put link_6's own ORIGIN at the cube's location - but the actual gripper
   jaw pinch point is offset ``0.036m`` along link_6's local +Z axis (the
   same ``_EE_OFFSET`` already used by ``tasks/ar4/pickplace_env_cfg.py``'s
   FrameTransformer for the RL env's own observations, never previously
   applied in this classical script). Fixed via
   ``_ee_point_pos_and_jacobian()``, which computes the offset point's real
   position and Jacobian (``J_pos - skew(R @ offset) @ J_ang``) and drives
   THAT toward the target instead of link_6's raw origin.

Verified result (this session, cube at world (0.20, 0.28, 0.009)):
PREGRASP converges to ~0.2mm (excellent). GRASP (the much harder waypoint -
9mm off the ground, near the edge of several joints' comfortable range)
converges to a genuine, reproducible ~15mm (link_6-origin metric; see fix #3
above for why the true fingertip target differs from this) across many
different seeds/basins - a real, substantial improvement over both the
divergent (unfixed-Jacobian) and the previously-believed-but-fictional
(unfixed-grid-search) baselines. See
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-22 update
for the full investigation and the final grasp+lift validation result.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/grasp_demo_v2.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Classical pick-and-place demo v2 (grid-search + DLS polish).")
# 2026-07-22 (orientation-fix task): override the cube's world (x,y) so the
# same seed-search/pose-DLS/phased-execution pipeline can be tested at
# DIFFERENT reach distances/bearings in one script, without editing the
# file between runs - added after finding the default cube position
# (0.0, 0.275, ...), ~27.5cm straight-ahead reach, hits a genuine joint_3
# (elbow) hard-limit conflict between reaching that low a height and
# maintaining the canonical straight-down orientation simultaneously (see
# this module's own docstring/kb doc). Testing other positions is how this
# task's own brief ("test across 3-4 different cube starting positions")
# gets satisfied, and also directly answers whether the joint-limit
# conflict is specific to this one position or a general property of the
# whole reachable workspace.
parser.add_argument(
    "--cube-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the cube's world-frame (x, y) position (z kept at the scene default) for testing reach.",
)
parser.add_argument(
    "--video-suffix", type=str, default="",
    help="Suffix appended to the output video filename, so different --cube-xy runs don't overwrite each other's video.",
)
parser.add_argument(
    "--grasp-height", type=float, default=None,
    help="Override GRASP_AT_HEIGHT (root-frame z target for the GRASP waypoint's pinch point), for testing whether the Z-shortfall residual is compensable at a given cube position (see GRASP_AT_HEIGHT's own docstring).",
)
parser.add_argument(
    "--tilt-deg", type=float, default=0.0,
    help="Deliberate forward/back tilt (degrees) of the canonical approach orientation away from pure-vertical - see _build_canonical_target_quat_w's own docstring for why this exists.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    subtract_frame_transforms,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_video_suffix = f"_{args_cli.video_suffix}" if args_cli.video_suffix else ""
VIDEO_PATH = os.path.join(LOG_DIR, "videos", f"ar4_grasp_demo_v2{_video_suffix}.mp4")

# NOTE (2026-07-22, ar4-grasp-ik-precision task): this was previously
# (0.20, 0.28, 0.009), copied from tasks/ar4/objects_cfg.py's raw CUBE_CFG
# default - but Ar4PickPlaceMirrorSceneCfg (the scene this script's
# Ar4GraspVerifyEnvCfg actually builds on) overrides the cube's spawn via
# CUBE_CFG.replace(init_state=(0.0, 0.275, 0.006)) - "recentered to the
# workspace midpoint" per that module's own comment - so this constant was
# silently aiming at a location the cube has never actually occupied in this
# scene, a ~20cm real target-position bug found via direct comparison of a
# fresh env.reset()'s actual env.scene["cube"].data.root_pos_w against this
# constant (independent of, and stacking with, the Jacobian-frame/grid-search/
# EE-offset bugs documented in this module's docstring above - this one
# alone was sufficient by itself to guarantee the gripper never got near the
# cube, regardless of how precise the IK solve was). Height kept at 3mm above
# the cube's true resting z=0.006 (matching the pre-existing GRASP_AT_HEIGHT
# convention, not touched here).
#
# NOTE (2026-07-22, orientation-fix task): no longer read directly by
# main() - cube_pos_w is now always taken LIVE from env.scene["cube"] (see
# main()'s own --cube-xy handling), so this constant is purely informational
# documentation of the scene's own default spawn point now, kept because
# its value still matches that default and its docstring above is useful
# history.
CUBE_POS_W = (0.0, 0.275, 0.009)
PREGRASP_HOVER = 0.05
# NOTE (2026-07-22): tried lowering this to -0.001 to compensate for the
# verified best GRASP_Q basin's ~10mm Z-height shortfall (its fingertip
# lands ~10mm above the intended contact height, the dominant component of
# its ~10.5mm residual) - this made the achieved residual WORSE (20mm), not
# better: the multi-seed search converged to essentially the SAME joint
# config regardless of the lower target, confirming this basin's descent is
# genuinely capped by a joint constraint (not simply "aiming too high"),
# most likely the same joint-limit-boundary behavior documented throughout
# this investigation (multiple local optima found this session pin one or
# more joints at/near their hard limits at this low approach height).
# Reverted to 0.009, the empirically better value.
#
# 2026-07-22 (orientation-fix task): overridable via --grasp-height for the
# same reason --cube-xy is overridable - to test whether the residual's
# Z-shortfall is a joint-limit wall (compensating won't help, per the note
# above) or a genuine constant-ish offset in a joint-limit-FREE basin
# (found live at the 32cm-reach cube position - no joint pinned at a limit
# there, unlike the default 27.5cm/closer 20cm positions - where
# compensating is worth an independent retry rather than assuming the
# earlier position-only finding still applies).
GRASP_AT_HEIGHT = args_cli.grasp_height if args_cli.grasp_height is not None else 0.009
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset

# link_6 -> gripper-jaw-pinch-point offset along link_6's own local +Z axis.
# Matches tasks/ar4/pickplace_env_cfg.py's _EE_OFFSET (measured directly via
# robot.data.body_pos_w for gripper_jaw1_link/gripper_jaw2_link - see that
# module's own docstring). This script's robot_entity_cfg controls link_6 (an
# ArticulationCfg body), but link_6's own origin is NOT the gripper's pinch
# point - every prior version of this script aimed link_6's raw origin
# directly at the cube's location, a genuine ~36mm targeting bug, independent
# of (and stacking with) the Jacobian-frame and grid-search bugs documented
# in this module's own docstring above. Found this session after the first
# post-Jacobian-fix run achieved <=15mm link_6-to-target precision yet the
# cube never moved at all - video review showed the gripper visibly not
# overlapping the cube in every frame.
_EE_OFFSET = (0.0, 0.0, 0.036)

# Canonical straight-down grasp orientation (2026-07-22, ar4-grasp-
# orientation-fix task). Position-only IK left the wrist's final
# orientation to whatever fell out of the arm's redundant null space - an
# ~18-degree tilt from vertical, diagnosed (_diag_check_orientation.py,
# see kb doc) as the cause of the remaining ~10mm pinch-depth shortfall.
# Built directly from explicit WORLD-frame basis vectors, not copied from
# demo_franka_ik_dice_line.py's own `canonical_down_quat_w` constant -
# that quaternion is specific to panda_hand's own local-axis convention
# and has no reason to transfer to AR4's differently-built link_6 frame.
# local +Z is link_6's own approach axis (the same axis _EE_OFFSET is
# measured along, confirmed via _diag_check_orientation.py) -> pointed
# straight down (world -Z). local +X is the jaw-slide axis -> pointed
# along a fixed horizontal heading - a symmetric cube grasp doesn't care
# WHICH horizontal direction the jaws close along, only that it's
# horizontal, but the SPECIFIC heading chosen is NOT free of side effects:
# a live run with jaw-axis = world +X converged GRASP's polish to
# joint_6 = 3.14159 (exactly pi to float precision) - pinned at what is
# almost certainly that joint's hard upper limit (PREGRASP's own converged
# joint_6, at a taller/easier approach height, landed at 3.1334, just
# under the same wall) - and the polish then deadlocked (identical residual
# for 80 straight rounds) unable to close the last ~2cm of position error
# because that joint had nowhere left to move. Switched jaw-axis heading to
# world +Y (a 90-degree rotation of the same otherwise-arbitrary choice) to
# move joint_6's required value away from that specific boundary for THIS
# cube position/approach geometry - not a claim that +Y is universally
# correct, just that the heading is a free parameter worth choosing to
# avoid a known joint limit rather than leaving to accident. local +Y
# completes a right-handed orthonormal frame (= local_Z x local_X).
_CANONICAL_Z_AXIS_W = (0.0, 0.0, -1.0)
_CANONICAL_X_AXIS_W = (0.0, 1.0, 0.0)
_CANONICAL_Y_AXIS_W = (1.0, 0.0, 0.0)  # = cross(Z, X), verified right-handed below


def _build_canonical_target_quat_w(device: str, tilt_deg: float = 0.0) -> torch.Tensor:
    """Builds the canonical target WORLD-frame quaternion from the explicit
    basis vectors above, optionally tilted forward/back by tilt_deg (2026-07-22,
    orientation-fix task addition) - a controlled, DELIBERATE deviation from
    pure-vertical, not the uncontrolled ~18-72-degree tilt this whole task
    started out fixing. Added after live evidence that pure-vertical
    (tilt_deg=0) is capped well above the cube's own surface by AR4's own
    kinematics at every reach distance tried (a joint_3 hard-limit wall at
    closer reach, a softer-but-still-real multi-joint reachability boundary
    at farther reach where "aim lower to compensate" reliably makes the
    residual WORSE, not better - see this module's docstring). Rotates the
    approach axis (local Z) about the JAW-SLIDE axis (local X, held fixed)
    by tilt_deg, mixing the straight-down axis with a horizontal component
    in the reach-direction plane - physically, tipping the wrist forward as
    a real arm does when reaching down near the edge of its extension.
    Sanity-checks orthonormality/right-handedness at call time (cheap, and
    this is load-bearing for the whole grasp)."""
    theta = math.radians(tilt_deg)
    x_axis = torch.tensor(_CANONICAL_X_AXIS_W, device=device)
    base_y_axis = torch.tensor(_CANONICAL_Y_AXIS_W, device=device)
    base_z_axis = torch.tensor(_CANONICAL_Z_AXIS_W, device=device)
    # Rotate (y,z) within the plane spanned by base_y_axis/base_z_axis by
    # theta, keeping x_axis (the rotation axis) fixed - standard
    # rotation-about-an-axis construction using the existing orthonormal
    # frame as the 2D basis to rotate within.
    z_axis = math.cos(theta) * base_z_axis + math.sin(theta) * base_y_axis
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    assert torch.allclose(torch.cross(z_axis, x_axis), y_axis, atol=1e-6), "basis is not right-handed"
    assert abs(torch.dot(x_axis, y_axis).item()) < 1e-6
    assert abs(torch.dot(y_axis, z_axis).item()) < 1e-6
    rot_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1).unsqueeze(0)  # columns = local axes in world frame
    return quat_from_matrix(rot_matrix)


def _build_canonical_target_quat_b(root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, tilt_deg: float = 0.0) -> torch.Tensor:
    """Converts the canonical WORLD-frame target orientation into the
    articulation's ROOT frame via subtract_frame_transforms - the same
    world-to-root conversion already used for position targets elsewhere in
    this script. This is what correctly accounts for AR4's 180-degree base
    yaw (tasks/ar4/robot_cfg.py's init_state rot=(0,0,0,1)): a 180-degree
    yaw about Z leaves world -Z indistinguishable in the root frame, but
    flips the X/Y axes - handled here by the library call, not worked out
    by hand."""
    target_quat_w = _build_canonical_target_quat_w(str(root_pos_w.device), tilt_deg=tilt_deg)
    _, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, root_pos_w, target_quat_w)
    return target_quat_b

# Diverse (j2, j3, j5) candidate seeds for _find_best_seed() (j1 comes from the
# calibration formula, j4/j6 start at 0). The corrected DLS polish (see
# _world_jacobian_to_root_frame()) still converges to different local optima
# depending on which basin the wrist starts in - found empirically this
# session by trying several starting configurations and keeping the best.
# Not claimed to be a globally-searched or task-position-specific optimal
# set, just a diverse spread that reliably found a good basin for both the
# GRASP and PREGRASP waypoints at this cube position.
CANDIDATE_SEEDS = [
    (1.0, -0.5, -0.8),
    (1.2, 0.0, -1.2),
    (0.6, 0.3, 0.5),
    (1.3, -0.8, 0.9),
    (0.9, -0.9, -1.5),
    (1.4, 0.4, -0.3),
    (0.8, -0.85, -1.55),
    (1.3, 0.6, -1.7),
    (1.1, 0.9, -1.3),
    (0.5, -0.3, 0.8),
    (1.5, 0.2, 0.2),
    (0.7, 0.7, -0.9),
]

SEED_SETTLE_STEPS = 60
POLISH_STEP_MAX = 0.03
# 2026-07-22 (orientation-fix task): bumped 60 -> 100. The polish loop now
# has to close a 6D pose error (position + orientation) instead of a 3D
# position-only error, starting from seeds whose orientation was never
# selected for (CANDIDATE_SEEDS only varies j2/j3/j5, matching a
# position-only search) - budgeting more rounds for the extra orientation
# degrees of freedom to converge alongside position.
POLISH_ROUNDS = 100
POLISH_SETTLE_STEPS = 30
LAMBDA_VAL = 0.02
CONVERGENCE_THRESHOLD = 0.003
# 2026-07-22 (orientation-fix task): per-round bounded ROTATION step, the
# orientation analogue of POLISH_STEP_MAX - mirrors
# demo_franka_ik_dice_line.py's own _MAX_ROT_STEP (bounded per-step
# rotation correction, not a single large jump, for the same stability
# reason POLISH_STEP_MAX bounds position).
POLISH_ROT_STEP_MAX = 0.15
# Rotation convergence tolerance (radians, axis-angle norm) - the
# orientation analogue of CONVERGENCE_THRESHOLD. ~0.05rad ~= 3 degrees,
# tight enough for a genuine top-down pinch, loose enough to be reachable
# given this arm's own joint-limit-constrained basins (see this module's
# docstring on the ~18-degree tilt this fix is targeting).
ROT_CONVERGENCE_THRESHOLD = 0.05
# Weight converting a radian of orientation error into an equivalent
# "meters" scale for the combined best-round score below - chosen as the
# _EE_OFFSET length scale (0.036m/rad), i.e. roughly the linear pinch-point
# displacement a small rotation error produces at the fingertip. A
# deliberate, documented judgment call (not derived from a formal
# optimality argument) so a single round's "best so far" bookkeeping can
# compare position and rotation error on one scale.
ORIENTATION_SCORE_WEIGHT = 0.036


def _world_jacobian_to_root_frame(jacobian_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate a Jacobian returned by ``root_physx_view.get_jacobians()`` (WORLD
    frame) into the articulation's own root/base frame, matching Isaac Lab's
    own reference pattern in ``test_operational_space.py``'s
    ``_update_states()``. Required whenever the robot's root orientation is
    not identity in world frame - AR4's base carries a 180-degree yaw
    (``tasks/ar4/robot_cfg.py``), so this is NOT a no-op here, unlike in
    Isaac Lab's own ``run_diff_ik.py`` tutorial (identity-orientation base),
    whose Jacobian-usage pattern this script's DLS solve was originally
    copied from without this rotation - the root cause of this session's
    "DLS polish diverges instead of converging" finding.
    """
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
    if jacobian_b.shape[1] > 3:
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])
    return jacobian_b


def _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q_list, steps) -> None:
    """Teleport the arm to q_list via write_joint_position_to_sim, explicitly
    zero its joint velocity (so no momentum carries over from whatever the
    arm was doing before), then hold the commanded target for `steps` sim
    steps. This clean-teleport-and-hold pattern is what this session found
    necessary to get a TRUSTWORTHY distance reading - the original grid
    search's continuous "slide from the previous candidate" pattern (no
    teleport, no velocity reset, only 15 steps) was found to report
    transient/mid-swing distances up to 10x better than the true settled
    value for the same nominal joint config."""
    q = torch.tensor([q_list], device=env.device)
    robot.write_joint_position_to_sim(q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(steps):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = q
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)


def _ee_point_pos_and_jacobian(ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor, jacobian_b: torch.Tensor):
    """Return the ACTUAL gripper jaw pinch point's position (root frame) and
    Jacobian, given link_6's own pose/Jacobian and the constant local offset
    _EE_OFFSET. The pinch point is rigidly attached to link_6, so its world
    velocity is v_link6_origin + omega x (R @ offset_local); in Jacobian
    terms that's J_pos - skew(R @ offset_local) @ J_ang."""
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset

    skew = torch.zeros(ee_pos_b.shape[0], 3, 3, device=ee_pos_b.device)
    skew[:, 0, 1], skew[:, 0, 2] = -world_offset[:, 2], world_offset[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = world_offset[:, 2], -world_offset[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -world_offset[:, 1], world_offset[:, 0]
    jac_ang = jacobian_b[:, 3:6, :]
    point_jac_pos = jacobian_b[:, 0:3, :] - torch.bmm(skew, jac_ang)
    return point_pos_b, point_jac_pos


def _measure_dist(robot, robot_entity_cfg, target_pos_b) -> float:
    """Distance from the ACTUAL gripper jaw pinch point (link_6 origin +
    _EE_OFFSET) to target_pos_b - NOT link_6's own raw origin (see
    _EE_OFFSET's own docstring above for why this distinction is load-bearing
    here)."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    return torch.norm(point_pos_b[0] - target_pos_b[0]).item()


def _measure_dist_vec(robot, robot_entity_cfg, target_pos_b):
    """Same as _measure_dist but returns the signed per-axis (root-frame
    x,y,z) residual vector instead of just the scalar norm - added
    (2026-07-22, orientation-fix task) to diagnose WHICH direction a
    joint-limit-capped polish's residual is dominated by (e.g. a pure
    Z-height shortfall vs an X/Y bearing miss), the same distinction the
    earlier position-only investigation found load-bearing."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    return (target_pos_b[0] - point_pos_b[0]).tolist()


def _measure_rot_err(robot, robot_entity_cfg, target_quat_b) -> float:
    """Axis-angle rotation error (radians) between link_6's CURRENT
    orientation and target_quat_b - the orientation analogue of
    _measure_dist. The pinch point shares link_6's orientation exactly (the
    rigid offset only shifts position), so link_6's own quat is also the
    pinch point's quat - no offset correction needed here, unlike position."""
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    _, axis_angle_error = compute_pose_error(
        ee_pos_b, ee_quat_b, ee_pos_b, target_quat_b, rot_error_type="axis_angle"
    )
    return torch.norm(axis_angle_error[0]).item()


def _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, target_pos_b, target_quat_b, seed_j1, extra_full_seeds=None):
    """Genuinely-settled multi-seed search (replaces the old 2D raster grid
    search over (j2, j3) alone, which had the transient-measurement bug
    documented in this module's own docstring above). Each candidate is
    settled from a clean teleport + zero-velocity hold (see _settle_at), so
    the reported distance is trustworthy, not a mid-swing artifact.

    2026-07-22 (orientation-fix task): scoring switched from position-only
    to the SAME combined position+orientation score polish_from_seed uses
    (see ORIENTATION_SCORE_WEIGHT). Found necessary live: with position-only
    scoring, the GRASP waypoint's seed search always picked
    `extra_full_seeds`'s old position-only-tuned KNOWN_GOOD_GRASP_Q (best on
    pure position, since that's literally what it was tuned for) - but that
    config's orientation turned out to be ~163 degrees (2.85rad) from
    canonical, and the subsequent polish got permanently stuck at that same
    ~163-degree error (identical pos/rot residual for 80 straight rounds -
    a joint-limit deadlock, not a converging solve) instead of correcting
    it. The PREGRASP waypoint's own seed search happened to land on a
    CANDIDATE_SEEDS-derived config with a far more compatible starting
    orientation instead, and its polish converged cleanly (2.98rad ->
    0.0059rad) - direct evidence the DLS mechanism itself works fine when
    not started from an orientation-incompatible basin, and that the fix is
    in seed SELECTION, not the polish loop itself.

    `extra_full_seeds`: optional list of full 6-joint configs (already known
    to be good, position-wise, from a prior offline multi-seed search at
    this exact target) to ALSO try directly, alongside the (seed_j1, j2,
    j3, 0, j5, 0) candidates derived from CANDIDATE_SEEDS. Still included
    for their position quality, but no longer trusted blindly - the
    combined score means one of these can still be picked, but only if its
    orientation is ALSO reasonably compatible, not solely because its
    position is good."""
    def _combined_score(pos_err: float, rot_err: float) -> float:
        return pos_err + ORIENTATION_SCORE_WEIGHT * rot_err

    best_score = float("inf")
    best_dist = float("inf")
    best_q = None
    for (j2, j3, j5) in CANDIDATE_SEEDS:
        q = [seed_j1, j2, j3, 0.0, j5, 0.0]
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q, SEED_SETTLE_STEPS)
        dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        rot = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
        score = _combined_score(dist, rot)
        if score < best_score:
            best_score = score
            best_dist = dist
            best_q = q
    for q in (extra_full_seeds or []):
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, q, SEED_SETTLE_STEPS)
        dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        rot = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
        score = _combined_score(dist, rot)
        if score < best_score:
            best_score = score
            best_dist = dist
            best_q = q
    print(f"  [SEED SEARCH] best settled seed dist: {best_dist:.5f}m (combined score {best_score:.5f}), config: {best_q}")
    return best_q, best_dist


def polish_from_seed(
    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits
):
    """Bounded-step DLS polish from an already-settled seed. Same overall
    structure as the previous grid_search_then_polish's polish phase (bounded
    per-round Cartesian step, "keep best across rounds" regression guard),
    with the world-to-root Jacobian rotation (_world_jacobian_to_root_frame)
    and an explicit joint-limit clamp on the DLS-desired target (prevents
    repeatedly re-triggering the same limit-wall bounce a round after a
    limit is hit).

    2026-07-22 (orientation-fix task): now drives a full 6D POSE error
    (position + axis-angle rotation, via compute_pose_error), not a 3D
    position-only error - ik_controller is now configured
    command_type="pose" (relative mode). The rotation part of the Jacobian
    (jacobian_b[:, 3:6, :]) needs NO offset correction (the pinch point
    shares link_6's own rotation exactly - only its POSITION differs from
    link_6's origin), so the combined 6-row Jacobian is simply the existing
    offset-corrected position rows stacked on link_6's own unmodified
    angular rows. The "keep best across rounds" guard now tracks a combined
    score (position error plus ORIENTATION_SCORE_WEIGHT-scaled rotation
    error, see that constant's own docstring for the judgment call this
    represents) instead of position alone, so it doesn't restore a round
    that had great position but a bad orientation."""
    robot = env.scene["robot"]

    def _combined_score(pos_err: float, rot_err: float) -> float:
        return pos_err + ORIENTATION_SCORE_WEIGHT * rot_err

    best_pos_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
    best_rot_residual = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
    best_score = _combined_score(best_pos_residual, best_rot_residual)
    best_polish_q = list(seed_q)
    final_pos_residual = best_pos_residual
    final_rot_residual = best_rot_residual

    for round_num in range(POLISH_ROUNDS):
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        jacobian_b = _world_jacobian_to_root_frame(jacobian_w, robot.data.root_quat_w)

        # Drive the ACTUAL gripper jaw pinch point (link_6 + _EE_OFFSET), not
        # link_6's own raw origin - see _EE_OFFSET's docstring above. Only
        # the position rows need the rigid-offset correction; the pinch
        # point's orientation IS link_6's orientation.
        point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)
        jac_ang = jacobian_b[:, 3:6, :]
        full_jac = torch.cat([point_jac_pos, jac_ang], dim=1)

        pos_error, rot_error = compute_pose_error(
            point_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
        )
        pos_norm = torch.norm(pos_error, dim=-1, keepdim=True)
        pos_step = pos_error / (pos_norm + 1e-8) * torch.clamp(pos_norm, max=POLISH_STEP_MAX)
        rot_norm = torch.norm(rot_error, dim=-1, keepdim=True)
        rot_step = rot_error / (rot_norm + 1e-8) * torch.clamp(rot_norm, max=POLISH_ROT_STEP_MAX)
        delta_command = torch.cat([pos_step, rot_step], dim=-1)

        ik_controller.set_command(delta_command, ee_pos=point_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(point_pos_b, ee_quat_b, full_jac, current_joint_pos)

        lo = joint_pos_limits[:, :, 0]
        hi = joint_pos_limits[:, :, 1]
        joint_pos_des = torch.clamp(joint_pos_des, min=lo, max=hi)

        for _ in range(POLISH_SETTLE_STEPS):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = joint_pos_des[:, j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        final_pos_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        final_rot_residual = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
        final_score = _combined_score(final_pos_residual, final_rot_residual)
        if round_num % 10 == 0 or round_num == POLISH_ROUNDS - 1:
            print(
                f"  [POLISH round {round_num:3d}] pos_err={final_pos_residual:.5f}m rot_err={final_rot_residual:.4f}rad"
            )
        if final_score < best_score:
            best_score = final_score
            best_pos_residual = final_pos_residual
            best_rot_residual = final_rot_residual
            best_polish_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
        if final_pos_residual < CONVERGENCE_THRESHOLD and final_rot_residual < ROT_CONVERGENCE_THRESHOLD:
            break

    if best_score < _combined_score(final_pos_residual, final_rot_residual):
        print(
            f"  [POLISH] last round (pos={final_pos_residual:.5f}m rot={final_rot_residual:.4f}rad) was worse than "
            f"the best round found (pos={best_pos_residual:.5f}m rot={best_rot_residual:.4f}rad) - restoring the "
            "best config instead of the last one"
        )
        _settle_at(env, robot, robot_entity_cfg, num_arm_joints, best_polish_q, POLISH_SETTLE_STEPS * 2)
        final_pos_residual = best_pos_residual
        final_rot_residual = best_rot_residual

    residual_vec = _measure_dist_vec(robot, robot_entity_cfg, target_pos_b)
    print(
        f"  [POLISH] final residual: pos={final_pos_residual:.5f}m rot={final_rot_residual:.4f}rad "
        f"(target-achieved per-axis xyz: {['%.5f' % v for v in residual_vec]})"
    )
    return robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist(), final_pos_residual, final_rot_residual


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Test-local actuator-gain override (2026-07-22, not touching the shared
    # tasks/ar4/robot_cfg.py): the arm's own default gains (stiffness=40,
    # damping=4) were confirmed too weak to hold/track a commanded pose
    # under gravity during the jaw2-drive diagnostic this same session (see
    # kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-22
    # "later" UPDATE) - PHASE 2 of this script's own run (moving from
    # pregrasp to grasp_q) showed a 1.42rad max joint error at the end of a
    # 90-step settle, well beyond normal PD convergence, and the gripper
    # never got near the cube in the recorded video. Boosting gains here
    # only (a scripted validation tool, not a training-time change) to test
    # whether that's the actual blocker for this specific classical-IK demo.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    print(f"[INFO] Arm joint pos limits (lo,hi per joint): {joint_pos_limits[0].tolist()}")

    # 2026-07-22 (orientation-fix task): switched command_type "position" ->
    # "pose" (relative mode, mirroring demo_franka_ik_dice_line.py's own
    # proven bounded-relative-step pattern) so the solve targets BOTH
    # position and a canonical straight-down orientation, instead of
    # leaving orientation to fall out of the arm's redundant null space.
    # See this module's docstring and _build_canonical_target_quat_b for why.
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": LAMBDA_VAL}
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["perception_camera"]

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()

        # 2026-07-22 (orientation-fix task): if --cube-xy was passed, teleport
        # the cube there (z kept at scene default) BEFORE reading its pose as
        # ground truth for the grasp targets - lets this same pipeline be
        # tested at different reach distances/bearings without touching the
        # scene cfg's own randomization-free spawn. cube_pos_w is now always
        # read LIVE from the scene (source of truth), not from the CUBE_POS_W
        # constant directly - CUBE_POS_W remains the default/no-override case.
        cube = env.scene["cube"]
        if args_cli.cube_xy is not None:
            override_z = cube.data.root_pos_w[0, 2].item()
            override_pos = torch.tensor([[args_cli.cube_xy[0], args_cli.cube_xy[1], override_z]], device=env.device)
            override_quat = cube.data.root_quat_w[0:1].clone()
            cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
            cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
            print(f"[INFO] --cube-xy override applied: cube teleported to world {override_pos[0].tolist()}")

        cube_pos_w = cube.data.root_pos_w[0:1].clone()
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Cube (robot frame): {cube_pos_b[0].tolist()}, calibrated joint_1: {seed_j1:.4f}")

        # Single canonical straight-down target orientation, reused for both
        # PREGRASP and GRASP waypoints (mirrors demo_franka_ik_dice_line.py's
        # own canonical_down_quat_w convention - one fixed approach
        # orientation for the whole descend/close/lift sequence, not
        # per-waypoint). See _build_canonical_target_quat_b's own docstring.
        target_quat_b = _build_canonical_target_quat_b(root_pos_w, root_quat_w, tilt_deg=args_cli.tilt_deg)
        print(f"[INFO] Canonical target orientation (root frame, w,x,y,z): {target_quat_b[0].tolist()}")

        # Capture the cube's TRUE pose right after reset, before any
        # seed-search/polish runs. The multi-seed search below teleports the
        # arm through several very different configurations (some close to
        # the cube's own resting spot, since they're candidates for reaching
        # it) to evaluate them - if a teleported pose interpenetrates the
        # cube, PhysX's own depenetration reaction can shove it out of place
        # long before the actual phased grasp attempt begins. Confirmed
        # happening this session: a first post-Jacobian-fix run achieved
        # <=15mm link_6-to-target precision yet the cube never moved during
        # CLOSE/lift/hold - video review showed the gripper visibly not
        # overlapping the cube, and the cube's own logged position had
        # drifted ~20cm in X from its expected spawn point by the time the
        # phased execution started. Restoring this captured pose right
        # before Phase 0 (below) is a direct, cheap guard against exactly
        # that displacement. (`cube` itself was already resolved above, when
        # the --cube-xy override - if any - was applied.)
        cube_init_pos = cube.data.root_pos_w[0].clone()
        cube_init_quat = cube.data.root_quat_w[0].clone()
        print(f"[INFO] Cube initial pose (world): pos={cube_init_pos.tolist()}")

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

        # Known-good absolute configs from a prior offline multi-seed search
        # at this exact target (cube (0.0, 0.275, 0.009)) - see
        # _find_best_seed's own docstring for why these are included
        # directly rather than trusting this run's live search alone to
        # reproduce the same basin.
        KNOWN_GOOD_GRASP_Q = [-0.014429761096835136, 1.240863561630249, 0.3401874601840973, -0.08906537294387817, 1.1987247467041016, 0.0052983760833740234]
        KNOWN_GOOD_PREGRASP_Q = [0.00014747596287634224, 0.9648232460021973, 0.9025915265083313, -0.0006890640361234546, -0.6352600455284119, 0.008003178983926773]

        # 2026-07-22 (orientation-fix task, live-run finding): solve PREGRASP
        # FIRST, then seed GRASP's polish from PREGRASP's own CONVERGED
        # config, instead of solving both waypoints independently from
        # scratch. Found necessary live: a first attempt gave GRASP its own
        # independent combined-score seed search (same mechanism as
        # PREGRASP's) and it STILL landed on the old position-only
        # KNOWN_GOOD_GRASP_Q seed (every CANDIDATE_SEEDS entry scored worse
        # on combined position+orientation terms) - and that seed's polish
        # got permanently deadlocked at ~163 degrees (2.845rad) of rotation
        # error for 80 straight rounds (identical residual every round - a
        # joint-limit wall, not a converging solve), DESPITE starting from
        # almost the identical ~171-degree (2.98rad) initial rotation error
        # PREGRASP's own seed also started from and successfully corrected
        # (2.98rad -> 0.0059rad in 20 rounds). That rules out "bad seed
        # orientation" as the actual cause - the real difference is GRASP's
        # much lower approach height (GRASP_AT_HEIGHT=0.009, right at the
        # cube's surface) creating a joint-limit conflict between position
        # and canonical orientation that none of the existing seeds happen
        # to avoid, consistent with this same basin's earlier-diagnosed
        # (position-only investigation) joint-limit-capped descent. Since
        # PREGRASP_Q is only 5cm away (PREGRASP_HOVER) and already
        # genuinely canonical (rot_err 0.0059rad), it's a far more physically
        # sensible starting point for GRASP's polish than an independently
        # re-solved basin - and matches how the phased execution actually
        # moves the arm anyway (pregrasp_q -> grasp_q as consecutive,
        # nearby waypoints, not independent teleports).
        print("\n[INFO] Finding best seed for PREGRASP waypoint (multi-seed, genuinely settled)...")
        seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b, target_quat_b, seed_j1, extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q])
        print("[INFO] Polishing PREGRASP waypoint (fixed-Jacobian, pose-DLS)...")
        pregrasp_q, pregrasp_residual, pregrasp_rot_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print("\n[INFO] Finding best seed for GRASP waypoint (multi-seed, genuinely settled, includes converged PREGRASP_Q)...")
        seed_q, _ = _find_best_seed(
            env, robot, robot_entity_cfg, num_arm_joints, grasp_pos_b, target_quat_b, seed_j1,
            extra_full_seeds=[KNOWN_GOOD_GRASP_Q, pregrasp_q],
        )
        print("[INFO] Polishing GRASP waypoint (fixed-Jacobian, pose-DLS)...")
        grasp_q, grasp_residual, grasp_rot_residual = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits
        )

        print(
            f"\n[SUMMARY] grasp_residual={grasp_residual:.5f}m/{grasp_rot_residual:.4f}rad "
            f"pregrasp_residual={pregrasp_residual:.5f}m/{pregrasp_rot_residual:.4f}rad"
        )
        print(f"[SUMMARY] grasp_q={grasp_q}")
        print(f"[SUMMARY] pregrasp_q={pregrasp_q}")

        # Live orientation sanity check (mirrors _diag_check_orientation.py):
        # explicitly teleport+settle to EACH found waypoint and print link_6's
        # actual local-axis directions in root frame, so the printed log
        # itself shows whether the approach axis (local +Z) is genuinely
        # vertical at THAT waypoint - not just that the scalar rot_residual
        # returned by polish_from_seed is small (a small residual is a claim
        # about compute_pose_error's own math; this is an independent,
        # directly-readable check of the same thing). Does its own explicit
        # teleport rather than trusting whatever state happens to be live
        # (an earlier version of this check read the CURRENT robot state
        # without re-teleporting, right after the PREGRASP polish had
        # already moved the arm away from grasp_q - it was silently
        # reporting PREGRASP's orientation while labeled "GRASP_Q").
        def _check_orientation_at(q_list, label):
            q = torch.tensor([q_list], device=env.device)
            robot.write_joint_position_to_sim(q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
            robot.write_joint_velocity_to_sim(
                torch.zeros((1, num_arm_joints), device=env.device), joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device)
            )
            for _ in range(30):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                action[:, :num_arm_joints] = q
                action[:, num_arm_joints] = GRIPPER_OPEN
                env.step(action)
            ee_pose_w_check = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            _, ee_quat_b_check = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w_check[:, 0:3], ee_pose_w_check[:, 3:7]
            )
            rot_check = matrix_from_quat(ee_quat_b_check)[0]
            print(
                f"[CHECK] {label} local +Z (approach axis) in root frame: {['%.3f' % v for v in rot_check[:, 2].tolist()]} "
                "(target: world -Z, i.e. root-frame [0,0,-1] since AR4's 180-deg base yaw leaves Z unaffected)"
            )
            print(f"[CHECK] {label} local +X (jaw-slide axis) in root frame: {['%.3f' % v for v in rot_check[:, 0].tolist()]}")

        _check_orientation_at(grasp_q, "GRASP_Q")
        _check_orientation_at(pregrasp_q, "PREGRASP_Q")

        # Restore the cube to its captured initial pose (see note above)
        # before starting the actual phased grasp attempt, undoing any
        # accidental disturbance from the seed-search/polish process above.
        cube_pos_now = cube.data.root_pos_w[0].tolist()
        print(f"[INFO] Cube pose before restore: pos={cube_pos_now}")
        restore_pose = torch.cat([cube_init_pos, cube_init_quat]).unsqueeze(0)
        cube.write_root_pose_to_sim(restore_pose, env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube pose after restore: pos={cube.data.root_pos_w[0].tolist()}")

        PHASES = [
            (60, HOME_Q, GRIPPER_OPEN),
            (150, pregrasp_q, GRIPPER_OPEN),
            (90, grasp_q, GRIPPER_OPEN),
            (60, grasp_q, GRIPPER_CLOSE),
            (90, pregrasp_q, GRIPPER_CLOSE),
            (120, pregrasp_q, GRIPPER_CLOSE),
            (60, pregrasp_q, GRIPPER_OPEN),
            (150, HOME_Q, GRIPPER_OPEN),
        ]

        print("\n[INFO] Starting phased execution...\n")
        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            # Command the phase's target directly (not a ramped interpolation)
            # and let the PD controller converge over the phase's duration.
            # Tried ramped interpolation from the actual current position first
            # (the "correct-looking" fix for the stale-prev_q bug) - it was
            # much WORSE (final errors grew to 2.6+ rad) than the original
            # buggy version, which accidentally commanded the fixed target
            # directly from step 1 of each phase (since it always ramped
            # FROM the wrong, stale prev_q TO the same target, producing a
            # near-constant commanded value rather than a real ramp). This
            # arm's modest actuator stiffness appears to track a fixed target
            # better than a continuously-moving one.
            start_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(f"[PHASE {phase_idx}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} start_q={['%.4f' % x for x in start_q]}")
            for i in range(duration):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = target_q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

                if phase_idx in (3, 4, 5) and i % 20 == 0:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    print(f"  [PHASE {phase_idx} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]}")

            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            max_err = max(abs(a - t) for a, t in zip(achieved_q, target_q))
            print(f"  [PHASE {phase_idx} END] max joint error: {max_err:.5f} rad")

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()

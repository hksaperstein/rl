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
import random
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
parser.add_argument(
    "--lambda-val", type=float, default=None,
    help="Override LAMBDA_VAL (DLS damping factor) - added (2026-07-22, ar4-tilt-fix task) to test whether more damping avoids the near-singular-Jacobian instability found at a tilted GRASP target (a huge single-step joint_4 swing, ~1.1rad, coinciding with the polish falling into a bad, stuck basin) without editing the file between runs.",
)
parser.add_argument(
    "--num-descent-steps", type=int, default=30,
    help="Number of incremental sub-waypoints for the PREGRASP->GRASP height descent (2026-07-22, ar4-grasp-descent-continuity task). Instead of solving GRASP as an independent one-shot target from a multi-seed search (which the prior session found deadlocks at ~1.1-1.4rad rotation error, a stable tilt/damping/seed-independent local optimum), interpolate the target height from PREGRASP's own converged height down to GRASP_AT_HEIGHT in this many steps, re-solving polish_from_seed at each sub-height WITHOUT re-teleporting in between (chaining calls this way is what makes each sub-step start from the previous one's genuinely converged live state, mirroring Experiment 11's RL-driven incremental IK and demo_franka_ik_dice_line.py's own per-step continuous resolve). Set to 1 to reproduce the old one-shot independent-GRASP-target behavior for direct comparison.",
)
# 2026-07-22 (ar4-grasp-z-envelope task): the descent-continuity task above
# left an explicit follow-up - "directly sweep the reachable Z-height
# envelope at this XY position ... to map exactly how low this basin can
# genuinely descend before the position residual starts growing, and
# cross-reference against each joint's own live margin at that specific
# height". Both new flags below run their sweep in a SINGLE Isaac Sim
# launch (avoiding per-height/per-bearing app-startup overhead) and exit
# before the one-shot GRASP solve / phased pick-and-place execution - they
# are envelope-mapping diagnostics, not grasp attempts.
parser.add_argument(
    "--z-sweep", type=float, nargs="+", default=None,
    help="Sweep multiple GRASP-waypoint target heights (root-frame z, meters) at the CURRENT --cube-xy/--tilt-deg bearing, via the same incremental-descent method used for the real GRASP waypoint. PREGRASP is solved once as usual; each swept height then settles back to PREGRASP's own converged config first (so every sweep point starts from an identical place, not compounding drift from the previous point) before descending via --num-descent-steps sub-waypoints from PREGRASP's height down to that swept height. Logs the final position/rotation residual, the per-axis (x,y,z) residual, and the full joint config against robot.data.joint_pos_limits for every swept height - this is the direct evidence for whether the Z-shortfall is a hard cliff at one height or a smoothly growing shortfall, and which specific joint(s) are pinned/near-pinned when it appears. Exits after the sweep (skips the one-shot GRASP solve and phased pick execution).",
)
parser.add_argument(
    "--bearing-sweep", type=float, nargs="+", default=None,
    help="Sweep multiple cube BEARINGS (degrees; 0 = the scene's own default straight-ahead position at the given --bearing-sweep-radius, positive rotates toward world +X) at a FIXED reach radius, instead of varying reach distance (already tested in an earlier session) or tilt (also already tested). For each bearing, teleports the cube to that angle/radius, then runs the full seed-search + PREGRASP polish + incremental descent to --grasp-height (default 0.009, the true cube grasp point) and logs the same residual/joint-margin data as --z-sweep. Tests whether the Z-height reachability floor found at the default bearing is a property of THIS specific approach direction (a different bearing might relieve the joint_3-vs-vertical-orientation conflict) or holds at every angle (a genuine, direction-independent kinematic limit). Overrides/ignores --cube-xy when set. Exits after the sweep.",
)
parser.add_argument(
    "--bearing-sweep-radius", type=float, default=0.275,
    help="Reach radius (meters, world-frame distance from the cube to the robot base) used by --bearing-sweep. Defaults to 0.275, the scene's own default cube distance (matches the (0.0, 0.275) default cube position at bearing=0), a distance already confirmed reachable at the default bearing.",
)
# 2026-07-23 (ar4-grasp-position-search task): the prior session's bearing
# sweep held reach radius FIXED at 0.275m and swept angle only, finding a
# bearing-independent ~19mm Z-shortfall. That same session's earlier
# (separate, one-shot, non-sweep) 3-point reach-distance test found the
# shortfall SHRINKS as reach grows (20cm: 4.6cm worse: 27.5cm: 2.8cm;
# 32cm: 2.0cm, no single joint pinned) but never tested past 32cm or
# cross-referenced joint_3's own live margin (only the z-sweep/bearing-sweep
# machinery logs that) at any reach beyond the scene default. This sweep
# closes that gap directly: fixed bearing (straight-ahead, world +Y, matching
# the scene's own default direction), varying reach radius, at the TRUE
# --grasp-height (0.009 default) - looking for where joint_3's margin
# becomes clearly healthy (well above the ~0.08-0.13rad range already seen
# at 20-32cm), not just numerically less bad. Mirrors --bearing-sweep's own
# structure/logging exactly (same per-point seed-search + PREGRASP polish +
# incremental descent), just varying radius instead of angle.
parser.add_argument(
    "--radius-sweep", type=float, nargs="+", default=None,
    help="Sweep multiple cube REACH RADII (meters, world-frame distance from the cube to the robot base) at a FIXED bearing (0 degrees = straight-ahead, world +Y - the scene's own default approach direction), at --grasp-height (default 0.009, the true grasp point). For each radius, teleports the cube to (0, radius) and runs the full seed-search + PREGRASP polish + incremental descent, logging the same residual/per-axis/joint-margin data as --z-sweep/--bearing-sweep. Added after the z-envelope/bearing-sweep investigation only tested reach distances up to 32cm (already shown to have a smaller shortfall than the 27.5cm default) - this sweeps a wider range to find where joint_3's own margin becomes clearly healthy at the true grasp height, not just less-bad. Overrides/ignores --cube-xy when set. Exits after the sweep.",
)
# 2026-07-23 (ar4-capstone-grasp task): the 2026-07-23 (later) radius-sweep
# found TWO qualitatively different comfortable-joint_3-margin basins
# (0.30m: joint_4 far from its own limit; 0.39-0.42m: joint_4 near-pi, also
# far from ITS own limit) where the ~18mm Z-shortfall nonetheless persists
# unchanged - but that same session never tried a deliberate, non-zero tilt
# AT any of these newly-found comfortable positions (prior tilt tests were
# only run at the joint_3-TIGHT-margin 27.5cm/32cm positions, where the
# elbow had little room left to accommodate the extra reach a tilt demands).
# This flag tests exactly that untested combination: multiple tilt angles,
# all at the SAME fixed cube position (whatever --cube-xy/the scene default
# resolves to), in a single Isaac Sim launch (mirrors --z-sweep/--bearing-
# sweep/--radius-sweep's own single-launch-multi-point structure). Position
# is held fixed (unlike --radius-sweep/--bearing-sweep) since tilt is the
# only free variable this sweep is testing; run this AT one of the already-
# confirmed comfortable-margin radii via --cube-xy 0 <radius> if that's the
# combination being tested.
parser.add_argument(
    "--tilt-sweep", type=float, nargs="+", default=None,
    help="Sweep multiple --tilt-deg values (degrees) at the CURRENT fixed cube position (--cube-xy override, or the scene default if not given), at --grasp-height (default 0.009). For each tilt angle, rebuilds the canonical target orientation with that tilt and runs the full seed-search + PREGRASP polish + incremental descent, logging the same residual/per-axis/joint-margin data as the other sweep flags. Added to test whether a moderate tilt resolves the Z-shortfall specifically AT one of the comfortable-joint-margin positions the radius-sweep found (untested combination as of 2026-07-23) - prior tilt tests were only run at joint-margin-TIGHT positions. Exits after the sweep.",
)
# 2026-07-22/23 (ar4-grasp-deployability-check task, coordinator-directed):
# `_find_best_seed` teleports the robot (via write_joint_position_to_sim)
# through several candidate configs and picks the best-scoring one BEFORE
# ever running the real DLS resolve - a real AR4 can never do this (there
# is no "try a config, see how close it is, undo, try another" operation on
# physical hardware). This flag swaps that teleport-based search for
# `_wiggle_and_resolve` (see its own docstring): starts from wherever the
# robot's live state actually is (after env.reset(), that's HOME_Q -
# tasks/ar4/robot_cfg.py's own init_state, all-zero joint_pos - a real,
# natural starting pose, not a special case invented for this flag) and, if
# the direct continuous DLS resolve doesn't converge, retries from a SMALL,
# BOUNDED joint perturbation reached via normal commanded-target PD motion
# (env.step, matching every other real move in this script) - never a
# teleport. Only affects PREGRASP's own seeding (the only place in the
# CURRENTLY-DEFAULT --num-descent-steps>1 pipeline that ever calls
# _find_best_seed at all - the incremental GRASP descent already reuses
# PREGRASP's own converged config with no separate seed search, so it was
# already teleport-free before this flag existed).
parser.add_argument(
    "--deployable-seed", action="store_true",
    help="Replace _find_best_seed's teleport-based PREGRASP seed search with _wiggle_and_resolve's bounded, PD-driven local retry (no teleportation anywhere) - tests whether the already-fixed continuous-DLS-resolve/interpolated-descent pipeline still converges without a simulation-only search mechanism.",
)
parser.add_argument(
    "--max-wiggles", type=int, default=6,
    help="Max number of bounded local perturbation retries for --deployable-seed's _wiggle_and_resolve before giving up and reporting whatever it last converged to.",
)
parser.add_argument(
    "--wiggle-max-rad", type=float, default=0.3,
    help="Max per-joint perturbation magnitude (radians, ~17 degrees at the default) for --deployable-seed's wiggle retries - deliberately small/bounded (a real jog/dither move, not a big jump) per the coordinator's own framing of what makes this physically realizable on real hardware.",
)
# 2026-07-22/23 (ar4-grasp-deployability-check task, coordinator-directed,
# follow-up after --deployable-seed's own bounded wiggle retries were found
# NOT to escape a ~1.0-1.4rad rotation-error basin from HOME_Q at any of 6
# attempts): a real robot CAN make one large, DELIBERATE commanded move to a
# specific known/precomputed reference posture (that's an ordinary PD-driven
# joint move, not a teleport) - it just cannot cheaply "try many candidates
# and roll back" the way write_joint_position_to_sim does. This flag tests
# the coordinator's own suggested fallback ("a smarter single initial guess
# ... rather than either teleporting or wiggling"): before any DLS resolve
# attempt, commands a genuine PD move (env.step loop, held for
# SEED_SETTLE_STEPS, NO write_joint_position_to_sim anywhere) from wherever
# the robot currently is to KNOWN_GOOD_PREGRASP_Q - a fixed reference
# posture already independently established across several earlier sessions
# (and consistent with what the teleport search itself usually converges
# near for this cube height/bearing) - then proceeds with the normal
# --deployable-seed resolve/wiggle logic from THAT starting point instead of
# from raw HOME_Q.
parser.add_argument(
    "--fixed-posture-move", action="store_true",
    help="With --deployable-seed: before resolving, make one real PD-driven move (no teleport) to the fixed KNOWN_GOOD_PREGRASP_Q reference posture, then resolve/wiggle from there instead of from raw HOME_Q.",
)
# 2026-07-24 (ar4-grasp-ik-convergence-tightening task): the best-known
# configuration to date (65deg tilt, reach 0.30-0.36m) converges to a
# genuine ~9.5mm/~4.2deg residual via the existing incremental descent
# (DESCENT_SUBSTEP_MAX_STEPS=400/DESCENT_SUBSTEP_STAGNATION_STEPS=150 per
# sub-step, 30 sub-steps by default) - a real question left open by every
# prior session: is that residual a genuine local-optimum floor, or does it
# reflect the descent's own comparatively small per-sub-step iteration
# budget rather than the solver's true convergence limit? This flag adds
# ONE extra polish_from_seed call, at the GRASP waypoint's full-precision
# target (not a sub-height interpolation), immediately after the existing
# descent (or one-shot) GRASP resolution finishes - continuing from
# whatever the robot's live converged state already is (matches
# polish_from_seed's own no-teleport-on-seed design, so this genuinely
# extends the same solve rather than restarting it). Disabled by default
# (0 steps) so it never changes any existing sweep/behavior unless
# explicitly requested.
parser.add_argument(
    "--grasp-deep-polish-steps", type=int, default=0,
    help="If > 0, run one additional polish_from_seed pass at the GRASP waypoint's full target (continuing live from wherever the descent/one-shot resolve left off, no teleport) with this many MAX physics steps, to test whether more solver effort shrinks the ~9.5mm/4.2deg residual found at the 65deg-tilt/reach=0.30-0.36m configuration, or whether it's already a genuine plateau. 0 (default) disables this pass entirely.",
)
parser.add_argument(
    "--grasp-deep-polish-stagnation-steps", type=int, default=None,
    help="Stagnation-break threshold (consecutive no-improvement steps) for --grasp-deep-polish-steps' extra pass. Defaults to 20%% of --grasp-deep-polish-steps (min 500) if not given - deliberately generous relative to the existing DESCENT_SUBSTEP_STAGNATION_STEPS=150 so this pass can distinguish 'still slowly improving' from 'genuinely plateaued' rather than breaking early on the same tight budget already used per descent sub-step.",
)
parser.add_argument(
    "--grasp-pos-threshold", type=float, default=None,
    help="Override CONVERGENCE_THRESHOLD (position, meters) for --grasp-deep-polish-steps' extra pass only - lets this pass demand a tighter position convergence bound than the module default (0.003m) without affecting PREGRASP or the descent sub-steps.",
)
parser.add_argument(
    "--grasp-rot-threshold", type=float, default=None,
    help="Override ROT_CONVERGENCE_THRESHOLD (radians) for --grasp-deep-polish-steps' extra pass only - lets this pass demand a tighter rotation convergence bound than the module default (0.05rad) without affecting PREGRASP or the descent sub-steps.",
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
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

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
# POLISH_SETTLE_STEPS: still used by _settle_at() (multi-seed search
# settling, and the final "restore best config" step at the end of
# polish_from_seed) - NOT by polish_from_seed's own per-step loop anymore,
# see POLISH_MAX_STEPS below and polish_from_seed's own docstring for why
# that changed (2026-07-22, ar4-tilt-fix task).
POLISH_SETTLE_STEPS = 30
# 2026-07-22 (ar4-tilt-fix task): replaces the old POLISH_ROUNDS (100,
# briefly bumped to 200) x POLISH_SETTLE_STEPS(30)-physics-steps-per-round
# open-loop design. polish_from_seed now re-solves the DLS Jacobian and
# takes one bounded step EVERY physics step (see that function's own
# docstring for why the old "solve once, hold blindly for 30 steps" design
# was the actual cause of a --tilt-deg 15/30 divergence that persisting
# reducing POLISH_ROT_STEP_MAX alone did not fix) - so this is now a
# PHYSICS STEP budget, not a "round" budget. Sized generously (old design's
# worst-case physics-step budget was 200*30=6000) since the continuous
# re-solve converges much faster per physics-step than the old design did
# per round (confirmed live: PREGRASP's old design took ~50-60 rounds x 30
# steps/round = 1500-1800 physics steps to converge from a ~1.5rad seed
# error; the continuous version needs far fewer steps for the same
# correction since it isn't wasting 29 of every 30 steps holding a stale
# target).
POLISH_MAX_STEPS = 3000
# 2026-07-22 (ar4-tilt-fix task): break out of polish_from_seed's loop early
# once the combined score hasn't improved for this many consecutive physics
# steps - added after live evidence (--tilt-deg 10/15 at the default 27.5cm
# reach) that GRASP's polish can reach a genuine STATIONARY point (residual
# frozen to 4+ decimal places for 300+ steps straight) with no chance of
# further improvement, well before POLISH_MAX_STEPS - running the full
# budget in that state only burns GPU time for zero benefit. Generous
# relative to how quickly the continuous per-step solve now converges
# genuinely-improving cases (PREGRASP converges from a ~1.5rad seed error in
# ~150 steps) so this won't cut off a still-improving run early.
STAGNATION_BREAK_STEPS = 500
LAMBDA_VAL = args_cli.lambda_val if args_cli.lambda_val is not None else 0.02
CONVERGENCE_THRESHOLD = 0.003
# 2026-07-22 (orientation-fix task): per-round bounded ROTATION step, the
# orientation analogue of POLISH_STEP_MAX - mirrors
# demo_franka_ik_dice_line.py's own _MAX_ROT_STEP (bounded per-step
# rotation correction, not a single large jump, for the same stability
# reason POLISH_STEP_MAX bounds position).
#
# 2026-07-22 (ar4-tilt-fix task): 0.15 -> 0.03, matching
# demo_franka_ik_dice_line.py's own _MAX_ROT_STEP EXACTLY (a proven-stable
# reference value, not a guess) - found to be the most likely cause of the
# --tilt-deg 30 divergence diagnosed in this module's own docstring/kb doc
# ("rotation error monotonically INCREASING round over round instead of
# converging"). The two scripts' polish/step loops are structurally
# different (Franka's `_step_toward` re-solves the DLS Jacobian EVERY
# physics step and takes one small bounded step per step, closed-loop;
# this script's `polish_from_seed` solves ONCE per "round" then holds that
# single joint_pos_des target open-loop for POLISH_SETTLE_STEPS=30 physics
# steps before re-measuring) - but the per-round/per-step BOUND on how far
# a single DLS linearization is trusted to extrapolate is the same kind of
# safety margin in both, and this script's value (0.15rad, ~8.6 degrees)
# was 5x Franka's own (0.03rad, ~1.7 degrees) with no stated justification
# for the larger figure. At a genuine non-zero tilt target, AR4's basin is
# already close to a joint-limit-constrained boundary (see this module's
# docstring) - a single round's Jacobian-linearized correction is only a
# valid local approximation over a small neighborhood, and demanding up to
# 8.6 degrees of rotational correction in one open-loop round (then holding
# that target for 30 full settle steps, giving the arm's actual nonlinear
# dynamics plenty of time to overshoot past where the linear model predicted)
# is a straightforward mechanism for the observed instability: each round's
# real endpoint lands further from the linearization point than the next
# round's Jacobian can correctly account for, so error compounds rather than
# shrinks. Matching Franka's proven bound directly, rather than re-deriving
# a new one from scratch, since this exact bound has already been validated
# stable (10 dice x 2 passes) in that script.
POLISH_ROT_STEP_MAX = 0.03
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
# 2026-07-22 (ar4-grasp-descent-continuity task): per-sub-step budgets for
# the incremental PREGRASP->GRASP height descent (see --num-descent-steps
# and main()'s descent loop). Each sub-step only needs to correct a small
# (~1-3mm, for the default 30-step/5cm-total descent) height change from an
# already-converged neighboring state, unlike PREGRASP's/the old one-shot
# GRASP's much larger corrections from a generic multi-seed-search start -
# a much smaller step/stagnation budget than POLISH_MAX_STEPS/
# STAGNATION_BREAK_STEPS is deliberately used so a genuinely-stuck sub-step
# (the disconnected-basin hypothesis predicts this might still happen near
# the bottom of the descent) fails fast rather than burning the full 3000-step
# budget per sub-step across dozens of sub-steps.
DESCENT_SUBSTEP_MAX_STEPS = 400
DESCENT_SUBSTEP_STAGNATION_STEPS = 150


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


def _wiggle_and_resolve(
    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, num_arm_joints,
    joint_pos_limits, max_wiggles=6, wiggle_settle_steps=60, wiggle_max_rad=0.3, rng_seed=0,
):
    """Real-robot-deployable replacement for _find_best_seed's teleport-based
    multi-candidate search (2026-07-22/23, ar4-grasp-deployability-check
    task, coordinator-directed). _find_best_seed calls
    write_joint_position_to_sim to instantly snap the robot into several
    candidate configs and score each one before committing - a genuine
    simulation-only capability with no real-hardware analogue (a physical
    AR4 cannot "try" a configuration and undo it). This function instead:

    1. Attempts the direct continuous DLS resolve (polish_from_seed, already
       real/deployable - the Jacobian-frame fix, EE-offset fix, and
       per-physics-step continuous resolve it uses are all genuine, no
       teleportation inside polish_from_seed itself) from wherever the
       robot's live state ACTUALLY is right now - no teleport.
    2. If that doesn't converge (final pos/rot residual still above the
       normal convergence thresholds), applies a SMALL, BOUNDED per-joint
       perturbation (uniform in [-wiggle_max_rad, wiggle_max_rad], default
       ~17 degrees - a real jog/dither move, not a distant jump) as a
       normal COMMANDED joint-position target, held for wiggle_settle_steps
       physics steps of ordinary PD-driven motion via env.step - the exact
       same mechanism every other move in this script uses, never
       write_joint_position_to_sim - then retries the resolve from that new,
       physically-reached nearby state.
    3. Repeats up to max_wiggles times, returning whatever the best attempt
       achieved (by the same combined position+orientation score
       polish_from_seed itself uses) if none fully converges.

    This is deliberately NOT claimed to be equivalent in search power to the
    teleport-based multi-seed search - it is a bounded LOCAL search around
    wherever the robot currently is, which is exactly the constraint a real
    robot actually has. Comparing its result quality against the
    teleport-assisted version's is the whole point of this function.
    """
    robot = env.scene["robot"]
    rng = random.Random(rng_seed)

    def _combined_score(pos_err: float, rot_err: float) -> float:
        return pos_err + ORIENTATION_SCORE_WEIGHT * rot_err

    cur_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
    best_q, best_pos_res, best_rot_res = polish_from_seed(
        env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, cur_q,
        num_arm_joints, joint_pos_limits,
    )
    best_score = _combined_score(best_pos_res, best_rot_res)
    print(f"  [WIGGLE 0/{max_wiggles}] direct resolve (no teleport, no wiggle): pos={best_pos_res:.5f}m rot={best_rot_res:.4f}rad score={best_score:.5f}")

    attempt = 0
    lo = joint_pos_limits[0, :, 0].tolist()
    hi = joint_pos_limits[0, :, 1].tolist()
    while (best_pos_res >= CONVERGENCE_THRESHOLD or best_rot_res >= ROT_CONVERGENCE_THRESHOLD) and attempt < max_wiggles:
        attempt += 1
        base_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
        wiggle_q = [
            min(max(q + rng.uniform(-wiggle_max_rad, wiggle_max_rad), l), h)
            for q, l, h in zip(base_q, lo, hi)
        ]
        for _ in range(wiggle_settle_steps):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, :num_arm_joints] = torch.tensor([wiggle_q], device=env.device)
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)
        print(f"  [WIGGLE {attempt}/{max_wiggles}] PD-driven nudge (no teleport) to q={['%.4f' % v for v in wiggle_q]}, retrying resolve...")
        cand_q, cand_pos_res, cand_rot_res = polish_from_seed(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, wiggle_q,
            num_arm_joints, joint_pos_limits,
        )
        cand_score = _combined_score(cand_pos_res, cand_rot_res)
        print(f"  [WIGGLE {attempt}/{max_wiggles}] result: pos={cand_pos_res:.5f}m rot={cand_rot_res:.4f}rad score={cand_score:.5f}")
        if cand_score < best_score:
            best_q, best_pos_res, best_rot_res, best_score = cand_q, cand_pos_res, cand_rot_res, cand_score
        else:
            # This attempt was worse than an earlier one - the robot's LIVE
            # state is now sitting at this worse config, not best_q.
            # polish_from_seed always resumes from the robot's actual live
            # state (never teleports to its seed_q argument, see its own
            # docstring), so a caller chaining another polish_from_seed call
            # after this function returns would silently start from the
            # wrong place unless we explicitly move back to best_q here -
            # via a normal commanded PD move, same as every other move in
            # this function, never a teleport.
            for _ in range(wiggle_settle_steps):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                action[:, :num_arm_joints] = torch.tensor([best_q], device=env.device)
                action[:, num_arm_joints] = GRIPPER_OPEN
                env.step(action)

    print(
        f"  [WIGGLE] final (after {attempt} wiggle attempt(s), no teleport used anywhere): "
        f"pos={best_pos_res:.5f}m rot={best_rot_res:.4f}rad"
    )
    return best_q, best_pos_res, best_rot_res


def polish_from_seed(
    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits,
    max_steps=None, stagnation_break_steps=None, pos_threshold=None, rot_threshold=None,
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
    that had great position but a bad orientation.

    2026-07-22 (ar4-tilt-fix task): switched from ONE Jacobian solve per
    "round" held open-loop for POLISH_SETTLE_STEPS=30 physics steps, to a
    CONTINUOUS per-physics-step re-solve - one full re-measure + re-solve +
    bounded-step + env.step() per iteration, no blind hold in between.
    Mirrors demo_franka_ik_dice_line.py's own `_step_toward` exactly (that
    function also re-reads the Jacobian/error and takes one bounded step
    EVERY physics step, proven stable over 10 full pick-and-place cycles).
    Found necessary live: reducing POLISH_ROT_STEP_MAX alone (0.15->0.03,
    matching Franka's own bound) was NOT sufficient - a --tilt-deg 15 run
    with the smaller step bound still diverged (rot_err 0.019rad -> 1.32rad
    within 10 rounds, then plateaued/deadlocked there), and the JUMP size
    (>1rad of measured orientation change) was far larger than 10 rounds'
    worth of a 0.03rad-capped COMMANDED step could produce even in the
    worst case (max 0.3rad cumulative) - meaning the actual achieved motion
    diverged from what the single-shot linearization at the round's start
    predicted, not just "step too big". The old design solves the DLS
    Jacobian ONCE per round from the PRE-round state, then blindly holds
    that single computed target for 30 physics steps before ever
    re-checking - if that configuration turns out to require passing near a
    Jacobian near-singularity (AR4 has a non-spherical wrist, unlike
    Franka's, so wrist-singularity-adjacent configs are a real risk for a
    tilted, non-canonical orientation target) or a joint-limit boundary
    along the way, the actual 30-step trajectory can overshoot far past
    where the linearization predicted before anything corrects it - and the
    NEXT round's solve then starts from that already-bad state, compounding
    rather than correcting. Re-solving every single physics step (this fix)
    means any such overshoot is caught and corrected on the very next
    physics step, exactly like Franka's proven closed-loop tracking, instead
    of being locked in for 30 steps first.

    2026-07-22 (ar4-grasp-descent-continuity task): added optional
    ``max_steps``/``stagnation_break_steps`` overrides (defaulting to the
    module-level ``POLISH_MAX_STEPS``/``STAGNATION_BREAK_STEPS`` constants
    when not given), so a caller doing an incremental multi-sub-waypoint
    descent (see ``main()``'s ``--num-descent-steps`` handling) can give each
    small sub-step a much smaller budget than a single big one-shot solve
    needs, without touching the module constants used elsewhere. Also note
    this function NEVER teleports the robot to ``seed_q`` itself - it always
    starts its DLS loop from whatever the robot's actual LIVE state is at
    call time (``seed_q`` is only used as the initial "best" bookkeeping
    value for the keep-best-round guard). This is exactly what makes calling
    this function repeatedly back-to-back, with no ``_settle_at``/teleport
    in between, behave as a genuine continuous resolve from one call's
    converged end-state to the next call's starting state - the mechanism
    the incremental descent below relies on.

    2026-07-24 (ar4-grasp-ik-convergence-tightening task): added optional
    ``pos_threshold``/``rot_threshold`` overrides (defaulting to the
    module-level ``CONVERGENCE_THRESHOLD``/``ROT_CONVERGENCE_THRESHOLD``
    constants when not given), so a caller wanting a tighter early-exit
    convergence bound for one specific waypoint/pass (see ``main()``'s
    ``--grasp-deep-polish-steps`` handling) can do so without touching the
    module constants used everywhere else (PREGRASP, the descent
    sub-steps)."""
    robot = env.scene["robot"]
    max_steps = POLISH_MAX_STEPS if max_steps is None else max_steps
    stagnation_break_steps = STAGNATION_BREAK_STEPS if stagnation_break_steps is None else stagnation_break_steps
    pos_threshold = CONVERGENCE_THRESHOLD if pos_threshold is None else pos_threshold
    rot_threshold = ROT_CONVERGENCE_THRESHOLD if rot_threshold is None else rot_threshold

    def _combined_score(pos_err: float, rot_err: float) -> float:
        return pos_err + ORIENTATION_SCORE_WEIGHT * rot_err

    best_pos_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
    best_rot_residual = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
    best_score = _combined_score(best_pos_residual, best_rot_residual)
    best_polish_q = list(seed_q)
    final_pos_residual = best_pos_residual
    final_rot_residual = best_rot_residual
    # 2026-07-22 (ar4-tilt-fix task): stagnation counter for
    # STAGNATION_BREAK_STEPS (see that constant's own docstring).
    steps_since_improvement = 0

    for step_num in range(max_steps):
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

        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = joint_pos_des
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)

        final_pos_residual = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        final_rot_residual = _measure_rot_err(robot, robot_entity_cfg, target_quat_b)
        final_score = _combined_score(final_pos_residual, final_rot_residual)
        if step_num % 100 == 0 or step_num == max_steps - 1:
            # 2026-07-22 (ar4-tilt-fix task): also print the live joint config
            # against joint_pos_limits, so a plateaued/deadlocked residual can
            # be immediately attributed to a specific joint sitting at (or
            # very near) its hard limit, without a separate diagnostic pass.
            live_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            lo_list = joint_pos_limits[0, :, 0].tolist()
            hi_list = joint_pos_limits[0, :, 1].tolist()
            margins = [min(q - l, h - q) for q, l, h in zip(live_q, lo_list, hi_list)]
            print(
                f"  [POLISH step {step_num:4d}] pos_err={final_pos_residual:.5f}m rot_err={final_rot_residual:.4f}rad "
                f"q={['%.4f' % v for v in live_q]} limit_margin={['%.4f' % v for v in margins]}"
            )
        if final_score < best_score:
            best_score = final_score
            best_pos_residual = final_pos_residual
            best_rot_residual = final_rot_residual
            best_polish_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        if final_pos_residual < pos_threshold and final_rot_residual < rot_threshold:
            break
        if steps_since_improvement >= stagnation_break_steps:
            print(
                f"  [POLISH] no improvement for {stagnation_break_steps} consecutive steps - stopping early "
                f"at step {step_num} instead of running the full {max_steps}-step budget"
            )
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

    # 2026-07-23 (coordinator-directed, mid-task correction): the prior
    # session's "gripper confirmed OPEN throughout" claim was a SOURCE-CODE
    # check of the COMMANDED action tensor (GRIPPER_OPEN=1.0 fed into
    # BinaryJointPositionAction) - not a live read of the gripper's ACTUAL
    # physical joint position. Given this whole investigation's repeated
    # history of commanded-vs-actual divergence bugs for this exact gripper
    # (missing jaw2 physics drive, an inverted mimic sign, jaw-mimic-limit
    # conflicts - all found and fixed earlier in this same investigation),
    # a direct visual observation of a live run (gripper appeared CLOSED
    # throughout, not open during descent as commanded) means this needs its
    # own independent, physical-joint-position confirmation, not another
    # commanded-action-tensor argument. Resolved via robot.find_joints, the
    # same mechanism BinaryJointPositionAction itself uses internally, so
    # this reads the identical joints the action term drives.
    gripper_joint_ids, gripper_joint_names_found = robot.find_joints(GRIPPER_JOINT_NAMES)
    print(f"[INFO] Gripper joint ids resolved: {gripper_joint_names_found} -> {gripper_joint_ids}")

    def _print_gripper_state(label: str) -> None:
        live_gripper_q = robot.data.joint_pos[0, gripper_joint_ids].tolist()
        print(f"  [GRIPPER-CHECK] {label}: actual joint_pos {gripper_joint_names_found}={['%.5f' % v for v in live_gripper_q]}")

    # 2026-07-23 (ar4-jaw-bisector-hypothesis task): direct LIVE measurement
    # of both gripper jaw fingertips' real world body positions -
    # robot.data.body_pos_w, NOT the arm's own single _EE_OFFSET-derived
    # target point - to test whether _EE_OFFSET (a fixed constant, measured
    # near-vertical per pickplace_env_cfg.py's own docstring) still
    # represents the TRUE bisector point between the two jaw fingertips once
    # the whole gripper is oriented at a large, non-near-vertical tilt (the
    # 2026-07-23 ar4-capstone-grasp task's own flagged next diagnostic - see
    # kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching
    # UPDATE). This is independent of (and a live-asset cross-check against)
    # a pure-FK-model calculation done offline against tasks/ar4/
    # fk_verification.py's vendor-URDF joint table, which found the
    # local-frame offset between link_6 and the true jaw bisector is a
    # CONSTANT ~0.0001mm regardless of arm joint values (confirmed
    # numerically across several very different joint configs) - a rigid,
    # fixed-body-frame quantity by construction, since the whole gripper
    # subtree hangs off link_6 through fixed joints plus the jaws' own
    # prismatic joints only, none of which depend on the arm's own joint_1-6
    # values. That FK-model result predicts this hypothesis should be
    # FALSIFIED on the real built asset too, unless the built USD itself
    # carries an import-time defect the idealized vendor-URDF model
    # wouldn't predict (this project's own history has real precedent for
    # exactly that class of surprise - see fk_verification.py's own
    # docstring on its first-draft calibration being wrong until a live
    # integration run caught it) - hence checking directly on the live
    # asset here rather than trusting the offline calculation alone.
    gripper_jaw_body_ids, gripper_jaw_body_names_found = robot.find_bodies(["gripper_jaw1_link", "gripper_jaw2_link"])
    print(f"[INFO] Gripper jaw body ids resolved: {gripper_jaw_body_names_found} -> {gripper_jaw_body_ids}")

    def _measure_jaw_bisector_vs_ee_offset(label: str, cube_pos_w=None):
        """Live-measure both jaw fingertips' world positions, the true
        bisector between them, and the arm's own _EE_OFFSET-derived assumed
        pinch point - all from robot.data.body_pos_w/body_pose_w directly
        (not this script's own root-frame IK-target bookkeeping). Prints the
        discrepancy between the true bisector and the assumed point, and (if
        cube_pos_w is given) each jaw's own distance to the cube - the
        latter directly tests whether a nonzero position residual has an
        ASYMMETRIC effect (one jaw meaningfully closer to the cube than the
        other), the mechanism this task's step 4 flags as the next most
        likely explanation for jaw1-only contact if the bisector/offset
        discrepancy itself turns out to be negligible."""
        jaw_pos_w = robot.data.body_pos_w[0, gripper_jaw_body_ids].clone()
        jaw1_pos_w, jaw2_pos_w = jaw_pos_w[0], jaw_pos_w[1]
        true_bisector_w = (jaw1_pos_w + jaw2_pos_w) / 2.0
        jaw_sep_mm = torch.norm(jaw1_pos_w - jaw2_pos_w).item() * 1000.0

        ee_pose_w = robot.data.body_pose_w[0, robot_entity_cfg.body_ids[0]]
        ee_pos_w, ee_quat_w = ee_pose_w[0:3], ee_pose_w[3:7]
        rot = matrix_from_quat(ee_quat_w.unsqueeze(0))[0]
        offset_local = torch.tensor(_EE_OFFSET, device=ee_pos_w.device)
        assumed_pinch_w = ee_pos_w + rot @ offset_local

        disc_mm = torch.norm(true_bisector_w - assumed_pinch_w).item() * 1000.0
        print(f"[BISECTOR-CHECK] {label}: jaw1_w={jaw1_pos_w.tolist()} jaw2_w={jaw2_pos_w.tolist()} sep={jaw_sep_mm:.4f}mm")
        print(f"[BISECTOR-CHECK] {label}: true_bisector_w={true_bisector_w.tolist()} assumed_pinch_w(_EE_OFFSET)={assumed_pinch_w.tolist()}")
        print(f"[BISECTOR-CHECK] {label}: DISCREPANCY (true bisector vs _EE_OFFSET assumed point) = {disc_mm:.4f}mm")
        if cube_pos_w is not None:
            bisector_vs_cube_mm = torch.norm(true_bisector_w - cube_pos_w).item() * 1000.0
            assumed_vs_cube_mm = torch.norm(assumed_pinch_w - cube_pos_w).item() * 1000.0
            jaw1_to_cube_mm = torch.norm(jaw1_pos_w - cube_pos_w).item() * 1000.0
            jaw2_to_cube_mm = torch.norm(jaw2_pos_w - cube_pos_w).item() * 1000.0
            print(f"[BISECTOR-CHECK] {label}: true_bisector vs cube = {bisector_vs_cube_mm:.4f}mm; assumed_pinch vs cube = {assumed_vs_cube_mm:.4f}mm")
            print(
                f"[BISECTOR-CHECK] {label}: jaw1-to-cube={jaw1_to_cube_mm:.4f}mm  jaw2-to-cube={jaw2_to_cube_mm:.4f}mm  "
                f"(asymmetry={abs(jaw1_to_cube_mm - jaw2_to_cube_mm):.4f}mm - a large asymmetry here, with a small "
                "bisector/offset discrepancy above, would point at the pre-existing position residual's own DIRECTION "
                "as the explanation for one-sided contact, not an _EE_OFFSET calibration bug)"
            )
        return true_bisector_w, assumed_pinch_w, disc_mm

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
    # 2026-07-22 (orientation-fix task): perception_camera is a tight
    # close-up framing tuned for object detection (per its own docstring in
    # tasks/ar4/grasp_verify_env_cfg.py), not for a human watching arm
    # motion - a review of ar4_grasp_demo_v2.mp4 found it nearly impossible
    # to tell what's happening from that camera alone. demo_camera (a wide
    # 3/4 view, same module) is recorded to a second file for actual visual
    # verification purposes, alongside the existing perception_camera video.
    demo_video_path = VIDEO_PATH.replace(".mp4", "_demo_camera.mp4")
    demo_video_writer = imageio.get_writer(demo_video_path, fps=int(1.0 / env.step_dt), codec="libx264")
    demo_camera = env.scene["demo_camera"]

    with torch.inference_mode():
        env.reset()

        # Baseline bisector-vs-offset check at HOME_Q (near-zero joint
        # values, the arm's default reset pose) - per the offline FK-model
        # calculation (see _measure_jaw_bisector_vs_ee_offset's own
        # docstring), this discrepancy should be joint-config-independent
        # (a rigid local-frame quantity), so this baseline reading should
        # already match whatever is found later at the tilted grasp_q - a
        # live cross-check of that prediction on the REAL built asset,
        # not just the idealized vendor-URDF model.
        _measure_jaw_bisector_vs_ee_offset("HOME_Q baseline (post-reset)")

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

        # Known-good absolute configs from a prior offline multi-seed search
        # at this exact target (cube (0.0, 0.275, 0.009)) - see
        # _find_best_seed's own docstring for why these are included
        # directly rather than trusting this run's live search alone to
        # reproduce the same basin. Moved up here (2026-07-22, ar4-grasp-
        # z-envelope task) from further down in main() so --bearing-sweep's
        # own per-bearing seed search (below, which runs BEFORE the rest of
        # main()'s normal single-position pipeline) can also use
        # KNOWN_GOOD_PREGRASP_Q as an extra seed candidate.
        KNOWN_GOOD_GRASP_Q = [-0.014429761096835136, 1.240863561630249, 0.3401874601840973, -0.08906537294387817, 1.1987247467041016, 0.0052983760833740234]
        KNOWN_GOOD_PREGRASP_Q = [0.00014747596287634224, 0.9648232460021973, 0.9025915265083313, -0.0006890640361234546, -0.6352600455284119, 0.008003178983926773]

        # 2026-07-23 (ar4-grasp-position-search task): --radius-sweep, same
        # supersede-the-normal-pipeline pattern as --bearing-sweep below (own
        # per-point cube teleport + seed-search + PREGRASP polish + descent),
        # checked first since it's cheap to check and mutually exclusive with
        # --bearing-sweep in practice (both exit before reaching the rest of
        # main()).
        if args_cli.radius_sweep is not None:
            grasp_h = GRASP_AT_HEIGHT
            print(
                f"\n[RADIUS-SWEEP] bearing=0deg (straight-ahead) grasp_height={grasp_h}m "
                f"radii(m)={args_cli.radius_sweep}"
            )
            for radius in args_cli.radius_sweep:
                bx, by = 0.0, radius
                override_z = cube.data.root_pos_w[0, 2].item()
                override_pos = torch.tensor([[bx, by, override_z]], device=env.device)
                override_quat = cube.data.root_quat_w[0:1].clone()
                cube.write_root_pose_to_sim(
                    torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
                )
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

                cube_pos_w_i = cube.data.root_pos_w[0:1].clone()
                cube_pos_b_i, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w_i)
                bearing_i = math.atan2(cube_pos_b_i[0, 1].item(), cube_pos_b_i[0, 0].item())
                seed_j1_i = -(bearing_i - CALIBRATION_C)

                pregrasp_pos_b_i = cube_pos_b_i.clone()
                pregrasp_pos_b_i[:, 2] = grasp_h + PREGRASP_HOVER
                grasp_pos_b_i = cube_pos_b_i.clone()
                grasp_pos_b_i[:, 2] = grasp_h

                seed_q_i, _ = _find_best_seed(
                    env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b_i, target_quat_b, seed_j1_i,
                    extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q],
                )
                pregrasp_q_i, _, _ = polish_from_seed(
                    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b_i, target_quat_b, seed_q_i,
                    num_arm_joints, joint_pos_limits,
                )

                _settle_at(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_q_i, SEED_SETTLE_STEPS)
                cur_q_i = pregrasp_q_i
                z_start_i = grasp_h + PREGRASP_HOVER
                last_target_i = None
                pos_res_i = rot_res_i = None
                for sub_idx in range(1, args_cli.num_descent_steps + 1):
                    sub_z = z_start_i + (grasp_h - z_start_i) * sub_idx / args_cli.num_descent_steps
                    sub_target_i = grasp_pos_b_i.clone()
                    sub_target_i[:, 2] = sub_z
                    last_target_i = sub_target_i
                    cur_q_i, pos_res_i, rot_res_i = polish_from_seed(
                        env, ik_controller, robot_entity_cfg, ik_jacobi_idx, sub_target_i, target_quat_b, cur_q_i,
                        num_arm_joints, joint_pos_limits,
                        max_steps=DESCENT_SUBSTEP_MAX_STEPS, stagnation_break_steps=DESCENT_SUBSTEP_STAGNATION_STEPS,
                    )
                residual_vec_i = _measure_dist_vec(robot, robot_entity_cfg, last_target_i)
                live_q_i = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                lo_i = joint_pos_limits[0, :, 0].tolist()
                hi_i = joint_pos_limits[0, :, 1].tolist()
                margins_i = [min(q - l, h - q) for q, l, h in zip(live_q_i, lo_i, hi_i)]
                print(
                    f"[RADIUS-SWEEP] radius={radius:.4f} cube_xy_w=({bx:.4f},{by:.4f}) "
                    f"pos_err={pos_res_i:.5f}m rot_err={rot_res_i:.4f}rad "
                    f"xyz_residual={['%.5f' % v for v in residual_vec_i]} "
                    f"q={['%.4f' % v for v in live_q_i]} margins={['%.4f' % v for v in margins_i]}"
                )
            print("[RADIUS-SWEEP] done - exiting before one-shot GRASP solve / phased execution.")
            video_writer.close()
            demo_video_writer.close()
            env.close()
            return

        # 2026-07-23 (ar4-capstone-grasp task): --tilt-sweep - fixed cube
        # position (whatever --cube-xy resolved to above, or the scene
        # default), varying ONLY the canonical target's tilt_deg. See this
        # flag's own docstring for why (untested tilt-at-comfortable-margin
        # combination).
        if args_cli.tilt_sweep is not None:
            grasp_h = GRASP_AT_HEIGHT
            print(
                f"\n[TILT-SWEEP] cube_pos_b={cube_pos_b[0].tolist()} grasp_height={grasp_h}m "
                f"tilts(deg)={args_cli.tilt_sweep}"
            )
            for tilt_deg_i in args_cli.tilt_sweep:
                target_quat_b_i = _build_canonical_target_quat_b(root_pos_w, root_quat_w, tilt_deg=tilt_deg_i)

                pregrasp_pos_b_i = cube_pos_b.clone()
                pregrasp_pos_b_i[:, 2] = grasp_h + PREGRASP_HOVER
                grasp_pos_b_i = cube_pos_b.clone()
                grasp_pos_b_i[:, 2] = grasp_h

                seed_q_i, _ = _find_best_seed(
                    env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b_i, target_quat_b_i, seed_j1,
                    extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q],
                )
                pregrasp_q_i, _, _ = polish_from_seed(
                    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b_i, target_quat_b_i, seed_q_i,
                    num_arm_joints, joint_pos_limits,
                )

                _settle_at(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_q_i, SEED_SETTLE_STEPS)
                cur_q_i = pregrasp_q_i
                z_start_i = grasp_h + PREGRASP_HOVER
                last_target_i = None
                pos_res_i = rot_res_i = None
                for sub_idx in range(1, args_cli.num_descent_steps + 1):
                    sub_z = z_start_i + (grasp_h - z_start_i) * sub_idx / args_cli.num_descent_steps
                    sub_target_i = grasp_pos_b_i.clone()
                    sub_target_i[:, 2] = sub_z
                    last_target_i = sub_target_i
                    cur_q_i, pos_res_i, rot_res_i = polish_from_seed(
                        env, ik_controller, robot_entity_cfg, ik_jacobi_idx, sub_target_i, target_quat_b_i, cur_q_i,
                        num_arm_joints, joint_pos_limits,
                        max_steps=DESCENT_SUBSTEP_MAX_STEPS, stagnation_break_steps=DESCENT_SUBSTEP_STAGNATION_STEPS,
                    )
                residual_vec_i = _measure_dist_vec(robot, robot_entity_cfg, last_target_i)
                live_q_i = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                lo_i = joint_pos_limits[0, :, 0].tolist()
                hi_i = joint_pos_limits[0, :, 1].tolist()
                margins_i = [min(q - l, h - q) for q, l, h in zip(live_q_i, lo_i, hi_i)]
                print(
                    f"[TILT-SWEEP] tilt_deg={tilt_deg_i:.1f} pos_err={pos_res_i:.5f}m rot_err={rot_res_i:.4f}rad "
                    f"xyz_residual={['%.5f' % v for v in residual_vec_i]} "
                    f"q={['%.4f' % v for v in live_q_i]} margins={['%.4f' % v for v in margins_i]}"
                )
            print("[TILT-SWEEP] done - exiting before one-shot GRASP solve / phased execution.")
            video_writer.close()
            demo_video_writer.close()
            env.close()
            return

        # 2026-07-22 (ar4-grasp-z-envelope task): --bearing-sweep supersedes
        # the normal single-position pipeline entirely (it does its own
        # per-bearing cube teleport + seed-search + PREGRASP polish + descent,
        # independent of whatever --cube-xy/the scene default set cube_pos_b
        # to above) - checked here, before the rest of main() commits to one
        # specific cube position.
        if args_cli.bearing_sweep is not None:
            grasp_h = GRASP_AT_HEIGHT
            print(
                f"\n[BEARING-SWEEP] radius={args_cli.bearing_sweep_radius}m grasp_height={grasp_h}m "
                f"bearings(deg)={args_cli.bearing_sweep}"
            )
            for bearing_deg in args_cli.bearing_sweep:
                theta = math.radians(bearing_deg)
                bx = args_cli.bearing_sweep_radius * math.sin(theta)
                by = args_cli.bearing_sweep_radius * math.cos(theta)
                override_z = cube.data.root_pos_w[0, 2].item()
                override_pos = torch.tensor([[bx, by, override_z]], device=env.device)
                override_quat = cube.data.root_quat_w[0:1].clone()
                cube.write_root_pose_to_sim(
                    torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
                )
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

                cube_pos_w_i = cube.data.root_pos_w[0:1].clone()
                cube_pos_b_i, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w_i)
                bearing_i = math.atan2(cube_pos_b_i[0, 1].item(), cube_pos_b_i[0, 0].item())
                seed_j1_i = -(bearing_i - CALIBRATION_C)

                pregrasp_pos_b_i = cube_pos_b_i.clone()
                pregrasp_pos_b_i[:, 2] = grasp_h + PREGRASP_HOVER
                grasp_pos_b_i = cube_pos_b_i.clone()
                grasp_pos_b_i[:, 2] = grasp_h

                seed_q_i, _ = _find_best_seed(
                    env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b_i, target_quat_b, seed_j1_i,
                    extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q],
                )
                pregrasp_q_i, _, _ = polish_from_seed(
                    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b_i, target_quat_b, seed_q_i,
                    num_arm_joints, joint_pos_limits,
                )

                _settle_at(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_q_i, SEED_SETTLE_STEPS)
                cur_q_i = pregrasp_q_i
                z_start_i = grasp_h + PREGRASP_HOVER
                last_target_i = None
                pos_res_i = rot_res_i = None
                for sub_idx in range(1, args_cli.num_descent_steps + 1):
                    sub_z = z_start_i + (grasp_h - z_start_i) * sub_idx / args_cli.num_descent_steps
                    sub_target_i = grasp_pos_b_i.clone()
                    sub_target_i[:, 2] = sub_z
                    last_target_i = sub_target_i
                    cur_q_i, pos_res_i, rot_res_i = polish_from_seed(
                        env, ik_controller, robot_entity_cfg, ik_jacobi_idx, sub_target_i, target_quat_b, cur_q_i,
                        num_arm_joints, joint_pos_limits,
                        max_steps=DESCENT_SUBSTEP_MAX_STEPS, stagnation_break_steps=DESCENT_SUBSTEP_STAGNATION_STEPS,
                    )
                residual_vec_i = _measure_dist_vec(robot, robot_entity_cfg, last_target_i)
                live_q_i = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                lo_i = joint_pos_limits[0, :, 0].tolist()
                hi_i = joint_pos_limits[0, :, 1].tolist()
                margins_i = [min(q - l, h - q) for q, l, h in zip(live_q_i, lo_i, hi_i)]
                print(
                    f"[BEARING-SWEEP] bearing_deg={bearing_deg:.1f} cube_xy_w=({bx:.4f},{by:.4f}) "
                    f"pos_err={pos_res_i:.5f}m rot_err={rot_res_i:.4f}rad "
                    f"xyz_residual={['%.5f' % v for v in residual_vec_i]} "
                    f"q={['%.4f' % v for v in live_q_i]} margins={['%.4f' % v for v in margins_i]}"
                )
            print("[BEARING-SWEEP] done - exiting before one-shot GRASP solve / phased execution.")
            video_writer.close()
            demo_video_writer.close()
            env.close()
            return

        # Capture the cube's TRUE pose right after reset (and after any
        # --cube-xy override), before any seed-search/polish runs, THEN park
        # it far outside the whole reachable workspace for the entire
        # duration of that search - not just capture-then-restore.
        #
        # 2026-07-23 (ar4-grasp-position-search task, coordinator-directed):
        # the multi-seed search/PREGRASP polish/incremental descent below
        # teleports and PD-drives the arm through several very different
        # configurations (some close to the cube's own resting spot, since
        # they're candidates for reaching it) to evaluate them - if the arm
        # interpenetrates the cube during this process, PhysX's own
        # depenetration reaction can shove/impart velocity to it well before
        # the actual phased grasp attempt begins. The original fix
        # (capture-then-restore: read the true pose once, restore position
        # AND zero velocity right before Phase 0) was ALREADY committed and
        # is a real, adequate guard against this for the phased-execution's
        # OWN starting state (confirmed: this reverts the cube to a
        # deterministic pos+zero-vel state regardless of what happened
        # during the search) - but it does not prevent interpenetration
        # DURING the search itself, which is wasted physics/contact-solver
        # work and (per the coordinator's own concern) could theoretically
        # still leave a transient residual effect a pure position+velocity
        # overwrite might not fully undo (e.g. if the search's own last step
        # ends with the arm still overlapping the cube's real position at
        # the instant of restore, see below). Parking is strictly better:
        # no interpenetration can occur at all while the cube sits at
        # _CUBE_PARK_POS_W, far outside the whole workspace. (`cube` itself
        # was already resolved above, when the --cube-xy override - if any -
        # was applied.)
        cube_init_pos = cube.data.root_pos_w[0].clone()
        cube_init_quat = cube.data.root_quat_w[0].clone()
        print(f"[INFO] Cube initial pose (world): pos={cube_init_pos.tolist()}")

        _CUBE_PARK_POS_W = torch.tensor([5.0, 5.0, -5.0], device=env.device)
        park_pose = torch.cat([_CUBE_PARK_POS_W, cube_init_quat]).unsqueeze(0)
        cube.write_root_pose_to_sim(park_pose, env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube parked at {_CUBE_PARK_POS_W.tolist()} for the duration of seed-search/polish/descent.")

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

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
        if args_cli.deployable_seed:
            # 2026-07-22/23 (ar4-grasp-deployability-check task): real-robot-
            # deployable path - no write_joint_position_to_sim teleportation
            # anywhere in this branch. The robot is already sitting at
            # HOME_Q (tasks/ar4/robot_cfg.py's own init_state, all-zero
            # joint_pos) right after env.reset() - that IS the starting
            # point, not a teleport target.
            if args_cli.fixed_posture_move:
                # 2026-07-22/23 (deployability-check follow-up): one real,
                # deliberate PD-driven move (NOT a teleport) from HOME_Q to a
                # fixed reference posture, before any DLS resolve attempt -
                # see this flag's own docstring above.
                print(f"\n[INFO] --fixed-posture-move: commanding one real PD move (no teleport) to KNOWN_GOOD_PREGRASP_Q={KNOWN_GOOD_PREGRASP_Q}...")
                for _ in range(SEED_SETTLE_STEPS):
                    action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                    action[:, :num_arm_joints] = torch.tensor([KNOWN_GOOD_PREGRASP_Q], device=env.device)
                    action[:, num_arm_joints] = GRIPPER_OPEN
                    env.step(action)
            print("\n[INFO] --deployable-seed: resolving PREGRASP from the robot's live state via bounded PD-driven wiggle retries, NO teleportation...")
            pregrasp_q, pregrasp_residual, pregrasp_rot_residual = _wiggle_and_resolve(
                env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, target_quat_b, num_arm_joints,
                joint_pos_limits, max_wiggles=args_cli.max_wiggles, wiggle_max_rad=args_cli.wiggle_max_rad,
            )
        else:
            print("\n[INFO] Finding best seed for PREGRASP waypoint (multi-seed, genuinely settled)...")
            seed_q, _ = _find_best_seed(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_pos_b, target_quat_b, seed_j1, extra_full_seeds=[KNOWN_GOOD_PREGRASP_Q])
            print("[INFO] Polishing PREGRASP waypoint (fixed-Jacobian, pose-DLS)...")
            pregrasp_q, pregrasp_residual, pregrasp_rot_residual = polish_from_seed(
                env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits
            )

        # 2026-07-22 (ar4-grasp-z-envelope task): map the reachable Z-height
        # envelope at THIS cube_pos_b (the descent-continuity task's own
        # flagged follow-up) - reuses the already-converged pregrasp_q as a
        # common, identical starting point for every swept height (settled
        # back to it before each height, so sweep points don't compound), and
        # descends via the exact same incremental method already validated to
        # avoid the one-shot rotation-deadlock. Exits after logging every
        # swept height's residual + joint-limit-margin data.
        if args_cli.z_sweep is not None:
            z_start = pregrasp_pos_b[0, 2].item()
            sweep_heights = sorted(args_cli.z_sweep, reverse=True)
            print(f"\n[Z-SWEEP] cube_pos_b={cube_pos_b[0].tolist()} pregrasp_height(z_start)={z_start:.5f} heights={sweep_heights}")
            for h in sweep_heights:
                _settle_at(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_q, SEED_SETTLE_STEPS)
                cur_q = pregrasp_q
                last_target = None
                pos_res = rot_res = None
                for sub_idx in range(1, args_cli.num_descent_steps + 1):
                    sub_z = z_start + (h - z_start) * sub_idx / args_cli.num_descent_steps
                    sub_target = cube_pos_b.clone()
                    sub_target[:, 2] = sub_z
                    last_target = sub_target
                    cur_q, pos_res, rot_res = polish_from_seed(
                        env, ik_controller, robot_entity_cfg, ik_jacobi_idx, sub_target, target_quat_b, cur_q,
                        num_arm_joints, joint_pos_limits,
                        max_steps=DESCENT_SUBSTEP_MAX_STEPS, stagnation_break_steps=DESCENT_SUBSTEP_STAGNATION_STEPS,
                    )
                residual_vec = _measure_dist_vec(robot, robot_entity_cfg, last_target)
                live_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                lo_list = joint_pos_limits[0, :, 0].tolist()
                hi_list = joint_pos_limits[0, :, 1].tolist()
                margins = [min(q - l, hh - q) for q, l, hh in zip(live_q, lo_list, hi_list)]
                print(
                    f"[Z-SWEEP] height={h:.4f} pos_err={pos_res:.5f}m rot_err={rot_res:.4f}rad "
                    f"xyz_residual={['%.5f' % v for v in residual_vec]} "
                    f"q={['%.4f' % v for v in live_q]} margins={['%.4f' % v for v in margins]}"
                )
            print("[Z-SWEEP] done - exiting before one-shot GRASP solve / phased execution.")
            video_writer.close()
            demo_video_writer.close()
            env.close()
            return

        # 2026-07-22 (ar4-tilt-fix task): also try several WRIST-PERTURBED
        # variants of pregrasp_q (small offsets to j4/j6, the two joints
        # CANDIDATE_SEEDS never varies - it always leaves them at 0.0,
        # matching the original position-only search this pool was designed
        # for), alongside the existing single pregrasp_q/KNOWN_GOOD_GRASP_Q
        # seeds. Added after live evidence that pregrasp_q's own basin is a
        # genuine LOCAL OPTIMUM for GRASP at a tilted target - the "keep best
        # across rounds" guard correctly avoids reporting a worse result, but
        # every polish attempt from this exact seed hits the same ~2.6-4.7cm
        # ceiling (never improves past the seed's own quality) before falling
        # into an orientation-breaking branch, at 10deg, 15deg, lambda=0.02,
        # AND lambda=0.1 - consistent with this being a real feature of THIS
        # basin, not a solver tuning issue. A small nudge in the two
        # never-explored wrist DOFs gives the multi-seed search a chance to
        # land in a DIFFERENT basin whose local polish might not hit the same
        # wall - still evaluated via the same genuine settle + combined-score
        # comparison as every other candidate, not assumed better a priori.
        def _perturb(q, j4_delta=0.0, j6_delta=0.0):
            q2 = list(q)
            q2[3] += j4_delta
            q2[5] += j6_delta
            return q2

        wrist_perturbed_seeds = [
            _perturb(pregrasp_q, j4_delta=0.3),
            _perturb(pregrasp_q, j4_delta=-0.3),
            _perturb(pregrasp_q, j6_delta=0.3),
            _perturb(pregrasp_q, j6_delta=-0.3),
            _perturb(pregrasp_q, j4_delta=0.3, j6_delta=0.3),
            _perturb(pregrasp_q, j4_delta=-0.3, j6_delta=-0.3),
        ]

        if args_cli.num_descent_steps <= 1:
            # Old one-shot behavior, preserved for direct comparison against
            # the incremental descent below (--num-descent-steps 1).
            print("\n[INFO] Finding best seed for GRASP waypoint (multi-seed, genuinely settled, includes converged PREGRASP_Q + wrist-perturbed variants)...")
            seed_q, _ = _find_best_seed(
                env, robot, robot_entity_cfg, num_arm_joints, grasp_pos_b, target_quat_b, seed_j1,
                extra_full_seeds=[KNOWN_GOOD_GRASP_Q, pregrasp_q] + wrist_perturbed_seeds,
            )
            print("[INFO] Polishing GRASP waypoint (fixed-Jacobian, pose-DLS, one-shot independent target)...")
            grasp_q, grasp_residual, grasp_rot_residual = polish_from_seed(
                env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, target_quat_b, seed_q, num_arm_joints, joint_pos_limits
            )
        else:
            # 2026-07-22 (ar4-grasp-descent-continuity task): disconnected-
            # basin hypothesis test. The prior session found GRASP solved as
            # an independent one-shot target (multi-seed search, including
            # PREGRASP's own converged config as a candidate seed) deadlocks
            # at a stable ~1.1-1.4rad rotation error, tilt/damping/seed
            # independent, with no single joint pinned at a hard limit -
            # "a big jump from PREGRASP's config can't reach [GRASP's] basin
            # directly." Instead of jumping to GRASP in one shot, interpolate
            # ONLY the height (x,y and orientation are already shared between
            # PREGRASP and GRASP - see grasp_pos_b/pregrasp_pos_b above) from
            # PREGRASP's converged height down to GRASP_AT_HEIGHT in
            # args_cli.num_descent_steps increments, re-solving
            # polish_from_seed at each sub-height. Critically, this loop does
            # NOT call _settle_at/teleport between sub-steps (aside from the
            # one explicit re-settle immediately below, a safety net to
            # guarantee the descent's OWN starting point is genuinely
            # PREGRASP's converged config) - polish_from_seed's own
            # documented behavior (never teleports to its seed_q argument,
            # always continues from the robot's live state) is exactly what
            # makes chaining these calls back-to-back a genuine continuous
            # resolve through configuration space, not a series of
            # independent re-solves.
            print(
                f"\n[INFO] Descending to GRASP waypoint via {args_cli.num_descent_steps} incremental "
                "height sub-waypoints from PREGRASP's converged config (disconnected-basin hypothesis test)..."
            )
            _settle_at(env, robot, robot_entity_cfg, num_arm_joints, pregrasp_q, SEED_SETTLE_STEPS)
            z_start = GRASP_AT_HEIGHT + PREGRASP_HOVER
            z_end = GRASP_AT_HEIGHT
            grasp_q = pregrasp_q
            grasp_residual = pregrasp_residual
            grasp_rot_residual = pregrasp_rot_residual
            for sub_idx in range(1, args_cli.num_descent_steps + 1):
                sub_z = z_start + (z_end - z_start) * sub_idx / args_cli.num_descent_steps
                sub_target_pos_b = cube_pos_b.clone()
                sub_target_pos_b[:, 2] = sub_z
                grasp_q, grasp_residual, grasp_rot_residual = polish_from_seed(
                    env, ik_controller, robot_entity_cfg, ik_jacobi_idx, sub_target_pos_b, target_quat_b, grasp_q,
                    num_arm_joints, joint_pos_limits,
                    max_steps=DESCENT_SUBSTEP_MAX_STEPS, stagnation_break_steps=DESCENT_SUBSTEP_STAGNATION_STEPS,
                )
                print(
                    f"  [DESCENT {sub_idx:2d}/{args_cli.num_descent_steps}] z_target={sub_z:.5f}m "
                    f"pos_err={grasp_residual:.5f}m rot_err={grasp_rot_residual:.4f}rad"
                )

        # 2026-07-24 (ar4-grasp-ik-convergence-tightening task): optional
        # extra deep-polish pass AT GRASP's own true full-precision target
        # (grasp_pos_b, not a sub-height interpolation), continuing live
        # from whatever the descent/one-shot resolve above already converged
        # to (polish_from_seed never teleports to its seed_q argument - see
        # its own docstring - so this genuinely extends the same solve,
        # exactly like chaining descent sub-steps does). Tests directly
        # whether the ~9.5mm/4.2deg residual found at the 65deg-tilt/
        # reach=0.30-0.36m configuration shrinks further with more solver
        # effort/a tighter convergence bound, or is already a genuine
        # plateau - see this flag's own --grasp-deep-polish-steps docstring.
        if args_cli.grasp_deep_polish_steps > 0:
            deep_stagnation = args_cli.grasp_deep_polish_stagnation_steps
            if deep_stagnation is None:
                deep_stagnation = max(500, int(0.2 * args_cli.grasp_deep_polish_steps))
            pre_deep_pos, pre_deep_rot = grasp_residual, grasp_rot_residual
            print(
                f"\n[DEEP-POLISH] starting extra GRASP polish pass: max_steps={args_cli.grasp_deep_polish_steps} "
                f"stagnation_break_steps={deep_stagnation} pos_threshold={args_cli.grasp_pos_threshold} "
                f"rot_threshold={args_cli.grasp_rot_threshold} (pre-pass residual: "
                f"pos={pre_deep_pos:.5f}m rot={pre_deep_rot:.4f}rad)"
            )
            grasp_q, grasp_residual, grasp_rot_residual = polish_from_seed(
                env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, target_quat_b, grasp_q,
                num_arm_joints, joint_pos_limits,
                max_steps=args_cli.grasp_deep_polish_steps, stagnation_break_steps=deep_stagnation,
                pos_threshold=args_cli.grasp_pos_threshold, rot_threshold=args_cli.grasp_rot_threshold,
            )
            print(
                f"[DEEP-POLISH] done: pos={grasp_residual:.5f}m rot={grasp_rot_residual:.4f}rad "
                f"(was pos={pre_deep_pos:.5f}m rot={pre_deep_rot:.4f}rad before this pass - "
                f"delta pos={grasp_residual - pre_deep_pos:.5f}m rot={grasp_rot_residual - pre_deep_rot:.4f}rad)"
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
        # 2026-07-23 (ar4-jaw-bisector-hypothesis task): the robot is now
        # genuinely settled AT grasp_q (gripper OPEN, per _check_orientation_at's
        # own settle loop) - the exact configuration one of the capstone
        # session's 3 failed grasp+lift attempts used. Measure the real jaw
        # fingertip bisector here, against both the arm's own _EE_OFFSET
        # assumption and the cube's TRUE captured position (cube_init_pos,
        # captured before parking, above) - this is the core test this task
        # was dispatched to run.
        _measure_jaw_bisector_vs_ee_offset("GRASP_Q (converged, this run's actual grasp target)", cube_pos_w=cube_init_pos)
        _check_orientation_at(pregrasp_q, "PREGRASP_Q")

        # Un-park the cube: move it from _CUBE_PARK_POS_W back to its
        # captured real initial pose, right before starting the actual
        # phased grasp attempt (Phase 0 commands HOME_Q, moving the arm away
        # from the cube before it ever approaches it again, so there is no
        # instant-of-restore interpenetration risk here either).
        cube_pos_now = cube.data.root_pos_w[0].tolist()
        print(f"[INFO] Cube pose before un-park: pos={cube_pos_now}")
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

        # 2026-07-23 (coordinator-directed, mid-task correction): explicit
        # per-phase snapshot directory - a cropped, zoomed still frame of the
        # ACTUAL demo_camera view saved at each phase's midpoint, so the
        # gripper's real visual state can be directly cross-checked against
        # the numeric _print_gripper_state readout at the SAME instant,
        # instead of guessing which frame index of the full video corresponds
        # to which phase after the fact.
        snapshot_dir = os.path.join(LOG_DIR, "videos", f"ar4_grasp_gripper_check{_video_suffix}")
        os.makedirs(snapshot_dir, exist_ok=True)

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
            _print_gripper_state(f"PHASE {phase_idx} START (commanded={'OPEN' if gripper_cmd > 0 else 'CLOSE'})")
            for i in range(duration):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = target_q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))
                demo_rgb = demo_camera.data.output["rgb"][0].cpu().numpy()
                demo_video_writer.append_data(demo_rgb[:, :, :3].astype("uint8"))

                if i == duration // 2:
                    # Numeric readout AND a saved still frame at the exact
                    # same physics step, so the two can be directly
                    # cross-checked against each other (coordinator-directed).
                    _print_gripper_state(f"PHASE {phase_idx} MIDPOINT (step {i})")
                    imageio.imwrite(
                        os.path.join(snapshot_dir, f"phase{phase_idx}_mid_demo.png"), demo_rgb[:, :, :3].astype("uint8")
                    )
                    imageio.imwrite(
                        os.path.join(snapshot_dir, f"phase{phase_idx}_mid_perception.png"), rgb[:, :, :3].astype("uint8")
                    )

                if phase_idx in (2, 3, 4, 5, 6) and i % 20 == 0:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    # 2026-07-23 (ar4-capstone-grasp task): jaw contact forces
                    # alongside cube pose - this project's own Experiment 16
                    # precedent (a video that LOOKED like a lift but was
                    # actually the object wedged, not grasped) is exactly why
                    # a real grasp+lift claim needs contact-force evidence,
                    # not just cube-z-height video review. jaw1_contact/
                    # jaw2_contact are filtered against the Cube prim only
                    # (tasks/ar4/pickplace_mirror_env_cfg.py), so a nonzero
                    # reading here is unambiguously cube contact, not e.g.
                    # ground/table contact.
                    jaw1_force = torch.linalg.vector_norm(
                        env.scene["gripper_jaw1_contact"].data.force_matrix_w.view(1, 3)[0]
                    ).item()
                    jaw2_force = torch.linalg.vector_norm(
                        env.scene["gripper_jaw2_contact"].data.force_matrix_w.view(1, 3)[0]
                    ).item()
                    print(
                        f"  [PHASE {phase_idx} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]} "
                        f"jaw1_cube_force={jaw1_force:.4f}N jaw2_cube_force={jaw2_force:.4f}N"
                    )

            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            max_err = max(abs(a - t) for a, t in zip(achieved_q, target_q))
            print(f"  [PHASE {phase_idx} END] max joint error: {max_err:.5f} rad")
            _print_gripper_state(f"PHASE {phase_idx} END (commanded={'OPEN' if gripper_cmd > 0 else 'CLOSE'})")

        env.reset()

    video_writer.close()
    demo_video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")
    print(f"Demo-camera video recorded to: {demo_video_path}")
    print(f"Gripper-check snapshots recorded to: {snapshot_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Standalone Isaac Sim App-API AR4 grasp+lift — a GENUINE physics grasp via
NVIDIA's native SurfaceGripper (with a runtime PhysX fixed-joint fallback),
built on `isaacsim.core.api.World` + `SingleArticulation` + `World.step()`,
NOT Isaac Lab's `ManagerBasedRLEnv` (2026-07-29, ar4-isaacsim-standalone-pick
task).

WHY this exists — the immediately-prior task
(`scripts/ar4_isaacsim_surfacegripper_pick.py`, and
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-29 UPDATE)
established: in Isaac Sim, motion planning + trajectory control WORK, but the
grasp-HOLD fails under `ManagerBasedRLEnv` because (a) the SurfaceGripper C++
manager subscribes to physics-step events that `env.sim.step()` never fires,
and (b) runtime PhysX joints don't inject into `ManagerBasedRLEnv`'s
once-built physics views. The recommended fix, and this task: rebuild on the
LOWER-LEVEL standalone App API, where NVIDIA's own SurfaceGripper example runs
the manager correctly (`World` + `timeline.play()` + physics-step callbacks
fire on `world.step()`).

Authored directly against the actually-installed Isaac Sim 5.1.0 API — the
SurfaceGripper recipe mirrors NVIDIA's own
`isaacsim.robot.surface_gripper/tests/test_surface_gripper.py`
(robot_schema.CreateSurfaceGripper + an IsaacAttachmentPointAPI D6 joint on
link_6 + `GripperView.apply_gripper_action`/`get_gripped_objects`).

Scene geometry reproduces this repo's validated pedestal+15mm-cube scene
(tasks/ar4/objects_cfg.py) and the height-corrected grasp config
(scripts/ar4_isaacsim_surfacegripper_pick.py's POINT dict).

Run (in the isaac-lab container, headless on cloud):
    /isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py \
        [--mechanism friction] [--squeeze_mode effort|position] \
        [--squeeze_force 3.0] [--grasp_depth_extra 0.225] [--no_video]

UPDATE 2 (2026-07-30, jaw-closure-fix continued — PURE-FRICTION mandate): per
direct user directive, the grasp MUST be a genuine contact-friction hold with
NO joint/weld of ANY kind, not even as a fallback. Three coupled fixes make
that real (default `--mechanism friction`):
  1. GRASP HEIGHT. The prior run's jaws closed ~11mm ABOVE the cube center
     (fingertip world-z 0.0588 vs cube center 0.0475), catching the top edge,
     so the cube never moved (ground-truth gain 0.0mm). `--grasp_depth_extra`
     now defaults to 0.225, derived OFFLINE from the standing pure-Python FK
     framework (tasks/ar4/fk_verification.py) calibrated to that live d=0
     ground truth: it lands the fingertip midpoint at world-z=0.0475 == the
     cube's vertical CENTER (FK err 0.03mm), jaws straddling the two flat faces.
  2. REAL SQUEEZE FORCE. The jaws apply a genuine, KNOWN closing force via
     `--squeeze_mode effort` (zero the jaw position gains, apply a constant
     closing joint EFFORT every step) or `position` (drive target past the
     cube with maxForce capped to `--squeeze_force`). Default 3N -- the old
     500N maxForce would eject a 10g (0.098N) cube, which needs only ~0.12N of
     grip normal force to hold at mu=0.8.
  3. MEASURED PROOF. `contact_report()` reads the actual jaw<->cube contact
     normal force each phase (ContactSensor per jaw) -- nonzero + symmetric +
     stable through lift+retreat is the proof the squeeze is real, alongside
     the ground-truth cube pose rising with the gripper. NO SurfaceGripper /
     FixedJoint is authored or engaged in friction mode.

UPDATE (2026-07-30, jaw-closure fix task) — the immediately-prior version of
this script achieved a physics-held lift, but via a runtime PhysX FixedJoint
that engaged while the jaws were STILL VISUALLY OPEN (confirmed live from the
recorded video: the cube floats between open jaw sides, welded rather than
gripped). Root-caused to a real "missing gripper physics drive" bug (this
repo's own recurring bug class, see the kb doc's intro) -- this standalone
script bypasses Isaac Lab's `ImplicitActuatorCfg` layer, and
`articulation_controller.set_gains()` alone was NOT a reliable guarantee that
a real PhysX position drive exists on the two gripper prismatic joints, so the
commanded GRIP_CLOSED target never physically moved them. Fixed by explicitly
authoring `UsdPhysics.DriveAPI("linear")` directly on both joint prims before
reset (see the "gripper-jaw REAL physics setup" block below), plus added
world-frame jaw-separation/cube-centering instrumentation (`jaw_geometry()`)
to prove closure numerically rather than by eyeballing a frame, plus a new
`--mechanism friction` mode to genuinely retest a pure-friction hold (no
grasp-assist at all) now that the jaws can actually close on the cube, before
falling back to the grasp-assist mechanisms (now engaged only once jaws are
measured closed on the cube, not around an open gap).
"""

import argparse
import math
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Standalone Isaac Sim AR4 genuine physics grasp+lift.")
parser.add_argument("--mechanism", choices=["surface_gripper", "fixed_joint", "friction"], default="friction",
                    help="GRASP HOLD mechanism. Default is 'friction': a GENUINE contact-friction "
                         "grasp (jaws squeezing the cube's two faces) with NO joint/weld of any "
                         "kind -- per direct user directive (2026-07-30). surface_gripper/fixed_joint "
                         "are retained only as historical diagnostics, NOT to be used for the deliverable.")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--max_grip_distance", type=float, default=0.10)
parser.add_argument("--cpu", action="store_true", help="run PhysX on CPU (avoids first-time GPU-pipeline init stall)")
parser.add_argument("--stage", choices=["ground", "scene", "robot_noxform", "full"], default="full",
                    help="diagnostic: build scene incrementally to isolate a reset hang")
parser.add_argument("--grasp_depth_extra", type=float, default=0.0,
                    help="OPTIONAL extra VERTICAL descent past the IK grasp pose, in units of "
                         "VERT_DESCENT_DIR (1.0 == 11.1mm straight down). Default 0 (the IK pose "
                         "already targets cube center); the in-run closed loop handles residual. "
                         "[legacy help follows] extrapolate GRASP_Q further past PREGRASP_Q by this fraction of the "
                         "PREGRASP->GRASP joint-space vector (e.g. 0.3 = 30%% deeper than GRASP_Q) "
                         "-- a bounded numeric tuning knob for the measured jaw/cube height gap, "
                         "reusing the SAME validated descent direction rather than a fresh IK solve. "
                         "DEFAULT 0.225 is derived offline from the standing pure-Python FK framework "
                         "(tasks/ar4/fk_verification.py), CALIBRATED to the live d=0 ground truth "
                         "(achieved fingertip world-z=0.0588m at GRASP_Q): d=0.225 lands the "
                         "fingertip midpoint at world-z=0.0475m == the 15mm cube's vertical CENTER "
                         "(FK err 0.03mm), <2mm horizontal shift, so the jaws straddle the two "
                         "opposing FLAT faces at mid-height instead of catching the top edge. "
                         "The prior run's d=0 (11mm too high) and the earlier 0.4/0.7 sweep (too "
                         "deep, into the pedestal) both missed the cube -- see the FK derivation.")
parser.add_argument("--squeeze_mode", choices=["effort", "position"], default="position",
                    help="How the jaws apply CLOSING force onto the cube. 'position' (default): hold a "
                         "position target PAST the cube surface (toward 0 aperture) with the drive's "
                         "maxForce CAPPED at --squeeze_force, which physically delivers a CONSTANT "
                         "squeeze force = that force limit once the jaws stall against cube contact "
                         "(genuine force control via the drive force limit). 'effort': zero the jaw "
                         "position gains and apply a constant closing joint EFFORT each step -- tried "
                         "first per user directive but found NOT to move these prismatic jaws on the "
                         "standalone App API (live 2026-07-30: jaw sep stayed 28mm, contact 0N), the "
                         "API snag the directive anticipated -- so 'position' is the working default.")
parser.add_argument("--squeeze_force", type=float, default=5.0,
                    help="Constant inward (closing) squeeze force per jaw, Newtons. The 10g (0.098N) "
                         "cube needs only ~0.12N of grip normal force to hold at mu=0.8, so a few N "
                         "is a large safety margin; kept modest (default 5N, NOT the old 500N which "
                         "would eject/penetrate a 10g cube) so the small cube is squeezed, not "
                         "launched. Measured contact normal force is logged as the proof it is real.")
parser.add_argument("--droop_diagnostic", action="store_true",
                    help="DIAGNOSTIC MODE (2026-07-30, ar4-gravity-droop task). Instead of the full "
                         "pick, command the arm to the reachable GRASP_Q, settle to STEADY STATE, and "
                         "report per-arm-joint commanded-vs-achieved angle + measured joint effort vs "
                         "the drive's maxForce/effort limit, introspect the articulation's "
                         "generalized-gravity-force API, run a GRAVITY-OFF control (does the pad reach "
                         "cube-center Z with gravity disabled?), and test gravity-feedforward comp "
                         "(both signs) -- the decisive joint-level diagnostic for the persistent "
                         "11-16mm vertical droop that blocked runs 1-4. No video, no pick. Exits after.")
parser.add_argument("--gravity_comp", action="store_true",
                    help="Enable FEEDFORWARD GRAVITY COMPENSATION on the arm joints: each physics step, "
                         "query the articulation's generalized gravity force G(q) and apply it as an "
                         "additive joint EFFORT on the arm DOFs so the PD drive only handles the "
                         "residual (a plain PD controller sags under gravity because it counters "
                         "position error, not the gravitational load). The feedforward sign is "
                         "auto-calibrated live at the grasp pose (pick whichever sign drives the pad "
                         "DOWN toward the cube center). This is the fix for the droop that resisted "
                         "60x drive maxForce -- feedforward injects the holding torque directly rather "
                         "than relying on drive stiffness/force headroom.")
parser.add_argument("--assist_mode", choices=["cmd_integral", "integral", "gravity"], default="cmd_integral",
                    help="Tracking assist. 'cmd_integral' (default, RUN-7 fix): accumulate a per-arm-"
                         "joint COMMAND offset each step so achieved -> commanded FK pose, via the STABLE "
                         "position drive (the effort 'integral' ran away -- the wrist has no effective "
                         "drive to stabilize applied effort). 'integral': PI on applied EFFORT (UNSTABLE "
                         "here, kept only for the record). 'gravity': feedforward G(q) (proved a no-op -- "
                         "the gap is non-gravitational). "
                         "RUN-5 fix): a PI integrator on per-arm-joint (commanded-achieved) error, "
                         "applied as additive joint effort, ramping until the steady-state joint "
                         "tracking offset is nulled -- the cause-agnostic fix for the measured "
                         "NON-gravitational +9-16mm vertical gap (wrist joints settle 4-9deg off "
                         "command at near-zero effort: friction/weak-drive, NOT gravity, per the "
                         "droop diagnostic). 'gravity': legacy feedforward of G(q) (retained; the "
                         "diagnostic proved it does nothing here because the gap is not gravitational).")
parser.add_argument("--integral_ki", type=float, default=6.0,
                    help="Per-step integral gain for the integral tracking assist (effort += ki*err "
                         "each physics step, err in rad). Ramps applied joint effort until per-joint "
                         "position error is nulled. Heavy joint damping (ARM_KD=600) stabilizes it.")
parser.add_argument("--integral_clamp", type=float, default=400.0,
                    help="Anti-windup clamp (N*m) on each arm joint's integral effort -- well below "
                         "the 1500 N*m drive maxForce, generous vs the few-N*m loads involved.")
parser.add_argument("--cmd_kc", type=float, default=0.35,
                    help="cmd_integral gain: command_offset += kc*(commanded-achieved) each step. "
                         "Drives achieved -> commanded via the position drive. 0.35 converges in "
                         "~tens of steps without overshoot given the position drive's own damping.")
parser.add_argument("--cmd_clamp_deg", type=float, default=25.0,
                    help="Clamp (deg) on each arm joint's accumulated command offset -- bounds the "
                         "offset for a non-responsive joint (e.g. wrist roll) so it can't wind up "
                         "unboundedly, while easily covering the few-deg tracking errors on the "
                         "pad-controlling joints.")
parser.add_argument("--arm_armature", type=float, default=0.0,
                    help="RUN-6 fix: joint ARMATURE (reflected actuator inertia, PhysxJointAxisAPI) "
                         "authored on each arm joint before reset. The diagnostic proved the AR4 wrist "
                         "joints (4,5,6) don't hold their commanded position (near-zero drive effort at "
                         "large error, worse with higher stiffness) -- the classic low/zero distal-link "
                         "inertia drive pathology, whose textbook cure is armature. 0 = off. Try ~0.5.")
parser.add_argument("--quick", action="store_true",
                    help="Quick diagnostic: only section A (per-joint tracking at GRASP_Q) + a +5deg "
                         "per-joint command-tracking probe, then exit -- to validate the armature fix "
                         "cheaply without the full A-E battery.")
parser.add_argument("--empty_scene", action="store_true",
                    help="Place pedestal+cube 5m out of reach so nothing collides with the arm -- the "
                         "decisive test for whether GRASP_Q is blocked by an EXTERNAL scene collision "
                         "(the arm should then reach the FK pad z ~0.0562 freely) vs a self-collision.")
parser.add_argument("--side_grasp", action="store_true",
                    help="SIDE / HORIZONTAL grasp mode (2026-07-31, ar4-side-grasp task). The proven "
                         "blocker (kb 2026-07-31 UPDATE): a centered TOP-DOWN approach is collision-"
                         "blocked -- the gripper palm/body (48.5mm palm-to-fingertip depth) jams on the "
                         "cube TOP FACE ~9mm early; shallow fingers cannot straddle a 15mm cube from "
                         "directly above. This mode instead orients the gripper HORIZONTALLY (approach "
                         "axis along world +X, jaw-slide/closing axis along world Y) so the pads close "
                         "onto the cube's two VERTICAL +Y/-Y faces at mid-height and the palm sits on the "
                         "-X SIDE of the cube, never above it -- sidestepping the top collision entirely. "
                         "Uses a NARROW pedestal (narrow in Y) so the pads clear it at the cube faces. "
                         "All poses (PREGRASP/GRASP/LIFT) are FK/IK-derived with joint-limit margin "
                         "(tightest joint_5 ~12.6deg at grasp) -- see scripts/_design_ar4_side_grasp.py.")
parser.add_argument("--prism", action="store_true",
                    help="TALL-PRISM TOP-DOWN grasp (2026-07-31, ar4-tall-prism-pick task). The proven "
                         "blocker for a 15mm cube (kb 2026-07-31 UPDATEs): the gripper collision envelope "
                         "hangs 15-23mm BELOW the pad plane and its body SURROUNDS a 15mm cube on a work "
                         "surface -> geometrically infeasible top-down AND side-on. Fix (Principal decision): "
                         "grasp a PROPERLY-SIZED object. This mode replaces the pedestal+15mm-cube with a "
                         "TALL, NARROW rectangular prism resting directly on the GROUND (no pedestal), sized "
                         "so the FIXED reachable pad plane (z=0.05621 at the proven near-vertical GRASP_Q) "
                         "grips NEAR THE PRISM'S TOP: the prism is tall enough that the gripper's below-pad "
                         "underhang clears the ground (pad 56mm up, underhang <=23mm -> >=33mm clearance) and "
                         "NARROW enough (<28mm jaw aperture) to fit between the open jaws. Prism dims are "
                         "--prism_width/--prism_depth/--prism_height (the last swept cheaply physics-only to "
                         "find the height whose top clears the gripper's central body while giving a solid "
                         "pad contact band). Reuses the proven top-down GRASP_Q/servo/squeeze machinery "
                         "unchanged; only the object + the servo's grip-height target change.")
parser.add_argument("--prism_width", type=float, default=0.018,
                    help="Prism width along the jaw-SLIDE/closing axis (world X, top-down), meters. Must be "
                         "< the ~28mm open jaw aperture; 18mm leaves ~5mm clearance per side.")
parser.add_argument("--prism_depth", type=float, default=0.018,
                    help="Prism depth along the third axis (world Y, top-down), meters.")
parser.add_argument("--prism_height", type=float, default=0.056,
                    help="Prism height (rests on ground z=0, so top = this value), meters. Chosen so the "
                         "fixed pad plane (0.05621) grips near the top. Swept physics-only to clear the "
                         "gripper's central-body underhang while keeping a solid pad contact band.")
parser.add_argument("--prism_mass", type=float, default=0.025, help="Prism mass, kg.")
parser.add_argument("--pin_pick", action="store_true",
                    help="SUPPORT-FREE THIN-PIN top-down grasp (2026-07-31, ar4-support-free-pin-pick "
                         "task). The DEFINITIVE blocker (kb 2026-07-31 UPDATEs, proven by "
                         "--dump_gripper_aabb): the AR4 gripper's central body (gripper_base_link) bottoms "
                         "only ~1.3mm BELOW the pad plane, so it CANNOT straddle ANY object resting on a "
                         "support surface -- the body hits the object top / the fingers hit the support -- "
                         "in BOTH top-down and side-on. Arm/drives/tracking/servo/squeeze/jaw-closure are "
                         "ALL proven flawless. Fix (Principal decision): present the object SUPPORT-FREE. "
                         "This mode holds a small object up on a THIN VERTICAL PIN (much thinner than the "
                         "gripper's jaw clearance) so there is NO support surface in the grasp region for "
                         "the body/fingers to collide with. The object's TOP is placed just BELOW the "
                         "gripper body's z_min (so the descending body clears it) and its faces span down "
                         "through the finger/pad contact band; the pin rises up the CENTER (between the two "
                         "closing jaws, below the fingertips) and bears the object's weight until the grasp "
                         "lifts it OFF. Reuses the proven near-vertical top-down GRASP_Q/servo/squeeze/"
                         "friction machinery unchanged; only the object presentation changes. Recommended "
                         "with --no_servo (the Jacobian probe would bump the object).")
parser.add_argument("--pin_obj_width", type=float, default=0.015,
                    help="Object width along the jaw-SLIDE/closing axis (world X, top-down), meters. Must "
                         "be < the ~28mm open jaw aperture; 15mm leaves the open jaws clearance to descend "
                         "beside it and close onto its two +/-X faces.")
parser.add_argument("--pin_obj_depth", type=float, default=0.018,
                    help="Object depth along the third axis (world Y, top-down), meters.")
parser.add_argument("--pin_obj_height", type=float, default=0.020,
                    help="Object height, meters. The object hangs DOWN from its top face; a taller object "
                         "gives a taller pad-face contact band (better friction) but its bottom must stay "
                         "above the pin top. 20mm spans the finger contact band comfortably.")
parser.add_argument("--pin_obj_top", type=float, default=0.0510,
                    help="World-Z of the object's TOP face, meters. MUST be below the gripper body's z_min "
                         "(~0.0548 at the reachable top-down GRASP_Q) so the descending central body clears "
                         "it -- the whole point of the support-free presentation. Default 0.0510 gives "
                         "~3.8mm body clearance while the object's faces (0.031..0.051) overlap the finger/"
                         "pad contact band (~0.038..0.056). Swept physics-only to pick the highest top that "
                         "clears the body yet gives solid symmetric contact.")
parser.add_argument("--pin_obj_mass", type=float, default=0.015, help="Object mass, kg (10g cube analog).")
parser.add_argument("--pin_size", type=float, default=0.009,
                    help="Thin support-pin cross-section (square), meters. Must be small enough to clear "
                         "the CLOSING jaws (which reach ~+/-7.5mm on a 15mm object) and rise between the "
                         "fingertips -- 9mm (+/-4.5mm) leaves ~3mm clearance per side. The pin is a static "
                         "kinematic collider from the ground up to the object bottom; the gripper body/"
                         "fingers genuinely clear it (verify via --dump_gripper_aabb --pin_pick).")
parser.add_argument("--jaw_max_force", type=float, default=50.0,
                    help="Jaw prismatic drive maxForce CEILING (N). The closing force is min(GRIP_KP*error, "
                         "this). 50N (old default) + heavy damping left the jaws stalling BEFORE reaching a "
                         "narrow object (2026-07-31 tall-prism sweep: jaw1 crept to dof 0.011 and stalled, "
                         "jaw2 never moved) -> no symmetric grip. Raise to firmly/quickly close both jaws.")
parser.add_argument("--grip_kd", type=float, default=200.0,
                    help="Jaw prismatic drive damping. 200 is ~6x overdamped for the light jaws and makes "
                         "them creep; lower it so both jaws actually reach the object within the close steps.")
parser.add_argument("--close_steps", type=int, default=40,
                    help="Physics steps per jaw-close sub-phase (there are ~3). Raise so the jaws finish "
                         "closing onto the object rather than stalling mid-travel.")
parser.add_argument("--no_servo", action="store_true",
                    help="Skip the empirical-Jacobian centering servo and grasp DIRECTLY at GRASP_Q "
                         "(2026-07-31): the arm tracks GRASP_Q to the object XY to <0.02deg in free space "
                         "(the 'droop' was a proven phantom), so the servo is unnecessary -- and its "
                         "6-joint Jacobian PROBE (+/-0.9deg jostles) BUMPS a tall ground-resting object with "
                         "the open jaws' tight clearance, sliding it ~18mm off-center so only one jaw grips "
                         "(tall-prism retry root cause). Descend PREGRASP->GRASP and close, no probing.")
parser.add_argument("--jaw_close_test", action="store_true",
                    help="DIAGNOSTIC (2026-07-31): at HOME (gripper in free space, no object contact), "
                         "command the jaws CLOSED with the real squeeze and log BOTH jaw joint positions "
                         "over 200 steps -> a clean control for whether BOTH jaw drives physically close to "
                         "~0 in free space (rules out a one-jaw-stuck drive bug vs an object-interaction "
                         "centering failure -- the tall-prism sweep showed jaw2 never moving). Exits after.")
parser.add_argument("--dump_gripper_aabb", action="store_true",
                    help="DIAGNOSTIC (2026-07-31, side-grasp): drive to GRASP_Q, settle, and dump the "
                         "WORLD-frame collision-mesh AABB (min/max xyz) of gripper_base_link + both jaw "
                         "links + link_6, plus the pad midpoint -- to measure exactly how far the gripper "
                         "BODY hangs below the pad plane in the grasp orientation (the suspected support-"
                         "collision cause: FK origins clear the support but the mesh body does not). Use "
                         "with --empty_scene so the arm reaches the true pose unobstructed. Exits after.")
args_cli = parser.parse_args()

# enable_cameras (and thus the RTX render pipeline) ONLY when capturing video.
# The first render-pipeline init on a fresh cloud instance is a known ~10-min+
# stall (docs/cloud/dispatch-checklist.md); a physics-only ground-truth run
# must not pay it.
RENDER = not args_cli.no_video

# Kit init on this 4-vCPU instance deadlocks nondeterministically at
# startup/World()/reset() with "carb.tasking is likely stuck" (a Kit
# thread-pool deadlock: all worker threads blocked waiting on a task that
# needs a free thread). Enlarging the tasking pool and disabling async
# rendering are the documented mitigations. SimulationApp forwards leftover
# sys.argv to Kit, so inject these as Kit CLI settings.
import sys  # noqa: E402
sys.argv += [
    "--/plugins/carb.tasking.plugin/threadCount=16",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
]
simulation_app = SimulationApp({"headless": True, "enable_cameras": RENDER})

# breadcrumb: written from the very first line so a hang during the heavy
# isaac imports below is pinpointable (Kit swallows stdout). NOTE: logs/ does
# not exist in a fresh git-archive checkout (gitignored, empty dirs not
# shipped) -- create it BEFORE opening any log file here.
os.makedirs("/workspace/rl/logs", exist_ok=True)
_BC = open("/workspace/rl/logs/standalone_pick_breadcrumb.txt", "w")


def _bc(s):
    _BC.write(s + "\n"); _BC.flush()


_bc("app_constructed")
import numpy as np  # noqa: E402
_bc("numpy")
import omni.timeline  # noqa: E402
_bc("omni.timeline")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema  # noqa: E402
_bc("pxr")
from usd.schema.isaac import robot_schema  # noqa: E402
_bc("robot_schema")
from isaacsim.core.api import World  # noqa: E402
_bc("World")
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
_bc("objects")
from isaacsim.core.prims import SingleArticulation  # noqa: E402
_bc("SingleArticulation")
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
_bc("stage_utils")
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
_bc("ArticulationAction")
from isaacsim.robot.surface_gripper import GripperView  # noqa: E402
_bc("GripperView")
from isaacsim.robot.surface_gripper._surface_gripper import GripperStatus  # noqa: E402
_bc("GripperStatus_imports_done")


# --- self-contained quaternion/rotation helpers (wxyz), numpy -------------
# (deliberately NOT `import isaaclab.*` -- importing the isaaclab package in a
# raw standalone SimulationApp hangs before main() even starts; confirmed live
# 2026-07-29. Keep this script dependency-free of the Isaac Lab layer.)
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / n


def quat_apply(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_inv(q))[1:]


def mat_to_quat_wxyz(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def lookat_quat_opengl(eye, target):
    """USD/OpenGL camera orientation quat (wxyz): camera -Z -> target, +Y up."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, fwd)
    # camera basis: X=right, Y=up, Z=-fwd (OpenGL)
    R = np.column_stack([right, up, -fwd])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

# ----------------------------------------------------------------------------
# Validated scene + grasp geometry (see module docstring for provenance).
USD_PATH = "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd"
AR4_ROOT = "/World/AR4"
ART_ROOT = "/World/AR4/root_joint/root_joint"
LINK6_PATH = "/World/AR4/root_joint/link_6"
GRIP1_PATH = "/World/AR4/root_joint/gripper_jaw1_link"
GRIP2_PATH = "/World/AR4/root_joint/gripper_jaw2_link"
CUBE_PATH = "/World/Cube"

# SCENE + GRASP CONFIG (2026-07-30 REACHABLE-CONFIG FIX, ar4-reachable-friction-
# grasp task). The prior hand-tuned GRASP_Q was a near-HORIZONTAL branch
# (~79deg from vertical) at cube Y~0.39, near AR4's full forward extension:
# its jaw LINK origins bottomed out at z~0.0686m -- ~21mm ABOVE the cube center
# -- so the jaws closed above the cube (0N contact, cube never moved). That was
# a GEOMETRIC reach limit, not a servo/squeeze problem (kb 2026-07-30
# "PURE-FRICTION grasp attempt -- ground-truth reach-limit blocker").
#
# THIS config comes from the standing pure-FK graspable-workspace sweep
# (scripts/ar4_graspable_workspace.py, 8M-sample near-vertical solve) then
# re-verified offline with the CORRECTED pad geometry (12.5mm along each jaw
# link's LOCAL -Y, the measured pad centroid -- NOT the old 18.475mm world-Z
# model). At this config the arm approaches near-VERTICALLY (tilt 5.55deg from
# straight-down) with 36deg+ joint-limit margin on EVERY joint, and the true
# jaw-pad MIDPOINT lands at world (-0.12246, 0.37247, 0.05621). CUBE_XY is set
# to that pad-midpoint XY (pads centered on the cube horizontally) and
# PEDESTAL_HEIGHT so the cube's vertical CENTER == the pad-midpoint z -- i.e.
# the pads straddle the two flat faces at mid-height in ALL THREE axes at the
# nominal grasp pose (FK err 0.000mm x/y/z), leaving the in-run servo only
# residual gravity-droop to trim. See scratchpad verify_config.py derivation
# recorded in the kb 2026-07-30 reachable-config UPDATE.
# RUN-4 STRATEGY (2026-07-30, "fix the DROOP at the actuator, not the scene").
# Runs 1-3 all failed at PLACEMENT (0N contact every time): the live pad floats
# ~10-16mm ABOVE the commanded descent and ~9mm off in Y, and neither commanding
# deeper (run 2, pad moved only 3.4mm of 16mm) nor raising the cube to the
# drooped pose (run 3, the taller cube propped the gripper 12mm HIGHER) helped.
# KEY PHYSICS INSIGHT: a 16mm origin-level droop is FAR larger than steady-state
# PD error should give for this small arm's gravity torque at KP=8000 (that
# predicts sub-mm) -- the arm joints' drive FORCE/EFFORT LIMIT is saturating, so
# the drive physically cannot hold the arm against gravity at this pose. Fix at
# the actuator: author high-maxForce angular drives on the ARM joints (+ firmer
# gains) so the arm actually HOLDS its commanded descent, and return the cube to
# the FK-cube-center low-pedestal scene so a now-tracking arm lands the pad at
# the cube center (FK 0.05621). See the arm-drive authoring block below.
PEDESTAL_CENTER_XY = (-0.12246, 0.37247)
PEDESTAL_FOOTPRINT = (0.30, 0.14)
PEDESTAL_HEIGHT = 0.04871
CUBE_XY = (-0.12246154740662964, 0.37247094274942993)
CUBE_SIZE = 0.015
CUBE_REST_Z = PEDESTAL_HEIGHT + CUBE_SIZE / 2.0  # 0.05621 == FK pad-midpoint z

# NOMINAL near-vertical GRASP_Q (the FK-reachable config; the cube is now placed
# at THIS pose's live drooped pad position -- see the SCENE block above). Run-2's
# droop-pre-compensated deeper config is retired: the pad could not actually
# descend to it (torque-limited floor), so the fix moved the CUBE up to the pad
# instead of the pad down to the cube.
GRASP_Q_DEG = [-19.4595, 53.2382, 14.9828, -14.9277, 21.9517, 114.6585]
PREGRASP_Q_DEG = [-19.5117, 47.5355, 16.5996, -14.892, 22.9696, 114.6585]
HOME_Q = [0.0] * 6
GRASP_Q_BASE = [math.radians(d) for d in GRASP_Q_DEG]
PREGRASP_Q = [math.radians(d) for d in PREGRASP_Q_DEG]

# VERTICAL-DESCENT IK GRASP POSE. At the base GRASP_Q the gripper sits directly
# above the cube (fingertip X,Y dead-on the cube) but ~11mm too HIGH. The
# earlier fix extrapolated GRASP_Q along the PREGRASP->GRASP direction to
# descend -- but that direction is DIAGONAL: live 2026-07-30 showed it lowered
# the fingertip ~8mm while shoving it ~15-18mm FORWARD in Y, PAST the cube, so
# the cube ended up beside (not between) the jaws (0 contact, scoop artifact).
# IK_GRASP_Q_DEG is instead a LOCAL DLS-IK solution (derived offline via the
# standing pure-Python FK framework tasks/ar4/fk_verification.py, converged to
# 0.002mm) that lowers the fingertip 11.1mm STRAIGHT DOWN to the cube's center
# Z while holding X,Y fixed on the cube (drift 0.00mm) -- a tiny perturbation
# of GRASP_Q (joint_2 +2.09deg, joint_3 -0.74deg, joint_5 -0.38deg), so it
# stays well within reach. VERT_DESCENT_DIR (IK - base) is that pure-vertical
# 11.1mm descent direction in joint space, reused by the in-run closed loop to
# correct any residual gravity-droop under-tracking WITHOUT drifting off the
# cube horizontally.
# REACHABLE-CONFIG FIX (2026-07-30): the recommended near-vertical GRASP_Q
# above ALREADY places the pad midpoint at the cube center (FK-verified 0.000mm
# in x/y/z), so no artificial vertical-descent nudge is needed. Set IK_GRASP_Q
# == GRASP_Q so VERT_DESCENT_DIR ~= 0 and the legacy grasp_depth_extra knob is
# a no-op by default; the in-run empirical-Jacobian servo below does the real
# (droop-compensating) final centering.
IK_GRASP_Q_DEG = list(GRASP_Q_DEG)
IK_GRASP_Q = [math.radians(d) for d in IK_GRASP_Q_DEG]
VERT_DESCENT_DIR = [ik - b for ik, b in zip(IK_GRASP_Q, GRASP_Q_BASE)]  # ~11.1mm world descent
VERT_DESCENT_M = 0.0111  # world-meters of fingertip descent for +1.0 of VERT_DESCENT_DIR
GRASP_Q = list(IK_GRASP_Q)  # grasp target = vertical-descent-to-cube-center pose
if args_cli.grasp_depth_extra != 0.0:  # optional extra vertical nudge (same dir)
    _k = args_cli.grasp_depth_extra
    GRASP_Q = [g + _k * d for g, d in zip(IK_GRASP_Q, VERT_DESCENT_DIR)]

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIP_JOINTS = ["gripper_jaw1_joint", "gripper_jaw2_joint"]
GRIP_OPEN = 0.014
GRIP_CLOSED = 0.0

# Arm gains. HISTORY: 12000/600 OSCILLATED at the deep grasp pose (2026-07-30),
# and 7000/350 left the fingertip unable to reach the cube's mid-height at a
# graspable XY. But raising stiffness to 11000 (run 6) did NOT lower that
# frontier -- it's GEOMETRIC (the gripper approached tilted, so descending
# dragged X/Y), not a droop-force wall -- and 11000's residual oscillation
# actually corrupted the servo's online Jacobian. The real fix is the 6-DOF
# orientation-HOLDING servo below (vertical descent), which works best with a
# MODERATE, well-damped gain: keep stiffness clean so the fingertip/orientation
# measurements feeding the Broyden update are quiet. 8000/600 (KD/KP=0.075).
# RUN-4: firmer arm gains + high-maxForce arm drives (see the arm-drive
# authoring block) to eliminate the ~16mm gravity droop that blocked runs 1-3.
# RUN-5 (2026-07-30, ar4-gravity-droop task): run 4 PROVED brute force is the
# wrong lever -- 60x maxForce (3000 N*m) + 2.5x stiffness (20000) cut the droop
# only ~30% AND its extreme stiffness introduced a gripper-won't-close solver
# artifact and oscillation that corrupted the servo's online Jacobian. The
# principled fix is FEEDFORWARD GRAVITY COMPENSATION (--gravity_comp): inject
# G(q) as joint effort so the PD holds pose with ~0 steady-state error at MODERATE
# gains. So revert to the documented clean, well-damped 8000/600 (KD/KP=0.075) --
# the choice the servo measures best at -- with a healthy (not extreme) maxForce
# ceiling. Gravity comp, not stiffness, closes the droop now.
ARM_KP, ARM_KD = 8000.0, 600.0
ARM_MAX_FORCE = 1500.0  # N*m per arm joint drive -- generous ceiling; feedforward carries the gravity load
# GRIP_KP sets the closing/squeeze force: force = GRIP_KP * jaw position error.
# At cube contact each jaw sits ~0.0065m short of its 0-target, so 5000*0.0065
# ~= 33N squeeze -- firm, well above the ~0.12N needed for the 10g cube, but
# far from the 500N that risks penetrating the small cube. The squeeze is
# delivered by this PD drive directly (a proven closing path -- the prior task
# closed the jaws this way); the earlier RUNTIME DriveAPI maxForce mutation is
# removed (it desynced the articulation drive and left the jaws stuck open).
GRIP_KP, GRIP_KD = 5000.0, args_cli.grip_kd

JAW1_LOWER_EXTENT = 0.018475  # fingertip offset below jaw link origin

# ---- SIDE / HORIZONTAL grasp config (2026-07-31, ar4-side-grasp task) -------
# Overrides the near-vertical top-down scene+poses above with a horizontal side
# grasp that sidesteps the proven palm-vs-cube-top collision (kb 2026-07-31
# UPDATE). Derived offline via scripts/_design_ar4_side_grasp.py (pure-FK/IK on
# tasks/ar4/fk_verification.py's vendor-URDF chain + the measured 12.5mm local-Y
# pad centroid). Geometry:
#   - approach axis = world +X (link_6/palm on the -X side of the cube, fingers
#     reach +X toward the cube); closing/jaw-slide axis = world Y (pads grip the
#     cube's +Y and -Y VERTICAL faces at mid-height).
#   - cube center world = (-0.12, 0.28, 0.09); pedestal top z = 0.0825 (cube
#     half-height 7.5mm below center). The whole gripper sits at z~0.09, 7.5mm
#     above the pedestal top, with the finger LONG axis pointing horizontally
#     (+X), so the fingers do not dip toward the pedestal.
#   - NARROW pedestal (30mm X x 10mm Y): narrower than the 15mm cube in Y, so the
#     pads closing onto the +/-7.5mm cube faces clear the pedestal top laterally.
#   - joint-limit margins (deg): GRASP tightest joint_5=12.6; PREGRASP j5=8.8;
#     LIFT(+50mm) j5=7.9. All comfortably inside limits.
SIDE_GRASP = args_cli.side_grasp
SIDE_LIFT_Q_DEG = [-36.8047, 51.8462, 26.3231, -53.7862, 97.0561, -9.5172]  # +50mm up
SIDE_LIFT_Q = [math.radians(d) for d in SIDE_LIFT_Q_DEG]
if SIDE_GRASP:
    GRASP_Q_DEG = [-36.8089, 62.1258, 23.7873, -53.2924, 92.4411, -3.2426]
    PREGRASP_Q_DEG = [-41.132, 62.6465, 17.8947, -49.2824, 96.1933, -7.1265]
    GRASP_Q_BASE = [math.radians(d) for d in GRASP_Q_DEG]
    PREGRASP_Q = [math.radians(d) for d in PREGRASP_Q_DEG]
    IK_GRASP_Q_DEG = list(GRASP_Q_DEG)
    IK_GRASP_Q = [math.radians(d) for d in IK_GRASP_Q_DEG]
    VERT_DESCENT_DIR = [ik - b for ik, b in zip(IK_GRASP_Q, GRASP_Q_BASE)]  # ~0
    GRASP_Q = list(IK_GRASP_Q)
    if args_cli.grasp_depth_extra != 0.0:
        _k = args_cli.grasp_depth_extra
        GRASP_Q = [g + _k * d for g, d in zip(IK_GRASP_Q, VERT_DESCENT_DIR)]
    # scene: cube on a NARROW pedestal, gripper approaches horizontally from -X
    PEDESTAL_CENTER_XY = (-0.12, 0.28)
    PEDESTAL_FOOTPRINT = (0.03, 0.010)   # (legacy; unused in side mode -- see plate below)
    PEDESTAL_HEIGHT = 0.0825
    CUBE_XY = (-0.12, 0.28)
    CUBE_REST_Z = PEDESTAL_HEIGHT + CUBE_SIZE / 2.0  # 0.09 == FK pad-midpoint z
# SIDE-grasp support: a THIN FLOATING PLATE, NOT a ground-rooted pedestal.
# RUN-1 (2026-07-31) proved (via the free-space vs obstacles A/B diagnostic) that
# a tall 82.5mm ground-rooted pedestal RAMS the arm mid-swing: joint_5 jammed 28deg
# short (64 vs 92 commanded), joint_1/2/3 pushing 88-138 N*m, pad 40mm low -- while
# the SAME horizontal GRASP_Q in FREE SPACE tracks to <0.02deg and lands the pads
# dead-on the cube center. The blocker is purely the tall support wall in the
# horizontal approach/swing corridor. Fix: a thin static plate whose TOP is the
# cube's rest surface (z=0.0825) but which is only ~10mm thick (a FixedCuboid is
# kinematic -- it can float, no ground rooting needed), positioned UNDER + toward
# +X of the cube so it (a) supports the cube's footprint and (b) leaves the -X
# horizontal approach corridor open and never rises as a wall the arm swings into.
SIDE_SUPPORT_CENTER = (-0.10, 0.28, 0.0775)   # plate center; top at 0.0825, bottom 0.0725
SIDE_SUPPORT_SCALE = (0.06, 0.06, 0.010)       # 60x60x10mm thin plate; covers cube footprint (X -0.13..-0.07)

# ---- TALL-PRISM top-down grasp config (2026-07-31, ar4-tall-prism-pick task) -
# The measured, definitive blocker (kb 2026-07-31 UPDATEs): the AR4 gripper's
# collision envelope hangs 15-23mm below the pad plane and SURROUNDS a 15mm cube
# on a work surface, so a pure-friction grasp of a 15mm cube is geometrically
# infeasible both top-down (palm jams on cube top) and side-on (body jams on the
# support). Fix (Principal decision): grasp a PROPERLY-SIZED object. Here: a TALL
# NARROW rectangular prism resting directly on the GROUND, reusing the SAME proven
# near-vertical top-down GRASP_Q (which lands the pad midpoint dead-on
# (-0.12246, 0.37247, 0.05621) with all joints tracking <0.02deg in free space --
# empty_scene-confirmed). The prism is:
#   - NARROW (18mm along the jaw-slide axis X) -> fits between the 28mm open jaws
#     with ~5mm clearance per side, so the open jaws descend BESIDE it, not onto it.
#   - TALL (height ~= the 56mm pad plane) -> its top reaches the fixed pad plane so
#     the pads grip near the top, while its body extends down to the ground, putting
#     the whole grasp 56mm up where the gripper's <=23mm below-pad underhang clears
#     the ground by >=33mm. No support/pedestal to jam the body on (the cube's
#     side-on blocker), and nothing above the pads for the palm to jam on
#     (the cube's top-down blocker).
# PRISM_GRIP_Z is the reachable pad-plane z the servo targets (unchanged from the
# cube's CUBE_REST_Z, which WAS this exact pad plane). Prism dims are CLI-swept.
TALL_PRISM = args_cli.prism
PRISM_GRIP_Z = 0.05621  # measured pad-midpoint z at the reachable top-down GRASP_Q
if TALL_PRISM:
    PRISM_W = args_cli.prism_width
    PRISM_D = args_cli.prism_depth
    PRISM_H = args_cli.prism_height
    PRISM_CENTER_Z = PRISM_H / 2.0      # rests on ground (bottom z=0), so center = H/2
    PRISM_XY = (-0.12246154740662964, 0.37247094274942993)  # == pad-midpoint XY at GRASP_Q

# ---- SUPPORT-FREE THIN-PIN top-down grasp config (2026-07-31, ar4-support-free-
# pin-pick task). Reuses the SAME proven near-vertical top-down GRASP_Q (pad
# midpoint lands dead-on (-0.12246, 0.37247, 0.05621) with all joints tracking
# <0.02deg in free space -- empty_scene-confirmed). Measured gripper geometry at
# that pose (--dump_gripper_aabb, kb 2026-07-31 tall-prism UPDATE):
#   pad_mid z = 0.05621; gripper_base_link z_min = 0.0548 (1.3mm below pad plane);
#   jaw/finger links z_min ~0.038 (18mm below pad); link_6 z_min 0.074.
# The object is presented SUPPORT-FREE on a thin central pin:
#   - object TOP (PIN_OBJ_TOP) placed BELOW the body z_min (0.0548) so the
#     descending central body clears it (this is the exact 1.3mm-window blocker
#     that made every surface-supported object fail; removing the support does
#     NOT remove it -- placing the object top below the body does);
#   - object faces span DOWN from the top through the finger/pad contact band
#     (~0.038..0.056) so the closing jaws grip the two +/-X faces by real flat-
#     face friction;
#   - a THIN square pin (PIN_SIZE) rises up the CENTER from the ground to the
#     object bottom, thin enough to clear the closing jaws (+/-7.5mm) and rise
#     between the fingertips; it bears the object's weight until the grasp lifts
#     it OFF the pin.
PIN_PICK = args_cli.pin_pick
if PIN_PICK:
    PIN_OBJ_W = args_cli.pin_obj_width       # X, jaw-slide/closing axis
    PIN_OBJ_D = args_cli.pin_obj_depth       # Y
    PIN_OBJ_H = args_cli.pin_obj_height      # vertical
    PIN_OBJ_TOP = args_cli.pin_obj_top       # world-Z of object top face
    PIN_OBJ_CENTER_Z = PIN_OBJ_TOP - PIN_OBJ_H / 2.0
    PIN_OBJ_BOTTOM = PIN_OBJ_TOP - PIN_OBJ_H
    PIN_XY = (-0.12246154740662964, 0.37247094274942993)  # == pad-midpoint XY at GRASP_Q
    PIN_SZ = args_cli.pin_size
    PIN_HEIGHT = PIN_OBJ_BOTTOM               # pin spans [0, object bottom]
    PIN_CENTER_Z = PIN_OBJ_BOTTOM / 2.0
    # servo grip target: the object's own vertical CENTER (real face midpoint),
    # clamped to the reachable pad plane -- but with --no_servo the arm just goes
    # to GRASP_Q (pad_mid 0.05621) and closes; the object faces (below the body)
    # are gripped by the sub-centroid finger band.
    PIN_GRIP_Z = PIN_OBJ_CENTER_Z
    # LIFT pose: extrapolate along the proven, near-vertical PREGRASP->GRASP
    # descent direction (mostly joint_2 shoulder-pitch) to raise the gripper
    # up-and-back, clearly lifting the object OFF the thin pin. 3x PREGRASP's
    # offset gives a solid (~several-cm) lift; the arm has 30deg+ joint-limit
    # margin here so it stays reachable. Retreat then returns to PREGRASP (still
    # well up off the pin, object still gripped) rather than the violent all-the-
    # way-to-HOME retreat, so the friction hold is demonstrated stably.
    PIN_LIFT_Q = [g + 3.0 * (p - g) for g, p in zip(GRASP_Q_BASE, PREGRASP_Q)]

RESULT_PATH = f"/workspace/rl/logs/standalone_pick_result_{args_cli.mechanism}.txt"
VIDEO_DIR = f"/workspace/rl/logs/videos/ar4_isaacsim_standalone_pick/{args_cli.mechanism}"
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
_R = open(RESULT_PATH, "w")


def log(s):
    _R.write(str(s) + "\n")
    _R.flush()
    print(str(s), flush=True)


# ----------------------------------------------------------------------------
def link_world_pose(stage, path):
    """(pos[3], quat_wxyz[4]) world pose of a prim via USD xform cache."""
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    imag = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]]),
            np.array([q.GetReal(), imag[0], imag[1], imag[2]]))


def find_prim_by_name(stage, root_path, name):
    """First prim under root_path (inclusive) whose own name == `name`,
    found by plain traversal -- robust to not knowing the exact nested
    path a joint/link prim lives at inside the referenced USD asset."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for p in Usd.PrimRange(root):
        if p.GetName() == name:
            return p
    return None


def make_physics_material(stage, path, static_friction, dynamic_friction, restitution, combine_mode="max"):
    """Author a UsdPhysics material prim directly (bypassing the
    isaacsim.core.api.materials.PhysicsMaterial wrapper class, whose exact
    attribute names aren't confirmed against this specific Isaac Sim build
    -- raw pxr.UsdPhysics/PhysxSchema calls are the stable, documented
    layer). `combine_mode` deliberately defaults to "max" (not PhysX's
    "average" default, and NEVER "min") so a low friction value authored on
    either side of a contact pair can't silently cap the effective
    friction -- this repo's own dispatch brief flagged exactly this risk."""
    mat = UsdShade.Material.Define(stage, Sdf.Path(path))
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(restitution)
    physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    physx_mat_api.CreateFrictionCombineModeAttr().Set(combine_mode)
    return mat.GetPrim()


def bind_physics_material(stage, prim_path, material_prim):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), materialPurpose="physics")
    return True


def apply_contact_offsets_under(stage, root_path, contact_offset, rest_offset):
    """Set contact/rest offset on every collision-API prim found under
    root_path -- appropriate small values for a 15mm cube (PhysX defaults
    are tuned for much larger objects and can make contact register well
    before the actual meshes touch)."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0
    n = 0
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            capi = PhysxSchema.PhysxCollisionAPI.Apply(p)
            capi.CreateContactOffsetAttr().Set(contact_offset)
            capi.CreateRestOffsetAttr().Set(rest_offset)
            n += 1
    return n


def main():
    log(f"=== standalone AR4 pick, mechanism={args_cli.mechanism} ===")
    _bc("main_start")
    world_kwargs = dict(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
    if args_cli.cpu:
        world_kwargs["device"] = "cpu"
        world_kwargs["sim_params"] = {"use_gpu_pipeline": False, "use_gpu": False}
    world = World(**world_kwargs)
    _bc("World_constructed")
    stage = get_current_stage()

    # LOCAL ground only. Deliberately NOT `add_default_ground_plane()` -- that
    # fetches a USD from NVIDIA's remote Nucleus asset server, which a cloud
    # instance with no Nucleus access blocks on forever ("carb.tasking stuck"
    # at world.reset()); confirmed live 2026-07-29. A big thin static box whose
    # top sits at z=0 is a fully-procedural, no-network floor.
    FixedCuboid(
        prim_path="/World/Ground",
        position=np.array([0.0, 0.0, -0.5]),
        scale=np.array([50.0, 50.0, 1.0]),
        color=np.array([0.55, 0.55, 0.58]) if SIDE_GRASP else np.array([0.3, 0.3, 0.3]),
    )
    _bc("ground_added")

    # dome light (brighter for side-grasp / pin-pick video legibility -- the
    # gripper is dark, so a small low-lit scene reads as a black blob without it)
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(3000.0 if (SIDE_GRASP or PIN_PICK) else 2000.0)
    if PIN_PICK:
        # add a bright distal key light angled down onto the grasp point so the
        # dark gripper + object are lit from a 3/4 angle, not a flat dome wash --
        # avoids the "black blob against a dark background" the user complained of.
        _key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
        _key.CreateIntensityAttr(2500.0)
        _key.CreateAngleAttr(1.5)
        _kx = UsdGeom.Xformable(_key.GetPrim())
        _kx.ClearXformOpOrder()
        # aim down-and-toward the grasp from +X+Z (matches the 3/4 camera side)
        _kx.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 35.0))

    cube = None
    robot = None
    # --empty_scene: place pedestal+cube FAR out of reach so nothing can collide
    # with the arm -- the decisive test for whether GRASP_Q is blocked by an
    # EXTERNAL scene collision (cube/pedestal) vs a self-collision/arm issue.
    _obs_dx = 5.0 if args_cli.empty_scene else 0.0
    if args_cli.stage in ("scene", "robot_noxform", "full"):
        if SIDE_GRASP and PIN_PICK:
            # SUPPORT-FREE side grasp (2026-07-31): a THIN vertical pin under the
            # cube CENTER, rising from the ground to the cube bottom (0.0825). The
            # side grasp grips the cube's +Y/-Y vertical faces at MID-HEIGHT (cube
            # center 0.09 == pad plane) while the body sits laterally on -X -- so
            # UNLIKE top-down, the pad grip band is NOT tied to the body's vertical
            # position, and the ONLY prior blocker was the SUPPORT under the cube
            # intersecting the jaw z-band (kb 2026-07-31 side-grasp UPDATE). A thin
            # central pin (much thinner than the 15mm cube) threads up through the
            # cube's own footprint, between the two Y-face jaws, clearing them and
            # the body (body +X edge -0.122 vs pin at -0.12). The cube rests on it
            # until the grasp lifts it straight off.
            _side_pin_h = CUBE_REST_Z - CUBE_SIZE / 2.0   # cube bottom == 0.0825
            FixedCuboid(
                prim_path="/World/Pin",
                position=np.array([CUBE_XY[0] + _obs_dx, CUBE_XY[1], _side_pin_h / 2.0]),
                scale=np.array([PIN_SZ, PIN_SZ, _side_pin_h]),
                color=np.array([0.55, 0.55, 0.60]),
            )
            log(f"[SIDE_PIN] support-free side grasp: thin pin {PIN_SZ*1000:.0f}x{PIN_SZ*1000:.0f}mm "
                f"height {_side_pin_h*1000:.1f}mm (ground -> cube bottom {_side_pin_h:.4f}) at cube center; "
                f"grips cube Y-faces at mid-height 0.09, body lateral on -X")
        elif SIDE_GRASP:
            # thin FLOATING support plate (top == cube rest surface 0.0825), NOT a
            # tall ground-rooted pedestal -- so it never rises as a wall in the
            # horizontal approach/swing corridor (RUN-1 proved the tall pedestal
            # rammed the arm 40mm low; free space was flawless). See SIDE_SUPPORT_*.
            FixedCuboid(
                prim_path="/World/Pedestal",
                position=np.array([SIDE_SUPPORT_CENTER[0] + _obs_dx, SIDE_SUPPORT_CENTER[1], SIDE_SUPPORT_CENTER[2]]),
                scale=np.array([SIDE_SUPPORT_SCALE[0], SIDE_SUPPORT_SCALE[1], SIDE_SUPPORT_SCALE[2]]),
                color=np.array([0.45, 0.32, 0.22]),
            )
        elif TALL_PRISM:
            # NO pedestal: the tall prism rests directly on the ground (z=0). The
            # whole grasp sits 56mm up (the reachable pad plane) so the gripper's
            # below-pad underhang clears the ground -- and there is no support
            # surface for the gripper body to jam on (the side-grasp blocker).
            log("[TALL_PRISM] no pedestal -- prism rests on the ground")
        elif PIN_PICK:
            # THIN support pin ONLY (no pedestal/plate): a static kinematic square
            # post rising from the ground up to the object bottom, thin enough for
            # the gripper body/fingers to genuinely clear it. High friction so the
            # object rests stably on it until the grasp lifts it off. Colored bright
            # so it reads in video against the ground.
            _pin_mat = None
            FixedCuboid(
                prim_path="/World/Pin",
                position=np.array([PIN_XY[0] + _obs_dx, PIN_XY[1], PIN_CENTER_Z]),
                scale=np.array([PIN_SZ, PIN_SZ, PIN_HEIGHT]),
                color=np.array([0.55, 0.55, 0.60]),
            )
            log(f"[PIN_PICK] thin support pin {PIN_SZ*1000:.0f}x{PIN_SZ*1000:.0f}mm, "
                f"height {PIN_HEIGHT*1000:.1f}mm (ground -> object bottom {PIN_OBJ_BOTTOM:.4f})")
        else:
            # pedestal (static collider)
            FixedCuboid(
                prim_path="/World/Pedestal",
                position=np.array([PEDESTAL_CENTER_XY[0] + _obs_dx, PEDESTAL_CENTER_XY[1], PEDESTAL_HEIGHT / 2.0]),
                scale=np.array([PEDESTAL_FOOTPRINT[0], PEDESTAL_FOOTPRINT[1], PEDESTAL_HEIGHT]),
                color=np.array([0.45, 0.32, 0.22]),
            )
        if TALL_PRISM:
            # tall narrow prism (dynamic), resting on the ground, gripped near top
            cube = DynamicCuboid(
                prim_path=CUBE_PATH,
                position=np.array([PRISM_XY[0] + _obs_dx, PRISM_XY[1], PRISM_CENTER_Z]),
                scale=np.array([PRISM_W, PRISM_D, PRISM_H]),
                color=np.array([0.1, 0.35, 0.85]),
                mass=args_cli.prism_mass,
            )
            log(f"[TALL_PRISM] prism {PRISM_W*1000:.0f}x{PRISM_D*1000:.0f}x{PRISM_H*1000:.0f}mm "
                f"mass={args_cli.prism_mass*1000:.0f}g on ground; top_z={PRISM_H:.4f} "
                f"pad_plane_z={PRISM_GRIP_Z:.4f} (grip {(PRISM_H-PRISM_GRIP_Z)*1000:+.1f}mm relative to top)")
        elif PIN_PICK and not SIDE_GRASP:
            # support-free object hanging down from just below the gripper body,
            # resting on the thin pin. Gripped on its two +/-X faces by the jaws.
            cube = DynamicCuboid(
                prim_path=CUBE_PATH,
                position=np.array([PIN_XY[0] + _obs_dx, PIN_XY[1], PIN_OBJ_CENTER_Z]),
                scale=np.array([PIN_OBJ_W, PIN_OBJ_D, PIN_OBJ_H]),
                color=np.array([0.85, 0.15, 0.12]),
                mass=args_cli.pin_obj_mass,
            )
            log(f"[PIN_PICK] object {PIN_OBJ_W*1000:.0f}x{PIN_OBJ_D*1000:.0f}x{PIN_OBJ_H*1000:.0f}mm "
                f"mass={args_cli.pin_obj_mass*1000:.0f}g on thin pin; top_z={PIN_OBJ_TOP:.4f} "
                f"bottom_z={PIN_OBJ_BOTTOM:.4f} center_z={PIN_OBJ_CENTER_Z:.4f}; "
                f"body_z_min~0.0548 -> top clearance {(0.0548-PIN_OBJ_TOP)*1000:+.1f}mm; "
                f"pad_plane 0.05621 -> top {(PIN_OBJ_TOP-0.05621)*1000:+.1f}mm relative to pad centroid")
        else:
            # cube (dynamic, 15mm)
            cube = DynamicCuboid(
                prim_path=CUBE_PATH,
                position=np.array([CUBE_XY[0] + _obs_dx, CUBE_XY[1], CUBE_REST_Z]),
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=np.array([0.8, 0.1, 0.1]),
                mass=0.01,
            )
        if args_cli.empty_scene:
            log(f"[EMPTY_SCENE] scene objects shifted +{_obs_dx}m in X (out of reach) to test for external collision")
        _bc("scene_built")

    if args_cli.stage in ("robot_noxform", "full"):
        # robot: reference the USD, rotate base 180deg about Z (arm faces +Y),
        # BEFORE physics init.
        add_reference_to_stage(USD_PATH, AR4_ROOT)
        if args_cli.stage == "full":
            ar4_prim = stage.GetPrimAtPath(AR4_ROOT)
            xf = UsdGeom.Xformable(ar4_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            # quat wxyz (0,0,0,1) -> 180deg about Z
            xf.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
        robot = SingleArticulation(prim_path=ART_ROOT, name="ar4")
        world.scene.add(robot)
        _bc("robot_added")

    # --- gripper-jaw REAL physics setup (BEFORE reset) ----------------------
    # Root-cause fix for the user-flagged defect: the prior task's video shows
    # the jaws staying visually OPEN through the whole grasp+lift despite
    # commanding GRIP_CLOSED -- i.e. the position command never actually moved
    # the jaw joints. This repo's own history has hit exactly this bug class
    # before ("a missing gripper physics drive", see kb/wiki/concepts/
    # ar4-vs-franka-root-cause-comparison.md's intro). This standalone script
    # bypasses Isaac Lab's ImplicitActuatorCfg layer entirely (by design, see
    # module docstring) and relies solely on
    # `articulation_controller.set_gains()` -- which this task found is NOT a
    # reliable guarantee that a real PhysX drive exists on these two prismatic
    # joints. Author an explicit UsdPhysics.DriveAPI("linear") directly on
    # both joint prims as a USD-level guarantee, independent of whatever the
    # higher-level Python gains wrapper does or doesn't do.
    if args_cli.stage == "full":
        # RUN-4 arm-drive authoring: raise the ARM joints' drive force/effort
        # ceiling so the drive can actually HOLD the arm against gravity at the
        # grasp pose (runs 1-3's ~16mm droop is drive-force saturation, not a
        # stiffness deficit -- see the SCENE comment block above). Author an
        # explicit angular force-type drive on each arm joint with a high
        # maxForce (ARM_MAX_FORCE), stiffness ARM_KP, damping ARM_KD. The
        # per-step controller.apply_action(joint_positions=...) still sets the
        # live target each step (same pattern proven on the gripper joints).
        for _ajname in ARM_JOINTS:
            _ajprim = find_prim_by_name(stage, AR4_ROOT, _ajname)
            if _ajprim is None or not _ajprim.IsValid():
                log(f"[ARM DRIVE] WARNING: joint prim not found for {_ajname} -- explicit drive NOT authored")
                continue
            _adrive = UsdPhysics.DriveAPI.Apply(_ajprim, "angular")
            _adrive.CreateTypeAttr().Set("force")
            _adrive.CreateStiffnessAttr().Set(ARM_KP)
            _adrive.CreateDampingAttr().Set(ARM_KD)
            _adrive.CreateMaxForceAttr().Set(ARM_MAX_FORCE)
            log(f"[ARM DRIVE] authored angular force-type drive on {_ajprim.GetPath()} "
                f"(stiffness={ARM_KP}, damping={ARM_KD}, maxForce={ARM_MAX_FORCE} N*m)")
            # RUN-6 (2026-07-30, ar4-gravity-droop): the wrist joints (4,5,6)
            # do NOT hold their commanded position -- near-zero drive effort at
            # large error, worse with higher stiffness (diagnostic D2/D3) -- the
            # signature of low/zero distal-link inertia making the position drive
            # ill-conditioned. Author ARMATURE (reflected actuator inertia) on
            # each arm joint axis so the drive is well-conditioned and actually
            # tracks. Attribute: PhysxJointAxisAPI(<axis>):armature. Try the
            # likely axis instance names for a revolute drive.
            if args_cli.arm_armature > 0.0:
                _armed = False
                for _axis in ("angular", "rotX", "X"):
                    try:
                        _axapi = PhysxSchema.PhysxJointAxisAPI.Apply(_ajprim, _axis)
                        _axapi.CreateArmatureAttr().Set(float(args_cli.arm_armature))
                        _armed = _axis
                        break
                    except Exception:
                        continue
                log(f"[ARM ARMATURE] joint {_ajname}: armature={args_cli.arm_armature} "
                    f"authored via PhysxJointAxisAPI axis='{_armed}'")

        for _jname in GRIP_JOINTS:
            _jprim = find_prim_by_name(stage, AR4_ROOT, _jname)
            if _jprim is None or not _jprim.IsValid():
                log(f"[DRIVE] WARNING: joint prim not found for {_jname} -- explicit drive NOT authored")
                continue
            _drive = UsdPhysics.DriveAPI.Apply(_jprim, "linear")
            _drive.CreateTypeAttr().Set("force")
            _drive.CreateStiffnessAttr().Set(GRIP_KP)
            _drive.CreateDampingAttr().Set(GRIP_KD)
            # maxForce is a generous CEILING for the approach/open phase only.
            # The actual squeeze force is set later, at the grasp phase, either
            # by capping this same maxForce to --squeeze_force (position mode)
            # or by zeroing the position gains and applying a direct joint
            # effort (effort mode). The old 500N here would eject a 10g cube.
            _drive.CreateMaxForceAttr().Set(float(args_cli.jaw_max_force))
            _drive.CreateTargetPositionAttr().Set(GRIP_OPEN)
            log(f"[DRIVE] authored explicit linear force-type drive on {_jprim.GetPath()} "
                f"(stiffness={GRIP_KP}, damping={GRIP_KD}, maxForce ceiling=50N; "
                f"squeeze force set at grasp per --squeeze_mode/--squeeze_force)")

        # articulation-wide solver iteration counts (default PhysX iteration
        # counts can be too low to resolve a stiff small-object pinch grasp).
        try:
            _art_prim = stage.GetPrimAtPath(ART_ROOT)
            if _art_prim.IsValid():
                _aapi = PhysxSchema.PhysxArticulationAPI.Apply(_art_prim)
                _aapi.CreateSolverPositionIterationCountAttr().Set(32)
                _aapi.CreateSolverVelocityIterationCountAttr().Set(4)
                log(f"[PHYSICS] articulation {ART_ROOT}: solver_pos_iters=32 solver_vel_iters=4")
        except Exception as e:
            log(f"[PHYSICS] WARNING: articulation solver-iteration authoring failed: {e}")

        # --- friction materials: cube high-friction, jaws high-friction,
        # combine mode "max" (never "min") so neither side's material can
        # silently cap the effective grip friction below the other's value.
        try:
            _cube_mat = make_physics_material(
                stage, "/World/PhysicsMaterials/cube_mat",
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0, combine_mode="max",
            )
            bind_physics_material(stage, CUBE_PATH, _cube_mat)
            _jaw_mat = make_physics_material(
                stage, "/World/PhysicsMaterials/jaw_mat",
                static_friction=0.9, dynamic_friction=0.9, restitution=0.0, combine_mode="max",
            )
            _n_bound = 0
            for _jaw_link_path in (GRIP1_PATH, GRIP2_PATH):
                if bind_physics_material(stage, _jaw_link_path, _jaw_mat):
                    _n_bound += 1
                # also bind directly onto any nested collision-API mesh prims
                # (a link-level bind should already inherit down, this is
                # belt-and-suspenders in case the collision mesh is a sibling
                # rather than a descendant of the named link Xform).
                _root = stage.GetPrimAtPath(_jaw_link_path)
                if _root.IsValid():
                    for _p in Usd.PrimRange(_root):
                        if _p.HasAPI(UsdPhysics.CollisionAPI):
                            bind_physics_material(stage, str(_p.GetPath()), _jaw_mat)
                            _n_bound += 1
            log(f"[FRICTION] cube_mat(0.8/0.8) bound to cube; jaw_mat(0.9/0.9) bound to {_n_bound} jaw prim(s); "
                f"combine_mode=max on both materials")
        except Exception as e:
            log(f"[FRICTION] WARNING: physics-material authoring failed: {e}")

        # --- contact/rest offsets tuned for a 15mm cube (PhysX defaults are
        # sized for much larger objects and can register contact well before
        # the meshes actually touch, or fail to hold a tight small-object
        # pinch).
        try:
            n_cube = apply_contact_offsets_under(stage, CUBE_PATH, contact_offset=0.001, rest_offset=0.0002)
            n_jaw1 = apply_contact_offsets_under(stage, GRIP1_PATH, contact_offset=0.001, rest_offset=0.0002)
            n_jaw2 = apply_contact_offsets_under(stage, GRIP2_PATH, contact_offset=0.001, rest_offset=0.0002)
            log(f"[PHYSICS] contact_offset=1mm rest_offset=0.2mm applied to "
                f"{n_cube} cube prim(s), {n_jaw1} jaw1 prim(s), {n_jaw2} jaw2 prim(s)")
        except Exception as e:
            log(f"[PHYSICS] WARNING: contact/rest offset authoring failed: {e}")

    # --- SurfaceGripper authoring (BEFORE reset) ---------------------------
    gripper_view = None
    fixed_joint_enabled_attr = None
    if args_cli.mechanism == "surface_gripper" and args_cli.stage == "full":
        gripper_path = "/World/SurfaceGripper"
        robot_schema.CreateSurfaceGripper(stage, gripper_path)
        joints_scope = "/World/Surface_Gripper_Joints"
        UsdGeom.Scope.Define(stage, Sdf.Path(joints_scope))
        attach_path = joints_scope + "/AttachPoint"
        joint = UsdPhysics.Joint.Define(stage, Sdf.Path(attach_path))
        jp = joint.GetPrim()
        robot_schema.ApplyAttachmentPointAPI(jp)
        jp.GetAttribute(robot_schema.Attributes.FORWARD_AXIS.name).Set("Z")
        jp.GetAttribute(robot_schema.Attributes.CLEARANCE_OFFSET.name).Set(0.008)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(LINK6_PATH)])
        # attach point sits on link_6 +Z (which points toward the grasped cube,
        # baked cube-in-link6 ~ (-0.0025,0.0005,0.0632)); ray searches +Z.
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.0025, 0.0005, 0.040))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        gp = stage.GetPrimAtPath(gripper_path)
        gp.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([Sdf.Path(attach_path)])
        log("[surface_gripper] authored /World/SurfaceGripper + AttachPoint on link_6")

    # --- reset (initializes physics + articulation, plays timeline) --------
    _bc("before_reset")
    timeline = omni.timeline.get_timeline_interface()
    world.reset()
    _bc("after_reset")
    log(f"RESET_OK stage={args_cli.stage}")
    timeline.play()
    _bc("after_play")
    # let things settle a few steps (heartbeat each step so a slow-but-working
    # 4-vCPU physics loop is not mistaken for a hang by the watchdog)
    for _s in range(10):
        world.step(render=False)
        _bc(f"settle_step_{_s}")
    _bc("settled")
    log(f"STEP_OK stage={args_cli.stage}")

    if args_cli.stage != "full":
        log(f"DIAG stage={args_cli.stage} completed reset+step OK; exiting (VERDICT: DIAG_OK)")
        _R.close()
        simulation_app.close()
        return

    dof_names = list(robot.dof_names)
    log(f"[dof_names] {dof_names}  num_dof={robot.num_dof}")
    arm_idx = [dof_names.index(n) for n in ARM_JOINTS]
    grip_idx = [dof_names.index(n) for n in GRIP_JOINTS]

    # boost gains for crisp trajectory tracking
    kps = np.zeros(robot.num_dof, dtype=np.float32)
    kds = np.zeros(robot.num_dof, dtype=np.float32)
    for i in arm_idx:
        kps[i], kds[i] = ARM_KP, ARM_KD
    for i in grip_idx:
        kps[i], kds[i] = GRIP_KP, GRIP_KD
    controller = robot.get_articulation_controller()
    controller.set_gains(kps=kps, kds=kds)

    # --- contact-force instrumentation on each jaw (the squeeze PROOF) ------
    # Direct measurement of the normal force the jaws transmit onto the cube's
    # two faces -- the number that proves a real squeeze (per user directive).
    contact_sensors = {}
    try:
        from isaacsim.sensors.physics import ContactSensor
        for _cn, _cp in (("jaw1", GRIP1_PATH), ("jaw2", GRIP2_PATH)):
            _cs = ContactSensor(prim_path=_cp + "/grip_contact_sensor",
                                name=f"{_cn}_contact", min_threshold=0.0,
                                max_threshold=1.0e8, radius=-1.0)
            _cs.initialize()
            contact_sensors[_cn] = _cs
        log(f"[CONTACT] ContactSensor initialized on jaws: {list(contact_sensors)}")
    except Exception as e:
        log(f"[CONTACT] ContactSensor unavailable ({e}); relying on measured joint forces")

    def contact_report(label):
        cs_out = {}
        for _cn, _cs in contact_sensors.items():
            try:
                fr = _cs.get_current_frame()
                cs_out[_cn] = round(float(fr.get("force", 0.0)), 4)
            except Exception as _e:
                cs_out[_cn] = f"err:{_e}"
        # measured actuation efforts on the gripper DOFs (backup squeeze signal)
        me = None
        try:
            me = [round(float(v), 4) for v in np.asarray(robot.get_measured_joint_efforts())[grip_idx]]
        except Exception:
            me = None
        log(f"[CONTACT {label}] jaw_normal_force_N={cs_out} measured_grip_efforts_N={me}")
        return cs_out

    # --- squeeze machinery: a REAL closing force on the jaws (no joint/weld) -
    squeeze = {"active": False, "eff": -abs(args_cli.squeeze_force)}

    def start_squeeze():
        squeeze["active"] = True
        if args_cli.squeeze_mode == "effort":
            # zero the jaw POSITION gains so the applied joint effort is the
            # ONLY force the jaws exert -> genuine direct force control.
            for i in grip_idx:
                kps[i], kds[i] = 0.0, 30.0
            controller.set_gains(kps=kps, kds=kds)
            log(f"[SQUEEZE] effort mode: jaw position gains zeroed; applying constant closing "
                f"effort {squeeze['eff']:.3f} N/jaw (kd=30 damping for stability)")
        else:
            # position mode: simply drive the jaw target PAST the cube
            # (GRIP_CLOSED). The squeeze force comes from the PD drive itself
            # (GRIP_KP * contact error ~= 33N at the cube faces) -- NO runtime
            # DriveAPI mutation (that desynced the articulation drive last run
            # and left the jaws stuck open). This is the proven closing path.
            log(f"[SQUEEZE] position mode: jaw target -> GRIP_CLOSED (past cube); squeeze delivered by the "
                f"jaw PD drive (GRIP_KP={GRIP_KP:.0f} -> ~{GRIP_KP*0.0065:.0f}N at cube contact). No runtime "
                f"maxForce mutation.")

    if args_cli.mechanism == "surface_gripper":
        gripper_view = GripperView(paths="/World/SurfaceGripper")
        gripper_view.set_surface_gripper_properties(
            max_grip_distance=[args_cli.max_grip_distance],
            coaxial_force_limit=[1000.0],
            shear_force_limit=[1000.0],
            retry_interval=[3.0],
        )
        log(f"[surface_gripper] GripperView ready, max_grip_distance={args_cli.max_grip_distance}")

    grasp_state = {"engaged": False, "mechanism": "none"}

    # ----- fixed_joint fallback: author a DISABLED joint now, enable at grasp
    if args_cli.mechanism == "fixed_joint":
        fj_path = "/World/AR4_grasp_fixed_joint"
        fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(fj_path))
        fj.CreateBody0Rel().SetTargets([Sdf.Path(LINK6_PATH)])
        fj.CreateBody1Rel().SetTargets([Sdf.Path(CUBE_PATH)])
        fj.GetPrim().CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)
        fixed_joint_enabled_attr = fj.CreateJointEnabledAttr()
        fixed_joint_enabled_attr.Set(False)
        log("[fixed_joint] authored DISABLED link_6<->cube fixed joint (frames baked at grasp)")

    # ---- helpers ----------------------------------------------------------
    def full_targets(q_arm, grip_val):
        t = np.array(robot.get_joint_positions(), dtype=np.float32)
        for k, i in enumerate(arm_idx):
            t[i] = q_arm[k]
        for i in grip_idx:
            t[i] = grip_val
        return t

    def cube_pose():
        p, q = cube.get_world_pose()
        return np.array(p), np.array(q)

    # camera setup (best-effort)
    cams = {}
    writers = {}
    if not args_cli.no_video:
        try:
            from isaacsim.sensors.camera import Camera
            cams["closeup"] = Camera(prim_path="/World/closeup_cam", resolution=(640, 480))
            cams["elbow"] = Camera(prim_path="/World/elbow_cam", resolution=(640, 480))
            for c in cams.values():
                c.initialize()
            import imageio
            fps = 20
            writers["closeup"] = imageio.get_writer(os.path.join(VIDEO_DIR, "closeup.mp4"), fps=fps, codec="libx264")
            writers["elbow"] = imageio.get_writer(os.path.join(VIDEO_DIR, "elbow.mp4"), fps=fps, codec="libx264")
            log("[video] cameras initialized")
        except Exception as e:
            log(f"[video] camera init failed (continuing without video): {e}")
            cams = {}

    frame = {"n": 0}

    # Two FIXED wide 3/4 overviews from opposite sides, each aimed at the
    # middle of the lift arc (grasp z~0.05 -> retreat z~0.46 -> aim ~0.25).
    # Static + far + opposite sides => at least one keeps a clear sightline
    # to the red cube through the whole pick despite arm-link occlusion. The
    # camera prim's USD orientation is set DIRECTLY via a lookat matrix
    # (unambiguous), bypassing the isaacsim camera_axes convention.
    if SIDE_GRASP:
        # Side grasp: cube at (-0.12,0.28,0.09), gripper approaches horizontally
        # from -X (palm on -X side), pads grip the +Y/-Y faces, lift straight up
        # to z~0.14. RUN-1's cameras were too far (0.6-0.8m) for a 15mm cube --
        # closeup showed only the horizon. Bring them CLOSE (~0.25m) and frame the
        # cube+jaws directly. Closeup from +X+Y (looks -X at the cube's +X face,
        # sees the two jaws close along Y); wide from -X-Y (over the incoming
        # gripper's shoulder, sees the full approach + straight-up lift).
        # PROVEN values: RUN-1's elbow camera (CAM_B_EYE=[-0.75,0.62,0.42],
        # target [-0.12,0.28,0.11]) demonstrably showed the gripper (dark gripper
        # against the scene, off-center but clearly visible in extracted frames).
        # RUN-2/3/4 "improvements" (closer / near-level / look-up) all lost the
        # gripper -- so revert to the RUN-1 elevated-3/4 geometry (per the
        # isaac-sim-video-capture skill: reuse a verified position, don't re-derive)
        # and mirror it for a second view. ~0.8m elevated 3/4 from +X+Y and -X+Y.
        CAM_A_EYE = [0.75, 0.62, 0.42]    # +X +Y elevated 3/4 (mirror of the proven elbow)
        CAM_B_EYE = [-0.75, 0.62, 0.42]   # -X +Y elevated 3/4 (RUN-1's confirmed-visible elbow eye)
        CAM_TGT = [-0.12, 0.28, 0.11]
    elif TALL_PRISM:
        # Tall prism at (-0.122,0.372), gripped near its top (~0.056), lifted up
        # (grasp->PREGRASP->HOME retreat) to ~0.2-0.3. Reuse the side-grasp lesson:
        # a ~0.8m ELEVATED-3/4 framing reads the small dark gripper + blue prism
        # legibly (near-level / too-far framings lost it). Two opposite-X views.
        # NOTE (2026-07-31): the first tall-prism videos framed empty space (closeup
        # showed only the horizon, elbow caught the base corner) -- the small object
        # at (-0.122,0.372,~0.05) and the low grasp action need a CLOSER camera aimed
        # at the grip point, not the mid-lift arc. Closeup ~0.35m on the grip point;
        # elbow ~0.7m to also catch the lift. (Framing improvement; not re-verified
        # this session since the grasp itself is geometrically infeasible -- see kb.)
        CAM_A_EYE = [0.12, 0.60, 0.26]    # +X +Y elevated 3/4, ~0.35m, tight on grip
        CAM_B_EYE = [-0.48, 0.74, 0.42]   # -X +Y elevated 3/4, ~0.7m, catches the lift
        CAM_TGT = [-0.122, 0.372, 0.055]  # aim at the grip point (pad plane)
    elif PIN_PICK:
        # Support-free pin pick: object at (-0.1224,0.3724), gripped ~0.045 (top
        # 0.051), lifted up-and-back off the thin pin. Framing lessons applied
        # (isaac-sim-video-capture skill + side-grasp RUN-1 + tall-prism note):
        #   - ELEVATED 3/4 aimed DEAD AT THE GRIP POINT (not the mid-lift arc, which
        #     framed empty space in the tall-prism run) reads the dark gripper +
        #     object legibly; near-level / look-up variants lose it to the ground.
        #   - Both eyes on the +Y side (BEYOND the object) so the arm body -- which
        #     sits between the base (origin) and the object -- does not occlude.
        #   - CLOSEUP pulled to ~0.55m so the ~15-20mm object is well-sized (bigger
        #     than the 0.8m side-grasp shot), still elevated/aimed-down so the eye
        #     stays well outside gripper geometry (no black frames). WIDE ~0.9m to
        #     catch the whole approach->close->liftoff arc for context.
        CAM_A_EYE = [0.30, 0.62, 0.34]    # +X +Y elevated 3/4, ~0.55m closeup on the grip
        CAM_B_EYE = [-0.55, 0.90, 0.58]   # -X +Y elevated 3/4, ~0.90m wide context
        CAM_TGT = [-0.1224, 0.3724, 0.075]  # aim at the grip point (a touch up to keep early lift in frame)
    else:
        CAM_A_EYE = [0.9, 1.0, 0.85]      # +X +Y high 3/4
        CAM_B_EYE = [-1.1, 1.0, 0.85]     # -X +Y high 3/4
        CAM_TGT = [-0.122, 0.372, 0.20]   # aim at new cube XY / mid lift arc

    def _set_cam_lookat(prim_path, eye, target):
        eye = np.asarray(eye, float); target = np.asarray(target, float)
        fwd = target - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_w); right /= (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, fwd)
        # USD camera looks down -Z, +Y up, +X right
        m = Gf.Matrix4d(
            float(right[0]), float(right[1]), float(right[2]), 0.0,
            float(up[0]), float(up[1]), float(up[2]), 0.0,
            float(-fwd[0]), float(-fwd[1]), float(-fwd[2]), 0.0,
            float(eye[0]), float(eye[1]), float(eye[2]), 1.0,
        )
        prim = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)

    def position_cameras():
        if "closeup" in cams:
            _set_cam_lookat("/World/closeup_cam", CAM_A_EYE, CAM_TGT)
        if "elbow" in cams:
            _set_cam_lookat("/World/elbow_cam", CAM_B_EYE, CAM_TGT)

    def capture():
        if not cams:
            return
        frame["n"] += 1
        for name, c in cams.items():
            try:
                rgba = c.get_rgba()
                if rgba is not None and rgba.size > 0:
                    writers[name].append_data(rgba[..., :3].astype("uint8"))
            except Exception:
                pass

    step_ctr = {"n": 0}

    _arm_idx_np = np.array(arm_idx, dtype=np.int32)
    _grip_idx_np = np.array(grip_idx, dtype=np.int32)

    # --- FEEDFORWARD GRAVITY COMPENSATION (2026-07-30, ar4-gravity-droop) ----
    # Root-cause fix for the persistent 11-16mm vertical droop that resisted 60x
    # arm-drive maxForce (runs 1-4): a plain PD position drive sags under gravity
    # because its force is proportional ONLY to position error, so it must droop
    # to generate any holding torque at all. Feedforward the generalized gravity
    # force G(q) as an additive joint EFFORT each step -> the PD then only handles
    # the residual, and the commanded pose is held with ~0 steady-state error
    # regardless of the drive's effective stiffness/force headroom. The applied
    # joint effort is an ADDITIVE actuation force in PhysX's reduced-coordinate
    # articulation (summed with the position drive's force), NOT a replacement.
    # RUN-5 DIAGNOSTIC RESULT (2026-07-30, ar4-gravity-droop): the +9-16mm
    # vertical gap is NOT gravity droop -- feedforward of the reported G(q)
    # (tiny: joint_2 only ~2.3 N*m) changed nothing, and the settled pose shows
    # large tracking errors concentrated in the WRIST joints (joint_5 +4.5deg,
    # joint_6 +8.6deg) at near-ZERO effort -- i.e. the arm settles OFF its
    # commanded FK pose (which is verified pad-at-cube-center 0.000mm) for a
    # position-tracking reason (joint friction / weak effective wrist drive),
    # not a gravitational load. The robust, cause-agnostic fix is an INTEGRAL
    # term on applied joint EFFORT: accumulate per-arm-joint (cmd-achieved) into
    # an effort that ramps until the steady-state joint error is nulled --
    # overcoming whatever static offset (friction/deadband/weak drive) the plain
    # position PD leaves. Nulling all 6 arm-joint errors drives the achieved
    # pose ONTO the commanded FK pose == pad at cube center. This is the "closed-
    # loop integral term that nulls the steady-state gap" fix. A small gravity
    # feedforward is retained as an option but is NOT the operative mechanism.
    grav = {"active": False, "sign": 1.0, "api": None,
            "mode": args_cli.assist_mode, "integ": np.zeros(robot.num_dof, dtype=np.float64),
            "ki": args_cli.integral_ki, "clamp": args_cli.integral_clamp,
            # RUN-7 (2026-07-31): joint-space COMMAND pre-compensation. The +9mm
            # pad-Z gap is a repeatable joint TRACKING error (achieved != command)
            # on joints 2,3,5 (which DO respond to commands). Accumulate a per-arm-
            # joint command OFFSET each step so achieved -> the commanded FK pose,
            # using the STABLE position drive (the effort-integral ran away because
            # the wrist has no effective drive to stabilize applied effort; adding a
            # command offset instead rides the position drive, which IS stable).
            "cmd_off": np.zeros(robot.num_dof, dtype=np.float64),
            "kc": args_cli.cmd_kc, "cmd_clamp": math.radians(args_cli.cmd_clamp_deg)}

    def _reset_integrator():
        grav["integ"][:] = 0.0

    def _reset_cmd_off():
        grav["cmd_off"][:] = 0.0

    def _apply_cmd_offset(q_arm):
        """Return arm-target array = q_arm + accumulated per-joint command offset,
        and advance the offset by kc*(q_arm - achieved) (clamped). Drives achieved
        -> q_arm via the stable position drive. Active only in 'cmd_integral' mode."""
        out = np.array(q_arm, dtype=np.float32)
        if not (grav["active"] and grav["mode"] == "cmd_integral"):
            return out
        ach = np.array(robot.get_joint_positions())
        cc = grav["cmd_clamp"]
        for k, i in enumerate(arm_idx):
            err = float(q_arm[k]) - float(ach[i])       # want achieved == q_arm
            grav["cmd_off"][i] += grav["kc"] * err
            if grav["cmd_off"][i] > cc:
                grav["cmd_off"][i] = cc
            elif grav["cmd_off"][i] < -cc:
                grav["cmd_off"][i] = -cc
            out[k] = float(q_arm[k]) + grav["cmd_off"][i]
        return out

    def _query_gravity_forces():
        """Return the generalized gravity force vector G(q) over all DOFs, or
        None. Defensive: the exact method name on this Isaac Sim 5.1
        SingleArticulation is introspected at runtime (candidates below), the
        winner cached in grav['api']."""
        candidates = [robot] + [getattr(robot, a, None) for a in
                                ("_articulation_view", "_physics_view", "_articulation")]
        method_names = ("get_generalized_gravity_forces",
                        "get_gravity_compensation_forces")
        if grav["api"] is not None:
            obj, name = grav["api"]
            try:
                return np.asarray(getattr(obj, name)()).reshape(-1)
            except Exception:
                grav["api"] = None
        for obj in candidates:
            if obj is None:
                continue
            for name in method_names:
                fn = getattr(obj, name, None)
                if fn is None:
                    continue
                try:
                    g = np.asarray(fn()).reshape(-1)
                    if g.size >= robot.num_dof:
                        grav["api"] = (obj, name)
                        return g
                except Exception:
                    continue
        return None

    def _assist_effort(q_arm):
        """Full num_dof additive-effort array for the active assist mode, or
        None when inactive. 'integral': PI-style integrator on per-arm-joint
        (commanded - achieved) position error -> nulls steady-state tracking
        offset. 'gravity': feedforward sign*G(q) (retained; NOT operative --
        the droop was measured to be non-gravitational). Grip DOFs stay 0 here
        (their effort is handled by the squeeze path)."""
        if not grav["active"] or grav["mode"] == "cmd_integral":
            return None  # cmd_integral adjusts position TARGETS, not effort
        eff = np.zeros(robot.num_dof, dtype=np.float32)
        if grav["mode"] == "gravity":
            g = _query_gravity_forces()
            if g is None or g.size < robot.num_dof:
                return None
            for i in arm_idx:
                eff[i] = grav["sign"] * float(g[i])
            return eff
        # integral (default): accumulate effort until per-joint error -> 0
        ach = np.array(robot.get_joint_positions())
        c = grav["clamp"]
        for k, i in enumerate(arm_idx):
            err = float(q_arm[k]) - float(ach[i])       # rad
            grav["integ"][i] += grav["ki"] * err        # discrete integrator
            if grav["integ"][i] > c:
                grav["integ"][i] = c
            elif grav["integ"][i] < -c:
                grav["integ"][i] = -c
            eff[i] = grav["integ"][i]
        return eff

    def drive(q_arm, grip_val, steps, render=True):
        do_render = render and RENDER
        for _ in range(steps):
            gff = _assist_effort(q_arm)          # effort assist (gravity/integral modes)
            q_cmd = _apply_cmd_offset(q_arm)     # cmd_integral: tracking-compensated arm targets
            if squeeze["active"] and args_cli.squeeze_mode == "effort":
                # arm on position control (arm DOFs only); jaws on pure EFFORT.
                if gff is not None:
                    controller.apply_action(ArticulationAction(
                        joint_positions=q_cmd, joint_efforts=gff[_arm_idx_np],
                        joint_indices=_arm_idx_np))
                else:
                    controller.apply_action(ArticulationAction(
                        joint_positions=q_cmd, joint_indices=_arm_idx_np))
                controller.apply_action(ArticulationAction(
                    joint_efforts=np.full(len(grip_idx), squeeze["eff"], dtype=np.float32),
                    joint_indices=_grip_idx_np))
            else:
                # position control on all DOFs. Once squeezing in position mode,
                # force the jaw target CLOSED (past the cube) regardless of the
                # grip_val the caller passed, so the capped-force squeeze holds.
                gv = GRIP_CLOSED if (squeeze["active"] and args_cli.squeeze_mode == "position") else grip_val
                tgt = full_targets(q_cmd, gv)  # q_cmd already has the arm cmd-offset applied
                if gff is not None:
                    controller.apply_action(ArticulationAction(joint_positions=tgt, joint_efforts=gff))
                else:
                    controller.apply_action(ArticulationAction(joint_positions=tgt))
            world.step(render=do_render)
            step_ctr["n"] += 1
            if step_ctr["n"] % 10 == 0:
                _bc(f"drive_step_{step_ctr['n']}")  # heartbeat for the watchdog
            if do_render:
                capture()

    def traj(q0, q1, grip_val, steps, label, render=True):
        q0, q1 = np.asarray(q0), np.asarray(q1)
        for i in range(1, steps + 1):
            drive(((1 - i / steps) * q0 + (i / steps) * q1).tolist(), grip_val, 2, render)
        cz = cube_pose()[0][2]
        log(f"[TRAJ {label}] cube_z={cz:.4f}m")
        return cz

    def report(label):
        j1, _ = link_world_pose(stage, "/World/AR4/root_joint/gripper_jaw1_link")
        cp = cube_pose()[0]
        log(f"[REPORT {label}] fingertip_z={j1[2]-JAW1_LOWER_EXTENT:.4f}m cube_z={cp[2]:.4f}m")

    def jaw_geometry(label):
        """WORLD-frame jaw separation + cube-vs-jaw-midpoint offset -- the
        direct proof (or disproof) that the jaws are genuinely closing onto
        the cube rather than commanding a target that never takes physical
        effect (this task's central defect to catch).

        Reports BOTH the raw jaw-LINK-origin separation/offset (for
        continuity with the module's other diagnostics) AND a
        fingertip-corrected version, since `report()`'s own pre-existing
        `fingertip_z = j1[2] - JAW1_LOWER_EXTENT` formula (already trusted
        by this repo's prior height-fix work) establishes that the actual
        gripping surface sits JAW1_LOWER_EXTENT=18.475mm below each jaw
        LINK's own world-Z, not at the link origin itself -- using the raw
        link origin alone overstates any real cube-to-fingertip gap by
        that same ~18.5mm. Applying the identical world-Z correction to
        BOTH jaws (matching the existing, already-validated convention,
        rather than inventing a new per-jaw local-frame rotation that could
        introduce a fresh sign bug) gives the fingertip-corrected numbers
        that should be trusted for judging real contact."""
        p1, _ = link_world_pose(stage, GRIP1_PATH)
        p2, _ = link_world_pose(stage, GRIP2_PATH)
        sep_mm = float(np.linalg.norm(p1 - p2) * 1000.0)
        mid = (p1 + p2) / 2.0
        cp = cube_pose()[0]
        off_vec_mm = (cp - mid) * 1000.0
        off_mm = float(np.linalg.norm(off_vec_mm))

        ext = np.array([0.0, 0.0, JAW1_LOWER_EXTENT])
        f1, f2 = p1 - ext, p2 - ext
        fsep_mm = float(np.linalg.norm(f1 - f2) * 1000.0)
        fmid = (f1 + f2) / 2.0
        foff_vec_mm = (cp - fmid) * 1000.0
        foff_mm = float(np.linalg.norm(foff_vec_mm))

        dof_grip = np.array(robot.get_joint_positions())[grip_idx]
        log(f"[JAW {label}] RAW: sep={sep_mm:.2f}mm cube_vs_jaw_mid_offset={off_mm:.2f}mm "
            f"(dxyz_mm=[{off_vec_mm[0]:.2f},{off_vec_mm[1]:.2f},{off_vec_mm[2]:.2f}]) | "
            f"FINGERTIP-CORRECTED: sep={fsep_mm:.2f}mm cube_vs_fingertip_mid_offset={foff_mm:.2f}mm "
            f"(dxyz_mm=[{foff_vec_mm[0]:.2f},{foff_vec_mm[1]:.2f},{foff_vec_mm[2]:.2f}]) | "
            f"grip_dof={dof_grip.tolist()} jaw1_pos_m={p1.tolist()} jaw2_pos_m={p2.tolist()}")
        return fsep_mm, foff_mm

    def _pad_mid_xyz():
        """Tilt-correct true jaw-pad midpoint (world). Same measured 12.5mm
        local-Y offset as the servo's fingertip_mid_xyz, duplicated here so the
        diagnostic is usable independent of the servo block."""
        p1, q1 = link_world_pose(stage, GRIP1_PATH)
        p2, q2 = link_world_pose(stage, GRIP2_PATH)
        off = np.array([0.0, -0.0125, 0.0])
        return (p1 + quat_apply(q1, off) + p2 + quat_apply(q2, off)) / 2.0

    # ======================================================================
    # DROOP DIAGNOSTIC MODE (2026-07-30, ar4-gravity-droop task)
    # ======================================================================
    if args_cli.droop_diagnostic:
        log("\n" + "#" * 70)
        log("# JOINT-LEVEL DROOP DIAGNOSTIC")
        log("#" * 70)
        cube_p = cube_pose()[0]
        cube_center_z = float(cube_p[2])
        log(f"[DIAG] cube center world = {cube_p.tolist()}  (cube_center_z={cube_center_z:.5f})")

        # introspect the articulation API surface (once) so we KNOW the exact
        # generalized-gravity / effort-limit method names on this 5.1 build.
        api_hits = sorted([m for m in dir(robot) if any(
            k in m.lower() for k in ("grav", "mass", "coriolis", "jacob", "effort", "force", "max"))])
        log(f"[DIAG API] robot methods of interest: {api_hits}")

        # command the reachable GRASP_Q and settle HARD to true steady state
        drive(HOME_Q, GRIP_OPEN, 40, render=False)
        traj(HOME_Q, PREGRASP_Q, GRIP_OPEN, 80, "DIAG HOME->PREGRASP", render=False)
        traj(PREGRASP_Q, GRASP_Q, GRIP_OPEN, 60, "DIAG PREGRASP->GRASP", render=False)
        drive(GRASP_Q, GRIP_OPEN, 400, render=False)  # long settle -> steady state

        def _per_joint_report(tag):
            ach = np.array(robot.get_joint_positions())
            cmd = np.array(GRASP_Q)
            try:
                meff = np.array(robot.get_measured_joint_efforts()).reshape(-1)
            except Exception as e:
                meff = None
                log(f"[DIAG {tag}] get_measured_joint_efforts failed: {e}")
            pad = _pad_mid_xyz()
            log(f"[DIAG {tag}] pad_mid_z={pad[2]:.5f}  pad_z_err_vs_cube_center_mm={(pad[2]-cube_center_z)*1000:+.2f} "
                f"pad_xyz={['%.5f'%v for v in pad]}")
            for k, i in enumerate(arm_idx):
                err_deg = math.degrees(float(ach[i] - cmd[k]))
                eff = float(meff[i]) if meff is not None else float('nan')
                log(f"[DIAG {tag}] {ARM_JOINTS[k]:8s} cmd={math.degrees(cmd[k]):+8.3f}deg "
                    f"ach={math.degrees(float(ach[i])):+8.3f}deg  err={err_deg:+7.3f}deg  "
                    f"measured_effort={eff:+9.3f}  drive_maxForce={ARM_MAX_FORCE:.0f}")
            return pad, meff

        log("\n[DIAG] --- (A) GRAVITY ON, PD only (baseline droop) ---")
        pad_base, meff_base = _per_joint_report("A_grav_on")

        if args_cli.quick:
            # RUN-7 finding: cmd_integral offsets SATURATE yet joints stay off
            # command (joint_2 pinned ~52deg pushing ~480 N*m) -> GRASP_Q may be
            # physically UNREACHABLE (joint limit or collision), not a control issue.
            # Read the actual USD joint limits (pxr is available inside the app) to
            # distinguish a LIMIT block from a COLLISION block, and check arm-link
            # net contact via measured joint forces.
            log("[DIAG LIMITS] USD revolute-joint limits vs GRASP_Q:")
            for k, _jn in enumerate(ARM_JOINTS):
                _jp = find_prim_by_name(stage, AR4_ROOT, _jn)
                lo = hi = None
                if _jp is not None and _jp.IsValid():
                    try:
                        _rj = UsdPhysics.RevoluteJoint(_jp)
                        lo = _rj.GetLowerLimitAttr().Get()
                        hi = _rj.GetUpperLimitAttr().Get()
                    except Exception as _e:
                        log(f"[DIAG LIMITS] {_jn}: read failed {_e}")
                gq = math.degrees(GRASP_Q[k])
                _flag = ""
                if lo is not None and hi is not None:
                    if gq <= lo + 1.0:
                        _flag = f">>> GRASP_Q AT/BELOW LOWER LIMIT ({lo})"
                    elif gq >= hi - 1.0:
                        _flag = f">>> GRASP_Q AT/ABOVE UPPER LIMIT ({hi})"
                log(f"[DIAG LIMITS] {_jn:8s} USD_limits=[{lo},{hi}]deg  GRASP_Q={gq:+.2f}deg  {_flag}")

            # QUICK mode: validate the cmd_integral tracking assist cheaply. Report
            # baseline joint errors (section A above), then activate cmd_integral,
            # settle, and report whether the per-joint errors are nulled and the pad
            # reaches cube-center Z.
            log(f"[DIAG QUICK] activating cmd_integral (kc={grav['kc']} clamp_deg="
                f"{math.degrees(grav['cmd_clamp']):.0f}) at GRASP_Q")
            grav["mode"] = "cmd_integral"
            _reset_cmd_off()
            grav["active"] = True
            for _w in (100, 100, 100, 100, 100, 100):
                drive(GRASP_Q, GRIP_OPEN, _w, render=False)
                pad_q, _ = _per_joint_report(f"QUICK_cmdint_{step_ctr['n']}")
                log(f"[DIAG QUICK] pad_z residual_vs_cube_center={(pad_q[2]-cube_center_z)*1000:+.2f}mm "
                    f"cmd_off_deg={[round(math.degrees(float(grav['cmd_off'][i])),2) for i in arm_idx]}")
            log(f"[DIAG QUICK] VERDICT: baseline pad_z {(pad_base[2]-cube_center_z)*1000:+.2f}mm -> "
                f"after cmd_integral {(pad_q[2]-cube_center_z)*1000:+.2f}mm "
                f"({'NULLED (cmd_integral fix works)' if abs(pad_q[2]-cube_center_z)<0.003 else 'residual remains'})")
            grav["active"] = False
            log("\n[DIAG] DIAGNOSTIC COMPLETE (VERDICT: DROOP_DIAG_DONE)")
            _R.close(); simulation_app.close(); return

        # generalized gravity force query
        g_vec = _query_gravity_forces()
        if g_vec is not None:
            log(f"[DIAG] generalized gravity forces G(q) [all dof] = {[round(float(v),3) for v in g_vec]}")
            log(f"[DIAG] G(q) on ARM dofs = {[round(float(g_vec[i]),3) for i in arm_idx]}  "
                f"(via {grav['api'][1] if grav['api'] else '?'})")
            if meff_base is not None:
                log("[DIAG] SATURATION CHECK per arm joint: |measured_effort| vs drive_maxForce "
                    f"({ARM_MAX_FORCE:.0f}) and |G|:")
                for k, i in enumerate(arm_idx):
                    log(f"[DIAG]   {ARM_JOINTS[k]:8s} |meas|={abs(float(meff_base[i])):8.3f} "
                        f"|G|={abs(float(g_vec[i])):8.3f} maxForce={ARM_MAX_FORCE:.0f} "
                        f"-> {'SATURATING' if abs(float(meff_base[i]))>0.95*ARM_MAX_FORCE else 'below cap'}")
        else:
            log("[DIAG] generalized gravity force API NOT available on this build (tried candidates)")

        log("\n[DIAG] --- (B) GRAVITY-OFF CONTROL: robot.disable_gravity(), re-settle ---")
        # DEFINITIVE gravity-off via the articulation's own disable_gravity()
        # (the RUN-5 introspection confirmed robot exposes disable/enable_gravity;
        # the earlier physics_context.set_gravity path was suspect -- results were
        # suspiciously identical, so use the proper API here).
        goff_ok = False
        try:
            robot.disable_gravity()
            goff_ok = True
            log("[DIAG] robot.disable_gravity() called")
        except Exception as e:
            log(f"[DIAG] robot.disable_gravity() failed: {e}")
        drive(GRASP_Q, GRIP_OPEN, 400, render=False)
        pad_goff, _ = _per_joint_report("B_grav_off")
        log(f"[DIAG] GRAVITY-OFF verdict (disable_gravity ok={goff_ok}): pad_z moved "
            f"{(pad_goff[2]-pad_base[2])*1000:+.2f}mm vs grav-on; "
            f"residual_vs_cube_center={(pad_goff[2]-cube_center_z)*1000:+.2f}mm "
            f"({'REACHES cube center (gap WAS gravity)' if abs(pad_goff[2]-cube_center_z)<0.003 else 'STILL off center (gap is NOT gravity -> tracking/friction)'})")
        try:
            robot.enable_gravity()
            log("[DIAG] robot.enable_gravity() -- gravity restored")
        except Exception as e:
            log(f"[DIAG] robot.enable_gravity() failed: {e}")
        drive(GRASP_Q, GRIP_OPEN, 200, render=False)

        log("\n[DIAG] --- (C) FEEDFORWARD GRAVITY COMP test (both signs) ---")
        if g_vec is None:
            log("[DIAG] no gravity-force API -> cannot test feedforward comp; skipping (C)")
        else:
            grav["mode"] = "gravity"
            _per_joint_report("C0_grav_on_recheck")
            for sgn in (1.0, -1.0):
                grav["active"] = True
                grav["sign"] = sgn
                drive(GRASP_Q, GRIP_OPEN, 300, render=False)
                pad_ff, _ = _per_joint_report(f"C_ff_sign{int(sgn):+d}")
                log(f"[DIAG] feedforward sign={sgn:+.0f}: pad_z residual_vs_cube_center="
                    f"{(pad_ff[2]-cube_center_z)*1000:+.2f}mm "
                    f"(baseline droop was {(pad_base[2]-cube_center_z)*1000:+.2f}mm)")
                grav["active"] = False
                drive(GRASP_Q, GRIP_OPEN, 150, render=False)  # relax back
            grav["mode"] = args_cli.assist_mode

        log("\n[DIAG] --- (D) POSITION-DRIVE investigation (why wrist joints don't track) ---")
        # The +9mm gap is a POSITION-TRACKING failure: wrist joints settle 4-9deg
        # off command at near-zero effort (an 8000-stiffness drive would exert
        # ~1200 N*m at 8.6deg). Effort feedforward is unstable (no effective drive
        # to stabilize it -> runaway). So investigate the POSITION drive directly.
        # (D1) read back the effective per-DOF drive gains.
        gains_str = "unavailable"
        try:
            gk, gd = controller.get_gains()
            gk = np.asarray(gk).reshape(-1); gd = np.asarray(gd).reshape(-1)
            gains_str = "; ".join(f"{ARM_JOINTS[k]}:kp={gk[i]:.1f},kd={gd[i]:.1f}" for k, i in enumerate(arm_idx))
        except Exception as e:
            log(f"[DIAG D1] controller.get_gains() failed: {e}")
        log(f"[DIAG D1] effective arm drive gains: {gains_str}")

        # (D2) per-joint command-tracking probe: from GRASP_Q, add +5deg to ONE
        # arm joint's target, settle, measure how far that joint ACTUALLY moved.
        # delta~5deg => drive tracks; delta~0 => drive dead on that joint.
        drive(GRASP_Q, GRIP_OPEN, 200, render=False)
        base_ach = np.array(robot.get_joint_positions())
        for k, i in enumerate(arm_idx):
            qtest = list(GRASP_Q); qtest[k] += math.radians(5.0)
            drive(qtest, GRIP_OPEN, 200, render=False)
            moved = math.degrees(float(np.array(robot.get_joint_positions())[i] - base_ach[i]))
            log(f"[DIAG D2] {ARM_JOINTS[k]:8s} commanded +5.00deg -> joint moved {moved:+6.2f}deg "
                f"({'TRACKS' if abs(moved-5.0)<1.5 else 'WEAK/DEAD drive'})")
            drive(GRASP_Q, GRIP_OPEN, 150, render=False)  # return

        # (D3) FIX test: raise ALL arm drive gains hard (position-only, no effort),
        # re-settle at GRASP_Q, and see if the joints now track -> pad at center.
        for stiff in (50000.0, 200000.0):
            kps2 = np.array(kps, dtype=np.float32); kds2 = np.array(kds, dtype=np.float32)
            for i in arm_idx:
                kps2[i] = stiff; kds2[i] = 2.0 * math.sqrt(stiff)  # ~critical-ish
            controller.set_gains(kps=kps2, kds=kds2)
            drive(GRASP_Q, GRIP_OPEN, 400, render=False)
            pad_s, _ = _per_joint_report(f"D3_stiff{int(stiff)}")
            log(f"[DIAG D3] arm stiffness={stiff:.0f}: pad_z residual_vs_cube_center="
                f"{(pad_s[2]-cube_center_z)*1000:+.2f}mm "
                f"({'REACHES center (stiffness fix works)' if abs(pad_s[2]-cube_center_z)<0.003 else 'still off'})")
        # restore nominal gains
        controller.set_gains(kps=kps, kds=kds)

        log("\n[DIAG] --- (E) ARMATURE stabilization test (distal-joint drive fix) ---")
        # D2/D3 show the distal joints (esp joint_6) don't track: near-zero drive
        # effort at large error, WORSE with higher stiffness -> low-inertia drive
        # pathology. Textbook cure: add ARMATURE (reflected inertia) to the DOFs so
        # the position drive is well-conditioned. Introspect the API, read joint
        # limits (rule out a limit), then set armature and retest tracking+settle.
        arm_methods = sorted([m for m in dir(robot) if 'armature' in m.lower()])
        log(f"[DIAG E] armature-related robot methods: {arm_methods}")
        try:
            lim = robot.get_dof_limits()
            lim = np.asarray(lim)
            lim = lim.reshape(-1, 2) if lim.ndim > 1 else lim
            for k, i in enumerate(arm_idx):
                lo, hi = lim[i]
                log(f"[DIAG E] {ARM_JOINTS[k]:8s} limits=[{math.degrees(float(lo)):+.1f},{math.degrees(float(hi)):+.1f}]deg")
        except Exception as e:
            log(f"[DIAG E] get_dof_limits failed: {e}")

        def _try_set_armature(val):
            av = np.zeros(robot.num_dof, dtype=np.float32)
            for i in arm_idx:
                av[i] = val
            for name in ("set_armatures", "set_armature"):
                fn = getattr(robot, name, None)
                if fn is None:
                    continue
                try:
                    fn(av)
                    return name
                except Exception as e:
                    log(f"[DIAG E] {name}({val}) failed: {e}")
            return None

        for armv in (0.05, 0.5):
            used = _try_set_armature(armv)
            log(f"[DIAG E] set armature={armv} via {used}")
            controller.set_gains(kps=kps, kds=kds)  # nominal 8000/600
            drive(GRASP_Q, GRIP_OPEN, 400, render=False)
            pad_a, _ = _per_joint_report(f"E_armature{armv}")
            # per-joint tracking retest under armature
            base_a = np.array(robot.get_joint_positions())
            for k, i in enumerate(arm_idx):
                qt = list(GRASP_Q); qt[k] += math.radians(5.0)
                drive(qt, GRIP_OPEN, 150, render=False)
                mv = math.degrees(float(np.array(robot.get_joint_positions())[i] - base_a[i]))
                log(f"[DIAG E armature={armv}] {ARM_JOINTS[k]:8s} +5deg cmd -> moved {mv:+6.2f}deg "
                    f"({'TRACKS' if abs(mv-5.0)<1.5 else 'off'})")
                drive(GRASP_Q, GRIP_OPEN, 120, render=False)
            log(f"[DIAG E] armature={armv}: pad_z residual_vs_cube_center="
                f"{(pad_a[2]-cube_center_z)*1000:+.2f}mm "
                f"({'REACHES center (armature fix works)' if abs(pad_a[2]-cube_center_z)<0.003 else 'still off'})")
        _try_set_armature(0.0)

        log("\n[DIAG] DIAGNOSTIC COMPLETE (VERDICT: DROOP_DIAG_DONE)")
        _R.close()
        simulation_app.close()
        return

    # ======================================================================
    # GRIPPER-AABB DUMP (2026-07-31, side-grasp support-collision diagnostic)
    # ======================================================================
    if args_cli.dump_gripper_aabb:
        log("\n" + "#" * 70)
        log("# GRIPPER COLLISION-MESH WORLD AABB @ GRASP_Q (measure body hang below pads)")
        log("#" * 70)
        drive(HOME_Q, GRIP_OPEN, 40, render=False)
        traj(HOME_Q, PREGRASP_Q, GRIP_OPEN, 80, "AABB HOME->PREGRASP", render=False)
        traj(PREGRASP_Q, GRASP_Q, GRIP_OPEN, 60, "AABB PREGRASP->GRASP", render=False)
        drive(GRASP_Q, GRIP_OPEN, 300, render=False)  # settle
        pad = _pad_mid_xyz()
        log(f"[AABB] pad_mid world = {['%.4f'%v for v in pad]}")
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        for _lp in ("/World/AR4/root_joint/gripper_base_link", GRIP1_PATH, GRIP2_PATH, LINK6_PATH):
            _pr = stage.GetPrimAtPath(_lp)
            if not _pr.IsValid():
                # gripper_base_link path may differ; find by name
                _pr2 = find_prim_by_name(stage, AR4_ROOT, _lp.split("/")[-1])
                _pr = _pr2 if (_pr2 is not None and _pr2.IsValid()) else _pr
            if not _pr.IsValid():
                log(f"[AABB] {_lp}: prim not found")
                continue
            try:
                rng = bbc.ComputeWorldBound(_pr).ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                log(f"[AABB] {_pr.GetName():20s} world min=({mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f}) "
                    f"max=({mx[0]:.4f},{mx[1]:.4f},{mx[2]:.4f}) | "
                    f"z_below_pad_mm={(pad[2]-mn[2])*1000:.1f} z_above_pad_mm={(mx[2]-pad[2])*1000:.1f} "
                    f"x_min_mm={mn[0]*1000:.1f} x_max_mm={mx[0]*1000:.1f}")
            except Exception as _e:
                log(f"[AABB] {_pr.GetName()}: bbox failed: {_e}")
        # combined gripper-assembly lowest point vs the cube-mid-height support plane
        log(f"[AABB] SUPPORT-CLEARANCE: cube center z=0.09, cube bottom (support top)=0.0825. "
            f"Any gripper AABB z_min BELOW 0.0825 while over the support XY footprint => collision.")
        if PIN_PICK:
            # pin-pick clearance verdicts: (1) does the descending gripper BODY
            # clear the object TOP? (2) is the thin pin (center, thin) clear of the
            # fingers/body? Read the body z_min from the just-logged AABBs.
            _bb2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
            _body = find_prim_by_name(stage, AR4_ROOT, "gripper_base_link")
            _body_zmin = None
            if _body is not None and _body.IsValid():
                try:
                    _body_zmin = _bb2.ComputeWorldBound(_body).ComputeAlignedRange().GetMin()[2]
                except Exception:
                    _body_zmin = None
            log(f"[AABB PIN] object_top={PIN_OBJ_TOP:.4f} object_bottom={PIN_OBJ_BOTTOM:.4f} "
                f"pin: center=({PIN_XY[0]:.4f},{PIN_XY[1]:.4f}) half_width={PIN_SZ/2*1000:.1f}mm "
                f"top_z={PIN_HEIGHT:.4f}")
            if _body_zmin is not None:
                _clr = (_body_zmin - PIN_OBJ_TOP) * 1000.0
                log(f"[AABB PIN] BODY-vs-OBJECT-TOP clearance = body_z_min({_body_zmin:.4f}) - "
                    f"object_top({PIN_OBJ_TOP:.4f}) = {_clr:+.1f}mm "
                    f"({'CLEARS (body descends past object top)' if _clr > 0.5 else 'COLLISION RISK -- lower object_top'})")
            log(f"[AABB PIN] the pin rises only at the CENTER (x={PIN_XY[0]:.4f},y={PIN_XY[1]:.4f}) to "
                f"z={PIN_HEIGHT:.4f}; fingertips reach ~z=0.038 at X-extremes -> the thin central pin is "
                f"below/between them. Confirm jaw X-spans above straddle the object, not the pin.")
        log("\n[AABB] DUMP COMPLETE (VERDICT: AABB_DUMP_DONE)")
        _R.close(); simulation_app.close(); return

    # ======================================================================
    # FREE-SPACE JAW-CLOSE TEST (2026-07-31): do BOTH jaw drives close to ~0 in
    # free space? -- clean control to separate a one-jaw-stuck DRIVE bug from an
    # object-interaction CENTERING failure (tall-prism sweep showed jaw2 never
    # moving). At HOME the gripper is well clear of any object.
    # ======================================================================
    if args_cli.jaw_close_test:
        log("\n" + "#" * 70)
        log(f"# FREE-SPACE JAW-CLOSE TEST (jaw_max_force={args_cli.jaw_max_force} "
            f"grip_kd={args_cli.grip_kd} close_steps={args_cli.close_steps})")
        log("#" * 70)
        drive(HOME_Q, GRIP_OPEN, 60, render=False)
        d0 = np.array(robot.get_joint_positions())[grip_idx]
        log(f"[JAWTEST] jaws OPEN in free space: grip_dof={d0.tolist()}")
        start_squeeze()
        for w in range(10):
            drive(HOME_Q, GRIP_CLOSED, 30, render=False)
            dj = np.array(robot.get_joint_positions())[grip_idx]
            ef = None
            try:
                ef = [round(float(v), 3) for v in np.asarray(robot.get_measured_joint_efforts())[grip_idx]]
            except Exception:
                pass
            log(f"[JAWTEST] after {(w+1)*30} close steps: grip_dof={[round(float(v),5) for v in dj]} "
                f"efforts_N={ef}  (target GRIP_CLOSED={GRIP_CLOSED}; both -> ~0 == both drives close)")
        dj = np.array(robot.get_joint_positions())[grip_idx]
        both_closed = bool(abs(dj[0]) < 0.002 and abs(dj[1]) < 0.002)
        log(f"[JAWTEST] VERDICT: both jaws closed in free space = {both_closed} "
            f"(final grip_dof={[round(float(v),5) for v in dj]})")
        log("[JAWTEST] JAW-CLOSE TEST COMPLETE (VERDICT: JAWTEST_DONE)")
        _R.close(); simulation_app.close(); return

    # ---- execute the pick -------------------------------------------------
    drive(HOME_Q, GRIP_OPEN, 40, render=False)
    cube_rest_z = cube_pose()[0][2]
    log(f"[INFO] cube resting z={cube_rest_z:.4f}m")
    jaw_geometry("P0_HOME(open)")
    if not args_cli.no_video:
        position_cameras()

    cz = {"P0_HOME": cube_rest_z}
    cz["P1_PREGRASP"] = traj(HOME_Q, PREGRASP_Q, GRIP_OPEN, 80, "HOME->PREGRASP")
    if not args_cli.no_video:
        position_cameras()
    cz["P2_GRASP"] = traj(PREGRASP_Q, GRASP_Q, GRIP_OPEN, 60, "PREGRASP->GRASP")
    report("at GRASP (open)")
    jaw_geometry("P2_GRASP(open, before descent)")

    # ---- TRACKING ASSIST ACTIVATION -------------------------------------
    # RUN-5: the +9-16mm vertical gap was measured to be NON-gravitational (a
    # joint tracking offset, esp. the wrist joints, from friction/weak drive --
    # see the droop diagnostic). Default assist is the INTEGRAL-effort term:
    # activate it here and settle, so the achieved pose is pulled ONTO the
    # commanded FK pose (pad at cube center) BEFORE the servo probes -- then
    # leave it ON for the whole pick (servo, close, lift, retreat) so the arm
    # keeps tracking every commanded pose.
    if args_cli.gravity_comp:
        cube_center_z = float(cube_pose()[0][2])
        if grav["mode"] == "gravity":
            # legacy feedforward path with live sign calibration
            g_probe = _query_gravity_forces()
            if g_probe is None:
                log("[ASSIST] WARNING: gravity API unavailable -- gravity assist DISABLED.")
            else:
                drive(GRASP_Q, GRIP_OPEN, 120, render=False)
                z_none = _pad_mid_xyz()[2]
                grav["active"] = True; grav["sign"] = 1.0
                drive(GRASP_Q, GRIP_OPEN, 200, render=False)
                z_plus = _pad_mid_xyz()[2]
                grav["sign"] = -1.0
                drive(GRASP_Q, GRIP_OPEN, 200, render=False)
                z_minus = _pad_mid_xyz()[2]
                grav["sign"] = 1.0 if abs(z_plus-cube_center_z) <= abs(z_minus-cube_center_z) else -1.0
                grav["active"] = True
                drive(GRASP_Q, GRIP_OPEN, 200, render=False)
                log(f"[ASSIST gravity] chose sign={grav['sign']:+.0f}; residual_vs_center="
                    f"{(_pad_mid_xyz()[2]-cube_center_z)*1000:+.2f}mm (no-comp {(z_none-cube_center_z)*1000:+.2f}mm)")
        elif grav["mode"] == "cmd_integral":
            # CMD_INTEGRAL assist (default, RUN-7 fix): accumulate a per-joint
            # command offset so achieved -> GRASP_Q (pad at FK cube center), via the
            # stable position drive. Reset, activate, settle. Stays ON for the whole
            # pick so every commanded pose (servo targets, lift) is tracking-comped.
            drive(GRASP_Q, GRIP_OPEN, 100, render=False)
            z_none = _pad_mid_xyz()[2]
            _reset_cmd_off()
            grav["active"] = True
            drive(GRASP_Q, GRIP_OPEN, 600, render=False)  # converge the command offset
            z_final = _pad_mid_xyz()[2]
            log(f"[ASSIST cmd_integral] kc={grav['kc']}: pad_z no-assist={z_none:.5f} "
                f"-> after={z_final:.5f} cube_center={cube_center_z:.5f}; "
                f"residual_vs_center={(z_final-cube_center_z)*1000:+.2f}mm "
                f"(no-assist gap was {(z_none-cube_center_z)*1000:+.2f}mm); "
                f"cmd_off_deg={[round(math.degrees(float(grav['cmd_off'][i])),2) for i in arm_idx]}")
        else:
            # INTEGRAL (effort) assist -- retained for the record; UNSTABLE here.
            drive(GRASP_Q, GRIP_OPEN, 100, render=False)
            z_none = _pad_mid_xyz()[2]
            _reset_integrator()
            grav["active"] = True
            drive(GRASP_Q, GRIP_OPEN, 500, render=False)
            log(f"[ASSIST integral] residual_vs_center="
                f"{(_pad_mid_xyz()[2]-cube_center_z)*1000:+.2f}mm (no-assist {(z_none-cube_center_z)*1000:+.2f}mm)")

    # ---- 3-AXIS EMPIRICAL-JACOBIAN SERVO TO CUBE CENTER ------------------
    # ROOT CAUSE this replaces (measured live 2026-07-30): the prior 1-D
    # "vertical" descent nudged along a PRECOMPUTED joint-space direction
    # (VERT_DESCENT_DIR). Under real gravity droop that joint direction is NOT
    # vertical in the ACHIEVED frame -- live data showed it improved fingertip
    # Z by ~5mm but simultaneously dragged it ~12mm sideways in Y, ending 18mm
    # off the cube in Y. The jaws then closed along X onto EMPTY AIR (contact
    # 0N, jaw sep collapsed 28mm->0.3mm passing PAST each other, cube gain
    # 0.0mm). Squeeze force / friction are irrelevant until the jaws physically
    # straddle the cube. Fix: measure a full 3x6 world-frame Jacobian
    # d(fingertip_mid)/d(arm_q) EMPIRICALLY in the sim (so droop is baked into
    # the measurement, and error + Jacobian + correction all live in the SAME
    # world frame -- no offline-FK base-frame alignment to get wrong), then
    # DLS-solve the arm-joint correction that drives the fingertip to the cube
    # CENTER in x, y AND z at once. The Jacobian is probed ONCE while the cube
    # is still ~18mm away in Y (jaws open, clear of the cube) so probing can't
    # disturb it; the columns are ~constant over the small final correction.
    def fingertip_mid_xyz():
        # TRUE pad-center midpoint, tilt-correct. MEASURED directly from the USD
        # jaw collision geometry (scripts/_inspect_jaw_geometry.py, 2026-07-30):
        # each jaw's pad collision centroid sits at (0, -0.0125, 0) in that jaw
        # LINK's OWN local frame -- 12.5mm along local -Y, NOT 18.475mm along
        # world/local -Z as the standing JAW1_LOWER_EXTENT constant assumed. At
        # this grasp pose the gripper is ~79deg from vertical, so that 12.5mm
        # offset points nearly horizontally in world; the old world-Z
        # subtraction targeted a phantom point ~15mm off the real pads (the root
        # cause of 9 straight zero-contact runs). Rotate the measured local
        # offset by each jaw's ACTUAL orientation to get the true world pad
        # centers, whose midpoint is the grasp centerline the cube must sit on.
        p1, q1 = link_world_pose(stage, GRIP1_PATH)
        p2, q2 = link_world_pose(stage, GRIP2_PATH)
        off = np.array([0.0, -0.0125, 0.0])   # pad centroid in jaw-link local frame (measured)
        f1 = p1 + quat_apply(q1, off)
        f2 = p2 + quat_apply(q2, off)
        return (f1 + f2) / 2.0

    def gripper_quat():
        _, q = link_world_pose(stage, LINK6_PATH)
        return np.asarray(q, dtype=float)  # wxyz

    def rot_err_vec(q_to, q_from):
        """World-frame rotation VECTOR (axis*angle, rad) that rotates q_from
        onto q_to. Zero when aligned. Small-angle-safe."""
        qe = quat_mul(q_to, quat_inv(q_from))
        if qe[0] < 0:
            qe = -qe
        v = qe[1:4]
        s = float(np.linalg.norm(v))
        if s < 1e-12:
            return np.zeros(3)
        return v / s * (2.0 * math.atan2(s, qe[0]))

    # ---- 6-DOF POSE SERVO: reach cube center AND HOLD gripper orientation --
    # ROOT CAUSE the 3-DOF position-only servo could not beat (mapped across 6
    # runs): controlling only the fingertip POSITION leaves the 3 redundant
    # DOF (gripper ORIENTATION) free, and the DLS pseudo-inverse let it TILT --
    # so descending toward the cube's Z dragged the fingertip sideways. The
    # reachable frontier was a dead-end: fingertip-low needed X~13mm (open jaws
    # then COLLIDE the cube) OR Y~19mm (gripper past the cube); no pose had all
    # of X<6.5 (jaw straddle) & Y<7.5 (pad overlap) & fingertip below the cube
    # top. Higher stiffness (run 6) did NOT lower that frontier -- it's
    # geometric, not droop-force. Fix: control the FULL 6-DOF pose -- drive the
    # fingertip to the cube center WHILE holding the gripper's orientation fixed
    # at its initial (jaws-along-X, pointing-down) grasp orientation, so the
    # descent stays VERTICAL and does not drag X/Y. Same empirical + Broyden
    # Jacobian machinery, now 6x6. Orientation error is scaled by a lever arm
    # L (m/rad) into position units so the DLS balances translation & rotation.
    # L=0 disables orientation-hold: now that fingertip_mid_xyz targets the TRUE
    # (tilt-correct) pad midpoint, this reduces to a pure, stable 3-DOF position
    # servo centering the REAL pads on the cube -- orientation is a free
    # byproduct and does not need constraining.
    L = 0.0
    cube_p0 = cube_pose()[0]
    # For the tall prism, target the fixed reachable pad plane (grip NEAR THE TOP),
    # NOT the object center (which is ~H/2 down and would drive the pad into the
    # ground). For the cube, target its center as before.
    _servo_target_z = PIN_GRIP_Z if (PIN_PICK and not SIDE_GRASP) else (PRISM_GRIP_Z if TALL_PRISM else cube_rest_z)
    target_xyz = np.array([cube_p0[0], cube_p0[1], _servo_target_z])  # grip point, world
    q_arm_cur = np.array(GRASP_Q, dtype=float)

    # --no_servo: skip the Jacobian-probe centering servo entirely and grasp at
    # GRASP_Q directly (the arm already lands the pad dead-on the object XY; the
    # probe otherwise BUMPS the object off-center). Settle briefly, then commit.
    if args_cli.no_servo:
        drive(q_arm_cur.tolist(), GRIP_OPEN, 60, render=(not args_cli.no_video))
        m_ns = fingertip_mid_xyz()
        log(f"[SERVO] --no_servo: grasping directly at GRASP_Q; pad_mid={['%.4f'%v for v in m_ns]} "
            f"target={['%.4f'%v for v in target_xyz]} "
            f"pos_err_mm={['%.2f'%((t-f)*1000) for t, f in zip(target_xyz, m_ns)]}")
        grasp_q_use = list(q_arm_cur)
        report("at GRASP (open, no_servo)")
        jaw_open_sep, jaw_open_off = jaw_geometry("P2_GRASP(open, no_servo)")
        _do_servo = False
    else:
        _do_servo = True

    if _do_servo:
        drive(q_arm_cur.tolist(), GRIP_OPEN, 45, render=(not args_cli.no_video))
        # ---- build a TRUE-VERTICAL gripper orientation target ----------------
        # Run 8 proved: HOLDING orientation makes the descent vertical (Z reached
        # 0.0501, below the cube top) -- but holding the INITIAL grasp orientation
        # (which is itself tilted) pushed the gripper 33mm past the cube in Y. The
        # correct hold target is a genuinely vertical gripper: jaw-separation axis
        # along world X, fingertips pointing straight down (-Z). Derive it live from
        # the actual jaw geometry: the sep-axis and approach-axis are FIXED in
        # link_6's frame, so find the link_6 orientation that maps them onto world
        # (X, -Z). Then descending moves purely in Z, not dragging X/Y.
        _pj1, _ = link_world_pose(stage, GRIP1_PATH)
        _pj2, _ = link_world_pose(stage, GRIP2_PATH)
        _pl6, _ql6 = link_world_pose(stage, LINK6_PATH)
        _ftm = fingertip_mid_xyz()
        _sep_w = (_pj2 - _pj1); _sep_w = _sep_w / (np.linalg.norm(_sep_w) + 1e-12)
        _app_w = (_ftm - _pl6); _app_w = _app_w / (np.linalg.norm(_app_w) + 1e-12)
        _qi = quat_inv(_ql6)
        _sep_l6 = quat_apply(_qi, _sep_w)          # sep axis in link_6 frame (constant)
        _app_l6 = quat_apply(_qi, _app_w)          # approach axis in link_6 frame (constant)
        # orthonormal link_6-frame basis from (sep, approach)
        _e1 = _sep_l6 / (np.linalg.norm(_sep_l6) + 1e-12)
        _e2 = _app_l6 - np.dot(_app_l6, _e1) * _e1
        _e2 = _e2 / (np.linalg.norm(_e2) + 1e-12)
        _e3 = np.cross(_e1, _e2)
        _Bl6 = np.column_stack([_e1, _e2, _e3])
        if SIDE_GRASP:
            # SIDE grasp desired WORLD basis: sep -> world Y (grip the cube's +Y/-Y
            # vertical faces), approach -> world +X (palm on the -X side). Match the
            # live sep-axis SIGN so the target is the nearest axis-aligned horizontal
            # frame to the current (already-horizontal) GRASP_Q, not a 180deg flip.
            _sep_sign = 1.0 if _sep_w[1] >= 0 else -1.0
            _sep_tgt = np.array([0.0, _sep_sign, 0.0])
            _app_tgt = np.array([1.0, 0.0, 0.0])
            _third = np.cross(_sep_tgt, _app_tgt)
            _D = np.column_stack([_sep_tgt, _app_tgt, _third])
            _orient_desc = "horizontal (sep->Y, approach->+X)"
        else:
            # desired WORLD basis: sep -> +X, approach -> -Z, third = X x (-Z) = +Y
            _D = np.column_stack([np.array([1.0, 0.0, 0.0]),
                                  np.array([0.0, 0.0, -1.0]),
                                  np.array([0.0, 1.0, 0.0])])
            _orient_desc = "vertical (sep->X, approach->-Z)"
        _Rtgt = _D @ _Bl6.T
        q_orient_target = mat_to_quat_wxyz(_Rtgt)
        _init_oerr = float(np.linalg.norm(rot_err_vec(q_orient_target, _ql6)))
        log(f"[SERVO] orientation target built ({_orient_desc}); initial gripper is {_init_oerr:.3f}rad "
            f"({math.degrees(_init_oerr):.1f}deg) from target")
        target_m6 = np.concatenate([target_xyz, np.zeros(3)])

        def measure6():
            ft = fingertip_mid_xyz()
            ov = L * rot_err_vec(gripper_quat(), q_orient_target)  # 0 when aligned
            return np.concatenate([ft, ov])

        m0 = measure6()
        log(f"[SERVO] start fingertip={['%.4f'%v for v in m0[:3]]} cube_center={['%.4f'%v for v in target_xyz]} "
            f"pos_err_mm={['%.2f'%((t-f)*1000) for t, f in zip(target_xyz, m0[:3])]}")
        # --- estimate the 6x6 empirical pose Jacobian ONCE --------------------
        Jac = np.zeros((6, 6))
        dq_probe = math.radians(0.9)  # small enough not to reach the ~18mm-away cube
        for j in range(6):
            qp = q_arm_cur.copy(); qp[j] += dq_probe
            drive(qp.tolist(), GRIP_OPEN, 24, render=False)
            Jac[:, j] = (measure6() - m0) / dq_probe
            drive(q_arm_cur.tolist(), GRIP_OPEN, 16, render=False)  # return to base pose
        log(f"[SERVO] empirical 6x6 pose Jacobian estimated (rows: pos_xyz m/rad, orient*L m/rad)")
        # --- iterate DLS corrections; refine J ONLINE via Broyden -------------
        # Position acceptance box: X self-centers on close (~6mm), Y needs pad-face
        # overlap (~6mm), Z must put the fingertip below the cube top (~4mm).
        tol = np.array([6.0, 6.0, 4.0]) / 1000.0

        def reprobe_jacobian(qc):
            """Re-measure the full 6x6 pose Jacobian at pose qc (keeps it fresh so
            Broyden drift / a wandering pose can't corrupt it into instability)."""
            base = measure6_at(qc)
            Jn = np.zeros((6, 6))
            for jj in range(6):
                qpj = np.array(qc, dtype=float); qpj[jj] += dq_probe
                drive(qpj.tolist(), GRIP_OPEN, 22, render=False)
                Jn[:, jj] = (measure6() - base) / dq_probe
                drive(list(qc), GRIP_OPEN, 14, render=False)
            return Jn

        def measure6_at(qc):
            drive(list(qc), GRIP_OPEN, 20, render=False)
            return measure6()

        best_q = list(q_arm_cur)
        best_metric = 1.0e9
        prev_m6 = None
        prev_dq = None
        # STABILITY (run 7 diverged: lam 0.0006 + noisy 11000 gains -> huge steps,
        # orientation drifted 25deg). Strong DLS damping, small bounded steps,
        # moderate quiet gains, and a periodic Jacobian RE-PROBE so a bad Broyden
        # update can't compound.
        # GENTLER/BOUNDED (2026-07-30 run-1 fix): run 1's servo (maxstep 1.8deg, 52
        # iters) WANDERED 12-29mm off and bumped the cube when it couldn't descend.
        # With the droop-pre-compensated GRASP_Q above the pad now starts already in
        # the box, so the servo should converge on iteration 0; keep it small-step
        # and short so that if the pad lands slightly off it can only fine-trim, not
        # wander far enough to knock the cube.
        lam = 0.004
        maxstep = math.radians(0.8)
        for _it in range(26):
            drive(q_arm_cur.tolist(), GRIP_OPEN, 32, render=(not args_cli.no_video))
            m6 = measure6()
            # Broyden good-update from the LAST actually-observed move.
            if prev_m6 is not None and prev_dq is not None:
                dy = m6 - prev_m6
                denom = float(prev_dq @ prev_dq)
                if denom > 1e-12:
                    Jac = Jac + np.outer((dy - Jac @ prev_dq), prev_dq) / denom
            e6 = target_m6 - m6                       # [pos_err(m); -orient*L]
            perr = e6[:3]                             # position error (m)
            oerr = 0.0 if L <= 0.0 else float(np.linalg.norm(e6[3:]) / L)  # orient err (rad)
            pnorm = np.abs(perr) / tol                # per-axis pos error / tolerance
            ninf = float(np.max(pnorm))               # <=1 => position graspable
            hxy_mm = float(np.linalg.norm(perr[:2]) * 1000.0)
            metric = ninf + 0.5 * oerr                # prioritize pos box, penalize tilt
            log(f"[SERVO it{_it}] pad_mid={['%.4f'%v for v in m6[:3]]} "
                f"pos_err_mm=[{perr[0]*1000:+.2f},{perr[1]*1000:+.2f},{perr[2]*1000:+.2f}] "
                f"pos_normLinf={ninf:.2f} horiz={hxy_mm:.2f}mm orient_err_rad={oerr:.3f}")
            if metric < best_metric:
                best_metric = metric; best_q = list(q_arm_cur)
            if ninf <= 1.0 and oerr <= 0.10:
                log(f"[SERVO] converged: position within box (X<{tol[0]*1000:.0f} Y<{tol[1]*1000:.0f} "
                    f"Z<{tol[2]*1000:.0f}mm), orientation held ({oerr:.3f}rad) -- jaws straddle the cube")
                best_q = list(q_arm_cur); break
            # periodic re-probe keeps J trustworthy (Broyden alone drifted in run 7)
            if _it > 0 and _it % 16 == 0:
                log(f"[SERVO] re-probing Jacobian at it{_it}")
                Jac = reprobe_jacobian(q_arm_cur)
                prev_m6 = None; prev_dq = None
                continue
            dq = Jac.T @ np.linalg.solve(Jac @ Jac.T + lam * np.eye(6), e6)
            _m = float(np.max(np.abs(dq)))
            if _m > maxstep:
                dq = dq * (maxstep / _m)
            prev_m6 = m6
            prev_dq = dq.copy()
            q_arm_cur = q_arm_cur + dq
        grasp_q_use = list(best_q)         # commit the best pose reached
        log(f"[SERVO] committed pose: best metric {best_metric:.2f} (pos_normLinf<=1 & orient held == graspable)")
        report("at GRASP (open, after servo)")
        jaw_open_sep, jaw_open_off = jaw_geometry("P2_GRASP(open, after servo)")

    # ---- CLOSE the jaws onto the cube with a REAL squeeze force ----------
    contact_report("P2_before_close(open)")
    if args_cli.mechanism == "friction":
        # engage the genuine squeeze (effort or capped-position), then drive it
        # home. Every step goes through drive() so the closing effort is
        # re-applied continuously (a bare world.step() would NOT re-assert it).
        start_squeeze()
        drive(grasp_q_use, GRIP_CLOSED, args_cli.close_steps, render=True)
        sep_mid, _ = jaw_geometry("P3a_mid_close")
        contact_report("P3a_mid_close")
        # effort-mode sign self-check: a genuine CLOSE must DECREASE jaw
        # separation. If it didn't, the closing-force sign is wrong -- flip it.
        if args_cli.squeeze_mode == "effort" and sep_mid > jaw_open_sep - 1.0:
            squeeze["eff"] = -squeeze["eff"]
            log(f"[SQUEEZE] jaw sep did not decrease ({jaw_open_sep:.2f}->{sep_mid:.2f}mm); "
                f"flipping closing-effort sign to {squeeze['eff']:.3f} N/jaw and re-driving")
            drive(grasp_q_use, GRIP_CLOSED, args_cli.close_steps, render=True)
        # settle the squeeze firmly against the two faces
        drive(grasp_q_use, GRIP_CLOSED, args_cli.close_steps, render=True)
    else:
        # legacy diagnostics only (NOT the deliverable): plain position close.
        drive(grasp_q_use, GRIP_CLOSED, 80, render=True)
    l6p, l6q = link_world_pose(stage, LINK6_PATH)
    cp = cube_pose()[0]
    log(f"[GEO] link_6 pos={['%.4f'%v for v in l6p]} quat={['%.4f'%v for v in l6q]} cube={['%.4f'%v for v in cp]}")
    jaw_closed_sep, jaw_closed_off = jaw_geometry("P3_GRASP(after CLOSE command, before engage)")
    log(f"[JAW SUMMARY] open_sep={jaw_open_sep:.2f}mm -> closed_sep={jaw_closed_sep:.2f}mm "
        f"(target: ~28mm -> ~15mm-on-cube, NOT ~0mm/pass-through); "
        f"cube_centering_offset={jaw_closed_off:.2f}mm (want small, jaws straddling cube)")
    contact_report("P3_after_close(squeezing)")

    # ---- ENGAGE the grasp mechanism + VERIFY it registers ----------------
    if args_cli.mechanism == "surface_gripper":
        gripper_view.apply_gripper_action([0.5])  # close
        for _ in range(60):  # give the manager time to detect + attach
            world.step(render=RENDER)
            capture()
        status = gripper_view.get_surface_gripper_status()
        gripped = gripper_view.get_gripped_objects()
        log(f"[SURFACE_GRIPPER] status={status}  gripped_objects={gripped}")
        grasp_state["engaged"] = (len(gripped) > 0 and len(gripped[0]) > 0)
        grasp_state["mechanism"] = "SurfaceGripper"
        if not grasp_state["engaged"]:
            log("[SURFACE_GRIPPER] WARNING: no object gripped after close.")
    elif args_cli.mechanism == "fixed_joint":
        # bake grasp-relative frames from measured link_6<->cube transform
        cqv = cube_pose()[1]
        q6_inv = quat_inv(l6q)
        rel_pos = quat_apply(q6_inv, (cp - l6p))
        rel_quat = quat_mul(q6_inv, cqv)
        fj_path = "/World/AR4_grasp_fixed_joint"
        fj = UsdPhysics.FixedJoint.Get(stage, Sdf.Path(fj_path))
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_pos]))
        fj.CreateLocalRot0Attr().Set(Gf.Quatf(float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2]), float(rel_quat[3])))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed_joint_enabled_attr.Set(True)
        grasp_state["mechanism"] = "RuntimeFixedJoint"
        for _ in range(30):
            world.step(render=RENDER)
            capture()
        # verify PhysX registered the joint (cube shouldn't fall)
        grasp_state["engaged"] = True
        log("[fixed_joint] enabled runtime fixed joint link_6<->cube")
    else:  # "friction" (the deliverable): NO joint/weld of any kind. The cube
        # is held ONLY by the jaws squeezing its two opposing faces. Settle via
        # drive() so the closing squeeze keeps being re-applied every step.
        grasp_state["engaged"] = True
        grasp_state["mechanism"] = f"PureFriction(squeeze={args_cli.squeeze_mode},{args_cli.squeeze_force}N)"
        drive(grasp_q_use, GRIP_CLOSED, 30, render=True)
        log("[friction] PURE contact-friction grasp -- jaws squeezing cube faces, NO joint/weld")

    report("after engage")
    jaw_geometry("P3b_after_engage")
    contact_report("P3b_after_engage")
    test_gain = cube_pose()[0][2] - cube_rest_z
    log(f"[ENGAGE] cube_z gain after engage = {test_gain*1000:.1f}mm")

    # ---- LIFT + HOLD + RETREAT (jaws keep squeezing throughout) ----------
    # grip_hold is only used by the legacy position path; in friction mode the
    # squeeze (effort or capped-force) persists via drive() regardless.
    grip_hold = GRIP_CLOSED
    # SIDE grasp: the grip is on the cube's two VERTICAL faces, so a genuine
    # friction hold must resist gravity by lifting the cube straight UP (not the
    # near-vertical mode's back-off-to-PREGRASP, which retreats horizontally).
    # LIFT to SIDE_LIFT_Q (+50mm straight up, FK/IK-derived, j5 margin 7.9deg),
    # HOLD there, then RETREAT back down/out via PREGRASP.
    _lift_target = SIDE_LIFT_Q if SIDE_GRASP else (PIN_LIFT_Q if PIN_PICK else PREGRASP_Q)
    cz["P4_LIFT"] = traj(grasp_q_use, _lift_target, grip_hold, 60, "LIFT")
    report("after LIFT")
    jaw_geometry("P4_after_LIFT")
    contact_report("P4_after_LIFT")
    drive(_lift_target, grip_hold, 40, render=True)
    cz["P5_HOLD"] = cube_pose()[0][2]
    _retreat_target = PREGRASP_Q if (SIDE_GRASP or PIN_PICK) else HOME_Q
    cz["P6_RETREAT"] = traj(_lift_target, _retreat_target, grip_hold, 100, "RETREAT")
    report("after RETREAT")
    jaw_geometry("P6_after_RETREAT")
    contact_report("P6_after_RETREAT")

    # re-check gripper status at the end
    if args_cli.mechanism == "surface_gripper" and gripper_view is not None:
        log(f"[SURFACE_GRIPPER final] status={gripper_view.get_surface_gripper_status()} "
            f"gripped={gripper_view.get_gripped_objects()}")

    if writers:
        for wv in writers.values():
            wv.close()

    # ---- verdict ----------------------------------------------------------
    max_z = max(cz.values())
    final_z = cz["P6_RETREAT"]
    gain = max_z - cube_rest_z
    real_lift = gain > 0.01
    held = final_z > cube_rest_z + 0.01
    log("\n" + "=" * 70)
    log(f"SUMMARY mechanism={grasp_state['mechanism']} engaged={grasp_state['engaged']}")
    for k, v in cz.items():
        log(f"  cube_z {k}: {v:.4f}m")
    log(f"cube resting z={cube_rest_z:.4f}m  max z={max_z:.4f}m (gain={gain*1000:.1f}mm)  final z={final_z:.4f}m")
    log(f"real lift(>1cm)={real_lift}  held through retreat={held}")
    log(f"VERDICT: {'PICK CONFIRMED' if (real_lift and held) else 'PICK NOT CONFIRMED'} "
        f"(mechanism={grasp_state['mechanism']})")
    log("=" * 70)
    _R.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

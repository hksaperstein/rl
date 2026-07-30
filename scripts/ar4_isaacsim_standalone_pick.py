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
        [--mechanism surface_gripper|fixed_joint] [--no_video]
"""

import argparse
import math
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Standalone Isaac Sim AR4 genuine physics grasp+lift.")
parser.add_argument("--mechanism", choices=["surface_gripper", "fixed_joint"], default="surface_gripper")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--max_grip_distance", type=float, default=0.10)
parser.add_argument("--cpu", action="store_true", help="run PhysX on CPU (avoids first-time GPU-pipeline init stall)")
parser.add_argument("--stage", choices=["ground", "scene", "robot_noxform", "full"], default="full",
                    help="diagnostic: build scene incrementally to isolate a reset hang")
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema  # noqa: E402
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
CUBE_PATH = "/World/Cube"

PEDESTAL_CENTER_XY = (-0.0262, 0.3660)
PEDESTAL_FOOTPRINT = (0.30, 0.14)
PEDESTAL_HEIGHT = 0.040
CUBE_XY = (-0.04511343308636716, 0.3926929901804897)
CUBE_SIZE = 0.015
CUBE_REST_Z = PEDESTAL_HEIGHT + CUBE_SIZE / 2.0  # 0.0475

GRASP_Q_DEG = [-6.486502296738718, 55.02598117736985, 13.479564958718077,
               0.9246593576759368, 21.857063034307192, 96.18412135786986]
PREGRASP_Q_DEG = [-6.486502296738718, 46.38313360110892, 13.354865607777075,
                  1.3196911944959482, 30.254696468370202, 95.82074708404251]
HOME_Q = [0.0] * 6
GRASP_Q = [math.radians(d) for d in GRASP_Q_DEG]
PREGRASP_Q = [math.radians(d) for d in PREGRASP_Q_DEG]

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIP_JOINTS = ["gripper_jaw1_joint", "gripper_jaw2_joint"]
GRIP_OPEN = 0.014
GRIP_CLOSED = 0.0

ARM_KP, ARM_KD = 4000.0, 200.0
GRIP_KP, GRIP_KD = 10000.0, 200.0

JAW1_LOWER_EXTENT = 0.018475  # fingertip offset below jaw link origin

RESULT_PATH = "/workspace/rl/logs/standalone_pick_result.txt"
VIDEO_DIR = "/workspace/rl/logs/videos/ar4_isaacsim_standalone_pick"
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
        color=np.array([0.3, 0.3, 0.3]),
    )
    _bc("ground_added")

    # dome light
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(2000.0)

    cube = None
    robot = None
    if args_cli.stage in ("scene", "robot_noxform", "full"):
        # pedestal (static collider)
        FixedCuboid(
            prim_path="/World/Pedestal",
            position=np.array([PEDESTAL_CENTER_XY[0], PEDESTAL_CENTER_XY[1], PEDESTAL_HEIGHT / 2.0]),
            scale=np.array([PEDESTAL_FOOTPRINT[0], PEDESTAL_FOOTPRINT[1], PEDESTAL_HEIGHT]),
            color=np.array([0.45, 0.32, 0.22]),
        )
        # cube (dynamic, 15mm)
        cube = DynamicCuboid(
            prim_path=CUBE_PATH,
            position=np.array([CUBE_XY[0], CUBE_XY[1], CUBE_REST_Z]),
            scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            color=np.array([0.8, 0.1, 0.1]),
            mass=0.01,
        )
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
    CAM_A_EYE = [1.0, 1.0, 0.85]      # +X +Y high 3/4
    CAM_B_EYE = [-1.0, 1.0, 0.85]     # -X +Y high 3/4
    CAM_TGT = [-0.03, 0.36, 0.24]

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

    def drive(q_arm, grip_val, steps, render=True):
        do_render = render and RENDER
        tgt = full_targets(q_arm, grip_val)
        for _ in range(steps):
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

    # ---- execute the pick -------------------------------------------------
    drive(HOME_Q, GRIP_OPEN, 40, render=False)
    cube_rest_z = cube_pose()[0][2]
    log(f"[INFO] cube resting z={cube_rest_z:.4f}m")
    if not args_cli.no_video:
        position_cameras()

    cz = {"P0_HOME": cube_rest_z}
    cz["P1_PREGRASP"] = traj(HOME_Q, PREGRASP_Q, GRIP_OPEN, 80, "HOME->PREGRASP")
    if not args_cli.no_video:
        position_cameras()
    cz["P2_GRASP"] = traj(PREGRASP_Q, GRASP_Q, GRIP_OPEN, 60, "PREGRASP->GRASP")
    report("at GRASP (open)")

    # close jaws
    drive(GRASP_Q, GRIP_CLOSED, 40, render=True)
    l6p, l6q = link_world_pose(stage, LINK6_PATH)
    cp = cube_pose()[0]
    log(f"[GEO] link_6 pos={['%.4f'%v for v in l6p]} quat={['%.4f'%v for v in l6q]} cube={['%.4f'%v for v in cp]}")

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
    else:  # fixed_joint
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

    report("after engage")
    test_gain = cube_pose()[0][2] - cube_rest_z
    log(f"[ENGAGE] cube_z gain after engage = {test_gain*1000:.1f}mm")

    # ---- LIFT + HOLD + RETREAT (grip stays closed) -----------------------
    grip_hold = GRIP_CLOSED
    cz["P4_LIFT"] = traj(GRASP_Q, PREGRASP_Q, grip_hold, 60, "LIFT")
    report("after LIFT")
    drive(PREGRASP_Q, grip_hold, 40, render=True)
    cz["P5_HOLD"] = cube_pose()[0][2]
    cz["P6_RETREAT"] = traj(PREGRASP_Q, HOME_Q, grip_hold, 100, "RETREAT")
    report("after RETREAT")

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

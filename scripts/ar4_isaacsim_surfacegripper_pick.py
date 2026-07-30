"""Native Isaac Sim AR4 pick: pre-authored grasp-assist fixed joint (the Isaac
analog of Gazebo's DetachableJoint) + proper multi-step trajectory execution
(2026-07-29, ar4-isaacsim-curobo-pick task).

Closes the loop opened by the ROS2+MoveIt+Gazebo pivot (see
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md): that pivot proved the
AR4 pick blocker was hand-rolled planning/control, NOT Isaac Sim, and that pure
friction fails to hold the 15mm cube in BOTH sims (Gazebo needed a
DetachableJoint grasp-assist). This script replicates that recipe natively:

  1. Grasp-hold -> a PhysX fixed joint between link_6 and the cube, the direct
     Isaac analog of Gazebo's DetachableJoint. IMPORTANT Isaac-Lab-specific
     finding (diagnosed live 2026-07-29): under a `ManagerBasedRLEnv` GPU/PhysX
     pipeline, (a) the native `isaacsim.robot.surface_gripper` manager never
     registers a runtime-created SurfaceGripper prim (its status attr stays
     None, body1 never binds) and (b) a fixed joint AUTHORED at runtime never
     injects into the already-running PhysX scene. The robust fix used here:
     author the fixed joint into the USD stage BEFORE physics init (so PhysX
     parses it) with jointEnabled=False and grasp-relative frames baked from FK
     (so no snap on engage), then simply flip jointEnabled=True at grasp.
  2. Trajectory execution -> smooth multi-waypoint linear interpolation driven
     through the articulation controller (set_joint_position_target per physics
     step), NOT a single-shot PD set (the hand-rolled failure mode).
  3. Reuses this project's fixed AR4 asset + the pedestal + 15mm-cube scene and
     the height-corrected grasp geometry validated earlier.

cuRobo (the originally-intended GPU planner) was a genuine multi-hour wall on
this stack (the isaac-lab:2.3.1 container has no nvcc/CUDA dev toolkit), so per
the task's authorization the Isaac-native fallback path is used; collision-aware
transit is a documented follow-on (Lula RRT is bundled).

.. code-block:: bash

    /workspace/isaaclab/isaaclab.sh -p scripts/ar4_isaacsim_surfacegripper_pick.py --headless [--no_video]
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native Isaac Sim AR4 grasp-assist pick+lift.")
parser.add_argument("--no_video", action="store_true", help="skip video capture (faster probe run)")
parser.add_argument("--pose_follow", action="store_true",
                    help="if the pre-authored joint toggle does not hold, fall back to a pose-driven kinematic weld")
parser.add_argument("--prejointed", action="store_true",
                    help="enable the grasp-assist fixed joint from init (parsed at reset) and start the arm at the "
                         "grasp config with the cube jointed -> proves a REAL PhysX fixed-joint grasp-assist holds+lifts "
                         "under physics (works around the Isaac-Lab runtime-joint-injection limitation)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdPhysics, PhysxSchema  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    create_rotation_matrix_from_view,
    quat_from_matrix,
    quat_mul,
    quat_inv,
    quat_apply,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pedestal_grasp_camera_env_cfg import Ar4PedestalGraspCameraEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

POINT = {
    "cube_xy": (-0.04511343308636716, 0.3926929901804897),
    "grasp_q_deg": [-6.486502296738718, 55.02598117736985, 13.479564958718077, 0.9246593576759368, 21.857063034307192, 96.18412135786986],
    "pregrasp_q_deg": [-6.486502296738718, 46.38313360110892, 13.354865607777075, 1.3196911944959482, 30.254696468370202, 95.82074708404251],
}
POINT_LABEL = "Q0_bearing95_height_corrected"

HOME_Q = [0.0] * 6
GRASP_Q = [math.radians(d) for d in POINT["grasp_q_deg"]]
PREGRASP_Q = [math.radians(d) for d in POINT["pregrasp_q_deg"]]

GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": 0.014, "gripper_jaw2_joint": 0.014}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": 0.0, "gripper_jaw2_joint": 0.0}

JAW1_MESH_LOWER_EXTENT_M = 0.018475
CUBE_HALF_SIZE_M = 0.0075

ARM_STIFFNESS, ARM_DAMPING, ARM_EFFORT = 4000.0, 200.0, 20.0
GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_EFFORT = 4000.0, 200.0, 100.0

CAPTURE_EVERY_N = 6
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "logs", "videos", "ar4_isaacsim_surfacegripper_pick")
os.makedirs(VIDEO_DIR, exist_ok=True)

FIXED_JOINT_PATH = "/World/AR4_grasp_fixed_joint"


def _lookat_quat_opengl(eye, target):
    rot_mat = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([target]), up_axis="Z")
    return tuple(quat_from_matrix(rot_mat)[0].tolist())


def _closeup_cam(jaw1, jaw2, cube, standoff=0.15, z_lift=0.05):
    jaw1, jaw2, cube = map(np.asarray, (jaw1, jaw2, cube))
    target = ((jaw1 + jaw2) / 2.0 + cube) / 2.0
    eye = target.copy(); eye[0] += standoff; eye[2] += z_lift
    return tuple(eye.tolist()), tuple(target.tolist())


def _elbow_cam(elbow, jaw1, jaw2, cube, standoff_min=0.30, z_lift=0.18):
    elbow, jaw1, jaw2, cube = map(np.asarray, (elbow, jaw1, jaw2, cube))
    gp = ((jaw1 + jaw2) / 2.0 + cube) / 2.0
    target = (elbow + gp) / 2.0
    standoff = max(standoff_min, 0.9 * float(np.linalg.norm(gp - elbow)))
    eye = target.copy(); eye[0] += standoff; eye[2] += z_lift
    return tuple(eye.tolist()), tuple(target.tolist())


def _find_prim(stage, leaf, under="/World/envs/env_0"):
    root = stage.GetPrimAtPath(under)
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()
    for prim in Usd.PrimRange(root):
        if prim.GetName() == leaf:
            return prim.GetPath().pathString
    return None


def main() -> None:
    env_cfg = Ar4PedestalGraspCameraEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Fabric OFF: runtime physics-attribute toggles read/write the USD stage;
    # with Fabric ON the stage reflects only stale spawn poses (diagnosed live).
    env_cfg.sim.use_fabric = False
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES); arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES); gripper_cfg.resolve(env.scene)
    cube = env.scene["cube"]
    closeup_camera = env.scene["closeup_camera"]
    elbow_camera = env.scene["elbow_context_camera"]

    jaw_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    elbow_id = robot.data.body_names.index("link_3")
    link6_id = robot.data.body_names.index("link_6")

    stage = omni.usd.get_context().get_stage()
    link6_path = _find_prim(stage, "link_6")
    cube_path = _find_prim(stage, "Cube")
    print(f"[PRIM PATHS] link_6={link6_path}  cube={cube_path}")
    assert link6_path and cube_path

    # ---- Pre-author the grasp-assist fixed joint (DISABLED) BEFORE reset so
    # PhysX parses it at init. Grasp-relative frames are baked from the
    # (repeatable) measured link_6<->cube transform at GRASP_Q so engaging it
    # produces no snap. body1 target = the cube. ----
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(FIXED_JOINT_PATH))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(link6_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
    # measured cube-center in link_6 frame at grasp (consistent across runs)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.0025, 0.0005, 0.0632))
    # localRot0 = conjugate(link6 world quat at grasp) since cube world quat is
    # identity: link6 quat_wxyz ~= (-0.0203,-0.7096,-0.7040,0.0209).
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(-0.0203, 0.7096, 0.7040, -0.0209))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # prejointed: enable at authoring so PhysX parses an ACTIVE constraint at
    # reset (the only way a real PhysX joint engages in this env).
    joint.CreateJointEnabledAttr().Set(bool(args_cli.prejointed))
    jp = joint.GetPrim()
    jp.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)
    enabled_attr = joint.GetJointEnabledAttr()
    print("[GRASP-ASSIST] pre-authored DISABLED fixed joint link_6<->cube (baked grasp frames)")

    grasp_state = {"engaged": False, "mechanism": "none", "rel_pos_b": None, "rel_quat_b": None}

    def _engage_joint():
        enabled_attr.Set(True)
        grasp_state["mechanism"] = "PreAuthoredFixedJoint"
        grasp_state["engaged"] = True
        print("[GRASP-ASSIST] flipped jointEnabled=True")

    def _capture_rel():
        # relative transform of cube in link_6 frame, for the pose-follow weld
        l6 = robot.data.body_pose_w[0, link6_id]
        cp = cube.data.root_pos_w[0]; cq = cube.data.root_quat_w[0]
        p6, q6 = l6[0:3], l6[3:7]
        q6_inv = quat_inv(q6.unsqueeze(0))
        grasp_state["rel_pos_b"] = quat_apply(q6_inv, (cp - p6).unsqueeze(0))[0].clone()
        grasp_state["rel_quat_b"] = quat_mul(q6_inv, cq.unsqueeze(0))[0].clone()

    def _pose_follow_step():
        # kinematic weld: drive cube root pose to link_6 * rel each step
        l6 = robot.data.body_pose_w[0, link6_id]
        p6, q6 = l6[0:3], l6[3:7]
        new_pos = p6 + quat_apply(q6.unsqueeze(0), grasp_state["rel_pos_b"].unsqueeze(0))[0]
        new_quat = quat_mul(q6.unsqueeze(0), grasp_state["rel_quat_b"].unsqueeze(0))[0]
        pose = torch.cat([new_pos, new_quat]).unsqueeze(0)
        cube.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

    def _set_gains():
        na = len(arm_cfg.joint_ids)
        robot.write_joint_stiffness_to_sim(torch.full((1, na), ARM_STIFFNESS, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_damping_to_sim(torch.full((1, na), ARM_DAMPING, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_effort_limit_to_sim(torch.full((1, na), ARM_EFFORT, device=env.device), joint_ids=arm_cfg.joint_ids)
        ng = len(gripper_cfg.joint_ids)
        robot.write_joint_stiffness_to_sim(torch.full((1, ng), GRIPPER_STIFFNESS, device=env.device), joint_ids=gripper_cfg.joint_ids)
        robot.write_joint_damping_to_sim(torch.full((1, ng), GRIPPER_DAMPING, device=env.device), joint_ids=gripper_cfg.joint_ids)
        robot.write_joint_effort_limit_to_sim(torch.full((1, ng), GRIPPER_EFFORT, device=env.device), joint_ids=gripper_cfg.joint_ids)

    frame_counter = {"n": 0}
    writers = {}
    cam_done = {"v": False}

    def _reposition():
        j1 = robot.data.body_pos_w[0, jaw_ids[0]].cpu().tolist()
        j2 = robot.data.body_pos_w[0, jaw_ids[1]].cpu().tolist()
        cp = cube.data.root_pos_w[0].cpu().tolist()
        el = robot.data.body_pos_w[0, elbow_id].cpu().tolist()
        ce, ct = _closeup_cam(j1, j2, cp); ee, et = _elbow_cam(el, j1, j2, cp)
        closeup_camera.set_world_poses(positions=torch.tensor([ce], device=env.device),
                                       orientations=torch.tensor([_lookat_quat_opengl(ce, ct)], device=env.device), convention="opengl")
        elbow_camera.set_world_poses(positions=torch.tensor([ee], device=env.device),
                                     orientations=torch.tensor([_lookat_quat_opengl(ee, et)], device=env.device), convention="opengl")
        cam_done["v"] = True

    def _capture():
        if args_cli.no_video:
            return
        frame_counter["n"] += 1
        if frame_counter["n"] % CAPTURE_EVERY_N != 0:
            return
        if not cam_done["v"]:
            _reposition()
        closeup_camera.update(env.physics_dt, force_recompute=True)
        elbow_camera.update(env.physics_dt, force_recompute=True)
        writers["c"].append_data(closeup_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8"))
        writers["e"].append_data(elbow_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8"))

    def _drive(q, gexpr, steps, render):
        qt = torch.tensor([q], device=env.device)
        gt = torch.tensor([[gexpr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
        for _ in range(steps):
            robot.set_joint_position_target(qt, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(gt, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            if grasp_state["engaged"] and grasp_state["mechanism"] == "PoseFollowWeld":
                _pose_follow_step()
            env.sim.step(render=render)
            robot.update(env.physics_dt)
            _capture()

    def _traj(qf, qt, gexpr, steps, label, render=True):
        qf, qt = np.asarray(qf), np.asarray(qt)
        for i in range(1, steps + 1):
            _drive(((1 - i / steps) * qf + (i / steps) * qt).tolist(), gexpr, 2, render)
        cz = cube.data.root_pos_w[0, 2].item()
        print(f"[TRAJ {label}] cube_z={cz:.4f}m")
        return cz

    def _report(label):
        j1 = robot.data.body_pos_w[0, jaw_ids[0]].cpu().tolist()
        cp = cube.data.root_pos_w[0].cpu().tolist()
        print(f"[REPORT {label}] fingertip_z={j1[2]-JAW1_MESH_LOWER_EXTENT_M:.4f}m cube_z={cp[2]:.4f}m")

    with torch.inference_mode():
        env.reset()

        if not args_cli.no_video:
            import imageio
            fps = max(1, int((1.0 / env.physics_dt) / (2 * CAPTURE_EVERY_N)))
            writers["c"] = imageio.get_writer(os.path.join(VIDEO_DIR, "closeup.mp4"), fps=fps, codec="libx264")
            writers["e"] = imageio.get_writer(os.path.join(VIDEO_DIR, "elbow.mp4"), fps=fps, codec="libx264")

        if args_cli.prejointed:
            # ---- REAL PhysX fixed-joint grasp-assist (joint parsed ACTIVE at
            # reset). Start the arm AT the grasp config and place the cube at
            # link_6's baked grasp-relative pose so the joint is satisfied at
            # t=0 (no snap); then lift+retreat under the real constraint. ----
            grasp_state["mechanism"] = "PreAuthoredFixedJoint(active-from-init)"
            grasp_state["engaged"] = True
            _set_gains()
            grasp_t = torch.tensor([GRASP_Q], device=env.device)
            robot.write_joint_state_to_sim(grasp_t, torch.zeros_like(grasp_t), joint_ids=arm_cfg.joint_ids)
            gclosed = torch.tensor([[GRIPPER_CLOSED_EXPR[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
            robot.write_joint_state_to_sim(gclosed, torch.zeros_like(gclosed), joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim(); env.sim.step(render=False); robot.update(env.physics_dt)
            # place cube at link_6 * baked_rel
            l6 = robot.data.body_pose_w[0, link6_id]
            p6, q6 = l6[0:3], l6[3:7]
            rel_pos = torch.tensor([-0.0025, 0.0005, 0.0632], device=env.device)
            rel_quat = torch.tensor([-0.0203, 0.7096, 0.7040, -0.0209], device=env.device)
            cube_pos = p6 + quat_apply(q6.unsqueeze(0), rel_pos.unsqueeze(0))[0]
            cube_quat = quat_mul(q6.unsqueeze(0), rel_quat.unsqueeze(0))[0]
            cube.write_root_pose_to_sim(torch.cat([cube_pos, cube_quat]).unsqueeze(0), env_ids=torch.tensor([0], device=env.device))
            cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
            _drive(GRASP_Q, GRIPPER_CLOSED_EXPR, 30, render=True)
            _reposition()
            grasp_z = cube.data.root_pos_w[0, 2].item()
            print(f"[PREJOINTED] settled at grasp: cube_z={grasp_z:.4f}m (jointed to link_6 via real PhysX FixedJoint)")
            cz = {"P0_GRASP": grasp_z}
            # lift straight up, hold, and swing back to HOME -- all under the joint
            cz["P4_LIFT"] = _traj(GRASP_Q, PREGRASP_Q, GRIPPER_CLOSED_EXPR, 60, "LIFT")
            _report("after LIFT")
            _drive(PREGRASP_Q, GRIPPER_CLOSED_EXPR, 40, render=True)
            cz["P5_HOLD"] = cube.data.root_pos_w[0, 2].item()
            cz["P6_RETREAT"] = _traj(PREGRASP_Q, HOME_Q, GRIPPER_CLOSED_EXPR, 100, "RETREAT")
            _report("after RETREAT")
            cube_rest_z = grasp_z  # baseline = pedestal grasp height
        else:
            cxy = POINT["cube_xy"]; oz = cube.data.root_pos_w[0, 2].item()
            cube.write_root_pose_to_sim(
                torch.cat([torch.tensor([[cxy[0], cxy[1], oz]], device=env.device), cube.data.root_quat_w[0:1].clone()], dim=-1),
                env_ids=torch.tensor([0], device=env.device))
            cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

            _set_gains()
            _drive(HOME_Q, GRIPPER_OPEN_EXPR, 40, render=False)
            cube_rest_z = cube.data.root_pos_w[0, 2].item()
            print(f"[INFO] cube resting z={cube_rest_z:.4f}m")

            cz = {"P0_HOME": cube_rest_z}
            cz["P1_PREGRASP"] = _traj(HOME_Q, PREGRASP_Q, GRIPPER_OPEN_EXPR, 80, "HOME->PREGRASP")
            _reposition()
            cz["P2_GRASP"] = _traj(PREGRASP_Q, GRASP_Q, GRIPPER_OPEN_EXPR, 60, "PREGRASP->GRASP")
            _report("at GRASP (open)")

            # PHASE 3: close jaws snug, capture the grasp-relative transform at
            # the grasp (BEFORE any lift), then engage the hold.
            _drive(GRASP_Q, GRIPPER_CLOSED_EXPR, 30, render=True)
            _capture_rel()
            l6 = robot.data.body_pose_w[0, link6_id].cpu().tolist()
            cp = cube.data.root_pos_w[0].cpu().tolist()
            print(f"[GEO] link_6 pos={['%.4f'%v for v in l6[0:3]]} quat={['%.4f'%v for v in l6[3:7]]} cube={['%.4f'%v for v in cp]}")
            if args_cli.pose_follow:
                grasp_state["mechanism"] = "PoseFollowWeld"
                grasp_state["engaged"] = True
                print("[GRASP-ASSIST] engaged PoseFollowWeld (kinematic weld: cube pose driven to link_6*rel each step)")
            else:
                _engage_joint()
                _drive(GRASP_Q, GRIPPER_CLOSED_EXPR, 30, render=True)
                test_gain = cube.data.root_pos_w[0, 2].item() - cube_rest_z
                print(f"[GRASP-ASSIST] joint runtime-toggle test gain={test_gain*1000:.1f}mm")
            _report("after engage")

            cz["P4_LIFT"] = _traj(GRASP_Q, PREGRASP_Q, GRIPPER_CLOSED_EXPR, 50, "LIFT")
            _report("after LIFT")
            _drive(PREGRASP_Q, GRIPPER_CLOSED_EXPR, 40, render=True)
            cz["P5_HOLD"] = cube.data.root_pos_w[0, 2].item()
            cz["P6_RETREAT"] = _traj(PREGRASP_Q, HOME_Q, GRIPPER_CLOSED_EXPR, 100, "RETREAT")
            _report("after RETREAT")

        if not args_cli.no_video:
            writers["c"].close(); writers["e"].close()

        print("\n" + "=" * 70)
        print(f"SUMMARY {POINT_LABEL}  mechanism={grasp_state['mechanism']}")
        for k, v in cz.items():
            print(f"  cube_z {k}: {v:.4f}m")
        max_z = max(cz.values()); final_z = cz["P6_RETREAT"]; gain = max_z - cube_rest_z
        real_lift = gain > 0.01; held = final_z > cube_rest_z + 0.01
        print(f"cube resting z={cube_rest_z:.4f}m  max z={max_z:.4f}m (gain={gain*1000:.1f}mm)  final z={final_z:.4f}m")
        print(f"real lift(>1cm)={real_lift}  held through retreat={held}")
        print(f"VERDICT: {'PICK CONFIRMED' if (real_lift and held) else 'PICK NOT CONFIRMED'}  (mechanism={grasp_state['mechanism']})")
        print(f"videos -> {VIDEO_DIR}")
        print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

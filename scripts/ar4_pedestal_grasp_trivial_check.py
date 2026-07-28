# scripts/ar4_pedestal_grasp_trivial_check.py
"""Dead-simple, single-grasp-point diagnostic (2026-07-28,
ar4-grasp-trivial-friction-check task, direct user instruction: "STOP
adding complexity and check the trivial things"). Re-attempts exactly ONE
of the 3 pedestal validation points already used by
scripts/ar4_pedestal_grasp_confirm.py (Q0_bearing95 - all 3 failed
identically in that script's own prior runs, so one repro is sufficient),
with two additions that script deliberately did not have (it was the
"numeric/fast variant only" by design):

1. Real close-up + elbow-inclusive VIDEO/frame capture through the whole
   PHASE0-6 sequence (tasks/ar4/pedestal_grasp_camera_env_cfg.py's
   closeup_camera/elbow_context_camera, repositioned at runtime from a live
   jaw1/jaw2/cube/elbow measurement - reuses
   scripts/grasp_demo_v2.py's own _compute_closeup_camera/
   _compute_elbow_context_camera verbatim, and
   scripts/_record_jaw_fix_open_close_cycle.py's live-measure-then-
   reposition convention), so the CLOSE moment can be directly LOOKED AT
   instead of only inferred from force/position numbers.
2. Explicit numeric checks for the two remaining trivial hypotheses this
   task was dispatched to test in order:
   - Grasp HEIGHT: is the REAL fingertip (jaw1 link origin minus the
     known 18.475mm mesh extent - see scripts/ar4_graspable_workspace.py's
     _JAW1_MESH_LOWER_EXTENT_M) inside the cube's own vertical span at the
     CLOSE moment, or above/below it?
   - Aperture/alignment: does the cube's own XY position sit between the
     two open, descended jaws along their separation axis?

Friction (the first trivial hypothesis) is addressed structurally, not
just measured, per direct user decision made mid-task: tasks/ar4/
objects_cfg.py's CUBE_CFG now carries its own explicit, realistic
wood/plastic/resin-range physics_material (static=0.8/dynamic=0.7,
CUBE_PHYSICS_MATERIAL) instead of only inheriting the scene-wide default -
see that module's own docstring for the full rationale (the scene default
was already mu=1.0, HIGHER than Franka's own working setup's unset 0.5
default, so this was already not the likely culprit, but the user directed
setting an explicit, physically-labeled value on the object itself
regardless). This script's own printed friction line confirms the applied
values match that decision.

No IK search, no workspace re-derivation, no Cartesian correction loop -
reuses the already-solved GRASP_Q/PREGRASP_Q values and the already-
validated settle_to_joint_pose closed-loop primitive exactly as
scripts/ar4_pedestal_grasp_confirm.py does, per this task's explicit
"strip it down" instruction.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ar4_pedestal_grasp_trivial_check.py"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Single-point AR4 pedestal grasp+lift trivial-cause check, with close-up video.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for camera-sensor rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    create_rotation_matrix_from_view,
    matrix_from_quat,
    quat_from_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pedestal_grasp_camera_env_cfg import Ar4PedestalGraspCameraEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402
from tasks.ar4.fk_verification import compute_link_pose_from_joint_values  # noqa: E402
from tasks.ar4.joint_tracking import settle_to_joint_pose  # noqa: E402
from tasks.ar4.objects_cfg import CUBE_PHYSICS_MATERIAL  # noqa: E402

# Copied verbatim from scripts/ar4_pedestal_grasp_confirm.py's
# VALIDATION_POINTS["Q0_bearing95"] - one repro point is sufficient since
# all 3 points there failed IDENTICALLY (see kb doc's 2026-07-28
# "ar4-pedestal-ground-clearance-fix task" UPDATE, Part 3).
POINT = {
    "cube_xy": (-0.03690307221845676, 0.3902115233016254),
    "grasp_q_deg": [-6.486502296738718, 53.23822236377881, 14.982846754337578, -14.92769805402622, 21.951674171245376, 114.65851419966206],
    "pregrasp_q_deg": [-6.486502296738718, 45.26345050158713, 13.541632691134666, -8.35706089858409, 34.462683836708806, 112.67242887258548],
}
POINT_LABEL = "Q0_bearing95"

HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}

EE_OFFSET_LOCAL_LIST = [0.0, 0.0, 0.036]  # matches grasp_demo_v2.py/_EE_OFFSET

# Known jaw1 mesh extent (scripts/ar4_graspable_workspace.py's
# _JAW1_MESH_LOWER_EXTENT_M, 2026-07-24 finding, reused not re-derived):
# jaw1_link's own world origin sits 18.475mm ABOVE the real fingertip.
JAW1_MESH_LOWER_EXTENT_M = 0.018475

CUBE_HALF_SIZE_M = 0.0075  # 15mm cube, tasks/ar4/objects_cfg.py CUBE_CFG

STIFFNESS = 4000.0
DAMPING = 200.0
EFFORT_LIMIT = 20.0

# Render throttle (2026-07-28, added after this script's first cloud run
# took ~29min real time - confirmed via live nvidia-smi/CPU% inspection to
# be genuinely rendering, not hung - because camera.update() was called for
# BOTH cameras on EVERY physics step, only skipping the video WRITE, not
# the render+GPU-readback itself). Only actually render/capture every
# CAPTURE_EVERY_N-th step; forced snapshots always render regardless.
CAPTURE_EVERY_N = 8

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "ar4_pedestal_grasp_trivial_check")
os.makedirs(VIDEO_DIR, exist_ok=True)


def base_to_world(p_base):
    import numpy as np
    out = np.array(p_base, dtype=float).copy()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _quat_wxyz_to_matrix(quat_wxyz):
    import numpy as np
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fk_predicted_pinch_point_world(joint_values_rad: dict):
    pos_b, quat_b = compute_link_pose_from_joint_values(joint_values_rad, "link_6")
    rot_b = _quat_wxyz_to_matrix(quat_b)
    import numpy as np
    pinch_b = pos_b + rot_b @ np.array(EE_OFFSET_LOCAL_LIST)
    return base_to_world(pinch_b)


def _achieved_pinch_world(robot, link6_body_id):
    link6_pose_w = robot.data.body_pose_w[0, link6_body_id]
    link6_pos_w, link6_quat_w = link6_pose_w[0:3], link6_pose_w[3:7]
    rot_w = matrix_from_quat(link6_quat_w.unsqueeze(0))[0]
    offset_t = torch.tensor(EE_OFFSET_LOCAL_LIST, device=link6_pos_w.device)
    return (link6_pos_w + rot_w @ offset_t).tolist()


def _lookat_quat_opengl(eye, target):
    """OpenGL look-at convention (forward=-Z, up=+Y) - same helper as
    scripts/grasp_demo_v2.py / scripts/_record_jaw_fix_open_close_cycle.py."""
    eyes = torch.tensor([eye])
    targets = torch.tensor([target])
    rot_mat = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    return tuple(quat_from_matrix(rot_mat)[0].tolist())


def _compute_closeup_camera(jaw1_pos, jaw2_pos, cube_pos, standoff=0.15, z_lift=0.05):
    """Copied verbatim from scripts/grasp_demo_v2.py's
    _compute_closeup_camera - see that function's own docstring for the
    full geometry rationale (side-profile view along world +X, perpendicular
    to the jaw-slide axis, avoids the documented axial-eye black-frame
    failure mode)."""
    import numpy as np
    jaw1 = np.asarray(jaw1_pos, dtype=float)
    jaw2 = np.asarray(jaw2_pos, dtype=float)
    cube = np.asarray(cube_pos, dtype=float)
    jaw_mid = (jaw1 + jaw2) / 2.0
    target = (jaw_mid + cube) / 2.0
    eye = target.copy()
    eye[0] += standoff
    eye[2] += z_lift
    return tuple(eye.tolist()), tuple(target.tolist())


def _compute_elbow_context_camera(elbow_pos, jaw1_pos, jaw2_pos, cube_pos, standoff_scale=0.9, standoff_min=0.30, z_lift=0.18):
    """Copied verbatim from scripts/grasp_demo_v2.py's
    _compute_elbow_context_camera - see that function's own docstring."""
    import numpy as np
    elbow = np.asarray(elbow_pos, dtype=float)
    jaw1 = np.asarray(jaw1_pos, dtype=float)
    jaw2 = np.asarray(jaw2_pos, dtype=float)
    cube = np.asarray(cube_pos, dtype=float)
    jaw_mid = (jaw1 + jaw2) / 2.0
    gripper_point = (jaw_mid + cube) / 2.0
    target = (elbow + gripper_point) / 2.0
    span = float(np.linalg.norm(gripper_point - elbow))
    standoff = max(standoff_min, standoff_scale * span)
    eye = target.copy()
    eye[0] += standoff
    eye[2] += z_lift
    return tuple(eye.tolist()), tuple(target.tolist())


def _dist_mm(a, b):
    return math.dist(a, b) * 1000.0


def main() -> None:
    env_cfg = Ar4PedestalGraspCameraEnvCfg()
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    contact_sensors = [env.scene["gripper_jaw1_contact"], env.scene["gripper_jaw2_contact"]]
    cube = env.scene["cube"]
    closeup_camera = env.scene["closeup_camera"]
    elbow_camera = env.scene["elbow_context_camera"]

    link6_body_id = robot.find_bodies(["link_6"])[0][0]
    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    elbow_body_id = robot.data.body_names.index("link_3")
    print(f"[body_names] {robot.data.body_names}")

    print("=" * 70)
    print("[FRICTION] cube CUBE_PHYSICS_MATERIAL (tasks/ar4/objects_cfg.py, "
          f"direct user decision this task): static_friction={CUBE_PHYSICS_MATERIAL.static_friction} "
          f"dynamic_friction={CUBE_PHYSICS_MATERIAL.dynamic_friction} "
          f"friction_combine_mode={CUBE_PHYSICS_MATERIAL.friction_combine_mode}")
    print("[FRICTION] gripper/scene-wide default (Ar4PickPlaceGraspGoalEnvCfg.__post_init__, "
          "unchanged, applies to the gripper jaw links since robot_cfg.py sets no per-body "
          "override): static_friction=1.0 dynamic_friction=1.0, combine_mode=average (Isaac Lab default) "
          "-> effective cube-vs-jaw ~0.9 static / 0.85 dynamic")
    print("=" * 70)

    with torch.inference_mode():
        env.reset()

        grasp_q = [math.radians(d) for d in POINT["grasp_q_deg"]]
        pregrasp_q = [math.radians(d) for d in POINT["pregrasp_q_deg"]]
        cube_xy = POINT["cube_xy"]

        joint_values_commanded = {name: grasp_q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
        fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

        override_z = cube.data.root_pos_w[0, 2].item()
        override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
        override_quat = cube.data.root_quat_w[0:1].clone()
        cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

        home_t = torch.tensor([HOME_Q], device=env.device)
        open_t = torch.tensor([[GRIPPER_OPEN_EXPR[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
        for _ in range(30):
            robot.set_joint_position_target(home_t, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(open_t, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(env.physics_dt)
        cube_z_on_pedestal = cube.data.root_pos_w[0, 2].item()
        print(f"[INFO] Cube teleported to: {override_pos[0].tolist()}, settled resting height on pedestal={cube_z_on_pedestal:.4f}m")

        n_arm = len(arm_cfg.joint_ids)
        robot.write_joint_stiffness_to_sim(torch.full((1, n_arm), STIFFNESS, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_damping_to_sim(torch.full((1, n_arm), DAMPING, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_effort_limit_to_sim(torch.full((1, n_arm), EFFORT_LIMIT, device=env.device), joint_ids=arm_cfg.joint_ids)

        fps = max(1, int((1.0 / env.physics_dt) / (4 * CAPTURE_EVERY_N)))
        closeup_writer = imageio.get_writer(os.path.join(VIDEO_DIR, "closeup.mp4"), fps=fps, codec="libx264")
        elbow_writer = imageio.get_writer(os.path.join(VIDEO_DIR, "elbow_context.mp4"), fps=fps, codec="libx264")

        frame_counter = {"n": 0}
        cameras_positioned = {"done": False}

        def _reposition_cameras():
            # Repositioned ONCE, the FIRST time a frame is actually captured
            # (i.e. after PHASE1-PREGRASP-OPEN-SETTLE has already moved the
            # arm near the cube/pedestal - NOT at the HOME reset pose used
            # by an earlier version of this script, which put the arm's own
            # resting jaw position and the far-away cube position on
            # opposite sides of a midpoint+offset computation and landed the
            # closeup eye INSIDE a link's own mesh - confirmed live via a
            # solid-blue-link-fills-the-frame snapshot, 2026-07-28 first
            # cloud run of this exact script). Live-measured, not guessed,
            # matching grasp_demo_v2.py's own "camera computed once after
            # GRASP_Q is actually reached" convention.
            jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
            jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
            cube_pos = cube.data.root_pos_w[0].cpu().tolist()
            elbow_pos = robot.data.body_pos_w[0, elbow_body_id].cpu().tolist()
            closeup_eye, closeup_target = _compute_closeup_camera(jaw1_pos, jaw2_pos, cube_pos)
            elbow_eye, elbow_target = _compute_elbow_context_camera(elbow_pos, jaw1_pos, jaw2_pos, cube_pos)
            closeup_camera.set_world_poses(
                positions=torch.tensor([closeup_eye], device=env.device),
                orientations=torch.tensor([_lookat_quat_opengl(closeup_eye, closeup_target)], device=env.device),
                convention="opengl",
            )
            elbow_camera.set_world_poses(
                positions=torch.tensor([elbow_eye], device=env.device),
                orientations=torch.tensor([_lookat_quat_opengl(elbow_eye, elbow_target)], device=env.device),
                convention="opengl",
            )
            print(f"[CAMERA] closeup eye={closeup_eye} target={closeup_target}")
            print(f"[CAMERA] elbow_context eye={elbow_eye} target={elbow_target}")
            cameras_positioned["done"] = True

        def _capture_frame(force_snapshot_name=None):
            frame_counter["n"] += 1
            # Render throttle (2026-07-28, second cloud run of this script):
            # the first attempt called camera.update() - a real render+GPU
            # readback for BOTH cameras - on EVERY physics step (only
            # skipping the VIDEO WRITE, not the render itself), inflating a
            # ~90-480-step phase into a ~29-minute run. Only actually render
            # every CAPTURE_EVERY_N-th step (always render on a forced
            # snapshot regardless of phase).
            if force_snapshot_name is None and (frame_counter["n"] % CAPTURE_EVERY_N != 0):
                return
            if not cameras_positioned["done"]:
                _reposition_cameras()
            closeup_camera.update(env.physics_dt, force_recompute=True)
            elbow_camera.update(env.physics_dt, force_recompute=True)
            closeup_rgb = closeup_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
            elbow_rgb = elbow_camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
            closeup_writer.append_data(closeup_rgb)
            elbow_writer.append_data(elbow_rgb)
            if force_snapshot_name is not None:
                imageio.imwrite(os.path.join(VIDEO_DIR, f"closeup_{force_snapshot_name}.png"), closeup_rgb)
                imageio.imwrite(os.path.join(VIDEO_DIR, f"elbow_{force_snapshot_name}.png"), elbow_rgb)
                print(f"[SNAPSHOT] saved '{force_snapshot_name}'")

        force_tracker = {"jaw1_max": 0.0, "jaw2_max": 0.0, "open_gripper_max_force": 0.0}

        def _track_forces(is_pre_close: bool):
            for sensor in contact_sensors:
                sensor.update(env.physics_dt, force_recompute=True)
            jaw1_force = contact_sensors[0].data.net_forces_w[0, 0].norm().item()
            jaw2_force = contact_sensors[1].data.net_forces_w[0, 0].norm().item()
            force_tracker["jaw1_max"] = max(force_tracker["jaw1_max"], jaw1_force)
            force_tracker["jaw2_max"] = max(force_tracker["jaw2_max"], jaw2_force)
            if is_pre_close:
                force_tracker["open_gripper_max_force"] = max(force_tracker["open_gripper_max_force"], jaw1_force, jaw2_force)
            return jaw1_force, jaw2_force

        def _height_aperture_report(label):
            jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
            jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
            cube_pos = cube.data.root_pos_w[0].cpu().tolist()
            real_fingertip_z = jaw1_pos[2] - JAW1_MESH_LOWER_EXTENT_M
            cube_top = cube_pos[2] + CUBE_HALF_SIZE_M
            cube_bottom = cube_pos[2] - CUBE_HALF_SIZE_M
            if real_fingertip_z > cube_top:
                height_verdict = f"ABOVE cube top by {(real_fingertip_z - cube_top) * 1000:.2f}mm (clamping top face)"
            elif real_fingertip_z < cube_bottom:
                height_verdict = f"BELOW cube bottom by {(cube_bottom - real_fingertip_z) * 1000:.2f}mm (below the cube entirely)"
            else:
                height_verdict = f"INSIDE cube's vertical span (correct side-grip height)"
            jaw_sep = math.dist(jaw1_pos, jaw2_pos)
            jaw_mid_xy = [(jaw1_pos[i] + jaw2_pos[i]) / 2.0 for i in range(2)]
            cube_offset_from_jaw_mid_xy = math.dist(jaw_mid_xy, cube_pos[:2])
            print(
                f"[HEIGHT/APERTURE @ {label}] jaw1_z={jaw1_pos[2]:.4f}m real_fingertip_z={real_fingertip_z:.4f}m "
                f"cube_z={cube_pos[2]:.4f}m cube_span=[{cube_bottom:.4f},{cube_top:.4f}]m -> {height_verdict}"
            )
            print(
                f"[HEIGHT/APERTURE @ {label}] jaw_separation={jaw_sep * 1000:.2f}mm "
                f"jaw_mid_xy={['%.4f' % v for v in jaw_mid_xy]} cube_xy={['%.4f' % v for v in cube_pos[:2]]} "
                f"cube_offset_from_jaw_midline_xy={cube_offset_from_jaw_mid_xy * 1000:.2f}mm "
                f"(cube half-size={CUBE_HALF_SIZE_M * 1000:.1f}mm - offset should be well under this for the cube "
                "to sit BETWEEN the jaws, not beside/outside them)"
            )

        def _settle_tracked(desired_q, gripper_expr, label, snapshot_name=None, capture=True):
            def on_step(outer, i):
                _track_forces(is_pre_close=True)
                if capture:
                    _capture_frame()

            gripper_target = [gripper_expr[n] for n in GRIPPER_JOINT_NAMES]
            result = settle_to_joint_pose(
                env, robot, arm_cfg.joint_ids, desired_q,
                tol_rad=math.radians(0.15), max_outer_iters=8, inner_settle_steps=150,
                integral_gain=1.0, integral_clamp=0.5,
                gripper_joint_ids=gripper_cfg.joint_ids, gripper_target=gripper_target,
                render=True, on_step=on_step, label=label,
            )
            if snapshot_name is not None and capture:
                _capture_frame(force_snapshot_name=snapshot_name)
            return result

        def _drive_naive(target_q, gripper_expr, duration, label, snapshot_name=None):
            target_t = torch.tensor([target_q], device=env.device)
            g_t = torch.tensor([[gripper_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
            for i in range(duration):
                robot.set_joint_position_target(target_t, joint_ids=arm_cfg.joint_ids)
                robot.set_joint_position_target(g_t, joint_ids=gripper_cfg.joint_ids)
                robot.write_data_to_sim()
                env.sim.step(render=True)
                robot.update(env.physics_dt)
                j1, j2 = _track_forces(is_pre_close=False)
                _capture_frame()
                if i % 20 == 0 or i == duration - 1:
                    cube_z = cube.data.root_pos_w[0, 2].item()
                    print(f"  [{label} step {i:3d}] cube_z={cube_z:.4f}m jaw1={j1:.3f}N jaw2={j2:.3f}N")
            if snapshot_name is not None:
                _capture_frame(force_snapshot_name=snapshot_name)
            return cube.data.root_pos_w[0, 2].item()

        cube_z_by_phase = {}
        cube_z_by_phase["PHASE0-HOME-OPEN"] = cube.data.root_pos_w[0, 2].item()
        # No camera capture here - the arm is still at HOME, far from the
        # cube/pedestal, and the closeup/elbow cameras haven't been
        # positioned yet (see _reposition_cameras's own docstring for why
        # positioning from a HOME-pose measurement was the exact bug found
        # in this script's first cloud run).

        pregrasp_result = _settle_tracked(pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN-SETTLE", capture=False)
        cube_z_by_phase["PHASE1-PREGRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
        corrected_pregrasp_target = [d + c for d, c in zip(pregrasp_q, pregrasp_result["correction"])]
        print(f"[INFO] PREGRASP settle: converged={pregrasp_result['converged']} iters={pregrasp_result['n_outer_iters']} max_err_deg={pregrasp_result['max_err_deg']:.4f}")

        # The arm is now genuinely near the cube/pedestal (PREGRASP settled)
        # - position the cameras from this live state, then take the
        # PHASE1 snapshot for reference before continuing to PHASE2.
        _reposition_cameras()
        _capture_frame(force_snapshot_name="phase1_pregrasp_open")

        grasp_result = _settle_tracked(grasp_q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN-SETTLE", snapshot_name="phase2_grasp_open_descended")
        cube_z_by_phase["PHASE2-GRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
        corrected_grasp_target = [d + c for d, c in zip(grasp_q, grasp_result["correction"])]
        print(f"[INFO] GRASP settle: converged={grasp_result['converged']} iters={grasp_result['n_outer_iters']} max_err_deg={grasp_result['max_err_deg']:.4f}")
        print(f"[INFO] open_gripper_max_force so far (PHASE0-2, pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")
        _height_aperture_report("end of PHASE2-GRASP-OPEN-SETTLE (open, descended, pre-CLOSE)")

        achieved_pinch_w = _achieved_pinch_world(robot, link6_body_id)
        pinch_disc_mm = _dist_mm(achieved_pinch_w, fk_pred_pinch_w)
        print(f"[INFO] GRASP pinch-point discrepancy vs FK prediction: {pinch_disc_mm:.3f}mm")

        cube_z_by_phase["PHASE3-GRASP-CLOSE"] = _drive_naive(corrected_grasp_target, GRIPPER_CLOSED_EXPR, 90, "PHASE3-GRASP-CLOSE", snapshot_name="phase3_close")
        _height_aperture_report("end of PHASE3-GRASP-CLOSE")
        cube_z_by_phase["PHASE4-LIFT-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE4-LIFT-CLOSE", snapshot_name="phase4_lift")
        cube_z_by_phase["PHASE5-HOLD-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE5-HOLD-CLOSE", snapshot_name="phase5_hold")
        cube_z_by_phase["PHASE6-RETREAT-CLOSE"] = _drive_naive(HOME_Q, GRIPPER_CLOSED_EXPR, 150, "PHASE6-RETREAT-CLOSE", snapshot_name="phase6_retreat")

        closeup_writer.close()
        elbow_writer.close()

        print("\n" + "=" * 70)
        print(f"SUMMARY: {POINT_LABEL}")
        print("=" * 70)
        for label, z in cube_z_by_phase.items():
            print(f"  cube_z at end of {label}: {z:.4f}m")
        height_gain = max(cube_z_by_phase.values()) - cube_z_on_pedestal
        final_z = cube_z_by_phase["PHASE6-RETREAT-CLOSE"]
        held_through_retreat = final_z > cube_z_on_pedestal + 0.01
        real_lift = height_gain > 0.01
        both_jaws_contacted = force_tracker["jaw1_max"] > 0.001 and force_tracker["jaw2_max"] > 0.001
        open_gripper_clean = force_tracker["open_gripper_max_force"] < 1.0
        print(f"Cube resting height (on pedestal): {cube_z_on_pedestal:.4f}m")
        print(f"Max cube height reached: {max(cube_z_by_phase.values()):.4f}m (gain={height_gain * 1000:.2f}mm)")
        print(f"Final cube height (end of RETREAT): {final_z:.4f}m")
        print(f"Max jaw-cube contact force WHILE GRIPPER STILL OPEN (pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")
        print(f"Max jaw1/jaw2 contact force (any phase): {force_tracker['jaw1_max']:.4f}N / {force_tracker['jaw2_max']:.4f}N")
        print(f"BOTH jaws registered real contact force (post-close): {both_jaws_contacted}")
        print(f"Real height gain (>1cm): {real_lift}")
        print(f"Held through retreat (>1cm above pedestal-resting height): {held_through_retreat}")
        print(f"Open-gripper collision-free (<1N while nominally open): {open_gripper_clean}")
        verdict = (
            "GRASP+LIFT CONFIRMED" if (open_gripper_clean and both_jaws_contacted and real_lift and held_through_retreat)
            else "GRASP+LIFT NOT CONFIRMED"
        )
        print(f"VERDICT [{POINT_LABEL}]: {verdict}")
        print(f"Videos/frames written to: {VIDEO_DIR}")
        print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

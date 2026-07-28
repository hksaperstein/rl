"""Height-corrected grasp+lift attempt (2026-07-28, ar4-pedestal-grasp-
height-fix task, direct continuation of scripts/ar4_pedestal_grasp_trivial_check.py).

That script visually and numerically confirmed the AR4 capstone grasp's real
cause: the REAL fingertip (jaw1's own world z minus the known 18.475mm mesh
extent) settled at 0.0548m against a cube vertical span of [0.0400, 0.0550]m
- just 0.2mm below the cube's own TOP FACE, not its vertical center
(0.0475m). The jaws close on the cube's top edge (jamming, not a genuine
side-pinch), so `jaw_separation` never actually decreases and the cube is
left behind the instant the arm retreats.

This script re-derives a LOWERED grasp joint configuration and re-attempts
the identical PHASE0-6 grasp+lift+retreat sequence, close-up + elbow-inclusive
video included, to see whether a corrected height produces a real lift.

**Height correction methodology** (Pi-local, pure-FK search, no GPU/Isaac -
see scripts/ar4_graspable_workspace.py / scripts/_ar4_pedestal_select_grasp_points.py
for the established sweep/search machinery this reuses, not re-derives):

- The prior trivial-check run measured a real, reproducible PHYSICS-VS-
  KINEMATICS tracking bias at this exact validation point: the FK-commanded
  design target (`GRASP_AT_HEIGHT` = PEDESTAL_HEIGHT_M + 0.0105 = 0.0505m)
  produced a PHYSICALLY-achieved real fingertip of 0.0548m after
  `settle_to_joint_pose` converged - i.e. achieved = commanded + ~4.3mm
  (the arm settles ~4.3mm short of its own commanded descent, consistent
  with this investigation's already-documented "joint-tracking-closed-loop"
  and "Cartesian-fingertip-correction" findings of a several-mm physics
  tracking residual - see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
  matching UPDATEs).
- To get the PHYSICALLY-achieved fingertip to the cube's own vertical center
  (PEDESTAL_HEIGHT_M + CUBE_HALF_SIZE_M = 0.040 + 0.0075 = 0.0475m), the
  FK-commanded design target must be lowered by that same ~4.3mm bias:
  NEW_GRASP_HEIGHT_FK_TARGET = 0.0475 - 0.0043 = 0.0432m.
- A local Gaussian-perturbation search (identical method to
  `_ar4_pedestal_select_grasp_points.py`'s own `search_pregrasp`, just
  targeting this new height instead of a hover height) around the EXISTING
  Q0_bearing95 GRASP_Q seed, joint_1 held fixed (same approach azimuth),
  found a new GRASP config with FK fingertip_z=0.0418m (within the
  established 2mm tolerance of the 0.0432m target) and excellent margins
  (tilt=0.50deg, roll_offset=0.56deg, min_margin=34.97deg - all comfortably
  better than the original point). A matching PREGRASP hover config
  (target height = new GRASP height + 0.05m) was found the same way. Both
  reuse the SAME joint_1/approach azimuth and land within ~1cm (xy) of the
  original Q0_bearing95 cube position - well inside the existing pedestal's
  own (0.30m x 0.14m) footprint, no pedestal resize needed.
- Even in the worst case (zero physics-tracking bias at this specific new
  config), the FK-kinematic fingertip (0.0418m) still sits 1.8mm ABOVE the
  pedestal's own top surface (0.040m) - no collision risk even without any
  compensating bias.

If the corrected height still doesn't produce a real lift, the per-phase
height/aperture diagnostics and close-up video below will show exactly
what the achieved fingertip height/jaw state look like at CLOSE, same as
the trivial-check script's own diagnostic method - not a re-litigation of
the whole workspace/FK machinery, just re-running the same lightweight
per-phase checks at the corrected height.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ar4_pedestal_grasp_height_corrected_check.py"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Height-corrected single-point AR4 pedestal grasp+lift check, with close-up video.")
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

# Height-corrected GRASP/PREGRASP config (2026-07-28, ar4-pedestal-grasp-
# height-fix task) - see module docstring above for full derivation. Same
# approach azimuth (joint_1) as the original Q0_bearing95 point this task
# continues from, but joints 2-6 (and therefore cube_xy, ~8.6mm from the
# original point) re-derived via a local FK search targeting a LOWERED
# height that compensates for the measured ~4.3mm physics-tracking bias.
POINT = {
    "cube_xy": (-0.04511343308636716, 0.3926929901804897),
    "grasp_q_deg": [-6.486502296738718, 55.02598117736985, 13.479564958718077, 0.9246593576759368, 21.857063034307192, 96.18412135786986],
    "pregrasp_q_deg": [-6.486502296738718, 46.38313360110892, 13.354865607777075, 1.3196911944959482, 30.254696468370202, 95.82074708404251],
}
POINT_LABEL = "Q0_bearing95_height_corrected"

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

# GRIPPER actuator gain boost (2026-07-28, ar4-pedestal-grasp-height-fix task,
# added after this script's own FIRST cloud run at the corrected height: real
# fingertip landed genuinely INSIDE the cube's vertical span (a real
# improvement over the old top-face-jamming bug), but jaw_separation still
# stayed frozen at 28.07mm through the entire CLOSE phase, with 48-79N of
# jaw-cube contact force already present BEFORE the intentional close command
# - the SAME magnitude range documented at EVERY OTHER validation point ever
# tested in this whole investigation (48.29N, 66.65N, 79.17N - see
# kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching
# UPDATEs). tasks/ar4/robot_cfg.py's gripper ImplicitActuatorCfg has ALWAYS
# used effort_limit_sim=20.0 (Newtons, since gripper_jaw[12]_joint are
# prismatic) unboosted in every grasp script in this investigation - genuinely
# less than the observed 48-79N contact resistance range, so the position
# controller is force-capped and CANNOT physically push the jaw further once
# real resistance appears, independent of the commanded target - this is the
# same "implicit actuator PD droops under real load" mechanism this repo
# already diagnosed and fixed for the ARM (STIFFNESS/DAMPING/EFFORT_LIMIT
# above), just never applied to the GRIPPER before (every prior close-success
# precedent, e.g. scripts/_record_jaw_fix_open_close_cycle.py, only ever
# demonstrated closing with NO object in the way - zero resistance - so this
# gap was never stress-tested). Boosted here by direct analogy, not a new
# mechanism: same fix, same actuator-cfg pattern, applied to the other set of
# joints on the same robot.
GRIPPER_STIFFNESS = 4000.0
GRIPPER_DAMPING = 200.0
GRIPPER_EFFORT_LIMIT = 100.0  # comfortably above the observed 48-79N range

# Render throttle (see ar4_pedestal_grasp_trivial_check.py's own comment for
# the full story - camera.update() for both cameras on every physics step
# inflates run time ~15x even though the render itself is real work, not a
# hang). Only actually render/capture every CAPTURE_EVERY_N-th step; forced
# snapshots always render regardless.
CAPTURE_EVERY_N = 8

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "ar4_pedestal_grasp_height_corrected_check")
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
    print("[FRICTION] cube CUBE_PHYSICS_MATERIAL (tasks/ar4/objects_cfg.py): "
          f"static_friction={CUBE_PHYSICS_MATERIAL.static_friction} "
          f"dynamic_friction={CUBE_PHYSICS_MATERIAL.dynamic_friction} "
          f"friction_combine_mode={CUBE_PHYSICS_MATERIAL.friction_combine_mode}")
    print("[HEIGHT FIX] this run uses the height-corrected GRASP/PREGRASP config derived "
          "in this script's own module docstring (FK target 0.0432m, compensating for a "
          "measured ~4.3mm physics-tracking bias, targeting real fingertip ~0.0475m = cube center).")
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

        # GRIPPER gain boost (2026-07-28, see GRIPPER_EFFORT_LIMIT's own
        # comment above for full rationale) - same write pattern as the arm
        # above, applied to gripper_cfg.joint_ids instead.
        n_gripper = len(gripper_cfg.joint_ids)
        robot.write_joint_stiffness_to_sim(torch.full((1, n_gripper), GRIPPER_STIFFNESS, device=env.device), joint_ids=gripper_cfg.joint_ids)
        robot.write_joint_damping_to_sim(torch.full((1, n_gripper), GRIPPER_DAMPING, device=env.device), joint_ids=gripper_cfg.joint_ids)
        robot.write_joint_effort_limit_to_sim(torch.full((1, n_gripper), GRIPPER_EFFORT_LIMIT, device=env.device), joint_ids=gripper_cfg.joint_ids)
        print(f"[INFO] Gripper actuator gains boosted: stiffness={GRIPPER_STIFFNESS} damping={GRIPPER_DAMPING} effort_limit={GRIPPER_EFFORT_LIMIT}N (was 1000/50/20N, tasks/ar4/robot_cfg.py default)")

        fps = max(1, int((1.0 / env.physics_dt) / (4 * CAPTURE_EVERY_N)))
        closeup_writer = imageio.get_writer(os.path.join(VIDEO_DIR, "closeup.mp4"), fps=fps, codec="libx264")
        elbow_writer = imageio.get_writer(os.path.join(VIDEO_DIR, "elbow_context.mp4"), fps=fps, codec="libx264")

        frame_counter = {"n": 0}
        cameras_positioned = {"done": False}

        def _reposition_cameras():
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
            cube_center = cube_pos[2]
            if real_fingertip_z > cube_top:
                height_verdict = f"ABOVE cube top by {(real_fingertip_z - cube_top) * 1000:.2f}mm (clamping top face)"
            elif real_fingertip_z < cube_bottom:
                height_verdict = f"BELOW cube bottom by {(cube_bottom - real_fingertip_z) * 1000:.2f}mm (below the cube entirely)"
            else:
                dist_from_center_mm = (real_fingertip_z - cube_center) * 1000
                height_verdict = f"INSIDE cube's vertical span (correct side-grip height), {dist_from_center_mm:+.2f}mm from center"
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

        pregrasp_result = _settle_tracked(pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN-SETTLE", capture=False)
        cube_z_by_phase["PHASE1-PREGRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
        corrected_pregrasp_target = [d + c for d, c in zip(pregrasp_q, pregrasp_result["correction"])]
        print(f"[INFO] PREGRASP settle: converged={pregrasp_result['converged']} iters={pregrasp_result['n_outer_iters']} max_err_deg={pregrasp_result['max_err_deg']:.4f}")

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

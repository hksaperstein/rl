"""Live grasp+lift confirmation at a cube position CHOSEN BY the FK-based
graspable-workspace sweep (scripts/ar4_graspable_workspace.py), 2026-07-27
ar4-graspable-workspace-from-fk task.

Background: this project's multi-week AR4 grasp investigation
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md) repeatedly found a
genuine multi-joint reachability shortfall trying to reach a vertical grasp
pose AT THE CUBE'S DEFAULT POSITION (world (0.0, 0.275, ...), joint_3 running
out of comfortable range) - every fix attempt tuned the IK/solver/approach
instead of questioning whether that position was reachable at all.
scripts/ar4_graspable_workspace.py inverted this: forward-sampled the arm's
own 6-joint configuration space (pure FK, zero solver risk) to find where a
genuinely graspable pinch pose (near-vertical approach, correct height for
the 15mm cube, comfortable margin on every joint, AND - 2026-07-27 addition,
see ROLL_TOL_DEG in that script - a jaw-slide-axis heading that actually
straddles a face of the cube instead of colliding with it while open)
actually exists, and recommended a specific interior point: world (x, y) =
(-0.0238, 0.3436), radius 0.3445m, bearing ~94 deg (near the scene's
existing "straight ahead" bearing=90 convention) - see that script's own run
output / the kb doc's matching update for the full characterization and the
visualization PNG.

This script is the live confirmation: places the cube at that EXACT
recommended point and drives the arm DIRECTLY to the two FK-precomputed
joint configurations below (GRASP_Q, PREGRASP_Q - both produced by the same
forward-sampling method, not solved via IK at runtime) through the normal
PD-actuator control loop (env.step, holding each phase's target constant -
this project's own established best practice, see grasp_demo_v2.py's PHASES
loop comment: a ramped interpolation was tried and found WORSE than just
holding the fixed target for the whole phase). No IK/DLS solver is used
anywhere in this script - the whole point is that a joint configuration this
project already knows reaches the target pose (by construction, via FK) can
just be commanded directly, sidestepping this investigation's entire
solver-local-minima history.

Checks the same real physical evidence this project's verification standard
requires (not a shaped metric or a single eyeballed video frame): contact
force on BOTH jaws (env.scene["gripper_jaw{1,2}_contact"], filtered to the
Cube prim only - a nonzero reading is unambiguously cube contact), real
cube height gain during lift (env.scene["cube"].data.root_pos_w), and
whether the grasp survives close+lift+hold+retreat.

Reuses (does not reinvent):
  - tasks/ar4/grasp_verify_env_cfg.py's Ar4GraspVerifyEnvCfg (contact
    sensors + cube + demo_camera/perception_camera, already built for
    exactly this kind of verification).
  - The test-local arm actuator stiffness/damping boost (40/4 -> 4000/200)
    this investigation established is necessary for the arm to actually
    track a commanded multi-joint pose under gravity (see
    kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-22
    "later, same day" UPDATE) - NOT applied to the shared tasks/ar4/robot_cfg.py.
  - grasp_demo_v2.py's own PHASES-loop/contact-force-logging conventions
    (constant-target-per-phase, force_matrix_w reads every ~20 steps).

Cloud runs headless (this repo's own standing exception to "never headless
locally" - see docs/cloud/dispatch-checklist.md). Copy the AppLauncher
boilerplate ordering below verbatim if extending this script - AppLauncher
must be constructed before any other isaaclab import.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_graspable_workspace_confirm.py --headless"
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live grasp+lift confirmation at an FK-chosen graspable workspace point.")
parser.add_argument(
    "--cube-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the recommended cube world (x,y) - defaults to the FK sweep's own recommended point.",
)
parser.add_argument(
    "--video-suffix", type=str, default="",
    help="Suffix appended to output video filenames, so multiple runs (e.g. testing several graspable-region points) don't overwrite each other.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# This script uses the scene's cameras (demo_camera/perception_camera) for
# video recording - Isaac Sim refuses to spawn any camera at all without
# this flag (RuntimeError: "A camera was spawned without the
# --enable_cameras flag"). Matches grasp_demo_v2.py's own established
# pattern (set programmatically here rather than requiring an extra CLI
# flag on every invocation).
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402


def _lookat_quat_opengl(eye, target):
    """OpenGL look-at convention (forward=-Z, up=+Y) - duplicated verbatim
    from scripts/grasp_demo_v2.py's own helper of the same name (that
    script has its own AppLauncher/Isaac-Sim import-time side effects, so
    it isn't importable here - same reasoning scripts/ar4_graspable_workspace.py
    already documents for its own small constant duplications)."""
    import numpy as np

    eyes = torch.tensor([eye])
    targets = torch.tensor([target])
    rot_mat = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    return tuple(quat_from_matrix(rot_mat)[0].tolist())


def _compute_elbow_context_camera(elbow_pos, jaw1_pos, jaw2_pos, cube_pos, standoff_scale=0.9, standoff_min=0.30, z_lift=0.18):
    """Duplicated verbatim from scripts/grasp_demo_v2.py's own function of
    the same name (2026-07-24, ar4-axis-align-ik task) - given LIVE-MEASURED
    (never guessed) link_3 (elbow)/jaw1/jaw2/cube world positions, returns
    (eye, target) for a WIDE camera keeping elbow, forearm, wrist, and
    gripper/cube all visible together. Reused here (2026-07-27,
    ar4-graspable-roll-constraint task) per this task's own explicit
    requirement for an "elbow-inclusive camera framing so the whole grasp
    is visible" - this repo's standing named camera for exactly that."""
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
    return tuple(eye.tolist()), tuple(target.tolist()), span, standoff

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_video_suffix = f"_{args_cli.video_suffix}" if args_cli.video_suffix else ""
VIDEO_PATH = os.path.join(LOG_DIR, "videos", f"ar4_graspable_workspace_confirm{_video_suffix}.mp4")

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# 2026-07-27 UPDATE (ar4-graspable-roll-constraint task): superseded by the
# ROLL-CONSTRAINED sweep - scripts/ar4_graspable_workspace.py's Stage 1
# filter now also constrains the gripper jaw-slide axis's world-frame
# heading (ROLL_TOL_DEG=12deg from parallel to world X or Y), after the
# original (roll-unconstrained) point below was live-confirmed to collide
# with the cube at 52-61N even gripper-OPEN (kb/wiki/concepts/
# ar4-vs-franka-root-cause-comparison.md's 2026-07-27 "ar4-graspable-
# workspace-from-fk task" UPDATE). Same Stage-A survivor (identical
# joint_2-6), re-swept over joint_1 with roll filtered: closest-to-bearing-90
# joint_1 that also satisfies roll landed at joint_1=-4.32deg, bearing=94.0deg,
# world (x,y)=(-0.0238, 0.3436) - same radius (0.3445m) and min joint margin
# (27.68deg) as before, PLUS jaw-slide-axis roll/heading offset 10.08deg
# (within the new 12deg tolerance).
RECOMMENDED_CUBE_XY = (-0.023809635667085667, 0.3436444906384511)

HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# GRASP_Q: the exact FK-forward-sampled joint configuration whose pinch
# point (link_6 + _EE_OFFSET, matching grasp_demo_v2.py's own convention)
# lands at RECOMMENDED_CUBE_XY at GRASP_AT_HEIGHT=0.0105m, tilt=2.91deg from
# vertical, jaw-slide-axis roll/heading offset=10.08deg (2026-07-27 addition).
# Degrees, converted to radians below.
GRASP_Q_DEG = [-4.32433486449247, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994]

# PREGRASP_Q: a SEPARATE FK-forward-sampled config (same method - local
# Gaussian perturbation search around GRASP_Q's own joint_2-6 values, joint_1
# held fixed at GRASP_Q's value since azimuth shouldn't change for a pure
# vertical-hover waypoint, NOW also filtered by the same roll criterion -
# scripts/_ar4_pregrasp_search_roll_constrained.py) at the SAME (x,y) (within
# 1.7mm) but hover height = GRASP_AT_HEIGHT + 0.05m = 0.0605m, tilt=0.91deg,
# roll offset=3.98deg, min joint margin=24.12deg. This is a genuinely
# different, independently-found valid FK config, not an interpolation/
# guess - both waypoints came from the same "sample forward, keep what
# satisfies the filters" method this whole task is built around.
PREGRASP_Q_DEG = [-4.32433486449247, 52.434178877076384, 27.87529831816333, -2.2493188573178595, 8.860705727687604, 100.52702739544871]

GRASP_Q = [math.radians(d) for d in GRASP_Q_DEG]
PREGRASP_Q = [math.radians(d) for d in PREGRASP_Q_DEG]


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Test-local actuator-gain boost (NOT touching the shared
    # tasks/ar4/robot_cfg.py) - established necessary by this whole
    # investigation (kb doc, 2026-07-22 UPDATEs) for the arm to actually
    # track a commanded multi-joint pose under gravity rather than sagging.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    num_arm_joints = len(ARM_JOINT_NAMES)

    gripper_joint_ids, gripper_joint_names_found = robot.find_joints(GRIPPER_JOINT_NAMES)
    print(f"[INFO] Gripper joint ids resolved: {gripper_joint_names_found} -> {gripper_joint_ids}")

    # 2026-07-27 (ar4-graspable-roll-constraint task): resolve link_3
    # (elbow) + gripper jaw body ids for the elbow-context camera below -
    # same bodies scripts/grasp_demo_v2.py's own --elbow-camera resolves.
    elbow_body_ids, elbow_body_names_found = robot.find_bodies(["link_3"])
    print(f"[INFO] Elbow (link_3) body id resolved: {elbow_body_names_found} -> {elbow_body_ids}")
    gripper_jaw_body_ids, gripper_jaw_body_names_found = robot.find_bodies(["gripper_jaw1_link", "gripper_jaw2_link"])
    print(f"[INFO] Gripper jaw body ids resolved: {gripper_jaw_body_names_found} -> {gripper_jaw_body_ids}")

    def _print_gripper_state(label: str) -> None:
        live_gripper_q = robot.data.joint_pos[0, gripper_joint_ids].tolist()
        print(f"  [GRIPPER-CHECK] {label}: actual joint_pos {gripper_joint_names_found}={['%.5f' % v for v in live_gripper_q]}")

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["perception_camera"]
    demo_video_path = VIDEO_PATH.replace(".mp4", "_demo_camera.mp4")
    demo_video_writer = imageio.get_writer(demo_video_path, fps=int(1.0 / env.step_dt), codec="libx264")
    demo_camera = env.scene["demo_camera"]
    # 2026-07-27 (ar4-graspable-roll-constraint task): THIRD video output -
    # elbow-inclusive framing (this task's own explicit requirement: "so
    # the whole grasp is visible"), reusing grasp_verify_env_cfg.py's
    # already-defined elbow_context_camera (placeholder pose, repositioned
    # below from a LIVE measurement once the arm settles at GRASP_Q - same
    # pattern grasp_demo_v2.py's own --elbow-camera uses, per the
    # isaac-sim-video-capture skill's "measure the real subject, don't
    # guess coordinates" rule).
    elbow_video_path = VIDEO_PATH.replace(".mp4", "_elbow_context.mp4")
    elbow_video_writer = imageio.get_writer(elbow_video_path, fps=int(1.0 / env.step_dt), codec="libx264")
    elbow_camera = env.scene["elbow_context_camera"]
    elbow_camera_positioned = False

    with torch.inference_mode():
        env.reset()

        cube = env.scene["cube"]
        cube_xy = args_cli.cube_xy if args_cli.cube_xy is not None else RECOMMENDED_CUBE_XY
        override_z = cube.data.root_pos_w[0, 2].item()
        override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
        override_quat = cube.data.root_quat_w[0:1].clone()
        cube.write_root_pose_to_sim(
            torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
        )
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube teleported to FK-recommended world position: {override_pos[0].tolist()}")

        cube_z_on_table = cube.data.root_pos_w[0, 2].item()
        print(f"[INFO] Cube resting height (table): {cube_z_on_table:.4f}m")
        print(f"[INFO] GRASP_Q (deg): {GRASP_Q_DEG}")
        print(f"[INFO] PREGRASP_Q (deg): {PREGRASP_Q_DEG}")

        PHASES = [
            (60, HOME_Q, GRIPPER_OPEN),
            (150, PREGRASP_Q, GRIPPER_OPEN),
            (90, GRASP_Q, GRIPPER_OPEN),
            (90, GRASP_Q, GRIPPER_CLOSE),
            (120, PREGRASP_Q, GRIPPER_CLOSE),
            (120, PREGRASP_Q, GRIPPER_CLOSE),  # explicit HOLD phase, unchanged target - tests whether the grasp slips over time
            (150, HOME_Q, GRIPPER_CLOSE),  # RETREAT while still holding the cube
        ]

        cube_z_history = []
        jaw_force_history = []

        print("\n[INFO] Starting phased execution (direct joint-target control, no IK)...\n")
        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            start_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(
                f"[PHASE {phase_idx}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} "
                f"target_q={['%.4f' % x for x in target_q]} start_q={['%.4f' % x for x in start_q]}"
            )
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

                # Reposition the elbow-context camera ONCE, right after the
                # arm has settled at GRASP_Q (end of PHASE 2, the last step
                # before CLOSE is commanded) - a live measurement of the
                # ACTUAL settled pose, not a pre-computed guess, per the
                # isaac-sim-video-capture skill. From this point on
                # (CLOSE/LIFT/HOLD/RETREAT - the actual grasp being
                # confirmed), every frame is recorded to elbow_video_writer.
                if phase_idx == 2 and i == duration - 1:
                    elbow_pos_w_for_cam = robot.data.body_pos_w[0, elbow_body_ids].cpu().tolist()[0]
                    jaw_pos_w_for_cam = robot.data.body_pos_w[0, gripper_jaw_body_ids].cpu().tolist()
                    cube_pos_for_cam = cube.data.root_pos_w[0].cpu().tolist()
                    elbow_eye, elbow_target, elbow_span, elbow_standoff = _compute_elbow_context_camera(
                        elbow_pos_w_for_cam, jaw_pos_w_for_cam[0], jaw_pos_w_for_cam[1], cube_pos_for_cam,
                    )
                    print(
                        f"[ELBOW-CONTEXT CAMERA] elbow_world(link_3)={['%.5f' % v for v in elbow_pos_w_for_cam]} "
                        f"jaw1_world={['%.5f' % v for v in jaw_pos_w_for_cam[0]]} "
                        f"jaw2_world={['%.5f' % v for v in jaw_pos_w_for_cam[1]]} "
                        f"cube_world={['%.5f' % v for v in cube_pos_for_cam]} elbow_to_gripper_span={elbow_span:.4f}m "
                        f"standoff={elbow_standoff:.4f}m -> EYE={['%.5f' % v for v in elbow_eye]} TARGET={['%.5f' % v for v in elbow_target]}"
                    )
                    elbow_quat = _lookat_quat_opengl(elbow_eye, elbow_target)
                    elbow_eye_t = torch.tensor([elbow_eye], device=env.device)
                    elbow_quat_t = torch.tensor([elbow_quat], device=env.device)
                    elbow_camera.set_world_poses(positions=elbow_eye_t, orientations=elbow_quat_t, convention="opengl")
                    elbow_camera_positioned = True
                    # NOTE: do NOT append a frame this exact step - the
                    # camera's cached RGB output was rendered BEFORE this
                    # reposition took effect (set_world_poses only updates
                    # the buffer; it doesn't force a synchronous re-render),
                    # so reading it now would write one stale/placeholder
                    # frame. Start appending from the NEXT step onward.
                elif elbow_camera_positioned:
                    elbow_rgb = elbow_camera.data.output["rgb"][0].cpu().numpy()
                    elbow_video_writer.append_data(elbow_rgb[:, :, :3].astype("uint8"))

                if i == duration // 2:
                    _print_gripper_state(f"PHASE {phase_idx} MIDPOINT (step {i})")

                if i % 20 == 0 or i == duration - 1:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy_now = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    jaw1_force = torch.linalg.vector_norm(
                        env.scene["gripper_jaw1_contact"].data.force_matrix_w.view(1, 3)[0]
                    ).item()
                    jaw2_force = torch.linalg.vector_norm(
                        env.scene["gripper_jaw2_contact"].data.force_matrix_w.view(1, 3)[0]
                    ).item()
                    live_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                    max_track_err = max(abs(a - b) for a, b in zip(live_q, target_q))
                    cube_z_history.append((phase_idx, i, cube_z))
                    jaw_force_history.append((phase_idx, i, jaw1_force, jaw2_force))
                    print(
                        f"  [PHASE {phase_idx} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy_now]} "
                        f"jaw1_cube_force={jaw1_force:.4f}N jaw2_cube_force={jaw2_force:.4f}N "
                        f"max_joint_track_err={max_track_err:.4f}rad"
                    )

        video_writer.close()
        demo_video_writer.close()
        elbow_video_writer.close()
        print(f"[INFO] elbow-context video written to: {elbow_video_path} (camera positioned: {elbow_camera_positioned})")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        max_cube_z = max(z for _, _, z in cube_z_history)
        height_gain = max_cube_z - cube_z_on_table
        max_jaw1_force = max(j1 for _, _, j1, j2 in jaw_force_history)
        max_jaw2_force = max(j2 for _, _, j1, j2 in jaw_force_history)
        # Final cube height at the very end of the RETREAT phase (phase 6) -
        # is the cube still elevated/carried, or did it fall back to the
        # table at some point during CLOSE/LIFT/HOLD/RETREAT?
        final_cube_z = cube_z_history[-1][2]
        print(f"Cube resting height (table): {cube_z_on_table:.4f}m")
        print(f"Max cube height reached during sequence: {max_cube_z:.4f}m (gain={height_gain*1000:.2f}mm)")
        print(f"Final cube height (end of RETREAT, phase 6): {final_cube_z:.4f}m")
        print(f"Max jaw1-cube contact force observed: {max_jaw1_force:.4f}N")
        print(f"Max jaw2-cube contact force observed: {max_jaw2_force:.4f}N")
        both_jaws_contacted = max_jaw1_force > 0.001 and max_jaw2_force > 0.001
        real_lift = height_gain > 0.01  # >1cm real gain, well above noise
        held_through_retreat = final_cube_z > cube_z_on_table + 0.01
        print(f"BOTH jaws registered real contact force: {both_jaws_contacted}")
        print(f"Real height gain (>1cm) at some point: {real_lift}")
        print(f"Cube still elevated at end of RETREAT (>1cm above table): {held_through_retreat}")
        verdict = "GRASP+LIFT CONFIRMED" if (both_jaws_contacted and real_lift and held_through_retreat) else "GRASP+LIFT NOT CONFIRMED"
        print(f"VERDICT: {verdict}")
        print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

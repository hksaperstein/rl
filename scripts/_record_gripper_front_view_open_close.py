# scripts/_record_gripper_front_view_open_close.py
"""FRONT-VIEW AR4 gripper OPEN -> CLOSE -> OPEN recorder (2026-08-04,
ar4-gripper-front-view task, direct user request).

Context: prior open/close-cycle recordings
(scripts/_record_jaw_fix_open_close_cycle.py and its revisions) were all
shot with the eye ALONG the world-Y forearm axis (from the base side, or
from beyond the gripper aimed past it at the elbow). The user reviewed
those and said "that's the wrong view, use the front view." This script
delivers a genuine FRONT view of the gripper's jaws.

Geometry (verified analytically against tasks/ar4/fk_verification.py's
vendor-URDF joint table AND against this project's own live-measured jaw
world positions - the two agree to <1mm):
  - The two jaws slide open/closed along world-X (jaw1 at +x, jaw2 at -x,
    28mm apart when open). So the open/close GAP is a world-X separation.
  - The forearm (elbow link_3 -> gripper) lies along world-Y at Z~0.475 at
    the all-zero reset pose. ANY camera looking along Y therefore looks
    straight down the forearm (this is exactly why every prior view read as
    "elbow-to-wrist"/axial, and why a base-side level eye landed INSIDE the
    forearm and rendered black frames - see that script's own comments).
  - A camera on the +Y (open-workspace / "front") side of the gripper,
    at the gripper's own height, looking back along -Y at the jaw midpoint,
    sees the world-X gap at FULL lateral width (the gap is perpendicular to
    the sight-line) with the jaws appearing left/right and nothing in front
    of them to occlude - the true head-on front view the task asks for. The
    eye sits in open space (Y beyond the fingertips), so it cannot land
    inside geometry -> no black frames.

The eye/target are COMPUTED AT RUNTIME from the live-measured jaw1/jaw2
world positions (never hardcoded/guessed), per the isaac-sim-video-capture
skill. A small upward lift gives a slightly-above-level look so the gap
reads clearly whether the fingers project forward or downward.

Run (local, non-headless per repo convention):
    DISPLAY=:1 flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_record_gripper_front_view_open_close.py"
Run (cloud container, headless - no display on a cloud instance):
    /workspace/isaaclab/isaaclab.sh -p scripts/_record_gripper_front_view_open_close.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record AR4 gripper OPEN->CLOSE->OPEN from a FRONT view.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_gripper_open_close_front_view.mp4")

# --- Front-view camera geometry (see module docstring) --------------------
# "Front" = the +Y open-workspace side of the gripper. The eye sits that far
# beyond the jaw midpoint along +Y, at the jaws' own height plus a small
# lift, looking back along -Y at the midpoint. All three numbers were chosen
# to (a) frame the jaws large and clear, (b) keep the eye well outside any
# solid geometry, (c) show the world-X gap at full lateral width.
_EYE_STANDOFF_FRONT_M = 0.20   # eye distance in front (+Y) of the jaw midpoint
_EYE_Z_LIFT_M = 0.06           # small upward lift for a slightly-above-level look
_FOCAL_LENGTH_MM = 60.0        # frames the full gripper with margin; 28mm gap clearly resolved (~20% of frame width), low crop risk
_CLIPPING = (0.02, 3.0)

# Placeholders - overwritten at runtime from the live jaw measurement before
# any frame is recorded (kept here only so the CameraCfg has something to
# construct from).
_EYE = (0.0, 0.56, 0.53)
_TARGET = (0.0, 0.364, 0.475)


def _compute_front_camera(jaw_mid_pos):
    """Eye on the +Y front side of the gripper, at jaw height + small lift,
    aimed at the jaw midpoint. X kept equal to the midpoint's X so the two
    jaws stay left/right-symmetric in frame."""
    mx, my, mz = jaw_mid_pos
    eye = (mx, my + _EYE_STANDOFF_FRONT_M, mz + _EYE_Z_LIFT_M)
    target = (mx, my, mz)
    return eye, target


def _lookat_quat_opengl(eye, target):
    eyes = torch.tensor([eye])
    targets = torch.tensor([target])
    R = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    return tuple(quat_from_matrix(R)[0].tolist())


def main() -> None:
    global _EYE, _TARGET
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0
    # Brighten the scene so the dark gripper reads clearly, not as a black
    # blob (task requirement). Dome boosted from the scene default (2000).
    env_cfg.scene.light.spawn.intensity = 3200.0
    # Repurpose the existing demo_camera as the front-view camera (same
    # CameraCfg mechanism the prior recorder used - reposition/re-zoom only).
    env_cfg.scene.demo_camera.spawn.focal_length = _FOCAL_LENGTH_MM
    env_cfg.scene.demo_camera.spawn.clipping_range = _CLIPPING
    env_cfg.scene.demo_camera.offset = CameraCfg.OffsetCfg(
        pos=_EYE, rot=_lookat_quat_opengl(_EYE, _TARGET), convention="opengl"
    )

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    demo_camera = env.scene["demo_camera"]

    fps = int(1.0 / env.step_dt)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=fps, codec="libx264")

    with torch.inference_mode():
        env.reset()
        arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].clone().tolist()

        print(f"[body_names] {robot.data.body_names}")
        jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
        jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
        jaw_mid = [(a + b) / 2.0 for a, b in zip(jaw1_pos, jaw2_pos)]
        print(
            f"[live measurement] jaw1_world={['%.5f' % v for v in jaw1_pos]} "
            f"jaw2_world={['%.5f' % v for v in jaw2_pos]} jaw_mid_world={['%.5f' % v for v in jaw_mid]}"
        )

        _EYE, _TARGET = _compute_front_camera(jaw_mid)
        print(f"[front camera] EYE={['%.5f' % v for v in _EYE]} TARGET={['%.5f' % v for v in _TARGET]}")

        # Front fill light near the eye, aimed at the gripper, so the jaws
        # are lit from the camera side (not silhouetted against the arm).
        # Defensive: a light-spawn API hiccup must not kill the whole render
        # (the boosted dome above already guarantees a lit, non-black scene).
        try:
            fill_cfg = sim_utils.SphereLightCfg(intensity=25000.0, radius=0.08, color=(1.0, 1.0, 1.0))
            fill_cfg.func("/World/FrontFillLight", fill_cfg, translation=tuple(_EYE))
            print("[fill light] spawned front sphere light at eye")
        except Exception as exc:  # noqa: BLE001
            print(f"[fill light] WARNING: could not spawn front fill light ({exc}); relying on boosted dome")

        eye_t = torch.tensor([_EYE], device=env.device)
        quat_t = torch.tensor([_lookat_quat_opengl(_EYE, _TARGET)], device=env.device)
        demo_camera.set_world_poses(positions=eye_t, orientations=quat_t, convention="opengl")

        def _measure(label):
            j1 = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
            j2 = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
            actual_q = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
            sep = sum((a - b) ** 2 for a, b in zip(j1, j2)) ** 0.5
            print(
                f"[{label}] actual_joint_pos={['%.5f' % v for v in actual_q]} "
                f"jaw1_world={['%.5f' % v for v in j1]} jaw2_world={['%.5f' % v for v in j2]} "
                f"separation_dist={sep:.5f}m"
            )
            return sep

        def _hold(jaw_target_dict, seconds, label):
            steps = int(seconds * fps)
            jaw1 = jaw_target_dict["gripper_jaw1_joint"]
            jaw2 = jaw_target_dict["gripper_jaw2_joint"]
            for i in range(steps):
                robot.set_joint_position_target(
                    torch.tensor([[jaw1, jaw2]], device=env.device), joint_ids=gripper_cfg.joint_ids
                )
                robot.set_joint_position_target(
                    torch.tensor([arm_hold_target], device=env.device), joint_ids=arm_cfg.joint_ids
                )
                robot.write_data_to_sim()
                env.sim.step(render=True)
                robot.update(env.physics_dt)
                demo_camera.update(env.physics_dt)
                rgb = demo_camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))
                if i == steps - 1:
                    _measure(label)

        print("=" * 100)
        _measure("reset (initial)")
        print("Phase 1: OPEN")
        _hold(GRIPPER_OPEN_COMMAND_EXPR, seconds=3.0, label="end of Phase 1 (OPEN)")
        print("Phase 2: CLOSE")
        _hold(GRIPPER_CLOSED_COMMAND_EXPR, seconds=3.0, label="end of Phase 2 (CLOSE)")
        print("Phase 3: OPEN again")
        _hold(GRIPPER_OPEN_COMMAND_EXPR, seconds=3.0, label="end of Phase 3 (OPEN again)")
        print("=" * 100)

    video_writer.close()
    print(f"Video recorded to: {VIDEO_PATH}")
    env.close()


if __name__ == "__main__":
    main()

# scripts/_record_jaw_bug_open_close_cycle.py
"""One-off diagnostic video capture (2026-07-23, record-jaw-bug-video task):
records the AR4 gripper's OPEN -> CLOSE -> OPEN cycle through the actual
production GRIPPER_OPEN_COMMAND_EXPR / GRIPPER_CLOSED_COMMAND_EXPR constants
in tasks/ar4/robot_cfg.py, so the CORRECTED jaw dynamics (real pincer
separation, not the collapsed-onto-one-point bug) can be directly observed
on video and numerically confirmed.

Context: the jaw-collapse bug (commanding "open" made both jaws converge to
the IDENTICAL world point, ~0.00001m separation) was found live by
scripts/_sweep_jaw2_symmetry.py and root-caused to a double-negation in the
jaw2 open/close command convention (see tasks/ar4/robot_cfg.py's 2026-07-23
UPDATE comment and kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md).
The fix (same-signed command for both jaws, plus re-derived jaw2 hard
limits in scripts/build_asset.py, asset rebuilt this session) has already
been applied BEFORE this script runs - this script's job is purely to
directly measure + record the result, not to apply or verify the fix's
logic itself.

Uses the exact same live open/close command mechanism as
scripts/_sweep_jaw2_symmetry.py (direct set_joint_position_target writes,
holding the arm fixed at its default reset pose - no arm motion, only the
gripper's own two jaw joints are commanded), but drives the jaws through
GRIPPER_OPEN_COMMAND_EXPR / GRIPPER_CLOSED_COMMAND_EXPR exactly as authored
(not swept/widened), and records a tight close-up camera on the gripper
(repurposing Ar4GraspVerifyEnvCfg's existing demo_camera mechanism,
repositioned/re-zoomed for this close range via
isaaclab.utils.math.create_rotation_matrix_from_view + quat_from_matrix,
per this repo's own established convention - see scripts/_compute_democam_quat.py -
rather than building a new camera system from scratch).

Run:
    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_record_jaw_bug_open_close_cycle.py"
"""
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(
    description="Record AR4 gripper OPEN->CLOSE->OPEN cycle, confirming the corrected (unfixed->fixed) jaw dynamics."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

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
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_gripper_jaw_open_close_cycle_fixed_zoomedout.mp4")

# Close-up camera target/eye anchored on the gripper's own resting world
# position at the default reset pose, (~0.014, 0.364, 0.470) - measured
# directly by scripts/_sweep_jaw2_symmetry.py's own printed jaw1_world/
# base_pos output at this identical reset configuration. Eye placed at
# lower Y (in front of the gripper) and slightly higher Z, looking down/
# back at the jaws so the world-X jaw separation reads as left-right
# motion in frame (jaw1/jaw2 differ only in X per that same sweep data;
# Y and Z stayed fixed across the whole swept range).
#
# TRIED AND REVERTED (same session): a second attempt leveled the eye to
# the jaws' own Z height (0, 0.20, 0.475) with a tighter focal_length=70,
# aiming for a less-occluded frontal view of the fingertips specifically -
# this instead rendered solid black/near-black frames end to end (almost
# certainly the eye landing inside/behind solid gripper geometry at that
# closer, level position - not investigated further given limited
# remaining desktop time). Reverted back to the known-good values below
# (confirmed via extracted, non-black video frames), even though the jaw
# fingertips were smaller/more foreshortened in frame than that failed
# attempt was aiming for.
#
# ZOOMED-OUT REVISION (same day, follow-up task): the known-good close-up
# above was delivered to the user but judged "too close/zoomed-in,
# weird-looking." This revision starts from that SAME known-good eye/
# target line-of-sight (not the reverted, black-frame-producing leveled
# attempt above) and pulls the eye back along that same line (larger Y/Z
# offset from target, preserving the elevated down/back look angle that
# avoided the black-frame failure) while also widening the FOV
# (focal_length 50 -> 30), so the full gripper plus some surrounding
# wrist/arm context is visible with margin, not cropped.
_EYE = (0.0, 0.05, 0.55)
_TARGET = (0.0, 0.364, 0.47)


def _lookat_quat_opengl(eye, target):
    eyes = torch.tensor([eye])
    targets = torch.tensor([target])
    R = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    return tuple(quat_from_matrix(R)[0].tolist())


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0
    # Repurpose the existing demo_camera as a tight close-up on the gripper
    # jaws (reposition/re-zoom only - same CameraCfg class/mechanism
    # grasp_demo_v2.py already uses for its own demo_camera recording).
    env_cfg.scene.demo_camera.spawn.focal_length = 30.0
    env_cfg.scene.demo_camera.spawn.clipping_range = (0.05, 2.0)
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
    num_arm_joints = len(ARM_JOINT_NAMES)
    demo_camera = env.scene["demo_camera"]

    fps = int(1.0 / env.step_dt)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=fps, codec="libx264")

    with torch.inference_mode():
        env.reset()
        arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].clone().tolist()

        def _measure(label):
            jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
            jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
            actual_q = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
            sep = [a - b for a, b in zip(jaw1_pos, jaw2_pos)]
            sep_dist = sum(v * v for v in sep) ** 0.5
            print(
                f"[{label}] actual_joint_pos={['%.5f' % v for v in actual_q]} "
                f"jaw1_world={['%.5f' % v for v in jaw1_pos]} jaw2_world={['%.5f' % v for v in jaw2_pos]} "
                f"separation_dist={sep_dist:.5f}m"
            )
            return sep_dist

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
        print("Phase 1: OPEN (corrected GRIPPER_OPEN_COMMAND_EXPR)")
        _hold(GRIPPER_OPEN_COMMAND_EXPR, seconds=3.0, label="end of Phase 1 (OPEN)")
        print("Phase 2: CLOSE (corrected GRIPPER_CLOSED_COMMAND_EXPR)")
        _hold(GRIPPER_CLOSED_COMMAND_EXPR, seconds=3.0, label="end of Phase 2 (CLOSE)")
        print("Phase 3: OPEN again")
        _hold(GRIPPER_OPEN_COMMAND_EXPR, seconds=3.0, label="end of Phase 3 (OPEN again)")
        print("=" * 100)

    video_writer.close()
    print(f"Video recorded to: {VIDEO_PATH}")
    env.close()


if __name__ == "__main__":
    main()

# scripts/_record_jaw_fix_open_close_cycle.py
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
    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_record_jaw_fix_open_close_cycle.py"

REVERSED-VIEW REVISION (2026-07-23, reversed-camera-view task) was run on a
fresh GCP cloud instance (desktop unreachable that day), headless, per
docs/cloud/dispatch-checklist.md:
    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 python scripts/_record_jaw_fix_open_close_cycle.py --headless"
(inside the cloud instance's isaac-venv, cwd ~/rl). See the _EYE/_TARGET
comment block below for what changed and why.
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
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_gripper_jaw_open_close_cycle_fixed_reversed_view.mp4")

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
# _EYE = (0.0, 0.05, 0.55)  # superseded by the REVERSED-VIEW REVISION below
# _TARGET = (0.0, 0.364, 0.47)  # superseded by the REVERSED-VIEW REVISION below
#
# REVERSED-VIEW REVISION (2026-07-23, reversed-camera-view task, direct user
# request): the ZOOMED-OUT REVISION above was watched and described as
# "oriented from the elbow down to the wrist" - _EYE sat at low Y (near the
# elbow/base side) looking toward _TARGET at the gripper's own Y (the
# wrist side). This revision reverses that viewing direction: camera now
# sits on the wrist/gripper side, looking back across the gripper toward
# the elbow/upper-arm.
#
# The elbow link's world position was measured EMPIRICALLY, not guessed:
# link_3 is the child link of joint_3, the AR4's third arm joint - the
# conventional "elbow" joint of a 6-DOF serial arm (joint_1=base yaw,
# joint_2=shoulder, joint_3=elbow, joint_4/5/6=wrist roll/pitch/yaw),
# cross-checked against tasks/ar4/fk_verification.py's own vendor-URDF
# joint table (joint_2: link_1->link_2 "shoulder", joint_3: link_2->link_3
# "elbow"). main() below reads robot.data.body_pos_w for "link_3" live,
# right after the identical env.reset() this script has always used (all
# ARM_JOINT_NAMES at their init_state default of 0.0 - unchanged from every
# prior revision of this script), and prints it before computing the
# camera pose from it - see the printed "[elbow measurement]" line in this
# revision's own recorded run log for the exact live numbers.
#
# _EYE/_TARGET are computed at runtime (via _compute_reversed_camera below)
# directly from that live elbow measurement plus the live-measured gripper
# jaw1/jaw2 midpoint (the same body_pos_w read the prior revisions already
# used), rather than hardcoded, so this revision cannot silently drift from
# whatever the live asset/reset config actually produces:
#   - _TARGET = the measured elbow (link_3) world position directly -
#     "looking toward the elbow/upper arm" per the task's own framing.
#   - _EYE = placed on the FAR side of the gripper from the elbow (i.e.
#     beyond the gripper along the elbow->gripper axis, the OPPOSITE side
#     from the zoomed-out revision's eye), then pulled back an additional
#     standoff further along that same line specifically so the gripper
#     isn't cropped/too-close/occluding the frame - the same "keep the
#     look direction, increase eye-to-subject distance" logic the
#     zoomed-out revision itself already used, just with eye and target on
#     reversed sides of the gripper.
_ELBOW_LINK_NAME = "link_3"
# TRIED AND REVISED (same task, first live cloud attempt): a first attempt at
# 0.55m standoff (eye-to-elbow distance ~0.86m) produced real, non-black,
# correctly-reversed frames, but with the whole arm rendered tiny and the
# gripper barely distinguishable - confirmed by inspecting extracted frames,
# not just asserted. Root cause, confirmed via the live measurement itself:
# at this reset pose, link_3 (elbow) and the gripper jaw midpoint are exactly
# colinear along world-Y (both at world-Z 0.475, world-X ~0.000 - only Y
# differs), so ANY camera with eye/target both on that same axis necessarily
# looks straight down the forearm - the same axial framing the ZOOMED-OUT
# REVISION's own eye/target line already used in the opposite direction
# (confirmed by inspecting ITS frames too: an extreme close-in, gripper-
# filling-the-frame shot, not a clean side profile). That revision's
# eye-to-gripper distance was ~0.32m, not ~0.86m - reduced the standoff here
# to bring this revision's eye-to-gripper distance back to that same
# established, already-proven-non-black/non-tiny scale.
_EYE_STANDOFF_PAST_GRIPPER_M = 0.25  # distance the eye sits beyond the gripper, along the elbow->gripper axis
_EYE_Z_LIFT_M = 0.12  # small upward offset for a slightly-elevated look-back angle


def _compute_reversed_camera(elbow_pos, gripper_mid_pos):
    """Given live-measured elbow (link_3) and gripper-jaw-midpoint world
    positions, return (eye, target) for a camera on the far side of the
    gripper from the elbow, looking back across the gripper toward the
    elbow, pulled back far enough that the gripper isn't cropped."""
    import numpy as np

    elbow = np.asarray(elbow_pos, dtype=float)
    gripper = np.asarray(gripper_mid_pos, dtype=float)
    direction = gripper - elbow
    dist = float(np.linalg.norm(direction))
    d_hat = direction / dist
    eye = gripper + d_hat * _EYE_STANDOFF_PAST_GRIPPER_M
    eye[2] += _EYE_Z_LIFT_M
    return tuple(eye.tolist()), tuple(elbow.tolist())


# Placeholder values only - overwritten at runtime in main() by
# _compute_reversed_camera() using the live-measured elbow/gripper
# positions above, before any frame is recorded. Kept here (rather than
# leaving _EYE/_TARGET undefined at module scope) purely so the
# CameraCfg.OffsetCfg construction below has something to start from.
_EYE = (0.0, 0.9, 0.6)
_TARGET = (0.0, 0.1, 0.47)


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
    elbow_body_id = robot.data.body_names.index(_ELBOW_LINK_NAME)
    num_arm_joints = len(ARM_JOINT_NAMES)
    demo_camera = env.scene["demo_camera"]

    fps = int(1.0 / env.step_dt)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=fps, codec="libx264")

    with torch.inference_mode():
        env.reset()
        arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].clone().tolist()

        # Empirical elbow measurement (task requirement: don't guess the
        # elbow's world position). Printed body_names list is the ground
        # truth for which link names actually exist on the live asset -
        # cross-check _ELBOW_LINK_NAME ("link_3") against it directly.
        print(f"[body_names] {robot.data.body_names}")
        elbow_pos = robot.data.body_pos_w[0, elbow_body_id].cpu().tolist()
        jaw1_pos_reset = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
        jaw2_pos_reset = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
        gripper_mid_reset = [(a + b) / 2.0 for a, b in zip(jaw1_pos_reset, jaw2_pos_reset)]
        print(
            f"[elbow measurement] {_ELBOW_LINK_NAME}_world={['%.5f' % v for v in elbow_pos]} "
            f"jaw1_world={['%.5f' % v for v in jaw1_pos_reset]} jaw2_world={['%.5f' % v for v in jaw2_pos_reset]} "
            f"gripper_mid_world={['%.5f' % v for v in gripper_mid_reset]}"
        )

        # Compute the reversed-view camera pose from the live measurement
        # above (see the REVERSED-VIEW REVISION comment block for the
        # geometry reasoning) and reposition the already-spawned demo_camera
        # at runtime, before any frame is recorded - CameraCfg.OffsetCfg's
        # construction-time values above are only a placeholder.
        _EYE, _TARGET = _compute_reversed_camera(elbow_pos, gripper_mid_reset)
        print(f"[reversed camera] EYE={['%.5f' % v for v in _EYE]} TARGET={['%.5f' % v for v in _TARGET]}")
        eye_t = torch.tensor([_EYE], device=env.device)
        quat_t = torch.tensor([_lookat_quat_opengl(_EYE, _TARGET)], device=env.device)
        demo_camera.set_world_poses(positions=eye_t, orientations=quat_t, convention="opengl")

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

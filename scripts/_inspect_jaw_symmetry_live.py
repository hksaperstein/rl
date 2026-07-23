# scripts/_inspect_jaw_symmetry_live.py
"""One-off live diagnostic (coordinator-directed, 2026-07-23, following a
direct user visual observation that the two gripper jaws look shifted to one
side rather than symmetric about a shared centerline, even though the
commanded JOINT VALUES are confirmed correct: +0.014/-0.014).

The static USD inspection (_inspect_jaw_symmetry.py) already found both
joints' own localPos0 (origin, in the shared gripper_base_link parent frame)
IDENTICAL to float precision - (0, -0.036, 0) for both - and both jaw links'
REST-POSE (joint value 0) world translations identical to ~1e-7m. That rules
out a baked-in asymmetric ORIGIN offset in the authored asset. This script
is the live-dynamics complement: drive the gripper to its real commanded
OPEN state under actual PD actuation (not just read the static rest pose),
then read each jaw LINK's actual world-frame XYZ position directly and check
whether they are genuinely symmetric about their own midpoint - i.e. whether
correct joint VALUES actually produce a symmetric physical result once
combined with the asset's real actuator dynamics, not just at t=0.

Reuses the same env/settle pattern as scripts/_verify_gripper_mirror_fix.py
(arm actuator gains boosted test-locally so the arm base doesn't sag under
gravity and inject spurious motion into the gripper links during settling).

Run: flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_inspect_jaw_symmetry_live.py"
"""
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Live check: are the two gripper jaws symmetric in world space at OPEN?")
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Same test-local gain boost as _verify_gripper_mirror_fix.py / grasp_demo_v2.py
    # - isolates the gripper's own dynamics from the arm base sagging under gravity.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    base_body_id = robot.data.body_names.index("gripper_base_link")
    num_arm_joints = len(ARM_JOINT_NAMES)

    os.makedirs(os.path.join(LOG_DIR, "videos"), exist_ok=True)
    demo_video_path = os.path.join(LOG_DIR, "videos", "ar4_jaw_symmetry_check_demo_camera.mp4")
    demo_video_writer = imageio.get_writer(demo_video_path, fps=int(1.0 / env.step_dt), codec="libx264")
    demo_camera = env.scene["demo_camera"]

    with torch.inference_mode():
        env.reset()
        arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].clone().tolist()

        for i in range(120):
            # This env's ActionsCfg (Ar4GraspVerifyEnvCfg) is: 6 arm joint
            # position targets + 1 scalar gripper open/close command (>=0 ->
            # open per BinaryJointPositionAction) - same action-tensor shape
            # grasp_demo_v2.py's _settle_at already uses, reused here rather
            # than the raw sim.step()/write_data_to_sim() pattern
            # _verify_gripper_mirror_fix.py used (that script drives a
            # ManagerBasedRLEnv directly; this one uses the plain
            # ManagerBasedEnv from grasp_verify_env_cfg.py, so the normal
            # env.step(action) action-manager path is the simpler match).
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = arm_hold_target[j]
            action[:, num_arm_joints] = 1.0  # GRIPPER_OPEN
            env.step(action)
            demo_rgb = demo_camera.data.output["rgb"][0].cpu().numpy()
            demo_video_writer.append_data(demo_rgb[:, :, :3].astype("uint8"))
            if i % 20 == 0 or i == 119:
                jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
                jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
                base_pos = robot.data.body_pos_w[0, base_body_id].cpu().tolist()
                midpoint = [(a + b) / 2 for a, b in zip(jaw1_pos, jaw2_pos)]
                sep_vec = [a - b for a, b in zip(jaw1_pos, jaw2_pos)]
                jaw_q = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
                print(
                    f"[step {i:3d}] jaw_joint_pos={['%.5f' % v for v in jaw_q]} "
                    f"jaw1_link_world={['%.5f' % v for v in jaw1_pos]} "
                    f"jaw2_link_world={['%.5f' % v for v in jaw2_pos]} "
                    f"midpoint={['%.5f' % v for v in midpoint]} "
                    f"gripper_base_world={['%.5f' % v for v in base_pos]} "
                    f"jaw1-jaw2_vec={['%.5f' % v for v in sep_vec]}"
                )

        # Final settled readout with explicit symmetry verdict.
        jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
        jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
        base_pos = robot.data.body_pos_w[0, base_body_id].cpu().tolist()
        midpoint = [(a + b) / 2 for a, b in zip(jaw1_pos, jaw2_pos)]
        # base_pos is gripper_base_link's ORIGIN, not the joint's own local
        # offset point (0,-0.036,0 in gripper_base_link's local frame, from
        # the static USD inspection) - report the raw offset so it can be
        # cross-checked against that number directly rather than assumed.
        offset_base_to_midpoint = [m - b for m, b in zip(midpoint, base_pos)]
        print("=" * 70)
        print(f"[FINAL] jaw1_link_world = {jaw1_pos}")
        print(f"[FINAL] jaw2_link_world = {jaw2_pos}")
        print(f"[FINAL] midpoint(jaw1,jaw2) = {midpoint}")
        print(f"[FINAL] gripper_base_link_world = {base_pos}")
        print(f"[FINAL] midpoint - gripper_base_link = {offset_base_to_midpoint} (expect ~(0,-0.036,0) in base's LOCAL frame, world-frame numbers here will differ by the base's own orientation)")
        print("=" * 70)

    demo_video_writer.close()
    env.close()
    print(f"Video: {demo_video_path}")


if __name__ == "__main__":
    main()

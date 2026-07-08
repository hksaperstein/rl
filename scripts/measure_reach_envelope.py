"""Directly measure the AR4 mk5 arm's real Cartesian reach envelope via
forward kinematics (no IK, no solver, no convergence questions at all) and
compare it against the cube's distance from the robot base.

This exists to test, cleanly, whether the target position (0.20, 0.28, 0.009)
that every IK-driven script this session has failed to reach is simply
outside the arm's physical workspace - a single-env, no-iteration, ground-
truth measurement, deliberately avoiding every source of ambiguity in the
IK-convergence diagnostics that came before it.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/measure_reach_envelope.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure AR4 reach envelope via forward kinematics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Ar4GraspVerifyEnvCfg's scene includes a camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

CUBE_POS_W = (0.20, 0.28, 0.009)


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    min_limits = joint_pos_limits[0, :, 0].tolist()
    max_limits = joint_pos_limits[0, :, 1].tolist()
    print(f"[INFO] Joint limits (rad): {list(zip(ARM_JOINT_NAMES, min_limits, max_limits))}")

    num_arm_joints = len(ARM_JOINT_NAMES)
    settle_steps = 60

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        cube_dist_horizontal = torch.norm(cube_pos_b[0, :2]).item()
        cube_dist_3d = torch.norm(cube_pos_b[0]).item()
        print(f"[INFO] Cube position (robot frame): {cube_pos_b[0].tolist()}")
        print(f"[INFO] Cube horizontal distance from base: {cube_dist_horizontal:.4f}m")
        print(f"[INFO] Cube full 3D distance from base: {cube_dist_3d:.4f}m")

        def settle_and_measure(q):
            for _ in range(settle_steps):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = q[j]
                action[:, num_arm_joints] = 1.0
                env.step(action)
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            ee_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            return ee_pos_b[0]

        # Sample a spread of joint_1/2/3 combinations at/near their extremes
        # (joints 4-6 held at 0 - wrist orientation doesn't add horizontal
        # reach for this arm's geometry, only rotates the end effector) to
        # empirically find the maximum horizontal reach the arm can achieve,
        # rather than assuming any particular "outstretched" combination is
        # the true maximum.
        j1_min, j2_min, j3_min = min_limits[0], min_limits[1], min_limits[2]
        j1_max, j2_max, j3_max = max_limits[0], max_limits[1], max_limits[2]

        candidates = []
        for j1 in [0.0, j1_min, j1_max, -0.9273]:  # -0.9273 rad = atan2(-0.28,-0.20), pointing at cube
            for j2 in [j2_min, j2_min * 0.5, 0.0, j2_max * 0.5, j2_max]:
                for j3 in [j3_min, j3_min * 0.5, 0.0, j3_max * 0.5, j3_max]:
                    candidates.append([j1, j2, j3, 0.0, 0.0, 0.0])

        print(f"\n[INFO] Sweeping {len(candidates)} joint_1/2/3 combinations to find max reach...")

        best_horizontal = -1.0
        best_config = None
        best_ee_pos = None
        results = []

        for i, q in enumerate(candidates):
            ee_pos_b = settle_and_measure(q)
            horizontal = torch.norm(ee_pos_b[:2]).item()
            results.append((horizontal, ee_pos_b.tolist(), q))
            if horizontal > best_horizontal:
                best_horizontal = horizontal
                best_config = q
                best_ee_pos = ee_pos_b.tolist()
            if i % 20 == 0:
                print(f"  [{i}/{len(candidates)}] q={['%.3f' % x for x in q]} -> ee_pos_b={['%.4f' % x for x in ee_pos_b.tolist()]} horizontal={horizontal:.4f}m")

        print("\n" + "=" * 80)
        print("REACH ENVELOPE RESULT")
        print("=" * 80)
        print(f"Maximum horizontal reach found across {len(candidates)} sampled configs: {best_horizontal:.4f}m")
        print(f"Achieved at joint config: {best_config}")
        print(f"EE position (robot frame) at max reach: {best_ee_pos}")
        print(f"\nCube horizontal distance from base: {cube_dist_horizontal:.4f}m")
        print(f"Cube full 3D distance from base: {cube_dist_3d:.4f}m")

        if best_horizontal < cube_dist_horizontal:
            deficit = cube_dist_horizontal - best_horizontal
            print(f"\n*** CUBE IS OUTSIDE THE SAMPLED REACH ENVELOPE by {deficit:.4f}m (horizontal) ***")
            print("This is consistent with the repeated ~0.3m IK-solve plateau seen across every")
            print("script this session - the target may be genuinely beyond this arm's reach.")
        else:
            margin = best_horizontal - cube_dist_horizontal
            print(f"\nCube horizontal distance IS within the sampled reach envelope (margin: {margin:.4f}m).")
            print("This would mean the target is geometrically reachable and the repeated IK stall")
            print("is a solver/path issue, not a workspace-boundary issue.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""3D matplotlib visualization of the AR4 arm's kinematic skeleton at a given pose.

Reads real simulated body positions from Isaac Lab (not reimplemented forward
kinematics), plots the arm chain, gripper fingers, end-effector target, and cube
in 3D space.

Usage:
    /home/saps/IsaacLab/isaaclab.sh -p scripts/plot_arm_skeleton.py [--joint-angles q1 q2 q3 q4 q5 q6] [--gripper open|closed]

Examples:
    # Default rest pose, gripper open
    /home/saps/IsaacLab/isaaclab.sh -p scripts/plot_arm_skeleton.py

    # Custom pose: all joints at pi/4, gripper closed
    /home/saps/IsaacLab/isaaclab.sh -p scripts/plot_arm_skeleton.py --joint-angles 0.785 0.785 0.785 0.785 0.785 0.785 --gripper closed
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Plot AR4 arm skeleton at a given pose.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--joint-angles",
    type=float,
    nargs=6,
    default=None,
    help="Target joint angles (radians) for joint_1 through joint_6. If not given, uses default reset pose.",
)
parser.add_argument(
    "--gripper",
    type=str,
    choices=["open", "closed"],
    default="open",
    help="Gripper state: 'open' or 'closed'.",
)
args_cli = parser.parse_args()

# Force headless mode since this script doesn't need the GUI
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS, ARM_JOINT_NAMES  # noqa: E402

# Time-based settle pattern (same as interactive_joint_demo.py's fix)
SETTLE_TIME_S = 0.5  # seconds to let the arm settle to commanded pose


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Disable cameras to avoid --enable_cameras requirement
    env_cfg.scene.perception_camera = None
    env_cfg.scene.demo_camera = None
    env = ManagerBasedEnv(cfg=env_cfg)

    # Convert time-based settle to step count
    settle_steps = max(1, round(SETTLE_TIME_S / env.physics_dt))

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)

    with torch.inference_mode():
        env.reset()

        # Command custom joint angles if provided
        if args_cli.joint_angles is not None:
            print(f"[INFO] Commanding joint angles: {args_cli.joint_angles}")
            arm_target = torch.tensor([args_cli.joint_angles], device=env.device)

            # Set gripper target
            gripper_target_val = GRIPPER_CLOSED_POS if args_cli.gripper == "closed" else GRIPPER_OPEN_POS
            gripper_target = torch.tensor([[gripper_target_val, gripper_target_val]], device=env.device)

            # Step for settle_steps to let the pose stabilize
            for _ in range(settle_steps):
                robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
                robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
                robot.write_data_to_sim()
                env.sim.step(render=False)
                robot.update(env.physics_dt)

            print(f"[INFO] Settled for {settle_steps} steps (~{settle_steps * env.physics_dt:.3f}s)")
        else:
            print("[INFO] Using default reset pose")

        # Read the actual gripper state
        gripper_state = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
        print(f"[INFO] Gripper jaw positions: {gripper_state}")

        # Introspect body names at runtime
        body_names = robot.data.body_names
        print(f"[INFO] Available body names: {body_names}")

        # Expected arm chain and gripper links
        expected_chain = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
        expected_gripper = ["gripper_jaw1_link", "gripper_jaw2_link"]

        # Resolve indices for arm chain
        arm_chain_indices = []
        for name in expected_chain:
            if name in body_names:
                arm_chain_indices.append(body_names.index(name))
            else:
                available = ", ".join(body_names)
                raise ValueError(
                    f"Expected body '{name}' not found in robot.\n"
                    f"Available bodies: {available}"
                )

        # Resolve indices for gripper
        gripper_indices = {}
        for name in expected_gripper:
            if name in body_names:
                gripper_indices[name] = body_names.index(name)
            else:
                available = ", ".join(body_names)
                raise ValueError(
                    f"Expected body '{name}' not found in robot.\n"
                    f"Available bodies: {available}"
                )

        # Read world positions for all bodies in the chain
        body_pos_w = robot.data.body_pos_w[0].cpu().numpy()  # Shape: (num_bodies, 3)

        arm_positions = np.array([body_pos_w[idx] for idx in arm_chain_indices])  # Shape: (7, 3)
        gripper_jaw1_pos = body_pos_w[gripper_indices["gripper_jaw1_link"]]  # Shape: (3,)
        gripper_jaw2_pos = body_pos_w[gripper_indices["gripper_jaw2_link"]]  # Shape: (3,)

        # Read EE frame target (the actual pinch point)
        ee_target_pos = env.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu().numpy()  # Shape: (3,)

        # Read cube position
        cube_pos = env.scene["cube"].data.root_pos_w[0].cpu().numpy()  # Shape: (3,)

        # Print numeric positions for sanity checking
        print("\n" + "=" * 70)
        print("NUMERIC POSITIONS (meters, world frame):")
        print("=" * 70)
        for i, name in enumerate(expected_chain):
            pos = arm_positions[i]
            print(f"  {name:20s}: ({pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f})")
        print(f"  {'gripper_jaw1_link':20s}: ({gripper_jaw1_pos[0]:+.6f}, {gripper_jaw1_pos[1]:+.6f}, {gripper_jaw1_pos[2]:+.6f})")
        print(f"  {'gripper_jaw2_link':20s}: ({gripper_jaw2_pos[0]:+.6f}, {gripper_jaw2_pos[1]:+.6f}, {gripper_jaw2_pos[2]:+.6f})")
        print(f"  {'EE_target (pinch)':20s}: ({ee_target_pos[0]:+.6f}, {ee_target_pos[1]:+.6f}, {ee_target_pos[2]:+.6f})")
        print(f"  {'cube':20s}: ({cube_pos[0]:+.6f}, {cube_pos[1]:+.6f}, {cube_pos[2]:+.6f})")
        print("=" * 70 + "\n")

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the arm chain as a line
        ax.plot(arm_positions[:, 0], arm_positions[:, 1], arm_positions[:, 2], "b-", linewidth=2, label="Arm chain")

        # Plot arm joints as scatter points, labeled
        ax.scatter(arm_positions[:, 0], arm_positions[:, 1], arm_positions[:, 2], c="blue", s=50, marker="o")
        for i, name in enumerate(expected_chain):
            pos = arm_positions[i]
            ax.text(pos[0], pos[1], pos[2], f"  {name}", fontsize=8, color="blue")

        # Plot gripper fingers: two short segments from link_6 to each jaw
        link6_pos = arm_positions[-1]
        ax.plot([link6_pos[0], gripper_jaw1_pos[0]], [link6_pos[1], gripper_jaw1_pos[1]], [link6_pos[2], gripper_jaw1_pos[2]], "g--", linewidth=1.5)
        ax.plot([link6_pos[0], gripper_jaw2_pos[0]], [link6_pos[1], gripper_jaw2_pos[1]], [link6_pos[2], gripper_jaw2_pos[2]], "g--", linewidth=1.5)

        # Plot gripper jaw positions
        ax.scatter([gripper_jaw1_pos[0]], [gripper_jaw1_pos[1]], [gripper_jaw1_pos[2]], c="green", s=80, marker="s", label="Gripper jaw1")
        ax.scatter([gripper_jaw2_pos[0]], [gripper_jaw2_pos[1]], [gripper_jaw2_pos[2]], c="lime", s=80, marker="s", label="Gripper jaw2")

        ax.text(gripper_jaw1_pos[0], gripper_jaw1_pos[1], gripper_jaw1_pos[2], "  jaw1", fontsize=8, color="green")
        ax.text(gripper_jaw2_pos[0], gripper_jaw2_pos[1], gripper_jaw2_pos[2], "  jaw2", fontsize=8, color="lime")

        # Plot EE frame target (the actual pinch point, not link_6)
        ax.scatter([ee_target_pos[0]], [ee_target_pos[1]], [ee_target_pos[2]], c="red", s=100, marker="*", label="EE target (pinch point)")
        ax.text(ee_target_pos[0], ee_target_pos[1], ee_target_pos[2], "  EE_target", fontsize=8, color="red")

        # Plot cube position with size indicator (~12mm = 0.006m half-size)
        cube_half_size = 0.006  # meters
        ax.scatter([cube_pos[0]], [cube_pos[1]], [cube_pos[2]], c="orange", s=150, marker="^", label="Cube (center)")
        ax.text(cube_pos[0], cube_pos[1], cube_pos[2], "  cube", fontsize=8, color="orange")

        # Draw a small box around the cube to show its size
        # Simple wireframe cube corners
        d = cube_half_size
        cube_corners = np.array(
            [
                cube_pos + np.array([d, d, d]),
                cube_pos + np.array([d, d, -d]),
                cube_pos + np.array([d, -d, d]),
                cube_pos + np.array([d, -d, -d]),
                cube_pos + np.array([-d, d, d]),
                cube_pos + np.array([-d, d, -d]),
                cube_pos + np.array([-d, -d, d]),
                cube_pos + np.array([-d, -d, -d]),
            ]
        )

        # Draw cube edges (a few representative ones)
        cube_edges = [
            (0, 1),
            (0, 2),
            (4, 5),
            (4, 6),
        ]
        for edge in cube_edges:
            corner1, corner2 = cube_corners[edge[0]], cube_corners[edge[1]]
            ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]], [corner1[2], corner2[2]], "orange", alpha=0.4, linewidth=0.5)

        # Set equal aspect ratio
        all_positions = np.vstack([arm_positions, gripper_jaw1_pos, gripper_jaw2_pos, ee_target_pos, cube_pos])
        max_range = np.array([all_positions[:, 0].max() - all_positions[:, 0].min(), all_positions[:, 1].max() - all_positions[:, 1].min(), all_positions[:, 2].max() - all_positions[:, 2].min()]).max() / 2.0
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)

        title = "AR4 Arm Skeleton"
        if args_cli.joint_angles is not None:
            title += f" (q={args_cli.joint_angles})"
        title += f", gripper={args_cli.gripper}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.legend(loc="upper left", fontsize=8)

        # Create output directory
        plots_dir = Path("/home/saps/projects/rl/logs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = plots_dir / f"arm_skeleton_{timestamp}.png"

        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved plot to: {output_path}")

        # Verify the file was created and is non-trivial
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"[INFO] File size: {file_size} bytes")
            if file_size > 5000:  # Should be at least a few KB for a real plot
                print("[SUCCESS] Plot saved successfully!")
            else:
                print("[WARNING] Plot file seems very small, may be blank or corrupt")
        else:
            print("[ERROR] Plot file was not created!")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

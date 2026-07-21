"""Pure-matplotlib 3D visualization module for the toy_env proxy environment.

No Isaac Sim/rendering dependency — uses standard matplotlib 3D plotting for
static and animated visualization of arm poses, trajectories, and episode
rollouts in the CPU-only kinematic-arm environment.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from toy_env import kinematic_arm as ka


# Lightweight snapshot of environment state at a single step, for
# reconstructing correct historical poses during GIF animation.
EpisodeSnapshot = namedtuple(
    "EpisodeSnapshot",
    ["step_index", "theta", "target_pos", "object_pos", "carrying", "trajectory_so_far", "info"],
)


def plot_arm_pose_3d(
    env: Any,
    ax: Any = None,
    show_trajectory: bool = True,
    ghost_thetas: list[np.ndarray] | None = None,
    title: str | None = None,
) -> Any:
    """Plot arm pose in 3D, with optional trajectory and ghost poses.

    Args:
        env: ArmReachEnv instance, with attributes theta, target_pos,
             object_pos, carrying, trajectory, action_mode.
        ax: matplotlib 3D Axes to plot on. If None, creates a new figure.
        show_trajectory: if True and env.trajectory has >1 point, plots
                        the trajectory as a thin dashed line.
        ghost_thetas: optional iterable of (7,) joint-angle arrays
                     representing earlier poses; plotted as faint ghosts.
        title: title string. If None, uses f"ArmReachEnv ({env.action_mode})".

    Returns:
        The Axes object (caller can access .figure to save/close).
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax.clear()

    # Plot the current arm pose.
    fk = ka.forward_kinematics(env.theta)
    joint_pos = fk.joint_positions

    # Plot arm as a connected line through all joint positions.
    ax.plot(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], "b-", linewidth=2, label="Arm")

    # Scatter markers at joints: base, intermediate joints, end-effector.
    # Base origin at index 0.
    ax.scatter(
        joint_pos[0, 0],
        joint_pos[0, 1],
        joint_pos[0, 2],
        c="darkblue",
        s=100,
        marker="o",
        label="Base",
    )

    # Intermediate joints (indices 1 to N_JOINTS).
    if ka.N_JOINTS > 0:
        ax.scatter(
            joint_pos[1:-1, 0],
            joint_pos[1:-1, 1],
            joint_pos[1:-1, 2],
            c="blue",
            s=50,
            marker="o",
            label="Joints",
        )

    # End-effector tip (last index).
    ax.scatter(
        joint_pos[-1, 0],
        joint_pos[-1, 1],
        joint_pos[-1, 2],
        c="red",
        s=150,
        marker="*",
        label="EE Tip",
    )

    # Plot ghost poses (earlier poses in faint lines).
    if ghost_thetas:
        for ghost_theta in ghost_thetas:
            ghost_fk = ka.forward_kinematics(ghost_theta)
            ghost_joint_pos = ghost_fk.joint_positions
            ax.plot(
                ghost_joint_pos[:, 0],
                ghost_joint_pos[:, 1],
                ghost_joint_pos[:, 2],
                "b-",
                linewidth=0.5,
                alpha=0.25,
            )

    # Plot target as a green star/diamond.
    ax.scatter(
        env.target_pos[0],
        env.target_pos[1],
        env.target_pos[2],
        c="green",
        s=200,
        marker="*",
        label="Target",
    )

    # Plot object position, unless it overlaps with target (within 1e-6).
    dist_to_target = np.linalg.norm(env.object_pos - env.target_pos)
    if dist_to_target > 1e-6:
        ax.scatter(
            env.object_pos[0],
            env.object_pos[1],
            env.object_pos[2],
            c="purple",
            s=100,
            marker="s",
            label="Object",
        )

    # Plot trajectory if requested and has multiple points.
    if show_trajectory and len(env.trajectory) > 1:
        traj_array = np.array(env.trajectory)
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            traj_array[:, 2],
            "--",
            color="gray",
            linewidth=1,
            alpha=0.5,
            label="Trajectory",
        )

    # Set equal-ish axis limits for undistorted view.
    # Collect all plotted points to determine axis range.
    all_points = [joint_pos, np.array([env.target_pos, env.object_pos])]
    if show_trajectory and len(env.trajectory) > 1:
        all_points.append(np.array(env.trajectory))

    all_points_array = np.vstack(all_points)
    x_min, x_max = all_points_array[:, 0].min(), all_points_array[:, 0].max()
    y_min, y_max = all_points_array[:, 1].min(), all_points_array[:, 1].max()
    z_min, z_max = all_points_array[:, 2].min(), all_points_array[:, 2].max()

    # Pad slightly to avoid cramping.
    margin = 0.05
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    z_range = max(z_max - z_min, 0.1)

    max_range = max(x_range, y_range, z_range)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    ax.set_xlim(x_mid - max_range / 2 - margin, x_mid + max_range / 2 + margin)
    ax.set_ylim(y_mid - max_range / 2 - margin, y_mid + max_range / 2 + margin)
    ax.set_zlim(z_mid - max_range / 2 - margin, z_mid + max_range / 2 + margin)

    # Try to set equal aspect ratio; fall back gracefully if not supported.
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    # Labels and title.
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    if title is None:
        title = f"ArmReachEnv ({env.action_mode})"
    ax.set_title(title)

    ax.legend(loc="upper left", fontsize=8)

    return ax


def render_episode_gif(
    env: Any,
    policy: Callable[[np.ndarray], np.ndarray] | None = None,
    out_path: str = "episode.gif",
    max_steps: int | None = None,
    fps: int = 10,
    ghost_every: int = 15,
) -> dict[str, Any]:
    """Render an episode rollout as an animated GIF using matplotlib.

    Args:
        env: ArmReachEnv instance.
        policy: callable that takes observation and returns action. If None,
               uses random actions from env.action_space.sample(). Supports
               both plain action arrays and SB3-style (action, state) tuples.
        out_path: path to save the output GIF.
        max_steps: max steps to run. If None, uses env.max_episode_steps.
        fps: frames per second for the GIF.
        ghost_every: record a ghost pose every N steps to show in the GIF.

    Returns:
        dict with keys: n_steps, final_dist, success, out_path.
    """
    # Reset environment.
    obs, info = env.reset()

    if max_steps is None:
        max_steps = env.max_episode_steps

    # Run the full episode and collect snapshots at each step.
    snapshots = []
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps and not terminated and not truncated:
        # Record current state before stepping.
        snapshot = EpisodeSnapshot(
            step_index=step_count,
            theta=env.theta.copy(),
            target_pos=env.target_pos.copy(),
            object_pos=env.object_pos.copy(),
            carrying=env.carrying,
            trajectory_so_far=[p.copy() for p in env.trajectory],
            info=info,
        )
        snapshots.append(snapshot)

        # Compute action.
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(obs)
            # Handle SB3-style (action, state) tuple return.
            if isinstance(action, tuple):
                action = action[0]

        # Step environment.
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    # Record final snapshot.
    if step_count < max_steps:
        snapshot = EpisodeSnapshot(
            step_index=step_count,
            theta=env.theta.copy(),
            target_pos=env.target_pos.copy(),
            object_pos=env.object_pos.copy(),
            carrying=env.carrying,
            trajectory_so_far=[p.copy() for p in env.trajectory],
            info=info,
        )
        snapshots.append(snapshot)

    # Create figure and axes for animation.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Store the last info dict for summary.
    final_info = snapshots[-1].info if snapshots else info

    def update_frame(frame_idx: int) -> Any:
        """Update function for FuncAnimation."""
        snapshot = snapshots[frame_idx]

        # Determine which ghost poses to show (every ghost_every steps).
        ghost_indices = [
            i
            for i in range(frame_idx + 1)
            if i % ghost_every == 0 and i < frame_idx
        ]
        ghost_thetas = [snapshots[i].theta for i in ghost_indices]

        # Temporarily restore env state for plotting (plot_arm_pose_3d reads from env).
        env.theta = snapshot.theta
        env.target_pos = snapshot.target_pos
        env.object_pos = snapshot.object_pos
        env.carrying = snapshot.carrying
        env.trajectory = snapshot.trajectory_so_far

        # Create title with step number and distance info.
        dist = snapshot.info.get("dist", 0.0)
        step_title = f"Step {snapshot.step_index} (dist={dist:.3f}m)"

        # Plot the frame.
        plot_arm_pose_3d(env, ax=ax, show_trajectory=True, ghost_thetas=ghost_thetas, title=step_title)

    # Create animation.
    anim = FuncAnimation(fig, update_frame, frames=len(snapshots), repeat=True, interval=1000 // fps)

    # Save as GIF.
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)

    # Build summary.
    summary = {
        "n_steps": step_count,
        "final_dist": float(final_info.get("dist", 0.0)),
        "success": bool(final_info.get("success", False)),
        "out_path": out_path,
    }

    return summary

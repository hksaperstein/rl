# scripts/measure_planarity_residual.py
"""One-off measurement script for ROADMAP.md item 2 (shape classifier misclassifies
cube/rectangular-prism as sphere against real depth data).

Holds the cube motionless at its real production resting pose (objects_cfg.py's
CUBE_CFG init position, (0.20, 0.28, 0.009) - NOT centered under the camera at
(0, 0.31): an earlier dead-center measurement turned out to be unrepresentative for
two reasons found during this investigation - (1) the camera-center point coincides
with the robot's resting end-effector, which occludes the cube entirely at some
sim steps, and (2) a perfectly axis-aligned, dead-center view removes the oblique
partial-side-wall visibility that turns out to be the real source of "residual"
in this pipeline (Isaac Sim's render noise itself is negligible - see the
report). (0.20, 0.28) is where the cube actually sits during real eval/demo runs
and is clear of the robot's resting pose.

Captures depth frames across several simulation steps, segments the cube's point
cluster, and computes the planarity residual via shape_classifier._fit_plane - the
exact function `classify_shape` uses - so the measured distribution is directly
comparable to what the real classifier sees. Reports mean/std/min/max, which is
then used to set PLANARITY_RESIDUAL_THRESHOLD = mean + 3*std in
perception/shape_classifier.py.

Not part of any automated flow - a one-time calibration measurement, same spirit as
scripts/perception_calibration.py's qualitative check but producing a number.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/measure_planarity_residual.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the real camera's planarity residual on the cube.")
parser.add_argument("--num-frames", type=int, default=20, help="Number of captured frames.")
parser.add_argument("--warmup-steps", type=int, default=5, help="Steps to let the render/scene settle before capture.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from perception.pipeline import build_world_point_grid  # noqa: E402
from perception.segmentation import segment_objects  # noqa: E402
from perception.shape_classifier import _fit_plane  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlacePerceptionEnvCfg  # noqa: E402

CUBE_X = 0.20  # cube's real resting x, see objects_cfg.py's CUBE_CFG init_state
CUBE_Y = 0.28  # cube's real resting y, see objects_cfg.py's CUBE_CFG init_state
CUBE_Z = 0.009  # cube's resting half-height, see objects_cfg.py's CUBE_CFG
CUBE_HEIGHT = 0.018  # expected cluster height (m) - sanity check against occlusion/merge


def main() -> None:
    env_cfg = Ar4PickPlacePerceptionEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    camera = env.scene["perception_camera"]

    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)
    actions[:, -1] = 1.0  # keep the gripper open

    pose = torch.tensor([[CUBE_X, CUBE_Y, CUBE_Z, 1.0, 0.0, 0.0, 0.0]], device=env.device)

    residuals: list[float] = []
    cluster_sizes: list[int] = []

    with torch.inference_mode():
        env.reset()
        for step in range(args_cli.warmup_steps + args_cli.num_frames):
            env.scene["cube"].write_root_pose_to_sim(pose)
            env.step(actions)

            if step < args_cli.warmup_steps:
                continue

            depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
            intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
            cam_pos = camera.data.pos_w[0].cpu().numpy()
            cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

            labeled, valid_ids = segment_objects(depth)
            world_points = build_world_point_grid(depth, intrinsics, cam_pos, cam_quat_ros)

            if not valid_ids:
                print(f"[frame {step - args_cli.warmup_steps}] no clusters detected, skipping", flush=True)
                continue

            # Pick the cluster nearest the cube's known (x, y) - other objects
            # (rect_prism, sphere, wedge, and the robot itself) are also in frame.
            best_id, best_dist = None, np.inf
            for label_id in valid_ids:
                mask = labeled == label_id
                centroid_xy = world_points[mask][:, :2].mean(axis=0)
                dist = float(np.hypot(centroid_xy[0] - CUBE_X, centroid_xy[1] - CUBE_Y))
                if dist < best_dist:
                    best_id, best_dist = label_id, dist

            cluster_points = world_points[labeled == best_id]
            height = float(cluster_points[:, 2].max())
            # Guard against picking up the robot/another object instead of the cube
            # (e.g. if the gripper happens to occlude it) - skip anything that isn't
            # plausibly the cube itself.
            if best_dist > 0.02 or abs(height - CUBE_HEIGHT) > 0.003:
                print(
                    f"[frame {step - args_cli.warmup_steps}] nearest cluster doesn't look like the cube "
                    f"(dist={best_dist:.4f}m height={height:.4f}m) - skipping",
                    flush=True,
                )
                continue

            _, residual = _fit_plane(cluster_points)
            residuals.append(residual)
            cluster_sizes.append(int(cluster_points.shape[0]))
            print(
                f"[frame {step - args_cli.warmup_steps}] cluster_dist_to_cube={best_dist:.4f}m "
                f"n_points={cluster_points.shape[0]} height={height:.5f}m residual={residual:.9f}m",
                flush=True,
            )

    env.close()

    residuals_arr = np.array(residuals)
    print("\n=== Planarity residual distribution (cube, real perception_camera) ===", flush=True)
    print(f"n_frames={len(residuals_arr)} total_points={sum(cluster_sizes)}", flush=True)
    print(f"mean={residuals_arr.mean():.7f}m std={residuals_arr.std():.7f}m", flush=True)
    print(f"min={residuals_arr.min():.7f}m max={residuals_arr.max():.7f}m", flush=True)
    print(f"mean + 3*std = {residuals_arr.mean() + 3 * residuals_arr.std():.7f}m", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

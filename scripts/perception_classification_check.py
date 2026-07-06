# scripts/perception_classification_check.py
"""Classifies all four objects (cube, rect_prism, sphere, wedge) at their default
resting poses (objects_cfg.py's fixed init_state positions, no movement) using the
real perception_camera, printing per-object planarity residual/height/tilt/
circularity/label. A quick end-to-end regression check for shape_classifier.py
changes: run this after touching any classify_shape threshold and confirm every
object still gets its correct label against the actual rendered depth camera, not
just the synthetic point clouds in perception/tests/test_shape_classifier.py.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/perception_classification_check.py --headless
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from perception.pipeline import build_world_point_grid  # noqa: E402
from perception.segmentation import segment_objects  # noqa: E402
from perception.shape_classifier import classify_shape  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlacePerceptionEnvCfg  # noqa: E402

# Known default resting (x, y) per objects_cfg.py's *_CFG.init_state.
KNOWN_OBJECTS = {
    "cube": (0.20, 0.28),
    "rect_prism": (0.20, 0.34),
    "sphere": (-0.20, 0.28),
    "wedge": (-0.20, 0.34),
}


def main():
    env_cfg = Ar4PickPlacePerceptionEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    camera = env.scene["perception_camera"]

    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)
    actions[:, -1] = 1.0

    with torch.inference_mode():
        env.reset()
        for _ in range(5):  # warmup
            env.step(actions)

        depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
        cam_pos = camera.data.pos_w[0].cpu().numpy()
        cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

        labeled, valid_ids = segment_objects(depth)
        world_points = build_world_point_grid(depth, intrinsics, cam_pos, cam_quat_ros)

        print(f"n_clusters={len(valid_ids)}", flush=True)
        for name, (ox, oy) in KNOWN_OBJECTS.items():
            best_id, best_dist = None, np.inf
            for label_id in valid_ids:
                mask = labeled == label_id
                centroid_xy = world_points[mask][:, :2].mean(axis=0)
                dist = float(np.hypot(centroid_xy[0] - ox, centroid_xy[1] - oy))
                if dist < best_dist:
                    best_id, best_dist = label_id, dist
            mask = labeled == best_id
            cluster_points = world_points[mask]
            result = classify_shape(cluster_points, ground_z=GROUND_Z)
            print(
                f"{name:12s} dist={best_dist:.4f} pixels={int(mask.sum())} label={result.label:16s} "
                f"height={result.height:.5f} residual={result.planarity_residual:.7f} "
                f"tilt_deg={np.degrees(result.tilt_rad):.2f} circ={result.circularity:.3f}",
                flush=True,
            )

    env.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()

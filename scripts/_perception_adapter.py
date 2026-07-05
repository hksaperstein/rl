# rl/scripts/_perception_adapter.py
"""Shared helper for eval_loop.py --perception and interactive_demo.py: builds
the trained policy's cube-position observation slot from the real perception
pipeline instead of privileged simulation state.
"""

import numpy as np
import torch

from isaaclab.utils.math import subtract_frame_transforms

from perception.pipeline import run_perception
from perception.tracker import find_by_shape


def cube_position_obs_slice(env) -> tuple[int, int]:
    """Column range of the 'cube_position' term within the concatenated policy
    observation tensor, computed from the observation manager rather than
    hardcoded, so it can't silently drift out of sync with pickplace_env_cfg.py.
    """
    term_names = env.observation_manager.active_terms["policy"]
    term_dims = env.observation_manager.group_obs_term_dim["policy"]
    offset = 0
    for name, dim in zip(term_names, term_dims):
        size = int(np.prod(dim))
        if name == "cube_position":
            return offset, offset + size
        offset += size
    raise ValueError("cube_position term not found in the policy observation group.")


def perceive_cube(env, camera, tracker, ground_z: float):
    """Runs perception on the camera's current frame, updates `tracker`, and
    returns (cube_position_in_robot_root_frame_or_None, tracked_objects, rgb_frame).
    `env` must be the raw ManagerBasedRLEnv (e.g. `env.unwrapped`), not the
    rsl_rl-wrapped env."""
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos = camera.data.pos_w[0].cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[0].cpu().numpy()

    detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=ground_z)
    tracked = tracker.update(detections)
    cube = find_by_shape(tracked, "cube")
    if cube is None:
        return None, tracked, rgb

    object_pos_w = torch.tensor(cube.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot = env.scene["robot"]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b, tracked, rgb

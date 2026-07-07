# scripts/_perception_adapter.py
"""Shared helper for eval_loop.py --perception, interactive_demo.py, and
train.py's camera-observed training experiment: builds the trained policy's
sphere-position observation slot from the real perception pipeline instead of
privileged simulation state.
"""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import subtract_frame_transforms

from perception.pipeline import run_perception
from perception.tracker import ObjectTracker, find_by_shape


def observation_term_slice(env, group_name: str, term_name: str) -> tuple[int, int]:
    """Column range of `term_name` within `group_name`'s concatenated observation
    tensor, computed from the observation manager rather than hardcoded, so it
    can't silently drift out of sync with the task's env cfg.
    """
    term_names = env.observation_manager.active_terms[group_name]
    term_dims = env.observation_manager.group_obs_term_dim[group_name]
    offset = 0
    for name, dim in zip(term_names, term_dims):
        size = int(np.prod(dim))
        if name == term_name:
            return offset, offset + size
        offset += size
    raise ValueError(f"{term_name!r} term not found in the {group_name!r} observation group.")


def sphere_position_obs_slice(env) -> tuple[int, int]:
    """Column range of the 'sphere_position' term within the concatenated policy
    observation tensor. Thin wrapper around `observation_term_slice` kept for
    call-site readability at the two existing eval/demo entry points."""
    return observation_term_slice(env, "policy", "sphere_position")


def perceive_object(env, camera, tracker, ground_z: float, shape_label: str, env_index: int = 0):
    """Runs perception on `camera`'s current frame for one env, updates `tracker`,
    and returns (object_position_in_robot_root_frame_or_None, tracked_objects,
    rgb_frame) for the first tracked object matching `shape_label`. `env` must
    be the raw ManagerBasedRLEnv (e.g. `env.unwrapped`), not the rsl_rl-wrapped
    env. `env_index` selects which parallel env's camera/robot data to read
    (default 0, matching eval_loop.py/interactive_demo.py/
    classical_pickplace_demo.py's single-env usage)."""
    depth = camera.data.output["distance_to_image_plane"][env_index, ..., 0].cpu().numpy()
    rgb = camera.data.output["rgb"][env_index, ..., :3].cpu().numpy().astype(np.uint8)
    intrinsics = camera.data.intrinsic_matrices[env_index].cpu().numpy()
    cam_pos = camera.data.pos_w[env_index].cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[env_index].cpu().numpy()

    detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=ground_z)
    tracked = tracker.update(detections)
    obj = find_by_shape(tracked, shape_label)
    if obj is None:
        return None, tracked, rgb

    object_pos_w = torch.tensor(obj.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot = env.scene["robot"]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_index : env_index + 1],
        robot.data.root_quat_w[env_index : env_index + 1],
        object_pos_w,
    )
    return object_pos_b, tracked, rgb


def perceive_sphere(env, camera, tracker, ground_z: float, env_index: int = 0):
    """Thin wrapper around perceive_object for the sphere-specific call sites
    (eval_loop.py --perception, interactive_demo.py) - neither needs any
    changes as a result of perceive_object's addition."""
    return perceive_object(env, camera, tracker, ground_z, "sphere", env_index)


class PerceptionObservationWrapper(gym.Wrapper):
    """Overrides the policy's 'sphere_position' observation columns with a
    position derived from the real top-down RGB-D `perception_camera` + this
    repo's classical perception pipeline (`perception.pipeline.run_perception`
    + `perception.tracker.find_by_shape`), per parallel env, instead of the
    privileged ground-truth `mdp.object_position_in_robot_root_frame` value
    the observation manager computes by default. The reward function is
    untouched - it keeps using privileged simulation state; only this one
    observation term changes.

    When a given env's sphere isn't detected this frame, that env's column
    slice is left as whatever the observation manager already computed (i.e.
    the privileged value for that step) rather than some stale synthetic
    fallback - the same "no override this frame" behavior
    `perceive_sphere`/eval_loop.py already uses for a `None` detection.

    Implementation note - wrapper order and why this isn't a plain step()/
    reset() override: `RslRlVecEnvWrapper` must be the *last* wrapper in the
    chain (its own docstring), so this wrapper must sit between the raw
    `ManagerBasedRLEnv` and it. But `rsl_rl`'s `OnPolicyRunner.learn()` fetches
    its very first observation via `RslRlVecEnvWrapper.get_observations()`,
    which calls `self.unwrapped.observation_manager.compute()` directly -
    bypassing every `gym.Wrapper` in the chain, including a naive override of
    this wrapper's own `step()`/`reset()`. To intercept all three entry points
    (`step`, `reset`, `get_observations`) uniformly, this wrapper instead
    monkeypatches the raw env's `observation_manager.compute` bound method at
    construction time - every entry point bottoms out in that same call
    (`isaaclab.envs.manager_based_env.ManagerBasedEnv.step`/`reset` and
    `manager_based_rl_env.ManagerBasedRLEnv.step` all set
    `self.obs_buf = self.observation_manager.compute(...)` directly).
    """

    def __init__(self, env: gym.Env, ground_z: float = 0.0):
        super().__init__(env)
        base_env = env.unwrapped
        self._col_start, self._col_end = sphere_position_obs_slice(base_env)
        self._camera = base_env.scene["perception_camera"]
        self._robot = base_env.scene["robot"]
        self._ground_z = ground_z
        self._device = base_env.device
        self._num_envs = base_env.num_envs
        self._trackers = [ObjectTracker() for _ in range(self._num_envs)]

        self._original_compute = base_env.observation_manager.compute
        base_env.observation_manager.compute = self._patched_compute

    def _patched_compute(self, *args, **kwargs):
        obs_dict = self._original_compute(*args, **kwargs)
        policy_obs = obs_dict["policy"].clone()

        for i in range(self._num_envs):
            depth = self._camera.data.output["distance_to_image_plane"][i, ..., 0].cpu().numpy()
            intrinsics = self._camera.data.intrinsic_matrices[i].cpu().numpy()
            cam_pos = self._camera.data.pos_w[i].cpu().numpy()
            cam_quat_ros = self._camera.data.quat_w_ros[i].cpu().numpy()

            detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=self._ground_z)
            tracked = self._trackers[i].update(detections)
            sphere = find_by_shape(tracked, "sphere")
            if sphere is None:
                continue  # leave this env's already-computed (privileged) value in place

            object_pos_w = torch.tensor(sphere.position, dtype=torch.float32, device=self._device).unsqueeze(0)
            object_pos_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w[i : i + 1], self._robot.data.root_quat_w[i : i + 1], object_pos_w
            )
            policy_obs[i, self._col_start : self._col_end] = object_pos_b[0]

        obs_dict["policy"] = policy_obs
        return obs_dict

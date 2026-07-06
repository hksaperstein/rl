# tasks/ar4/mdp.py
"""Local MDP reward terms for the AR4 pick-and-place task that don't exist
in any of Isaac Lab's built-in `mdp` modules.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def contact_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus when both gripper fingers register real contact force
    against the sphere specifically - a ground-truth grasp signal
    (ContactSensor, filtered via force_matrix_w), replacing the geometric
    position/closure proxies every prior experiment in this repo's
    grasp-reward history used (see ROADMAP.md's "grasp/lift never emerges"
    entry for why those failed: either reward-hackable via a loose distance
    check, or too sparse to discover via a tight alignment check). Adapted
    from isaaclab_tasks' manipulation/place/agibot task's object_grasped
    pattern (bilateral force-threshold check), using one sensor per jaw and
    the filtered force_matrix_w field rather than the unfiltered
    net_forces_w the reference used - see
    docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md
    for why both corrections were necessary.
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    # force_matrix_w shape: (num_envs, 1 body, 1 filter, 3) for each sensor.
    jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    both_fingers_contact = (jaw1_force > force_threshold) & (jaw2_force > force_threshold)
    return both_fingers_contact.float()

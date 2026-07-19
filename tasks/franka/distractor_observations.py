# tasks/franka/distractor_observations.py
"""Pure-tensor math for the `distractor_distance_summary` observation term
(Task 2 of docs/superpowers/plans/2026-07-19-target-selection-clutter-
implementation.md, spec: docs/superpowers/specs/2026-07-19-target-
selection-clutter-design.md). NO isaaclab/pxr/torch-sim imports beyond
plain torch - follows the same pure-function split
tasks/franka/shape_observations.py (and, before it, tasks/franka/
lift_reward.py) already established: tasks/franka/mdp.py reads live
simulated state (the target/distractor_1/distractor_2 root positions,
env.cfg.active_distractor_count) and delegates the actual computation to
the function below.

Implements DexSinGrasp's own `d_t^S` mechanism (Xu et al., "DexSinGrasp:
Learning a Unified Policy for Dexterous Object Singulation and Grasping in
Densely Cluttered Environments," arXiv:2504.04516, §III-A Eq. 1) as
literally as this project's own schema conventions allow: a fixed-size
K=2 vector of target-to-distractor Euclidean distances, one scalar per
distractor slot (`distractor_1`, `distractor_2`), NOT a richer per-object
position/state encoding - matching the design doc's explicit choice of the
paper's own fixed-size zero-padded aggregate over a per-object one-hot flag
or an attention/pointer mechanism.

Unlike `shape_observations.py`'s `shape_class_onehot`/`geometry_descriptor`
(per-env-cfg CONSTANTS, broadcast identically to every parallel env), this
term is genuinely per-environment-VARYING - each env's own live
target/distractor positions differ - so there is no broadcast helper here,
only a single batched function operating on already-batched (num_envs, 3)
position tensors.

Zero-padding (the load-bearing correctness property): whenever
`active_distractor_count` makes a slot inactive for this env cfg's own
curriculum stage (Stage SO: both slots inactive; Stage D1: slot 1 only;
Stage D2: both active), that slot's output column is HARD-ZEROED - never
the real, possibly-large, parked-off-table distance - mirroring the
paper's own literal convention ("padded with zeros if the true distractor
count is below" the max, §III-A). This keeps the output a fixed (num_envs,
2) shape across all 3 curriculum stages, so a single policy architecture
works throughout the curriculum with checkpoint-resume (Stage D1 resumes
from Stage SO's checkpoint, Stage D2 from Stage D1's - see the design
doc's "Curriculum" section).

`active_distractor_count` is a per-env-cfg constant (0/1/2, read from
`env.cfg.active_distractor_count` by tasks/franka/mdp.py's thin wrapper),
NOT a per-environment-varying value - every env under one training run
shares the same curriculum stage, same convention as
`die_shape_classes_per_env` etc. World-frame positions in throughout;
frame choice doesn't affect a scalar Euclidean distance (matches the
design doc's own framing note)."""

from __future__ import annotations

import torch


def distractor_distance_summary(
    target_pos: torch.Tensor,
    distractor_1_pos: torch.Tensor,
    distractor_2_pos: torch.Tensor,
    active_distractor_count: int,
) -> torch.Tensor:
    """(num_envs, 2) tensor: column 0 = Euclidean distance from
    `target_pos` to `distractor_1_pos`, column 1 = Euclidean distance from
    `target_pos` to `distractor_2_pos` - each column HARD-ZEROED (not the
    real distance) whenever its own slot index (0-based: column i needs
    `active_distractor_count > i`) is inactive for the current curriculum
    stage. `target_pos`/`distractor_1_pos`/`distractor_2_pos` are all
    (num_envs, 3) world-frame position tensors (frame choice doesn't affect
    a scalar distance)."""
    num_envs = target_pos.shape[0]
    out = torch.zeros((num_envs, 2), dtype=target_pos.dtype, device=target_pos.device)

    if active_distractor_count >= 1:
        out[:, 0] = torch.linalg.norm(distractor_1_pos - target_pos, dim=-1)
    if active_distractor_count >= 2:
        out[:, 1] = torch.linalg.norm(distractor_2_pos - target_pos, dim=-1)

    return out

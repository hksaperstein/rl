# tasks/franka/shape_observations.py
"""Pure-tensor math for the shape-class one-hot + geometry-descriptor
observation terms (Task 1 of docs/superpowers/plans/2026-07-16-unified-
multi-die-specialist-distillation.md, spec:
docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-
design.md). NO isaaclab/pxr/torch-sim imports beyond plain torch - follows
the same pure-function split tasks/franka/lift_reward.py established for
reward math (that module's own docstring: "mdp.py reads live simulated
state and delegates the actual ... computation to the functions below"),
extended here to observations. tasks/franka/mdp.py's
object_shape_class_onehot/object_geometry_descriptor are thin wrappers that
read a per-env-cfg constant (which shape this training run's object is -
see scope note below) and call into the pure functions here.

Scope note (2026-07-16, controller decision recorded in task-1-report.md):
shape-class and geometry-descriptor are per-ENV-CFG constants, NOT
per-environment-varying values. Every consumer of this task (Task 0's
single-shape specialist env cfg classes, Task 2's specialist training) has
exactly one shape per training run - every one of a run's `num_envs`
parallel environments holds the SAME shape. Building a mechanism for a
single run to mix different shapes across its parallel envs simultaneously
is explicitly out of scope here (that is Task 3's d20 size-domain-
randomization retry and Phase 2/3's actual multi-shape-per-episode unified
policy). Both functions below simply broadcast one constant to
`(num_envs, K)`.

Geometry descriptor (K=1): a single scalar, the Wadell sphericity of each
die's own baked mesh (Wadell, H. 1935, "Volume, shape, and roundness of
quartz particles", Journal of Geology 43(3):250-280 - the standard
convex-hull-based particle-shape descriptor, still in routine use across
geology/powder-metallurgy/granular-mechanics literature):

    psi = pi^(1/3) * (6*V)^(2/3) / A

where V, A are the volume/surface-area of the mesh's own 3-D convex hull.
psi = 1.0 for a perfect sphere, decreasing as a shape deviates from
spherical - naturally distinguishes the die shapes by "how round" they are
(more faces -> more sphere-like -> higher psi), which is exactly the kind
of geometry a policy might need to condition on to generalize its grasp
strategy across shapes. This ratio is scale-invariant (V ~ L^3, A ~ L^2,
so (6V)^(2/3)/A ~ dimensionless), so it was computed once per shape
directly from each baked USD's own native (unscaled) mesh point cloud via
scripts/_diag_shape_sphericity_check.py (headless SimulationApp + pxr mesh
read + scipy.spatial.ConvexHull for the real volume/area of the actual
bevelled/manufacturing-style dice mesh - not an idealized Platonic-solid
formula), then hardcoded below as SHAPE_GEOMETRY_DESCRIPTORS - the same
"one-off diagnostic script measures a real derived constant, hardcode with
derivation documented" convention already established by
tasks/franka/dice_lift_joint_env_cfg.py's own per-shape scale constants.
"""

from __future__ import annotations

import torch

# Canonical one-hot order - K=4 dims, one per supported die shape. Any new
# caller/consumer must agree on this exact order.
SHAPE_CLASSES = ("d8", "d10", "d12", "d20")

# Wadell sphericity per shape, computed via scripts/_diag_shape_sphericity_check.py
# from each shape's own baked mesh convex hull (assets/dice/{shape}_physics.usd) -
# see this module's own docstring for the exact formula/derivation. K=1 dim.
# Live-measured (stage units, native/unscaled mesh - sphericity itself is
# scale-invariant): d8 V=745.021343 A=446.730607 (486 mesh points), d10
# V=1165.970252 A=597.952736 (812 points), d12 V=15330.053595 A=3213.801746
# (1220 points), d20 V=14663.586524 A=3041.874482 (1212 points). Monotonic
# with face count as expected (more faces -> more sphere-like).
SHAPE_GEOMETRY_DESCRIPTORS = {
    "d8": 0.889647,
    "d10": 0.895933,
    "d12": 0.928597,
    "d20": 0.952437,
}

GEOMETRY_DESCRIPTOR_K = 1


def shape_class_onehot(shape_class: str, num_envs: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """One-hot (num_envs, 4) tensor over SHAPE_CLASSES, identical across every
    row (a per-env-cfg constant broadcast to every parallel env - see module
    docstring's scope note, NOT read off live per-env object state)."""
    if shape_class not in SHAPE_CLASSES:
        raise ValueError(f"unknown shape_class {shape_class!r}, expected one of {SHAPE_CLASSES}")
    row = torch.zeros(len(SHAPE_CLASSES), device=device)
    row[SHAPE_CLASSES.index(shape_class)] = 1.0
    return row.unsqueeze(0).expand(num_envs, -1).clone()


def geometry_descriptor_broadcast(
    shape_class: str, num_envs: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """(num_envs, GEOMETRY_DESCRIPTOR_K) tensor, every row equal to
    SHAPE_GEOMETRY_DESCRIPTORS[shape_class] - same per-env-cfg-constant
    broadcast pattern as shape_class_onehot above."""
    if shape_class not in SHAPE_GEOMETRY_DESCRIPTORS:
        raise ValueError(f"unknown shape_class {shape_class!r}, expected one of {tuple(SHAPE_GEOMETRY_DESCRIPTORS)}")
    value = SHAPE_GEOMETRY_DESCRIPTORS[shape_class]
    return torch.full((num_envs, GEOMETRY_DESCRIPTOR_K), value, dtype=torch.float32, device=device)

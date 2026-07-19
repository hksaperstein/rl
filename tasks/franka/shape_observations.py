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
was explicitly out of scope here (that was Task 3's d20 size-domain-
randomization retry's own scope note, which used per-env SIZE variation
only, not shape). `shape_class_onehot`/`geometry_descriptor_broadcast`
below still simply broadcast one constant to `(num_envs, K)` and remain
the single-shape-per-run path every existing env cfg in this file's sibling
module (tasks/franka/dice_lift_joint_env_cfg.py) uses unchanged.

**Extended 2026-07-19 (Task 5, BACKLOG.md's controller decision "(b) single
mixed-population env"):** `FrankaDieLiftJointD12D20MixedEnvCfg` mixes TWO
shapes (d12, d20) across one run's parallel envs via a deterministic
`MultiAssetSpawnerCfg(random_choice=False)` round-robin - the same
per-env-fixed-at-spawn-time mechanism `FrankaDieLiftJointMixedEnvCfg`
already uses for per-env size (not shape) variation. `shape_class_onehot_
per_env`/`geometry_descriptor_per_env` (below) extend this module to that
one additional case: still per-env-cfg-*determined* (known at env-cfg-
construction time from the assets-list order/length, not read off live
per-env simulated state), just no longer a single constant broadcast to
every row - each env's row is picked by the same `index % len(assets)`
formula the live spawner itself uses. This is additive: every existing
single-shape env cfg's observations are produced by the original two
broadcast functions above, completely unchanged.

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


# =====================================================================
# Per-env (mixed-population) variants (Task 5, BACKLOG.md's 2026-07-19
# controller decision "(b) single mixed-population env"):
# FrankaDieLiftJointD12D20MixedEnvCfg (tasks/franka/dice_lift_joint_env_cfg.py)
# spawns a DETERMINISTIC round-robin mix of shapes across its parallel envs
# via `MultiAssetSpawnerCfg(assets_cfg=[...], random_choice=False)`. This is
# the exact same mechanism FrankaDieLiftJointMixedEnvCfg already uses for
# per-env SIZE variation (that class's own docstring, confirmed again by a
# fresh direct source read for this task,
# isaaclab/sim/spawners/wrappers/wrappers.py::spawn_multi_asset:
# `proto_path = proto_prim_paths[index % len(proto_prim_paths)]`, where
# `index` enumerates the env's own prim paths in ascending env-index order,
# i.e. env i is assigned `cfg.assets_cfg[i % len(cfg.assets_cfg)]`).
# Since this assignment is deterministic and known at env-cfg-construction
# time (not something that requires querying live USD/spawner state), these
# functions just replicate that exact `i % len(shapes)` formula directly
# given the SAME shapes-list order/length the env cfg used to build its
# MultiAssetSpawnerCfg - see tasks/franka/mdp.py's
# object_shape_class_onehot/object_geometry_descriptor for the thin wrappers
# that branch to these functions instead of the single-shape broadcast ones
# above, based on whether the env cfg sets `die_shape_classes_per_env`.
# =====================================================================


def shape_class_onehot_per_env(
    shape_classes_per_env: tuple[str, ...], num_envs: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """One-hot (num_envs, 4) tensor over SHAPE_CLASSES, where env i's row is
    `shape_class_onehot(shape_classes_per_env[i % len(shape_classes_per_env)], ...)`
    - i.e. a deterministic round-robin per-env assignment, mirroring
    MultiAssetSpawnerCfg(random_choice=False)'s own `i % len(assets_cfg)`
    formula exactly (see module docstring section above)."""
    if not shape_classes_per_env:
        raise ValueError("shape_classes_per_env must be non-empty")
    for shape_class in shape_classes_per_env:
        if shape_class not in SHAPE_CLASSES:
            raise ValueError(f"unknown shape_class {shape_class!r}, expected one of {SHAPE_CLASSES}")
    table = torch.stack(
        [shape_class_onehot(shape_class, 1, device=device)[0] for shape_class in shape_classes_per_env], dim=0
    )  # (len(shape_classes_per_env), 4)
    idx = torch.arange(num_envs, device=device) % len(shape_classes_per_env)
    return table[idx]


def geometry_descriptor_per_env(
    shape_classes_per_env: tuple[str, ...], num_envs: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """(num_envs, GEOMETRY_DESCRIPTOR_K) tensor, same deterministic
    round-robin per-env assignment as shape_class_onehot_per_env above."""
    if not shape_classes_per_env:
        raise ValueError("shape_classes_per_env must be non-empty")
    for shape_class in shape_classes_per_env:
        if shape_class not in SHAPE_GEOMETRY_DESCRIPTORS:
            raise ValueError(
                f"unknown shape_class {shape_class!r}, expected one of {tuple(SHAPE_GEOMETRY_DESCRIPTORS)}"
            )
    table = torch.stack(
        [geometry_descriptor_broadcast(shape_class, 1, device=device)[0] for shape_class in shape_classes_per_env],
        dim=0,
    )  # (len(shape_classes_per_env), GEOMETRY_DESCRIPTOR_K)
    idx = torch.arange(num_envs, device=device) % len(shape_classes_per_env)
    return table[idx]

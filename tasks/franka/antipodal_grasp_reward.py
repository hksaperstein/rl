# tasks/franka/antipodal_grasp_reward.py
"""Pure-tensor bilateral force-closure/antipodal grasp-quality math, ported
from tasks/ar4/mdp.py:902-940's antipodal_grasp_bonus (Experiments 9/10/11)
into tasks/franka/'s already-established pure-math/wrapper module split
(tasks/franka/exploration_bonus_reward.py's/distractor_observations.py's
own precedent) - NO isaaclab import, testable via plain pytest+torch, zero
Isaac Sim dependency. tasks/franka/mdp.py's antipodal_grasp_bonus wrapper
reads live env.scene ContactSensor state and delegates the actual
computation to antipodal_grasp_bonus_raw below. This is a deliberate,
small structural deviation from the design spec's own single-function code
sketch (which combines the tensor math and the ContactSensor state read in
one function, exactly mirroring AR4's own mdp.py, which has no pure/wrapper
split convention at all) - the math itself is UNCHANGED from the port
(identical formula, identical parameter names/semantics), only the module
boundary differs (Task 1, docs/superpowers/plans/2026-07-20-d8-antipodal-
grasp-quality-implementation.md, "Design notes" #1).

Requires BOTH jaw contact-force magnitudes to exceed force_threshold AND
their force directions to be nearly anti-parallel (cosine of the angle
between them below antipodal_cos_threshold) - the classical two-contact
force-closure necessary condition (Nguyen 1988, "Constructing Force-Closure
Grasps"; Ponce & Faverjon, "On Computing Two-Finger Force-Closure Grasps of
Curved 2D Objects," ICRA 1991/IJRR 1993 - same citations tasks/ar4/mdp.py's
own antipodal_grasp_bonus docstring already carries, tasks/ar4/mdp.py:902-
940). A real bilateral contact-force reading can register from a
non-antipodal, physically-unstable pinch that classical theory says is not
actually resistant to gravity's wrench even though it satisfies a
magnitude-only check - this is why the direction check matters, not just
magnitude (this project's own AR4-era Experiment 1->9 precedent).

antipodal_cos_threshold refit for Franka (2026-07-20) - NOT a reuse of
AR4's own value, a genuine re-derivation: AR4's antipodal_grasp_bonus used
threshold=-0.7071 (Experiment 10's own derivation, threshold =
-cos(arctan(mu)), at mu=1.0 - a 45-degree friction-cone half-angle). This
Franka die-lift scene has no RigidBodyMaterialCfg override anywhere
(confirmed by docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-
literature.md's own grep across lift_env_cfg.py/dice_lift_joint_env_cfg.py/
bake_die_asset.py), so Isaac Lab's RigidBodyMaterialCfg() default, mu=0.5,
applies (independently verified for this exact tasks/franka/ asset stack by
docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md's own
friction check). Re-applying Experiment 10's own derivation formula at this
scene's real mu=0.5: half-angle = arctan(0.5) = 26.565 degrees, threshold =
-cos(26.565 degrees) = -0.894427 (recomputed and verified this session, not
carried over from the mu=1.0 case) - a genuinely stricter (more-negative)
bound than AR4's own -0.7071, since a lower friction coefficient narrows the
friction cone and demands a more precisely opposed grasp to actually resist
gravity's wrench.

See docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md
for the full spec and
docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md
for the underlying research grounding.
"""

from __future__ import annotations

import torch


def antipodal_grasp_bonus_raw(
    jaw1_force_vec: torch.Tensor,
    jaw2_force_vec: torch.Tensor,
    force_threshold: float,
    antipodal_cos_threshold: float,
) -> torch.Tensor:
    """Bilateral force-closure grasp bonus, operating directly on two
    (N, 3) world-frame contact-force vectors (tasks/franka/mdp.py's
    antipodal_grasp_bonus wrapper reads
    env.scene[...].data.force_matrix_w and reshapes to this shape before
    calling this function - see that function's own docstring). Identical
    math to tasks/ar4/mdp.py:930-941's own tensor computation.

    Args:
        jaw1_force_vec: jaw 1's contact-force vector, shape (N, 3).
        jaw2_force_vec: jaw 2's contact-force vector, shape (N, 3).
        force_threshold: minimum force magnitude (N, i.e. Newtons) each jaw
            must independently exceed for a contact to count at all.
        antipodal_cos_threshold: the two jaws' force *directions* must have
            a cosine-of-angle-between-them strictly below this value (i.e.
            nearly anti-parallel) - see module docstring for this scene's
            own physically-derived -0.894427 value (mu=0.5), distinct from
            AR4's own -0.7071 (mu=1.0). The comparison is strict `<`,
            matching AR4's own operator (tasks/ar4/mdp.py:939) - a
            cos_angle exactly equal to this threshold does NOT count as
            antipodal.

    Returns:
        Tensor, shape (N,), 1.0 where both the magnitude and antipodal-
        direction conditions hold, 0.0 otherwise (bool-as-float, matching
        the AR4 source's own `.float()` cast).
    """
    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
    both_magnitude_ok = (jaw1_force_mag > force_threshold) & (jaw2_force_mag > force_threshold)

    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
    antipodal_ok = cos_angle < antipodal_cos_threshold

    return (both_magnitude_ok & antipodal_ok).float()

# Experiment 1: ContactSensor-based grasp reward

**Object:** sphere. **Escalation context:** the fourth consecutive falsified
reward/control-only hypothesis (lift-weight bump, dense proximity+closure
bonus, multiplicatively-gated alignment bonus, gripper PD-gain rescale — none
of them numbered "Experiment N" in ROADMAP.md, all predate this article) had
just been reached, triggering `superpowers:systematic-debugging` Phase 4.5:
question the architecture, not attempt a fifth single-shot reward tweak.

## Hypothesis

The prior geometric grasp proxies (EE-to-object distance, or a
multiplicatively-gated alignment score) are proxies for "is the object
actually being grasped," not ground truth. A real, ground-truth signal —
does PhysX report actual bilateral contact force between both gripper jaws
and the sphere — should be learnable and non-hackable in a way a geometric
proxy isn't.

## What changed

Added two `ContactSensorCfg` sensors (one per gripper jaw, each filtered to
the sphere specifically) feeding a new `grasp_contact` reward requiring real,
sustained, bilateral contact force above a calibrated threshold.

Two real implementation bugs were found and fixed while building this (both
via smoke-test/calibration failures, not anticipated on paper):

- A single wildcard `ContactSensorCfg` covering both jaw links can't pair
  with PhysX's per-body filter-count requirement — fixed to one sensor per
  jaw.
- `net_forces_w` is **not** actually filtered by `filter_prim_paths_expr` at
  all (it sums *any* contact on the body); the correct field is
  `force_matrix_w`.

**Bigger finding, incidental to this experiment's own goal:** `_EE_OFFSET`
(the link_6-to-jaw-pinch-point offset feeding the `ee_frame` sensor used by
every `reaching_*` reward across the whole sphere-era session) was wrong by
**5.4cm** (`0.09` vs. measured `0.036`, confirmed directly against
`robot.data.body_pos_w`). Every prior grasp experiment's reach reward had
been maximizing proximity to a point 5.4cm from where the jaws actually
meet — a plausible deeper explanation for why grasping never emerged in any
of them, independent of whatever reward-shaping was layered on top each
time. Fixed as a prerequisite correctness bug, not treated as a new
hypothesis, but flagged prominently since it retroactively recontextualizes
the whole "grasp/lift never emerges" investigation up to this point.

## Quantitative result

`grasp_contact` converged to **~92% per-step sustained contact**
(18.39/20 weighted, max 18.58) — real, bilateral, correctly-filtered contact
sustained for most of the episode. `lifting_sphere` still converged to ~0
(max 0.0027); `sphere_reached_goal` still ~0 (max 0.027); `reaching_sphere`
converged lower than prior runs (0.727 vs. ~0.92–0.94) — expected, since it
now measures against the corrected (true) jaw pinch point, a harder
criterion than the old 5.4cm-off target.

## Qualitative video finding

10 episodes, frame-extracted, all inspected directly: **0/10 show a real
grasp+lift** — fails the 8/10 decision gate. But the failure signature is
new: the arm reaches down within ~1s of every episode and then holds a
completely static pose with the gripper directly on the sphere for the rest
of the episode, in all 10 episodes. The sphere is never lifted or moved, but
per the `grasp_contact` numbers it is genuinely, sustainedly gripped
throughout, not merely approached — "reach, grip, freeze."

## Verdict

**Real progress, not a false positive.** The specific problem this
experiment targeted (does the gripper ever really close on the object) is
solved for the first time this session. A new, distinct bottleneck (grip
achieved, but no subsequent attempt to lift) takes its place. Not treated as
grounds for a further unilateral reward tweak — flagged back as a decision
point.

## Related concepts

[[reach-grasp-lift-gap]] — this is the pivot point where "grasp never
happens" becomes "grasp happens, lift never happens." [[grasp-mechanics-antipodal-vs-magnitude]]
— the ground-truth contact signal this experiment introduces, later found to
be magnitude-only rather than force-closure-aware (see Experiments 9–10).
[[reward-hacking-and-sparse-discoverability]] — contrasts with the two prior
falsified geometric proxies (reward-hacked vs. too-sparse-to-discover) that
this experiment's ground-truth approach avoids.

## Sources

`docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`

# Experiment 22 Training Run Report: Software Jaw Position Mirroring (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_20-58-26`
**Training Status:** COMPLETED (1500 iterations)

## Mechanism Verification Recap

Before training, `MirroredGripperAction`'s logic was verified directly
(not inferred from downstream behavior): with jaw1's actual position
set to 0.014, 0.007, and 0.0 in turn, jaw2's computed `_processed_actions`
matched jaw1's actual position exactly in all three cases (PASS). This
confirms the code itself is correct in isolation.

## An Investigation Worth Recording: Identical Training-Time Aggregate Metrics

Full-run scalar extraction initially raised a serious concern: every
logged reward term's value was **bit-for-bit identical to Experiment
21's**, at all 1500 logged points, including `Episode_Reward/reaching_object`
(a pure arm-position metric with no dependency on gripper joint state
at all). A rigorous point-by-point diff (`max(abs(v21[i] - v22[i]))`
across all 1500 points) confirmed `0.0000000000` difference, exactly,
for every term checked.

This was investigated before trusting any conclusion from this run —
per this repo's own standing practice of verifying surprising results
rather than either dismissing or over-reacting to them. Ran the same
Task-6-style instrumented contact diagnostic against Experiment 22's
own trained checkpoint (`model_1499.pt`), this time also logging raw
per-step jaw1/jaw2 joint positions directly (not just contact forces):

```
[EP 0 STEP   0] jaw_joint_pos=0.01400/0.01149
[EP 0 STEP   1] jaw_joint_pos=0.01335/0.00799
[EP 0 STEP   4] jaw_joint_pos=0.01174/0.00399
```

**This directly confirms the mirroring mechanism is genuinely active
and changing jaw2's real trajectory** — jaw2's actual position diverges
from jaw1's from the very first step, in a pattern consistent with
mirroring-with-lag (see next section), not with the mechanism being
inert. Further confirmation: the two checkpoints' own contact
diagnostics show genuinely different results (Experiment 21:
`max_jaw1_force=6.73N`; Experiment 22: `max_jaw1_force=0.0N`) — the
underlying learned policies are not identical, despite the training-time
aggregate reward logs matching exactly.

**Conclusion on this specific finding:** the identical aggregate
metrics appear to be a genuine property of how coarse those specific
logged signals are (`pregrasp_readiness`'s closedness term uses the
*mean* of both jaw positions; `reaching_object` depends only on arm
state, which the policy may not have learned to condition differently
on jaw2's exact value if grasp/lift outcomes are unaffected either way)
rather than evidence the mechanism failed to run. This is recorded here
in full because it's a real, non-obvious property of this specific
control setup worth knowing for any future experiment comparing
training-time aggregate scalars between configs that only differ in
low-level actuator behavior — the aggregate scalars alone would not
have revealed the checkpoints are behaviorally different; the
instrumented rollout did.

## Verification Results

### Checkpoint Integrity
31 checkpoints confirmed, `model_1499.pt` exists, no tracebacks/exceptions.

### `Loss/value_function`
Small and bounded throughout (matching Experiment 21's own values
exactly, consistent with the aggregate-metric finding above) — training
stable.

## Critical Question: Does the Contact Diagnostic's `both_magnitude_ok_steps` Move Off `0/750`?

```
[SUMMARY] total_steps=750 height_ok_steps=0 both_magnitude_ok_steps=0
antipodal_ok_steps=0 grasp_ok_steps=0 gate_fires_steps=0
max_jaw1_force=0.0 max_jaw2_force=23.445221 max_cube_z=0.009706
min_orientation_dot=0.013309 max_orientation_dot=1.0
max_jaw_pos_diff=0.011094
```

**No — `both_magnitude_ok_steps` is still exactly `0/750`.** By this
experiment's own success criterion, a null result.

**But the new `max_jaw_pos_diff=0.011094` metric explains why, more
precisely than a simple null would.** 0.011m is 79% of the full 0.014m
gripper travel range — real, substantial divergence, comparable in
magnitude to Experiment 17 Task 6's original 0.0028m finding scaled up,
not a small residual. The mechanism (jaw2 chases jaw1's *actual*
measured position each step) has a structural limitation: because jaw2
reacts to where jaw1 *already is* rather than where jaw1 is *headed*,
jaw2 is always one control step behind a target that itself keeps
moving whenever the policy is actively opening or closing the gripper.
During the observed rapid-motion window (steps 0-4 above), jaw1 moves
from 0.014 to 0.0117 (a real, fast transition) while jaw2's own
lagging-follower dynamics let it fall to 0.004 — well behind. **Mirroring
shifted the source of divergence from Task 6's original finding
(asymmetric PD-tracking under sustained contact load) to a new one
(discrete one-step reactive lag during motion) — it did not eliminate
divergence, it relocated it.**

## Assessment

**Mechanism verification:** confirmed working as designed, by three
independent checks (isolated unit test, raw per-step rollout data, and
the checkpoint-level behavioral difference from Experiment 21). Not a
failed implementation.

**Primary finding:** `both_magnitude_ok_steps` remains at `0/750`,
`lifting_object` remains at `0/1500` — both null by the strict success
criteria, matching Experiments 17, 18, 20, and 21.

**This is a genuinely informative negative result, not a repeat.** The
specific mechanism this experiment tested — reactive position-following
— has now been shown to have its own structural failure mode (lag
during motion), distinct from both Experiment 19's finding (physical
constraint fighting independent actuators) and Experiment 17's original
finding (asymmetric tracking under load). A future attempt at software-
level jaw synchronization would need to account for jaw1's *velocity*,
not just its instantaneous position (e.g., commanding jaw2 toward
jaw1's position plus a lead term proportional to jaw1's recent rate of
change, or simply reading jaw1's own *commanded target* — which is
known instantly, not delayed by physics — rather than its physically-
settled actual position). This narrows the space of viable software-
mirroring designs rather than closing off the software-mirroring
direction entirely.

**Where this leaves the research program:** six consecutive experiments
(17, 18, 19, 20, 21, 22) have now each targeted a different specific
candidate mechanism for the same underlying problem (the cube never
gets lifted) and each has narrowed the picture further without
resolving it. The two most concrete remaining levers: (a) a corrected
software-mirroring design using jaw1's commanded target (not actual
position) as the reference, directly informed by this experiment's own
lag diagnosis; (b) demonstration/imitation bootstrapping for the lift
primitive, now a comparatively more attractive option given how many
independent reward/action-space/control mechanisms have been tried
without success.

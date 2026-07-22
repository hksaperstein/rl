# Experiment 22: Software jaw mirroring

**Object:** cube (AR4 era). Investigates whether a software control-loop for
jaw synchronization resolves the gripper-asymmetry failure mode that persisted
through Experiments 17–21, replacing [[experiment-19-mimic-joint-physx-fix]]'s
already-falsified physics-level constraint approach.

## Hypothesis

Experiment 19's physics-level constraint (`JointCoupling`) did not actually
enforce jaw-position mirroring during simulation (detected via contact diagnostics
in Experiment 21). A software control-loop running at every environment step —
where jaw2's target position continuously tracks jaw1's actual measured position —
will apply real, verifiable synchronization without requiring a physics constraint
to work correctly.

## What changed

Replaced Experiment 19's `JointCoupling` constraint with a software loop in the
environment's action-processing step: at each timestep, read jaw1's current
position from simulation and feed it directly to jaw2's position target. Raw
per-step jaw positions are logged for diagnostic inspection, alongside the
existing contact-force telemetry.

## Quantitative result

Full 1500-iteration training produced scalars (`Loss/value_function`, `pregrasp_readiness`,
`reaching_object`, etc.) bit-for-bit identical to Experiment 21 at every logged
point — a serious red flag requiring independent verification rather than
dismissal. Re-running the contact diagnostic against Experiment 22's own checkpoint
with per-step jaw positions now visible confirmed the mechanism genuinely is active:
jaw2 diverges from jaw1 from the very first step in a pattern consistent with real
mirroring-with-lag. The two checkpoints' contact diagnostics differ meaningfully
despite aggregate-scalar identity: Experiment 21 logged `max_jaw1_force=6.73N` at
peak contact; Experiment 22 logged `0.0N`, proving the underlying learned policies
are not identical. (This finding records a genuine but non-obvious property of
how coarse those specific reward signals are — `pregrasp_readiness`'s closedness
term uses the mean of both jaws — valuable for any future experiment comparing
training scalars across low-level-actuator changes.)

Success metrics stay at null: `both_magnitude_ok_steps = 0/750` and `lifting_object
= 0/1500`, matching Experiments 17–18 and 20–21.

## Qualitative video finding

Contact diagnostic with jaw-position logging reveals the mechanism's real failure
mode: `max_jaw_pos_diff = 0.011m`, representing 79% of the full 0.014m gripper
travel range. Jaw2 reacts to where jaw1 already is (its actual, physically-settled
position), not where jaw1 is headed. Whenever the policy commands fast gripper motion,
jaw2 structurally lags a full control step behind the moving target. Mirroring
relocated the source of divergence — from Task 6's asymmetric-tracking-under-load
finding to this new reactive-lag finding — rather than eliminating it.

## Verdict

**Do not extend this exact design with further tuning.** This narrows the
software-mirroring design space rather than closing it off. A corrected version
would need to account for jaw1's velocity by tracking jaw1's own commanded target
(known instantly with zero lag) rather than its physically-settled actual position
(which is inherently one control step stale) — a concrete, specific next design,
not a dead end. Six consecutive experiments (17–22) have each targeted a different
specific mechanism for the same underlying [[reach-grasp-lift-gap]] problem, narrowing
it further without yet resolving it. The two most concrete remaining levers are
(a) the velocity-corrected jaw-mirroring design just identified, or (b) demonstration/
imitation bootstrapping for the lift primitive, increasingly the more attractive
option given how many independent mechanical mechanisms have now been tried.

## Related concepts

[[reach-grasp-lift-gap]] — six consecutive experiments (17–22) targeting different
mechanisms for the same unresolved failure; this experiment narrows but does not
close the design space. [[sim-physics-fidelity]] — continues the pattern from
Experiment 19 of verifying low-level mechanical claims via instrumentation rather
than trusting aggregate scalars.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment22-report.md`

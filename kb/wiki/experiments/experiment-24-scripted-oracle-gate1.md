# Experiment 24 Gate 1: Scripted-oracle demonstration bootstrapping

**Object:** cube. Demonstration/imitation bootstrapping as the next concretely justified direction after [[experiment-23-warmstarted-residual-rl]] exhausted the reward/action-space-engineering family.

## Hypothesis

A non-learned, reactive-differential-IK oracle following a pre-computed 5-waypoint pick-and-place path can demonstrate stable, repeatable behavior suitable for bootstrapping behavior-cloning pretraining, which would then warm-start an RL finetune toward the full task.

## What changed

Built `scripts/oracle_rollout.py`, a non-learned reactive controller using differential inverse kinematics to follow a 5-waypoint pick-and-place trajectory without human teleoperation. The oracle computes receding-horizon IK corrections at each environment step to steer the end-effector toward successive waypoints.

## Quantitative result

The oracle stalls before reaching the grasp waypoint in the overwhelming majority of sampled episodes: only ~1/24 episodes advanced past waypoint 0. When stalls occur, the end-effector position freezes bit-identical for 50+ consecutive steps, indicating the control loop reaches a fixed point and does not recover. This diagnostic behavior is independent of episode selection — the stall signature is repeatable and widespread, not rare or boundary-case-specific.

## Qualitative diagnostic evidence trail

Three architecturally distinct fixes were attempted:

1. **Cartesian-space escape-perturbation** (ported from `classical_pickplace_demo.py`): verified firing correctly and reaching the commanded targets, but produced zero effect on actual end-effector motion recovery.

2. **Joint-space perturbation**: verified reaching the commanded target configurations in joint space, but similarly had zero effect on actual end-effector position.

3. **Integral/accumulated-error pursuit correction**: at conservative gain, produced no effect; when gain was strengthened to pursue more aggressively, the arm actively diverged away from the target — the only attempted fix that made behavior demonstrably worse when strengthened.

**Root-cause diagnostics independently re-verified** (not accepted from first pass alone): torque saturation never clips (14.1 N·m peak vs. 20 N·m limit), hard joint-limit clearance is ~13° (substantial margin), contact and collision forces remain at 0.0 N throughout, and Jacobian rank collapse is ruled out (smallest singular value plateaus ~0.15, not ~0). Evidence instead points to a genuine fixed point of the receding-horizon control loop in a poorly-conditioned kinematic region, where the linearized IK correction stops reliably pointing toward the goal.

## Verdict

**FAIL.** The oracle does not produce a usable demonstration trajectory. This finding independently echoes [[experiment-20-vertical-orientation-lock]]'s own prior conclusion — reached through a structurally different investigation path (waypoint-pursuit stall vs. orientation-holding drift) — that single-Newton-step-per-`env.step()` differential IK is not a stable mechanism on this arm. The fixed-point-in-poorly-conditioned-region signature also aligns with the parallel classical-IK investigation thread documented in [[reach-grasp-lift-gap]] and [[sim-physics-fidelity]], where similar converge-partway-then-stall behavior emerged via completely independent scripts and mechanisms. Gate 2's implementation plan was not written per the original plan's scope note for a FAIL verdict.

## Related concepts

[[reach-grasp-lift-gap]] — the parallel classical-IK investigation thread independently identified the same fixed-point stall signature via a different investigation path. [[sim-physics-fidelity]] — physics-fidelity work relevant to understanding the poorly-conditioned kinematic region. [[experiment-20-vertical-orientation-lock]] — prior experiment whose own IK-instability finding this result directly echoes. [[experiment-23-warmstarted-residual-rl]] — the prior experiment whose null result motivated this direction.

## Sources

`docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md`, `scripts/oracle_rollout.py`

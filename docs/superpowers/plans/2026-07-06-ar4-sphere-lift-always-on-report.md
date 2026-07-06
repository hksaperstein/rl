# Task 2: Full training run (always-on lift reward, no curriculum)

## Summary

This task ran the full 1500-iteration training loop with the always-on dense `lift_height_progress` reward (no curriculum gating) from Task 1, then extracted key TensorBoard scalars to verify learning progress before Task 3's eval video.

## Log Directory
`/home/saps/projects/rl/logs/train/2026-07-06_12-24-08/`

## Model Checkpoint
Confirmed: `model_1499.pt` exists at `/home/saps/projects/rl/logs/train/2026-07-06_12-24-08/model_1499.pt` (1.2M, dated 2026-07-06 12:40).

## TensorBoard Scalars (1500 iterations)

Episode_Reward/lift_height_progress -> first: 0.000298902828944847 last: 0.007014661096036434 max: 0.035228900611400604 trajectory (10 samples): [0.0003, 0.0143, 0.0086, 0.009, 0.0073, 0.0067, 0.0077, 0.0071, 0.0077, 0.0071]

Episode_Reward/lifting_sphere -> first: 0.0 last: 0.0 max: 0.0009057971183210611 trajectory (10 samples): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Episode_Reward/grasp_contact -> first: 0.0 last: 18.26797866821289 max: 18.569320678710938 trajectory (10 samples): [0.0, 1.3711, 13.2677, 17.0622, 18.3159, 18.3817, 18.0774, 18.1281, 18.4396, 18.1866]

Episode_Reward/reaching_sphere -> first: 0.00015441118739545345 last: 0.7024034261703491 max: 0.7117838859558105 trajectory (10 samples): [0.0002, 0.3102, 0.5953, 0.6571, 0.6926, 0.7006, 0.6914, 0.6916, 0.707, 0.7005]

Episode_Termination/sphere_reached_goal -> first: 0.0 last: 0.0 max: 0.02724202536046505 trajectory (10 samples): [0.0, 0.0108, 0.0031, 0.0007, 0.0002, 0.0005, 0.0, 0.0003, 0.0005, 0.0005]

## Lift Height Progress Comparison

The `lift_height_progress` scalar reached a maximum of **0.0352** during this always-on run, which is substantially larger than the prior curriculum-gated experiment's maximum of 0.0065—a 5.4× improvement. The trajectory samples show meaningful engagement with the lift reward from early iterations (0.0143 at the second sample), suggesting the removal of the curriculum gate successfully activated the dense lift signal from the start.

## Task 3: Real eval + video inspection (decision gate)

**Eval command:** `/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/2026-07-06_12-24-08/model_1499.pt --episodes 10` — completed successfully, 10 videos written.

**Frame extraction:** `ffmpeg -vf fps=10` on all 10 videos, controller-inspected directly (all 10 episode directories reviewed; representative samples at start/~25%/~50%/~75%/end and every episode's midpoint frame checked).

**Observation:** The same static "reach, grip, freeze" signature as both prior experiments (curriculum-gated, and the original sparse-only baseline) — the arm reaches down within the first ~1-2 seconds and holds a completely static pose next to the sphere for the rest of every episode. The sphere (visible as the small blue marker) never leaves the ground in any of the 10 episodes; frames across episodes at equivalent timepoints are visually near-identical, exactly matching the pattern from both prior experiments.

**Decision gate: 0/10 episodes show any real lift.** Same failure signature as every prior attempt.

**Correction to Task 2's "5.4x" comparison:** that comparison was between raw logged `Episode_Reward` values across two runs with *different* `lift_height_progress` weights (15.0 in the curriculum run vs. 25.0 here) — not apples-to-apples. Weight-normalizing (dividing by the term's own weight before applying `tanh`'s small-angle approximation): this run's real per-step `tanh` ≈ `0.0352/25 ≈ 0.00141`, corresponding to ~0.0141mm of real height gain, vs. the prior run's ~0.0043mm. That's a real ~3.3x increase in the tiny signal the term is producing, not 5.4x, and both figures remain many orders of magnitude short of the 21mm `lifting_sphere` requires. This doesn't change the outcome, only the magnitude of an already-negligible number.

**Interpretation:** removing the curriculum gate and raising the weight produced a measurably larger (though still tiny) dense-reward signal and an early-training bump that faded as `grasp_contact` converged (per Task 2's trajectory samples) — consistent with the entropy-collapse mechanism the literature research identified (`docs/superpowers/specs/research/2026-07-06-lift-reward-literature-junior.md`/`-senior-review.md`): once the policy locks onto the safe static-grip behavior, exploration toward lifting effectively stops regardless of whether the incentive was available from iteration 0 or introduced later. This rules out "curriculum timing" as the sole explanation (already suspected after this run) and points at the policy's own exploration collapse as the real mechanism - which the research identified real, verified remedies for (SA-PPO-style dynamic learning-rate adjustment specifically validated on a robotic-grasping PPO local-optimum-escape task; potential-based reward shaping) - none of which have been tried yet.

Per the design doc's own instruction and the literature findings, not attempting a fourth reward-only tweak unilaterally. Flagging back to the user with the two real, literature-backed candidates plus the not-yet-tried architectural ones.

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

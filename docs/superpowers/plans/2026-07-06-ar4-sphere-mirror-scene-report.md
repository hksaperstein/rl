# AR4 Sphere Mirror-Scene Full Training Report

## Task 4: Full 1500-iteration training run (4096 envs)

### First attempt — INVALID, discarded (sign-convention bug)

**Log directory:** `logs/train/2026-07-06_15-44-58/` (do not use for eval).

`Episode_Reward/stillness_penalty` grew to **+1.3** over training
(trajectory: `[0.0, 0.000135, 0.000147, 0.000204, 0.000376, 0.005577,
0.843786, 1.115782, 1.211107, 1.185924]`) — impossible for a true penalty
term. Root cause: `stillness_penalty`'s function body already returns the
signed value (`-1.0` when triggered), but its `RewardsCfg` registration
used `weight=-2.0`. `RewardManager.compute()` computes `func(...) *
weight * dt`, so the double negative turned the intended penalty into a
`+2.0*dt` reward for the exact stay-still-after-grasp behavior this term
exists to punish. Fixed to `weight=2.0` (commit `e7742b5`); full
derivation in `ROADMAP.md`. This run's checkpoint and scalars are invalid
and must not be used for the eval/decision-gate step (Task 5).

### Second attempt — with corrected `weight=2.0`

Not yet run. A fresh training run with the corrected reward weight is
required before Task 5 (eval + decision gate) can proceed. This section
will be filled in with real data (log directory, verified
`model_1499.pt`, and the actual pulled TensorBoard scalar trajectories)
once that run completes — do not fill in numbers here ahead of time.

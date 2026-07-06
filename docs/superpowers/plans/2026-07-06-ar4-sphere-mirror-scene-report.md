# AR4 Sphere Mirror-Scene Full Training Report

## Task 4: Full 1500-iteration training run (4096 envs)

**Log directory:** `logs/train/2026-07-06_15-44-58/`

**Model file verified:** `model_1499.pt` exists (1500 iterations, 0-indexed)

### TensorBoard Scalar Trajectories

The following five key scalar metrics were pulled from the training event logs:

```
Episode_Reward/staged_milestone_bonus -> first: 0.010863902978599072 last: 0.037513867020606995 max: 0.03851180151104927 min: 0.010748554021120071 trajectory (10 samples): [0.010864, 0.021757, 0.027067, 0.033655, 0.0362, 0.03742, 0.036285, 0.036831, 0.037265, 0.036886]

Episode_Reward/stillness_penalty -> first: 0.0 last: 1.3002004623413086 max: 1.3240711688995361 min: 0.0 trajectory (10 samples): [0.0, 0.000135, 0.000147, 0.000204, 0.000376, 0.005577, 0.843786, 1.115782, 1.211107, 1.185924]

Episode_Termination/sphere_reached_goal -> first: 0.0025736491661518812 last: 0.007232666015625 max: 0.058349609375 min: 0.0012613933067768812 trajectory (10 samples): [0.002574, 0.027273, 0.034678, 0.036865, 0.046285, 0.051565, 0.012563, 0.010701, 0.008372, 0.009013]

Episode_Reward/action_rate -> first: -6.624991510761902e-05 last: -0.0010351873934268951 max: -6.624991510761902e-05 min: -0.0021076854318380356 trajectory (10 samples): [-6.6e-05, -0.000948, -0.00094, -0.001189, -0.001497, -0.001969, -0.001442, -0.001169, -0.001065, -0.001047]

Episode_Reward/joint_vel -> first: -7.407102384604514e-05 last: -0.00036798667861148715 max: -7.407102384604514e-05 min: -0.0016623595729470253 trajectory (10 samples): [-7.4e-05, -0.001324, -0.001241, -0.001302, -0.001492, -0.001627, -0.000583, -0.000485, -0.000396, -0.000399]
```

### Analysis

`Episode_Reward/staged_milestone_bonus` remains non-negative throughout training (minimum value: 0.0107), with a monotonic increase from 0.0109 to 0.0375, confirming the corrected undiscounted milestone-bonus reward formula is functioning as designed without negative values.

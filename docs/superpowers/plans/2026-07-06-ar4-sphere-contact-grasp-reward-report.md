# AR4 Sphere Contact-Grasp Reward Implementation Report

## Task 4: Full training run

**Log directory:** `logs/train/2026-07-06_08-38-09/`

**Final checkpoint:** `model_1499.pt` (verified present)

**Wall-clock duration:** ~16 minutes

**TensorBoard scalars:**
```
Episode_Reward/grasp_contact -> first: 0.0 last: 18.39236831665039 max: 18.58135986328125
Episode_Reward/lifting_sphere -> first: 0.0 last: 0.0 max: 0.0026785717345774174
Episode_Reward/reaching_sphere -> first: 0.00015441118739545345 last: 0.7274736166000366 max: 0.7340885996818542
Episode_Termination/sphere_reached_goal -> first: 0.0 last: 0.000244140625 max: 0.0268656425178051
```

**grasp_contact interpretation:** The grasp_contact reward moved meaningfully off 0, rising from 0.0 at iteration 0 to 18.39 at iteration 1500 (max: 18.58), indicating the reward signal is active and responding during training.

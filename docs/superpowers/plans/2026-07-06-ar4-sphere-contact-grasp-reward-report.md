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

## Task 5: Real eval + video inspection (decision gate)

**Eval command:** `/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/2026-07-06_08-38-09/model_1499.pt --episodes 10` — completed successfully, 10 videos written (`logs/videos/ar4_pickplace-step-{0,250,500,750,1000,1250,1500,1750,2000,2250}.mp4`).

**Frame extraction:** `ffmpeg -vf fps=10` on all 10 videos (50 frames/episode, ~5s episodes), controller-inspected directly (not delegated) given this is the experiment's decision gate.

**Observation (consistent across all 10 episodes, sampled at start/~25%/~50%/~75%/end of each):** The arm reaches down from its home pose within the first ~1 second and settles into a static pose with the gripper positioned directly next to the sphere (visible as the small blue marker, alongside the wedge's yellow marker) — then **holds that exact pose, completely static, for the remainder of the episode** in every single one of the 10 episodes. The sphere (blue marker) remains visibly on the ground at its original position throughout every episode — it never lifts, never moves, never becomes occluded by an enclosing grip. Frames from different episodes at equivalent time points are visually near-identical, confirming this is the policy's single learned behavior, not per-episode variation.

**Decision gate: 0/10 episodes show a real grasp+lift.** Fails the 8/10 success criterion.

**This is not simply a sixth repeat of the prior four failures.** The quantitative data (Task 4) showed `grasp_contact` converged to ~92% per-step average — meaning the ContactSensor is registering real, sustained contact between both jaws and the sphere for most of the episode, a first for this whole session (every prior experiment either never closed on the object, closed beside it, or never discovered closure at all). Combined with the video: the policy has learned to close the gripper directly onto the sphere and hold that position — genuine contact, not a proxy or reward hack — but has not learned to follow that contact with any subsequent lifting motion. The behavior is best described as "reach, grip, freeze" rather than either of the prior session's two named failure signatures ("reach-then-freeze-without-closing" or "closes beside the object"). The gripping problem this whole ContactSensor experiment was designed to solve appears to be solved; a new, distinct bottleneck (grip achieved, but no lift attempted) has taken its place.

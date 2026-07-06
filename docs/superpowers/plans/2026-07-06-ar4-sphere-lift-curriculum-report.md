# AR4 sphere lift-curriculum reward — Results Report

## Task 2: Full training run

**Command run:**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

**Log directory:** `logs/train/2026-07-06_10-52-19/`

**Wall-clock duration:** ~16 minutes 23 seconds (matches the prior similar
experiment's ~16-minute run). `model_1499.pt` confirmed present (1500
iterations, 0-indexed).

**TensorBoard scalars** (pulled via `event_accumulator` on
`logs/train/2026-07-06_10-52-19/events.out.tfevents.*`):

```
Episode_Reward/lift_height_progress -> first: 0.0 last: 0.004720306023955345 max: 0.006482542958110571 at_iter_699: 0.0 at_iter_701: 0.0006173293804749846
Episode_Reward/lifting_sphere -> first: 0.0 last: 0.0 max: 0.0026785717345774174 at_iter_699: 0.0 at_iter_701: 0.0
Episode_Reward/grasp_contact -> first: 0.0 last: 18.41428565979004 max: 18.58443832397461 at_iter_699: 17.839996337890625 at_iter_701: 18.021011352539062
Episode_Reward/reaching_sphere -> first: 0.00015441118739545345 last: 0.7290780544281006 max: 0.7348679900169373 at_iter_699: 0.7155972719192505 at_iter_701: 0.7212046384811401
Episode_Termination/sphere_reached_goal -> first: 0.0 last: 0.0 max: 0.0268656425178051 at_iter_699: 0.0012613933067768812 at_iter_701: 0.0006815592641942203
```

**Curriculum-gate check:** `lift_height_progress` is `0.0` at iteration 699
(just before the switch, weight still 0.0) and becomes nonzero
(`0.0006173293804749846`) at iteration 701 (weight now 15.0), confirming the
curriculum gate fired at the intended point (weight change from 0.0 to 15.0
between step 700 and 701 in `env.common_step_counter` terms as configured in
Task 1).

`lifting_sphere` did not move meaningfully off 0 after the curriculum switch:
it is `0.0` at both iteration 699 and 701, remains `0.0` at the final
iteration (`last: 0.0`), and its max over the whole run (`0.0026785717345774174`)
is negligible — the curriculum gate firing did not by itself produce a
sustained binary lift signal in this run.

## Task 3: Real eval + video inspection (decision gate)

**Eval command:** `/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/2026-07-06_10-52-19/model_1499.pt --episodes 10` — completed successfully, 10 videos written.

**Frame extraction:** `ffmpeg -vf fps=10` on all 10 videos, controller-inspected directly.

**Observation:** The same static "reach, grip, freeze" signature as the prior ContactSensor experiment — the arm reaches down and holds a completely static pose next to the sphere for the rest of the episode, across every sampled episode (0, 250, 500, 750, 1250, 1500, 1750). In two episodes (750, 1250) the sphere briefly appeared to vanish from view in a single sampled frame; checking adjacent frames in the same episode showed the sphere reappearing at the identical ground-level position next to the gripper — a viewing-angle occlusion artifact (the gripper body briefly blocking line-of-sight to the small sphere at that specific camera angle/pose), not a real lift. This is consistent with the quantitative data: `lift_height_progress`'s max value over the entire run (0.0065) corresponds to roughly 0.065mm of real height gain via `tanh`'s small-angle approximation — many orders of magnitude short of the 21mm needed to cross `lifting_sphere`'s threshold, and far too small to produce a real, sustained visual occlusion.

**Decision gate: 0/10 episodes show any real lift.** Same failure signature as the ContactSensor experiment before it — the curriculum-gated dense shaping term did not move the policy off its entrenched static-grip local optimum, despite the curriculum switch itself firing correctly (verified in Task 2).

Per the design doc's own instruction, not attempting a further reward-only tweak unilaterally. The curriculum apparently opened too late and/or too weak a window relative to how deeply the static-grip behavior had already converged by iteration 700 (`grasp_contact` was already at ~17.8/20, essentially its plateau, by that point) — the remaining ~800 iterations were not enough for a newly-introduced weight-15.0 dense term to meaningfully perturb a policy that stable. Remaining candidates: a hierarchical reach-then-grasp-policy split, or questioning whether the gripper's actual closed-jaw force is physically sufficient to support lifting this object at all.

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

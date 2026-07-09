# Experiment 26: reintroduce the gripper (grasp/lift/carry/goal)

## Context

Direct user instruction (2026-07-09): reintroduce the gripper (grasp/
lift back in scope, after Experiment 25 deliberately removed it), use
longer episodes, then iterate on training/reward hyperparameters. Per
CLAUDE.md's mandate to prefer a structurally new strategy over another
parameter tweak after a string of nulls, this is not a repeat of any
single prior attempt — it composes several previously-*individually*-
validated pieces that were never combined, plus one concrete,
previously-identified-but-untried fix, plus new evidence this session
established that didn't exist when Experiments 16-24 ran.

## What's already established (this project's own prior verified
evidence — legitimate grounding per CLAUDE.md)

- **The jaws are not mechanically coupled** (source URDF's `mimic`
  constraint confirmed unenforced by Isaac Sim's USD import). Two fix
  attempts: Experiment 19 (`PhysxMimicJointAPI`, two configurations,
  both regressions, reverted) and Experiment 22 (software mirroring,
  `MirroredGripperAction` in `tasks/ar4/actions.py` — jaw2's target
  tracks jaw1's **actual measured position**, one physics step stale) —
  both failed. Experiment 22's own report identifies the untried fix:
  track jaw1's **commanded target** instead (available with zero lag).
  Reading the current code (`tasks/ar4/actions.py:191-194`) confirms the
  exact bug: `jaw1_actual_pos = self._asset.data.joint_pos[:,
  self._joint_ids[0]]` — the settled/actual value, not `self.
  _processed_actions[:, 0]` (jaw1's own just-computed, gate-processed
  commanded target this same step).
- **Experiment 21's proximity gate genuinely fixed a specific asymmetry**
  (`ProximityGatedBinaryJointPositionAction`, `tasks/ar4/actions.py`):
  forces the gripper open unless within `proximity_threshold=0.05m` of
  the object. Moved the failure signature from "one jaw never touches at
  all" to "both jaws touch, just not simultaneously" — a real, measured
  improvement, not a null result.
- **Experiment 17's antipodal grasp gate works exactly as designed**
  (`antipodal_grasp_bonus`, `tasks/ar4/mdp.py`, `force_threshold=0.05`,
  `antipodal_cos_threshold=-0.7071`): in the one real contact event it
  saw, it correctly rejected a non-antipodal wedge (cosine angle 0.66
  degrees short of the threshold, held for 230 steps, never credited).
  The problem was never the gate's correctness — it was that the
  compound "position correctly AND close simultaneously" behavior was
  never discovered via pure exploration within the training budget used
  at the time (1500 iterations, 5-8s episodes, pre-physics-fidelity-pass
  dt).
- **Experiment 16's reward-hacking mode** (ungated reach+grasp+lift+goal
  additive sum lets the policy satisfy lift/goal via wrist-wedging
  without genuine grasp) is now understood well enough to structurally
  avoid: gate each stage's potential on the previous stage's genuine
  completion, the same principle [[staged-reward-co-satisfiability]]
  documents from Experiment 25's own dead-zone fix, extended to more
  stages.

## What's new this session that didn't exist when 16-24 ran

- **Physics-fidelity pass** (2026-07-09): finer PhysX substeps
  (`sim.dt` halved with `decimation` doubled, control period unchanged),
  collision offsets empirically bounded, `_EE_OFFSET` re-verified both
  numerically and visually. Every grasp experiment 16-24 ran on the
  coarser pre-fidelity-pass physics.
- **Experiment 25 validates reliable sub-2cm positioning precision** is
  achievable with the corrected physics + adequate episode length
  (deterministic policy: 100% of rollouts get the end-effector within
  2cm of a target point). Positioning precision was explicitly
  implicated as a limiting factor in earlier joint-space grasp attempts
  (Experiments 9/10: "implicating positioning precision, not reward
  design") — this is now a validated, not assumed, capability under
  today's corrected setup.
- **Episode-length grounding**: Isaac Lab's own reference tasks scale
  episode length with task *structure* (Reach 12.0s, Lift 5.0s, Cabinet
  8.0s, **Stack — reach+grasp+lift+move+place, the closest structural
  analog — 30.0s**). Experiment 25 independently confirmed the same
  principle empirically (5.0s: `goal_reached` peaked then declined;
  20.0s: converged and held). Grasp+lift+carry+place experiments before
  today ran on episode lengths far short of Stack's 30.0s precedent.

## Hypothesis

Composing three previously-individually-validated fixes (Experiment 21's
proximity gate, Experiment 22's mirroring mechanism corrected for its
own identified lag bug, Experiment 17's antipodal gate) with a
Stack-task-precedented 30s episode length and today's corrected physics,
using Experiment 25's now-validated monotonic staged-potential reward
mechanism (extended from 2 stages to 4: reach → grasp → lift → goal)
instead of either Experiment 16's ungated sum or Experiment 17/18's
separately-weighted gated terms, will produce measurably more reliable
grasp discovery than any prior single-variable attempt — because the
compound behavior's three previously-separate obstacles (jaw asymmetry,
lag-prone mirroring, insufficient episode time under coarser physics)
are addressed together rather than one at a time, and Experiment 25 has
already validated the positioning precision this design assumes.

**Falsifiable as**: if `antipodal_grasp_bonus`'s underlying antipodal
contact condition (or the new staged milestone's grasp-stage component)
still shows `0` or near-`0` nonzero rate after a full 1500-iteration run
under this composed setup, the hypothesis is falsified — the remaining
bottleneck is not (only) the three addressed obstacles.

## Grounding (methodology)

- Xu et al. 2026 ("Stage-Transition Dense Reward Modeling,"
  arXiv:2606.31377), already read from source and used in this
  project's Experiments 17/18/25 — directly applicable to the 4-stage
  gated monotonic potential design below (same mechanism class,
  extended).
- Wang & Jin, "Knowledge capture, adaptation and composition (KCAC): A
  framework for cross-task curriculum learning in robotic manipulation"
  (arXiv:2505.10522) — abstract confirms a structured cross-task
  curriculum with reward-function/sequencing redesign across sub-tasks
  in robotic manipulation RL, supporting the general approach of
  building a harder compound skill on curriculum structure rather than
  training it flat from scratch. (Cited only for this general framing,
  confirmed directly against the paper's own abstract — a more specific
  "compose an already-trained sub-skill" claim surfaced in initial web
  search was not independently verifiable against the source PDF and is
  NOT relied on here.)
- This project's own extensive prior verified evidence (Experiments
  16-25, all cited above with specifics) — the primary grounding, per
  CLAUDE.md's explicit allowance for prior-verified-evidence alongside
  literature.

## Design

### Scene

Reuse `Ar4PickPlaceTouchGoalSceneCfg`'s fixed cube position `(0.20,
0.28, 0.006)` — **no randomization reintroduced this pass**. Adding
gripper mechanics and randomized positions simultaneously would
confound which change caused which effect; randomization is deferred to
a follow-up once this pass's own effect is isolated and measured,
mirroring Experiment 25's identical scoping decision. Fixed goal point:
same `(-0.20, 0.28, 0.15)` (`GOAL_OFFSET`), now interpreted as where the
*cube* must end up (carried there by the arm), not just the
end-effector — this task reintroduces genuine object transport.

Re-add `gripper_jaw1_contact`/`gripper_jaw2_contact` `ContactSensorCfg`s
(present in `pickplace_mirror_env_cfg.py`, absent from the touch-goal
lineage since no grasp existed there) — required for
`antipodal_grasp_bonus`.

### Action space

```
joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
gripper_position = MirroredGripperActionCfg(
    asset_name="robot", joint_names=GRIPPER_JOINT_NAMES,
    open_command_expr={...}, close_command_expr={...},
    object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
    proximity_threshold=0.05,
)
```

Same wiring Experiment 22 already used. `MirroredGripperAction` itself
gets the one-line fix described above (track `_processed_actions[:, 0]`
— jaw1's own commanded target, already gate-processed this step — not
`self._asset.data.joint_pos[:, self._joint_ids[0]]`). This is a
correctness fix to existing, shared code (`tasks/ar4/actions.py`), used
by both this new task and the existing `pickplace_jawmirror_env_cfg.py`
— re-verify `pickplace_jawmirror_env_cfg.py` isn't silently relying on
the buggy lag behavior for anything (unlikely, since Experiment 22's own
conclusion was that the lag was actively harmful, not incidentally
load-bearing) before relying on this shared-code change.

### Reward: 4-stage gated monotonic potential

Extends Experiment 25's `touch_goal_reward.py` pattern (pure-tensor,
Isaac-Lab-free, unit-testable) from 2 stages to 4. Each stage's raw
potential is a **monotonic function of a single scalar distance/state
measure within its own stage**, not a proximity bump — the exact
property that made Experiment 25's post-touch potential immune to the
dead-zone bug class. Stage floors: reach floor 0.0, grasp floor 0.25,
lift floor 0.50, goal floor 0.75 (ceiling 1.0) — four equal segments,
no independent per-stage weight tuning this pass (Tier 2 hillclimbing,
once a baseline exists, is the intended venue for that).

```
reach_progress   = clamp(1 - ee_to_cube_dist / REACH_DIST_NORM, 0, 1)                      # dense, always active
grasped          = antipodal_grasp_bonus's own underlying bilateral-contact condition (latched bool, same force_threshold/antipodal_cos_threshold as Experiment 17)
grasp_progress   = 1.0 if grasped else 0.0                                                  # binary within its segment - genuine grasp is the gate itself, not a shaped sub-metric
lifted           = grasped AND (cube.root_pos_w[:, 2] > LIFT_MINIMAL_HEIGHT)                 # latched bool
lift_progress    = clamp(cube_height_above_ground / LIFT_TARGET_HEIGHT, 0, 1) if grasped else 0.0
goal_progress    = clamp(1 - cube_to_goal_dist / CUBE_TO_GOAL_DIST, 0, 1) if lifted else 0.0

raw = where(lifted,  0.75 + 0.25 * goal_progress,
      where(grasped, 0.50 + 0.25 * lift_progress,
                      0.00 + 0.25 * reach_progress))
```

Concrete first-pass constants (all tunable via Tier 2 hillclimbing once
a baseline exists, per this project's established "first-pass value now,
weight/threshold tuning deferred" convention — these are starting
points, not independently re-derived for this exact task):

- `REACH_DIST_NORM = 0.3` (dense-shaping normalizer for the reach
  segment — the cube's own default spawn distance from the arm's rest
  pose is on this order).
- `LIFT_MINIMAL_HEIGHT = 0.03` — reused verbatim from
  `_raw_lift_progress_mirrored`'s own `lift_minimal_height` param
  (`tasks/ar4/pickplace_mirror_env_cfg.py`), not re-derived.
- `LIFT_TARGET_HEIGHT = 0.10` — 10cm above ground, the height at which
  `lift_progress` reaches its 1.0 ceiling within its segment.
- `CUBE_TO_GOAL_DIST = norm(GOAL_OFFSET)` — **derived, not hardcoded**
  (Experiment 25's final review caught exactly this kind of drift hazard
  once already: a hardcoded literal silently diverging from the
  geometry it measures). `GOAL_OFFSET=(-0.40, 0.0, 0.144)` unchanged
  from Experiment 25, so `CUBE_TO_GOAL_DIST = sqrt(0.40² + 0.144²) ≈
  0.4251m` (measuring cube-center-to-goal-point distance directly, not
  Experiment 25's touch-point-to-goal distance — a different, slightly
  larger reference distance, since this stage measures the cube's own
  position, not an offset touch point above it).

(`grasp_progress`'s segment is a flat 0.25-jump-on-gate rather than a
shaped ramp — deliberately, since Experiment 18 already falsified a
dense pre-grasp-readiness shaping term as a fix for discoverability, so
this pass does not reintroduce that specific mechanism; the compound
behavior's discoverability is instead being addressed by the
non-reward-shaping fixes above.) Wrapped in the same running-max
milestone delta mechanism as `staged_milestone_bonus`/
`touch_goal_milestone_bonus`. `env._grasped`/`env._lifted` latches,
reset per-episode, same pattern as `env._touched_cube`.

### Termination

`cube_reached_goal`-equivalent: cube within `GOAL_TOLERANCE=0.02m` of
the fixed goal point AND `env._lifted` true for that env (i.e. genuine
grasp+lift occurred at some point, not just incidental proximity).
Standard timeout otherwise.

### Episode length / physics

`episode_length_s = 30.0` (Stack-task precedent). `decimation=4`,
`sim.dt=0.005` (today's fidelity-pass values, matching
`pickplace_mirror_env_cfg.py`/`pickplace_touchgoal_env_cfg.py` — NOT the
older `0.01` most other `pickplace_*_env_cfg.py` variants still use).

### Observations

Experiment 25's set (arm joint pos/vel, cube position, goal position,
last action) plus gripper joint pos/vel back in (`joint_names` no longer
restricted to `ARM_JOINT_NAMES` for the position/velocity terms) and a
grasp-state observation (whether `env._grasped`/`env._lifted` are
latched — gives the policy direct access to its own stage progress, a
detail none of Experiments 16-18 included and worth testing since it's
a cheap, plausible discoverability aid consistent with this project's
own emphasis on ground-truth state over inferred proxies).

### Training

`Ar4PickPlacePPORunnerCfg` unchanged (no reason yet to retune an
already-proven PPO config before a baseline exists under the new
mechanism — that's exactly what Tier 2 hillclimbing is for once this
runs).

## Success criteria

`Episode_Termination/cube_reached_goal`-equivalent nonzero and
non-degenerate (this project's standing real-success-metric
convention) — NOT the shaped milestone reward scalar alone. Given the
hypothesis's own falsification condition above, a full 1500-iteration
run with checkpoint-rollout instrumentation (same method used to verify
Experiment 25 — deterministic-policy rollout, not training-time
scalars alone) is required before any verdict, per this project's
Tier-1 standard.

## Deferred to a follow-up, not forgotten

- Randomized cube spawn / mirrored goal (once this pass's own effect is
  isolated).
- Per-stage reward weight tuning (Tier 2 hillclimbing, once a baseline
  exists — this is the concrete next step the user's "iterate over
  hyperparameters" instruction maps to).
- PPO hyperparameter retuning.

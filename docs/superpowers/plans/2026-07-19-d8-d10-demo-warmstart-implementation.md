# d8/d10 demonstration-augmented warm-start — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test H1 (DAPG-style behavior-cloning pretrain from a scripted
demonstration, followed by full PPO fine-tune) as a warm-start
intervention for d8/d10's genuine, robust 0/24-both-shapes null at the
48mm-parity anchor (`unified-multi-die-specialist-distillation`'s FINAL
VERDICT). H2 (checkpoint warm-start from the nearest-by-sphericity d12
specialist) is pre-authorized as a per-shape fallback if H1 falsifies for
that shape, per the design spec's "one fallback rung, no new spec"
convention.

**Architecture:** (0) re-verify `dice_pick_demo.py`'s scripted grasp
actually transfers to 48mm scale for d8/d10 before trusting it as a
demonstration source (the spec's own flagged real scale-mismatch risk);
(1) a new closed-form action-space-mapping module + a new
`regress_on_paired_batches` regression function in
`tasks/franka/distillation.py` + a new sibling capture script that logs 5
scripted-grasp reference trajectories per shape; (2) a new sibling
BC-pretrain CLI that replays each captured trajectory through the real
target env (closed-form joint-position → raw-action conversion) and
BC-pretrains a fresh student against the pooled paired data, then hands
off to `train_franka.py`'s existing `--checkpoint --policy_only_checkpoint`
resume mechanism; (3) the real H1 run — full 1500-iteration PPO fine-tune,
3 seeds × 2 shapes, instrumented eval + video; (4) the pre-authorized H2
fallback, conditional per-shape on H1 falsifying; (5) verdict + kb/ROADMAP
update.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`distillation.py`, `dice_lift_joint_env_cfg.py`'s already-existing
`FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`), `scripts/dice_pick_demo.py`'s
scripted DiffIK grasp controller (reused by import, never modified in
place), desktop-first GPU dispatch
(`scripts/check_gpu_availability.sh`/`scripts/run_on_desktop_gpu.sh`), GCP
cloud fallback (`docs/cloud/dispatch-checklist.md`).

Spec: `docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`.
Research: `docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`.
Template/precedent for this plan's own structure: the just-finished
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`
(TDD discipline, task/commit format) and the more recent
`docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md`
(desktop-first dispatch block, cost-cap reasoning against real measured
baselines — supersedes the 2026-07-16 template's cloud-only default,
which predates the desktop dispatch infra). Fallback-rung task structure
follows `docs/superpowers/plans/2026-07-11-joint-space-die-lift.md`'s own
precedent (its Task 4 embeds "if FAILED, one fallback rung, then STOP" as
a verdict-gated branch) — here promoted to its own conditional top-level
task since H2 is a full separate multi-seed training run per shape, not a
same-task sub-step.
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Global Constraints

- **Task 0 gates every later task.** Do not start Task 1 until Task 0's
  re-verification reports PASS for both d8 and d10 at 48mm scale. If Task
  0 fails for a shape, **stop and report to the controller** for that
  shape rather than silently proceeding — per the spec, this would mean
  H1's premise ("a known-feasible grasp trajectory already exists for
  this shape at this size") does not hold, undercutting H1 before any
  PPO training starts. A per-shape split (e.g. d8 PASSes, d10 fails) is a
  real possible outcome — report and let the controller decide whether
  d8 alone proceeds.
- **Shapes evaluated and reported independently throughout** (d8, d10) —
  per the spec's own reasoning (d10 has two additional compounding
  disadvantages beyond sphericity that d8 doesn't), never assume both
  shapes pass or fail together, and never average their results together.
- **Size: 48mm parity only** (`FrankaDieLiftJointD8BigEnvCfg`/
  `...D10BigEnvCfg` — already exist, no new env cfg classes needed in
  this plan). Do not introduce real ~16-18mm size anywhere in this plan's
  training/eval — that would reintroduce the scale confound Task 3.5 of
  the prior experiment was built to isolate.
- **No new reward terms, no PPO-hyperparameter changes, no
  observation-schema changes.** This plan is exactly two training-
  *initialization* mechanisms (BC-pretrain warm start for H1, checkpoint
  warm start for H2) layered onto the already-validated 41-dim
  observation / 8-dim action / `FrankaLiftPPORunnerCfg` recipe. Do not
  touch `tasks/franka/mdp.py`, `tasks/franka/lift_reward.py`, or
  `tasks/franka/agents/rsl_rl_ppo_cfg.py` in this plan.
- **`dice_pick_demo.py`'s own Gate A/G/V contracts are never modified in
  place.** Every new script in this plan that needs its machinery
  (`spawn_scene_and_settle`, `run_detector_subprocess`,
  `select_target_detection`, `run_pick_sequence`) imports those functions
  from that file unchanged — the same reuse pattern that file's own Gate
  V already uses against Gate G.
- **H3 (exploration-noise/entropy retuning) is out of scope for this
  plan.** If both H1 and H2 falsify for a shape, that is a stop-and-report
  point back to Principal — do not write or execute an H3 task without a
  new spec.
- **TDD discipline for pure-Python pieces** (Task 1's
  `regress_on_paired_batches` and the new action-mapping module): write
  failing unit tests first, confirm the failure, implement, confirm
  green — matching this project's established
  `tasks/franka/shape_observations.py` / `tests/test_mdp_shape_observations.py`
  and `tasks/franka/distillation.py` / `tests/test_distillation_data_collection.py`
  precedent exactly. Run via
  `/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/<file>.py -v -p no:launch_testing`
  (plain python3/pytest lacks torch/rsl_rl in this environment).
- **Gripper binary-action convention: confirm by direct source read, not
  memory**, before writing any code that depends on it (Task 1's own
  first step). Read `isaaclab.envs.mdp.actions`'s
  `BinaryJointPositionAction` class on the desktop (where Isaac Lab is
  installed) and record the exact raw-action-value convention that
  selects `open_command_expr` vs. `close_command_expr` in the new
  action-mapping module's own docstring, citing the source
  file/line — per this project's own citation/fact-verification
  discipline (the spec explicitly declines to assert this from memory).
- **Demonstration trajectory data is a dataset — not committed.** Task
  1's captured reference trajectories go under `data/franka_demo_trajectories/`
  (already-gitignored `data/`, per this repo's public-repo-since-2026-07-13
  no-datasets convention) — only code that produces/consumes them is
  committed.
- **Execution backend: desktop-first, cloud fallback** (2026-07-18
  standing policy, CLAUDE.md's "Pi-as-primary-agent GPU dispatch" — this
  supersedes the 2026-07-16 template's cloud-only default). For every
  task that launches Isaac Sim:
  1. Check `scripts/check_gpu_availability.sh`. `TARGET=desktop` (exit 0)
     → dispatch via `scripts/run_on_desktop_gpu.sh` (default blocking
     mode, not `--detach`, for any task that needs a real result before
     the next step). `TARGET=cloud` (exit 1, BUSY) or unclear (exit 2,
     UNKNOWN) → fall back to `docs/cloud/dispatch-checklist.md`'s recipe.
     **Never treat "can't tell" as a green light for desktop** — UNKNOWN
     routes to cloud (or stop), never an assumed-available desktop.
  2. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim** (its
     blocking instruction, cost-cap paragraph, environment-conventions
     block, and bug-handling-discipline block).
  3. **`flock -o` is not automatic.** `run_on_desktop_gpu.sh` does not
     wrap the dispatched command in a lock itself — the command string
     shipped to the desktop (or run on a cloud instance) must itself be
     `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."`.
     Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
     (empty = clear) before dispatch, not a process-name/path grep.
  4. **Non-headless on desktop** (CLAUDE.md's standing "the user wants to
     watch" instruction) — do not pass `--headless`. **Headless only** on
     the cloud fallback (the standing, confirmed exception).
  5. **Full teardown after any cloud task**: verify zero
     instances/disks/snapshots remain (`scripts/check_cloud_state.sh`)
     before marking the task done. Desktop dispatch: verify
     `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
     `check_gpu_availability.sh` are all clear/AVAILABLE afterward.
- **Cost cap: notify the controller if cumulative cloud spend across
  Tasks 1/2/3/4 combined exceeds $10.** This is a cloud-fallback safety
  backstop, not an expected spend — desktop dispatch is expected to bring
  actual cost to $0, matching Tasks 5/6 of the just-finished experiment
  (both $0, desktop-only). If cloud is needed: this plan's dominant cost
  is Task 3's 6 full 1500-iteration runs (3 seeds × {d8, d10}) plus, in
  the worst case, Task 4's H2 fallback re-running the same 6 (both shapes
  falsify H1). Using this arc's own measured baseline of **≈$0.91 per
  3-seed/1500-iteration single-shape batch** (`ROADMAP.md`'s d20-big-geom
  gate task entry, SPOT-preemption-adjusted) as the per-shape unit: Task 3
  ≈ 2 × $0.91 ≈ $1.82; Task 4's absolute worst case (both shapes fall
  through) ≈ another $1.82; Task 0-2's capture/replay/BC-pretrain/smoke-test
  steps are short, non-full-budget runs, generously $1 total. Sum ≈ $4.64,
  doubled for a SPOT-preemption-retry buffer (this project's own recent
  precedent — `BACKLOG.md`'s "session hit repeated cloud friction: SPOT
  preemptions on 3/3 cloud tasks" entry) ≈ $9.3, rounded to **$10**. This
  is a smaller-scope experiment than the just-finished 4-shape unified
  arc (≈$5.87 total across specialist training + distillation + fine-tune
  for 2 shapes' worth of *successful* runs, many more total runs than
  this plan's 2-shape/no-distillation scope) and larger than the
  single-seed, no-fallback-branch target-selection plan's $5 cap (this
  plan has 3 seeds × 2 hypotheses possible per shape) — $10 is a
  considered number, not a stale copy-paste of the $15 default.
- Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line — each task
  below is executed by a freshly dispatched session; use that session's
  real ID, do not copy a fixed ID from this document.
- Real evidence over proxies at every eval: instrumented
  `max_height_gain`/`max_consecutive_lifted_steps` numbers per seed AND a
  reviewed eval video — not a shaped reward scalar or an exit code alone.
  Independently re-derive the raw per-step `.npy` height trajectory for
  at least one seed per shape per hypothesis rather than trusting the
  summary JSON alone, per this arc's own repeated (3-bugs-found-and-fixed)
  settle-detection discipline; use `franka_checkpoint_review.py`'s
  current (post-`977a748`) MIN-over-a-fixed-early-window settle-detection
  logic.

---

## Task 0 — Re-verify scripted grasp at 48mm scale for d8/d10 (PREREQUISITE — gates all later tasks)

**Files:**
- Create: `scripts/_diag_d8d10_48mm_grasp_reverify.py` — new, bounded
  diagnostic. Imports `dice_pick_demo.py`'s
  `spawn_scene_and_settle`/`run_detector_subprocess`/
  `select_target_detection`/`run_pick_sequence` unchanged (never modifies
  that file). Defines (for reuse by Task 1's capture script, so this
  logic is written once, not duplicated) two small helper functions:
  - `override_die_scale(scene_cfg, die_type, scale)` — mutates
    `scene_cfg.die_{die_type}.spawn.scale` to the 48mm-parity constant
    (`0.003167` for d8, `0.002928` for d10 — `FrankaDieLiftJointD8BigEnvCfg`/
    `...D10BigEnvCfg`'s own already-derived values,
    `tasks/franka/dice_lift_joint_env_cfg.py`) **before** scene
    construction — the same "mutate `scene_cfg` fields before
    `InteractiveScene(scene_cfg)` is built" pattern
    `spawn_scene_and_settle`'s own `light_scale` override already uses
    (read that function first).
  - `measure_settled_rest_height(scene, die_type) -> float` — after
    `spawn_scene_and_settle`'s own settle-and-verify completes, reads the
    live settled die's actual world-frame z / geometry directly (do not
    assume `_DIE_REST_HEIGHT_M`'s real-size value scales linearly with
    the spawn-scale ratio — the spec explicitly declines to assume this
    transfers unchanged; measure it live instead, the same rigor
    `_DIE_REST_HEIGHT_M`'s own original derivation used at real size per
    its own module comment).

**Interfaces:**
- Consumes: `dice_pick_demo.py`'s existing functions (import only),
  `dice_lift_joint_env_cfg.py`'s already-derived 48mm-parity scale
  constants.
- Produces: `override_die_scale`/`measure_settled_rest_height`, reused
  unchanged by Task 1's `scripts/extract_demo_trajectory.py`; a
  PASS/FAIL verdict per shape gating Task 1.

- [ ] **Step 1: Write the diagnostic.** For each of d8, d10: build the
  scene with `override_die_scale` applied to that die only (other dice in
  the 5-die layout stay at real size — irrelevant to this shape's own
  grasp check, and changing the whole scene layout is out of scope), run
  `spawn_scene_and_settle`, call `measure_settled_rest_height`, then run
  `run_pick_sequence` (via `run_detector_subprocess` +
  `select_target_detection`, exactly `dice_pick_demo.py`'s own Gate G/V
  flow) using the measured height in place of `_DIE_REST_HEIGHT_M`'s
  real-size constant, and report PASS/FAIL per that file's own existing
  sim-ground-truth lift-check verdict mechanism (do not build a new
  verdict mechanism).

- [ ] **Step 2: Run it, non-headless, under flock**, desktop-first per
  Global Constraints. Two shapes, one run each (a handful of seeds is not
  needed here — this is a mechanical transfer check, not the real
  demonstration capture, which is Task 1's own 5-seeds-per-shape job).

- [ ] **Step 3: Report the measured rest heights and PASS/FAIL per shape
  explicitly.** If either shape FAILs, **stop and report to the
  controller for that shape** (per Global Constraints) rather than
  proceeding to Task 1 for it.

- [ ] **Step 4: Commit.**

```bash
git add scripts/_diag_d8d10_48mm_grasp_reverify.py
git commit -m "feat: re-verify dice_pick_demo.py's scripted grasp at 48mm scale for d8/d10 (Task 0)"
git push origin main
```

---

## Task 1 — `regress_on_paired_batches`, closed-form action mapping, and demonstration capture

**Files:**
- Modify: `tasks/franka/distillation.py` — add
  `regress_on_paired_batches(obs, actions, student, optimizer, batch_size,
  num_epochs, generator) -> float`, mirroring `regress_on_pooled_batches`'s
  shuffle/minibatch/epoch loop and its call to `behavior_cloning_loss`
  verbatim, but taking pre-paired `(obs, actions)` tensors directly
  instead of calling a `MultiShapeTeacherRouter`. Update the module
  docstring's "new, not reusable as-is" section to note this addition,
  per this file's own established documentation rigor.
- Create: `tasks/franka/demo_action_mapping.py` — pure-torch, NO
  isaaclab/pxr imports (same importable-without-Isaac-Sim split as
  `shape_observations.py`/`lift_reward.py`, cited by
  `distillation.py`'s own module docstring). Two functions:
  - `joint_pos_to_raw_action(target_joint_pos, default_joint_pos, scale=0.5)`
    — the closed-form inverse of `JointPositionAction`'s
    `target_joint_pos = default_joint_pos + scale * raw_action`, i.e.
    `raw_action = (target_joint_pos - default_joint_pos) / scale`, for
    the 7 arm joints.
  - `gripper_target_to_raw_action(gripper_target)` — the raw-action value
    selecting `open_command_expr` vs. `close_command_expr`, per the
    convention confirmed by this task's own direct source read (Global
    Constraints) — cite the exact source location in this function's
    docstring.
  Module docstring must record the direct source-read finding for the
  gripper convention explicitly (file + line), not assert it from
  memory.
- Create: `scripts/extract_demo_trajectory.py` — new sibling script (per
  spec; never modifies `dice_pick_demo.py` in place). Imports
  `dice_pick_demo.py`'s `spawn_scene_and_settle`/
  `run_detector_subprocess`/`select_target_detection`/`run_pick_sequence`
  and Task 0's `override_die_scale`/`measure_settled_rest_height`
  (imported from `scripts/_diag_d8d10_48mm_grasp_reverify.py` — do not
  duplicate this logic). Passes a logging `on_step` callback into
  `run_pick_sequence` (the same extension point Gate V's own video
  capture already uses) that appends, every physics step: the desired
  joint-position target just issued to `panda_joint.*`
  (`_step_toward`/`_joint_space_prep`'s own internal `joint_pos_des`
  tensor) and the currently-commanded gripper target tensor. Writes one
  file per (shape, seed) to `data/franka_demo_trajectories/{shape}/seed{N}.pt`
  (`torch.save` of a plain dict — no new serialization format needed).
  `--seed` CLI flag varies the layout per capture, matching
  `dice_pick_demo.py`'s own `--seed` semantics.
- Test: `tests/test_distillation_data_collection.py` — extend with a new
  `TestRegressOnPairedBatches` class (loss decreases over epochs on a
  tiny synthetic paired dataset, using the existing `_StubActorCritic`
  helper already in this file; deterministic shuffling with a seeded
  generator, mirroring `TestPoolAndShuffle`'s own test shape).
- Test: `tests/test_demo_action_mapping.py` — new. Unit tests for
  `joint_pos_to_raw_action` (round-trips a known `target_joint_pos` back
  to a known `raw_action` against `JointPositionAction`'s documented
  formula; rejects a `scale=0` degenerate input) and
  `gripper_target_to_raw_action` (returns the correct raw value for both
  the open and close case, per Step 1's confirmed convention).

**Interfaces:**
- Consumes: Task 0's `override_die_scale`/`measure_settled_rest_height`
  and its PASS verdict (gate), `dice_pick_demo.py`'s existing functions
  (import only), `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`'s
  existing action-cfg values (`scale=0.5, use_default_offset=True` for
  the arm; `BinaryJointPositionActionCfg` for the gripper).
- Produces: `regress_on_paired_batches` (consumed by Task 2's BC-pretrain
  script), `demo_action_mapping.py`'s two functions (consumed by Task 2's
  replay driver's `action_fn`), 5 captured reference trajectories per
  shape under `data/franka_demo_trajectories/` (consumed by Task 2).

- [ ] **Step 1: Direct source read of the gripper binary-action
  convention** (Global Constraints) — do this before writing
  `gripper_target_to_raw_action`. Record the finding (file/line, exact
  convention) in `demo_action_mapping.py`'s own docstring.

- [ ] **Step 2: Write failing unit tests** — `TestRegressOnPairedBatches`
  in `tests/test_distillation_data_collection.py` and the new
  `tests/test_demo_action_mapping.py` — before implementing either
  function.

- [ ] **Step 3: Run tests, confirm they fail** (`function not defined`).

- [ ] **Step 4: Implement `regress_on_paired_batches` and
  `demo_action_mapping.py`.**

- [ ] **Step 5: Run tests, confirm they pass.**

- [ ] **Step 6: Write `scripts/extract_demo_trajectory.py`** per the
  Files section above.

- [ ] **Step 7: Run the real capture** — 5 seeds per shape (10 runs
  total), non-headless, desktop-first per Global Constraints, one
  `flock`-wrapped invocation per (shape, seed). Confirm each capture logs
  a non-empty joint-position/gripper-target trajectory and that
  `run_pick_sequence` itself still reports PASS for that seed (a capture
  seed that fails the grasp is not usable as a demonstration — if any of
  the 5 seeds per shape fails, drop it and capture a replacement seed
  instead, noting this in the task report; do not silently pool a
  failed-grasp trajectory as if it were a valid demonstration).

- [ ] **Step 8: Commit.**

```bash
git add tasks/franka/distillation.py tasks/franka/demo_action_mapping.py \
        scripts/extract_demo_trajectory.py \
        tests/test_distillation_data_collection.py tests/test_demo_action_mapping.py
git commit -m "feat: regress_on_paired_batches, closed-form action mapping, demo trajectory capture (Task 1)"
git push origin main
```

(`data/franka_demo_trajectories/` is gitignored — not part of this commit.)

---

## Task 2 — BC-pretrain pipeline: replay + `regress_on_paired_batches` + PPO-fine-tune handoff

**Design choice (documented, per dispatch instruction to make this call
and record the reasoning): a new sibling CLI script, not an extension of
`scripts/distill_specialists.py`.** `distill_specialists.py` is built
end-to-end around `tasks/franka/distillation.py`'s two-*live*-teacher
DAgger machinery (`MultiShapeTeacherRouter`, `dagger_beta_schedule`,
`mix_actions`, the beta-mixture rollout policy) — none of which apply
here: H1 has exactly one *fixed, already-recorded* demonstration source
per shape, no live teacher network to query or mix against, and no
per-row routing decision to make (every row in a given shape's replay
batch gets that same shape's own logged action, always). Forcing this
through `distill_specialists.py`'s router/mixture contract would mean
either faking a "teacher" that just replays a fixed action sequence
(distorting that contract's own meaning) or adding an if/else fork
through code that assumes a live queryable policy throughout. A new
sibling script is smaller, mirrors this project's own repeated precedent
for exactly this situation (`dice_pick_demo.py`'s Gate V reusing Gate G
by import rather than the reverse; the design spec's own choice of a new
sibling script for `extract_demo_trajectory.py` rather than modifying
`dice_pick_demo.py`), and keeps `distill_specialists.py`'s own,
separately-tested DAgger contract untouched.

**Files:**
- Create: `scripts/bc_pretrain_demo_warmstart.py` — new sibling CLI.
  Per shape (`d8`, `d10`):
  1. Build the real `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`
     `ManagerBasedRLEnv` (`num_envs=1` — replay is deterministic open-loop
     tracking, per the spec).
  2. For each of the shape's 5 captured trajectories
     (`data/franka_demo_trajectories/{shape}/seed{N}.pt`): reset the env,
     call `collect_rollout(env, action_fn, num_steps=len(trajectory), device)`
     with a scripted-replay `action_fn` — a closure over a step counter
     that ignores its `obs` argument, looks up the trajectory's logged
     joint-position/gripper target at the current index, converts it via
     `demo_action_mapping.py`'s two functions, and increments.
     `collect_rollout` is reused completely unchanged (Task 1 confirmed
     its contract only requires `action_fn(obs) -> action`, nothing about
     `action` depending on `obs`).
  3. `env.close()` after all 5 replays for that shape (this Isaac Lab
     installation cannot hold two `ManagerBasedRLEnv`s open at once —
     confirmed the hard way in the prior experiment's Task 5,
     `tasks/franka/distillation.py`'s own module docstring — build one
     shape's env, use it, close it, before the other shape's env if both
     run in the same process).
  4. Pool the 5 replays' paired `(obs, action)` tensors, call
     `build_student_actor_critic` + `regress_on_paired_batches`, log BC
     loss per epoch, stop once it plateaus (mirroring the prior
     experiment's Task 4 `--dry-run` loss-decrease check as the stopping
     criterion, per the spec).
  5. `save_student_checkpoint` — same `rsl_rl`-compatible format Task
     5/6 of the prior experiment already proved loads cleanly.
  Includes a `--dry-run` mode (stub trajectories + a physics-free stub
  env, mirroring `distill_specialists.py`'s own `--dry-run` precedent)
  so the pipeline wiring itself is verifiable without an Isaac Sim
  launch.
- Test: `tests/test_bc_pretrain_demo_warmstart.py` — unit-test the
  per-shape replay-then-regress driver logic against stub trajectories/a
  stub env (no real checkpoints, no Isaac Sim), same scope discipline as
  `tests/test_distillation_data_collection.py`.

**Interfaces:**
- Consumes: Task 1's `regress_on_paired_batches`, `demo_action_mapping.py`,
  and the 10 captured trajectory files.
- Produces: `scripts/bc_pretrain_demo_warmstart.py`'s CLI (documented via
  `--help`) and 2 real BC-pretrained student checkpoints (one per shape),
  consumed by Task 3's PPO fine-tune.

- [ ] **Step 1: Write failing unit tests** for the replay-then-regress
  driver against stub trajectories/a stub env.

- [ ] **Step 2: Run tests, confirm they fail.**

- [ ] **Step 3: Implement `scripts/bc_pretrain_demo_warmstart.py`.**

- [ ] **Step 4: Run tests, confirm they pass.** Also run `--dry-run` as a
  mechanical smoke test (no Isaac Sim launch), matching
  `distill_specialists.py`'s own established verification pattern.

- [ ] **Step 5: Real run — BC-pretrain both shapes.** Non-headless,
  desktop-first, `flock`-wrapped. Confirm BC loss actually plateaus (log
  and report the curve) before trusting the resulting checkpoint, per the
  spec's own stopping criterion.

- [ ] **Step 6: Bounded smoke test of the PPO-fine-tune handoff** —
  `train_franka.py --checkpoint <BC checkpoint> --policy_only_checkpoint
  --variant joint-die-d8-big --num_envs 64 --max_iterations <BC
  checkpoint's own saved "iter" + a small N, e.g. +5>` for **one** shape,
  a few iterations only (not the full 1500) — confirms the resume mechanics
  work end to end (no crash on `load_optimizer`, correct iteration
  arithmetic given the BC checkpoint's own saved `"iter"` field) before
  Task 3 commits to the real 6-run dispatch, mirroring the prior
  experiment's own Task 6 smoke-test precedent (`ROADMAP.md`'s "Task 6"
  entry: "Verified via a bounded 3-iteration smoke test on the desktop
  before the real dispatch").

- [ ] **Step 7: Commit.**

```bash
git add scripts/bc_pretrain_demo_warmstart.py tests/test_bc_pretrain_demo_warmstart.py
git commit -m "feat: BC-pretrain-from-demonstration pipeline + PPO-fine-tune handoff smoke test (Task 2)"
git push origin main
```

(BC-pretrained checkpoint files themselves are logs/model artifacts —
not committed, per this repo's `models`/`data` gitignore convention.)

---

## Task 3 — Real H1 run: PPO fine-tune + eval, d8 and d10, 3 seeds each

**Files:** none new — runs Task 2's real BC checkpoints through
`train_franka.py`'s existing `--checkpoint --policy_only_checkpoint`
mechanism (already-existing `--variant joint-die-d8-big`/`joint-die-d10-big`
choices, wired in the prior experiment's Task 3.5 — no new `--variant`
wiring needed in this plan).

**Interfaces:**
- Consumes: Task 2's 2 real BC-pretrained checkpoints.
- Produces: 6 full PPO-fine-tuned checkpoints (3 seeds × 2 shapes) +
  per-seed, per-shape discovery-rate numbers — the spec's H1
  falsification-bar evidence.

- [ ] **Step 1**: Confirm `scripts/check_gpu_availability.sh` →
  desktop or cloud per Global Constraints; provision/record accordingly.

- [ ] **Step 2**: For each shape × seed (42, 123, 7) — 6 runs total:

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
  --variant joint-die-d8-big --checkpoint data/franka_demo_trajectories_or_checkpoint_path/model_bc.pt \
  --policy_only_checkpoint --seed 42 --num_envs 4096 \
  --max_iterations <BC checkpoint's own saved iter + 1500>"
```

(Repeat for d10 and for seeds 123/7 — 6 total invocations. Confirm the
exact `--max_iterations` arithmetic against Task 2 Step 6's smoke test
finding before the real dispatch; per the prior experiment's own Task 6
precedent this may need to be `bc_iter + 1500`, not a bare `1500`, to get
a true 1500-iteration PPO budget.)

- [ ] **Step 3**: Instrumented eval per shape/seed —
  `franka_checkpoint_review.py --variant joint-die-d8-big|joint-die-d10-big
  --checkpoint <run's final model> --num_envs 8`, undiluted 48mm,
  identical mechanism to every existing specialist baseline in this arc.
  Capture and review eval video for any run showing nonzero discovery
  (rest frame vs. peak-height frame, per Global Constraints).

- [ ] **Step 4**: Re-derive the raw per-step `.npy` height trajectory for
  at least one seed per shape (not summary-JSON-only trust), per this
  arc's own repeated settle-detection-bug discipline.

- [ ] **Step 5**: Report per-shape, per-seed discovery rate explicitly
  (not averaged) against the spec's falsification bar: **falsified for a
  shape if all 3 seeds show 0/8** (0/24 total for that shape); **any
  nonzero count in any seed is a real positive signal** per the spec's
  own "no spurious partial count has ever been observed" reasoning.
  Explicitly flag if any seed lands in the never-before-observed
  "partial" range (1/8-7/8) — a reportable finding on its own, per the
  spec.

- [ ] **Step 6**: Full teardown if cloud-dispatched; report elapsed cost
  against the $10 cap. If desktop-dispatched, verify
  `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh` all clear/AVAILABLE.

- [ ] **Step 7**: **Do not decide whether Task 4 (H2) runs** — report the
  clear per-shape H1 verdict (PASS/falsified) and stop for the
  controller's call on which shape(s), if any, proceed to Task 4, same
  discipline as the prior experiment's own gated tasks.

- [ ] **Step 8: Commit** any code changes only (training runs themselves
  aren't committed) — none expected unless Step 2's real dispatch
  surfaces a bug in Task 1/2's code, in which case fix it, re-verify
  those tasks' own unit tests still pass, and commit the fix separately
  with its own message.

---

## Task 4 — H2 fallback (CONDITIONAL — only for shapes where Task 3's H1 falsified)

**Trigger condition:** run this task **only** for a shape where Task 3
reported H1 falsified (all 3 seeds, 0/24). If H1 succeeded for both
shapes, **skip this task entirely** and proceed straight to Task 5. If
H1 succeeded for one shape and falsified for the other, run this task
for the falsifying shape only.

**Files:** none new — direct `train_franka.py --checkpoint` resume from
the existing d12 specialist checkpoint (no new pipeline code, per the
spec: "No new code in `tasks/franka/distillation.py` is needed for H2 at
all").

**Interfaces:**
- Consumes: `gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`
  (the nearest-by-sphericity, already-converged d12 specialist —
  `DEFAULT_D12_CHECKPOINT` in `tasks/franka/distillation.py`, reuse the
  same constant/path, do not re-type it by hand).
- Produces: per-shape H2 discovery-rate numbers for whichever shape(s)
  triggered this task.

- [ ] **Step 1**: Confirm which shape(s) this task actually runs for,
  per the trigger condition above — state this explicitly before
  dispatching anything.

- [ ] **Step 2**: For each triggering shape × seed (42, 123, 7):

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
  --variant joint-die-d8-big --checkpoint gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt \
  --seed 42 --num_envs 4096 --max_iterations 2999"
```

(Full optimizer-state resume — do **not** pass `--policy_only_checkpoint`,
since the d12 checkpoint carries real PPO optimizer state, per the spec.
`--max_iterations 2999` = the checkpoint's own saved `iter=1499` + 1500,
matching this arc's own established resume-arithmetic convention —
confirm against the checkpoint's actual saved `"iter"` field before
dispatching, do not assume 2999 is universally correct if a different
checkpoint is substituted.)

- [ ] **Step 3**: Instrumented eval + video review, identical mechanism
  to Task 3 Steps 3-4.

- [ ] **Step 4**: Report per-shape H2 verdict against the same
  falsification bar (0/24 = falsified). **If H2 also falsifies for a
  shape, that is a stop-and-report point for that shape** — report both
  null results back to Principal; do not proceed to H3 or any further
  intervention without a new spec.

- [ ] **Step 5**: Full teardown / desktop-clear verification, same
  discipline as Task 3 Step 6. Report elapsed cost against the $10 cap
  (cumulative across Tasks 1-4).

---

## Task 5 — Verdict, ROADMAP.md + kb update

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry (per-shape, per-hypothesis).
- Create or modify: a `kb/wiki/experiments/` article for this experiment
  (check whether extending `kb/wiki/experiments/dice-pick-demo.md` or
  `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`, or
  writing a new `kb/wiki/experiments/d8-d10-demo-warmstart.md`, better
  fits this repo's kb-maintenance convention before choosing — a new
  article is likely cleaner since this is a materially different
  mechanism (warm-start, not specialist/distillation) tested against the
  same two shapes the unified experiment's own FINAL VERDICT left open;
  cross-link from that article's "d8/d10 remain open" line either way).

**Interfaces:**
- Consumes: Task 3's H1 results and Task 4's H2 results (if run).
- Produces: the closing, evidence-backed verdict for this experiment.

- [ ] **Step 1**: Write the verdict, per shape: H1 result (PASS/falsified,
  with the exact per-seed discovery counts), H2 result if triggered
  (same), and the final disposition (which mechanism, if any, produced
  real discovery for that shape; both-null if both fell through).
  Include an explicit call-out of any never-before-observed "partial"
  per-seed result (1/8-7/8), per the spec's own requirement — this is a
  new empirical fact about this project's own discovery dynamics worth
  recording regardless of the overall verdict.
- [ ] **Step 2**: Report total cumulative cost across Tasks 1-4 against
  the $10 cap.
- [ ] **Step 3**: Update `ROADMAP.md` and the kb article, following this
  repo's continuous-kb-update convention (not batched to session end).
- [ ] **Step 4**: Commit.

```bash
git add ROADMAP.md kb/wiki/experiments/
git commit -m "verdict: d8/d10 demonstration-augmented warm-start experiment"
git push origin main
```

# Unified Multi-Die Grasp Policy (Specialist + Distillation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a single unified PPO policy that grasps-and-lifts a
commanded die of shape d8, d10, d12, or d20 (one die per episode, ground
truth observations, lift-only) by first training per-shape specialists,
then distilling them into one policy and RL-fine-tuning it.

**Architecture:** Three ordered phases against `tasks/franka/`'s existing
Isaac Lab / `rsl_rl` PPO stack: (1) bake missing d8/d10/d12 physics assets
at real target sizes and add a shape-class one-hot + geometry-descriptor
observation feature to the existing lift env cfg; (2) train 4 per-shape
specialists (d20's retries the already-falsified mixed-size-DR mechanism
with the new geometry feature added); (3) distill the 4 frozen specialists
into one policy (UniDexGrasp++ GiGSL pattern) and RL-fine-tune it.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`lift_env_cfg.py`, `dice_lift_joint_env_cfg.py`, `mdp.py`,
`agents/rsl_rl_ppo_cfg.py`), GCP cloud training pipeline
(`docs/cloud/franka-cloud-shakedown.md`).

Spec: `docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`.
Research: `.superpowers/sdd/research-multi-die-unified-policy.md`.
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Global Constraints

- **d4 is out of scope** — do not add it to any env cfg, observation
  schema, or specialist set in this plan.
- **Distractors/target-selection are out of scope** — every env cfg in
  this plan spawns exactly one die per environment. Do not add a second
  object or a target-flag observation term.
- **Ground-truth observations only** — no vision-detector integration in
  this plan.
- **Lift-only** — no goal-tracking/carry reward terms beyond what
  `FrankaLiftEnvCfg` already inherits; do not add a carry-to-goal
  criterion.
- **Execution backend: GCP cloud (SPOT g2-standard-4 + L4)** for every
  task that trains a policy (Tasks 2, 3, 5, 6) — this plan trains 6+
  separate policies (3 new-shape specialists + 1 d20 retry + 1
  distillation + 1 fine-tune, likely across multiple seeds each), too
  much sequential GPU time to tie up the local RTX 5070 Ti for a session
  the user wants to use for live viewing of other work. Cloud runs
  **headless** — the standing, confirmed exception to the local
  never-headless rule (`kb/wiki/concepts/cloud-training.md`). Recipe of
  record: `docs/cloud/franka-cloud-shakedown.md`. Every cloud task must
  still produce and sync real eval videos (per this repo's verification
  standard) even though the run itself is headless.
- **Local GPU tasks** (0, 1, 4): no Isaac Sim/Lab launch at all (asset
  baking is a headless batch process by established precedent — see
  `scripts/bake_die_asset.py`'s own docstring — not a "watch it run"
  case; the observation-schema and distillation-pipeline code in Tasks 1
  and 4 are pure config/Python changes verified by pytest, no sim launch
  needed until Task 2). If a task unexpectedly needs to launch Isaac Sim
  non-headlessly, flag to controller rather than defaulting to headless
  locally.
- **Cost cap: notify the controller if cumulative cloud spend across
  Tasks 2/3/5/6 combined exceeds $15** — track by instance-uptime ×
  the published ~$0.361/hr on-demand rate (or actual SPOT rate if
  cheaper), per this repo's standing practice (no BigQuery billing
  export exists). Each cloud task must log its own instance creation
  timestamp and report an elapsed-cost estimate before teardown.
- **One Isaac Sim process at a time; `flock -o` required** for any local
  step that does launch Isaac Sim (none currently planned, but if a task
  needs one): `flock -o /tmp/rl_isaac_sim.lock -c "..."`. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  before any GPU-touching step, local or deciding to sequence around a
  concurrent cloud job.
- **Full teardown after every cloud task**: verify zero instances/disks/
  snapshots remain before marking the task done.
- Commit messages end: `Claude-Session: https://claude.ai/code/session_01WLbJi5jaDPLxCZXSrXMycR`.
- Real evidence over proxies at every eval: instrumented discovery-rate/
  z-gain numbers per seed AND a reviewed eval video, not a shaped reward
  scalar or an exit code alone.

---

## Task 0 — Real target sizes + baked physics assets for d8/d10/d12

**Files:**
- Create: `assets/dice/d8_physics.usd`, `assets/dice/d10_physics.usd`,
  `assets/dice/d12_physics.usd` (via existing `scripts/bake_die_asset.py
  --die {d8,d10,d12}` — no code changes needed to that script, it already
  supports all 5 die choices).
- Create: `scripts/_diag_d8d10d12_standard_scale_check.py` (new — extend
  the existing no-physics headless-`SimulationApp` bounding-box-read
  pattern from `scripts/_diag_d20_standard_scale_check.py` to loop over
  d8/d10/d12).
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add
  `_D8_USD`/`_D10_USD`/`_D12_USD` path constants (same pattern as
  `_D20_USD`/`_CUBE48_USD`) and three new env cfg classes,
  `FrankaDieLiftJointD8StandardEnvCfg` /
  `FrankaDieLiftJointD10StandardEnvCfg` /
  `FrankaDieLiftJointD12StandardEnvCfg` (+ `_PLAY` variants each),
  following the exact structure of `FrankaDieLiftJointStandardEnvCfg`
  (mass pinned at a real-density estimate — do not reuse 0.216kg
  blindly, that value was DexCube's own measured mass at DexCube's own
  size; derive each shape's own real mass the same way, or flag to
  controller if no established method exists yet for non-d20 shapes).

**Interfaces:**
- Consumes: `scripts/bake_die_asset.py`'s existing `--die` CLI (no
  changes); `dice_lift_joint_env_cfg.py`'s existing `_D20_RIGID_PROPS`
  constant (reuse, don't duplicate).
- Produces: three new spawnable env cfg classes with a confirmed-correct
  `spawn.scale` tuple for each shape's real standard size, consumed by
  Task 1's observation-schema additions and Task 2's specialist training.

- [ ] **Step 1: Research real commercial "standard" sizes for d8, d10,
  d12** — same method already used for d20 (30.3mm "jumbo" → ~20-22mm
  "standard", `ROADMAP.md`'s 2026-07-15 entry): check real dice
  retailers/manufacturer spec sheets for each shape's standard
  millimeter size. Record the source(s) in the eventual task report,
  same rigor as the d20 correction.

- [ ] **Step 2: Bake physics assets.**

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py --die d8 2>&1 | tee /tmp/bake_d8.log"
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py --die d10 2>&1 | tee /tmp/bake_d10.log"
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py --die d12 2>&1 | tee /tmp/bake_d12.log"
```

Expected: each log ends with a `[BAKE] OK: ... root='Object' ...` line,
matching the existing d20/cube48 bake convention.

- [ ] **Step 3: Write and run the scale-derivation diagnostic**, extending
  `scripts/_diag_d20_standard_scale_check.py`'s exact no-physics
  headless-`SimulationApp` bounding-box-read pattern (read that file
  first) to compute and verify the correct `scale` tuple for each of
  d8/d10/d12 against Step 1's target mm sizes, using the same
  scale-per-mm linear-fit method `FrankaDieLiftJointStandardEnvCfg`'s own
  docstring documents (fit a ratio from this file's existing baked-asset
  scale constants, or derive fresh per-shape ratios if each mesh's native
  bounding box differs from d20's — check directly, don't assume the
  same 3.302305e-5/mm ratio transfers across shapes). Verify each
  computed scale reproduces its target size within the same 0.3mm
  tolerance the d20 diagnostic used.

- [ ] **Step 4: Add the three new env cfg classes** to
  `dice_lift_joint_env_cfg.py` per the Files section above.

- [ ] **Step 5: Commit.**

```bash
git add assets/dice/d8_physics.usd assets/dice/d10_physics.usd assets/dice/d12_physics.usd \
        scripts/_diag_d8d10d12_standard_scale_check.py tasks/franka/dice_lift_joint_env_cfg.py
git commit -m "feat: bake d8/d10/d12 physics assets, derive real standard sizes, add specialist env cfgs"
```

---

## Task 1 — Observation schema: shape-class one-hot + geometry-descriptor feature

**Files:**
- Modify: `tasks/franka/mdp.py` — add two new observation-term functions:
  `object_shape_class_onehot(env, asset_cfg) -> torch.Tensor` (shape
  `(num_envs, 4)`, one-hot over {d8, d10, d12, d20} — the class is a
  per-env static property set at spawn time via each env cfg's own object
  choice, not something read off the live object state) and
  `object_geometry_descriptor(env, asset_cfg) -> torch.Tensor` (shape
  `(num_envs, K)`, K = whatever the implementer picks per the spec's
  guidance — the simplest continuous feature actually computable from
  this project's existing baked-mesh pipeline, e.g. a sphericity proxy
  computed once per shape from each baked USD's mesh and broadcast as a
  per-env constant; document the exact formula and K in the function's
  own module-level comment referencing the spec).
- Modify: `tasks/franka/lift_env_cfg.py`'s `ObservationsCfg.PolicyCfg` —
  add `shape_class = ObsTerm(func=mdp.object_shape_class_onehot)` and
  `geometry_descriptor = ObsTerm(func=mdp.object_geometry_descriptor)`
  terms.
- Test: `tasks/franka/tests/test_mdp_shape_observations.py` (new — plain
  pytest, no Isaac Sim launch needed if the functions are written to
  accept a minimal mock/stub env object exposing just what they read;
  if that's not achievable cleanly, use this repo's existing
  `/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest -p no:launch_testing`
  convention instead — check `tasks/franka/tests/` for the established
  pattern before picking one).

**Interfaces:**
- Consumes: Task 0's new env cfg classes (to know which shape/class each
  is associated with) and existing baked USD mesh data (for the geometry
  descriptor's computation).
- Produces: `object_shape_class_onehot` and `object_geometry_descriptor`
  observation functions, importable as `mdp.object_shape_class_onehot`
  / `mdp.object_geometry_descriptor`, consumed unchanged by Task 2's
  specialist env cfgs and Task 4/5's distillation/fine-tune env cfgs.

- [ ] **Step 1: Write failing unit tests** for both new functions
  (correct one-hot for each of the 4 shapes; geometry descriptor returns
  a finite, shape-appropriate value — e.g. a cube-like die and d20
  produce measurably different descriptor values).

- [ ] **Step 2: Run tests, confirm they fail** (`function not defined`).

- [ ] **Step 3: Implement both functions** in `tasks/franka/mdp.py`.

- [ ] **Step 4: Run tests, confirm they pass.**

- [ ] **Step 5: Wire both ObsTerms into `ObservationsCfg.PolicyCfg`** and
  confirm (via a quick local, no-GPU-required Python import check, not a
  sim launch) that `FrankaLiftEnvCfg`'s observation-space size grows by
  exactly `4 + K` dims and nothing else changes.

- [ ] **Step 6: Commit.**

```bash
git add tasks/franka/mdp.py tasks/franka/lift_env_cfg.py tasks/franka/tests/test_mdp_shape_observations.py
git commit -m "feat: add shape-class one-hot + geometry-descriptor observation terms"
```

---

## Task 2 — Train d8/d10/d12 specialists (cloud)

**Files:**
- Create: `tasks/franka/agents/` PPO runner cfg entries for each new
  specialist variant (follow `FrankaLiftPPORunnerCfg`'s existing
  pattern in `tasks/franka/agents/rsl_rl_ppo_cfg.py` — one runner cfg
  class per specialist, or one shared class parameterized by
  experiment_name, matching whatever this file's existing convention
  already is for the d20 variants).
- No env cfg changes needed beyond Task 0/1's additions.

**Interfaces:**
- Consumes: Task 0's `FrankaDieLiftJointD8StandardEnvCfg` /
  `...D10...` / `...D12...`, Task 1's observation terms.
- Produces: 3 trained specialist checkpoints (one per shape) + their
  discovery-rate numbers, consumed by Task 4's distillation step as
  frozen teachers.

- [ ] **Step 1**: Provision a GCP SPOT g2-standard-4 + L4 instance per
  `docs/cloud/franka-cloud-shakedown.md`. Record creation timestamp.

- [ ] **Step 2**: Train each of d8/d10/d12 for the same 1500-iteration
  budget this project's existing lift variants use, at minimum 3 seeds
  each (matching asset-bisect's own seed count) — headless, per the
  cloud exception.

- [ ] **Step 3**: Run this project's existing instrumented eval (the
  `_PLAY` variant pattern + z-gain/lift-height instrumentation already
  used in asset-bisect/size-curriculum) for each shape/seed. Capture and
  sync eval videos to GCS.

- [ ] **Step 4**: Report per-shape, per-seed discovery rate (sustained
  lift vs. no lift) — do not average away per-seed results, matching
  this repo's existing bisect/curriculum reporting convention.

- [ ] **Step 5**: Full teardown; verify zero instances/disks/snapshots
  remain. Report elapsed cost estimate.

- [ ] **Step 6: Commit** any new runner-cfg code (training runs
  themselves aren't committed, only code/config).

```bash
git add tasks/franka/agents/rsl_rl_ppo_cfg.py
git commit -m "feat: add d8/d10/d12 specialist PPO runner configs"
```

---

## Task 3 — d20 specialist: size-DR + geometry-feature retry (cloud) — GATE before Task 4

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — new class
  `FrankaDieLiftJointRandomSizeEnvCfg` using `MultiAssetSpawnerCfg` with
  `random_choice=True` (not `False`, unlike the falsified
  `FrankaDieLiftJointMixedEnvCfg`) spanning a size range from ~22mm
  (Task 0-derived-equivalent real standard for d20, already established
  as `FrankaDieLiftJointStandardEnvCfg`'s 0.000727 scale) to 48mm
  (`FrankaDieLiftJointBigEnvCfg`'s 0.001585 scale).

**Interfaces:**
- Consumes: Task 1's `object_geometry_descriptor` term (this is the one
  new ingredient vs. the already-falsified `FrankaDieLiftJointMixedEnvCfg`).
- Produces: a discovery-rate number for this retry, gating whether Task
  4 includes a d20 specialist at all.

- [ ] **Step 1**: Before writing training code, **directly verify**
  whether `isaaclab.sim.spawners.wrappers.wrappers.py`'s
  `spawn_multi_asset` with `random_choice=True` resamples per-episode-reset
  or only assigns once per-env at scene-spawn time (read the source
  directly, the same way `FrankaDieLiftJointMixedEnvCfg`'s own docstring
  documents having done for `random_choice=False`). **If it only
  supports per-env-fixed assignment, STOP and report to controller**
  rather than silently training with a mechanism different from what the
  spec assumed (the spec flags this exact risk explicitly).

- [ ] **Step 2**: Implement `FrankaDieLiftJointRandomSizeEnvCfg` per the
  Files section, using whichever resampling mechanism Step 1 confirmed
  is real (adjust the class docstring to state which one, following this
  file's own convention of precise, load-bearing docstrings).

- [ ] **Step 3**: Train on GCP cloud, minimum 3 seeds, 1500 iterations
  (or checkpoint-resumed staged budget if the confirmed mechanism from
  Step 1 requires a different structure than a single flat run).

- [ ] **Step 4**: Run instrumented eval across the size range (not just
  one fixed size) + video review.

- [ ] **Step 5: Falsification check** — compare this retry's discovery
  rate directly against the size-curriculum's original 0/3 floor
  (`FrankaDieLiftJointMixedEnvCfg`'s verdict, `ROADMAP.md` 2026-07-13).
  Report the comparison explicitly; do not proceed to Task 4 with a d20
  specialist folded in silently either way — **STOP and report to
  controller** with the clear result (cleared the floor / did not) before
  Task 4 starts, since Task 4's specialist set depends on this outcome.

- [ ] **Step 6**: Full teardown; report elapsed cost.

- [ ] **Step 7: Commit.**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py
git commit -m "feat: d20 size-DR + geometry-feature specialist retry env cfg"
```

---

## Task 3.5 — 48mm-parity check for d8/d10/d12 (cloud) — inserted 2026-07-16, GATE before Task 4

**Why this task exists (not in the original plan):** Task 2 (d8/d10/d12 at
real small sizes, ~16-18mm) and Task 3 (d20 with size-DR across 22-48mm +
geometry conditioning) both returned 0 discovery — 0/9 and 0/120
respectively, independently confirmed as genuine zero-engagement results
(not measurement noise) by tracing raw per-step height data in both cases.
Task 3's own sweep included 48.0mm exactly (the one size where the
asset-bisect ladder got real discovery, 1/3, for a *single-size, undiluted*
d20 population) and still got 0/3 there — pointing at population dilution
across multiple sizes, not shape itself, as the dominant confound. Neither
Task 2 nor Task 3 ever tested d8/d10/d12 at a single, undiluted, 48mm
population the way the original asset-bisect did for the cube (3/3) and
d20 (1/3). This task closes that gap before any decision on Task 4's
distillation premise (currently unsatisfiable — zero working specialists
exist).

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — three new classes
  (+ `_PLAY` variants), `FrankaDieLiftJointD8BigEnvCfg` /
  `FrankaDieLiftJointD10BigEnvCfg` / `FrankaDieLiftJointD12BigEnvCfg`,
  mirroring `FrankaCubeBakedLiftJointEnvCfg`'s/`FrankaDieLiftJointBigEnvCfg`'s
  own pattern exactly: reuse Task 0's baked `d8_physics.usd`/
  `d10_physics.usd`/`d12_physics.usd`, override `scale` to
  `FrankaDieLiftJointBigEnvCfg`'s own already-verified 48.0mm constant
  (0.001585 — reuse directly, do not re-derive; d8/d10/d12's own native
  mesh bbox ratios differ from d20's per Task 0's own finding, but the
  *target* mm size — 48.0mm — is shape-independent by construction, so
  the scale constant that hits 48.0mm for d20 will NOT be correct for
  d8/d10/d12's own different native bboxes — **compute each shape's own
  48.0mm-targeting scale using Task 0's own per-shape native-bbox
  measurements** (`scripts/_diag_d8d10d12_standard_scale_check.py`'s
  already-measured native bboxes: d8 15.1544, d10 16.3931 max dim, d12
  32.5160 stage units — do not assume 0.001585 transfers across shapes,
  it doesn't), mass pinned 0.216kg (matching every other Big-rung class).
- Modify: `scripts/train_franka.py`, `scripts/franka_checkpoint_review.py`,
  `scripts/sync_run_to_gcs.py` — wire 3 new `--variant` choices
  (`joint-die-d8-big`/`d10-big`/`d12-big`), same pattern as Task 2/3.

**Interfaces:**
- Consumes: Task 0's baked assets + native-bbox measurements, Task 1's
  observation terms.
- Produces: 3 more specialist checkpoints (48mm, single-size, undiluted)
  + discovery-rate numbers, directly comparable to asset-bisect's own
  cube (3/3) and d20 (1/3) 48mm baselines.

- [ ] **Step 1**: Compute each shape's own 48.0mm-targeting scale from
  Task 0's already-measured native bboxes (no new Isaac Sim launch
  needed for this arithmetic — reuse the measured numbers directly:
  `scale = 48.0 / (native_max_dim_stage_units * 1000.0)`).
- [ ] **Step 2**: Add the three new env cfg classes + wire the three
  scripts, following Task 2's exact wiring pattern. Commit before any
  cloud provisioning.
- [ ] **Step 3**: Cloud (GCP SPOT g2-standard-4 + L4), one instance,
  3 seeds × 3 shapes = 9 runs, 1500 iterations each, headless.
- [ ] **Step 4**: Instrumented eval + video per seed/shape. Given Tasks 2
  and 3 both found a measurement artifact in
  `franka_checkpoint_review.py`'s `max_height_gain` (a spurious spike at
  an episode-reset boundary contaminating the naive `max()` over a
  multi-episode recording window) — independently re-derive the raw
  per-step trajectory for at least one seed per shape rather than
  trusting the summary JSON's `max_height_gain`/`sustained_lift` fields
  alone, per both prior tasks' own established practice. Consider
  whether this is worth actually fixing in `franka_checkpoint_review.py`
  itself now that it's recurred 2/2 times on every variant tested at a
  small-enough size/short-enough episode — your call, flag if you think
  it should be fixed rather than worked around a third time.
- [ ] **Step 5**: Report per-shape, per-seed discovery rate (not
  averaged), explicitly compared against asset-bisect's own cube
  (3/3-at-48mm) and d20 (1/3-at-48mm) baselines.
- [ ] **Step 6**: Full teardown; report cost against the cumulative
  $15 cap (Tasks 2+3 already spent ≈$2.96 — ≈$12.04 remains for this
  task plus Tasks 5/6).
- [ ] **Step 7**: **Do not decide** whether/how Task 4 proceeds — report
  the clear numeric result (does population dilution explain the prior
  nulls, or does shape remain a barrier even at matched, undiluted 48mm
  conditions) and stop for the controller's call, same discipline as
  Task 3.

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py scripts/train_franka.py \
        scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: d8/d10/d12 48mm-parity specialist check (Task 3.5)"
```

---

## Task 4 — Distillation pipeline (local, no GPU training yet)

**Files:**
- Create: `scripts/distill_specialists.py` — loads the frozen specialist
  checkpoints (however many Task 2/3 actually produced — 3 or 4 depending
  on Task 3's gate outcome), collects rollout data from each in its own
  single-shape env, and trains one unified policy via behavior
  cloning/DAgger against the shape-randomized-per-episode env (Task 1's
  observation schema, no distractors) — implementing the UniDexGrasp++
  GiGSL pattern's distillation step. Implementer picks the exact
  imitation-loss formulation (documented in the script's own module
  docstring, citing the spec) since the spec deliberately leaves this
  choice to the implementing task.
- Test: `tasks/franka/tests/test_distillation_data_collection.py` (new
  — unit-test the rollout-collection/relabeling logic in isolation from
  any actual GPU training, e.g. with a stub policy and a stub env
  producing deterministic fake trajectories).

**Interfaces:**
- Consumes: Task 2/3's specialist checkpoint file paths, Task 1's
  observation schema.
- Produces: `scripts/distill_specialists.py`'s CLI (documented in its own
  `--help`), consumed by Task 5's actual training run.

- [ ] **Step 1: Write failing unit tests** for the rollout-collection/
  relabeling logic using stub policies/envs (no real checkpoints or GPU
  needed for this test).

- [ ] **Step 2: Run tests, confirm they fail.**

- [ ] **Step 3: Implement `scripts/distill_specialists.py`.**

- [ ] **Step 4: Run tests, confirm they pass.**

- [ ] **Step 5: Commit.**

```bash
git add scripts/distill_specialists.py tasks/franka/tests/test_distillation_data_collection.py
git commit -m "feat: add specialist-to-unified-policy distillation pipeline"
```

---

## Task 5 — Run distillation (cloud)

**Files:** none new — runs Task 4's script against Task 2/3's real
checkpoints.

**Interfaces:**
- Consumes: `scripts/distill_specialists.py`, real specialist checkpoint
  paths from Tasks 2/3.
- Produces: one distilled unified-policy checkpoint, consumed by Task 6.

- [ ] **Step 1**: Provision cloud instance; record timestamp.
- [ ] **Step 2**: Run `scripts/distill_specialists.py` against the real
  specialist checkpoints produced by Tasks 2/3.
- [ ] **Step 3**: Instrumented eval of the distilled (pre-fine-tune)
  policy, per-shape discovery rate + video.
- [ ] **Step 4**: Full teardown; report cost.
- [ ] **Step 5**: No code changes expected; if the run surfaces a real
  bug in Task 4's script, fix it, re-verify Task 4's unit tests still
  pass, and commit the fix separately.

---

## Task 6 — RL fine-tune + final verdict

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry.
- Modify: `kb/wiki/experiments/dice-pick-demo.md` (or a new
  `kb/wiki/experiments/` article if this warrants its own, per this
  repo's kb-maintenance convention — check which fits before adding).

**Interfaces:**
- Consumes: Task 5's distilled checkpoint.
- Produces: final per-shape discovery-rate comparison (unified policy
  vs. each specialist), the spec's pre-registered falsification check.

- [ ] **Step 1**: Provision cloud instance; record timestamp.
- [ ] **Step 2**: RL-fine-tune the distilled policy (PPO, per GiGSL's
  iterate-distillation-and-RL pattern), checkpoint-resumed from Task 5's
  distilled weights.
- [ ] **Step 3**: Instrumented eval + video review per shape/seed.
- [ ] **Step 4**: Full teardown; report total cumulative cost across
  Tasks 2/3/5/6 combined against the $15 cap.
- [ ] **Step 5**: Verdict against the spec's falsifiable hypothesis:
  compare the fine-tuned unified policy's per-shape discovery rate
  against each specialist's own rate (Task 2/3's numbers). State clearly
  whether each shape passed or failed the "not meaningfully below its
  specialist" bar.
- [ ] **Step 6**: Update `ROADMAP.md` and the relevant kb article with
  the verdict, per this repo's continuous-kb-update convention.
- [ ] **Step 7: Commit.**

```bash
git add ROADMAP.md kb/wiki/experiments/dice-pick-demo.md
git commit -m "verdict: unified multi-die specialist-distillation experiment"
git push origin main
```

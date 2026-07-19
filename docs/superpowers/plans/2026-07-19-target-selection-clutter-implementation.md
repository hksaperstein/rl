# Target selection among distractor dice: implementation plan (Experiment 2 of the multi-die RL arc)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** extend the finished unified d12/d20 single-object policy
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-d20-mixed/seed42/2026-07-19_12-53-35/model_2998.pt`,
8/8 discovery for both shapes with exactly one die in the scene) into a
3-die scene (1 commanded target + 2 distractor dice, drawn from
{d12, d20}) via a 3-stage distractor-count curriculum (0 → 1 → 2 active
distractors), testing whether curriculum + a new fixed-size zero-padded
distractor-distance observation term (DexSinGrasp's own `d_t^S`
mechanism) is sufficient to preserve most of that 8/8 discovery rate
under clutter, with the reward function and target-identification
mechanism left completely unchanged.

**Architecture:** (1) a genuinely new scene topology — two new sibling
`RigidObjectCfg` distractor slots on a new `InteractiveSceneCfg`
subclass, following `tasks/franka/dice_scene_cfg.py`'s `DiceSceneCfg`
precedent for multiple simultaneous dice in one scene (never previously
exercised inside a `ManagerBasedRLEnv`); (2) one new, additive
observation term, `distractor_distance_summary` (K=2, hard-zero-padded
per curriculum stage); (3) three new curriculum-stage env cfg classes
(SO/D1/D2), SO trained fresh (new 43-dim schema, cannot resume the
41-dim checkpoint), D1/D2 checkpoint-resumed from the prior stage, all
single-seed (seed42); (4) desktop-first GPU dispatch, cloud fallback.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`lift_env_cfg.py`, `dice_lift_joint_env_cfg.py`, `mdp.py`, a new
`tasks/franka/distractor_observations.py`), desktop GPU dispatch
(`scripts/check_gpu_availability.sh` / `scripts/run_on_desktop_gpu.sh`),
GCP cloud fallback (`docs/cloud/dispatch-checklist.md`).

Spec: `docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md`.
Research: `docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`.
Template/precedent for this plan's own structure and rigor:
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`
(same arc, just-finished — see `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`
for its closing verdict).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Global Constraints

- **Two shapes only: d12 and d20**, for both target and distractor roles.
  Do not introduce d8/d10/d4 as distractors or targets in this plan (both
  are genuinely null shapes for this project's own grasp task — see the
  spec's "Distractor shape population" section for why untested-shape
  co-presence would confound this experiment's own variable).
- **Distractor count: 0 → 1 → 2, staged (Stage SO/D1/D2).** Do not train
  or evaluate a 3+-distractor configuration in this plan.
- **Flat, non-overlapping tabletop placement only** (disjoint reset-range
  regions per entity, reusing the existing `reset_root_state_uniform`
  event mechanism per-entity). Do not build a heap/piled/occluding
  arrangement, a minimum-spacing rejection sampler, or any
  singulation-specific mechanism (finger-flicking, palm-rubbing, or any
  other DexSinGrasp-specific dexterous-hand technique — does not transfer
  to a Franka parallel-jaw gripper, see spec's "Reward and termination"
  section).
- **Reward function and terminations unchanged.** `RewardsCfg` and
  `TerminationsCfg` (`object_dropping` scoped to `scene["object"]` only)
  are inherited byte-identical from `FrankaDieLiftJointD12D20MixedEnvCfg`.
  Do not add a distractor-avoidance/disturbance reward term in this plan
  (see spec's Scope section — this is a follow-on hypothesis only if the
  curriculum+observation mechanism is falsified, not a parallel mechanism
  to test simultaneously).
- **Target identity is NOT a new observation flag** — `scene["object"]`
  is structurally the commanded die by scene-topology construction (a
  per-env-cfg-constant, cfg-construction-time property, matching
  `die_shape_class`'s own existing convention), same as every prior env
  cfg in this file. The existing, unchanged `object_position` /
  `target_object_position` terms already are DexSinGrasp's `s_t^O`. Do
  not add a "which entity is the target" observation term.
- **Ground-truth object-state observations only** — no vision-detector
  integration in this plan.
- **Lift-only task horizon** — no new carry-to-goal criterion beyond what
  `FrankaDieLiftJointD12D20MixedEnvCfg` already inherits.
- **Single seed (seed42) per stage** — matching the checkpoint this plan
  starts from. Multi-seed replication is explicitly deferred to a
  follow-on.
- **Scene topology: exactly 3 simultaneous dice per env at most** (1
  target + 2 distractors). Do not extend the scene to more entities in
  this plan.
- **Execution backend: desktop-first, cloud fallback** (2026-07-18
  standing policy, CLAUDE.md's "Pi-as-primary-agent GPU dispatch" — this
  supersedes the *template* plan's cloud-only default, which predates the
  desktop dispatch infra). For every task that launches Isaac Sim:
  1. Check `scripts/check_gpu_availability.sh`. `TARGET=desktop` (exit 0)
     → dispatch via `scripts/run_on_desktop_gpu.sh` (see its own
     docstring; default mode blocks and streams output — use that, not
     `--detach`, for training runs this plan needs a real result from
     before proceeding). `TARGET=cloud` (exit 1, BUSY) or unclear (exit
     2, UNKNOWN) → fall back to `docs/cloud/dispatch-checklist.md`'s
     recipe. **Never treat "can't tell" as a green light for desktop** —
     UNKNOWN routes to cloud (or stop), never an assumed-available
     desktop.
  2. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim** (its
     blocking instruction, cost-cap paragraph, environment-conventions
     block, and bug-handling-discipline block) — this project's own
     standing instruction after repeatedly reconstructing these from
     memory and dropping them.
  3. **`flock -o` is not automatic.** `run_on_desktop_gpu.sh` does not
     wrap the dispatched command in a lock itself — the command string
     shipped to the desktop (or run on a cloud instance) must itself be
     `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."`.
     Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
     (empty = clear) before dispatch, not a process-name/path grep.
  4. **Non-headless on desktop** (CLAUDE.md's standing "the user wants to
     watch" instruction, and Tasks 5/6 of the just-finished experiment
     both ran this way) — do not pass `--headless`. **Headless only** on
     the cloud fallback (the standing, confirmed exception).
  5. **Full teardown after any cloud task**: verify zero
     instances/disks/snapshots remain (`scripts/check_cloud_state.sh`)
     before marking the task done. Desktop dispatch: verify
     `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
     `check_gpu_availability.sh` are all clear/AVAILABLE afterward — the
     exact verification Task 6 of the prior experiment already
     established as this project's own bar, not a new requirement.
- **Cost cap: notify the controller if cumulative cloud spend across
  Tasks 4/5/6 (the three training stages) combined exceeds $5.** This is
  a *cloud-fallback safety backstop*, not an expected spend — desktop
  dispatch is expected to bring actual cost to $0, matching Tasks 5/6 of
  the just-finished experiment (both $0, desktop-only). If cloud is
  needed: this plan trains exactly 3 sequential single-seed runs (SO
  fresh + D1/D2 resumed), a small fraction of the prior experiment's 21+
  runs across multiple shapes/seeds that together cost ≈$5.87 of a $15
  cap — a $15 cap would be a stale copy-paste, not a considered number
  for this plan's real scale. `docs/cloud/franka-cloud-shakedown.md`'s
  own measured baseline (one full 1500-iteration `g2-standard-4`+L4 SPOT
  run, including ~15-20min install overhead, ≈55min instance existence)
  cost ≈$0.35 total. Per `dispatch-checklist.md`'s "provision ONE
  instance, run jobs sequentially" rule, all 3 stages plus their eval
  runs would share one instance's install overhead — a pessimistic
  estimate (3× training wall-clock + one shared install + a 2× SPOT-
  preemption-retry buffer) lands around $2-2.5, so $5 leaves a real
  margin without inheriting a cap sized for a 21-run experiment. Track by
  instance-uptime × the published SKU rate (`franka-cloud-shakedown.md`)
  if cloud is actually used; each cloud-touching task logs its own
  instance creation timestamp and reports an elapsed-cost estimate before
  teardown.
- **TDD discipline for pure-Python pieces** (Task 2's observation-math
  module): write failing tests first, confirm the failure, implement,
  confirm green — matching the template plan's Task 1 and this project's
  established `tasks/franka/shape_observations.py` +
  `tests/test_mdp_shape_observations.py` precedent exactly.
- **Real evidence over proxies at every eval**: instrumented
  `max_height_gain`/`max_consecutive_lifted_steps` numbers per shape AND
  a reviewed eval video — not a shaped reward scalar or an exit code
  alone. Use `franka_checkpoint_review.py`'s current (post-`977a748`)
  MIN-over-a-fixed-early-window settle-detection logic, not the older
  flatness-window heuristic — this project has now found and fixed real
  settle-detection bugs on 2 separate occasions in the immediately prior
  experiment; do not re-derive a third, different measurement approach.
- **Report exactly as observed, including Stage SO's own gate result even
  if it passes cleanly** — this project's own "report exactly as
  observed" convention (spec's "Success/failure reporting" section), not
  just the final headline number.
- Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line — each task
  below is executed by a freshly dispatched session, so use that
  session's real ID (the just-finished experiment's own commits show
  different tasks using different session IDs than the one that wrote
  the plan — do not copy a fixed ID from this document).

---

## Task 1 — Scene topology: 3-die clutter scene + SO/D1/D2 curriculum env cfgs

**Files:**
- Modify: `tasks/franka/lift_env_cfg.py` — add a new per-env-cfg constant
  `active_distractor_count: int = 0` on `FrankaLiftEnvCfg` (same pattern
  as the existing `die_shape_class`/`die_shape_classes_per_env`
  constants immediately above it in that file — document it the same
  way, default 0 so every existing env cfg in the repo is unaffected).
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add:
  - `FrankaDieLiftTargetSelectionSceneCfg(FrankaLiftSceneCfg)` — a new
    `InteractiveSceneCfg` subclass adding two new sibling `RigidObjectCfg`
    fields, `distractor_1`/`distractor_2` (prim paths
    `{ENV_REGEX_NS}/Distractor1`/`Distractor2`), following
    `tasks/franka/dice_scene_cfg.py`'s `DiceSceneCfg` pattern for multiple
    simultaneous `RigidObjectCfg` siblings on one scene cfg (that file's
    `die_d4`/`die_d8`/.../`die_d20` fields are the direct precedent, cited
    by the spec) — the first time this pattern is exercised inside a
    `ManagerBasedRLEnv` rather than a scripted `InteractiveScene` demo.
    Give each field a placeholder single-`UsdFileCfg` spawn default at
    cfg-definition time (e.g. d12 at its own already-verified 48mm scale,
    0.001476, `_D20_RIGID_PROPS`, mass 0.216kg — matching every other
    rigid-body default in this file); every real stage class overrides
    `.spawn` in its own `__post_init__`, exactly like `self.scene.object`
    already does throughout this file.
  - `TargetSelectionEventCfg(EventCfg)` (import `EventCfg` from
    `.lift_env_cfg`) — a new `EventCfg` subclass declaring two additional
    `EventTermCfg` fields, `reset_distractor_1_position`/
    `reset_distractor_2_position`, both `mdp.reset_root_state_uniform`
    (same function the existing `reset_object_position` term uses) with
    a placeholder `pose_range`/`asset_cfg` at declaration time, real
    values set per-stage in `__post_init__`. (Design note for the
    implementer: declare these as real configclass fields on a subclass,
    not dynamically added post-`__init__` attributes on a plain `EventCfg`
    instance — this repo has no precedent either way for the latter, and
    the former exactly mirrors how `FrankaDieLiftTargetSelectionSceneCfg`
    above and every existing `scene: <SceneCfgSubclass> = ...`/
    `observations: <ObservationsCfgSubclass> = ...` field override in
    this codebase already works, so it's the lower-risk choice — no need
    to test an unproven Isaac Lab manager-collection behavior when a
    proven pattern is directly available.)
  - `active_distractor_count`-aware curriculum env cfg classes, each
    extending `FrankaDieLiftJointD12D20MixedEnvCfg` (inheriting its own
    already-proven d12/d20 round-robin *target* population, 48mm-parity
    scales, `scene.replicate_physics = False`, `die_shape_classes_per_env`
    mechanism unchanged) and overriding
    `scene: FrankaDieLiftTargetSelectionSceneCfg = FrankaDieLiftTargetSelectionSceneCfg(num_envs=4096, env_spacing=2.5)`
    and `events: TargetSelectionEventCfg = TargetSelectionEventCfg()`:
    - `FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg` — Stage SO:
      `active_distractor_count = 0`; both distractor slots **parked** (a
      degenerate zero-width `pose_range` at a fixed off-workspace
      position, outside the arm's reachable volume — reuses
      `reset_root_state_uniform` with a parameter choice, not new
      event-handling code, per the spec).
    - `FrankaDieLiftJointD12D20TargetSelectionD1EnvCfg` — Stage D1:
      `active_distractor_count = 1`; `distractor_1` gets a REAL
      `MultiAssetSpawnerCfg(assets_cfg=[d12_cfg, d20_cfg], random_choice=True)`
      population (see the design note below on why `random_choice=True`,
      not `False`) plus a real disjoint reset-range event; `distractor_2`
      stays parked exactly as in Stage SO.
    - `FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg` — Stage D2: both
      `distractor_1`/`distractor_2` get their own independent real
      `MultiAssetSpawnerCfg(random_choice=True)` populations + disjoint
      reset-range events.
    - `_PLAY` eval variants, **one pinned-target-shape class per stage
      per shape (6 total, not 3)**: `..._PLAY_D12Target` /
      `..._PLAY_D20Target` for each of SO/D1/D2. Each overrides
      `self.scene.object.spawn` to a single `UsdFileCfg` (d12 at 0.001476
      or d20 at 0.001585 — reuse directly, do not re-derive), sets
      `self.die_shape_class` explicitly (`"d12"`/`"d20"`) and
      `self.die_shape_classes_per_env = None`, reduces `num_envs=8`
      (this arc's own established eval `num_envs`, not the usual 50 —
      matching every specialist-eval `_PLAY` invocation's actual usage in
      this arc, e.g. `franka_checkpoint_review.py --num_envs 8`) and
      `enable_corruption = False`. **Distractor slots are NOT pinned to a
      single shape at eval** — only the target needs pinning per the
      spec ("target shape fixed per eval run... exactly as every
      specialist eval in this arc already does"); distractors keep
      whatever `MultiAssetSpawnerCfg`/parked state their own stage
      already set. Naming/exact class layout is the implementer's call
      within this constraint — 6 explicit classes matches this file's
      own established convention (e.g. separate `D8Big`/`D10Big`/`D12Big`
      classes rather than one parametrized class); do not build a new
      generic parametrization mechanism for this.
- Create: `scripts/_diag_target_selection_clutter_scene_check.py` — new
  diagnostic (extends `scripts/_diag_d12d20_mixed_env_check.py`'s exact
  pattern: build a real, small (`num_envs=8-16`) live env, no training,
  and cross-check predicted vs. actual live state).

**Interfaces:**
- Consumes: `FrankaDieLiftJointD12D20MixedEnvCfg` (target population/scale
  constants, `scene.replicate_physics=False`), `tasks/franka/dice_scene_cfg.py`'s
  `DiceSceneCfg` (multi-`RigidObjectCfg`-sibling precedent, read not
  imported), `mdp.reset_root_state_uniform`'s real semantics (see Step 1).
- Produces: 3 curriculum-stage env cfg classes (+ 6 pinned-target `_PLAY`
  eval variants), consumed by Task 2 (adds the new observation term to
  these same classes' `observations` field), Task 3 (wires `--variant`
  strings to them), and Tasks 4/5/6 (trains/evals each stage).

- [ ] **Step 1: Before writing any reset-range numbers, directly read
  `mdp.reset_root_state_uniform`'s real source** (Isaac Lab's own
  `isaaclab.envs.mdp.events` module — this repo's own established
  convention is to never assume an Isaac Lab library semantic without a
  direct source read, e.g. Task 3's `random_choice` mechanism correction
  and Task 5's `spawn_multi_asset` round-robin confirmation in the prior
  experiment) to confirm whether `pose_range`'s `x`/`y`/`z` keys are
  sampled as an offset added to each asset's own `default_root_state`
  position, or as an absolute world-frame range. This determines how to
  compute genuinely disjoint sub-regions for target/`distractor_1`/
  `distractor_2` (each entity's own `init_state.pos` differs, so an
  "offset" semantics means disjoint *offset ranges* is not the same thing
  as disjoint *world regions* unless each entity's own base `init_state.pos`
  is also chosen with that in mind). Record the finding in the class
  docstrings, matching this file's own existing documentation rigor.

- [ ] **Step 2: Add the scene/event/env-cfg classes** per the Files
  section above. Choose concrete disjoint placement sub-regions and
  each parked-distractor's off-workspace position — this is the spec's
  own explicitly-deferred design choice, not resolved by the spec itself;
  use Step 1's confirmed semantics, keep each entity's own workspace
  comfortably within the arm's reachable volume (`CommandsCfg`'s existing
  `pos_x=(0.4,0.6), pos_y=(-0.25,0.25)` goal range is one concrete
  existing reference point for "reachable"), and account for each die's
  own real ~18-30mm 48mm-scale footprint when checking for adjacent-region
  overlap. Do not treat this as verified until Step 4 confirms it live.

  **Design note — why each distractor's `MultiAssetSpawnerCfg` must use
  `random_choice=True`, not `False`:** the target's own population already
  uses `random_choice=False` (deterministic round-robin,
  `env_index % len(assets)`, confirmed by direct source read in the prior
  experiment). If a distractor slot reused the *same* 2-element
  `[d12_cfg, d20_cfg]` list with `random_choice=False`, its assigned shape
  would ALSO be `env_index % 2` — identical to the target's own formula,
  meaning every single env's distractor would deterministically match its
  own target's shape, never a cross-shape pairing. The spec requires both
  same-shape and cross-shape regimes pooled in one training population;
  `random_choice=True` (an independent per-env-index `random.choice(...)`
  draw, still fixed once at scene-spawn time, never per-episode-resampled
  — confirmed mechanism per the prior experiment's Task 3 finding) is
  required for that, not just a stylistic choice. `distractor_1` and
  `distractor_2` use two SEPARATE `MultiAssetSpawnerCfg` instances (one
  per scene field), so their draws are independent of each other too.

- [ ] **Step 3: Write and run `scripts/_diag_target_selection_clutter_scene_check.py`**
  (non-headless, no training — the two explicit empirical risk-checks the
  spec's "Explicit known-weak-points" section requires before any Stage
  D2 config is trusted for real training):
  (a) build a real, small Stage D2 env and confirm multiple simultaneous
  `MultiAssetSpawnerCfg` fields (target + both distractor slots) coexist
  correctly on one `InteractiveSceneCfg` with `replicate_physics=False` —
  cross-check each entity's live USD-authored scale/shape against its own
  spawner's expected per-env assignment, the same double-check method
  `_diag_d12d20_mixed_env_check.py` already used (predicted vs. live
  `observation_manager` value vs. live USD-authored scale, all three
  agreeing);
  (b) a spawn-and-settle check: step physics until entities settle, then
  verify no entity (target or either distractor, across every env and
  every random per-env shape assignment) ends up off the table or outside
  the arm's reachable workspace, and that no two entities' reset ranges
  can produce an overlapping footprint at reset time given each die's own
  real size. Report both results explicitly (pass/fail), not just "ran
  without crashing."

- [ ] **Step 4: If Step 3 finds a real bug** (misplacement, spawner
  interference, off-table entities, overlap) — fix it and re-run Step 3
  to confirm, in this same task, per this project's bug-handling
  discipline (do not defer a found bug to a later task).

- [ ] **Step 5: Commit.**

```bash
git add tasks/franka/lift_env_cfg.py tasks/franka/dice_lift_joint_env_cfg.py \
        scripts/_diag_target_selection_clutter_scene_check.py
git commit -m "feat: 3-die target-selection-in-clutter scene topology + SO/D1/D2 curriculum env cfgs"
```

---

## Task 2 — New observation term: `distractor_distance_summary`

**Files:**
- Create: `tasks/franka/distractor_observations.py` — pure-tensor math, NO
  isaaclab/pxr/torch-sim imports beyond plain `torch` (mirrors
  `tasks/franka/shape_observations.py`'s established split: `mdp.py`
  reads live simulated state and delegates computation to pure functions
  here). One function,
  `distractor_distance_summary(target_pos: torch.Tensor, distractor_1_pos: torch.Tensor, distractor_2_pos: torch.Tensor, active_distractor_count: int) -> torch.Tensor`
  — shape `(num_envs, 2)`, column `i` = Euclidean distance between
  `target_pos` and `distractor_{i+1}_pos` for `i < active_distractor_count`,
  **hard-zeroed** (not the real, possibly-large, parked-off-table
  distance) for `i >= active_distractor_count` — DexSinGrasp's own
  literal zero-padding convention (spec: "padded with zeros if the true
  distractor count is below" the max), matching K=2 fixed-size design.
  World-frame positions in, frame choice doesn't affect a scalar
  distance (matches spec's own framing note).
- Modify: `tasks/franka/mdp.py` — add a thin wrapper
  `distractor_distance_summary(env, object_cfg=SceneEntityCfg("object")) -> torch.Tensor`
  reading `env.scene["object"]`/`env.scene["distractor_1"]`/
  `env.scene["distractor_2"]`'s `.data.root_pos_w` and
  `env.cfg.active_distractor_count`, delegating to the pure function
  above — same thin-wrapper pattern `object_shape_class_onehot`/
  `object_geometry_descriptor` already use.
- Modify: `tasks/franka/lift_env_cfg.py` — add a NEW
  `TargetSelectionObservationsCfg(ObservationsCfg)` with a nested
  `PolicyCfg(ObservationsCfg.PolicyCfg)` adding
  `distractor_distance_summary = ObsTerm(func=mdp.distractor_distance_summary)`,
  re-declaring `policy: PolicyCfg = PolicyCfg()` at the outer level
  (follow this file's own nested-`configclass`-override idiom exactly —
  read `ObservationsCfg`/`ObservationsCfg.PolicyCfg`'s own
  `__post_init__` first). **Do NOT add this term to the shared base
  `ObservationsCfg.PolicyCfg`** — unlike `shape_class`/
  `geometry_descriptor` (safe to broadcast unconditionally since every
  env cfg has a `die_shape_class` constant), `distractor_distance_summary`
  requires `scene["distractor_1"]`/`scene["distractor_2"]` to exist,
  which only Task 1's new clutter-scene env cfgs have — adding it
  unconditionally would `KeyError` on every single-die env cfg already in
  this file (`ik-cube`, `joint-die`, `joint-die-d12-d20-mixed`, etc).
  This is an explicit correction relative to how the schema grew last
  time (Task 1 of the template plan), not an oversight to flag — the new
  term must be genuinely additive to the *new* clutter env cfgs only.
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — give Task 1's
  3 curriculum-stage classes (+ 6 `_PLAY` variants)
  `observations: TargetSelectionObservationsCfg = TargetSelectionObservationsCfg()`.
  This makes Task 2 depend on Task 1 having already landed those classes.
- Test: `tests/test_distractor_observations.py` (new — plain pytest, no
  Isaac Sim/env stub needed, matching `tests/test_mdp_shape_observations.py`'s
  own scope: test the pure function directly with raw tensors, not
  `mdp.py`'s thin wrapper, which imports isaaclab at module level and is
  only exercised by actually running the env).

**Interfaces:**
- Consumes: Task 1's `active_distractor_count` constant and
  `distractor_1`/`distractor_2` scene entities.
- Produces: `mdp.distractor_distance_summary`, importable as
  `mdp.distractor_distance_summary`, wired into Task 1's env cfgs via
  `TargetSelectionObservationsCfg`, growing the observation schema from
  41 to 43 dims for those env cfgs only (verified in Step 6 below).

- [ ] **Step 1: Write failing unit tests** for
  `distractor_distance_summary` in `tests/test_distractor_observations.py`:
  correct Euclidean distance for an active slot; exact zero for an
  inactive slot regardless of the input position's real value (the
  hard-zero-not-real-distance requirement — test this explicitly with a
  distractor position far from the target, confirming the output is
  still exactly `0.0`, not a large real distance); shape `(num_envs, 2)`;
  `active_distractor_count` of 0/1/2 each produce the right zero/nonzero
  column pattern; batch of >1 env with per-row-varying positions produces
  per-row-varying (not broadcast-constant) distances — this term is
  genuinely per-environment-varying, unlike `shape_class`/
  `geometry_descriptor`'s per-env-cfg-constant broadcast, so this test
  matters as a real behavioral difference to assert.

- [ ] **Step 2: Run tests, confirm they fail** (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement `distractor_distance_summary`** in the new
  `tasks/franka/distractor_observations.py`.

- [ ] **Step 4: Run tests, confirm they pass.**

- [ ] **Step 5: Implement `mdp.py`'s thin wrapper and
  `TargetSelectionObservationsCfg`**, wire it into Task 1's env cfg
  classes per the Files section.

- [ ] **Step 6: Confirm (via a quick local, no-GPU-required Python import
  check, not a sim launch) that Task 1's curriculum env cfgs' observation
  space grows by exactly 2 dims (41→43) and that every OTHER existing env
  cfg in `dice_lift_joint_env_cfg.py`/`lift_env_cfg.py` is completely
  unaffected** (still 41 dims, still uses the original
  `ObservationsCfg`/`PolicyCfg`, no `KeyError` risk introduced).

- [ ] **Step 7: Commit.**

```bash
git add tasks/franka/distractor_observations.py tasks/franka/mdp.py \
        tasks/franka/lift_env_cfg.py tasks/franka/dice_lift_joint_env_cfg.py \
        tests/test_distractor_observations.py
git commit -m "feat: add distractor_distance_summary observation term (DexSinGrasp d_t^S mechanism)"
```

---

## Task 3 — Wiring: `train_franka.py` / `franka_checkpoint_review.py` / `sync_run_to_gcs.py`

**Files:**
- Modify: `scripts/train_franka.py` — add 3 new `--variant` choices
  (`joint-die-target-selection-so` / `-d1` / `-d2`), each importing and
  constructing the matching Task 1 env cfg class (mirrors every existing
  `elif args_cli.variant == ...` branch exactly), new `_log_suffix`
  entries (`_jointdietargetselectionso` / `d1` / `d2`), and new help text
  describing each stage (cite this plan + the spec, matching this file's
  own existing per-variant help-text convention).
- Modify: `scripts/franka_checkpoint_review.py` — add the same 3
  `--variant` choices, importing the appropriate one of the **6**
  pinned-target `_PLAY` classes (Task 1) based on BOTH `--variant` and a
  new CLI flag to select which shape is pinned as target for this eval
  run (e.g. `--eval_target_shape {d12,d20}`, required when `--variant` is
  one of the 3 target-selection variants) — since a single `--variant`
  string alone is ambiguous between the two pinned-target classes per
  stage. Follow this script's own existing `if/elif` dispatch pattern;
  do not restructure the script's overall shape.
- Modify: `scripts/sync_run_to_gcs.py` — add 3 new `VARIANT_MAP` entries
  (`train_franka_jointdietargetselectionso` → `joint-die-target-selection-so`,
  same for `-d1`/`-d2`), matching `train_franka.py`'s own new
  `_log_suffix` values exactly (log-root dirname is `train_franka` +
  suffix, per this script's own `derive_variant` docstring).

**Interfaces:**
- Consumes: Task 1's 3 curriculum env cfg classes + 6 `_PLAY` classes,
  Task 2's observation schema (already wired into those classes, nothing
  further needed here).
- Produces: `--variant joint-die-target-selection-{so,d1,d2}` on all 3
  scripts, consumed by Tasks 4/5/6's real training/eval dispatches.

- [ ] **Step 1: Wire `train_franka.py`** per the Files section.
- [ ] **Step 2: Wire `franka_checkpoint_review.py`** per the Files
  section, including the new `--eval_target_shape` flag.
- [ ] **Step 3: Wire `sync_run_to_gcs.py`**'s `VARIANT_MAP`.
- [ ] **Step 4: Smoke-check** — a quick local, no-GPU-required Python
  import/argparse check (`--help` on all 3 scripts, no sim launch) that
  the new choices parse and resolve to the right classes without error.
- [ ] **Step 5: Commit.**

```bash
git add scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: wire --variant joint-die-target-selection-{so,d1,d2} into train/eval/sync scripts"
```

---

## Task 4 — Train Stage SO (fresh) — internal sanity gate before D1/D2

**Files:** none new — runs Task 1-3's code against a fresh PPO training
run. Per CLAUDE.md's flock/GPU-dispatch conventions and this plan's
Global Constraints' "Execution backend" section — copy
`docs/cloud/dispatch-checklist.md`'s blocks verbatim into this dispatch
if it falls back to cloud.

**Interfaces:**
- Consumes: `--variant joint-die-target-selection-so`
  (`FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg`), trained FROM
  SCRATCH — **not** resumed from `model_2998.pt`, per the spec's explicit
  resolution of the dimensionality-mismatch snag (41-dim checkpoint vs.
  this stage's 43-dim schema; `rsl_rl`'s `ActorCritic` network shape is
  fixed at construction from the checkpoint's own dims, no
  cross-dimensionality warm-start mechanism exists in this codebase).
- Produces: Stage SO's own checkpoint, consumed by Task 5 as the resume
  source. Produces the internal sanity-gate verdict that Task 5/6 are
  conditioned on.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch via
  `scripts/run_on_desktop_gpu.sh` if AVAILABLE, else cloud fallback per
  the checklist. Record dispatch target and (if cloud) instance creation
  timestamp.
- [ ] **Step 2**: Train `--variant joint-die-target-selection-so`,
  `--num_envs 4096`, 1500 iterations (this project's established
  from-scratch default — no `--checkpoint` flag), non-headless if
  desktop / headless if cloud, `--seed 42` if not already the agent cfg's
  default.
- [ ] **Step 3**: Run the internal sanity-gate eval — BOTH pinned-target
  `_PLAY` variants (`joint-die-target-selection-so` with
  `--eval_target_shape d12` and `d20`), `num_envs=8`, the full 3-die
  topology (both distractors present but parked/inert), matching this
  arc's own instrumented lift-threshold convention
  (`franka_checkpoint_review.py`, video + `max_height_gain`/
  `max_consecutive_lifted_steps`, current post-`977a748` settle logic).
- [ ] **Step 4: Internal sanity-gate check (pre-registered in the spec)**
  — if EITHER shape's discovery rate is below 7/8, this specifically
  falsifies "the new scene entities + new always-zero observation dims
  are inert," a materially different and earlier failure than real
  distractor-pressure collapse. **STOP and report to the controller**
  before Task 5 proceeds — do not silently continue to D1 if this gate
  fails, per this plan's Phase-1-gate-before-Phase-2 precedent (the prior
  experiment's Task 3.5 gate before Task 4). If the gate passes (≥7/8
  both shapes), report the exact numbers (not just pass/fail) and
  proceed.
- [ ] **Step 5**: Full teardown if cloud; report elapsed cost against the
  $5 cap. Desktop: verify `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh` clear.
- [ ] **Step 6**: No code changes expected; if the run surfaces a real
  bug in Tasks 1-3's code, fix it, re-verify, and commit the fix
  separately before re-running.

---

## Task 5 — Train Stage D1 (resume from Stage SO) — GATE before Task 6

**Files:** none new.

**Interfaces:**
- Consumes: Stage SO's checkpoint (Task 4), `--variant
  joint-die-target-selection-d1`.
- Produces: Stage D1's checkpoint, consumed by Task 6.

**Precondition:** Task 4's internal sanity gate passed (≥7/8 both
shapes). Do not start this task otherwise — report back to the
controller instead.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per the
  Execution backend constraints.
- [ ] **Step 2**: Train `--variant joint-die-target-selection-d1`,
  `--checkpoint <Stage SO's checkpoint path>`, 1500 ADDITIONAL iterations
  on top of the resumed count (this schema is stable across SO→D1→D2, a
  normal same-schema PPO resume — do NOT pass `--policy_only_checkpoint`,
  that flag is only for Task 6-of-the-prior-experiment's BC-to-PPO
  distillation-checkpoint special case, not applicable here). Compute the
  absolute `--max_iterations` target from Stage SO's own actual resumed-at
  iteration count (`train_franka.py`'s own printed "Resumed from ... at
  iteration ..." message — verify against that directly, per the prior
  experiment's Task 6 precedent of confirming this arithmetic rather than
  assuming it; illustratively, if SO's final checkpoint is `model_1499.pt`,
  resuming with `--max_iterations 2999` gives 1500 more iterations ending
  at `model_2998.pt`, matching that precedent exactly — verify the real
  numbers for this run rather than trusting this example).
- [ ] **Step 3**: Run the same 2-pinned-target-shape instrumented eval as
  Task 4 (`--variant joint-die-target-selection-d1`,
  `--eval_target_shape d12`/`d20`, `num_envs=8`, full 3-die topology with
  `distractor_1` now real/active and `distractor_2` still parked), video
  + instrumented height data for both shapes.
- [ ] **Step 4**: Report Stage D1's own numbers explicitly (not just
  proceed silently) — this is an intermediate data point on the
  curriculum's own trajectory, not itself the pre-registered
  falsification bar (that's Stage D2, Task 6), but worth recording per
  this plan's "report exactly as observed" convention.
- [ ] **Step 5**: Full teardown/verification per Task 4's Step 5.
- [ ] **Step 6**: Bug-handling per Task 4's Step 6.

---

## Task 6 — Train Stage D2 (resume from Stage D1) — PRIMARY FALSIFICATION CHECK + verdict data

**Files:** none new.

**Interfaces:**
- Consumes: Stage D1's checkpoint (Task 5), `--variant
  joint-die-target-selection-d2`.
- Produces: Stage D2's checkpoint + the spec's pre-registered primary
  falsification verdict, consumed by Task 7.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per the
  Execution backend constraints.
- [ ] **Step 2**: Train `--variant joint-die-target-selection-d2`,
  `--checkpoint <Stage D1's checkpoint path>` (full optimizer resume, no
  `--policy_only_checkpoint`), 1500 additional iterations on top of
  Stage D1's own resumed-at iteration count (same verify-don't-assume
  arithmetic as Task 5's Step 2).
- [ ] **Step 3**: Run the same 2-pinned-target-shape instrumented eval
  (`--variant joint-die-target-selection-d2`, both distractors now real,
  `num_envs=8`, video + instrumented height data) for both d12 and d20.
- [ ] **Step 4: Primary falsification check (pre-registered in the
  spec)** — for each shape independently: if discovery rate is below
  6/8 (75%), that shape FALSIFIES "curriculum + observation (reward
  unchanged) is sufficient to preserve discovery under 2-distractor
  clutter." Report both shapes' results explicitly and separately (not
  averaged), exactly as observed, whether the bar is cleared or not. This
  is the experiment's real primary result — do not soften or average
  away a failing shape if only one falls short.
- [ ] **Step 5: Escalation note (if falsified)** — if Stage D2 falsifies
  for either shape WHILE Stage SO's own gate (Task 4) cleared cleanly,
  per the spec's own pre-registered escalation guidance: report this as
  evidence the schema extension itself is fine but real distractor
  pressure specifically collapses discovery, and flag to the controller
  that the honest next step is a genuinely different mechanism (a
  Deep-Sets/attention architecture, or a distractor-avoidance reward
  term) rather than a parameter retune within the same two mechanisms —
  **do not decide or attempt that follow-on within this task**, only
  report the finding and the spec's own pre-registered recommendation.
- [ ] **Step 6**: Full teardown/verification per Task 4's Step 5. Report
  cumulative cost across Tasks 4/5/6 against the $5 cap.
- [ ] **Step 7**: Bug-handling per Task 4's Step 6.

---

## Task 7 — Verdict + documentation

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry (Stage SO gate result,
  Stage D1 intermediate numbers, Stage D2 primary falsification result
  per shape, checkpoints, cost, teardown verification — matching the
  depth/format of the prior experiment's own ROADMAP entries, e.g. its
  "Task 6 + FINAL VERDICT" entry).
- Modify: `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`
  — this experiment is this arc's own explicit follow-on (its "Open, not
  yet decided" section already anticipates it); either extend that
  article's own "Related"/follow-on section with a link to a NEW
  dedicated kb article, or append a new section directly — check which
  fits this repo's own kb-maintenance convention before choosing (a
  genuinely new experiment likely warrants its own
  `kb/wiki/experiments/target-selection-clutter.md`, cross-linked both
  ways, following how `dice-pick-demo.md` and the multi-die-specialist
  article cross-link each other today).
- Modify: `kb/wiki/index.md` — update the one-line summary/status,
  matching the prior experiment's own closing-verdict convention.

**Interfaces:**
- Consumes: Tasks 4/5/6's real numeric results (per-shape, per-stage
  discovery rates, checkpoints, cost).
- Produces: the closed-out, cross-referenced record of this experiment's
  real result.

- [ ] **Step 1**: Verdict against the spec's pre-registered falsifiable
  hypothesis — state clearly, per shape, whether Stage D2 passed (≥6/8)
  or falsified (<6/8) the primary bar, and whether Stage SO's own
  internal gate passed. Do not average d12/d20 into one combined number;
  report both separately, matching the spec's own per-shape falsification
  design.
- [ ] **Step 2**: Include the same rigor the prior experiment's closing
  verdict used — instrumented `max_height_gain`/
  `max_consecutive_lifted_steps` numbers (not just the summary discovery
  fraction), explicit before/after comparison against the single-object
  8/8 baseline at every stage (not just Stage D2), and confirmation the
  eval videos were actually watched/frames actually inspected, not just
  the JSON trusted.
- [ ] **Step 3**: Update `ROADMAP.md` and the kb article(s) per the Files
  section above.
- [ ] **Step 4**: If the primary bar was falsified for either shape,
  explicitly carry forward the spec's own pre-registered escalation
  recommendation (Deep-Sets/attention architecture or a distractor
  reward term) into the kb article's own "Open, not yet decided" section
  as the concrete next candidate follow-on — do not leave it as a bare
  "failed" note with no forward pointer, matching this project's own
  "document the next candidate direction, don't just record failure"
  convention (CLAUDE.md's "Claude's role" section).
- [ ] **Step 5: Commit and push.**

```bash
git add ROADMAP.md kb/wiki/experiments/unified-multi-die-specialist-distillation.md \
        kb/wiki/index.md
# (add a new kb/wiki/experiments/target-selection-clutter.md here too if Step 1's judgment call creates one)
git commit -m "verdict: target-selection-among-distractor-dice experiment (Stages SO/D1/D2)"
git push origin main
```

# Target-selection clutter, Stage E1 (2→3 distractors, d12/d20 only): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** extend the finished, checkpointed D2 policy
(`gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d2/seed42/2026-07-19_21-08-07/model_5096.pt`,
8/8 both shapes at 2 active distractors) to tolerate a 3rd simultaneous
distractor (d12/d20 population only, unchanged), via (a) a mechanical
K=2→K=3 extension of the `distractor_distance_summary` observation term
(41+3=44 dims), (b) a new 2×2-grid scene topology (the existing 3-lane
y-strip has no room left for a 4th entity), and (c) a checkpoint-warm-start
weight-surgery resume from `model_5096.pt` — testing whether the same,
already-twice-validated curriculum+observation mechanism continues to
preserve most of D2's 8/8-both-shapes discovery rate at one more
distractor, in a single checkpoint-resumed training run (no further
0/1/2/3 sub-curriculum within E1 itself).

**Architecture:** (1) new, additive K=3 sibling observation
function/wrapper/cfg alongside (not replacing) the existing K=2 pieces
still used by SO/D1/D2; (2) a new `distractor_3` scene entity + event term
on new sibling scene/event cfg subclasses; (3) a real scene-topology
redesign (2×2 grid using both x and y, replacing the y-strip for this new
env cfg only) verified live via spawn-and-settle diagnostic before any
training; (4) `scripts/extend_checkpoint_observation_dims.py` reused
verbatim (43→44 dims) as the checkpoint warm-start mechanism, with an
explicitly redefined meaning for what its `--verify` gate proves at E1
(see Task 4); (5) one checkpoint-resumed training run with a real
iteration budget (not a no-op resume), desktop-first/cloud-fallback GPU
dispatch.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`lift_env_cfg.py`, `dice_lift_joint_env_cfg.py`, `mdp.py`,
`distractor_observations.py`), `scripts/extend_checkpoint_observation_dims.py`
(pure `torch`, no Isaac/GPU needed), desktop GPU dispatch
(`scripts/check_gpu_availability.sh` / `scripts/run_on_desktop_gpu.sh`),
GCP cloud fallback (`docs/cloud/dispatch-checklist.md`).

Spec: `docs/superpowers/specs/2026-07-21-target-selection-clutter-e1-3distractors-design.md`.
Research (this spec's own Tier-1 gate): `docs/superpowers/specs/research/2026-07-21-multi-shape-clutter-extension-literature.md`.
Template/precedent for this plan's own structure, rigor, and granularity:
`docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md`
(the finished SO/D1/D2 experiment this plan directly extends — see
`kb/wiki/experiments/target-selection-clutter.md` for its closing verdict).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Global Constraints

- **Two shapes only: d12 and d20**, for target, `distractor_1`,
  `distractor_2`, AND `distractor_3` alike — unchanged from D2's own
  population. Do not introduce d4/d8/d10 in this plan (see spec's Scope
  section for why — d4 fails upstream of clutter entirely; d8/d10 are
  reserved for a later, separately-gated S1 stage).
- **Distractor count: exactly 3 active, one checkpoint-resumed step from
  D2's own 2-active checkpoint.** No 0/1/2/3 sub-curriculum within E1
  itself — that ground is already validated by the finished SO/D1/D2
  stages (this plan's own Task 5 trains the single 2→3 increment only).
- **The 2×2 grid scene-layout redesign is REQUIRED, not optional, and its
  design-time numbers are NOT to be trusted for training until a live
  spawn-and-settle diagnostic passes** (Task 2, Step 4) — per the spec's
  own explicit "Explicit known-weak-points" flag. Do not skip this gate
  even though the design-time arithmetic looks safe by analogy to D2's own
  already-measured precedent.
- **The observation-term extension is purely additive** — new
  `distractor_distance_summary_3` function/wrapper/cfg alongside the
  existing K=2 versions, which are left completely untouched (still used
  unchanged by SO/D1/D2 and their `_PLAY` eval variants). Do not edit
  `distractor_distance_summary`, its `mdp.py` wrapper, or
  `TargetSelectionObservationsCfg` in place.
- **`extend_checkpoint_observation_dims.py` is reused verbatim, not
  redesigned** — it is already generic over old/new dim counts (confirmed
  by direct reading: `extend_checkpoint`/`extend_first_layer_weight` take
  `old_dim`/`new_dim` as parameters, no hardcoded 41/43). Task 4 only adds
  a documentation note about how to interpret `--verify`'s result at E1
  (see Task 4 — this is the plan's answer to "what does verification mean
  when the new column isn't hard-zeroed in real training").
- **E1's checkpoint warm-start is NOT bit-for-bit behaviorally lossless at
  t=0**, unlike Stage SO's — `distractor_3` is real/active from iteration
  0 and the relocated grid also shifts `distractor_1`/`distractor_2`'s own
  learned distance distribution. Budget and expect real adaptation
  iterations (Task 5); do not treat the `--verify` pre-training gate
  passing as evidence E1 needs no training.
- **Reward function and terminations unchanged** — `RewardsCfg`/
  `TerminationsCfg` inherited byte-identical from
  `FrankaDieLiftJointD12D20MixedEnvCfg`, exactly as D2 left them. Do not
  add a distractor-avoidance/disturbance reward term in this plan (only
  the pre-registered fallback on falsification, per the spec).
- **Target identity is NOT a new observation flag** — `scene["object"]` is
  structurally the commanded die by scene-topology construction, unchanged
  from every prior stage.
- **Ground-truth object-state observations only** — no vision-detector
  integration in this plan.
- **Single seed (seed42)**, matching the checkpoint this plan starts from.
  Multi-seed replication remains explicitly deferred (unchanged from the
  original experiment's own deferral).
- **Do not touch files belonging to other concurrent workstreams.**
  `git status` at plan-writing time shows uncommitted changes to
  `scripts/train.py` and `tasks/ar4/pickplace_graspgoal_env_cfg.py`
  belonging to the concurrent AR4-transfer workstream (see
  `docs/superpowers/plans/2026-07-21-ar4-franka-fixes-transfer-implementation.md`)
  — every task below must `git add` only the specific files it actually
  changed, never a blanket `git add -A`/`git add .`, and must re-check
  `git status` immediately before committing to confirm no other
  workstream's unstaged changes get swept in.
- **Execution backend: desktop-first, cloud fallback**, current standing
  policy. At plan-writing time (this session), `scripts/check_gpu_availability.sh`
  reported `TARGET=cloud`/exit 2 (desktop UNKNOWN — the status server at
  `home.local:8077` was unreachable, DNS resolution timed out) — this is a
  live snapshot, not a standing fact; every dispatching task below must
  re-run the check itself at dispatch time, not trust this plan's own
  snapshot. For every task that launches Isaac Sim:
  1. Check `scripts/check_gpu_availability.sh`. `TARGET=desktop` (exit 0)
     → dispatch via `scripts/run_on_desktop_gpu.sh` (default blocking mode,
     not `--detach`, for training runs this plan needs a real result
     from). `TARGET=cloud` (exit 1, BUSY) or UNKNOWN (exit 2) → fall back
     to `docs/cloud/dispatch-checklist.md`'s recipe. Never treat "can't
     tell" as a green light for desktop.
  2. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim.**
  3. **`flock -o` is not automatic** — the command string shipped to the
     desktop/cloud instance must itself be
     `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."`.
     Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
     (empty = clear) before dispatch, not a process-name/path grep.
  4. **Non-headless on desktop** (the user wants to watch) — do not pass
     `--headless`. **Headless only** on cloud fallback.
  5. **Full teardown after any cloud task**: `scripts/check_cloud_state.sh`
     clean before marking the task done. Desktop dispatch: verify
     `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
     `check_gpu_availability.sh` all clear/AVAILABLE afterward.
  6. **Note other concurrent workstreams may be contending for the same
     shared GPU quota** — the prior experiment's own Task 6 discovered a
     project-wide `GPUS_ALL_REGIONS` quota of 1 (only one cloud GPU
     instance project-wide at a time). Check `scripts/check_cloud_state.sh`
     for other live instances before assuming cloud is free; if BUSY on
     both desktop and cloud-quota grounds, report to the controller rather
     than waiting silently or picking a workaround.
- **Cost cap: $2 for this plan's Task 5 (the only GPU-spend task).** A
  judgment call, calibrated against `BACKLOG.md`/the original SO/D1/D2
  plan's own real costs: that 3-stage curriculum (SO fresh + D1 + D2
  resumed, each ~800-1500 iterations) cost ≈$1.35 total against its own $5
  cap, and D2's own single 1000-iteration resumed stage alone cost
  ≈$0.16-0.36 (cloud-partial + desktop). E1 is architecturally the same
  shape as D2's own step (one checkpoint-resumed run, ~1000-iteration
  budget) — no new curriculum stages, no from-scratch training. $2 leaves
  roughly 5-10x real headroom over D2's own single-stage cost even
  accounting for a SPOT-preemption retry or two, without inheriting a cap
  sized for a multi-stage or multi-seed experiment. Notify the controller
  if exceeded; otherwise the existing "well under, no notification needed"
  convention applies. Track cloud cost by instance-uptime × the published
  SKU rate if cloud is actually used (`docs/cloud/franka-cloud-shakedown.md`).
- **TDD discipline for the observation-term extension (Task 1).** Write
  failing tests first, confirm the failure, implement, confirm green —
  matching the original plan's own Task 2 and this project's established
  `tasks/franka/shape_observations.py` +
  `tests/test_mdp_shape_observations.py` / `tests/test_distractor_observations.py`
  precedent exactly.
- **Real evidence over proxies at every eval**: instrumented
  `max_height_gain`/`max_consecutive_lifted_steps` numbers per shape AND a
  reviewed eval video, using `franka_checkpoint_review.py`'s current
  post-`977a748` MIN-over-fixed-early-window settle-detection logic.
  Explicitly check for a "grasped the wrong die" episode, per D2's own
  precedent and per this being a newly-possible failure mode at higher
  distractor count.
- **Report exactly as observed**, including the Task 4 pre-training gate's
  result even though it is expected to pass cleanly, and the final
  discovery-rate numbers exactly as observed (not just pass/fail).
- Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line — each task
  is executed by a freshly dispatched session; use that session's real ID,
  do not copy a fixed ID from this document.

---

## Task 1 — Observation term: mechanical K=2→K=3 extension (TDD)

**Files:**
- Modify: `tasks/franka/distractor_observations.py` — add
  `distractor_distance_summary_3(target_pos, distractor_1_pos,
  distractor_2_pos, distractor_3_pos, active_distractor_count) ->
  torch.Tensor`, a `(num_envs, 3)` tensor, same hard-zero-per-slot-index
  convention as the existing K=2 function
  (`if active_distractor_count >= {1,2,3}`). Columns 0/1 must be computed
  by the IDENTICAL Euclidean-norm formula, same argument order, as the
  existing `distractor_distance_summary`'s own columns 0/1 — this identity
  is load-bearing for Task 4's checkpoint warm-start (see that task). The
  existing `distractor_distance_summary` (K=2) is left completely
  untouched.
- Modify: `tasks/franka/mdp.py` — add
  `distractor_distance_summary_3(env) -> torch.Tensor`, reading
  `env.scene["object"]`/`["distractor_1"]`/`["distractor_2"]`/
  `["distractor_3"]`'s `.data.root_pos_w` and
  `env.cfg.active_distractor_count`, delegating to the new pure function.
  Same thin-wrapper idiom as the existing `distractor_distance_summary`
  wrapper. The existing wrapper is left completely untouched.
- Modify: `tasks/franka/lift_env_cfg.py` — add
  `TargetSelectionE1ObservationsCfg(ObservationsCfg)` with a nested
  `PolicyCfg(ObservationsCfg.PolicyCfg)` adding
  `distractor_distance_summary = ObsTerm(func=mdp.distractor_distance_summary_3)`,
  re-declaring `policy: PolicyCfg = PolicyCfg()` — same nested-override
  idiom the existing K=2 `TargetSelectionObservationsCfg` already uses
  (read that class first). **Do NOT add this term to the shared base
  `ObservationsCfg.PolicyCfg`** — same reasoning as the K=2 term (only
  env cfgs with all 3 distractor scene entities can use it without a
  `KeyError`). The existing `TargetSelectionObservationsCfg` (K=2, 43-dim)
  is left completely untouched, still used unchanged by SO/D1/D2.
- Modify: `tests/test_distractor_observations.py` (existing file from the
  original experiment) — add tests for the new K=3 function alongside the
  existing K=2 tests, not a new test file.

**Interfaces:**
- Consumes: nothing new from other tasks (pure-tensor math + a thin `env`
  wrapper, same as the existing K=2 term).
- Produces: `mdp.distractor_distance_summary_3` and
  `TargetSelectionE1ObservationsCfg`, consumed by Task 2 (wires it into
  the new E1 env cfg class) and Task 4 (the 44-dim target width the
  checkpoint surgery extends to).

- [ ] **Step 1: Write failing unit tests** in
  `tests/test_distractor_observations.py`:
  - Correct Euclidean distance for each active slot (1, 2, 3 active).
  - Exact zero for every inactive slot regardless of the input position's
    real value (test explicitly with a far-away distractor position,
    confirming output is still exactly `0.0`) — same hard-zero-not-real-
    distance requirement as the K=2 term.
  - Shape `(num_envs, 3)`.
  - `active_distractor_count` of 0/1/2/3 each produce the right zero/
    nonzero column pattern.
  - Batch of >1 env with per-row-varying positions produces per-row-
    varying (not broadcast-constant) distances.
  - **Load-bearing cross-check for Task 4's checkpoint warm-start:**
    `distractor_distance_summary_3(target, d1, d2, d3, count)[:, :2]`
    equals `distractor_distance_summary(target, d1, d2, count)` element-
    wise, for the same `target`/`d1`/`d2` tensors and the same
    `active_distractor_count` value in `{0, 1, 2}` (test at least
    `active_distractor_count=2`, matching D2's real trained configuration,
    as the strongest case) — this is the exact property Task 4's
    checkpoint-surgery weight-copy assumes holds for columns 0/1 to carry
    over meaningfully; if this test fails, the K=3 function's column
    order/formula has diverged from the K=2 function's and Task 4 must
    not proceed until it's fixed.

- [ ] **Step 2: Run tests, confirm they fail** (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement `distractor_distance_summary_3`** in
  `tasks/franka/distractor_observations.py`.

- [ ] **Step 4: Run tests, confirm they pass**, including the Step 1
  cross-check.

- [ ] **Step 5: Implement `mdp.py`'s thin wrapper and
  `TargetSelectionE1ObservationsCfg`** per the Files section.

- [ ] **Step 6: Quick local, no-GPU-required Python import check** —
  confirm `TargetSelectionE1ObservationsCfg` imports cleanly, its nested
  `PolicyCfg` has all 41 inherited base terms plus exactly one new
  (`distractor_distance_summary`) term, and every OTHER existing
  `ObservationsCfg`/`TargetSelectionObservationsCfg`-based env cfg in
  `lift_env_cfg.py`/`dice_lift_joint_env_cfg.py` is completely unaffected.
  (Full live 44-dim confirmation against a real env is Task 2's own live
  diagnostic, since that requires `distractor_3`'s scene entity to exist —
  not yet built at this point in the plan.)

- [ ] **Step 7: Commit.**

```bash
git add tasks/franka/distractor_observations.py tasks/franka/mdp.py \
        tasks/franka/lift_env_cfg.py tests/test_distractor_observations.py
git commit -m "feat: extend distractor_distance_summary to K=3 (E1, additive sibling to K=2)"
```

---

## Task 2 — Scene/event cfg: 2×2 grid layout + REQUIRED live spawn-and-settle diagnostic

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py`:
  - Add `_PARKED_DISTRACTOR_3_POS = (0.5, 0.0, -0.9)` (distinct x/y from
    the existing `_PARKED_DISTRACTOR_1_POS`/`_PARKED_DISTRACTOR_2_POS`, per
    the existing rationale for why those two already differ from each
    other).
  - Add the 2×2 grid constants exactly per the spec's own worked-out
    table (Cell A=target, B=`distractor_1`, C=`distractor_2`,
    D=`distractor_3`; all four share one `pose_range`):
    ```python
    _E1_GRID_POSE_RANGE = {"x": (-0.03, 0.03), "y": (-0.09, 0.09), "z": (0.0, 0.0)}
    _E1_TARGET_LANE_CENTER = (0.43, -0.125, 0.055)       # Cell A (near, left)
    _E1_DISTRACTOR_1_LANE_CENTER = (0.43, 0.125, 0.055)  # Cell B (near, right)
    _E1_DISTRACTOR_2_LANE_CENTER = (0.57, -0.125, 0.055) # Cell C (far, left)
    _E1_DISTRACTOR_3_LANE_CENTER = (0.57, 0.125, 0.055)  # Cell D (far, right)
    ```
  - Add `FrankaDieLiftTargetSelectionE1SceneCfg(FrankaDieLiftTargetSelectionSceneCfg)`
    — one new sibling `RigidObjectCfg` field, `distractor_3` (prim path
    `{ENV_REGEX_NS}/Distractor3`), same placeholder-spawn-at-cfg-time /
    override-in-`__post_init__` pattern as `distractor_1`/`distractor_2`,
    default `init_state.pos=_PARKED_DISTRACTOR_3_POS`. **Do not add this
    field to the base `FrankaDieLiftTargetSelectionSceneCfg`** — that
    class is still used unchanged by SO/D1/D2.
  - Add `TargetSelectionE1EventCfg(TargetSelectionEventCfg)` — one new
    `EventTermCfg` field, `reset_distractor_3_position`
    (`mdp.reset_root_state_uniform`, placeholder `pose_range`/
    `SceneEntityCfg("distractor_3", body_names="Distractor3")` at
    declaration, real values set in `__post_init__`). Do not mutate the
    base `TargetSelectionEventCfg`.
  - Add `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg(FrankaDieLiftJointD12D20MixedEnvCfg)`:
    - `scene: FrankaDieLiftTargetSelectionE1SceneCfg = FrankaDieLiftTargetSelectionE1SceneCfg(num_envs=4096, env_spacing=2.5)`
    - `events: TargetSelectionE1EventCfg = TargetSelectionE1EventCfg()`
    - `observations: TargetSelectionE1ObservationsCfg = TargetSelectionE1ObservationsCfg()` (Task 1)
    - `__post_init__`: `super().__post_init__()`; `self.active_distractor_count = 3`;
      **relocate the target too** (unlike D2, which never moved
      `object`'s own lane center) —
      `self.scene.object.init_state.pos = _E1_TARGET_LANE_CENTER` AND
      `self.events.reset_object_position.params["pose_range"] = dict(_E1_GRID_POSE_RANGE)`
      together, per the spec's own explicit "changing only one half
      silently produces the wrong region" warning; then set
      `distractor_1`/`distractor_2`/`distractor_3`'s own
      `init_state.pos` to their respective `_E1_..._LANE_CENTER`
      constants, each a real independent
      `MultiAssetSpawnerCfg(assets_cfg=[d12_cfg, d20_cfg], random_choice=True)`
      population (same d12/d20 `UsdFileCfg` definitions D2 already uses —
      reuse directly, do not re-derive), and each corresponding
      `events.reset_distractor_{1,2,3}_position.params["pose_range"] = dict(_E1_GRID_POSE_RANGE)`.
    - Two `_PLAY` eval variants,
      `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg_PLAY_D12Target` /
      `_PLAY_D20Target` — same pattern as D2's own `_PLAY` variants
      (pin `scene.object.spawn` to a single d12/d20 `UsdFileCfg`, set
      `die_shape_class`, `die_shape_classes_per_env = None`,
      `num_envs = 8`, `env_spacing = 2.5`,
      `observations.policy.enable_corruption = False`). Distractor slots
      (`1`/`2`/`3`) are NOT pinned to a single shape at eval — they keep
      their own real `MultiAssetSpawnerCfg`/`random_choice=True`
      populations, matching every prior stage's own eval convention.
- Create: `scripts/_diag_target_selection_clutter_e1_scene_check.py` —
  extends `scripts/_diag_target_selection_clutter_scene_check.py`'s exact
  pattern (predicted-vs-live cross-check + spawn-and-settle) to 4 co-
  present entities (target + 3 distractors) instead of 3:
  - **Check (a) multi-spawner coexistence**: target's deterministic
    round-robin still matches live observation/USD scale with 3
    additional `MultiAssetSpawnerCfg`-populated siblings present; each
    distractor's live scale is a valid d12/d20 value; both shapes appear
    across the batch for all 3 distractor slots.
  - **Check (b) spawn-and-settle**: step physics to settle
    (`SETTLE_STEPS` unchanged from the K=2 diagnostic, 150 steps), then
    per-env verify (1) no entity (target or any of the 3 distractors)
    ends up off-table/outside the reachable workspace — update
    `REACH_X`/`REACH_Y` bounds generously to cover BOTH grid rows
    (near x∈[0.40,0.46], far x∈[0.54,0.60]) and BOTH columns
    (left y∈[-0.215,-0.035], right y∈[0.035,0.215]), with the same kind
    of settle-drift margin the K=2 diagnostic already used; (2) no two
    entities' live post-settle positions fall below the same
    `MIN_SEPARATION_M = 0.06` (60mm) safety floor used by the K=2
    diagnostic — report the actual measured minimum pairwise separation
    for same-row, same-column, and diagonal pairs explicitly (the spec's
    own nominal estimates are 70mm/80mm/106mm respectively; this is the
    live check that confirms or refutes those design-time numbers, not a
    re-assertion of them).
  Report both checks explicitly PASS/FAIL, not just "ran without
  crashing" — matching the K=2 diagnostic's own convention.

**Interfaces:**
- Consumes: Task 1's `TargetSelectionE1ObservationsCfg`,
  `FrankaDieLiftJointD12D20MixedEnvCfg`'s target population/scale
  constants (reused, not re-derived).
- Produces: `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg` + 2 `_PLAY`
  variants, consumed by Task 3 (wiring) and Task 5 (training/eval). The
  live diagnostic's PASS result is a hard precondition for Task 5 — do
  not proceed to training if it fails.

- [ ] **Step 1: Add the constants/scene/event/env-cfg classes** per the
  Files section above, reusing Task 1's Step 1 offset-semantics finding
  (already confirmed in the original experiment: `pose_range` is an offset
  added to each entity's own `init_state.pos`, not an absolute world-frame
  range) — no need to re-derive this from Isaac Lab source again, just
  cite it.

- [ ] **Step 2: Write `scripts/_diag_target_selection_clutter_e1_scene_check.py`**
  per the Files section.

- [ ] **Step 3: Run it live** (non-headless, no training, `num_envs=16`,
  `flock`-wrapped):
  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_target_selection_clutter_e1_scene_check.py"
  ```
  Report both checks' PASS/FAIL explicitly, including the actual measured
  minimum pairwise separations (not just "PASS").

- [ ] **Step 4: If Step 3 finds a real bug** (misplacement, spawner
  interference, off-workspace entity, separation below 60mm) — fix it in
  this same task and re-run Step 3 to confirm, per this project's bug-
  handling discipline (do not defer a found bug to Task 5). If the grid's
  own design-time numbers turn out to be unsafe once live-tested, adjust
  the grid geometry (e.g. widen `_E1_GRID_POSE_RANGE`'s margins or shift
  cell centers) and re-verify — this task owns getting the topology to a
  genuinely passing state, not just reporting a failure.

- [ ] **Step 5: Commit.**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py \
        scripts/_diag_target_selection_clutter_e1_scene_check.py
git commit -m "feat: E1 2x2-grid scene topology (distractor_3 + target relocation), live-verified"
```

---

## Task 3 — Wiring: `train_franka.py` / `franka_checkpoint_review.py` / `sync_run_to_gcs.py`

**Files:**
- Modify: `scripts/train_franka.py` — add one new `--variant` choice
  (`joint-die-target-selection-e1`), importing
  `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg` (mirrors the existing
  `elif args_cli.variant == "joint-die-target-selection-d2":` branch
  exactly), a new `_log_suffix` entry (`_jointdietargetselectione1`), and
  help text describing the stage (cite this plan + the spec).
- Modify: `scripts/franka_checkpoint_review.py` — add
  `"joint-die-target-selection-e1"` to the existing `_TARGET_SELECTION_VARIANTS`
  list (this reuses the already-generic `--eval_target_shape` disambiguation
  mechanism the SO/D1/D2 variants already use — no new flag needed) and
  two new `elif` branches (both the argparser-choices dispatch and the
  eval-construction dispatch) importing/constructing
  `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg_PLAY_D12Target`/
  `_PLAY_D20Target` based on `--eval_target_shape`. Follow the existing
  `if/elif` pattern exactly (see the current `joint-die-target-selection-d2`
  branches at both dispatch points as the direct template) — do not
  restructure the script.
- Modify: `scripts/sync_run_to_gcs.py` — add one new `VARIANT_MAP` entry
  (`train_franka_jointdietargetselectione1` → `joint-die-target-selection-e1`),
  matching `train_franka.py`'s new `_log_suffix` value exactly.

**Interfaces:**
- Consumes: Task 2's `FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg` + 2
  `_PLAY` classes.
- Produces: `--variant joint-die-target-selection-e1` on all 3 scripts,
  consumed by Task 5.

- [ ] **Step 1: Wire `train_franka.py`** per the Files section.
- [ ] **Step 2: Wire `franka_checkpoint_review.py`** per the Files
  section.
- [ ] **Step 3: Wire `sync_run_to_gcs.py`**'s `VARIANT_MAP`.
- [ ] **Step 4: Smoke-check** — quick local, no-GPU-required `--help` on
  all 3 scripts, confirming the new choice parses and resolves without
  error.
- [ ] **Step 5: Commit.**

```bash
git add scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: wire --variant joint-die-target-selection-e1 into train/eval/sync scripts"
```

---

## Task 4 — Checkpoint weight-surgery: 43→44-dim extension of `model_5096.pt`

**Files:** none — this task only *invokes* the existing, already-generic
`scripts/extend_checkpoint_observation_dims.py` (no code change needed to
the surgery/verify logic itself, confirmed by direct reading: it takes
`--old-obs-dim`/`--new-obs-dim` as CLI args, no hardcoded 41/43). One
small documentation-only addition:
- Modify: `scripts/extend_checkpoint_observation_dims.py`'s module
  docstring — append a short note (no logic change) recording that this
  script was reused for E1's 43→44 case and clarifying what `--verify`
  proves there (see "Verification interpretation" below) — so a future
  reader doesn't assume a passing `--verify` always means the same thing
  it did at Stage SO.

**Interfaces:**
- Consumes: `model_5096.pt` (downloaded locally from
  `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d2/seed42/2026-07-19_21-08-07/model_5096.pt`).
- Produces: `model_5096_44dim.pt`, the extended checkpoint Task 5 resumes
  training from. Produces the internal pre-training gate's own PASS/FAIL
  verdict, a hard blocking precondition for Task 5.

**Verification interpretation (the plan's answer to "what counts as
acceptable verification when the new column isn't hard-zeroed"):**
`--verify`'s check (`verify_stage_so_equivalence`) feeds the EXTENDED
network a synthetic batch whose new column(s) are forced to exactly
`0.0`, and confirms it matches the ORIGINAL network fed the same batch's
old columns. This is an unconditional linear-algebra fact (a Linear
layer's new column contributes exactly `0.0` to its output whenever fed
exactly `0.0`, regardless of what the other columns carry or what
real-world distribution the network was trained under) — so **at E1 this
check still proves the surgery is MECHANICALLY correct**: the old 43
columns were copied unchanged (not corrupted/reordered) and the new
column was appended (not interleaved) at the right position. **It does
NOT prove — and this plan makes no claim that it proves — that E1 starts
from behaviorally-unperturbed D2 behavior**, because `distractor_3`'s real
observation value is never exactly `0.0` once E1 actually trains (the new
distractor is active from iteration 0, unlike Stage SO's genuinely-inert
extension). The real behavioral-equivalence question the old check
answered at Stage SO simply does not apply at E1 — E1's real evidence for
"did the warm start help/hurt" is Task 5's own trained, evaluated result,
not this gate. This gate exists only to catch a column-order/copy bug
before spending any training budget on a broken extension — exactly the
scope the spec's own "Internal pre-training gate" section already defines
it as.

- [ ] **Step 1: Download `model_5096.pt` locally** (`gcloud storage cp` /
  `gsutil cp`).
- [ ] **Step 2: Run the surgery + `--verify` locally** (throwaway CPU-only
  venv, no Isaac Sim/GPU required, matching how this script already ran
  on the Pi for the original SO warm start):
  ```bash
  python3 scripts/extend_checkpoint_observation_dims.py \
      --input model_5096.pt \
      --output model_5096_44dim.pt \
      --old-obs-dim 43 --new-obs-dim 44 --seed 42 --verify
  ```
- [ ] **Step 3: Pre-training gate check** — must show exactly `0.0` max
  abs diff for BOTH actor and critic branches. If it does not, STOP — do
  not proceed to Task 5 — investigate (most likely cause per the spec: a
  column-order mismatch between the K=2 and K=3 pure functions, i.e. Task
  1's own load-bearing cross-check test would have caught this already;
  re-check that test passed) and fix before re-attempting.
- [ ] **Step 4: Add the module-docstring note** described in the Files
  section, documenting the "mechanical correctness, not t=0 behavioral
  losslessness" distinction for future readers.
- [ ] **Step 5: Re-run Step 2's `--verify` a second time, on the actual
  training host, immediately before launching `train_franka.py` in Task
  5** — matching the original experiment's own double-verification
  protocol exactly (once locally before any cloud/desktop dispatch, once
  again on the real training host). Report both results explicitly.
- [ ] **Step 6: Commit** (docstring note only — the `.pt` checkpoint
  files themselves are not committed, matching this repo's `models/`/data
  gitignore convention; store `model_5096_44dim.pt` alongside the
  training dispatch instead).

```bash
git add scripts/extend_checkpoint_observation_dims.py
git commit -m "docs: document extend_checkpoint_observation_dims.py's E1 (43->44) reuse and --verify's mechanical-only meaning there"
```

---

## Task 5 — Train E1 (checkpoint-resumed from D2) + eval — PRIMARY FALSIFICATION CHECK

**Files:** none new — runs Tasks 1-4's code/checkpoint against a real
training run. Per the Global Constraints' "Execution backend" section —
copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into this
dispatch if it falls back to cloud.

**Interfaces:**
- Consumes: Task 4's `model_5096_44dim.pt` (re-verified per Task 4 Step
  5), `--variant joint-die-target-selection-e1` (Task 3).
- Produces: E1's own checkpoint + the spec's pre-registered primary
  falsification verdict, consumed by Task 6.

**Precondition:** Task 4's pre-training gate passed (exactly 0.0 max abs
diff, both branches, both verification runs). Do not start training
otherwise — report back to the controller instead.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch via
  `scripts/run_on_desktop_gpu.sh` if AVAILABLE, else cloud fallback per
  the checklist. Check `scripts/check_cloud_state.sh` first if falling
  back to cloud, in case another concurrent workstream already holds the
  project's single `GPUS_ALL_REGIONS` slot. Record dispatch target and
  (if cloud) instance creation timestamp.
- [ ] **Step 2**: Train:
  ```bash
  train_franka.py --variant joint-die-target-selection-e1 \
      --checkpoint model_5096_44dim.pt --policy_only_checkpoint \
      --max_iterations <5096 + budget>
  ```
  Verify the resumed-at iteration count directly from `train_franka.py`'s
  own printed "Resumed from ... at iteration ..." message (do not assume
  it's exactly 5096 from the filename alone — matching this project's own
  established verify-don't-assume convention) before computing the
  absolute `--max_iterations` target. `--policy_only_checkpoint` is
  required (the first-layer weight shape changed, so the old optimizer's
  Adam moments for that layer are shape-incompatible — a fresh PPO
  optimizer must be built, same reasoning as Stage SO's own warm start).

  **Iteration budget: start at 1000 additional iterations**
  (`--max_iterations 6096`, matching the spec's own recommended starting
  point and D2's own reasoning for its second-simultaneous-distractor
  increment) — **this is a starting point for this task's own judgment,
  not a rigid mandate**. Watch `Episode_Reward/lifting_object`/
  `Episode_Reward/object_goal_tracking`/`Episode_Termination/object_dropping`
  live during the run (the same signals every prior stage's own budget
  judgment was based on) and extend the budget if the curve is still
  climbing meaningfully as 1000 iterations approaches. Document the
  actual reasoning used, matching every prior stage's own write-up
  convention (see `kb/wiki/experiments/target-selection-clutter.md`'s D1/
  D2 sections for the expected depth).
- [ ] **Step 3**: Run the instrumented eval — both pinned-target-shape
  `_PLAY` variants (`--variant joint-die-target-selection-e1
  --eval_target_shape d12` and `d20`), `num_envs=8`, the full 4-entity E1
  scene active, `franka_checkpoint_review.py`'s current settle-detection
  logic, video + `max_height_gain`/`max_consecutive_lifted_steps` for both
  shapes.
- [ ] **Step 4: Primary falsification check (pre-registered in the
  spec)** — for each shape independently: if discovery rate is below 6/8
  (75%), that shape FALSIFIES "the count-curriculum + fixed-size zero-
  padded observation mechanism, already validated through K=2, continues
  to scale to K=3." Report both shapes' results explicitly and
  separately (not averaged), exactly as observed. Explicitly check for a
  "grasped the wrong die" episode across every inspected frame, per D2's
  own precedent and this being a newly-possible failure mode at 3
  distractors.
- [ ] **Step 5: Escalation note (if falsified)** — if E1 falsifies for
  either shape while Task 4's gate passed cleanly, report this per the
  spec's own pre-registered escalation guidance: the honest next step is
  a richer/architecturally-different observation mechanism (Deep-Sets/
  attention over distractor state), not another parameter retune (more
  iterations, a different grid) unless there's a specific, evidenced
  reason to believe a parameter was simply mis-set. Do not decide or
  attempt that follow-on within this task — only report the finding and
  the spec's own recommendation. **Do not proceed to a future E2/S1 stage
  if E1 falsifies** — per the research doc's own staging, E1 failing puts
  the count-scaling axis itself in doubt, not just this specific
  increment.
- [ ] **Step 6**: Full teardown/verification per the Global Constraints'
  Execution-backend section. Report cost against the $2 cap.
- [ ] **Step 7**: Bug-handling — if the run surfaces a real bug in Tasks
  1-4's code, fix it, re-verify, and commit the fix separately before
  re-running.

---

## Task 6 — Verdict + documentation

**Files:**
- Modify: `ROADMAP.md` — append the E1 verdict entry (pre-training gate
  result, primary falsification result per shape, checkpoint, cost,
  teardown verification — matching the depth/format of the SO/D1/D2
  experiment's own ROADMAP entries).
- Modify: `kb/wiki/experiments/target-selection-clutter.md` — this
  extends that same finished experiment's article (per this spec's own
  framing); add a new dated section (e.g. "Stage E1: scaling to 3
  distractors (2026-07-2x)") following the article's own existing
  per-stage section structure, rather than creating a separate new kb
  article — this is a direct continuation of the same experiment
  narrative, not a new one.
- Modify: `kb/wiki/index.md` — update the one-line summary/status if this
  repo's convention calls for it (check the existing entry for this
  experiment first).

**Interfaces:**
- Consumes: Task 5's real numeric results (per-shape discovery rates,
  checkpoint, cost).
- Produces: the closed-out, cross-referenced record of E1's real result.

- [ ] **Step 1**: Verdict against the spec's pre-registered falsifiable
  hypothesis — state clearly, per shape, whether E1 passed (≥6/8) or
  falsified (<6/8) the primary bar, and confirm Task 4's pre-training gate
  result explicitly (even though expected to pass cleanly). Do not average
  d12/d20 into one number.
- [ ] **Step 2**: Include the same rigor the SO/D1/D2 experiment's closing
  sections used — instrumented `max_height_gain`/
  `max_consecutive_lifted_steps` numbers, explicit before/after comparison
  against D2's own 8/8-both-shapes baseline, confirmation the eval videos
  were actually watched/frames inspected (not just the JSON trusted), and
  the explicit "grasped the wrong die" check result.
- [ ] **Step 3**: Update `ROADMAP.md` and the kb article(s) per the Files
  section above.
- [ ] **Step 4**: If the primary bar was falsified for either shape,
  explicitly carry forward the spec's own pre-registered escalation
  recommendation (Deep-Sets/attention architecture over distractor state)
  into the kb article's own "Open, not yet decided" section as the
  concrete next candidate — do not leave it as a bare "failed" note. Also
  record explicitly, regardless of outcome, that E2 (3→4 distractors) and
  S1 (fold in d8/d10) remain future specs gated on this result, per the
  research doc's own staging — E2/S1 should not start automatically off
  this plan's completion.
- [ ] **Step 5: Commit and push.**

```bash
git add ROADMAP.md kb/wiki/experiments/target-selection-clutter.md kb/wiki/index.md
git commit -m "verdict: target-selection-clutter Stage E1 (2->3 distractors, d12/d20)"
git push origin main
```

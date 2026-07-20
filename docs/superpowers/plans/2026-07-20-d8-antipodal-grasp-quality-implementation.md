# d8 antipodal/force-closure grasp-quality reward (dual action-space test) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test H_joint and H_taskspace (two distinct, individually falsifiable
hypotheses) — does porting `tasks/ar4/mdp.py:902-940`'s `antipodal_grasp_bonus`
(a bilateral force-closure/antipodal grasp-quality reward term, physically
refit to Franka's real μ=0.5 friction coefficient,
`antipodal_cos_threshold=-0.894427`) onto d8's robust, independently-
established 0/24 grasp-discoverability null
(`FrankaDieLiftJointD8BigEnvCfg`, 48mm-parity) unlock sustained-lift
discovery — under joint-space control (Condition A, H_joint) and,
separately, under task-space/relative-IK control (Condition B, H_taskspace).
Two-tier falsification per condition, per seed, never averaged:
**mechanism-level** (does the `antipodal_grasp_quality` reward term's own
final-100-iteration mean exceed 1e-4) and **behavioral** (does sustained
lift actually occur, `franka_checkpoint_review.py`'s existing protocol). A
SPLIT result (mechanism fires, lift still doesn't complete) is reported as
its own distinct outcome per condition, not collapsed to a binary — mirroring
this exact d8 env cfg's own immediately-prior exploration-bonus SPLIT
result. The combined 5-row outcome matrix (spec's own table) is reported in
full, not reduced to "confirmed/falsified."

**Why two conditions, not one:** this project's own AR4-era arc
(Experiments 9→10→11) found a correctly-thresholded antipodal check
regresses to exactly `0.000000` under joint-space control specifically —
not a reward-calibration problem but a gripper-positioning-precision
problem — and only task-space/IK control produced the first genuine
sustained antipodal signal. Testing only joint-space here would reproduce,
not test, that already-known result. Both conditions are run to completion
unconditionally (Condition B is not gated on Condition A's result).

**Architecture:** (1) a new pure-`torch` module
(`tasks/franka/antipodal_grasp_reward.py`, TDD) holding the antipodal
force-closure math as a *pure* function operating on two `(N,3)` force
vectors — a deliberate, small structural deviation from the spec's own
single-function code sketch (which combines the tensor math and the
`ContactSensor` state read in one function, exactly mirroring AR4's
original), made to match this project's own already-established Franka
convention of splitting sim-independent pure math (testable via plain
pytest+torch, no Isaac Sim) from a thin `mdp.py` wrapper that reads live
sim state (`lift_reward.py`/`exploration_bonus_reward.py`'s own precedent)
— see "Design notes" below, the math itself is unchanged from the port;
(2) scene wiring — a new `FrankaDieLiftContactSceneCfg`
(`tasks/franka/dice_lift_joint_env_cfg.py`, mirroring
`FrankaDieLiftTargetSelectionSceneCfg`'s "extend `FrankaLiftSceneCfg` with
new sibling fields" precedent in the same file) adapting
`dice_scene_cfg.py`'s already-proven `activate_contact_sensors=True`
copy-then-mutate + two-single-body-`ContactSensorCfg`-with-
`filter_prim_paths_expr` pattern onto `panda_leftfinger`/
`panda_rightfinger`, wired into a `ManagerBasedRLEnvCfg` for the first
time; (3) a new `AntipodalGraspRewardsCfg` (`tasks/franka/lift_env_cfg.py`,
mirroring `ExplorationBonusRewardsCfg`'s "new subclass, base untouched"
precedent); (4) two new leaf env-cfg classes
(`FrankaDieLiftJointD8BigAntipodalEnvCfg` for Condition A,
`FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg` for Condition B, both plus
`_PLAY` variants) plus `train_franka.py`/`franka_checkpoint_review.py`/
`sync_run_to_gcs.py` `--variant` wiring and a bounded smoke test for both
conditions; (5) two real 3-seed/1500-iteration training+eval runs (one per
condition, sequential due to this project's one-Isaac-Sim-process-at-a-time
`flock` convention, but each unconditionally executed); (6) verdict +
`ROADMAP.md`/kb update using the spec's own 5-row outcome matrix,
cross-linking the AR4-era Experiment 9/10/11 arc this design directly
tests transfer of, and the immediately-prior exploration-bonus SPLIT
result this spec's own research doc identifies as the forward pointer into
this experiment.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`lift_env_cfg.py`, `dice_lift_joint_env_cfg.py`, `mdp.py`, a new
`antipodal_grasp_reward.py`), desktop-first GPU dispatch
(`scripts/check_gpu_availability.sh`/`scripts/run_on_desktop_gpu.sh`), GCP
cloud fallback (`docs/cloud/dispatch-checklist.md`). **Desktop status
checked at plan-authoring time (2026-07-20): UNREACHABLE**
(`scripts/check_gpu_availability.sh` → `TARGET=cloud`, `curl` DNS
resolution to `home.local` timed out) — re-check fresh at each task's own
dispatch time per Global Constraints below; do not assume this result
still holds by the time execution starts.

Spec: `docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md`.
Research: `docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`.
Template/precedent for this plan's own structure and rigor:
`docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-discovery-implementation.md`
(TDD-first-task-split, design-notes-flagged-for-controller precedent,
bounded-smoke-test-before-real-dispatch precedent, cost-cap reasoning
methodology, and the exact `FrankaDieLiftJointD8BigEnvCfg` env cfg this
plan also extends) and
`docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-implementation.md`
(one-task-per-hypothesis-condition precedent, applied here as one-task-
per-action-space-condition).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Design notes (flagged for controller review, not silently assumed)

**1. Pure-math/wrapper module split, not a literal single-function port.**
The spec's own code sketch for `antipodal_grasp_bonus` (design spec,
"Exact mechanism proposed") is one function that both computes the
force-closure math AND reads `env.scene[...].data.force_matrix_w` —
verbatim from `tasks/ar4/mdp.py:902-940`, which is written the same way
(AR4's own `mdp.py` has no pure/wrapper split convention at all). Franka's
`tasks/franka/` package, however, already has an established, different
convention for exactly this situation:
`lift_reward.py`/`exploration_bonus_reward.py`/`distractor_observations.py`
hold sim-independent pure-`torch` math (no `isaaclab` import, testable via
plain pytest), and `mdp.py` holds thin wrappers that pull live `env.scene`
state and delegate to the pure function. This plan follows the *Franka*
convention, not the *AR4* one, for TDD-ability — Task 1's tests exercise
the pure math directly against synthetic force vectors, with zero Isaac
Sim dependency, matching every other Franka reward module in this
codebase. **The math itself is unchanged from the spec's port** (identical
formula, identical parameter names/semantics) — only the module boundary
differs. This is judged to be a small, implementation-level convention
choice (which file structure to use for testability) squarely within a
Senior's own discretion, not a cross-cutting architectural call — flagged
here for visibility, not escalated as an open question.

**2. The antipodal-threshold boundary must be tested explicitly, not just
the qualitative antipodal/non-antipodal cases.** Experiment 9's own
history (guess a threshold, discover it's wrong later) is the reason this
spec pins `antipodal_cos_threshold=-0.894427` (μ=0.5) as a fixed,
physically-derived constant rather than a tunable. Task 1's tests must
include a case that would silently pass with the *wrong* (AR4's own
μ=1.0, `-0.7071`) threshold but correctly fail with the *right* one (e.g.
a synthetic `cos_angle` of exactly `-0.80`, between the two thresholds) —
this is the one test that would actually catch an accidental
threshold-carryover regression, not just a generic "antipodal vs. not"
sanity check.

**3. `ContactSensorCfg` wired into a `ManagerBasedRLEnvCfg` for the first
time — the scripted-demo precedent is real but not identical.**
`dice_scene_cfg.py`'s `activate_contact_sensors=True` + per-finger
`ContactSensorCfg` pattern is proven in a plain scripted `InteractiveScene`
demo (`scripts/dice_pick_demo.py`), never inside an actual training loop
with a `RewardManager` reading `ContactSensorData.force_matrix_w` every
step. Task 1 includes a bounded, non-training empirical check (small
`num_envs`, random or scripted actions, no policy/checkpoint needed) that
confirms `force_matrix_w`'s shape and that it actually reads nonzero force
when the die is in contact with a finger, under this exact new
`ManagerBasedRLEnvCfg` wiring — not assumed to transfer automatically from
the scripted-demo precedent just because the same `ContactSensorCfg`
mechanics are involved.

## Global Constraints

- **Do not modify `tasks/franka/dice_lift_joint_env_cfg.py`'s existing
  `FrankaDieLiftJointD8BigEnvCfg`, or `tasks/franka/lift_env_cfg.py`'s
  existing `RewardsCfg`/`FrankaLiftSceneCfg`, in place.** Every new
  mechanism in this plan is additive: a new pure-math module, a new
  `mdp.py` function, a new `FrankaDieLiftContactSceneCfg` /
  `AntipodalGraspRewardsCfg`, and two new leaf env-cfg classes extending
  `FrankaDieLiftJointD8BigEnvCfg` — never editing it. This keeps every
  other concurrently-relevant workstream's own use of the plain,
  unmodified `FrankaDieLiftJointD8BigEnvCfg`/`RewardsCfg` (e.g. any future
  revisit of the d8/d10 demo-warmstart or exploration-bonus arcs)
  completely unaffected. Check `git status` fresh at the start of
  execution for any other concurrent workstream's untracked/in-progress
  files before touching anything, per this plan's own dispatch
  instructions — do not touch files outside this plan's own scope.
- **`antipodal_cos_threshold=-0.894427` and `force_threshold=0.05` are
  fixed, not tunable, for both conditions in this experiment** (spec's own
  "Global constraints": the cosine threshold is physically derived from
  this scene's real μ=0.5 and is explicitly NOT a free dial, unlike
  `force_threshold`/reward weight which are implementer-set Tier-2-hillclimb
  candidates later, not tuned here either). Reward weight fixed at `1.0`
  for both conditions.
- **No exploration-bonus terms, no demo warm-start, no PPO-runner-cfg
  change pre-authorized for Condition A** (spec's own "Global constraints"
  — this experiment isolates the antipodal mechanism as its own variable).
  `AntipodalGraspRewardsCfg` extends the plain `RewardsCfg`, never
  `ExplorationBonusRewardsCfg`.
- **Condition B's known critic-divergence risk (Experiment 11's AR4-era
  `Loss/value_function` blowup, fixed there by `clip_actions=5.0`) is
  flagged, not pre-resolved.** `train_franka.py` currently instantiates a
  single, shared `agent_cfg = FrankaLiftPPORunnerCfg()` with **no
  per-variant agent-cfg selection mechanism at all** (confirmed by direct
  read, `scripts/train_franka.py:285` — every `--variant` uses the same
  PPO runner cfg). If Task 4's real Condition B run shows the same
  divergence signature (watch `Loss/value_function` in TensorBoard
  throughout, not just at the end), applying Experiment 11's fix requires
  BOTH a new, Condition-B-scoped `PPORunnerCfg` subclass AND a small
  extension to `train_franka.py`'s variant dispatch to select it — real,
  if small, new surface area. **Do not build this preemptively.** Only
  build it if Task 4's real run actually shows divergence, and scope the
  fix to Condition B's own runner cfg only, never touching the shared
  default (per the spec's own explicit constraint). If this triggers,
  flag it explicitly in Task 4's own report (a real, if minor,
  train_franka.py change beyond this plan's original file list) rather
  than silently absorbing it.
- **Execution backend: desktop-first, cloud fallback** (2026-07-18
  standing policy, CLAUDE.md's "Pi-as-primary-agent GPU dispatch").
  **Desktop was UNREACHABLE at this plan's authoring time — re-check fresh
  at each task's own dispatch time, never reuse an earlier check's
  result.** For every task that launches Isaac Sim:
  1. Check `scripts/check_gpu_availability.sh`. `TARGET=desktop` (exit 0)
     → dispatch via `scripts/run_on_desktop_gpu.sh` (default blocking
     mode). `TARGET=cloud` (exit 1, BUSY) or unclear (exit 2, UNKNOWN) →
     fall back to `docs/cloud/dispatch-checklist.md`'s recipe. **Never
     treat "can't tell" as a green light for desktop.**
  2. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim** (its
     blocking instruction, cost-cap paragraph, environment-conventions
     block, and bug-handling-discipline block).
  3. **`flock -o` is not automatic** — the command string shipped to the
     desktop or cloud instance must itself be
     `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."`.
     Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
     (empty = clear) before dispatch, not a process-name/path grep.
  4. **Non-headless on desktop** (the user wants to watch) — do not pass
     `--headless`. **Headless only** on cloud fallback.
  5. **Full teardown after any cloud task**: `scripts/check_cloud_state.sh`
     shows zero instances/disks/snapshots before marking a task done.
     Desktop: `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
     `check_gpu_availability.sh` all clear/AVAILABLE afterward.
  6. **Other concurrent workstreams in this session may be holding this
     project's single cloud GPU quota** (`GPUS_ALL_REGIONS=1`
     project-wide, per the exploration-bonus and target-selection-clutter
     plans' own already-hit contention) **or dispatching to the same
     desktop.** Each task below must check fresh and genuinely
     queue/wait/retry if busy — never assume availability just because an
     earlier task in this same plan found it available, and never treat a
     busy quota as a reason to skip a run.
- **Cost cap: $6 cumulative across Tasks 1/2/3/4 combined, notify the
  controller if exceeded.** This is a cloud-fallback safety backstop,
  not an expected spend — desktop dispatch brings actual cost toward $0.
  Derivation: this plan's dominant cost is two full 3-seed/1500-iteration
  batches (Tasks 3 and 4), each the same scale as the exploration-bonus
  plan's own single such batch, which itself measured real actual spend
  of ≈$1.2 (its own $3 cap, doubled from a $0.91-per-batch estimate for a
  SPOT-preemption-retry buffer). Two batches instead of one: 2 ×
  $1.82(estimate, SPOT-buffered) ≈ $3.64, plus smoke tests for **two**
  conditions instead of one (Tasks 1's contact-sensor check + Task 2's
  dual-condition smoke test, roughly double the exploration-bonus plan's
  own $0.50 estimate) ≈ $1.00. Sum ≈ $4.64, rounded up to **$6** (not $5)
  for extra margin given this plan carries a real, if-triggered, retry
  risk the exploration-bonus plan's single condition did not (Condition
  B's known critic-divergence possibility, which could cost a wasted
  partial run before a scoped fix lands) — matching the task's own
  "roughly double the $3 cap" framing while grounding the exact number in
  this plan's real cost drivers rather than a bare 2x multiply.
- **Real evidence over proxies at every eval**: TensorBoard-logged
  `antipodal_grasp_quality` term mean (mechanism bar) AND
  `franka_checkpoint_review.py`'s `max_height_gain`/
  `max_consecutive_lifted_steps`/sustained-lift count (behavioral bar) AND
  a reviewed eval video (rest frame vs. peak-height frame, genuinely
  gripped pose) for any positive result — not a shaped reward scalar or
  exit code alone (Experiment 16 precedent, reused explicitly by every
  prior spec in this arc). Independently re-derive the raw per-step
  TensorBoard/`.npy` data for at least one seed per condition for both
  bars, not summary-JSON-only trust, per this arc's own repeated
  settle-detection-bug discipline (commit `977a748`).
- **Report both falsification bars per seed per condition, never averaged
  or collapsed.** Per the spec's own explicit rule: H_joint and
  H_taskspace are each falsified only if BOTH their own bars fail across
  all 3 of their own seeds. A SPLIT result (mechanism fires, behavioral
  bar still fails) is a genuinely novel, separately-reportable outcome per
  condition — Tasks 3/4/5 must preserve and surface this distinction
  explicitly if it occurs, never silently round it to a binary pass/fail,
  matching the immediately-prior exploration-bonus experiment's own SPLIT
  precedent on this identical env cfg.
- **Report the combined 5-row outcome matrix (spec's own table), not just
  two independent verdicts.** Task 5 must state which of the 5 rows the
  real (H_joint, H_taskspace) result pair lands in and what that row's own
  reading means, per the spec's "why both hypotheses must be run to
  completion" section — this is the single most important reporting
  requirement in this plan.
- **TDD discipline for the pure-Python piece** (Task 1's
  `antipodal_grasp_reward.py`): write failing tests first, confirm the
  failure, implement, confirm green — matching `lift_reward.py`/
  `exploration_bonus_reward.py`'s established precedent. Run via
  `/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_franka_antipodal_grasp_reward.py -v -p no:launch_testing`
  (plain python3/pytest lacks `torch` in this environment per this
  project's own standing note) — confirm which interpreter actually has
  `torch` before running, do not assume.
- Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line — each task
  below is executed by a freshly dispatched session; use that session's
  real ID, do not copy a fixed ID from this document.

---

## Task 1 — TDD antipodal reward math + `ContactSensorCfg` scene wiring

**Files:**
- Create: `tasks/franka/antipodal_grasp_reward.py` — pure `torch`-only
  math, NO `isaaclab` import (mirrors `lift_reward.py`'s established
  split, per "Design notes" #1 above). One function:
  - `antipodal_grasp_bonus_raw(jaw1_force_vec: torch.Tensor, jaw2_force_vec: torch.Tensor, force_threshold: float, antipodal_cos_threshold: float) -> torch.Tensor`
    — identical math to `tasks/ar4/mdp.py:930-941`'s own tensor
    computation (magnitude check on both jaws AND cosine-of-angle-between-
    force-directions check), operating directly on the two `(N, 3)` force
    vectors instead of two `ContactSensor` objects. Shape `(N,)` in for
    each force tensor, `(N,)` bool-as-float out. Module docstring must
    cite `tasks/ar4/mdp.py:902-940` as the exact source of the ported
    math (Nguyen 1988; Ponce & Faverjon 1991/93, same citations AR4's own
    docstring already carries) and this spec's own μ=0.5 →
    `antipodal_cos_threshold=-0.894427` derivation
    (`-cos(arctan(0.5))`), distinguishing it explicitly from AR4's own
    μ=1.0 → `-0.7071` value (per "Design notes" #2, this is a refit, not
    a reuse).
- Test: `tests/test_franka_antipodal_grasp_reward.py` — new. Covers:
  - Known antipodal pair (jaw1 force along -X, jaw2 force along +X, both
    magnitude well above `force_threshold`) → returns `1.0`.
  - Known non-antipodal pair (perpendicular forces, `cos_angle=0`, both
    magnitudes above threshold) → returns `0.0` despite both magnitudes
    passing — the direction-check-not-just-magnitude regression test this
    whole mechanism exists for (AR4 Experiment 1→9 precedent).
  - Magnitude-too-small case: perfectly antipodal direction, but at least
    one jaw's force magnitude below `force_threshold` → returns `0.0`.
  - **Threshold-boundary regression test (Design notes #2)**: a
    synthetic `cos_angle` of exactly `-0.80` (between AR4's own `-0.7071`
    and this scene's real `-0.894427`) — asserts the function returns
    `0.0` at `antipodal_cos_threshold=-0.894427` (this scene's real
    value), explicitly proving the function does NOT silently accept
    AR4's looser threshold.
  - Boundary exactness: `cos_angle` of exactly `-0.894427` → `0.0` (the
    `<` comparison is strict, matching AR4's own operator); `cos_angle`
    of `-0.90` (just past the boundary) → `1.0` (given magnitudes both
    pass).
  - Zero-force-vector case (both jaws report exactly zero contact) — the
    `+1e-8` epsilon guard must not produce a spurious antipodal match
    (assert `0.0`, not `NaN`).
  - Batch processing: `N > 1` with a mix of antipodal/non-antipodal/
    magnitude-failing envs in one call → correct per-env tensor, not a
    single collapsed scalar.
- Modify: `tasks/franka/mdp.py` — add
  `antipodal_grasp_bonus(env: ManagerBasedRLEnv, force_threshold: float, antipodal_cos_threshold: float, jaw1_contact_cfg: SceneEntityCfg, jaw2_contact_cfg: SceneEntityCfg) -> torch.Tensor`,
  a thin wrapper: reads
  `env.scene[jaw1_contact_cfg.name].data.force_matrix_w`/
  `jaw2_contact_cfg`, reshapes each to `(env.num_envs, 3)` (matching AR4's
  own `.view(env.num_envs, 3)` reshape — confirm this reshape is still
  correct for this scene's own single-body/single-filter sensor shape
  during Task 1's own empirical check below, do not assume it transfers
  byte-for-byte), and delegates to
  `antipodal_grasp_reward.antipodal_grasp_bonus_raw`. Docstring cites
  `tasks/ar4/mdp.py:902-940` (source of the port) and this file's own new
  pure-math module.
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add
  `FrankaDieLiftContactSceneCfg(FrankaLiftSceneCfg)` (near
  `FrankaDieLiftTargetSelectionSceneCfg`, line ~1024, mirroring its "new
  scene subclass, sibling fields" precedent): overrides `robot` to
  `_FRANKA_ROBOT_CFG_WITH_CONTACT.replace(prim_path="{ENV_REGEX_NS}/Robot")`
  (a new module-level constant in this file, adapting
  `dice_scene_cfg.py`'s own `_FRANKA_ROBOT_CFG_WITH_CONTACT` copy-then-
  mutate idiom — `FRANKA_PANDA_HIGH_PD_CFG.copy(); .spawn.activate_contact_sensors = True`
  — this file already imports `FRANKA_PANDA_HIGH_PD_CFG` transitively via
  `FrankaLiftSceneCfg`, confirm the import path before writing), and adds
  two new `ContactSensorCfg` fields,
  `panda_leftfinger_contact`/`panda_rightfinger_contact`, each pointing at
  `{ENV_REGEX_NS}/Robot/panda_leftfinger`/`panda_rightfinger` with
  `filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]` (per the spec's own
  code block — matches `tasks/ar4/pickplace_env_cfg.py`'s
  `gripper_jaw1_contact`/`gripper_jaw2_contact` two-single-body-sensors
  convention, not one two-body sensor). `object` field NOT overridden —
  stays inherited from `FrankaLiftSceneCfg` (the base DexCube); every
  leaf env cfg's own `__post_init__` chain already mutates
  `self.scene.object` at runtime regardless of which `SceneCfg` subclass
  `self.scene` resolves to (confirmed by direct read of this file's own
  existing pattern — every other leaf class does exactly this).
- Modify: `tasks/franka/lift_env_cfg.py` — add
  `AntipodalGraspRewardsCfg(RewardsCfg)` (near `ExplorationBonusRewardsCfg`,
  mirroring its "new subclass, base untouched" precedent) with one new
  `RewTerm`:
  `antipodal_grasp_quality = RewTerm(func=mdp.antipodal_grasp_bonus, params={"force_threshold": 0.05, "antipodal_cos_threshold": -0.894427, "jaw1_contact_cfg": SceneEntityCfg("panda_leftfinger_contact"), "jaw2_contact_cfg": SceneEntityCfg("panda_rightfinger_contact")}, weight=1.0)`
  per the spec exactly. Docstring cites the spec's own μ=0.5 derivation
  and states plainly this is a NEW class, base `RewardsCfg` untouched.

**Interfaces:**
- Consumes: nothing new (pure math has zero upstream dependency; scene/
  reward wiring consumes only already-existing Isaac Lab library classes).
- Produces: `antipodal_grasp_reward.antipodal_grasp_bonus_raw`,
  `mdp.antipodal_grasp_bonus`, `FrankaDieLiftContactSceneCfg`,
  `AntipodalGraspRewardsCfg` — all consumed by Task 2's two leaf env cfgs.

- [ ] **Step 1: Write failing tests** per the Files section, in
  `tests/test_franka_antipodal_grasp_reward.py`.
- [ ] **Step 2: Run tests, confirm they fail** (`ImportError`/`ModuleNotFoundError`).
- [ ] **Step 3: Implement `antipodal_grasp_reward.py`** per the Files section.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Implement `mdp.py`'s `antipodal_grasp_bonus` wrapper,
  `FrankaDieLiftContactSceneCfg`, and `AntipodalGraspRewardsCfg`** per the
  Files section.
- [ ] **Step 6: Bounded empirical check of the scene wiring** (Design
  notes #3 — this is genuinely new, not a re-trust of the scripted-demo
  precedent). `scripts/check_gpu_availability.sh` → dispatch per Global
  Constraints, `flock`-wrapped, non-headless if desktop. A small,
  non-training script/snippet (`num_envs` in the 8-16 range, random or
  scripted actions, no checkpoint needed) that builds a minimal env using
  `FrankaDieLiftContactSceneCfg` (temporarily, e.g. via a throwaway leaf
  class or an inline test harness — implementer's judgment on the
  cleanest bounded way to exercise this without yet building Task 2's
  real leaf classes) and directly prints/logs
  `env.scene["panda_leftfinger_contact"].data.force_matrix_w.shape` and
  its value across a few steps, confirming: (a) the shape matches what
  `mdp.antipodal_grasp_bonus`'s `.view(env.num_envs, 3)` reshape expects;
  (b) the value is genuinely zero when no contact is happening and
  genuinely nonzero when the gripper is scripted/driven into contact with
  the object (a simple closed-gripper-near-object pose is sufficient, no
  need for a real grasp). Record the exact observed shape/behavior in
  `FrankaDieLiftContactSceneCfg`'s own docstring, citing this empirical
  check — not asserted from the scripted-demo precedent alone.
- [ ] **Step 7: Verify GPU clear** per Global Constraints.
- [ ] **Step 8: Commit.**

```bash
git add tasks/franka/antipodal_grasp_reward.py tests/test_franka_antipodal_grasp_reward.py \
        tasks/franka/mdp.py tasks/franka/dice_lift_joint_env_cfg.py tasks/franka/lift_env_cfg.py
git commit -m "feat: port AR4's antipodal force-closure grasp reward to Franka, refit mu=0.5 (Task 1)"
git push origin main
```

---

## Task 2 — Two leaf env-cfg classes (Condition A/B) + script wiring + dual-condition smoke test

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add:
  - `FrankaDieLiftJointD8BigAntipodalEnvCfg(FrankaDieLiftJointD8BigEnvCfg)`
    (Condition A / H_joint): overrides only `scene:
    FrankaDieLiftContactSceneCfg = FrankaDieLiftContactSceneCfg()` and
    `rewards: AntipodalGraspRewardsCfg = AntipodalGraspRewardsCfg()` — no
    `__post_init__` override needed (matches
    `FrankaDieLiftJointD8BigExplorationBonusEnvCfg`'s own precedent
    immediately above/below it in this file: a pure field-override leaf,
    the inherited `__post_init__` chain from
    `FrankaDieLiftJointD8BigEnvCfg`→`FrankaDieLiftJointHeavyEnvCfg`→
    `FrankaDieLiftJointEnvCfg` still runs and mutates `self.scene.object`
    to the d8 die correctly regardless of which `SceneCfg` subclass
    `self.scene` resolves to). Arm action stays the inherited
    `JointPositionActionCfg` — joint-space, unchanged.
  - `FrankaDieLiftJointD8BigAntipodalEnvCfg_PLAY` (same
    `num_envs=50`/`env_spacing=2.5`/`enable_corruption=False` pattern as
    every other `_PLAY` class in this file).
  - `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg(FrankaDieLiftJointD8BigAntipodalEnvCfg)`
    (Condition B / H_taskspace): per the spec's own code block, a
    `__post_init__` that calls `super().__post_init__()` (running the
    entire inherited chain, INCLUDING `FrankaDieLiftJointEnvCfg`'s own
    joint-space override) and then re-asserts task-space control by
    reassigning `self.actions.arm_action` to a fresh
    `mdp.DifferentialInverseKinematicsActionCfg` matching
    `FrankaLiftEnvCfg.ActionsCfg.arm_action`'s exact stock values
    (`asset_name="robot"`, `joint_names=["panda_joint.*"]`,
    `body_name="panda_hand"`,
    `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")`,
    `scale=0.5`, `body_offset=(0.0, 0.0, 0.107)` — reuse
    `lift_env_cfg.py`'s own `_IK_BODY_OFFSET` constant by reference if
    importable cleanly, otherwise the literal tuple with a comment citing
    its source, implementer's judgment on the cleanest import path
    without creating a circular import). Class name deliberately drops
    "Joint" (unlike Condition A) to reflect it is NOT joint-space,
    matching the spec's own naming.
  - `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg_PLAY` (same `_PLAY`
    pattern).
- Modify: `scripts/train_franka.py` — add two new `--variant` choices:
  `joint-die-d8-big-antipodal` (Condition A, imports
  `FrankaDieLiftJointD8BigAntipodalEnvCfg`, `_log_suffix`
  `_jointdied8bigantipodal`) and `die-d8-big-taskspace-antipodal`
  (Condition B, imports `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`,
  `_log_suffix` `_died8bigtaskspaceantipodal`), mirroring the existing
  `joint-die-d8-big-exploration-bonus` branch's exact structure. Help text
  for both, citing this plan and the design spec, distinguishing the two
  action spaces explicitly (do not let the help text read as though they
  differ only in name).
- Modify: `scripts/franka_checkpoint_review.py` — same two new `--variant`
  choices, importing the two `_PLAY` classes, mirroring the existing
  `joint-die-d8-big-exploration-bonus` dispatch exactly (no new CLI flags
  needed — single target shape, no ambiguity to resolve, unlike the
  target-selection variants' `--eval_target_shape`).
- Modify: `scripts/sync_run_to_gcs.py` — add
  `"train_franka_jointdied8bigantipodal": "joint-die-d8-big-antipodal"`
  and `"train_franka_died8bigtaskspaceantipodal": "die-d8-big-taskspace-antipodal"`
  to `VARIANT_MAP`, matching `train_franka.py`'s new `_log_suffix` values
  exactly.

**Interfaces:**
- Consumes: Task 1's `FrankaDieLiftContactSceneCfg`,
  `AntipodalGraspRewardsCfg`, `mdp.antipodal_grasp_bonus`.
- Produces: both leaf env cfgs (`_PLAY` included), both new `--variant`
  choices on all 3 scripts, and two smoke-test checkpoints — consumed by
  Tasks 3 (Condition A) and 4 (Condition B).

- [ ] **Step 1: Implement both leaf env-cfg classes + `_PLAY` variants**
  per the Files section.
- [ ] **Step 2: Wire `train_franka.py`/`franka_checkpoint_review.py`/
  `sync_run_to_gcs.py`** per the Files section. Smoke-check with `--help`
  on all 3 scripts (no sim launch) to confirm the new choices parse.
- [ ] **Step 3: Bounded smoke test, Condition A** — confirm training
  doesn't crash and the new reward term produces plausible (non-NaN,
  bounded-magnitude) values. `scripts/check_gpu_availability.sh` →
  dispatch per Global Constraints, `flock`-wrapped, non-headless if
  desktop:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-antipodal --num_envs 256 --max_iterations 20 --seed 42"
  ```

  Confirm in the training log/TensorBoard: `antipodal_grasp_quality`
  appears, non-NaN, bounded in `[0, 1]` (this term is a bool-as-float, not
  a shaped continuous value — confirm it is not silently always exactly
  `0.0` for the entire 20-iteration run, which would indicate the contact
  sensor wiring or reshape is wrong, though genuinely rare/zero firing at
  this tiny iteration count alone is not itself damning — cross-check
  against Task 1's own empirical confirmation that contact force reads
  correctly). Keep the checkpoint.
- [ ] **Step 4: Bounded smoke test, Condition B** — same protocol,
  `--variant die-d8-big-taskspace-antipodal`. Additionally watch
  `Loss/value_function` in TensorBoard for this short run as an early,
  non-conclusive canary for Experiment 11's own critic-divergence failure
  mode (per Global Constraints — a clean 20-iteration run does NOT rule
  out divergence appearing later in the real 1500-iteration run; this is
  a cheap early check, not a substitute for watching Task 4's own real
  run throughout). Keep the checkpoint.
- [ ] **Step 5: Verify GPU clear** per Global Constraints.
- [ ] **Step 6: Commit.**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py \
        scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: wire d8 antipodal-grasp-quality env cfgs, joint-space and task-space conditions (Task 2)"
git push origin main
```

---

## Task 3 — Real H_joint run: Condition A, 3 seeds, both falsification bars

**Files:** none new — runs Tasks 1-2's code through a real, full
1500-iteration training + eval cycle, joint-space condition.

**Interfaces:**
- Consumes: `--variant joint-die-d8-big-antipodal` (Task 2).
- Produces: 3 full checkpoints (seeds 42/123/7) + both falsification bars'
  numbers per seed for H_joint — real evidence for the outcome matrix's
  H_joint axis.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per
  Global Constraints (re-check fresh). Record dispatch target and (if
  cloud) instance creation timestamp.
- [ ] **Step 2**: For each seed (42, 123, 7) — 3 runs total, from scratch
  (no `--checkpoint`, matching `FrankaDieLiftJointD8BigEnvCfg`'s own
  established 0/24-baseline recipe exactly, same as the exploration-bonus
  experiment's own precedent on this identical env cfg):

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-antipodal --seed 42 --num_envs 4096 --max_iterations 1500"
  ```

  (Repeat for seeds 123 and 7.)

- [ ] **Step 3: Mechanism-level bar, per seed** — pull the
  `antipodal_grasp_quality` reward term's own TensorBoard-logged mean,
  averaged over the final 100 of 1500 iterations, per seed. Per the spec:
  **H_joint's mechanism-level claim is falsified only if this mean is
  `< 1e-4` in all 3 seeds.** Report the exact per-seed numbers, not just
  pass/fail against the bar.
- [ ] **Step 4: Behavioral bar, per seed** — `franka_checkpoint_review.py
  --variant joint-die-d8-big-antipodal --checkpoint <seed's
  model_1499.pt> --num_envs 8`, existing post-`977a748` settle-detection
  logic, video + `max_height_gain`/`max_consecutive_lifted_steps`. Per
  the spec: **H_joint's behavioral bar is falsified only if all 3 seeds
  show 0/8 sustained-lift (0/24 total).**
- [ ] **Step 5: Re-derive the raw per-step TensorBoard/`.npy` data for at
  least one seed for BOTH bars** (not summary-JSON-only trust). Watch the
  eval video for any seed showing nonzero behavioral discovery (rest
  frame vs. peak-height frame, genuinely gripped pose).
- [ ] **Step 6: Report H_joint's own verdict** per the spec's own rule —
  falsified only if both bars fail across all 3 seeds; explicitly call
  out a SPLIT result (mechanism-level bar passes in ≥1 seed, behavioral
  bar still 0/24) as its own distinct outcome if it occurs, matching this
  exact env cfg's own immediately-prior exploration-bonus SPLIT
  precedent.
- [ ] **Step 7**: Full teardown if cloud-dispatched; report elapsed cost
  against the $6 cap (cumulative across Tasks 1/2/3/4). Desktop: verify
  `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh` all clear/AVAILABLE.
- [ ] **Step 8: Commit** any code fixes only (training runs themselves
  aren't committed) — none expected unless Step 2's real dispatch
  surfaces a bug in Tasks 1-2's code, in which case fix it, re-verify
  those tasks' own unit tests still pass, and commit the fix separately
  before re-running the affected seed(s).

---

## Task 4 — Real H_taskspace run: Condition B, 3 seeds, both falsification bars

**Files:** none new, unless Condition B's critic-divergence risk
materializes (Global Constraints — a scoped `PPORunnerCfg` subclass +
small `train_franka.py` variant-dispatch extension, built only if
triggered).

**Interfaces:**
- Consumes: `--variant die-d8-big-taskspace-antipodal` (Task 2).
- Produces: 3 full checkpoints (seeds 42/123/7) + both falsification bars'
  numbers per seed for H_taskspace — real evidence for the outcome
  matrix's H_taskspace axis. **Executed unconditionally, regardless of
  Task 3's own H_joint result** — Condition B is not gated on Condition A
  falsifying first (per the spec's own explicit design).

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per
  Global Constraints (re-check fresh, do not reuse Task 3's dispatch
  decision).
- [ ] **Step 2**: For each seed (42, 123, 7) — 3 runs total, from scratch:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant die-d8-big-taskspace-antipodal --seed 42 --num_envs 4096 --max_iterations 1500"
  ```

  (Repeat for seeds 123 and 7.) **Watch `Loss/value_function` in
  TensorBoard throughout each run, not just at the end** — Experiment
  11's own AR4-era critic-divergence signature (`Loss/value_function`
  exploding from ~0 to ~5e23) is the specific real risk this condition
  carries. If observed: per Global Constraints, build a
  Condition-B-scoped `PPORunnerCfg` subclass with `clip_actions=5.0`
  (Experiment 11's own fix), a minimal `train_franka.py` extension to
  select it for this variant only, commit that fix separately citing
  Experiment 11 (`[[ppo-critic-divergence]]`), and re-run the affected
  seed(s) — flag this explicitly in this task's own report as a real,
  if-triggered, scope addition beyond Task 2's original file list.
- [ ] **Step 3: Mechanism-level bar, per seed** — same protocol as Task 3
  Step 3, on Condition B's own checkpoints. **H_taskspace's
  mechanism-level claim is falsified only if the final-100-iteration mean
  is `< 1e-4` in all 3 seeds.**
- [ ] **Step 4: Behavioral bar, per seed** — same protocol as Task 3 Step
  4, `--variant die-d8-big-taskspace-antipodal`. **Falsified only if all
  3 seeds show 0/24 total.**
- [ ] **Step 5: Re-derive the raw per-step data for at least one seed for
  BOTH bars.** Watch the eval video for any seed showing nonzero
  behavioral discovery.
- [ ] **Step 6: Report H_taskspace's own verdict** per the spec's own
  rule, same SPLIT-aware reporting discipline as Task 3 Step 6.
- [ ] **Step 7**: Full teardown if cloud-dispatched; report elapsed cost
  against the $6 cap (cumulative across Tasks 1/2/3/4 — this is the
  final task in the cap's own scope, report the running total explicitly).
  Desktop: verify all clear/AVAILABLE.
- [ ] **Step 8: Commit** any code fixes (including the critic-divergence
  contingency fix, if triggered) separately, with their own messages.

---

## Task 5 — Verdict: 5-row outcome matrix, `ROADMAP.md` + kb update

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry: per-seed
  mechanism-level numbers, per-seed behavioral numbers, and each
  condition's own verdict (H_joint, H_taskspace), THEN the combined
  5-row-outcome-matrix classification (spec's own table: both confirmed /
  H_joint falsified+H_taskspace confirmed (the exact AR4 Experiment
  10→11 replay pattern) / mixed falsified+SPLIT / both SPLIT / both
  falsified — the only row that is a dispositive negative for Direction 1
  on Franka/d8), plus cost against the $6 cap.
- Create or modify: a `kb/wiki/experiments/` article (likely
  `kb/wiki/experiments/d8-antipodal-grasp-quality.md`, check this repo's
  kb-maintenance convention before finalizing the filename) cross-linking:
  - `kb/wiki/concepts/grasp-mechanics-antipodal-vs-magnitude.md` and
    `kb/wiki/experiments/experiment-09-antipodal-grasp-bonus.md`/
    `experiment-10-antipodal-threshold-action-scale-solver.md`/
    `experiment-11-taskspace-ik.md` — the exact AR4-era arc this
    experiment tests transfer of onto Franka, stating explicitly whether
    each condition's real result did or did not replay that arc's own
    pattern.
  - `kb/wiki/concepts/action-space-design.md` — the joint-space-vs-
    task-space finding this experiment's dual-condition structure was
    built to test.
  - `kb/wiki/experiments/exploration-bonus-grasp-discovery.md` — the
    immediately-prior SPLIT result on this identical env cfg whose own
    "forward pointer" explicitly named this antipodal-grasp-quality axis
    as the honest next candidate direction; record whether this
    experiment's own result confirms or complicates that forward pointer.
  - `kb/wiki/concepts/reach-grasp-lift-gap.md` and, if Task 4's
    contingency fired, `kb/wiki/concepts/ppo-critic-divergence.md`.

**Interfaces:**
- Consumes: Tasks 3 and 4's real per-seed results (both bars, both
  conditions).
- Produces: the closing, evidence-backed verdict for this experiment,
  including the honest next candidate direction per CLAUDE.md's "Claude's
  role" (don't just record a result without a forward pointer) — informed
  by which of the 5 outcome-matrix rows the real result actually landed
  in, not a generic template answer.

- [ ] **Step 1**: Write both conditions' own verdicts against the spec's
  pre-registered falsification rule, per seed, both bars — explicit,
  never averaged.
- [ ] **Step 2**: Classify the combined result against the spec's 5-row
  outcome matrix and state the row's own reading in plain language.
- [ ] **Step 3**: Include the instrumented `max_height_gain`/
  `max_consecutive_lifted_steps` numbers (not just discovery fractions)
  and confirm the eval videos were actually watched, not just JSON
  trusted.
- [ ] **Step 4**: Update `ROADMAP.md` and the kb article(s) per the Files
  section, per this repo's continuous-kb-update convention (not batched
  to session end).
- [ ] **Step 5**: State the honest next candidate direction given the
  real result — per the spec's own table, this could range from "closes
  Direction 1 for Franka/d8, escalate to Direction 2 (physical parameters)
  or a genuinely new direction" (both-falsified row) to "extend to
  d10/d12/d20 next" (a confirmed-in-≥1-condition row) — do not default to
  a generic template answer regardless of which row actually occurred.
- [ ] **Step 6**: Commit and push.

```bash
git add ROADMAP.md kb/wiki/experiments/
git commit -m "verdict: d8 antipodal/force-closure grasp-quality reward, joint-space + task-space (H_joint/H_taskspace)"
git push origin main
```

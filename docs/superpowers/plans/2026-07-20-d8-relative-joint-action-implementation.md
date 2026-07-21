# d8 relative/delta joint-position action (H_relative) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test H_relative — does swapping `FrankaDieLiftJointD8BigAntipodalEnvCfg`'s
inherited `JointPositionActionCfg` (absolute joint-space target) for
`RelativeJointPositionActionCfg` (delta/incremental joint-space target,
`scale=0.1`), with every other field (antipodal reward, contact-sensor
scene, PPO recipe) byte-identical, break the exact-`0.0`-contact-forever
pattern the root-cause doc found in all 8 previously-measured joint-space
checkpoints. Falsification/confirmation is defined **per seed, across a
5-checkpoint trajectory** (iterations 0/100/300/700/1499), not a
final-iteration snapshot alone — the whole point is distinguishing "the
diagnosed mechanism is fixed" from "the same collapse, just delayed."
Anything that doesn't cleanly land in FALSIFIED or CONFIRMED (per the
spec's exact numeric bars) is reported as SPLIT, this arc's own standing
precedent for not forcing an ambiguous result into a binary verdict.

**Why this, why now:** `kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s
"Root cause investigation (2026-07-21 follow-up)" section closed *why*
joint-space regresses to exact-zero contact (a configuration-dependent
absolute-target action-to-motion mapping, cross-validated against two
on-point action-space-comparison papers plus PPO's own documented
entropy-narrowing dynamic) and named `RelativeJointPositionActionCfg` as
"the most direct next test this investigation surfaces" — flagged for
Principal's own next-direction call, not decided by that investigation.
The design spec (`docs/superpowers/specs/2026-07-20-d8-relative-joint-
action-design.md`) picked it up, confirmed the exact mechanism contrast by
direct Isaac Lab v2.3.1 source read, grounded `scale=0.1` in a real shipped
precedent (Kuka Allegro dexsuite), and pre-registered the falsification
bar this plan executes.

**Architecture:** (1) one new leaf env-cfg class
(`FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`, `tasks/franka/
dice_lift_joint_env_cfg.py`) extending `FrankaDieLiftJointD8BigAntipodalEnvCfg`
(Condition A of the closed antipodal arc) purely additively — reuses its
entire inherited chain (d8 48mm-parity object, `AntipodalGraspRewardsCfg`,
`FrankaDieLiftContactSceneCfg`) and overwrites only `self.actions.arm_action`
in its own `__post_init__`, mirroring `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`'s
own "call `super().__post_init__()` first, then re-assert the one changed
field" pattern exactly; (2) `_PLAY` variant + `train_franka.py`/
`franka_checkpoint_review.py`/`sync_run_to_gcs.py` `--variant` wiring,
mirroring every prior leaf class's own three-script wiring; (3) an
empirical action-manager check (build the env, read
`env.action_manager.get_term("arm_action")`'s real class and
`env.action_manager.total_action_dim`/`action_term_dim`) — do not trust the
class definition alone, this arc's own standing "confirmed empirically, not
just asserted from the class hierarchy" discipline
(`FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`'s own docstring precedent for
exactly this kind of MRO/field-override claim); (4) a small extension to
`scripts/diag_antipodal_root_cause.py` — a new `--variant condition-relative`
choice importing the new `_PLAY` class, reusing 100% of its existing
per-step contact/reward/pose instrumentation and `REWARD_TERMS` list
unchanged (this new env cfg reuses `AntipodalGraspRewardsCfg` byte-identical,
so no new term needs adding); (5) one real 3-seed/1500-iteration training
run with checkpoints preserved and GCS-synced throughout (not just at the
end — the root-cause doc's own GRUB-corruption incident lost un-synced
intermediate checkpoints, and this experiment's own measurement plan
depends on having all 5 survive); (6) the 5-checkpoint diagnostic sweep
(15 rollouts: 5 checkpoints × 3 seeds) plus the final-checkpoint behavioral
bar (`franka_checkpoint_review.py`'s existing sustained-lift protocol); (7)
verdict — `ROADMAP.md` + kb update extending
`kb/wiki/experiments/d8-antipodal-grasp-quality.md` again (a direct
continuation of that arc, not a new article), explicitly comparing this
run's own 5-checkpoint trajectory against the root-cause doc's own
already-measured absolute-joint-space and task-space trajectories at the
same 5 checkpoints.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`dice_lift_joint_env_cfg.py` only — no new reward/scene/pure-math module,
unlike the antipodal plan this one is a template for), desktop-first GPU
dispatch (`scripts/check_gpu_availability.sh`/`scripts/run_on_desktop_gpu.sh`),
GCP cloud fallback (`docs/cloud/dispatch-checklist.md`). **Desktop status
checked at plan-authoring time (2026-07-20): UNREACHABLE**
(`scripts/check_gpu_availability.sh` → `TARGET=cloud`, exit 2, `curl`
DNS resolution to `home.local` timed out — treated as UNKNOWN per this
project's own "never treat can't-tell as a green light" rule, not BUSY) —
**re-check fresh at each task's own dispatch time**, do not assume this
result still holds by the time execution starts.

Spec: `docs/superpowers/specs/2026-07-20-d8-relative-joint-action-design.md`.
Template/precedent for this plan's own structure, cost-cap methodology, and
dispatch discipline: `docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-
quality-implementation.md` (the immediately-prior plan on this exact env-cfg
family — this plan reuses its Global Constraints almost verbatim, scaled
down for a single-condition/3-run scope instead of a two-condition/6-run
one) and `scripts/diag_antipodal_root_cause.py` (the just-built diagnostic
this plan extends rather than rebuilds — confirmed reusable with a small
addition, not a rewrite: it already takes contact-frequency/antipodal-
frequency/reward-term measurements at arbitrary checkpoints for either of
the antipodal arc's two existing conditions; it only needs a third
`--variant` choice pointed at this plan's new env cfg, since the reward
term list, contact-sensor names, and measurement logic are all unchanged).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Design notes (flagged for controller review, not silently assumed)

**1. No new reward/scene/pure-math code — the entire "new infrastructure"
surface is one env-cfg class plus one CLI choice per script.** Unlike the
antipodal plan this one templates off of, there is no TDD task here: the
spec's own "Scope" section is explicit that `AntipodalGraspRewardsCfg`/
`FrankaDieLiftContactSceneCfg`/the 41-dim observation schema/PPO runner cfg
are all reused byte-identical, and Task 1 of the antipodal plan already
empirically confirmed the contact-sensor wiring reads real, correctly-shaped
force data under this exact scene cfg. Re-verifying that here would
re-test an already-closed question for no isolation benefit. This plan's
own genuinely new empirical-verification burden is narrower and different
in kind: not "does the contact sensor work" (already proven) but "does the
action term actually swap to the relative/delta class, at the expected
dimensionality" (Task 1 below).

**2. The empirical action-manager check must inspect the live `ActionTerm`
object, not just print a config field.** A naive check that reads
`env_cfg.actions.arm_action.__class__.__name__` off the *unbuilt* cfg
object would not actually prove anything — the interesting failure mode is
the exact one `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`'s own docstring
already flags for this same class-hierarchy composition pattern: an
`__post_init__` override that runs but gets silently overwritten by a
later step in the MRO, or a field assignment that doesn't propagate into
the constructed `ActionTerm`. Task 1's check must build a real
`ManagerBasedRLEnv` (small `num_envs`, no training/checkpoint needed) and
read `env.action_manager.get_term("arm_action")`'s real Python type (must
be `isaaclab.envs.mdp.actions.joint_actions.RelativeJointPositionAction`,
not `JointPositionAction`) plus `env.action_manager.total_action_dim`
(must stay `8` — 7 relative-joint + 1 gripper, byte-identical to Condition
A's own absolute-joint action space) — confirmed by direct read of
`isaaclab/managers/action_manager.py` (this project's pinned `v2.3.1`,
local checkout at `/home/saps/IsaacLab/source/isaaclab/isaaclab/managers/
action_manager.py`): `ActionManager.get_term(name)` returns the live
`ActionTerm` instance from its own `self._terms` dict, and
`total_action_dim`/`active_terms` are real properties reflecting the
actually-constructed terms, not the cfg. `print(env.action_manager)` also
renders a table via `ActionManager.__str__` — a convenient secondary
sanity check, not a substitute for the type/dim assertions above.

**3. `scale=0.1`'s own stability is a real, if lower, risk here — flagged,
not pre-resolved, same treatment the antipodal plan gave Condition B's
critic-divergence risk.** The spec is explicit `scale=0.1` is an
implementer-set, precedent-grounded starting value, not load-bearing for
H_relative's own falsification bar, with `0.0625` (the UR10e reach
precedent) as a documented fallback if a short smoke-test rollout shows
visibly excessive per-step jitter or an unstable/oscillating gripper
approach. This is judged a real but bounded risk: relative/delta joint
control is a well-precedented, stable actuation family in Isaac Lab's own
shipped configs (unlike Condition B's task-space/IK swap, which had a
concrete prior AR4-era failure precedent — Experiment 11's
`Loss/value_function` blowup — motivating an explicit contingency-fix
plan). No contingency fix is pre-built here; Task 2's smoke test is the
one designated checkpoint for catching an obviously-wrong scale before
committing the full 3-seed budget to it.

## Global Constraints

- **Do not modify `FrankaDieLiftJointD8BigAntipodalEnvCfg`,
  `AntipodalGraspRewardsCfg`, or `FrankaDieLiftContactSceneCfg` in place.**
  Every new mechanism in this plan is additive: one new leaf env-cfg class
  (`FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`) extending Condition A,
  never editing it — keeps Condition A/B's own already-closed results and
  any other concurrent workstream's use of the same base classes completely
  unaffected. Check `git status` fresh at the start of execution for any
  other concurrent workstream's untracked/in-progress files before
  touching anything.
- **`scale=0.1`, `use_zero_offset=True` are this plan's own starting
  values, not tunable mid-experiment.** Per the spec: not load-bearing for
  H_relative's falsification bar, a Tier-2 hillclimb candidate later if the
  mechanism validates but needs retuning. If Task 2's smoke test shows
  visibly excessive jitter/instability, fall back to `scale=0.0625` (the
  documented UR10e precedent) — not a fresh guess — and note the
  substitution explicitly in that task's own report; do not silently swap
  it without recording why.
- **No reward, scene, or PPO-runner-cfg changes pre-authorized.**
  `AntipodalGraspRewardsCfg`/`FrankaDieLiftContactSceneCfg`/
  `FrankaLiftPPORunnerCfg` (`max_iterations=1500`, `gamma=0.98`,
  `save_interval=50`) reused unmodified — the action term is this
  experiment's only variable, per the spec's own "Global constraints".
  If a real critic-divergence signature appears (`Loss/value_function`
  exploding, `[[ppo-critic-divergence]]`'s own Experiment 11 signature —
  judged a lower-risk scenario here than Condition B's task-space swap
  since this stays within the same joint-space PD-servo actuation family
  Condition A already trains stably under, but not zero-risk given the
  action-magnitude-per-step semantics genuinely change), applying a scoped
  fix is implementation-plan-level judgment at the time, not pre-resolved
  here — do not build a contingency `PPORunnerCfg` subclass preemptively;
  only if Task 3's real run actually shows divergence, scoped to this
  variant only, flagged explicitly in that task's own report.
- **Checkpoints must be preserved AND GCS-synced throughout training, not
  only at the end.** `save_interval=50` already covers all 5 measurement
  checkpoints exactly (0, 100, 300, 700, 1499 are all multiples of 50, or
  the final iteration) with no schedule adjustment needed — confirmed by
  direct read of `tasks/franka/agents/rsl_rl_ppo_cfg.py:28-29`
  (`max_iterations=1500`, `save_interval=50`). Run a background
  `while true; do gsutil -m rsync -r <run-dir> gs://.../ ; sleep 300; done`
  loop in its own detached `tmux` session alongside each training run
  (the root-cause doc's own documented mitigation, `docs/cloud/dispatch-
  checklist.md`'s "Known infra gaps" — a repeat of that exact GRUB-
  corruption incident must not cost this plan its own measurement data a
  second time).
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
     project-wide) **or dispatching to the same desktop.** Each task below
     must check fresh and genuinely queue/wait/retry if busy — never
     assume availability just because an earlier task in this same plan
     found it available, and never treat a busy quota as a reason to skip
     a run.
  7. **Tasks 3 and 4 may be combined onto a single provisioned cloud
     instance if convenient** (avoids a second ~15-20min Isaac Lab install
     window) — implementer's judgment, not required. If combined, Task 4's
     own dispatch-check/teardown-verification steps still apply at the
     point Task 4's own work is actually done, not skipped because Task 3
     already "checked".
- **Cost cap: $5 cumulative across Tasks 1/2/3/4 combined, notify the
  controller if exceeded.** This is a cloud-fallback safety backstop, not
  an expected spend — desktop dispatch brings actual cost toward $0.
  Derivation, scaled down from the antipodal plan's own $6 cap for a
  smaller, single-condition/3-run scope (not this plan's own two-condition/
  6-run scope), grounded in that same arc's own real recorded spend:
  - Task 3 (real 3-seed/1500-iteration training, single condition) is the
    exact same shape as the closed antipodal plan's own Task 3 (H_joint:
    3 seeds, 1500 iterations, same `FrankaDieLiftJointD8BigAntipodalEnvCfg`
    lineage, same `num_envs=4096`), which measured real actual spend of
    **≈$1.6** (`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s own
    "Cost" section, 2.13hr on-demand `g2-standard-4`+`nvidia-l4`
    instance-uptime). Used directly, not re-derived from a per-hour rate
    guess.
  - Task 4 (measurement: 15 diagnostic rollouts — 5 checkpoints × 3 seeds,
    no additional training — plus the final-checkpoint behavioral eval,
    all 3 seeds) is the same kind of diagnostic-only work the root-cause
    doc's own follow-up did (rollouts via `diag_antipodal_root_cause.py`,
    no retraining needed since Task 3 already preserves all 5 checkpoints
    per seed): that follow-up's own total spend, **≈$2.5**, included TWO
    full 1500-iteration retrains (its own checkpoints weren't
    intermediate-synced, forcing fresh retrains for the trajectory) plus
    10 rollouts plus a genuine GRUB-corruption instance-replacement
    incident. This plan's own Task 4 needs zero retraining (Task 3's
    checkpoints are already GCS-synced throughout) and one more rollout
    set (15 vs. 10) — rollouts themselves are cheap (128-env, ~249-step
    headless episodes, no training loop) relative to a full retrain, so
    Task 4's own cost is estimated well under the root-cause doc's $2.5
    (which was dominated by its two retrains) — **≈$1.3** (15 short
    rollouts + 3-seed behavioral eval, with margin for per-checkpoint
    model-load overhead).
  - Tasks 1-2 (env-cfg wiring + empirical action-manager check + one
    bounded ~20-iteration smoke test, single condition unlike the
    antipodal plan's own dual-condition Tasks 1-2) ≈ half that plan's own
    $0.50 two-condition smoke-test estimate ≈ **$0.25**.
  - Sum ≈ $1.6 + $1.3 + $0.25 ≈ **$3.15**, rounded up to **$5** (not $6)
    for margin given this arc's own repeated real infra incidents (a
    genuine SPOT preemption + GRUB boot corruption, and separately a
    cluster of 3 preemptions in ~3 hours, both already hit on this exact
    diagnostic pipeline per `docs/cloud/dispatch-checklist.md`'s "Known
    infra gaps") — enough buffer for one preemption-retry cycle without
    reaching the antipodal plan's own larger two-condition $6 cap, which
    this plan's narrower single-condition scope does not need.
- **Real evidence over proxies at every eval**: the 5-checkpoint
  contact-frequency/antipodal-frequency/`reaching_object`-trajectory data
  (`diag_antipodal_root_cause.py`'s own `.npz`+summary JSON, re-derived
  from raw arrays for at least one seed, not summary-JSON-only trust) AND
  `franka_checkpoint_review.py`'s `max_height_gain`/
  `max_consecutive_lifted_steps`/sustained-lift count (behavioral bar) AND
  a reviewed eval video for any positive result (rest frame vs. peak-height
  frame, genuinely gripped pose) — not a shaped reward scalar or exit code
  alone (Experiment 16 precedent, reused by every prior spec in this arc).
- **Report the full per-seed 5-checkpoint trajectory, never collapsed to a
  single final number.** Per the spec's own falsification-bar design, a
  final-iteration-only number cannot distinguish "fixed" from "delayed
  collapse" — Tasks 3/4/5 must preserve and report all 5 checkpoints' own
  contact-frequency values per seed explicitly.
- **Report a SPLIT explicitly if the per-seed data doesn't cleanly match
  either the spec's CONFIRMED or FALSIFIED numeric bar** — this arc's own
  standing precedent (H_taskspace's own seed-level heterogeneity,
  `exploration-bonus-grasp-discovery`'s own SPLIT) for not forcing an
  ambiguous result into a binary verdict. A genuine per-seed split between
  the two shapes is itself a SPLIT, not resolved by majority vote beyond
  what the spec's own "at least 2 of 3 seeds" language already specifies.
- **Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line** — each task
  below is executed by a freshly dispatched session; use that session's
  real ID, do not copy a fixed ID from this document.

---

## Task 1 — New leaf env cfg + script wiring + empirical action-manager verification

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add, near
  `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg` (the file's own precedent
  for "a leaf that re-asserts `self.actions.arm_action` after
  `super().__post_init__()`"):
  - `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg(FrankaDieLiftJointD8BigAntipodalEnvCfg)`
    — `__post_init__` calls `super().__post_init__()` (running Condition
    A's full inherited chain: die-swap/mass/scale, `AntipodalGraspRewardsCfg`,
    `FrankaDieLiftContactSceneCfg`, and Condition A's own absolute
    `JointPositionActionCfg` assignment), then overwrites
    `self.actions.arm_action` to
    `mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.1, use_zero_offset=True)`
    — the exact spec-specified values. Docstring cites the design spec's
    own exact mechanism contrast (`applied_target = raw_action * scale +
    current_joint_pos`, read fresh each control step, vs. Condition A's
    fixed-`default_joint_pos`-offset absolute target) and states plainly
    this is a NEW leaf, `FrankaDieLiftJointD8BigAntipodalEnvCfg` untouched.
    Class name keeps "Joint" (unlike the taskspace condition) since this
    stays genuinely joint-space — only "Relative" distinguishes it,
    matching the spec's own naming logic.
  - `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY` — same
    `num_envs=50`/`env_spacing=2.5`/`enable_corruption=False` pattern as
    every other `_PLAY` class in this file.
- Modify: `scripts/train_franka.py` — add one new `--variant` choice,
  `joint-die-d8-big-relative-antipodal` (imports
  `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`, `_log_suffix`
  `_jointdied8bigrelativeantipodal`), mirroring the existing
  `joint-die-d8-big-antipodal`/`die-d8-big-taskspace-antipodal` branches'
  exact structure (both the `elif` dispatch around line 276-283 and the
  `_log_suffix` dict around line 337-339). Help text cites this plan and
  the design spec, states explicitly this is joint-space with a
  delta/relative target (not a rename of the existing absolute-target
  variant).
- Modify: `scripts/franka_checkpoint_review.py` — same new `--variant`
  choice, importing `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY`,
  mirroring the existing dispatch at both the import-selection block
  (~line 261-264) and the corresponding env-cfg-instantiation block further
  down (~line 390-392) exactly.
- Modify: `scripts/sync_run_to_gcs.py` — add
  `"train_franka_jointdied8bigrelativeantipodal": "joint-die-d8-big-relative-antipodal"`
  to `VARIANT_MAP` (~line 59-79), matching `train_franka.py`'s new
  `_log_suffix` value exactly.
- Create: a small, throwaway verification script (or an inline snippet run
  via `isaaclab.sh -p -c` if that's cleaner — implementer's judgment,
  doesn't need to be a permanent file) that builds a
  `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY` env at small
  `num_envs` (8-16 range) and asserts/prints:
  - `type(env.action_manager.get_term("arm_action")).__name__ ==
    "RelativeJointPositionAction"` (NOT `JointPositionAction`) — read
    directly off the live `ActionManager._terms` dict via its own
    `get_term()` accessor, confirmed present at
    `isaaclab/managers/action_manager.py:403-412` in this project's
    pinned `v2.3.1`.
  - `env.action_manager.total_action_dim == 8` (7 relative-joint + 1
    gripper, unchanged from Condition A's own absolute-joint action
    space).
  - `env.action_manager.active_terms` includes both `arm_action` and
    `gripper_action`, and `env.action_manager.action_term_dim` (or
    equivalent per-term dim listing, confirm exact property name in the
    installed Isaac Lab version) shows `arm_action` at dim 7.
  - Optionally: step the env a few times with a nonzero constant action
    on `arm_action` and confirm the robot's joint positions actually move
    by an amount that does NOT depend on which pose the arm started at
    (a lightweight empirical spot-check of the "locally consistent"
    property the spec's own mechanism claim rests on) — implementer's
    judgment on how much of this extra check is worth the wall-clock,
    not required to close this task.
- Record the exact observed output in
  `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`'s own docstring, citing
  this empirical check — not asserted from the class hierarchy alone,
  matching `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`'s own established
  precedent for this exact kind of claim.

**Interfaces:**
- Consumes: `FrankaDieLiftJointD8BigAntipodalEnvCfg`,
  `AntipodalGraspRewardsCfg`, `FrankaDieLiftContactSceneCfg` (Task 1/2 of
  the antipodal plan, already merged to `main`) — read-only, nothing
  modified.
- Produces: `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg` (+ `_PLAY`),
  the new `--variant` choice on all 3 scripts, and the empirical
  confirmation that the action term/dimensionality actually swapped as
  designed — consumed by Task 2's smoke test and Tasks 3/4's real runs.

- [ ] **Step 1: Implement the new leaf env-cfg class + `_PLAY` variant**
  per the Files section.
- [ ] **Step 2: Wire `train_franka.py`/`franka_checkpoint_review.py`/
  `sync_run_to_gcs.py`** per the Files section. Smoke-check with `--help`
  on all 3 scripts (no sim launch) to confirm the new choice parses.
- [ ] **Step 3: Empirical action-manager verification.**
  `scripts/check_gpu_availability.sh` → dispatch per Global Constraints,
  `flock`-wrapped, non-headless if desktop, small `num_envs`. Confirm the
  type/dimensionality assertions above; record the exact output in the
  new class's own docstring.
- [ ] **Step 4: Verify GPU clear** per Global Constraints.
- [ ] **Step 5: Commit.**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py \
        scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: wire d8 relative/delta joint-position action env cfg (H_relative, Task 1)"
git push origin main
```

---

## Task 2 — Bounded smoke test + `diag_antipodal_root_cause.py` extension

**Files:**
- Modify: `scripts/diag_antipodal_root_cause.py` — add a third
  `--variant` choice, `condition-relative` (alongside the existing
  `condition-a`/`condition-b`), importing
  `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY` and dispatching to
  it in `main()`'s existing `if/else` (~line 163-166), extended to a
  `if/elif/else`. `REWARD_TERMS`, `FORCE_THRESHOLD`,
  `ANTIPODAL_COS_THRESHOLD`, and every other measurement/instrumentation
  line are unchanged — this env cfg reuses `AntipodalGraspRewardsCfg`
  byte-identical, so the existing term list is already correct for this
  third variant with zero further edits. Update the module docstring's
  own "Loads a trained checkpoint (either condition)" line to "any of the
  three conditions" and cite this plan.
- No other files touched.

**Interfaces:**
- Consumes: Task 1's `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg_PLAY`.
- Produces: the extended diagnostic script, smoke-tested against a real
  (if short) checkpoint — consumed by Task 4's real 5-checkpoint
  measurement sweep.

- [ ] **Step 1: Implement the `--variant condition-relative` extension**
  per the Files section.
- [ ] **Step 2: Bounded smoke test, ~20-iteration training run** — confirm
  training doesn't crash and every reward term (including
  `antipodal_grasp_quality`, `action_rate`, `joint_vel`) produces plausible
  (non-NaN, bounded-magnitude) values under the new action term.
  `scripts/check_gpu_availability.sh` → dispatch per Global Constraints,
  `flock`-wrapped, non-headless if desktop:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-relative-antipodal --num_envs 256 --max_iterations 20 --seed 42"
  ```

  Watch `Loss/value_function` in TensorBoard as an early, non-conclusive
  canary for critic instability (Design notes #3 — a real if lower risk
  than Condition B's task-space swap; a clean 20-iteration run does NOT
  rule out divergence appearing later in the real 1500-iteration run).
  Watch for visibly excessive per-step action-magnitude jitter or an
  obviously oscillating/unstable gripper approach — if present, fall back
  to `scale=0.0625` (the documented UR10e precedent) in Task 1's env-cfg
  class, re-run this smoke test, and note the substitution explicitly in
  this task's own report; do not silently proceed with a visibly broken
  scale.
- [ ] **Step 3: Smoke-test the extended diagnostic script itself** against
  this smoke-test checkpoint (`--variant condition-relative
  --checkpoint <this task's own model_N.pt> --num_envs 16`) — confirm it
  runs end-to-end, produces a `.npz`+summary JSON, and the in-script
  training-time-function cross-check assertion
  (`antipodal_grasp_bonus_raw` vs. this script's own recomputation)
  passes. This is a real functional test of Task 2's own new code, not
  just Task 1's env cfg.
- [ ] **Step 4: Verify GPU clear** per Global Constraints.
- [ ] **Step 5: Commit.**

```bash
git add scripts/diag_antipodal_root_cause.py
git commit -m "feat: extend diag_antipodal_root_cause.py with condition-relative variant (Task 2)"
git push origin main
```

---

## Task 3 — Real training run: 3 seeds, 1500 iterations, checkpoints preserved + synced throughout

**Files:** none new — runs Tasks 1-2's code through a real, full
1500-iteration training cycle.

**Interfaces:**
- Consumes: `--variant joint-die-d8-big-relative-antipodal` (Task 1).
- Produces: 3 full checkpoint sets (seeds 42/123/7), each with the 5
  measurement-point checkpoints (iterations 0/100/300/700/1499, all
  landing exactly on `save_interval=50`'s own schedule) preserved and
  GCS-synced throughout training — consumed by Task 4's measurement sweep.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per
  Global Constraints (re-check fresh, do not reuse Task 2's dispatch
  decision). Record dispatch target and (if cloud) instance creation
  timestamp.
- [ ] **Step 2**: Start the background incremental GCS-sync loop (Global
  Constraints — a detached `tmux` session running
  `while true; do gsutil -m rsync -r <run-dir> gs://.../ ; sleep 300; done`)
  BEFORE starting training, so no early checkpoint is missed.
- [ ] **Step 3**: For each seed (42, 123, 7) — 3 runs total, from scratch
  (no `--checkpoint`, matching this arc's own established recipe):

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-relative-antipodal --seed 42 --num_envs 4096 --max_iterations 1500"
  ```

  (Repeat for seeds 123 and 7.) **Watch `Loss/value_function` in
  TensorBoard throughout each run, not just at the end** — per Global
  Constraints, a real if lower-probability risk than Condition B's
  task-space swap. If a genuine divergence signature appears, this is a
  real scope addition beyond this plan's original file list — flag it
  explicitly in this task's own report, apply a scoped fix (a
  variant-specific `PPORunnerCfg` subclass, built only if actually
  triggered, never touching the shared default), and re-run the affected
  seed(s).
- [ ] **Step 4**: Confirm all 3 seeds' checkpoints at iterations
  {0, 100, 300, 700, 1499} exist, are non-corrupt (file-size sanity check
  at minimum — a real checkpoint for this project's PPO configs is
  ~1.27MB, per `docs/cloud/dispatch-checklist.md`'s own documented SPOT-
  truncation gotcha — or a real `torch.load` try/except), and are synced
  to GCS. This is the single most important correctness check in this
  task — a repeat of the root-cause doc's own un-synced-intermediate-
  checkpoint gap would silently break Task 4's entire measurement plan.
- [ ] **Step 5**: Full teardown if cloud-dispatched (stop the sync loop
  first); report elapsed cost against the $5 cap (cumulative across Tasks
  1/2/3/4). Desktop: verify `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh` all clear/AVAILABLE.
- [ ] **Step 6: Commit** any code fixes only (training runs themselves
  aren't committed) — none expected unless Step 3's real dispatch surfaces
  a bug in Task 1-2's code or a triggered critic-divergence contingency,
  in which case fix it, re-verify Task 1's own empirical checks still
  hold, and commit the fix separately before re-running the affected
  seed(s).

---

## Task 4 — Measurement: 5-checkpoint diagnostic sweep + final behavioral bar, all 3 seeds

**Files:** none new — runs Task 2's extended diagnostic script and
`franka_checkpoint_review.py` against Task 3's real checkpoints.

**Interfaces:**
- Consumes: Task 3's 15 checkpoints (5 per seed × 3 seeds).
- Produces: the per-seed 5-checkpoint contact-frequency/antipodal-
  frequency/`reaching_object`-trajectory data, classified against the
  spec's exact CONFIRMED/FALSIFIED/SPLIT numeric bar, plus the
  final-checkpoint behavioral bar (sustained-lift, all 3 seeds) — real
  evidence for Task 5's verdict.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per
  Global Constraints (re-check fresh; may reuse Task 3's own instance if
  still up and convenient, per Global Constraints item 7, but re-verify
  availability/teardown state at this task's own boundary regardless).
- [ ] **Step 2**: For each seed (42, 123, 7), for each checkpoint
  (iterations 0, 100, 300, 700, 1499) — 15 rollouts total:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/diag_antipodal_root_cause.py \
    --variant condition-relative --checkpoint <path/to/seed42/model_0.pt> --num_envs 64 \
    --output_npz logs/diag_relative/seed42_iter0.npz --headless"
  ```

  (Repeat for all 15 combinations. Non-headless if dispatched to desktop,
  per Global Constraints — this script is headless-friendly by design but
  the "no `--headless` locally" convention still applies unless cloud.)
- [ ] **Step 3**: For each seed, tabulate the 5-checkpoint
  `contact_frequency` trajectory (matching the root-cause doc's own
  table format exactly) alongside the antipodal-satisfying frequency and
  `reaching_object` raw reward trajectory.
- [ ] **Step 4**: Apply the spec's exact falsification/confirmation bar
  per seed:
  - **FALSIFIED** if iter-1499 contact frequency `< 0.01` AND `<50%` of
    that seed's own peak value across the 5 checkpoints.
  - **CONFIRMED** if iter-1499 contact frequency `≥ 0.05` AND not less
    than that seed's own iter-700 value.
  - **Anything else is SPLIT** (per seed) — including a signal that grows
    but plateaus below 0.05.

  Report the overall verdict per the spec's "at least 2 of 3 seeds" rule
  for both FALSIFIED and CONFIRMED; anything not cleanly matching either
  is reported as an overall SPLIT.
- [ ] **Step 5**: Final-checkpoint behavioral bar, all 3 seeds —
  `franka_checkpoint_review.py --variant joint-die-d8-big-relative-antipodal
  --checkpoint <seed's model_1499.pt> --num_envs 8`, existing
  post-`977a748` settle-detection logic, video + `max_height_gain`/
  `max_consecutive_lifted_steps`. Report per-seed sustained-lift counts
  (out of 8) and the 24-total figure, independent of where each seed's
  mechanism-level trajectory landed above.
- [ ] **Step 6**: Re-derive the raw per-step `.npz` data for at least one
  seed's full 5-checkpoint trajectory (not summary-JSON-only trust).
  Watch the eval video for any seed showing nonzero behavioral discovery
  (rest frame vs. peak-height frame, genuinely gripped pose) before
  reporting it as a real positive.
- [ ] **Step 7**: Explicit comparison against the root-cause doc's own
  already-measured trajectories at the identical 5 checkpoints
  (absolute-joint-space: exact `0.0` at all 5; task-space seed 123:
  0 → 0.00047 → 0.6297 → 0.8539 → 0.8781) — the "did this actually change
  the shape of the curve" question this experiment exists to answer,
  answered from matched real data, not inference.
- [ ] **Step 8**: Full teardown if cloud-dispatched; report elapsed cost
  against the $5 cap (cumulative across Tasks 1/2/3/4 — this is the final
  task in the cap's own scope, report the running total explicitly).
  Desktop: verify all clear/AVAILABLE.
- [ ] **Step 9: Commit** any code fixes only, if Step 2 surfaces a bug in
  Task 2's diagnostic-script extension — fix, re-verify, commit separately.

---

## Task 5 — Verdict: `ROADMAP.md` + kb update (extend `d8-antipodal-grasp-quality.md`)

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry: the per-seed
  5-checkpoint contact-frequency trajectory (not collapsed to a final
  number), the overall CONFIRMED/FALSIFIED/SPLIT classification per the
  spec's numeric bar, the behavioral-bar numbers (per-seed and 24-total),
  the explicit curve-shape comparison against absolute joint-space and
  task-space's own already-measured trajectories, and cost against the $5
  cap.
- Modify: `kb/wiki/experiments/d8-antipodal-grasp-quality.md` — extend
  (do not create a new article; this is a direct continuation of that
  arc, matching the "Root cause investigation" section's own precedent of
  extending the same article rather than forking) with a new dated
  section (e.g. "H_relative test (2026-07-2X follow-up)") reporting:
  - The full per-seed 5-checkpoint trajectory table, matching Finding 1's
    own table format exactly, extended with this experiment's own rows.
  - The CONFIRMED/FALSIFIED/SPLIT classification and what it means for
    the root-cause doc's own diagnosed mechanism (did a locally-consistent
    action-to-motion mapping actually break the "transiently discover,
    then abandon" pattern Finding 3 identified, or reproduce it in a
    reshaped form).
  - Cross-links: `kb/wiki/concepts/action-space-design.md` (the
    joint-space-vs-task-space axis this experiment narrows further, into
    delta-vs-absolute within joint-space), `[[ppo-critic-divergence]]` (if
    Task 3's contingency fired), `[[reach-grasp-lift-gap]]`, and
    `CLAUDE.md`'s North Star (a genuinely joint-space fix, if confirmed,
    needs no arm-specific IK-controller configuration at all — call this
    out explicitly if CONFIRMED, since it is independently interesting for
    generalization beyond just closing this one experiment).
  - The honest next candidate direction given the real result — per the
    spec's own framing, this ranges from "a genuinely joint-space,
    arm-agnostic fix for the diagnosed collapse — extend to d10/d12/d20
    or reconsider whether task-space is even still needed as the default"
    (CONFIRMED) to "the collapse is intrinsic to absolute-vs-relative
    targeting only in a way this project hasn't yet isolated further, or
    is really about something else entirely — task-space remains this
    project's own working answer for d8" (FALSIFIED) to a SPLIT-specific
    honest read (e.g. "some seeds broke the pattern, worth a hillclimb on
    `scale` before concluding either way") — do not default to a generic
    template answer regardless of which outcome actually occurred.

**Interfaces:**
- Consumes: Task 4's real per-seed 5-checkpoint data and final-checkpoint
  behavioral bar.
- Produces: the closing, evidence-backed verdict for H_relative, plus a
  forward pointer for the next candidate direction.

- [ ] **Step 1**: Write the verdict against the spec's pre-registered
  falsification rule, per seed, the full 5-checkpoint trajectory — never
  collapsed to a final number.
- [ ] **Step 2**: Include the curve-shape comparison against the
  root-cause doc's own matched absolute-joint-space/task-space trajectories
  at the same 5 checkpoints.
- [ ] **Step 3**: Include the behavioral-bar numbers and confirm eval
  videos were actually watched, not just JSON trusted.
- [ ] **Step 4**: Update `ROADMAP.md` and the kb article per the Files
  section, per this repo's continuous-kb-update convention (not batched
  to session end).
- [ ] **Step 5**: State the honest next candidate direction given the
  real result.
- [ ] **Step 6**: Commit and push.

```bash
git add ROADMAP.md kb/wiki/experiments/d8-antipodal-grasp-quality.md
git commit -m "verdict: d8 relative/delta joint-position action (H_relative)"
git push origin main
```

# AR4 Franka-fixes transfer (relative joint-space action on Experiment 26) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test H_ar4_relative — does swapping `Ar4PickPlaceGraspGoalEnvCfg`'s
(Experiment 26) inherited `JointPositionActionCfg` (absolute joint-space,
`scale=0.5`) for `RelativeJointPositionActionCfg` (delta/incremental
joint-space, `scale=0.1`), with the antipodal-gated `RewardsCfg`/
`Ar4PickPlaceGraspGoalSceneCfg`/`TerminationsCfg`/`EventCfg`/PPO recipe
otherwise byte-identical, break AR4's own historical, never-cleanly-
root-caused Experiment 26 null (`cube_reached_goal 0.0000`, "the
antipodal grasp gate is apparently never satisfied") — the same fix
already CONFIRMED 3/3 seeds on Franka under the analogous mechanism.
Falsification/confirmation is defined **per seed, across a 5-checkpoint
trajectory** (iterations 0/100/300/700/1499, all landing on
`Ar4PickPlacePPORunnerCfg.save_interval=50`'s own schedule), not a
final-iteration snapshot alone. Anything that doesn't cleanly land in
FALSIFIED or CONFIRMED per the spec's exact numeric bars is reported as
SPLIT. **Condition A (unmodified Experiment 26) is also freshly retrained,
3 seeds, under current infra — not assumed from the historical run —**
since Isaac Lab has been updated since Experiment 26 last trained, and
whether the historical null still reproduces at all is itself part of
what this plan measures.

**Why this, why now:** `docs/superpowers/specs/2026-07-21-ar4-franka-
fixes-transfer-design.md` (this plan's spec) confirmed, via a live cloud
smoke test, that `Ar4PickPlaceGraspGoalEnvCfg` still builds and trains
cleanly under the currently-pinned Isaac Lab v2.3.1 / Isaac Sim 5.1.0
stack (three small, now-solved environment-setup gaps aside — asset
build, `xacro`, an `ament_index_python` shim), confirmed the antipodal
grasp-quality reward is already AR4-native and already wired into
Experiment 26's reward/observation/termination chain (no porting needed),
re-verified AR4's `mu=1.0`/`ANTIPODAL_COS_THRESHOLD=-0.7071` friction
constant is still physically correct (not a stale assumption), and chose
Experiment 26 over Experiment 11 (AR4's own prior positive task-space/IK
result) as the baseline specifically because Experiment 26's
absolute-joint-space null is the still-open analogue of Franka's own
falsified H_joint condition — testing whether the identical fix
(`RelativeJointPositionActionCfg`, `scale=0.1`) resolves it is a live,
undone question with direct North Star relevance (a genuinely joint-space
fix needs no arm-specific IK/kinematic-chain controller at all).

**Architecture:** (1) a small prerequisite CLI addition — `scripts/
train.py` currently has no `--seed` flag (unlike `scripts/train_franka.py`,
which already has one), needed before either condition can run 3 distinct
seeds; (2) one new leaf env-cfg class
(`Ar4PickPlaceGraspGoalRelativeEnvCfg`, added to the SAME file as
Condition A, `tasks/ar4/pickplace_graspgoal_env_cfg.py`) extending
`Ar4PickPlaceGraspGoalEnvCfg` purely additively — reuses its entire
inherited chain (scene, `RewardsCfg`, `TerminationsCfg`, `EventCfg`,
30s episode) and overwrites only `self.actions.joint_positions` in its own
`__post_init__`, mirroring Franka's own
`FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`'s "call `super().
__post_init__()` first, then re-assert the one changed field last"
pattern exactly, plus a new flat boolean CLI flag
(`--graspgoalrelative`, matching AR4's own existing flat-flag convention
in `scripts/train.py` rather than introducing a `--variant` string); (3)
an empirical action-manager check (build the env, read `env.
action_manager.get_term("joint_positions")`'s real class and `env.
action_manager.total_action_dim`) — do not trust the class definition
alone, this project's own standing "confirmed empirically, not just
asserted from the class hierarchy" discipline; (4) a new AR4-side
diagnostic script, `scripts/diag_ar4_antipodal_root_cause.py`, adapting
`scripts/diag_antipodal_root_cause.py`'s methodology (per-step contact-
force/reward/pose instrumentation, cross-checked against the exact
training-time `ar4_mdp.antipodal_grasp_bonus` function, not a
reimplementation) to AR4's own scene/entity names and both conditions;
(5) two real 3-seed/1500-iteration training runs (Condition A: unmodified
Experiment 26; Condition B: the new relative-joint leaf) with checkpoints
preserved and GCS-synced throughout via a raw `gsutil rsync` loop (AR4 has
no equivalent of `scripts/sync_run_to_gcs.py`, which is Franka-only
tooling keyed to `logs/train_franka*/` — see Design notes below); (6) the
5-checkpoint diagnostic sweep (30 rollouts: 5 checkpoints × 3 seeds × 2
conditions) plus the `cube_reached_goal` termination-rate behavioral bar
read directly from each run's own TensorBoard event file (reusing
`scripts/hillclimb_rewards.py`'s existing `extract_scalars` function, not
reimplementing it) plus targeted video review via the already-existing
`scripts/graspgoal_closeup_video.py` for any seed/condition showing a
real mechanism-level signal; (7) the explicit 3-signature jaw-mimic-
confound classification the spec requires for any null/partial result,
computed from the same diagnostic data (no new instrumentation); (8)
verdict — `ROADMAP.md` + a new dated section in
`kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md` (this plan's
chosen primary kb home — see Design notes below for the reasoning), plus
a small cross-link addition in
`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s own "Related"
section pointing at the new AR4 result rather than duplicating it there.

**Tech Stack:** Isaac Lab / Isaac Sim v2.3.1 / Isaac Sim 5.1.0, `rsl_rl`
PPO, `tasks/ar4/` (`pickplace_graspgoal_env_cfg.py` only — no new reward/
scene/pure-math module; the antipodal mechanism is already AR4-native and
already wired, per the spec's Design decision 2), `scripts/train.py`,
a new `scripts/diag_ar4_antipodal_root_cause.py`, GCP cloud dispatch
(`docs/cloud/dispatch-checklist.md`). **Execution backend: cloud ONLY for
this plan.** Desktop dispatch was checked live at this plan's authoring
time (2026-07-21) via `scripts/check_gpu_availability.sh` and returned
`TARGET=cloud` (exit 2, UNKNOWN — `curl` DNS resolution to `home.local`
timed out), per this project's own "never treat can't-tell as a green
light" rule. The task that commissioned this plan additionally stated the
desktop is unreachable/off-limits for this session — **do not re-check
desktop availability per task the way the Franka template plan does; every
task in this plan dispatches to cloud directly.** If a future execution of
this plan is picked up in a session where the desktop is confirmed
reachable again, that is a genuine judgment call for whoever executes it,
not something to silently assume from this document.

Spec: `docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`.
Templates/precedent for this plan's own structure, cost-cap methodology,
and dispatch discipline: `docs/superpowers/plans/2026-07-20-d8-relative-
joint-action-implementation.md` (the Franka version of this exact
experiment — direct structural template, and the source of the real,
**already-realized** critic-divergence contingency this plan budgets for,
not a hypothetical one — see Design notes below) and `scripts/
diag_antipodal_root_cause.py` (the diagnostic script this plan adapts,
not reuses directly, since AR4's task module, scene entity names, and
action space differ enough that a shared script would need per-platform
branching throughout rather than one clean `--variant` addition).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Design notes (flagged for controller review, not silently assumed)

**1. `sync_run_to_gcs.py` is Franka-only tooling and is NOT extended by
this plan.** Direct read confirms its `VARIANT_MAP`/`derive_variant()`
key off a `logs/train_franka*/` directory-name convention specific to
`scripts/train_franka.py`'s own log-root naming; AR4's `scripts/train.py`
writes to plain `logs/train/<timestamp>/` with no variant-tagged prefix,
so `derive_variant()` would fail immediately (`KeyError`-equivalent,
"Expected one of: [...]") on any AR4 run directory. Building AR4-side
variant-tagged GCS sync infrastructure is real, but out of scope for this
transfer-test plan — Tasks 4/5 instead use a plain, un-tagged `gsutil -m
rsync -r <run-dir> gs://rl-manipulation-hks-runs/ar4-franka-fixes-
transfer/<condition>/seed<K>/` loop in a detached `tmux` session, the same
raw mechanism `docs/cloud/dispatch-checklist.md`'s own "Known infra gaps"
section documents as the general mitigation, independent of
`sync_run_to_gcs.py`'s own Franka-specific variant-naming layer.

**2. The empirical action-manager check must inspect the live `ActionTerm`
object, not just a config field** — identical discipline to the Franka
template's own Task 1, for the identical reason (an `__post_init__`
override could run but be silently overwritten later in the MRO). Expected
values, derived directly from source rather than guessed: `tasks/ar4/
robot_cfg.py`'s `ARM_JOINT_NAMES` has exactly 6 entries
(`joint_1`..`joint_6`), and `tasks/ar4/actions.py`'s own comment
(`scripts/smoke_test_graspgoal_env.py:54`, "isaaclab's BinaryJointAction,
whose action_dim is hardcoded to [1]") confirms AR4's gripper action term
contributes exactly 1 dim regardless of its 2 underlying joint names — so
`env.action_manager.total_action_dim` must be **7** (6 arm + 1 gripper)
for BOTH Condition A and Condition B (the relative swap changes the arm
term's class, not its dimensionality). `env.action_manager.
get_term("joint_positions")`'s real Python type must be
`isaaclab.envs.mdp.actions.joint_actions.RelativeJointPositionAction`
(not `JointPositionAction`) for Condition B specifically.

**3. `scale=0.1`'s critic-divergence risk is a real, already-realized
precedent here, not a hypothetical one — budgeted into the cost cap, not
just flagged.** Two independent, concrete data points make this a
higher-probability risk for Condition B than the Franka template's own
framing suggested at spec-writing time: (a) AR4 itself already has a
direct precedent for exactly this failure mode under a *different* new
action term — `Ar4PickPlaceTaskspacePPORunnerCfg`
(`tasks/ar4/pickplace_taskspace_env_cfg.py:294-311`, `clip_actions=5.0`)
exists because Experiment 11's first full run under task-space/IK hit a
real, confirmed `Loss/value_function` divergence at iteration 67/1500;
(b) more directly, Franka's own H_relative arc — the *identical*
`RelativeJointPositionActionCfg(scale=0.1)` mechanism this plan transfers
— **actually hit this exact divergence** during its real 3-seed training
run (`tasks/franka/agents/rsl_rl_ppo_cfg.py`'s
`FrankaLiftRelativeJointPPORunnerCfg` docstring: seed 123 diverged,
`Mean value_function loss` going `181.0 → 2.2e8 → ... → inf` within 6 PPO
updates around iteration ~298, fixed with `clip_actions=5.0` and a full
3-seed re-run under the corrected runner cfg). **If Task 5's real run
shows the same signature, the fix is not a fresh guess**: add
`Ar4PickPlaceGraspGoalRelativePPORunnerCfg(Ar4PickPlacePPORunnerCfg)` to
`tasks/ar4/pickplace_graspgoal_env_cfg.py` (matching both
`Ar4PickPlaceTaskspacePPORunnerCfg`'s own file-placement convention — AR4
puts variant-specific runner cfgs in the env-cfg file itself, not
`agents/rsl_rl_ppo_cfg.py` the way Franka does — and Franka's own
`clip_actions=5.0` value directly, same ~3x-over-observed-action-noise-std
margin logic), scoped to Condition B only, and re-run all 3 seeds under it
(not just the diverged one, for a directly comparable clean set, mirroring
Franka's own real precedent exactly). **Do not pre-build this class before
Task 5's real run actually shows divergence** — per this plan's own Global
Constraints, matching the Franka template's identical rule.

**4. `scripts/diag_ar4_antipodal_root_cause.py`'s runner-cfg selection may
need a one-line update mid-plan, mirroring Franka's own real experience
exactly.** Task 3 builds this script assuming `Ar4PickPlacePPORunnerCfg`
for both conditions (today's actual state — no contingency class exists
yet). If Task 5's real run triggers the contingency in Design note 3
above, Task 6 (which runs after both training tasks complete) must select
`Ar4PickPlaceGraspGoalRelativePPORunnerCfg` for Condition B's rollouts
specifically (for the correct `clip_actions` at inference time) — a small,
expected edit to Task 3's own script, not a new scope item, flagged here
so it isn't mistaken for scope creep if it happens.

**5. AR4 has no equivalent of `franka_checkpoint_review.py`'s settle-
detection/sustained-lift behavioral protocol** — that script and its
`_detect_settle_step`/`max_height_gain` machinery are Franka-only. This
plan's own behavioral bar instead uses the `cube_reached_goal`
termination-rate scalar (`Episode_Termination/cube_reached_goal`),
already logged by every AR4 training run and already read by
`scripts/hillclimb_rewards.py`'s own `extract_scalars()` function (reused
directly, not reimplemented) over the final iterations of each run's own
TensorBoard event file, plus targeted video review via the
already-existing `scripts/graspgoal_closeup_video.py` (built for
Experiment 26 specifically) for any seed/condition whose mechanism-level
diagnostic data shows a real signal — not a new eval script.

**6. AR4's episode is 6x longer than Franka's** (30s/1500 steps vs.
Franka's 5s/250 steps, both at an identical 50Hz/0.02s control step) —
this directly inflates Task 6's per-rollout wall-clock/cost relative to a
naive "same shape, just AR4" assumption, and is factored explicitly into
the cost-cap derivation below rather than silently assumed free.

**7. kb write-back home: `ar4-vs-franka-root-cause-comparison.md` chosen
as primary, `d8-antipodal-grasp-quality.md` gets a cross-link only.** This
experiment is the direct empirical test of the three hypotheses that
concept doc already names as the pivot's own rationale — its own natural
continuation is a new dated section there, not a fork. `d8-antipodal-
grasp-quality.md`'s own H_relative section (the Franka result this plan
transfers) gets a short "AR4 transfer" cross-link addition in its
"Related" list, not a duplicated data table — avoiding writing the same
result twice, matching this project's own established "extend, don't
fork" discipline for continuation work.

## Global Constraints

- **Do not modify `Ar4PickPlaceGraspGoalEnvCfg`, `RewardsCfg`,
  `Ar4PickPlaceGraspGoalSceneCfg`, `TerminationsCfg`, or `EventCfg` in
  place.** Every new mechanism in this plan is additive: one new leaf
  env-cfg class (`Ar4PickPlaceGraspGoalRelativeEnvCfg`) extending
  Condition A, never editing it. Check `git status` fresh at the start of
  execution for any other concurrent workstream's untracked/in-progress
  files before touching anything (per `CLAUDE.md`'s "Claude's role" —
  multiple Seniors may be running in parallel across this repo).
- **`scale=0.1`, `use_zero_offset=True` are this plan's own starting
  values, not tunable mid-experiment.** Per the spec: precedent-grounded
  (Franka's own confirmed value under an identical 50Hz control step), not
  load-bearing for H_ar4_relative's own falsification bar, a Tier-2
  hillclimb candidate later if the mechanism validates but needs
  retuning.
- **No reward, scene, or PPO-runner-cfg changes pre-authorized beyond the
  one contingency in Design note 3.** `RewardsCfg`/
  `Ar4PickPlaceGraspGoalSceneCfg`/`Ar4PickPlacePPORunnerCfg`
  (`max_iterations=1500`, `gamma=0.98`, `save_interval=50`) reused
  unmodified — the action term is this experiment's only variable, per the
  spec's own "Global constraints." If a real critic-divergence signature
  appears (`Loss/value_function` exploding — judged a real, elevated-
  probability risk here per Design note 3, not a low-probability
  formality), apply the scoped `Ar4PickPlaceGraspGoalRelativePPORunnerCfg`
  fix described there and re-run all 3 seeds under it — do not build that
  class preemptively.
- **Checkpoints must be preserved AND GCS-synced throughout training, not
  only at the end.** `save_interval=50` already covers all 5 measurement
  checkpoints exactly (0, 100, 300, 700, 1499) with no schedule adjustment
  needed. Run a background `while true; do gsutil -m rsync -r <run-dir>
  gs://rl-manipulation-hks-runs/ar4-franka-fixes-transfer/<condition>/
  seed<K>/ ; sleep 300; done` loop in its own detached `tmux` session
  alongside each training run, started BEFORE training begins — per
  Design note 1, this plan does not use `sync_run_to_gcs.py`. A repeat of
  this project's own documented GRUB-corruption incident
  (`docs/cloud/dispatch-checklist.md`'s "Known infra gaps") must not cost
  this plan its own measurement data a second time.
- **Execution backend: cloud ONLY for every task in this plan** (see
  "Tech Stack" above for why desktop-first routing does not apply here).
  For every task that launches Isaac Sim:
  1. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim** — its
     blocking instruction, cost-cap paragraph, environment-conventions
     block, and bug-handling-discipline block. Do not paraphrase or
     reconstruct these from memory.
  2. `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1
     /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."` for
     every Isaac-Sim-touching invocation, even on a dedicated cloud
     instance running one job at a time — cheap insurance against a
     stray second process. Check `nvidia-smi --query-compute-apps=pid,
     used_memory --format=csv,noheader` (empty = clear) before dispatch.
  3. **Headless on cloud** — the standing, confirmed exception to this
     project's local "never headless" rule; this does not conflict with
     the desktop-only "non-headless, user wants to watch" convention
     since this plan never dispatches to desktop.
  4. **Full teardown after every cloud task**: `scripts/
     check_cloud_state.sh` shows zero instances/disks/snapshots before
     marking a task done.
  5. **Provision ONE instance per task-cluster where practical** (Tasks
     2+3's smoke tests may share one instance; Tasks 4+5's training runs
     may share one instance if convenient, avoiding a second ~15-20min
     Isaac Lab install window; Task 6's measurement sweep may reuse
     Task 5's own instance if still up) — implementer's judgment, not
     required, but re-verify availability/teardown state at each task's
     own boundary regardless of whether an instance is reused.
  6. **Other concurrent workstreams in this session may be holding this
     project's single cloud GPU quota** (`GPUS_ALL_REGIONS=1`
     project-wide). Each task below must check fresh and genuinely
     queue/wait/retry if busy — never assume availability just because an
     earlier task in this same plan found it available.
  7. **If a cloud interruption happens** (SPOT preemption, GRUB-corruption
     boot failure, pip-wheel-cache corruption from a preempted mid-
     install), diagnose via `gcloud compute operations list
     --filter="targetLink~<instance>"` before choosing a recovery path,
     per `docs/cloud/dispatch-checklist.md`'s "Known infra gaps" section —
     don't assume which failure mode occurred.
- **Cost cap: $10 cumulative across Tasks 1-6 combined, notify the
  controller if exceeded.** Derivation, real-number-anchored where
  possible, not a round-number guess:
  - Task 1 (add `--seed` flag): $0 — no Isaac Sim launch required, a
    `--help`-only smoke check suffices.
  - Task 2 (new leaf env cfg + empirical action-manager check + `--help`
    wiring check): ≈**$0.30** — comparable in scope to the spec's own
    Step 1 live smoke test, which cost a real, recorded **$0.14** for a
    fresh-instance asset build + 2-iteration training smoke test; this
    task's own action-manager check is smaller in scope (build the env,
    no training) but starts from the same fresh-instance asset-build
    overhead, so budgeted slightly above that recorded number for margin.
  - Task 3 (AR4 diagnostic script port + `--num_envs 256 --max_iterations
    20` smoke test + a smoke rollout of the new diagnostic script itself):
    ≈**$0.30** — same shape as the Franka template's own Task 2 estimate
    (**$0.25** there, for an equivalent smoke-test + diagnostic-script-
    smoke-test pairing), adjusted up slightly for AR4's own real,
    already-recorded asset-fidelity/build friction (Gaps 1-3 in the spec's
    Step 1) that a fresh cloud instance may re-encounter.
  - Task 4 (Condition A real training, 3 seeds/1500 iterations,
    `num_envs=4096`): ≈**$1.6** — directly reused, not re-derived, from
    the Franka template's own real recorded spend for the identically-
    shaped run (3 seeds/1500 iterations/`num_envs=4096`, same 50Hz
    control step, comparable single-arm/single-object scene complexity):
    `kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s own "Cost"
    section, **≈$1.6** for the equivalent absolute-joint-space condition.
    Training cost scales with `num_steps_per_env × max_iterations ×
    num_envs` (identical between AR4 and Franka: `num_steps_per_env=24`
    both platforms), not with episode length, so this number transfers
    directly despite AR4's 6x-longer episode (see Design note 6).
  - Task 5 (Condition B real training, 3 seeds/1500 iterations) +
    contingency reserve: baseline ≈**$1.6** (same reasoning as Task 4),
    **plus an explicit $1.6 contingency reserve for a full 3-seed re-run**
    under a `clip_actions=5.0` fix, since this exact mechanism (`Relative
    JointPositionActionCfg(scale=0.1)`) **already triggered this exact
    failure mode on Franka's own real run** (Design note 3) — this is
    budgeted as a probable cost driver, not a remote tail-risk margin.
    Task 5 subtotal: **≈$3.2**.
  - Task 6 (measurement: 30 diagnostic rollouts — 5 checkpoints × 3 seeds
    × 2 conditions — plus a handful of targeted `graspgoal_closeup_video.py`
    reviews, plus free TensorBoard-scalar reads from Tasks 4/5's own
    already-synced event files): ≈**$1.8**. Derivation: the Franka
    template's own equivalent measurement task (15 rollouts at 249-step
    episodes + 3 behavioral evals) cost a real, recorded **≈$0.25**
    (`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s "H_relative
    test" section, ≈40min instance-uptime). AR4's own episodes are 6x
    longer (1500 vs. 249 steps, Design note 6), and this task has 2x the
    rollout count (30 vs. 15, two conditions instead of one) — since
    install-window overhead is a fixed cost independent of rollout count/
    length while rollout compute itself scales with total env-steps
    processed, this task's own compute-bound portion is estimated at
    roughly `(30/15) × 6 ≈ 12x` Franka's own compute-bound share of that
    $0.25 (the bulk of which was actually fixed install overhead, not
    rollout compute, per that section's own "install + 15 diagnostic
    rollouts + 3 behavioral evals" framing) — netting to **≈$1.8** rather
    than a naive (and unrealistic) $3.0 literal 12x scale-up of the whole
    $0.25 figure.
  - Sum: 0 + 0.30 + 0.30 + 1.6 + 3.2 + 1.8 = **≈$7.2**, rounded up to
    **$10** for margin — a smaller proportional margin than the Franka
    template's own ~59% ($3.15→$5) specifically because this plan's own
    contingency reserve (Task 5) already prices in its single largest
    identified risk directly, rather than relying on the rounding margin
    alone to absorb it; the remaining ~39% headroom (`$7.2→$10`) covers
    one ordinary SPOT-preemption-retry cycle plus this platform's own
    documented environment-gap history (three real gaps already found and
    fixed in the spec's own Step 1; a fourth, narrower, non-blocking one —
    the missing `Link_5_Col.STL`/`Link_6_Col.STL` collision meshes — flagged
    but not expected to affect this specific experiment).
- **Real evidence over proxies at every eval**: the 5-checkpoint contact-
  frequency/antipodal-frequency/reward-term-mean data (re-derived from raw
  `.npz` arrays for at least one seed per condition, not summary-JSON-only
  trust) AND the `cube_reached_goal` termination-rate trajectory (read
  directly from TensorBoard event files) AND a reviewed close-up video for
  any positive result — not a shaped reward scalar or exit code alone
  (Experiment 16 precedent, reused by every prior spec in this arc).
- **Report the full per-seed 5-checkpoint trajectory, never collapsed to a
  single final number.** Per the spec's own falsification-bar design, a
  final-iteration-only number cannot distinguish "fixed" from "delayed
  collapse."
- **Report a SPLIT explicitly if the per-seed data doesn't cleanly match
  either the spec's CONFIRMED or FALSIFIED numeric bar** — this arc's own
  standing precedent for not forcing an ambiguous result into a binary
  verdict.
- **The three-way jaw-mimic-confound classification (spec's own hard
  requirement) must be reported for whichever condition(s) show a null or
  partial result, regardless of overall verdict** — a report that says
  only "FALSIFIED"/"CONFIRMED" without this breakdown does not satisfy
  the spec.
- **Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line** — each task
  below is executed by a freshly dispatched session; use that session's
  real ID, do not copy a fixed ID from this document.

---

## Task 1 — Add `--seed` to `scripts/train.py`

**Files:**
- Modify: `scripts/train.py` — add, mirroring `scripts/train_franka.py`'s
  existing pattern (its own `--seed` argument and the
  `if args_cli.seed is not None: agent_cfg.seed = args_cli.seed` /
  `env_cfg.seed = agent_cfg.seed` sequencing) exactly:
  - A new argparse argument, placed near the end of the flag list (after
    `--hierarchical`, before `AppLauncher.add_app_launcher_args(parser)`):
    ```python
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the PPO runner cfg's seed (for the 3-seed protocol used by this project's Tier 1 experiments; default: keep the agent cfg's own).",
    )
    ```
  - In `main()`, immediately after the existing `agent_cfg.max_iterations`
    override block (currently lines 347-348) and BEFORE the existing
    `env_cfg.seed = agent_cfg.seed` line (currently line 360):
    ```python
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    ```
    (`env_cfg.seed = agent_cfg.seed` already exists and needs no change —
    it already runs after this new block, picking up the override
    automatically, exactly as it does in `train_franka.py`.)

**Interfaces:**
- Consumes: nothing new.
- Produces: `--seed` CLI flag, consumed by Tasks 4/5's own real training
  dispatches (3 distinct seeds each).

- [ ] **Step 1: Add the `--seed` argument and the seed-override block**
  per the Files section above.
- [ ] **Step 2: Smoke-check** `python scripts/train.py --help` (no sim
  launch, no cloud/GPU needed) confirms the new flag parses and appears
  in help text; separately confirm (by direct code read, not a live run)
  that the override block is correctly positioned before
  `env_cfg.seed = agent_cfg.seed`.
- [ ] **Step 3: Commit.**

```bash
git add scripts/train.py
git commit -m "feat: add --seed flag to scripts/train.py (AR4 Franka-fixes-transfer plan, Task 1)"
git push origin main
```

---

## Task 2 — New relative-joint leaf env cfg (Condition B) + empirical action-manager verification

**Files:**
- Modify: `tasks/ar4/pickplace_graspgoal_env_cfg.py` — add, after
  `Ar4PickPlaceGraspGoalEnvCfg`'s own class body:
  ```python
  @configclass
  class Ar4PickPlaceGraspGoalRelativeEnvCfg(Ar4PickPlaceGraspGoalEnvCfg):
      """H_ar4_relative (Task 2, docs/superpowers/plans/2026-07-21-ar4-
      franka-fixes-transfer-implementation.md; spec: docs/superpowers/
      specs/2026-07-21-ar4-franka-fixes-transfer-design.md): IDENTICAL
      scene/rewards/observations/terminations/events/PPO recipe to
      Condition A (Ar4PickPlaceGraspGoalEnvCfg, Experiment 26) - the ONLY
      change is the arm action term, from Condition A's inherited
      ABSOLUTE JointPositionActionCfg (scale=0.5) to RELATIVE/delta
      RelativeJointPositionActionCfg (scale=0.1, use_zero_offset=True),
      mirroring Franka's own confirmed
      FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg
      (tasks/franka/dice_lift_joint_env_cfg.py) exactly - same scale/
      use_zero_offset values (AR4's control step is identical to
      Franka's own, 50Hz - see the design spec's own "why scale=0.1
      transfers without rescaling" section), same "call super, then
      re-assert the one changed field last" pattern.

      Empirical action-manager verification (Task 2, Step 3): [fill in
      exact observed output here after running the live check below -
      not asserted from the class hierarchy alone].
      """

      def __post_init__(self) -> None:
          super().__post_init__()
          self.actions.joint_positions = isaaclab_mdp.RelativeJointPositionActionCfg(
              asset_name="robot",
              joint_names=ARM_JOINT_NAMES,
              scale=0.1,
              use_zero_offset=True,
          )
  ```
  (`isaaclab_mdp` and `ARM_JOINT_NAMES` are already imported at the top of
  this file — no new imports needed.)
- Modify: `scripts/train.py` — add one new flat boolean flag,
  `--graspgoalrelative` (mirroring `--graspgoal`'s own structure exactly,
  placed immediately after it in the flag list), importing
  `Ar4PickPlaceGraspGoalRelativeEnvCfg` alongside the existing
  `Ar4PickPlaceGraspGoalEnvCfg` import, and adding one new `elif` branch
  in `main()`'s existing dispatch chain (immediately after the
  `args_cli.graspgoal` branch): `elif args_cli.graspgoalrelative:
  env_cfg_cls = Ar4PickPlaceGraspGoalRelativeEnvCfg`. Help text cites this
  plan and the design spec, states explicitly this is joint-space with a
  delta/relative target (not IK, not a rename of `--graspgoal`).
- Create: a small, throwaway verification script (or an inline snippet
  run via `isaaclab.sh -p -c` — implementer's judgment, doesn't need to
  be a permanent file) that builds an `Ar4PickPlaceGraspGoalRelativeEnvCfg`
  env at small `num_envs` (8-16 range) and asserts/prints:
  - `type(env.action_manager.get_term("joint_positions")).__name__ ==
    "RelativeJointPositionAction"` (NOT `JointPositionAction`).
  - `env.action_manager.total_action_dim == 7` (6 arm joints +
    1 gripper binary term — confirmed by direct source read, see Design
    note 2 above; NOT 8, unlike Franka's own equivalent check, since
    AR4's gripper action is a single binary command dim, not counted
    per-joint).
  - `env.action_manager.active_terms` includes both `joint_positions` and
    `gripper_position`, and the per-term dim listing shows
    `joint_positions` at dim 6 specifically (confirm the exact property
    name for per-term dims in the installed Isaac Lab v2.3.1 checkout).
  - Optionally: step the env a few times with a nonzero constant action on
    `joint_positions` and confirm the resulting joint-position delta does
    NOT depend on which pose the arm started at — implementer's judgment
    on how much of this extra check is worth the wall-clock.
- Record the exact observed output in
  `Ar4PickPlaceGraspGoalRelativeEnvCfg`'s own docstring (the `[fill in...]`
  placeholder above), matching this project's own established precedent
  for this exact kind of claim (`Ar4PickPlaceTaskspacePPORunnerCfg`'s own
  docstring, `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`'s own docstring).

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGoalEnvCfg`, `ARM_JOINT_NAMES` (read-only,
  nothing modified).
- Produces: `Ar4PickPlaceGraspGoalRelativeEnvCfg`, the new
  `--graspgoalrelative` flag on `scripts/train.py`, and the empirical
  confirmation that the action term/dimensionality actually swapped as
  designed — consumed by Task 3's smoke test and Tasks 4/5's real runs.

- [ ] **Step 1: Implement the new leaf env-cfg class** per the Files
  section.
- [ ] **Step 2: Wire `scripts/train.py`'s new `--graspgoalrelative`
  flag** per the Files section. Smoke-check with `--help` (no sim launch)
  to confirm the new choice parses.
- [ ] **Step 3: Empirical action-manager verification.** Dispatch to
  cloud per Global Constraints (`flock`-wrapped, headless, small
  `num_envs`). Confirm the type/dimensionality assertions above; record
  the exact output in the new class's own docstring.
- [ ] **Step 4: Verify GPU/instance clear** per Global Constraints
  (`scripts/check_cloud_state.sh`).
- [ ] **Step 5: Commit.**

```bash
git add tasks/ar4/pickplace_graspgoal_env_cfg.py scripts/train.py
git commit -m "feat: wire AR4 relative/delta joint-position action env cfg (H_ar4_relative, Task 2)"
git push origin main
```

---

## Task 3 — AR4 contact-force diagnostic port + bounded smoke test

**Files:**
- Create: `scripts/diag_ar4_antipodal_root_cause.py` — a new script
  adapting `scripts/diag_antipodal_root_cause.py`'s methodology to AR4
  (a new file, not a `--variant` extension of the Franka script — see
  "Tech Stack" above for why). Structure mirrors the Franka script
  closely:
  - `--variant {condition-a, condition-b}` (dispatches to
    `Ar4PickPlaceGraspGoalEnvCfg` / `Ar4PickPlaceGraspGoalRelativeEnvCfg`
    respectively — labeled "condition-a"/"condition-b" to match the
    spec's own naming, even though `scripts/train.py`'s own dispatch uses
    flat boolean flags; this is a deliberate, plan-level naming choice
    for the diagnostic script only, flagged here as a judgment call).
  - `--checkpoint`, `--num_envs` (default 64), `--num_steps` (default:
    one full episode, `max_episode_length - 1` ≈ 1499 for this scene —
    NOT truncated to save cost, despite AR4's episode being 6x longer
    than Franka's; the falsification bar's own trajectory-shape
    methodology depends on measuring real, full-episode contact-frequency
    behavior, not a partial window), `--output_npz`.
  - Loads `Ar4PickPlacePPORunnerCfg` for both variants initially (today's
    actual state — see Design note 4 for the expected mid-plan update if
    Task 5's contingency fires).
  - Per-step recorded arrays: both jaws' raw contact-force vectors
    (`gripper_jaw1_contact`/`gripper_jaw2_contact` `ContactSensorCfg`,
    `force_matrix_w.view(num_envs, 3)` — already confirmed correct shape
    by this scene's own existing reward/observation code), `magnitude_ok`/
    `antipodal_ok`/`cos_angle` (recomputed directly, then cross-checked via
    `assert` against `ar4_mdp.antipodal_grasp_bonus(env, FORCE_THRESHOLD,
    ANTIPODAL_COS_THRESHOLD, jaw1_contact_cfg, jaw2_contact_cfg)`'s own
    returned bool tensor — the exact training-time function, not a
    reimplementation, per Design decision 2 of the spec), the `ee_frame`
    FrameTransformer's `target_pos_w`/`target_quat_w`, the cube's
    `root_pos_w`, the policy's raw action tensor, and every
    `RewardsCfg` term's raw (pre-weight) value via direct function calls
    (`ar4_mdp.grasp_goal_milestone_bonus`, `mdp.action_rate_l2`,
    `mdp.joint_vel_l2`, `ar4_mdp.arm_ground_contact_penalty`,
    `ar4_mdp.slow_near_cube_bonus`) — the same direct-call pattern this
    repo's own `scripts/smoke_test_graspgoal_ground_penalty.py` already
    established.
  - `FORCE_THRESHOLD = 0.05`, `ANTIPODAL_COS_THRESHOLD = -0.7071` (AR4's
    own real, re-verified constants from `pickplace_graspgoal_env_cfg.py`
    — NOT Franka's `-0.894427`, a different `mu`).
  - Saves one `.npz` + summary JSON per (variant, checkpoint) invocation,
    identical schema to the Franka script (contact_frequency,
    antipodal_satisfying_frequency, fraction_of_contact_steps_that_are_
    antipodal, cos_angle stats, ee position/orientation step-diff stats,
    per-term reward means).

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGoalEnvCfg`,
  `Ar4PickPlaceGraspGoalRelativeEnvCfg` (Task 2), `ar4_mdp.
  antipodal_grasp_bonus` and the other `RewardsCfg` term functions
  (read-only).
- Produces: the new diagnostic script, smoke-tested against a real
  (if short) checkpoint — consumed by Task 6's real 5-checkpoint
  measurement sweep for both conditions.

- [ ] **Step 1: Implement `scripts/diag_ar4_antipodal_root_cause.py`**
  per the Files section.
- [ ] **Step 2: Bounded smoke test, ~20-iteration training run for
  Condition B** — confirm training doesn't crash and every reward term
  (including `grasp_goal_milestone_bonus`, `arm_ground_contact_penalty`,
  `slow_near_cube_bonus`) produces plausible (non-NaN, bounded-magnitude)
  values under the new action term:
  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py \
    --graspgoalrelative --num_envs 256 --max_iterations 20 --seed 42 --headless"
  ```
  Watch `Loss/value_function` in TensorBoard as an early, non-conclusive
  canary for critic instability (Design note 3 — a real, already-realized
  risk on the identical Franka mechanism; a clean 20-iteration run does
  NOT rule out divergence appearing later in the real 1500-iteration
  run). Watch for visibly excessive per-step action-magnitude jitter — if
  present, this is a real scope question for the controller (the spec
  treats `scale=0.1` as precedent-grounded and non-tunable mid-experiment,
  unlike the Franka template which had a documented `scale=0.0625`
  fallback; flag rather than silently substitute a different scale).
- [ ] **Step 3: Smoke-test the diagnostic script itself** against this
  smoke-test checkpoint (`--variant condition-b --checkpoint
  <this task's own model_N.pt> --num_envs 16`) — confirm it runs
  end-to-end, produces a `.npz` + summary JSON, and the in-script
  training-time-function cross-check assertion passes.
- [ ] **Step 4: Verify instance clear** per Global Constraints.
- [ ] **Step 5: Commit.**

```bash
git add scripts/diag_ar4_antipodal_root_cause.py
git commit -m "feat: add AR4 contact-force root-cause diagnostic (H_ar4_relative, Task 3)"
git push origin main
```

---

## Task 4 — Condition A: fresh rerun of unmodified Experiment 26, 3 seeds, 1500 iterations

**Files:** none new — runs the existing, unmodified
`Ar4PickPlaceGraspGoalEnvCfg` through a real, full 1500-iteration training
cycle, 3 seeds, under current infra (Isaac Lab has been updated since
Experiment 26 last trained — this is a genuine re-establishment of the
baseline, not assumed from the historical run).

**Interfaces:**
- Consumes: `--graspgoal` (already existing), `--seed` (Task 1).
- Produces: 3 full checkpoint sets (seeds 42/123/7), each with the 5
  measurement-point checkpoints (iterations 0/100/300/700/1499, all
  landing on `save_interval=50`'s own schedule) preserved and GCS-synced
  throughout — consumed by Task 6's measurement sweep, and this task's
  own final `cube_reached_goal` rate is itself a meaningful result (does
  the historical null still reproduce under today's stack at all).

- [ ] **Step 1**: Dispatch to cloud per Global Constraints. Record instance
  creation timestamp.
- [ ] **Step 2**: Start the background incremental GCS-sync loop (Global
  Constraints — a detached `tmux` session running `while true; do
  gsutil -m rsync -r <run-dir> gs://rl-manipulation-hks-runs/ar4-franka-
  fixes-transfer/condition-a/seed<K>/ ; sleep 300; done`) BEFORE starting
  training, so no early checkpoint is missed.
- [ ] **Step 3**: For each seed (42, 123, 7) — 3 runs total, from scratch:
  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py \
    --graspgoal --seed 42 --num_envs 4096 --max_iterations 1500 --headless"
  ```
  (Repeat for seeds 123 and 7.) **Watch `Loss/value_function` in
  TensorBoard throughout each run** — Condition A stays on AR4's own
  already-stable absolute joint-space actuation family (the family every
  prior AR4 joint-space experiment has trained stably under, per the
  design spec's own risk framing), so divergence is not expected here,
  but confirm rather than assume.
- [ ] **Step 4**: Confirm all 3 seeds' checkpoints at iterations
  {0, 100, 300, 700, 1499} exist, are non-corrupt (file-size sanity check
  — a real checkpoint for this project's PPO configs is ~1.27MB, per
  `docs/cloud/dispatch-checklist.md`'s own documented SPOT-truncation
  gotcha; verify this holds for AR4's own checkpoint size empirically
  rather than assuming it matches Franka's exactly, since the
  observation-space dimensionality differs), and are synced to GCS.
- [ ] **Step 5**: Full teardown; report elapsed cost against the $10 cap
  (cumulative across Tasks 1-6).
- [ ] **Step 6: Commit** any code fixes only (training runs themselves
  aren't committed) — none expected unless a real bug surfaces in Task 1's
  `--seed` wiring, in which case fix, re-verify, and commit separately
  before re-running the affected seed(s).

---

## Task 5 — Condition B: relative-joint leaf, 3 seeds, 1500 iterations

**Files:** none new, unless the critic-divergence contingency (Design
note 3) fires — in that case, modify:
- `tasks/ar4/pickplace_graspgoal_env_cfg.py` — add
  `Ar4PickPlaceGraspGoalRelativePPORunnerCfg(Ar4PickPlacePPORunnerCfg)`
  with `clip_actions = 5.0` (or an empirically-derived margin above the
  observed pre-divergence action-noise std, mirroring
  `Ar4PickPlaceTaskspacePPORunnerCfg`'s and Franka's own
  `FrankaLiftRelativeJointPPORunnerCfg`'s exact reasoning — see Design
  note 3 for the full precedent), scoped to Condition B only. Docstring
  must cite the real observed divergence signature (iteration, `Loss/
  value_function` trajectory) directly, matching this project's own
  established precedent for this exact kind of claim.
- `scripts/train.py` — add the corresponding runner-cfg selection branch
  (`if args_cli.graspgoalrelative: agent_cfg =
  Ar4PickPlaceGraspGoalRelativePPORunnerCfg() else: agent_cfg =
  Ar4PickPlacePPORunnerCfg()`, or equivalent given the existing
  taskspace-variant branch already present).
- `scripts/diag_ar4_antipodal_root_cause.py` (Task 3) — update the
  runner-cfg selection for `--variant condition-b` to match, per Design
  note 4.

**Interfaces:**
- Consumes: `--graspgoalrelative` (Task 2), `--seed` (Task 1).
- Produces: 3 full checkpoint sets (seeds 42/123/7) — consumed by Task 6.

- [ ] **Step 1**: Dispatch to cloud per Global Constraints (may reuse Task
  4's own instance if still up and convenient, per Global Constraints
  item 5, but re-verify availability/teardown state at this task's own
  boundary regardless).
- [ ] **Step 2**: Start the background incremental GCS-sync loop
  (`.../ar4-franka-fixes-transfer/condition-b/seed<K>/`) BEFORE starting
  training.
- [ ] **Step 3**: For each seed (42, 123, 7) — 3 runs total, from scratch:
  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py \
    --graspgoalrelative --seed 42 --num_envs 4096 --max_iterations 1500 --headless"
  ```
  **Watch `Loss/value_function` in TensorBoard throughout each run** — a
  real, already-realized risk on this exact mechanism (Design note 3), not
  a low-probability formality. If a genuine divergence signature appears
  (`Mean value_function loss` exploding toward `inf` over a handful of PPO
  updates, immediately followed by a NaN/Inf `std` crash — the exact
  signature Franka's own real run showed): stop the affected seed, apply
  the scoped `Ar4PickPlaceGraspGoalRelativePPORunnerCfg` fix described
  above (using the observed pre-divergence action-noise std to set
  `clip_actions` with the same ~3x margin Franka/AR4-taskspace both used,
  not a copy-pasted `5.0` without checking it fits this run's own
  observed std), and **re-run all 3 seeds** under the corrected runner
  cfg for a directly comparable clean set (mirroring Franka's own real
  precedent exactly — do not leave a mixed unclipped/clipped 3-seed set).
- [ ] **Step 4**: Confirm all 3 seeds' checkpoints at iterations
  {0, 100, 300, 700, 1499} exist, are non-corrupt, and are synced to GCS.
- [ ] **Step 5**: Full teardown; report elapsed cost against the $10 cap
  (cumulative across Tasks 1-6 — flag explicitly if the contingency fired
  and the reserve was consumed).
- [ ] **Step 6: Commit** the contingency fix (if triggered) and/or any
  other code fixes, separately from training runs:
  ```bash
  git add tasks/ar4/pickplace_graspgoal_env_cfg.py scripts/train.py scripts/diag_ar4_antipodal_root_cause.py
  git commit -m "fix: scoped clip_actions=5.0 for AR4 relative-joint condition after observed critic divergence (Task 5)"
  git push origin main
  ```
  (Only if the contingency actually fired — otherwise this step is a
  no-op for this task.)

---

## Task 6 — Measurement: 5-checkpoint diagnostic sweep + behavioral bar + 3-signature jaw-mimic classification, both conditions, all seeds

**Files:** none new — runs Task 3's diagnostic script and TensorBoard
scalar extraction against Tasks 4/5's real checkpoints and event files.

**Interfaces:**
- Consumes: Task 4's 15 checkpoints (Condition A, 5 per seed × 3 seeds)
  and Task 5's 15 checkpoints (Condition B) — 30 total.
- Produces: the per-seed, per-condition 5-checkpoint contact-frequency/
  antipodal-frequency/reward-term trajectory data, the
  `cube_reached_goal` termination-rate trajectory per run, the
  CONFIRMED/FALSIFIED/SPLIT classification per the spec's exact numeric
  bar, and — regardless of that classification — the explicit 3-signature
  jaw-mimic-confound classification for whichever condition(s) show a
  null/partial result.

- [ ] **Step 1**: Dispatch to cloud per Global Constraints (may reuse
  Task 5's own instance per item 5; re-verify availability/teardown state
  regardless).
- [ ] **Step 2**: For each condition (a, b), each seed (42, 123, 7), each
  checkpoint (iterations 0, 100, 300, 700, 1499) — 30 rollouts total:
  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/diag_ar4_antipodal_root_cause.py \
    --variant condition-a --checkpoint <path/to/seed42/model_0.pt> --num_envs 64 \
    --output_npz logs/diag_ar4/condition-a_seed42_iter0.npz --headless"
  ```
  (Repeat for all 30 combinations.)
- [ ] **Step 3**: For each (condition, seed), tabulate the 5-checkpoint
  `contact_frequency` trajectory alongside the antipodal-satisfying
  frequency and `grasp_goal_milestone_bonus`/other reward-term
  trajectories.
- [ ] **Step 4**: Read the `cube_reached_goal` termination-rate scalar
  (`Episode_Termination/cube_reached_goal`) over the final iterations for
  each of the 6 runs, reusing `scripts/hillclimb_rewards.py`'s existing
  `extract_scalars()` function directly rather than reimplementing
  TensorBoard event-file parsing.
- [ ] **Step 5**: Apply the spec's exact falsification/confirmation bar
  per seed, per condition (Condition A is expected, but not assumed, to
  reproduce the historical exact-zero pattern — report its own real
  trajectory regardless of expectation):
  - **FALSIFIED** if iter-1499 contact frequency `< 0.01` AND `< 50%` of
    that seed's own peak value across the 5 checkpoints, in ≥2 of 3
    seeds.
  - **CONFIRMED** if iter-1499 contact frequency `≥ 0.05` AND not less
    than that seed's own iter-700 value, in ≥2 of 3 seeds.
  - **Anything else is SPLIT.**
- [ ] **Step 6**: Explicit 3-signature jaw-mimic-confound classification
  (the spec's own hard requirement — applied to whichever condition(s)
  show a null or partial result, and reported even if H_ar4_relative is
  cleanly CONFIRMED, per the spec's "applied to both conditions" framing):
  1. Contact frequency near-zero in both conditions → exploration/
     action-space problem, not jaw-mimic-related.
  2. Contact frequency meaningfully nonzero but
     `fraction_of_contact_steps_that_are_antipodal` stays low → the
     signature directly consistent with the confirmed-broken jaw-mimic
     constraint (`docs/superpowers/specs/research/2026-07-20-ar4-vs-
     franka-root-cause-comparison.md` §2).
  3. Contact frequency nonzero AND mostly antipodal, but
     `cube_reached_goal` never rises → something else (e.g. the
     `slow_near_cube_bonus`/running-max-milestone confound already
     flagged in `pickplace_graspgoal_env_cfg.py`'s own docstrings, or a
     residual reach-grasp-lift gap) — not jaw-mimic, not exploration.
- [ ] **Step 7**: Re-derive the raw per-step `.npz` data for at least one
  seed per condition's full 5-checkpoint trajectory (not summary-JSON-only
  trust). Watch a close-up video (`scripts/graspgoal_closeup_video.py`)
  for any seed showing nonzero behavioral discovery before reporting it
  as a real positive.
- [ ] **Step 8**: Explicit comparison against (a) Franka's own H_relative
  trajectory (`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s own
  table: absolute joint-space exact `0.0` at all 5 checkpoints; relative
  joint-space seeds 42/123/7 rising to 0.88/0.83/0.89 by iter 1499) and
  (b) AR4's own historical Experiment 26 result
  (`kb/wiki/experiments/experiment-26-gripper-reintroduction.md`'s
  `cube_reached_goal 0.0000`) — the "did this actually change the shape
  of the curve, and does the historical AR4 null still reproduce under
  today's stack" questions this experiment exists to answer, from matched
  real data.
- [ ] **Step 9**: Full teardown; report elapsed cost against the $10 cap
  (cumulative across Tasks 1-6 — this is the final cost-bearing task,
  report the running total explicitly).
- [ ] **Step 10: Commit** any code fixes only, if Step 2 surfaces a bug in
  Task 3's diagnostic-script — fix, re-verify, commit separately.

---

## Task 7 — Verdict: `ROADMAP.md` + kb update

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry: the per-seed,
  per-condition 5-checkpoint contact-frequency trajectory (not collapsed
  to a final number), the overall CONFIRMED/FALSIFIED/SPLIT
  classification, the `cube_reached_goal` behavioral-bar numbers, the
  3-signature jaw-mimic classification for any null/partial result, the
  explicit curve-shape comparison against Franka's own H_relative result
  and AR4's own historical Experiment 26 null, and cost against the $10
  cap.
- Modify: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md` —
  extend with a new dated section (e.g. "H_ar4_relative transfer test
  (2026-07-2X follow-up)") — this plan's chosen primary kb home (Design
  note 7), reporting:
  - The full per-seed, per-condition 5-checkpoint trajectory table.
  - The CONFIRMED/FALSIFIED/SPLIT classification and what it means for
    this doc's own three named hypotheses (jaw-mimic, jaw-collision-
    geometry, classical-IK-miss) — does a locally-consistent action-to-
    motion mapping break AR4's own analogous "transiently discover, then
    abandon" pattern, or reproduce it in a reshaped form; does the
    3-signature classification implicate the jaw-mimic defect specifically
    or something else.
  - Cross-links: `kb/wiki/experiments/d8-antipodal-grasp-quality.md`
    (Franka's own H_relative result this test transfers — cross-link
    addition there too, per Design note 7, not a duplicated table),
    `kb/wiki/experiments/experiment-26-gripper-reintroduction.md` (the
    historical AR4 null this freshly reproduces or breaks),
    `kb/wiki/experiments/experiment-11-taskspace-ik.md` (AR4's own prior
    positive task-space result, the reason this plan did not retest that
    condition), `CLAUDE.md`'s North Star (a genuinely joint-space fix,
    confirmed on a *second* structurally different arm, is direct
    evidence the "drop in a new arm, train immediately" bar does not
    require an arm-specific IK layer — call this out explicitly if
    CONFIRMED).
  - The honest next candidate direction given the real result — ranging
    from "AR4's own grasp-discoverability problem may be substantially an
    action-space issue after all, not solely an asset defect — worth
    revisiting scope on AR4 vs. staying Franka-primary" (CONFIRMED) to
    "the fix does not transfer; AR4's null is not explained by the same
    mechanism Franka's was — the asset-level hypotheses (jaw-mimic,
    collision geometry) remain the more likely explanation, consistent
    with the Franka-pivot rationale" (FALSIFIED) to a SPLIT-specific
    honest read — do not default to a generic template answer regardless
    of which outcome actually occurred.

**Interfaces:**
- Consumes: Task 6's real per-seed, per-condition data.
- Produces: the closing, evidence-backed verdict for H_ar4_relative, plus
  a forward pointer for the next candidate direction.

- [ ] **Step 1**: Write the verdict against the spec's pre-registered
  falsification rule, per seed, per condition, the full 5-checkpoint
  trajectory — never collapsed to a final number.
- [ ] **Step 2**: Include the curve-shape comparison against Franka's own
  H_relative trajectory and AR4's own historical Experiment 26 null.
- [ ] **Step 3**: Include the `cube_reached_goal` behavioral-bar numbers
  and confirm close-up videos were actually watched, not just JSON
  trusted.
- [ ] **Step 4**: Include the explicit 3-signature jaw-mimic-confound
  classification, regardless of overall verdict.
- [ ] **Step 5**: Update `ROADMAP.md` and the kb article(s) per the Files
  section, per this repo's continuous-kb-update convention (not batched
  to session end).
- [ ] **Step 6**: State the honest next candidate direction given the
  real result.
- [ ] **Step 7**: Commit and push.

```bash
git add ROADMAP.md kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md kb/wiki/experiments/d8-antipodal-grasp-quality.md
git commit -m "verdict: AR4 Franka-fixes transfer, relative joint-position action (H_ar4_relative)"
git push origin main
```

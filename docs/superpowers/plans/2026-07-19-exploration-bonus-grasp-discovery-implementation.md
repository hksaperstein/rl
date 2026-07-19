# Exploration-bonus grasp discovery (H1: GRM D=1 attempt bonus) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test H1 (a GRM-`D=1`, action-dependent potential-based exploration
bonus for gripper-closure attempts near the object — `gripper_closure_attempt_bonus`
+ `gripper_closure_attempt_bonus_correction`) as a from-scratch-PPO fix for
d8's genuine, robust 0/24 grasp-discoverability null at the 48mm-parity
anchor (`FrankaDieLiftJointD8BigEnvCfg`, Task 3.5's own established
baseline). Two-tier falsification: **mechanism-level** (does the policy's
raw gripper action ever go negative near the object — currently a confirmed,
absolute `0.000` across 8 envs/a full episode on a structurally analogous
checkpoint) and **behavioral** (does sustained lift actually occur). A split
result — mechanism fires, lift still doesn't complete — is a genuine,
separately-reportable finding per the spec, not collapsed to a binary.

**Architecture:** (1) two new pure-tensor reward functions in a new
`tasks/franka/exploration_bonus_reward.py` module (TDD), including a design
note resolving two real ambiguities left open by the spec's literal
formula (see "Design notes" below — these are load-bearing correctness
decisions, flagged explicitly for the controller, not silently assumed);
(2) stateful integration — one stateless `mdp.py` function (term 1) + one
new `ManagerTermBase`-derived class (term 2, owning the one new persistent
per-env buffer this mechanism needs) — wired into a NEW env-cfg subclass
(`FrankaDieLiftJointD8BigExplorationBonusEnvCfg`, extending
`FrankaDieLiftJointD8BigEnvCfg` untouched, per the spec's own explicit
"Global constraints" requirement), plus `train_franka.py`/
`franka_checkpoint_review.py`/`sync_run_to_gcs.py` wiring and a bounded
smoke test; (3) a near-object-restricted mechanism-level diagnostic script,
generalizing `scripts/_diag_gripper_lowpass_check.py`; (4) the real
3-seed/1500-iteration training + eval run, both falsification bars computed
and reported per seed; (5) verdict + ROADMAP/kb update, with the split
"mechanism fired, grasp didn't complete" outcome reported explicitly if it
occurs, not rounded to a plain pass/fail.

**Tech Stack:** Isaac Lab / Isaac Sim, `rsl_rl` PPO, `tasks/franka/`
(`lift_env_cfg.py`, `dice_lift_joint_env_cfg.py`, `mdp.py`, a new
`exploration_bonus_reward.py`), desktop-first GPU dispatch
(`scripts/check_gpu_availability.sh`/`scripts/run_on_desktop_gpu.sh`), GCP
cloud fallback (`docs/cloud/dispatch-checklist.md`).

Spec: `docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-discovery-design.md`.
Research: `docs/superpowers/specs/research/2026-07-19-exploration-reward-expansion-literature.md`.
Template/precedent for this plan's own structure and rigor:
`docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-implementation.md`
(TDD-first-task-split, bounded-smoke-test-before-real-dispatch precedent,
desktop-first cost-cap reasoning) and
`docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md`
(new-env-cfg-subclass-not-shared-base-class precedent, per-shape/per-seed
"report exactly as observed" discipline).
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = a different Senior instance per task).

## Design notes (resolve two real ambiguities in the spec's literal formula — flagged for controller review, not silently assumed)

The spec's "Exact mechanism proposed" section gives `F'_t` (the fully
GRM-`D=1`-corrected per-step shaping reward) as a single 3-branch piecewise
formula, but names it as reward **term 2**'s own value while also saying
reward **term 1** (`F_t` itself) is a *separate*, simultaneously-added
`RewardsCfg` entry. Read completely literally, adding term 1 (`F_t`) AND
term 2 (`F'_t` verbatim) would **double-count `F_t`** at every step except
the terminal one (e.g. at `t=0`, `F_t + F'_t = F_0 + F_0 = 2F_0`) — which is
not what GRM's theorem describes and would silently reintroduce exactly the
class of "hand-derived non-Markovian potential with an incorrect informal
correctness argument" risk (Experiment 5) this spec was built to formally
avoid. This is judged, after working through the algebra below, to be an
artifact of the spec's prose (which describes `F'_t` as "the" GRM-corrected
total, not literally term 2's own isolated return value) rather than an
intended double-count — **but this is exactly the kind of subtle
reward-shaping-correctness call this project's own history (Experiment 5,
and `franka_checkpoint_review.py`'s two-bugs-in-a-row episode-boundary fix,
commit `977a748`, three days before this plan) shows is worth a second set
of eyes before Task 1 executes**, not a silent unilateral resolution.

**Resolution implemented by this plan:** term 1's `RewardsCfg` entry
returns `F_t` (the raw, action-dependent bonus, added every step,
unconditionally — this is `gripper_closure_attempt_bonus`, exactly as the
spec's own formula for it already specifies, no change there). Term 2's
`RewardsCfg` entry (`gripper_closure_attempt_bonus_correction`) does **not**
return `F'_t` verbatim — it returns `Correction_t := F'_t − F_t`, so that
`Term1 + Term2 == F'_t` (the spec's own literal formula) once both are
summed by the `RewardManager`, exactly as `RewardsCfg`'s weighted-sum
convention already works for every other term pair in this file. Solving
`Correction_t = F'_t - F_t` algebraically over the spec's own 3 branches:

```
Correction_t = 0                          if t = 0            (F'_0 - F_0 = 0)
             = -(1/γ) * F_{t-1}            if 1 <= t < N-1      (F'_t - F_t = -(1/γ)F_{t-1}, F_t cancels)
             = -(1/γ) * F_{t-1} - F_t      if t = N-1           (F'_{N-1} - F_{N-1} = -(1/γ)F_{N-2} - F_{N-1})
```

Note the middle and terminal branches are **the same formula
(`-(1/γ)*F_prev`) plus one extra `-F_t` term added only at the terminal
step** — i.e. "subtract last step's discounted raw bonus every step past
the first, and *additionally* cancel this step's own raw bonus if it's the
last step (since a bonus given on the very last step can never get its own
future matching step)." Task 1's unit tests must assert **both** (a) this
`Correction_t` formula directly, given synthetic `F_t`/`F_prev`/boundary
masks, **and** (b) the safety-net property `Term1_t + Term2_t == F'_t` for
a full synthetic multi-step episode against the spec's own literal 3-branch
formula, verbatim — (b) is the property that would actually catch a
double-counting regression regardless of which of the two of us has the
algebra right, and is the more important test of the two.

**Second ambiguity — `is_last_step` requires knowing termination in
advance, which the reward manager may not have computed yet.** The
`Correction_t` formula's terminal branch needs a per-env boolean "this is
my last step" *at reward-computation time*. `franka_checkpoint_review.py`'s
own `977a748` fix already found, empirically, that this env's step()
returns the **post-reset** observation on an episode's final step — meaning
some reset-related bookkeeping clearly happens within the same `step()`
call reward is computed in — but it does **not** by itself establish
whether the *reward manager* runs before or after that reset, or before or
after `episode_length_buf` itself is incremented/cleared. **This plan does
not assume an ordering — Task 2's own Step 1 is a mandatory, empirical
(not just source-read) confirmation of this exact timing before any
boundary-condition code is written**, mirroring the rigor
`franka_checkpoint_review.py`'s own fix used (a byte-for-byte empirical
check, not reasoning from source alone).

Independent of that ordering question, this plan avoids needing to know in
advance whether Isaac Lab's *own* termination manager has decided to
terminate this env this step, by having the correction term **independently
recompute** the same two termination predicates `TerminationsCfg` already
declares for this exact env cfg (`time_out` and `object_dropping`,
`lift_env_cfg.py:308-316`) using their own already-declared constants
(`minimum_height=-0.05`) — a self-contained, ordering-independent
computation, not a dependency on `env.termination_manager`'s own buffers
having already been populated this step. `is_last_step` is then
`(episode_length_buf + 1 >= max_episode_length) | (object.data.root_pos_w[:, 2] < -0.05)`
— reusing `TerminationsCfg`'s own `minimum_height` constant by reference,
not re-typing it — **with the exact `+1`/no-`+1` semantics on
`episode_length_buf` confirmed empirically in Task 2 Step 1 before this
line is written**, not assumed from this document.

## Global Constraints

- **Do not modify `tasks/franka/dice_lift_joint_env_cfg.py`'s existing
  `FrankaDieLiftJointD8BigEnvCfg` or `tasks/franka/lift_env_cfg.py`'s
  existing `RewardsCfg` in place.** Per the spec's own explicit "Global
  constraints" section: this plan's two new reward terms go on a NEW
  `RewardsCfg` subclass (`ExplorationBonusRewardsCfg(RewardsCfg)`, added to
  `lift_env_cfg.py` next to `TargetSelectionObservationsCfg`'s own
  established "new subclass, not a shared-base edit" precedent) wired into
  a NEW env-cfg subclass
  (`FrankaDieLiftJointD8BigExplorationBonusEnvCfg(FrankaDieLiftJointD8BigEnvCfg)`).
  This keeps the concurrently-running d8/d10 demo-warmstart plan's own use
  of the plain, unmodified `FrankaDieLiftJointD8BigEnvCfg` completely
  unaffected — do not touch any file that plan owns
  (`tasks/franka/distillation.py`, `tasks/franka/demo_action_mapping.py`,
  `scripts/extract_demo_trajectory.py`,
  `scripts/bc_pretrain_demo_warmstart.py`,
  `tests/test_bc_pretrain_demo_warmstart.py` — the latter two are currently
  untracked in `git status`; leave them alone).
- **`gamma` in both new reward terms MUST equal
  `FrankaLiftPPORunnerCfg.algorithm.gamma` exactly (`0.98`,
  `tasks/franka/agents/rsl_rl_ppo_cfg.py:50`), not an independently-chosen
  constant.** Theorem 1's own proof (the spec's "Research grounding"
  section) is a telescoping-sum identity over the *same* discount factor
  the agent's own returns/advantages are computed with — a mismatched
  `gamma` in the shaping term breaks the policy-invariance guarantee this
  whole mechanism exists to provide, silently. Hardcode `0.98` as a named
  constant sourced with a comment citing `rsl_rl_ppo_cfg.py:50`, not a
  bare literal.
- **`w_attempt`, `k`, `std_gate` are implementer-set starting values, not
  tuned in this plan** (spec's own "Global constraints": tuning these is a
  Tier 2 hillclimb candidate, not part of this experiment's falsification).
  Starting values: `w_attempt=1.0` (spec: "comparable to or smaller than
  `reaching_object`'s own weight, 1.0"), `k=1.0`, `std_gate=0.05` (spec:
  "matching `object_goal_tracking_fine_grained`'s existing fine-grained
  std, not `reaching_object`'s loose 0.1"). Define these three as one
  shared, named module-level constant (dict or three constants) referenced
  by **both** new `RewTerm`'s `params=` dicts — never copy-pasted twice —
  so term 1 and term 2's own redundant recomputation of `F_t` (term 2 must
  recompute `F_t` itself to compute `Correction_t`, see Design notes) can
  never silently diverge on these three values.
- **TDD discipline for the pure-Python piece** (Task 1's
  `exploration_bonus_reward.py`): write failing tests first, confirm the
  failure, implement, confirm green — matching this project's established
  `tasks/franka/lift_reward.py` / `tasks/franka/distractor_observations.py`
  precedent (pure `torch`-only math, no `isaaclab` import, testable via
  plain pytest). Run via
  `/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_exploration_bonus_reward.py -v -p no:launch_testing`
  (plain python3/pytest lacks `torch` in this environment) — or plain
  `python3 -m pytest` if a bare-torch environment is confirmed available;
  confirm which before running, do not assume.
- **Confirm Isaac Lab semantics by direct source read AND, where the
  question is about runtime ordering/timing (not just a static API
  contract), by an empirical check — not from memory.** This project's own
  immediately-prior precedent (`franka_checkpoint_review.py`'s `977a748`
  fix: a source-only read would not have caught the `height_history[249]
  == height_history[0]` off-by-one; it took an actual printed/compared
  array) is the standard this plan holds Task 2 Step 1 to. See "Design
  notes" above for exactly what must be confirmed: (a) `ManagerTermBase`
  class-based reward terms are supported by this installed Isaac Lab
  version and have their `.reset(env_ids)` called automatically on episode
  reset (needed for the spec's own "reset to 0 on episode reset" buffer
  requirement); (b) `episode_length_buf`'s value at reward-computation
  time, relative to its own per-step increment and to any same-step reset
  — confirmed empirically, e.g. by a small bounded script logging
  `episode_length_buf` from inside a custom reward term across a few real
  steps/one real episode boundary, not by reading source alone; (c)
  `env.action_manager.get_term("gripper_action").raw_actions`'s exact
  existence, shape, and timing at reward-computation time — the design
  spec asserts this is "confirmed exact API, reused verbatim from
  `scripts/_diag_gripper_lowpass_check.py`'s own already-working access
  pattern," but that diagnostic script actually reads the raw action from
  the **policy's own pre-`step()` output tensor** (`actions[:, -1]`), never
  from a `.raw_actions` attribute read post-`step()` — confirm this
  attribute genuinely exists, with the expected shape/value, before
  trusting the spec's claim uncritically.
- **Execution backend: desktop-first, cloud fallback** (2026-07-18 standing
  policy, CLAUDE.md's "Pi-as-primary-agent GPU dispatch"). **Other
  concurrent workstreams in this session (the d8/d10 demo-warmstart plan,
  the target-selection-clutter plan) may also be dispatching to the same
  desktop — each task below must re-check
  `scripts/check_gpu_availability.sh` fresh at its own dispatch time, never
  assume a backend decided earlier in this plan (or by a different
  workstream) still holds.** For every task that launches Isaac Sim:
  1. Check `scripts/check_gpu_availability.sh`. `TARGET=desktop` (exit 0)
     → dispatch via `scripts/run_on_desktop_gpu.sh` (default blocking mode,
     not `--detach`, for any task that needs a real result before the next
     step). `TARGET=cloud` (exit 1, BUSY) or unclear (exit 2, UNKNOWN) →
     fall back to `docs/cloud/dispatch-checklist.md`'s recipe. **Never
     treat "can't tell" as a green light for desktop** — UNKNOWN routes to
     cloud (or stop), never an assumed-available desktop.
  2. **Copy `docs/cloud/dispatch-checklist.md`'s blocks verbatim into any
     dispatch prompt that provisions cloud or launches Isaac Sim** (its
     blocking instruction, cost-cap paragraph, environment-conventions
     block, and bug-handling-discipline block).
  3. **`flock -o` is not automatic.** `run_on_desktop_gpu.sh` does not wrap
     the dispatched command in a lock itself — the command string shipped
     to the desktop (or run on a cloud instance) must itself be
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
- **Cost cap: notify the controller if cumulative cloud spend across Tasks
  2/3/4 combined exceeds $3.** This is a cloud-fallback safety backstop,
  not an expected spend — desktop dispatch is expected to bring actual
  cost to $0. If cloud is needed: this plan's dominant cost is Task 4's
  real dispatch, exactly **one** 3-seed/1500-iteration single-shape batch —
  the same scale `ROADMAP.md`'s d20-big-geom gate task entry measured at
  **≈$0.91** (2.39hr instance uptime, SPOT `g2-standard-4`+L4, zero
  preemptions). Doubled for a SPOT-preemption-retry buffer (this project's
  own recent repeated precedent of hitting SPOT preemptions on cloud
  training tasks) ≈ $1.82; Tasks 2/3's bounded smoke tests (a handful of
  iterations each, small `num_envs`, no full training budget) are short
  and cheap, generously $0.50 total. Sum ≈ $2.32, rounded up to **$3** for
  margin. This is smaller than the target-selection-clutter plan's $5 cap
  (3 sequential training stages sharing one instance's install overhead)
  and much smaller than the d8/d10 demo-warmstart plan's $10 cap (2 shapes
  × 3 seeds × up to 2 hypotheses) — this plan trains exactly one shape, one
  hypothesis, one 3-seed batch, the smallest-scope real training task in
  this project's recent history, so $3 is the proportionate number, not a
  stale copy-paste of a larger plan's cap.
- **Real evidence over proxies at every eval**: instrumented
  `max_height_gain`/`max_consecutive_lifted_steps` numbers AND
  `frac_steps_raw_action_negative_near_object` per seed AND a reviewed eval
  video (rest frame vs. peak-height frame) — not a shaped reward scalar or
  an exit code alone. Use `franka_checkpoint_review.py`'s current
  (post-`977a748`) settle-detection logic unchanged for the behavioral bar.
  Independently re-derive the raw per-step `.npy`/instrumented-rollout
  array for at least one seed for both bars, not summary-JSON-only trust,
  per this arc's own repeated settle-detection-bug discipline.
- **Report both falsification bars per seed, never averaged or collapsed.**
  Per the spec's own "Falsification bar"/"Success/failure reporting"
  sections: a split result (mechanism-level bar passes, behavioral bar
  still fails) is a genuinely novel, separately-reportable finding, not
  falsification of H1's core claim — Task 4 and Task 5 must both preserve
  and surface this distinction explicitly if it occurs, never silently
  round it to a binary pass/fail.
- Commit messages end with the *executing* session's own
  `Claude-Session: https://claude.ai/code/session_<ID>` line — each task
  below is executed by a freshly dispatched session; use that session's
  real ID, do not copy a fixed ID from this document.

---

## Task 1 — Pure reward math (TDD): `exploration_bonus_reward.py`

**Files:**
- Create: `tasks/franka/exploration_bonus_reward.py` — pure `torch`-only
  math, NO `isaaclab` import (mirrors `lift_reward.py`'s established
  split). Two functions:
  - `gripper_closure_attempt_bonus_raw(raw_gripper_action: torch.Tensor, cube_pos: torch.Tensor, ee_pos: torch.Tensor, w_attempt: float, k: float, std_gate: float) -> torch.Tensor`
    — `F_t = w_attempt * tanh(k * relu(-raw_gripper_action)) * (1 - tanh(||cube_pos - ee_pos|| / std_gate))`,
    matching `reaching_object_reward`'s own `(cube_pos, ee_pos, ...)`
    raw-position-input convention (not a pre-computed distance), shape
    `(N,)` in, `(N,)` out.
  - `gripper_closure_attempt_bonus_correction(F_t: torch.Tensor, F_prev: torch.Tensor, is_first_step: torch.Tensor, is_last_step: torch.Tensor, gamma: float) -> torch.Tensor`
    — `Correction_t` per the "Design notes" derivation above (**not**
    `F'_t` verbatim — see that section for why). `is_first_step`/
    `is_last_step` are `(N,)` bool tensors, caller-supplied (this function
    has zero knowledge of episode bookkeeping — pure arithmetic only,
    exactly the pattern this project's TDD discipline requires for
    something with no live-sim state to fake).
  Module docstring must state the Design notes' derivation explicitly
  (the `Term1 + Term2 == F'_t` identity, and why), matching this project's
  own documentation-rigor convention (e.g. `distractor_observations.py`'s
  module docstring citing its own source paper/equation).
- Test: `tests/test_exploration_bonus_reward.py` — new. Covers:
  - `gripper_closure_attempt_bonus_raw`: zero when `raw_gripper_action >=
    0` (no attempt) regardless of distance; positive and increasing in
    magnitude as `raw_gripper_action` becomes more negative, saturating
    (via `tanh`) rather than growing unboundedly for a very large negative
    input (assert a bound, e.g. `< w_attempt`); zero (or near-zero) far
    from the object even with a strongly negative action (gate actually
    gates); a known-value check at a specific `(action, distance)` pair
    against the formula computed by hand.
  - `gripper_closure_attempt_bonus_correction`: `Correction_t == 0` when
    `is_first_step` is `True` (regardless of `F_prev`'s value — this is
    the resolved episode-start boundary condition); the plain
    `-(1/γ)*F_prev` formula for a non-first, non-last step; the additional
    `-F_t` term present only when `is_last_step` is `True`; **explicitly
    assert `Correction_t` is NOT constrained to any particular sign**
    (construct a case where it comes out positive and one where it comes
    out negative) — a direct regression test against Experiment 5's own
    "always ≥ 0" failure class, matching the spec's own explicit
    non-claim.
  - **The safety-net test** (Design notes): construct a synthetic 4-step
    (or longer) episode's worth of `F_t` values, compute
    `Term1_t = F_t` and `Term2_t =` this function's own output at every
    step with correctly-threaded `F_prev`/boundary masks, and assert
    `Term1_t + Term2_t` equals the spec's own literal 3-branch `F'_t`
    formula, verbatim, at every step including both boundaries. This is
    the single most important test in this task — it is designed to catch
    a double-counting regression regardless of which side of the "Design
    notes" derivation is actually correct.

**Interfaces:**
- Consumes: nothing (pure math, no upstream dependency in this plan).
- Produces: both pure functions, consumed by Task 2's `mdp.py`
  wrappers/class.

- [ ] **Step 1: Write failing tests** per the Files section above, in
  `tests/test_exploration_bonus_reward.py`.
- [ ] **Step 2: Run tests, confirm they fail** (`ImportError`/`ModuleNotFoundError`).
- [ ] **Step 3: Implement `exploration_bonus_reward.py`** per the Files
  section, including the module docstring's explicit derivation.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit.**

```bash
git add tasks/franka/exploration_bonus_reward.py tests/test_exploration_bonus_reward.py
git commit -m "feat: pure GRM D=1 gripper-closure-attempt-bonus reward math (Task 1)"
git push origin main
```

---

## Task 2 — Stateful integration, new env cfg subclass, script wiring, bounded smoke test

**Files:**
- Modify: `tasks/franka/mdp.py` — add:
  - `gripper_closure_attempt_bonus(env, w_attempt, k, std_gate, object_cfg=SceneEntityCfg("object"), ee_frame_cfg=SceneEntityCfg("ee_frame"), action_term_name="gripper_action") -> torch.Tensor`
    — plain stateless function (term 1). Reads
    `env.action_manager.get_term(action_term_name).raw_actions` (shape/API
    confirmed per Step 1 below), `env.scene[object_cfg.name].data.root_pos_w`,
    `env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]` (same
    access pattern as `object_ee_distance`), delegates to
    `exploration_bonus_reward.gripper_closure_attempt_bonus_raw`.
  - `GripperClosureAttemptBonusCorrection` — a `ManagerTermBase`-derived
    class (term 2), the ONE new stateful mechanism this plan introduces.
    Owns `self._prev_raw = torch.zeros(env.num_envs, device=env.device)`
    (initialized in `__init__(self, cfg, env)`), implements
    `reset(self, env_ids=None)` (zeroes `self._prev_raw` for `env_ids`,
    matching the spec's own "reset to 0 on episode reset" requirement —
    confirm via Step 1 that this hook is actually invoked automatically on
    episode reset, do not assume), and `__call__(self, env, w_attempt, k,
    std_gate, gamma, object_cfg=..., ee_frame_cfg=..., action_term_name=...)`
    that: recomputes `F_t` via the same pure raw-bonus function (single
    source of truth, no duplicated formula), computes `is_first_step`/
    `is_last_step` per the "Design notes" derivation (using Step 1's
    confirmed `episode_length_buf` semantics), calls
    `exploration_bonus_reward.gripper_closure_attempt_bonus_correction`,
    updates `self._prev_raw = F_t` for the next call, and returns
    `Correction_t`. Module docstring addition must cite Step 1's exact
    confirmed findings (file/line for the `ManagerTermBase`/reset-hook
    confirmation, and the empirical `episode_length_buf` timing result),
    not assert them from memory.
- Modify: `tasks/franka/lift_env_cfg.py` — add
  `ExplorationBonusRewardsCfg(RewardsCfg)` (next to
  `TargetSelectionObservationsCfg`, same file, same "new subclass, base
  untouched" idiom) adding two new `RewTerm` fields:
  `gripper_closure_attempt_bonus = RewTerm(func=mdp.gripper_closure_attempt_bonus, params=_EXPLORATION_BONUS_PARAMS, weight=1.0)`
  and
  `gripper_closure_attempt_bonus_correction = RewTerm(func=mdp.GripperClosureAttemptBonusCorrection, params={**_EXPLORATION_BONUS_PARAMS, "gamma": _PPO_GAMMA}, weight=1.0)`,
  where `_EXPLORATION_BONUS_PARAMS = {"w_attempt": 1.0, "k": 1.0,
  "std_gate": 0.05}` and `_PPO_GAMMA = 0.98` are module-level constants
  defined once, each with a comment citing this plan's Global Constraints
  (`_PPO_GAMMA` citing `rsl_rl_ppo_cfg.py:50` by exact line), referenced by
  both `RewTerm`s (never copy-pasted).
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` — add
  `FrankaDieLiftJointD8BigExplorationBonusEnvCfg(FrankaDieLiftJointD8BigEnvCfg)`
  overriding only `rewards: ExplorationBonusRewardsCfg =
  ExplorationBonusRewardsCfg()` (everything else — scene/object
  scale/mass, observations, actions, events, terminations, PPO recipe —
  inherited byte-identical from `FrankaDieLiftJointD8BigEnvCfg`, per the
  spec's own "isolate one variable" design), plus
  `FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY` (same
  `num_envs=50`/`env_spacing=2.5`/`enable_corruption=False` pattern as
  every other `_PLAY` class in this file, e.g.
  `FrankaDieLiftJointD8BigEnvCfg_PLAY` immediately above it).
- Modify: `scripts/train_franka.py` — add `--variant
  joint-die-d8-big-exploration-bonus` (new `elif` branch importing
  `FrankaDieLiftJointD8BigExplorationBonusEnvCfg`, new `_log_suffix` entry
  `_jointdied8bigexplorationbonus`, new choices-list entry + help text
  citing this plan), mirroring the existing `joint-die-d8-big` branch
  exactly.
- Modify: `scripts/franka_checkpoint_review.py` — same new `--variant`
  choice, importing `FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY`,
  mirroring the existing `joint-die-d8-big` dispatch exactly (no new CLI
  flags needed — unlike the target-selection-clutter variants, this env
  cfg has no target-shape ambiguity to resolve).
- Modify: `scripts/sync_run_to_gcs.py` — add
  `"train_franka_jointdied8bigexplorationbonus": "joint-die-d8-big-exploration-bonus"`
  to `VARIANT_MAP`, matching `train_franka.py`'s new `_log_suffix` value
  exactly.

**Interfaces:**
- Consumes: Task 1's two pure functions.
- Produces: `mdp.gripper_closure_attempt_bonus`/
  `mdp.GripperClosureAttemptBonusCorrection`,
  `FrankaDieLiftJointD8BigExplorationBonusEnvCfg`(`_PLAY`), `--variant
  joint-die-d8-big-exploration-bonus` on all 3 scripts, and one smoke-test
  checkpoint — consumed by Task 3 (diagnostic-script smoke test) and Task
  4 (the real 3-seed dispatch).

- [ ] **Step 1: Mandatory direct-source-read + empirical confirmation**
  (Global Constraints; do this BEFORE writing any boundary-condition code)
  on the desktop (where Isaac Lab is installed), via a small bounded,
  non-training script (a handful of `num_envs`, no checkpoint/policy
  needed — random actions are sufficient, this is purely about manager
  timing):
  1. Confirm `isaaclab.managers`'s support for class-based
     (`ManagerTermBase`-derived) reward terms, and confirm the
     `RewardManager` calls `.reset(env_ids)` on such a term automatically
     when those envs reset (source read: cite file/line).
  2. Confirm `episode_length_buf`'s value at reward-computation time,
     empirically: build a tiny custom reward term that just prints
     `env.episode_length_buf` each step, run a couple of real episodes
     (letting at least one env reach both `time_out` and, if easy to
     trigger, `object_dropping`), and directly observe whether the value
     read during reward computation for a fresh episode's first control
     step is `0` or `1`, and whether the terminal step's own value is
     `max_episode_length - 1` or `max_episode_length`. Record the exact
     observed numbers, not an assumption.
  3. Confirm `env.action_manager.get_term("gripper_action").raw_actions`
     genuinely exists, its shape, and that its value at reward-computation
     time matches what `_diag_gripper_lowpass_check.py`'s own
     pre-`step()` `actions[:, -1]` capture would have recorded for the
     same step (a direct empirical cross-check, not just an attribute
     existence check).
  Record all three findings directly in `GripperClosureAttemptBonusCorrection`'s
  own module/class docstring, citing exact source file/line and the
  empirical numbers observed — per this project's "confirm by direct
  source read, not memory" discipline (Global Constraints).

- [ ] **Step 2: Implement `mdp.py`'s two additions** per the Files section,
  using Step 1's confirmed semantics (not this document's own draft
  assumptions).

- [ ] **Step 3: Implement `ExplorationBonusRewardsCfg` and
  `FrankaDieLiftJointD8BigExplorationBonusEnvCfg`(`_PLAY`)** per the Files
  section.

- [ ] **Step 4: Wire `train_franka.py`/`franka_checkpoint_review.py`/
  `sync_run_to_gcs.py`** per the Files section. Smoke-check with `--help`
  on all 3 scripts (no sim launch) to confirm the new choices parse.

- [ ] **Step 5: Bounded smoke test — confirm the new reward terms don't
  crash training and produce plausible (non-NaN, non-exploding) values.**
  `scripts/check_gpu_availability.sh` → dispatch per Global Constraints.
  `flock`-wrapped, non-headless if desktop:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-exploration-bonus --num_envs 256 --max_iterations 20 --seed 42"
  ```

  Confirm in the training log/TensorBoard: both new reward-term components
  appear with non-NaN, bounded-magnitude values (roughly `[-1, 1]`-scale
  given `w_attempt=1.0`, not blowing up or silently zero for the entire
  run — a silently-always-zero term would mean the near-object gate or the
  action-sign check is wired backwards); confirm at least one env's raw
  gripper action goes negative at least once during this short smoke run
  (sanity that the action-sign read is correct, independent of whether it
  happens *near the object* — that's Task 3's job to check specifically).
  Keep this checkpoint (`model_19.pt` or similar) — Task 3 reuses it for
  its own smoke test, avoiding a second GPU dispatch.

- [ ] **Step 6: Verify GPU clear** per Global Constraints (desktop:
  `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh`; cloud: full teardown + `check_cloud_state.sh`).

- [ ] **Step 7: Commit.**

```bash
git add tasks/franka/mdp.py tasks/franka/lift_env_cfg.py tasks/franka/dice_lift_joint_env_cfg.py \
        scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/sync_run_to_gcs.py
git commit -m "feat: wire GRM D=1 exploration-bonus reward terms into a new d8-big env-cfg subclass (Task 2)"
git push origin main
```

(The smoke-test checkpoint under `logs/` is a model artifact — not
committed, per this repo's `models`/`data`/`logs` gitignore convention.)

---

## Task 3 — Instrumentation: near-object-restricted mechanism-level diagnostic script

**Files:**
- Create: `scripts/_diag_gripper_closure_near_object_check.py` — new
  sibling of `scripts/_diag_gripper_lowpass_check.py` (never modifies that
  file in place), generalizing its rollout/instrumentation methodology per
  the spec's own "Falsification bar" §1: same per-step capture of raw
  gripper action (`actions[:, -1]`, pre-`step()`, matching the existing
  diagnostic's own proven access pattern) and object height, **plus** a
  new per-step capture of end-effector-to-object distance
  (`env.scene["ee_frame"].data.target_pos_w[..., 0, :]` vs.
  `env.scene["object"].data.root_pos_w[:, :3]`, the same access pattern
  `mdp.object_ee_distance` already uses). Computes, per env:
  `frac_steps_raw_action_negative_near_object` = fraction of steps where
  `raw_action < 0 AND distance < 0.05` (5cm — matching this design's own
  `std_gate`, per the spec), **restricted to the subset of steps where
  `distance < 0.05` at all**. Takes `--variant
  joint-die-d8-big-exploration-bonus` (Task 2's new `_PLAY` class) and
  `--checkpoint` CLI args, `--num_envs 8`/one full 250-step episode,
  deterministic inference policy — identical protocol to the existing
  diagnostic, generalized only in what's measured.

  **Explicit edge case (state this in the script and its output, do not
  silently default to 0.0):** if an env never gets within 5cm of the
  object during the episode (`near_object_mask.sum() == 0` for that env),
  report `frac_steps_raw_action_negative_near_object: null` for that env
  in the summary JSON, with a printed note — this is a categorically
  different finding ("never got close enough to test the mechanism at
  all") from a real, meaningful `0.000` ("got close, never attempted"),
  and conflating them would understate or overstate H1's mechanism-level
  result depending on which envs happen to fall into which bucket.

**Interfaces:**
- Consumes: Task 2's `FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY`
  and its smoke-test checkpoint (for this task's own smoke test).
- Produces: the mechanism-level falsification-bar measurement tool,
  consumed by Task 4's real eval.

- [ ] **Step 1: Write the script** per the Files section above.

- [ ] **Step 2: Smoke-test against Task 2's smoke-test checkpoint** (no
  new GPU dispatch — reuse that checkpoint), `scripts/check_gpu_availability.sh`
  → dispatch per Global Constraints if the checkpoint isn't already
  reachable from a live desktop session, `flock`-wrapped, non-headless if
  desktop:

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_gripper_closure_near_object_check.py \
    --variant joint-die-d8-big-exploration-bonus --checkpoint <Task 2's smoke-test checkpoint> --num_envs 8"
  ```

  Confirm the script runs end-to-end, writes a summary JSON with a
  well-formed `per_env` block (including at least one `null` or one real
  numeric value — whichever the smoke-test checkpoint's own behavior
  produces, either is an acceptable smoke-test outcome; the check here is
  "does the script run and produce sane, correctly-typed output," not "is
  the smoke-test checkpoint itself a real experimental result" — it's a
  20-iteration throwaway).

- [ ] **Step 3: Verify GPU clear**, per Global Constraints.

- [ ] **Step 4: Commit.**

```bash
git add scripts/_diag_gripper_closure_near_object_check.py
git commit -m "feat: near-object-restricted gripper-closure mechanism diagnostic (Task 3)"
git push origin main
```

---

## Task 4 — Real H1 run: PPO training + both falsification bars, d8, 3 seeds

**Files:** none new — runs Tasks 1-3's code through a real, full
1500-iteration training + eval cycle.

**Interfaces:**
- Consumes: `--variant joint-die-d8-big-exploration-bonus` (Task 2),
  `scripts/_diag_gripper_closure_near_object_check.py` (Task 3),
  `franka_checkpoint_review.py`'s existing behavioral-bar protocol.
- Produces: 3 full checkpoints (seeds 42/123/7) + both falsification bars'
  numbers per seed — this experiment's real evidence.

- [ ] **Step 1**: `scripts/check_gpu_availability.sh` → dispatch per
  Global Constraints (re-check fresh, do not reuse Task 2/3's dispatch
  decision). Record dispatch target and (if cloud) instance creation
  timestamp.

- [ ] **Step 2**: For each seed (42, 123, 7) — 3 runs total, from scratch
  (no `--checkpoint`, matching the existing 0/24 baseline's own recipe
  exactly so this is a clean, single-variable comparison):

  ```bash
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py \
    --variant joint-die-d8-big-exploration-bonus --seed 42 --num_envs 4096 --max_iterations 1500"
  ```

  (Repeat for seeds 123 and 7.)

- [ ] **Step 3: Mechanism-level bar, per seed** — run Task 3's
  `scripts/_diag_gripper_closure_near_object_check.py --variant
  joint-die-d8-big-exploration-bonus --checkpoint <seed's model_1499.pt>
  --num_envs 8` (one full 250-step episode, deterministic policy).
  Report `frac_steps_raw_action_negative_near_object` per env per seed
  explicitly (including any `null`s, per Task 3's edge-case handling) —
  per the spec, **H1's mechanism-level claim is falsified only if this
  fraction is exactly `0.000` in all 3 seeds** (treat a seed where every
  env reports `null` — never got near the object at all — as a
  **separate, explicitly-flagged** outcome, distinct from a confirmed
  `0.000`; do not silently fold "never got close" into "got close, never
  attempted" when reporting).

- [ ] **Step 4: Behavioral bar, per seed** — `franka_checkpoint_review.py
  --variant joint-die-d8-big-exploration-bonus --checkpoint <seed's
  model_1499.pt> --num_envs 8`, current post-`977a748` settle logic,
  video + `max_height_gain`/`max_consecutive_lifted_steps`. Per the spec:
  **H1 is falsified for d8 if all 3 seeds show 0/8** (0/24 total); any
  nonzero count in even one seed is a real positive signal, per this
  project's own "no spurious partial count has ever been observed"
  precedent.

- [ ] **Step 5: Re-derive the raw per-step `.npy`/instrumented-rollout
  array for at least one seed for BOTH bars** (not summary-JSON-only
  trust), per this arc's own repeated settle-detection-bug discipline.
  Watch the eval video for any seed showing nonzero behavioral discovery
  (rest frame vs. peak-height frame, genuinely gripped arm pose, not just
  a height number crossing a threshold).

- [ ] **Step 6: Report the overall verdict per the spec's own rule** —
  falsified only if BOTH bars fail across all 3 seeds; explicitly call out
  a split result (mechanism-level bar passes in ≥1 seed, behavioral bar
  still 0/24) as its own distinct, reportable outcome if it occurs, per
  Global Constraints and the spec's "Falsification bar" section — this is
  the single most important reporting requirement in this task, do not
  compress it to a plain pass/fail in the task's own output.

- [ ] **Step 7**: Full teardown if cloud-dispatched; report elapsed cost
  against the $3 cap (cumulative across Tasks 2/3/4). Desktop: verify
  `nvidia-smi`/`tmux ls`/`systemd-inhibit --list`/
  `check_gpu_availability.sh` all clear/AVAILABLE.

- [ ] **Step 8: Commit** any code changes only (training runs themselves
  aren't committed) — none expected unless Step 2's real dispatch surfaces
  a bug in Tasks 1-3's code, in which case fix it, re-verify those tasks'
  own unit tests still pass, and commit the fix separately with its own
  message before re-running the affected seed(s).

---

## Task 5 — Verdict, ROADMAP.md + kb update

**Files:**
- Modify: `ROADMAP.md` — append the verdict entry: per-seed mechanism-level
  numbers, per-seed behavioral numbers, overall verdict per the spec's own
  rule, and cost against the $3 cap.
- Create or modify: a `kb/wiki/experiments/` article — likely a new
  `kb/wiki/experiments/exploration-bonus-grasp-discovery.md` (a genuinely
  new mechanism — GRM/PBRS-based shaping — distinct from both the prior
  specialist/distillation arc and the demo-warmstart arc it runs alongside;
  check this repo's kb-maintenance convention before choosing, matching
  how `dice-pick-demo.md`/the multi-die-specialist article cross-link
  today), cross-linking `kb/wiki/concepts/reach-grasp-lift-gap.md`,
  `kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md`, and
  `kb/wiki/experiments/experiment-05-potential-based-reward-shaping.md`
  (the prior-failure this mechanism was built to formally avoid repeating —
  record explicitly whether it succeeded in doing so, per this task's own
  real evidence).

**Interfaces:**
- Consumes: Task 4's real per-seed results (both bars).
- Produces: the closing, evidence-backed verdict for this experiment.

- [ ] **Step 1**: Write the verdict against the spec's own pre-registered
  falsification rule, per seed, both bars — explicitly one of: (a) H1
  falsified (both bars fail, all 3 seeds), (b) H1 confirmed (both bars
  pass in ≥1 seed), or (c) **the split outcome** (mechanism-level bar
  passes in ≥1 seed, behavioral bar still 0/24) — reported as its own
  labeled category, not folded into (a) or (b), with an explicit
  discussion of what it would mean (per the spec: "the exploration
  mechanism worked... but some other downstream grasp-mechanics problem
  remains, distinct from and pointing away from the pure discoverability
  question this spec targets") and what the honest next candidate
  direction would be (per CLAUDE.md's "Claude's role": don't just record a
  failure/partial without a forward pointer).
- [ ] **Step 2**: Include the instrumented `max_height_gain`/
  `max_consecutive_lifted_steps` numbers (not just the summary discovery
  fraction) and confirmation the eval video(s) were actually watched, not
  just the JSON trusted, matching this arc's own established rigor.
- [ ] **Step 3**: Update `ROADMAP.md` and the kb article(s) per the Files
  section, following this repo's continuous-kb-update convention (not
  batched to session end).
- [ ] **Step 4**: Commit and push.

```bash
git add ROADMAP.md kb/wiki/experiments/
git commit -m "verdict: GRM D=1 exploration-bonus grasp discovery experiment (d8, H1)"
git push origin main
```

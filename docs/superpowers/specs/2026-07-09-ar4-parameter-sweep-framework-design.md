# Parameter-sweep framework: general Tier-2 hyperparameter/reward iteration

## Context

Direct user direction (2026-07-09): build a reusable framework for iterating
over hyperparameters and reward-function parameters in a consistent,
repeatable way across RL training experiments, with methods for collecting
data on what has been tried (coverage of the parameter space) and how it
relates to arriving at a working solution — usable continuously across
future experiments, not a one-off script for one task.

This is the generalized **Tier 2** tool (see `CLAUDE.md`'s Workflow
section). It tunes weights/thresholds and training hyperparameters *within
an already-validated mechanism*; it does **not** add reward terms, action
spaces, or mechanisms (that is Tier 1, a full hypothesis-gated spec/plan/SDD
process). It supersedes `scripts/hillclimb_rewards.py`, which was narrow:
one task, one file, six hardcoded reward literals, single-parameter greedy
only, results in a non-queryable markdown table.

## What it preserves from the existing hillclimb design

Non-negotiable disciplines carried forward verbatim (see
`2026-07-07-ar4-hillclimb-loop-design.md` and the kb concepts
`reward-hacking-and-sparse-discoverability`, `reward-rate-arithmetic`,
`staged-reward-co-satisfiability`):

- **Real success-termination rate as ground truth for every trial**, never a
  shaped/aggregate reward scalar. This project has repeatedly been burned by
  a shaped scalar improving while real behavior did not (Experiment 15 most
  sharply). Each task declares its own success tag (e.g. touchgoal ->
  `Episode_Termination/goal_reached`, graspgoal/baseproximity ->
  `Episode_Termination/cube_reached_goal`).
- **Automatic instability rejection**: a value-function-loss divergence check
  is a hard reject independent of the success metric (see "Instability gate"
  below).
- **Bounded, cheap proxy runs**: the fast loop defaults to the existing
  diagnostic scale (`num_envs=4096`, `max_iterations=300`, ~3-5 min). A
  sweep never silently balloons into a full 1500-iteration run.
- **Never `git push` from automation** — batch review and push stay a
  Principal decision. Extended here to **never edit or commit source per
  trial either** (see the decision below).
- **Every attempt logged regardless of outcome** — no silent discards.

## The central architectural decision: config-override, not source-edit

The old script's model was: edit a numeric literal in an env-cfg source
file, git-commit if the proxy improved, `git checkout --` if not. That is
appropriate when the only goal is to permanently bake one chosen value into
one file. It does not scale to a general framework meant to run many trials
across many parameters and files.

**Decision: trials apply config overrides at construction time; they do not
mutate or commit source files.**

`scripts/train.py` gained one additive, backward-compatible flag,
`--overrides_file <path.json>`: a flat dict of dotted-path keys -> JSON
scalar values, applied to the constructed `env_cfg`/`agent_cfg` *after*
construction and *before* the env/runner are built (so the dumped
`params/env.yaml`/`agent.yaml` reflect the actually-trained config).
Namespaces: `env.` (env cfg) and `agent.` (PPO runner cfg). Each path
segment resolves as an attribute, or as a dict key when the current object
is a dict — so reward-term `params` entries (`...params.touch_std`) are
reachable by the same syntax as configclass fields
(`...rewards.<term>.weight`, `actions.joint_positions.scale`,
`episode_length_s`, `algorithm.learning_rate`, ...). The resolver lives in
`sweeps/overrides.py` (pure python, no isaaclab import) and is shared by
train.py and unit-tested standalone.

Why config-override over source-edit:

1. **General across files and shared cfgs.** The env cfg and the PPO cfg
   (`tasks/ar4/agents/rsl_rl_ppo_cfg.py`, shared by nearly every task) are
   patched uniformly by dotted path. Editing the shared PPO file per trial
   would be actively dangerous (it would change every other task's config);
   the old script's literal-regex approach also had a real
   two-blocks-share-`weight=0.1,` ambiguity problem that vanishes entirely
   here.
2. **No source churn, no per-trial commits.** A 50-trial random sweep
   produces zero commits and zero source edits. Trial identity is captured
   reproducibly by a `config_hash` = sha256(baseline git sha + canonical
   override dict), so any recorded trial can be re-created exactly.
3. **Multi-parameter vectors are trivial.** Applying N overrides at once is
   the same code path as applying one — required for random/grid search,
   which the single-file-edit greedy model does not naturally support.
4. **Cleaner exploration/promotion split.** Exploration is ephemeral
   (overrides + a data store). Promoting a winning value into source stays a
   deliberate human/Principal git action after batch review — the same
   philosophy as "no push from automation," extended to "no source mutation
   from automation."

**Cost / boundary (documented honestly).** Only parameters that are *live
cfg fields at train time* are sweepable this way. A value hardcoded as a
bare module constant used directly inside a reward-function body (not passed
through a `params` dict) cannot be overridden without a source edit. In
practice the catalog's tunables are nearly all cfg fields — reward `weight`s,
reward `params` entries (the value is copied into the params dict even when
its source is a module constant, e.g. `touch_std`), action `scale`,
`episode_length_s`, and every PPO field. A future source-edit adapter could
be added for genuinely baked constants if one ever needs sweeping; it is not
needed for anything currently in scope.

## The other central decision: SQLite store, not a markdown table

The user explicitly wants "methods for collecting data on what's been tried
(coverage) and how it relates to arriving at a working solution" — queryable
coverage-vs-outcome analysis, not a file read top to bottom.

**Decision: a SQLite store (`logs/sweeps/sweeps.db`), stdlib `sqlite3`, no
new runtime dependency.** Two tables: `trials` (one row per trial: full
parameter vector as JSON, real success metric, stability metric, outcome,
config hash, baseline git sha, num_envs/max_iterations scale, timestamp,
run_dir/log paths) and `trial_params` (normalized `(trial_id, param_name,
value)` rows so per-parameter coverage queries are a trivial join). A
markdown table cannot be queried; a flat JSONL/CSV can but forces
re-implementing every group-by/join in Python. SQLite gives indexed queries,
joins, safe concurrent reads while a sweep writes, and a single portable
file — for zero dependency cost.

## Components

- `sweeps/spaces.py` — declarative `TaskSpace`/`ParameterSpec` and the
  `TASK_SPACES` registry (touchgoal, graspgoal, baseproximity today). Adding
  a task = add one entry; no execution-code change. A parameter maps to a
  *list* of override keys (one logical parameter can patch several duplicated
  fields together).
- `sweeps/store.py` — `TrialStore` / `TrialRecord` (the SQLite store above).
- `sweeps/strategies.py` — `HillclimbStrategy` (single-parameter greedy
  coordinate ascent, the old behavior generalized), `RandomStrategy`
  (multi-parameter random search over bounds, linear or log scale),
  `GridStrategy` (cartesian grid over a small axis set). One driver interface
  (`next_trial()` / `record()`) so all three share the store and runner.
- `sweeps/runner.py` — `TrialRunner`: launches `isaaclab.sh -p
  scripts/train.py <flag> --overrides_file ... --headless` with
  `PYTHONUNBUFFERED=1`, blocks until the checkpoint lands (ported from the
  proven hillclimb poll loop), and extracts the success + stability scalars
  via `sweeps/_extract_scalars.py` (a real script file, not an inline `-c`
  snippet — START_HERE.md warns inline snippets have hung reproducibly).
- `sweeps/overrides.py` — the shared, pure-python, unit-tested resolver.
- `scripts/sweep.py` — orchestrator CLI (plain `python3`).
- `scripts/sweep_report.py` — coverage/outcome report CLI (plain `python3`).

## Instability gate

`runner.py` hard-rejects a trial as `UNSTABLE` (never adopted, regardless of
success) when the run's max value-function loss is NaN/inf or exceeds
`--vf_max_reject` (default `1000.0`). The default is deliberately
conservative: this repo has a *documented benign* transient spike of ~17.66
(Experiment 15) that must not be discarded, while a genuine divergence runs
into the hundreds/thousands or NaN. This satisfies the hard "automatic
instability reject" requirement without discarding benign transients — the
exact judgment the original hillclimb spec had left to a human. Principal can
tighten it per batch.

## Invocation

```bash
# single-parameter greedy hillclimb, 15 rounds (round 0 = baseline)
python3 scripts/sweep.py --task touchgoal --strategy hillclimb --rounds 15

# random search: 20 trials over the whole space (multi-dim coverage)
python3 scripts/sweep.py --task touchgoal --strategy random --trials 20

# random search over a chosen parameter subset
python3 scripts/sweep.py --task touchgoal --strategy random --trials 20 \
    --params learning_rate entropy_coef action_scale

# grid over a small axis set
python3 scripts/sweep.py --task graspgoal --strategy grid \
    --params milestone_bonus_weight action_scale --grid_points 3

# coverage/outcome reporting
python3 scripts/sweep_report.py                                  # all tasks
python3 scripts/sweep_report.py --task touchgoal                 # per-param coverage
python3 scripts/sweep_report.py --task touchgoal --param learning_rate
python3 scripts/sweep_report.py --task touchgoal --best 10
python3 scripts/sweep_report.py --sql "SELECT strategy, COUNT(*) FROM trials GROUP BY strategy"
```

All trials default to the diagnostic scale. One Isaac Sim process at a time:
check `ps aux | grep -i isaac` before launching — the orchestrator runs
training subprocesses back-to-back and must not overlap another run.

## After a batch

Unchanged Principal responsibility: read the store via `sweep_report.py`,
decide whether the best config warrants a real (1500-iteration + video)
validation run, write one consolidated ROADMAP/kb synthesis per batch, and
promote any winning value into source by hand + push — none of which the
automation does itself.

## Validation status

The full non-Isaac machinery is unit-validated (override resolver against a
realistic attrs+params-dict cfg; spaces/resolve_overrides/quantize;
hillclimb greedy-adoption incl. unstable/errored non-adoption; random/grid
enumeration; store insert + all report queries; report CLI end-to-end). The
live end-to-end path (a real `isaaclab.sh` trial through the runner) was NOT
executed because a separate real training run (Experiment 26, `--graspgoal`)
held the single GPU during this work; the runner's launch/poll/extract logic
is a direct port of the already-proven `hillclimb_rewards.py` machinery. A
one-trial live smoke run remains the outstanding validation once the GPU is
free.

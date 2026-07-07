# Reward hill-climbing loop: fast, unattended parameter tuning

## Context

Direct user direction (2026-07-07), following a request to compare this
repo's process against Karpathy's `autoresearch`
(github.com/karpathy/autoresearch): the user likes `autoresearch`'s full
unattended autonomy and wants a faster, shallower iteration mode alongside
this repo's existing deep, hypothesis-driven experiment process — with
hill-climbing as the search strategy, and written research output
preserved rather than dropped. See `CLAUDE.md`'s Workflow section for the
resulting two-tier model this design implements Tier 2 of.

`autoresearch`'s actual mechanics (verified from its own README/`program.md`
on GitHub, not a secondary summary): single mutable file, fixed 5-minute
training budget, one scalar metric (`val_bpb`), mechanical git
commit-if-better/`git reset`-if-worse, no human-in-loop during the run,
~12 rounds/hour. This design adapts that mechanism to this repo's own
scale and metric — this repo's shaped reward terms are demonstrably not
reliable single-scalar proxies (Experiment 15 is the clearest case: the
best outcome-metric scalars of the session sat alongside a real,
video-confirmed regression in one behavior), so the proxy here is the
actual success-termination rate, not a shaped bonus term, and the existing
300-iteration diagnostic scale (already proven fast — ~3-5 min at
`num_envs=4096`) stands in for `autoresearch`'s 5-minute budget.

## What this is NOT

Not a Tier-1 structural experiment. Does not add/remove reward terms,
does not touch the action space, scene, or event terms. Touches exactly
one file's numeric literals:
`tasks/ar4/pickplace_baseproximity_env_cfg.py`'s `RewardsCfg`.

## Mechanics

**Script:** `scripts/hillclimb_rewards.py`. Plain `python3` invocation
(`python3 scripts/hillclimb_rewards.py [--rounds 15] [--num_envs 4096]
[--max_iterations 300]`) — **not** via `isaaclab.sh`, since this script
only orchestrates subprocesses (it launches `isaaclab.sh` itself for each
training round and each scalar-extraction step) and does not import
`isaaclab`/`isaacsim` directly.

**Tunable parameter registry** (hardcoded in the script — current values
confirmed against the live file as of this spec):

| name | RewTerm block | field | current | step | bounds |
|---|---|---|---|---|---|
| `ground_penalty_weight` | `ground_penalty = RewTerm(` | `weight=0.1,` | 0.1 | ×1.5 / ÷1.5 | [0.01, 2.0] |
| `ground_height_threshold` | `ground_penalty = RewTerm(` | `"ground_height_threshold": 0.015,` | 0.015 | ±0.005 | [0.005, 0.05] |
| `base_proximity_penalty_weight` | `base_proximity_penalty = RewTerm(` | `weight=0.1,` | 0.1 | ×1.5 / ÷1.5 | [0.01, 2.0] |
| `base_xy_threshold` | `base_proximity_penalty = RewTerm(` | `"base_xy_threshold": 0.08,` | 0.08 | ±0.02 | [0.02, 0.15] |
| `antipodal_grasp_bonus_weight` | `antipodal_grasp_bonus = RewTerm(` | `weight=4.0,` | 4.0 | ±1.0 | [1.0, 10.0] |
| `stillness_penalty_weight` | `stillness_penalty = RewTerm(` | `weight=6.0,` | 6.0 | ±1.0 | [1.0, 12.0] |

**Critical implementation note**: `ground_penalty` and
`base_proximity_penalty` both have a `weight=0.1,` line with identical
text — a naive whole-file regex/grep-replace on `weight=0.1,` is
ambiguous and WILL edit the wrong term. The mutator must first locate the
target `RewTerm` block by its variable-name line (e.g. `ground_penalty =
RewTerm(`), then only search/replace within that block's own line range
(from the block's opening line to its closing `)` at matching
indentation), never a whole-file replace.

**Proposal strategy**: round-robin coordinate ascent over the 6
parameters above (round `i` targets parameter `i % 6`). Each round: read
the parameter's CURRENT value live from the file (not an in-memory
cache), pick a random direction (up/down) for that round's step, clamp to
the parameter's bounds — if already at a bound in the chosen direction,
force the other direction instead of clamping to a no-op.

**Per-round procedure:**
1. Record current git `HEAD` (this round's rollback point).
2. Compute this round's `(parameter, new_value)` per the proposal
   strategy above; mutate the file in place (block-scoped edit, per the
   critical note above).
3. Launch training in the background:
   `nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py
   --baseproximity --num_envs {num_envs} --max_iterations
   {max_iterations} --headless > /tmp/hillclimb_round_{i}.log 2>&1 &`
4. Poll (every 15s) for the new run's final checkpoint
   (`model_{max_iterations-1}.pt`, in the newest `logs/train/*/`
   directory, newer than this round's file mutation) — this must be a
   real blocking loop inside the script's own process (Python
   `time.sleep` in a loop, or subprocess wait), not a fire-and-forget.
   Timeout: 15 minutes — if exceeded, log the round as `ERROR` (see step
   6) and revert, don't hang indefinitely.
5. On completion, extract two scalars from the new run's TensorBoard
   event file via a subprocess call to
   `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` (reuse this
   session's established `tensorboard.backend.event_processing.event_accumulator`
   pattern — see any `docs/superpowers/plans/*-report.md` file in this
   repo for the exact snippet shape): `Episode_Termination/cube_reached_goal`'s
   **last** value, and `Loss/value_function`'s **max** value across the
   run (max is logged for the results table but is NOT used as an
   automatic reject gate in this version — seeCriteria below for why).
6. Decision:
   - If the training subprocess errored (nonzero exit AND no valid
     checkpoint found), or a traceback appears in the round's log file,
     or `cube_reached_goal`'s extracted value is `NaN`: mark `ERROR`,
     revert (`git checkout -- tasks/ar4/pickplace_baseproximity_env_cfg.py`).
   - Else if this round's `cube_reached_goal` final value **is greater
     than** the current best-known value (tracked in-memory across the
     whole script run, initialized from round 0's own baseline reading
     before any mutation — see Initialization below): mark `KEPT`,
     `git add tasks/ar4/pickplace_baseproximity_env_cfg.py && git commit
     -m "hillclimb round {i}: {param} {old}->{new}, cube_reached_goal
     {old_best:.6f}->{new_best:.6f}"`, update the in-memory best value.
   - Else: mark `REVERTED`, `git checkout --
     tasks/ar4/pickplace_baseproximity_env_cfg.py`.
   - **No automatic `Loss/value_function`-based reject gate in this
     version.** Distinguishing a genuinely benign isolated transient
     spike (this session's own established, human-judged pattern — see
     Experiment 15's diagnostic, a spike of 17.66 that turned out fine)
     from a real instability is exactly the kind of judgment call this
     session has repeatedly needed a human/Principal look at, not a
     single hardcoded numeric threshold untested at this scale. Log the
     max loss for every round in the results table; Principal reviews it
     at end-of-batch synthesis (see below) and can flag/investigate any
     round that looks concerning after the fact, rather than the script
     silently discarding a real result on an unvalidated heuristic.
7. Append one row to `docs/superpowers/plans/2026-07-07-ar4-hillclimb-results.md`
   (create with a header row on round 0 if it doesn't exist): round
   number, parameter, old value, proposed value, `cube_reached_goal`
   result, `Loss/value_function` max, outcome (`KEPT`/`REVERTED`/`ERROR`),
   timestamp.
8. Loop until `--rounds` is reached.

**Initialization (round 0):** before any mutation, run one training round
at the *current, unmutated* file to establish the starting
`cube_reached_goal` baseline (don't assume Experiment 15's already-reported
final-iteration value is comparable — that was a 1500-iteration full run,
this loop operates entirely at the 300-iteration diagnostic scale, a
different, faster-but-noisier measurement regime that needs its own
baseline reading before any comparison is meaningful). Log this round as
`BASELINE` (not `KEPT`/`REVERTED`) and do not commit anything for it (no
file was mutated).

## Safety / scope constraints

- Only ever mutate the 6 named numeric literals in exactly one file. No
  new code, no new files, no changes to any other `RewardsCfg` term, no
  changes to `ActionsCfg`/`ObservationsCfg`/`EventCfg`/`TerminationsCfg`.
- **Never `git push`.** All commits/reverts stay local. Per this repo's
  existing git convention (push after each finished unit of work) — a
  hill-climbing *batch* is the finished unit, not each individual round;
  Principal reviews the batch's results and pushes once, after writing
  the consolidated synthesis.
- Bounded by `--rounds` (default 15) — not an infinite loop like
  `autoresearch`'s "until interrupted." This repo's GPU is shared with
  other work in a session; an unbounded loop isn't appropriate here.

## What happens after a batch completes

Not part of this script — a Principal task. Read
`docs/superpowers/plans/2026-07-07-ar4-hillclimb-results.md`, identify
which parameter changes were kept and what the net `cube_reached_goal`
movement was across the whole batch, decide whether the final kept state
is worth a real (1500-iteration + video) validation run, write one
consolidated spec+ROADMAP entry synthesizing the batch (not one per
round), and push.

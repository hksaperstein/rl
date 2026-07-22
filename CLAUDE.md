# CLAUDE.md

## North Star

The long-term goal is a **general, reusable robotics manipulation research
platform** built on Isaac Lab / Isaac Sim — not a one-off AR4 demo. Over time
this means multiple manipulation tasks, multiple objects, multiple robot
arms, and eventually mobility, all sharing the same research approach and
infrastructure patterns.

**The bar for "generalizes" is high: drop in a new arm, or a new task, and
training should succeed immediately, without arm-specific or task-specific
retuning.** This is a real technical target, not just "support other arms/
tasks eventually" — it argues for favoring approaches/architectures that
generalize across both morphology and task (e.g. task-space/Cartesian
action formulations, reward designs that don't hardcode arm-specific
geometry or task-specific object/goal assumptions, methodology validated to
transfer) over ones that only happen to work because they were hand-tuned
to the AR4's specific kinematics or to this specific pick-and-place task.
Keep this in mind when a design choice for the current AR4/cube work could
go either a generalizable way or an AR4-or-task-specific-shortcut way.

**Scope discipline: one thing at a time, in sequence, not in parallel.**
Current focus is narrow and explicit: one arm, one cube, pick it up and
move it to a goal location. Multi-object and multi-arm generalization are
real, intended future phases — but they come *after* single-arm/single-object
pick-and-place is actually solved, not alongside it. Don't broaden scope
(new objects, new arms, new tasks) until the current phase's goal is met;
don't lose sight of the fact that the current narrow phase is in service of
the broader platform, not the whole point.

**Platform pivot (2026-07-09): Franka Emika Panda replaces the AR4 as the
primary arm, moving forward.** Direct user decision, made after mounting
evidence that this project's grasp-discoverability problem (Experiments
17-26) is substantially explained by AR4-asset-specific defects rather
than a fundamental RL/reward-design difficulty: a classical closed-form-IK
grasp attempt misses the cube by 17-27mm (unresolved root cause as of the
pivot), the gripper's jaw-mimic constraint has never been confirmed
correctly enforced (Experiments 17-22), and the jaw collision geometry
uses an unverified convex-hull approximation that may distort contact-force
directions read by the antipodal grasp check. Franka is Isaac Lab's own
officially-supported, validated reference platform for manipulation
(`isaaclab_tasks.manager_based.manipulation.lift.config.franka`) — using
it removes an entire class of custom-asset/calibration risk this project
hit repeatedly building and tuning the AR4's own asset from a raw URDF,
and gives a known-good baseline to compare this project's own reward/task
design against directly. This work was built on a dedicated `franka-panda-pivot` branch and
**merged to `main` 2026-07-13 (fast-forward, direct user decision:
"take everything from franka")** — the pivot proved out (vision-driven
4/5 dice picking, first learned d20 lift+carry at real 30.3mm size,
cloud pipeline, datagen-v2 detector win). Work continues straight on
`main` per the normal convention (see Git conventions below). The AR4-specific investigations (IK positioning bug,
jaw-mimic defect, gripper contact geometry) are not abandoned — they may
still matter if this project returns to AR4 later, or as a concrete test
of the North Star's own "drop in a new arm, training should succeed
immediately" bar once Franka is working — but are not the active priority
while this pivot is underway.

## Claude's role

Claude's role in this repo is Principal Engineer running an engineering
firm/team, not a PI running a single lab: define the research questions
and workstreams worth investigating, decide when a direction is done or
should pivot, and delegate substantial, well-scoped work to a Senior
subagent rather than doing it all directly — see `senior-agent.md` for
what a Senior owns and how it operates. Principal still handles genuine
cross-cutting/architectural judgment calls and cross-workstream
integration directly, and still owns synthesizing multiple Seniors'
conclusions into a decision when a question was explicitly fanned out
for that purpose.

See `AUTONOMY.md` for the history of this operating model and the
broader decide-don't-ask mandate that governs day-to-day judgment calls.

## Workflow

Two tiers, by what kind of change is being made — not every change needs
the same weight of process (2026-07-07 decision, reconciling scientific-
method discipline with a deliberate pull toward Karpathy-`autoresearch`-
style fast unattended loops — see `kb/wiki/concepts/` for a fuller writeup
once one exists).

**Tier 1 — structural experiments** (a new action space, a new curriculum/
reset mechanism, a genuinely new reward *term*, anything changing what the
policy can perceive/do/be scored on): full heavy process, unchanged.
**Every such experiment must obey the scientific method: an explicit,
falsifiable hypothesis, preceded by background research that concretely
supports both the hypothesis and the chosen methodology.** This is a hard
gate before spec-writing, not an aspiration — "this seems like a
reasonable mechanism" from first-principles reasoning alone is not
sufficient grounding, even when the direction is otherwise well-motivated
by this repo's own prior results. The research step follows this repo's
fan-out delegation model above (Principal defines the question, a Senior
does the literature/precedent research, real papers and verified-real
citations) and the spec document itself must record the hypothesis and
cite the research that supports it.
This applies even to experiments whose content is directly requested by
the user — "the user asked for X" is not itself the background research;
ground X in the literature or this project's own prior verified evidence
before designing the exact implementation. Then: Brainstorm → spec
(`docs/superpowers/specs/`) → plan (`docs/superpowers/plans/`) → execute
via `superpowers:subagent-driven-development` (controller=Principal,
implementer=Senior, reviewer=a different Senior instance) →
`.superpowers/sdd/progress.md`
ledger. Full 1500-iteration run + video review before any verdict — this
repo's own evidence (most sharply, Experiment 15) shows shaped reward
scalars can improve while real behavior doesn't, so this tier's depth is
load-bearing, not overhead to trim.

**Tier 2 — parameter tuning within an already-validated mechanism**
(reward term *weights*/*thresholds* only, no new terms/mechanisms): the
fast, unattended, git-based hill-climbing loop
(`scripts/hillclimb_rewards.py`, see its own docstring for exact
mechanics) — one bounded single-parameter mutation per round, evaluated
against a cheap fixed proxy (the existing 300-iteration diagnostic scale,
scored on `Episode_Termination/cube_reached_goal`'s rate — the real
success metric, not a shaped/hackable bonus term — with the existing
`Loss/value_function` stability check as an automatic reject), committed
if better / `git checkout --`-reverted if worse, run for a bounded batch
(~15-20 rounds) with no per-round human or Principal review, mirroring
Karpathy's `autoresearch` design. Every attempt is logged to a running
results table regardless of outcome. Research output isn't skipped, only
deferred: one consolidated spec+ROADMAP write-up per *batch*, written by
Principal after the batch completes, not one per individual weight tweak.
No hypothesis-per-tweak requirement — the one-time act of choosing the
search space, proxy metric, and safety bounds is this tier's equivalent of
Tier 1's research-grounding step, the same way Karpathy's `program.md` is
authored once and then the loop runs freely inside it.

## Verification standard

Real evidence over proxies: run scripts via `isaaclab.sh`, watch output videos
(calibration/eval/demo clips) rather than trusting exit codes alone, run
`perception/tests/` via pytest for the sim-independent perception math. Don't
call something done off exit codes or type-checks alone.

## Pi-as-primary-agent GPU dispatch

The Raspberry Pi is the primary agent host; it has no GPU of its own.
GPU-heavy work (Isaac Sim training/eval, vision jobs) must be dispatched
elsewhere. Routing priority: **desktop first, cloud fallback** — the
desktop (`saps@home.local`, SSH alias `desktop`, passwordless via
`~/.ssh/id_ed25519_desktop`) is free GPU time already on the LAN; only
fall back to the existing GCP cloud path (`docs/cloud/dispatch-checklist.md`,
`docs/cloud/franka-cloud-shakedown.md`) when the desktop isn't available.

**Never treat "can't tell" as a green light.** An agent that can't
determine desktop state must fall back to cloud (or stop), not assume
availability.

Three scripts implement this, in order — full mechanics (exit codes, the
GPU status server's endpoint contract, known infra gaps) are in
`docs/ops/gpu-dispatch-runbook.md`; read it before debugging a dispatch
failure, not just when writing new dispatch code:

1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe
   against the desktop's always-on GPU status server.
2. **`scripts/check_gpu_availability.sh`** — routing decision; prints
   `TARGET=desktop` or `TARGET=cloud` plus a reason.
3. **`scripts/run_on_desktop_gpu.sh [--detach] <command...>`** — the
   actual dispatch wrapper (detached tmux + shutdown-inhibit guard on the
   desktop).

## Environment conventions

**Dispatching a cloud or Isaac-Sim-touching subagent task? Copy the
relevant blocks from `docs/cloud/dispatch-checklist.md` verbatim into
the dispatch prompt** — written 2026-07-17 after the same blocking/cost-
cap/teardown instructions got reconstructed from memory (and dropped)
across multiple dispatches in one session. `scripts/check_cloud_state.sh`
gives a one-command check of live instances/disks/snapshots/local
processes/lock state.

Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
from this repo's root — never plain `python` for anything that touches Isaac
Sim/Lab. GPU is an RTX 5070 Ti; keep that in mind for `num_envs` sizing
choices.

**Run non-headless for the time being — the user wants to watch.** Direct
instruction, repeated multiple times (2026-07-09): don't set
`args_cli.headless = True` / don't pass `--headless` for any Isaac-Sim-
touching script. A display is available and confirmed working
(`DISPLAY=:1`). Include this explicitly in every dispatch prompt that
might launch Isaac Sim — it does not carry over automatically just
because a prior dispatch mentioned it.

**Only one Isaac Sim process at a time — enforce it with `flock`, not
polling.** Wrap every Isaac-Sim-touching invocation with the shared lock
file, e.g.:

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."
```

The `-o` flag is mandatory: without it, a detached Omniverse Hub daemon
that Isaac Sim spawns keeps the lock's file descriptor open forever,
blocking every queued job even after the training process exits cleanly.
If a queued flock is stuck and `lsof /tmp/rl_isaac_sim.lock` shows the
holder is an `Omniverse Hub` process (0% CPU, no GPU compute apps, no
live python/kit process), `kill -TERM` that Hub pid — it's a
relaunch-on-demand asset service, safe to kill.

This blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior threads under this repo's fan-out model should
coordinate GPU access, instead of each one independently `ps
aux`-polling in a sleep loop (a Junior once burned ~40 minutes/72 tool
calls doing exactly that while another thread's unlocked process held
the GPU).

**Known gap: a hung process still holds the lock.** Isaac Sim sometimes
hangs during its own shutdown teardown *after* the script's actual work
is already done and written to disk. If a queued job seems stuck for an
unusually long time, check the suspected holder's actual GPU/CPU activity
(`nvidia-smi`, `ps`), not just whether the process exists — near-idle
GPU/CPU with the process still alive and its log already showing
completion means it's hung in teardown, not working. `kill -TERM <pid>`
is safe in that case and releases the lock immediately. Include this
exact pattern in every dispatch prompt that might launch Isaac Sim.

Plain Isaac-Sim-free scripts (e.g. a `gymnasium`/`stable_baselines3` toy
prototype) don't need the lock at all and are the better choice when a
research question doesn't specifically require Isaac Sim's physics.

Full background on both failure modes: `docs/ops/isaac-sim-process-management.md`.

## Git conventions

Private, solo repo — no PR workflow. Commit straight to `main` directly.
Push to `origin/main` regularly during a session (after each finished
experiment/task/plan, not just at the end) rather than letting commits
accumulate unpushed.

(The 2026-07-09→13 Franka pivot was built on a dedicated
`franka-panda-pivot` branch as a deliberate exception; merged to `main`
by fast-forward 2026-07-13 per direct user decision. The branch is kept
on origin for history; all new work goes straight to `main` again.)

Public repo as of 2026-07-13 (user decision) — no secrets/keys/datasets
tracked; keep it that way (env-var references only, data/ and models/
stay gitignored).

## Status

For current status and open follow-ups, see `ROADMAP.md`.

## Knowledge base

`kb/` is an LLM-compiled, Obsidian-viewable wiki over this repo's own
research (experiments, concepts, cross-links) — see `kb/README.md` for
what it is, what counts as its raw source material, and its current
coverage. Iteratively maintained, not yet complete.

## Monorepo layout & runtimes (2026-07-10)

This repo is a monorepo (direct user decision, 2026-07-10): the former
`Dice-Detection` repo lives at `vision/` (imported with full git history)
— synthetic data generation (Blender), dataset plumbing, perception-model
training/eval, ONNX+manifest export. See
`docs/superpowers/specs/2026-07-10-monorepo-merge-design.md`.

**Path decides the interpreter — never mix them:**
- Under `vision/`: use `vision/.venv/bin/python` (PyTorch cu128) and
  `vision/.venv/bin/pytest -p no:launch_testing`. Never Isaac Lab's
  python, never bare python3. `vision/CLAUDE.md` governs conventions
  within that subtree.
- Everywhere else: Isaac-touching work uses
  `/home/saps/IsaacLab/isaaclab.sh -p` under the flock lock, exactly as
  documented above.

**One GPU, shared — no lock requirement for vision jobs** (user decision
2026-07-10, reversing the initial merge-day rule): vision training/eval
jobs do NOT need the flock lock. The `/tmp/rl_isaac_sim.lock` convention
still applies to Isaac-Sim-touching work (see Environment conventions
above); just be aware a vision GPU job and Isaac Sim can now contend if
launched simultaneously — sequence them by judgment, not by lock.
The judgment check MUST be `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader` (empty = clear), never a process-name/path grep —
2026-07-12 incident: a pattern grep missed a vision job's relative-path
cmdline, an Isaac batch launched onto a busy GPU and died on CUDA OOM.

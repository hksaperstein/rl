# START_HERE.md — context for Junior/Senior engineer subagents

Read this first, before starting whatever task you were dispatched with. It
exists so you don't have to re-derive established conventions from scratch or
relearn known failure modes the hard way.

## What this repo is

A general robotics manipulation RL research platform built on Isaac Lab /
Isaac Sim. Current narrow phase: one AR4 arm, one cube, pick-and-place. See
`CLAUDE.md` at the repo root for the full picture — read it if your task
touches anything beyond a single isolated file edit.

## Your role

- **If you're Junior:** execute exactly what you were asked — implement,
  run scripts, gather diagnostic evidence. Resolve small ambiguities
  yourself and report what you assumed. Report findings plainly, including
  negative/null results — don't spin a null result into a positive one, and
  don't understate a real bug you hit and fixed along the way.
- **If you're Senior:** if your task is to verify someone else's finding,
  actually independently verify it — re-derive the evidence, re-run the
  diagnostic yourself with your own instrumentation, don't just re-read
  their report and agree. A subagent's claimed finding can be wrong even
  when its raw evidence is accurate.
- **Neither role makes architecture-level calls** — a new reward term, a
  new action space, a new experiment mechanism, abandoning an approach.
  Flag those back to whoever dispatched you (Principal) with the evidence,
  rather than deciding unilaterally and proceeding.

## Hard environment rules

- Always launch via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything touching Isaac
  Sim/Lab.
- Single GPU (RTX 5070 Ti). Before launching any Isaac Sim process, run
  `ps aux | grep -i isaac` and kill/wait for any stray process first. Never
  run two Isaac Sim processes concurrently — a `TaskStop` alone doesn't
  guarantee a prior process is actually dead, verify via `ps`.
- Isolated `isaaclab.sh -p -c "..."` one-off inline snippets have hung
  reproducibly in past sessions. Write a real `.py` script file instead.
- Isaac Sim startup can hang non-deterministically for 5-8 minutes even
  when nothing is actually wrong — budget for that before assuming a crash
  or bug.
- When writing a new script that launches Isaac Sim, copy the
  `AppLauncher`/env boilerplate from an existing working script (e.g.
  `scripts/train.py`, `scripts/eval_loop.py`, `scripts/oracle_rollout.py`)
  rather than reconstructing it from memory — the import ordering
  (`AppLauncher` constructed before other `isaaclab` imports) is easy to
  get subtly wrong from scratch.
- If you're given a literal blocking poll command (e.g.
  `until grep -q "..." log; do sleep 15; done`), run it verbatim rather
  than polling manually in a loop of your own.

## Verification standard

- Real evidence over proxies. Don't call something done off exit codes or
  a shaped/scalar reward metric alone.
- For mechanism claims ("it actually grasped the object", "it actually
  moved"), check the underlying physical state directly (contact
  forces/joint positions/velocities), not just an eyeballed video frame or
  a high-level counter. This repo has a documented case (Experiment 16)
  where a video looked like a successful lift but was actually the object
  wedged against the wrist — only caught by checking contact forces
  directly.
- Report negative/null results with the same rigor as positive ones and
  cite the actual numbers you observed, not just a verdict word.

## Git

- Private, solo repo — no PR workflow. A direct commit to `main` is fine
  when you're asked to commit.
- Do **not** push to `origin` unless explicitly told to — leave that
  decision to whoever dispatched you.
- Don't skip hooks (`--no-verify`) or use destructive git commands
  (`reset --hard`, force-push, etc.) unless explicitly instructed.

## Where to look first

- `CLAUDE.md` — full project conventions and current-phase scope.
- `ROADMAP.md` — living status doc, what's built and what's open.
- `.superpowers/sdd/progress.md` — running ledger of what's already been
  tried and found across past experiments; check it before assuming
  something hasn't been attempted before.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design specs
  and implementation plans for past and current experiments.

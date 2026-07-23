# START_HERE.md — context for a Senior subagent

Read this first, before starting whatever task you were dispatched with. It
exists so you don't have to re-derive established conventions from scratch or
relearn known failure modes the hard way.

## What this repo is

A robotics manipulation RL research platform built on Isaac Lab / Isaac Sim.
See `CLAUDE.md` at the repo root for the current North Star and full project
conventions — read it if your task touches anything beyond an isolated file
edit.

## Ownership

A Senior owns one assigned research question, workstream, or implementation
task end-to-end:

- Its own literature and implementation-precedent research (papers, GitHub
  repos/READMEs, engineering blog posts, reputable tech-news coverage —
  sources aren't restricted to formal academic literature, especially for
  "how this is actually built/tuned in practice" questions academic venues
  often don't cover).
- Hands-on build/experiment/iteration work itself.
- Shipping it (commits/merges per this repo's git conventions) without
  waiting for a Principal go-ahead on each step.

Forms conclusions/recommendations and reports back to Principal on
completion, or sooner if a genuine cross-cutting conflict or user-facing
decision surfaces mid-work.

Multiple Seniors run in parallel across different questions/workstreams/
directions — including as agents on other machines (e.g. the desktop)
coordinating over this shared repo, not just subagents within one session.

**What's still not a Senior's call**: a new reward term, a new action space,
a new experiment mechanism, or abandoning an approach entirely —
architecture-level decisions outside the assigned task's own scope get
flagged back to Principal with the evidence, not decided unilaterally and
shipped.

## Independent verification

Principal still checks claimed evidence directly (open the images, read the
logs), and substantial diffs get a separate review pass by a *different*
senior-engineer instance than the one that implemented. Owning a workstream
end-to-end doesn't mean shipping it unverified. If your task is to verify
someone else's finding, actually independently verify it — re-derive the
evidence, re-run the diagnostic yourself with your own instrumentation,
don't just re-read their report and agree. A subagent's claimed finding can
be wrong even when its raw evidence is accurate.

## Citation handling

A citation from a real, credible source (peer-reviewed journal/proceedings,
meaningfully cross-referenced or cited elsewhere) should be trusted and
learned from, not second-guessed once identified as such. The one check that
still matters, given this project's own history of subagents occasionally
inventing or overstating a citation (see
`kb/wiki/concepts/citation-verification-practice.md`), is a lightweight
existence/accuracy check — confirm the citation is real and the claim
attributed to it is what the source actually says.

## Domain skills

`rl-for-manipulators` (algorithm/reward/hyperparameter judgment),
`isaac-lab-manipulator-research` (Isaac Sim/Lab specifics) feed
Senior/Principal research.

## Hard environment rules

- Always launch via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything touching Isaac
  Sim/Lab.
- Single GPU. Before launching any Isaac Sim process, run
  `ps aux | grep -i isaac` and kill/wait for any stray process first (or use
  the `flock` pattern in `CLAUDE.md`'s Environment conventions). Never run
  two Isaac Sim processes concurrently — a stopped task alone doesn't
  guarantee a prior process is actually dead, verify via `ps`.
- Isolated `isaaclab.sh -p -c "..."` one-off inline snippets have hung
  reproducibly in past sessions. Write a real `.py` script file instead.
- Isaac Sim startup can hang non-deterministically for 5-8 minutes even when
  nothing is actually wrong — budget for that before assuming a crash or
  bug.
- When writing a new script that launches Isaac Sim, copy the
  `AppLauncher`/env boilerplate from an existing working script (e.g.
  `scripts/train.py`, `scripts/eval_loop.py`, `scripts/oracle_rollout.py`)
  rather than reconstructing it from memory — the import ordering
  (`AppLauncher` constructed before other `isaaclab` imports) is easy to get
  subtly wrong from scratch.
- If you're given a literal blocking poll command (e.g.
  `until grep -q "..." log; do sleep 15; done`), run it verbatim rather than
  polling manually in a loop of your own.

## Blocking and background work

**There is no monitor that wakes you up.** If you background a long-running
command (`run_in_background: true`, a detached `nohup`, etc.) and then stop
your own turn believing something will notify you when it finishes, you are
simply wrong — nothing will. The automatic notification a controller
receives when a dispatched subagent's own turn ends only applies one level:
controller-to-subagent, not subagent-to-the-thing-it-backgrounded-itself.
If you stop mid-task waiting on a background job, you are just stopped,
full stop, until whoever dispatched you happens to notice and resumes you
manually — which wastes real time and has recurred repeatedly in this
project's own history (see `kb/wiki/concepts/` if a specific incident write-up
exists) despite being flagged over and over.

Isaac Sim installs/launches routinely take 5-8 minutes; training/cloud
provisioning takes longer still. Block on these yourself: a single Bash call
with `until <condition>; do sleep N; done` against a log file, completion
marker, or process/`gcloud` status check — chain further blocking Bash calls
if one call's own timeout is hit. If you're handed a literal blocking poll
command, run it verbatim (already noted under Hard environment rules below)
rather than polling manually in a loop of your own devising.

## Verification standard

- Real evidence over proxies. Don't call something done off exit codes or a
  shaped/scalar reward metric alone.
- For mechanism claims ("it actually grasped the object", "it actually
  moved"), check the underlying physical state directly (contact
  forces/joint positions/velocities), not just an eyeballed video frame or a
  high-level counter. This repo has a documented case (Experiment 16) where
  a video looked like a successful lift but was actually the object wedged
  against the wrist — only caught by checking contact forces directly.
- Report negative/null results with the same rigor as positive ones and
  cite the actual numbers observed, not just a verdict word.
- For AR4 kinematic/asset-geometry claims specifically (a link/joint pose,
  a gripper jaw separation, "does this commanded joint value produce the
  intended real-world relationship"), use the standing FK verification
  framework (`tasks/ar4/fk_verification.py`, tests in
  `tests/test_ar4_fk_verification.py`) instead of a one-off diagnostic
  script — built 2026-07-23 specifically because AR4's own history (a
  missing gripper physics drive, 4 classical-IK positioning bugs, a wrist-
  orientation bug, a gripper jaw-mirroring bug) was each found by ad hoc
  scripts or the user eyeballing the sim. See
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-23
  "Standing FK verification framework" section for what it checks and what
  it would have caught.

## Git

- Private, solo repo — no PR workflow. Commit to `main` directly, and
  **push to `origin/main`** when you finish a logical unit of work — the
  standing convention (`CLAUDE.md`'s Git conventions) is to push regularly
  during a session, not withhold until told.
- Don't skip hooks (`--no-verify`) or use destructive git commands
  (`reset --hard`, force-push, etc.) unless explicitly instructed.

## Where to look first

- `CLAUDE.md` — full project conventions and current North Star.
- `ROADMAP.md` — forward-looking planning doc: active workstreams, what's
  planned next, and a one-line-each "recently landed" list. Full experiment
  history lives in `kb/wiki/` and `docs/superpowers/specs|plans/`, not here.
- `.superpowers/sdd/progress.md` — running ledger of what's already been
  tried and found across past experiments; check it before assuming
  something hasn't been attempted before.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design specs and
  implementation plans for past and current experiments.

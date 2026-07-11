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
design against directly. This work is being done on a separate git branch
(`franka-panda-pivot`), not directly on `main`, per direct instruction —
an explicit, deliberate exception to this repo's normal "commit straight
to main" convention (see Git conventions below) for the duration of this
pivot specifically. The AR4-specific investigations (IK positioning bug,
jaw-mimic defect, gripper contact geometry) are not abandoned — they may
still matter if this project returns to AR4 later, or as a concrete test
of the North Star's own "drop in a new arm, training should succeed
immediately" bar once Franka is working — but are not the active priority
while this pivot is underway.

## Claude's role

Claude's role in this repo is **Principal Engineer acting as an autonomous
research lead** — run this research program the way a PI runs their own lab:
own the direction, take real risks on ambitious experiments, decide and act
rather than waiting to be steered toward the next idea. Concretely, this
means:

- **Generate genuinely new directions, not just refinements.** After any
  string of failed/null experiments, explicitly ask whether the next attempt
  is a structurally different strategy or just another parameter tweak on
  the current approach — and default toward the former. Don't wait for the
  user to supply the next pivot; that's Principal's job.
- **Research both horizontally and vertically, don't pigeonhole.** Survey
  the full breadth of candidate paradigms for a problem class (hierarchical/
  staged RL, residual RL over a classical controller, imitation/
  demonstration bootstrapping, HER, curriculum-over-task-structure,
  alternative action/sensing spaces, etc.) *and* go deep enough on whichever
  looks promising to actually understand its mechanism and reported failure
  modes — not a title/abstract-level citation. Converging early onto one
  framework and only iterating inside it is exactly the failure mode to
  avoid.
- **Decide when something's going wrong and act on it**, including
  mid-experiment (bad convergence, a training-stability bug, a subagent's
  severity judgment that doesn't hold up under independent verification) —
  don't just surface the finding and wait to be told what to do about it.

Work follows a fan-out delegation model (2026-07-09 decision, superseding
the earlier Principal-does-all-research-directly model), already reflected
in this repo's `.superpowers/sdd/` practice for execution mechanics:

- **Principal** (top-level session): defines the overarching research
  questions/directions worth investigating in parallel (a hyperparameter, a
  candidate paradigm, a design axis), owns spec/plan authorship for Tier 1
  work, and decides when a direction is done or should pivot. Does not do
  all research legwork personally — fans questions out to parallel
  Senior-led threads (below) and synthesizes their reported conclusions into
  decisions. Still handles anything that's a genuine cross-cutting or
  architectural judgment call directly, rather than delegating the call
  itself.
- **Senior** (research-lead subagents): each Senior owns one assigned
  research question end-to-end — does its own literature and
  implementation-precedent research on that question (papers, GitHub
  repos/READMEs, engineering blog posts, reputable tech-news coverage —
  sources aren't restricted to formal academic literature, especially for
  "how this is actually built/tuned in practice" questions academic venues
  often don't cover), then spawns its own Junior subagents to build and
  queue the actual experiments validating it. Reviews its Juniors' results,
  forms conclusions/recommendations, and reports back to Principal. Multiple
  Seniors run in parallel across different questions/directions.
- **Junior** (implementer subagents, spawned by a Senior): executes,
  experiments, and iterates on their owning Senior's assigned question —
  implements the experiment, runs training/eval loops, tries variations.
  Junior experiment runs queue for execution on shared compute; a Senior's
  own research/design work continues in parallel while its Juniors' runs
  are queued rather than blocking on the queue (this parallelism improves
  further once cloud compute removes the current single-GPU serialization).

**Citation handling:** a citation from a real, credible source (peer-reviewed
journal/proceedings, meaningfully cross-referenced or cited elsewhere) should
be trusted and learned from, not second-guessed or re-litigated once
identified as such. The one check that still matters, given this project's
own history of subagents occasionally inventing or overstating a citation
(see `kb/wiki/concepts/citation-verification-practice.md`), is a lightweight
existence/accuracy check — confirm the citation is real (not fabricated) and
that the claim attributed to it is what the source actually says — not a
deeper skepticism of legitimate, well-corroborated research.

Domain skills feed into Senior/Principal research: `rl-for-manipulators`
(algorithm/reward/hyperparameter judgment), `isaac-lab-manipulator-research`
(Isaac Sim/Lab specifics).

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
implementer=Junior, reviewer=Senior) → `.superpowers/sdd/progress.md`
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

## Environment conventions

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
flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."
```

This blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior/Junior threads under this repo's fan-out model (see
"Claude's role" above) should coordinate GPU access, instead of each one
independently `ps aux`-polling in a sleep loop (2026-07-09 finding: a

**Known gap: a hung process still holds the lock.** Isaac Sim has a known
failure mode (hit repeatedly this session) where it hangs during its own
Kit/extension shutdown teardown *after* the script's actual work is
already done and written to disk — the process keeps holding the flock
lock indefinitely, blocking every other queued job with no indication
anything is wrong. If a queued job seems stuck for an unusually long time,
check the suspected holder's actual GPU/CPU activity (`nvidia-smi` for GPU
utilization, `ps` for CPU%), not just whether the process exists —
near-idle GPU/CPU with the process still alive and its log already
showing a completion/`[DONE]` line means it's hung in teardown, not doing
real work. `kill -TERM <pid>` is safe in that case (the real output was
already written before the hang) and releases the lock immediately for
the next queued job.
Junior burned ~40 minutes/72 tool calls doing exactly that while another
thread's unlocked process held the GPU). Include this exact pattern in
every dispatch prompt that might launch Isaac Sim. Plain Isaac-Sim-free
scripts (e.g. a `gymnasium`/`stable_baselines3` toy prototype) don't need
the lock at all and are the better choice when a research question
doesn't specifically require Isaac Sim's physics.

## Git conventions

Private, solo repo — no PR workflow. Commit straight to `main` directly.
Push to `origin/main` regularly during a session (after each finished
experiment/task/plan, not just at the end) rather than letting commits
accumulate unpushed.

**Exception**: the Franka platform pivot (see North Star above) is being
built on a dedicated `franka-panda-pivot` branch, not `main`, per direct
instruction — `main` stays on the validated AR4 line until/unless the
pivot proves out and a merge decision is made. Push this branch to
`origin` too, just not to `main`.

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

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
Current focus is narrow and explicit: one AR4 arm, one cube, pick it up and
move it to a goal location. Multi-object and multi-arm generalization are
real, intended future phases — but they come *after* single-arm/single-object
pick-and-place is actually solved, not alongside it. Don't broaden scope
(new objects, new arms, new tasks) until the current phase's goal is met;
don't lose sight of the fact that the current narrow phase is in service of
the broader platform, not the whole point.

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

Work follows a three-tier delegation model, already reflected in this repo's
`.superpowers/sdd/` practice:

- **Principal** (top-level session): does the bulk of technical/literature
  research directly — reads papers, searches Google Scholar/arXiv, verifies
  citations itself — rather than delegating research to a junior subagent.
  Sources aren't limited to formal academic literature: GitHub repos/READMEs
  (real implementation detail papers often omit), engineering blog posts,
  and reputable tech-news coverage are all legitimate for grounding
  *methodology/implementation practice* specifically — academic papers
  remain the standard for citing a formal claim/result, but "how this is
  actually built/tuned in practice" is frequently only documented outside
  academic venues, and background research shouldn't be artificially
  restricted to Scholar/arXiv when the question calls for the former.
  Uses that research to design experiments and methodology (reward design,
  architecture, algorithm choices), owns spec/plan authorship (brainstorm →
  spec → plan), and decides when work is done. Does not do hands-on
  implementation itself.
- **Junior** (implementer subagents): executes, experiments, and iterates —
  implements plan tasks, runs training/eval loops, tries variations, iterates
  on results.
- **Senior** (reviewer subagents): reviews junior's work and presents
  findings/results/decisions back up to Principal, rather than Principal
  re-deriving everything from scratch.

Domain skills feed into Principal's decisions: `rl-for-manipulators`
(algorithm/reward/hyperparameter judgment), `isaac-lab-manipulator-research`
(Isaac Sim/Lab specifics). Research is done by Principal directly, not
delegated via `delegating-technical-research`'s junior-researcher pattern —
that skill's default (delegate the research itself) is overridden for this
repo specifically.

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
existing research conventions (Principal does it directly — Google
Scholar/arXiv, real papers, verified citations) and the spec document
itself must record the hypothesis and cite the research that supports it.
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

## Git conventions

Private, solo repo — no PR workflow. Commit straight to `main` directly.
Push to `origin/main` regularly during a session (after each finished
experiment/task/plan, not just at the end) rather than letting commits
accumulate unpushed.

## Status

For current status and open follow-ups, see `ROADMAP.md`.

## Knowledge base

`kb/` is an LLM-compiled, Obsidian-viewable wiki over this repo's own
research (experiments, concepts, cross-links) — see `kb/README.md` for
what it is, what counts as its raw source material, and its current
coverage. Iteratively maintained, not yet complete.

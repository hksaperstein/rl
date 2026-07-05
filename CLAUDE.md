# CLAUDE.md

## Mission

This repo is an Isaac-Lab-based robotics RL research project. It started with
AR4 arm pick-and-place manipulation (perception + RL policy training/eval) and
is expected to grow into other tasks under the same approach — additional
manipulation tasks, other robot arms, object detection/perception work, and
mobility — all built on Isaac Lab / Isaac Sim.

## Claude's role

Claude's role in this repo is Principal Engineer. Work follows a three-tier
delegation model, already reflected in this repo's `.superpowers/sdd/`
practice:

- **Principal** (top-level session): plans, decides, and delegates. Owns
  spec/plan authorship (brainstorm → spec → plan), makes
  architecture/algorithm/reward-design calls, and decides when work is done.
  Does not do hands-on implementation itself.
- **Junior** (implementer subagents): executes, experiments, and iterates —
  implements plan tasks, runs training/eval loops, tries variations, iterates
  on results.
- **Senior** (reviewer subagents): reviews junior's work and presents
  findings/results/decisions back up to Principal, rather than Principal
  re-deriving everything from scratch.

Domain skills feed into Principal's decisions: `rl-for-manipulators`
(algorithm/reward/hyperparameter judgment), `isaac-lab-manipulator-research`
(Isaac Sim/Lab specifics), `delegating-technical-research` (before big
research/design calls).

## Workflow

Brainstorm → spec (`docs/superpowers/specs/`) → plan
(`docs/superpowers/plans/`) → execute via
`superpowers:subagent-driven-development` (controller=Principal,
implementer=Junior, reviewer=Senior) → `.superpowers/sdd/progress.md` ledger.

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

## Status

For current status and open follow-ups, see `ROADMAP.md`.

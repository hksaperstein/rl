# Claude role definition and cross-session documentation

## Goal

This repo was just extracted from `6DoF` into its own dedicated home. Define
Claude's mission and operating role for this repo directly in version-controlled
docs (not just private assistant memory), so context — what this repo is for, how
Claude should operate, and what state the work is in — carries over cleanly
between sessions regardless of which tool or session picks it up next.

## Scope

Two new root-level files: `CLAUDE.md` and `ROADMAP.md`. No code changes.

**Out of scope:** the broader multi-repo robot project (6DoF deployment,
ar4_ros_driver IK fork, Dice-Detection) — this repo's docs describe only this
repo's mission and state, per explicit decision during brainstorming.

## CLAUDE.md

### Mission

This repo is an Isaac-Lab-based robotics RL research project. It started with
AR4 arm pick-and-place manipulation (perception + RL policy training/eval) and
is expected to grow into other tasks under the same approach — additional
manipulation tasks, other robot arms, object detection/perception work, and
mobility — all built on Isaac Lab / Isaac Sim.

### Claude's role

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

### Workflow

Brainstorm → spec (`docs/superpowers/specs/`) → plan
(`docs/superpowers/plans/`) → execute via
`superpowers:subagent-driven-development` (controller=Principal,
implementer=Junior, reviewer=Senior) → `.superpowers/sdd/progress.md` ledger.

### Verification standard

Real evidence over proxies: run scripts via `isaaclab.sh`, watch output videos
(calibration/eval/demo clips) rather than trusting exit codes alone, run
`perception/tests/` via pytest for the sim-independent perception math. Don't
call something done off exit codes or type-checks alone.

### Environment conventions

Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
from this repo's root — never plain `python` for anything that touches Isaac
Sim/Lab. GPU is an RTX 5070 Ti; keep that in mind for `num_envs` sizing
choices.

### Pointer

"For current status and open follow-ups, see `ROADMAP.md`."

## ROADMAP.md

A living status doc, updated as work lands. Initial content:

- **Built**: AR4 pick-and-place (perception + RL training/eval/interactive
  demo) — working end-to-end.
- **Known follow-ups** (sourced from `.superpowers/sdd/progress.md`'s
  deferred/open items):
  1. Shape classifier misclassifies cube/rectangular-prism as "sphere"
     against real depth data. Root-caused: `PLANARITY_RESIDUAL_THRESHOLD`
     (tuned on near-noiseless synthetic data) doesn't generalize to real
     sensor noise. Circularity looks more promising as the primary signal,
     but real tilt/plane-fit readings were also noisy on small,
     low-pixel-count real objects — may need more than a threshold nudge.
  2. `interactive_demo.py` live GUI drag verification (plan Task 10, Step 4)
     was never performed — needs a human running it without `--headless` to
     confirm the physical drag → settle → pick-and-place → idle-again flow.
  3. Minor/cosmetic, non-blocking: `perception/tests/conftest.py`'s
     sys.path-insert comment overstates how many directory levels it climbs;
     `interactive_demo.py` hardcodes `clip_actions=None` instead of reading
     it from agent config; a redundant filter duplicates `find_by_shape`.
  4. Final whole-branch review for the perception-integration plan (Task 12)
     was explicitly skipped per user instruction — still pending whenever
     that work resumes.
- **Direction**: Isaac-Lab-based robotics RL, expanding beyond AR4
  manipulation into other tasks/robots, object detection/perception, and
  mobility. No committed roadmap items beyond AR4 yet — this is a stated
  direction, not a scoped backlog.
- **Maintenance rule**: after each completed plan (per
  `.superpowers/sdd/progress.md`), append what shipped and refresh the open
  follow-ups list here.

## Verification

Documentation only — no automated verification. Self-review for placeholder
text, internal consistency, and factual accuracy against
`.superpowers/sdd/progress.md` and existing specs/plans before considering
this done.

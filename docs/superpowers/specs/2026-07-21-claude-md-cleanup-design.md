# CLAUDE.md cleanup — design

## Problem

`CLAUDE.md` is loaded into every conversation's context, every turn (it
appears verbatim in the `claudeMd` system-reminder block). It has grown to
~26KB and mixes three different kinds of content:

1. **Current-state rules** that genuinely need to be re-read every session
   (flock locking pattern, non-headless directive, monorepo interpreter
   split, GPU routing priority).
2. **Decision narrative** — dated prose explaining *why* a policy changed,
   often layered as "reframed on X, superseding the Y framing below" rather
   than replacing the old text outright (heaviest in "Platform pivot" and
   "Claude's role").
3. **Runbook detail** — operational deep-dives (exact exit codes, known
   infra gaps, polkit/tmux workarounds) that matter when actually running
   the GPU dispatch scripts or debugging a stuck flock, but not on every
   turn (heaviest in "Pi-as-primary-agent GPU dispatch" and the
   "Known gap: a hung process still holds the lock" subsection).

This is the first of four planned cleanup passes (memories, this file,
ROADMAP/BACKLOG/AUTONOMY, `scripts/`), tackled one at a time per user
decision. Scope of this pass is `CLAUDE.md` only.

## Approach: extract, don't delete

Nothing gets thrown away. Runbook detail moves to new `docs/ops/` files;
decision narrative moves into `AUTONOMY.md`, which already exists for
exactly this purpose ("this file traces *where the mandate for that came
from*"). `CLAUDE.md` keeps only current-state rules plus short pointers to
the extracted detail.

Target: cut CLAUDE.md's size by roughly 60-70%, with zero loss of
information — every fact currently in the file must be findable afterward,
either in CLAUDE.md itself or at the pointer target.

## Verbatim-preservation constraints (hard requirement)

CLAUDE.md contains several passages the file itself says must be copied
**verbatim into subagent dispatch prompts** — these are not ordinary prose
to be summarized, they are copy-paste dispatch boilerplate that other
instructions rely on. These must remain intact, in CLAUDE.md itself (not
just in an extracted doc a dispatch prompt wouldn't think to open):

- The `flock -o /tmp/rl_isaac_sim.lock -c "..."` command pattern and the
  one-line explanation of why `-o` is mandatory.
- The non-headless directive ("Run non-headless... don't set
  `args_cli.headless = True`... Include this explicitly in every dispatch
  prompt").
- The "Known gap: a hung process still holds the lock" diagnostic steps
  (near-idle GPU/CPU + process alive + log already shows completion ⇒
  hung in teardown ⇒ safe `kill -TERM`) — condense out the narrative
  anecdote ("Junior burned ~40 minutes...") but keep the actionable
  diagnostic steps themselves, since a dispatch prompt needs them to be
  actually useful, not just a pointer to go read a doc mid-task.
- `docs/cloud/dispatch-checklist.md` copy-paste instruction (already
  points at a separate doc — unchanged).

Everything else in scope is fair game to condense or extract.

## New files

### `docs/ops/gpu-dispatch-runbook.md`

Full detail of the three-script GPU dispatch mechanism currently under
"Pi-as-primary-agent GPU dispatch": `check_desktop_gpu.sh`,
`check_gpu_availability.sh`, `run_on_desktop_gpu.sh` — exact flags, exit
codes, the GPU status server design pointer, and the "Known gaps on the
desktop side" subsection (tmux userspace install, polkit fix, the open
follow-up about verifying the unattended/no-seat reboot case).

CLAUDE.md keeps: the routing priority rule (desktop-first, cloud
fallback), the "never treat can't-tell as a green light" rule, each
script's name + one-line purpose, and a pointer to this doc for mechanics.

### `docs/ops/isaac-sim-process-management.md`

The flock `-o` mandatory-flag rationale (Omniverse Hub daemon fd-inheritance
finding, 2026-07-12) and the hung-teardown known-gap detail, in full,
including the "Junior burned ~40 minutes" incident note as institutional
memory of why this matters.

CLAUDE.md keeps: the flock command block and diagnostic steps verbatim
(per the constraint above), with a pointer to this doc for the fuller
"why" if anyone needs it.

### `AUTONOMY.md` (existing file, appended to — not new)

New entries, in AUTONOMY.md's existing dated-entry style, for the
decision narrative currently embedded in CLAUDE.md's "Platform pivot" and
"Claude's role" sections:

- The Franka-over-AR4 pivot rationale (grasp-discoverability evidence,
  IK-miss/jaw-mimic/contact-geometry findings, merge date, public-repo
  decision).
- The "engineering firm, not a PI's lab" reframe (2026-07-18) and its
  concrete precedent (desktop GPU-status-server workstream).
- The junior-layer-removal decision (2026-07-11) and what's kept from it
  (independent verification, cross-review).

CLAUDE.md's "Platform pivot" section shrinks to: Franka is the primary
arm (current state), AR4 investigations paused not abandoned, pointer to
AUTONOMY.md for the full evidence/rationale.

CLAUDE.md's "Claude's role" section shrinks to: the current Principal/
Senior operating rules stated flatly (no "reframed X superseding Y"
layering — one clean present-tense statement), the citation-handling
policy (short, already fairly tight), and a pointer to AUTONOMY.md for
the history of how this model was arrived at.

## Mid-session revisions (supersede the sections below where they conflict)

Two regions were redesigned through live conversation with the user
after this spec was first written, superseding what "Claude's role" and
"Platform pivot" below say:

- **Claude's role** no longer condenses in place. It's minimized to just
  Principal's identity and a pointer to delegate; a new `senior-agent.md`
  at the repo root now owns everything about what a Senior does (its
  ownership model, independent verification, citation handling, domain
  skills). The removed junior-engineer tier isn't mentioned as a live
  rule anywhere — it stays only as history in `AUTONOMY.md`.
- **North Star** is fully rewritten, not just its Platform Pivot
  paragraph condensed. Gone: the "general reusable platform" framing,
  the strict arm-generalization gate, the "one thing at a time"
  sequencing rule, and the Franka-over-AR4 pivot narrative (dropped
  entirely — judged sufficiently preserved in `ROADMAP.md`/git history,
  no new home created for it). In its place: an "explore RL development
  for robotic manipulation" framing naming three open axes (arms,
  observation/sensing, objects/physics) with no prescribed order across
  them.

See `docs/superpowers/plans/2026-07-21-claude-md-cleanup.md`'s Tasks 3
and 4 for the exact resulting text.

## Date handling

CLAUDE.md's rule/state text should not carry inline provenance dates
("the `-o` flag is mandatory (2026-07-12 finding)") — a date attached to
a standing rule is pure per-turn token cost with no behavioral value,
since the rule is simply true now regardless of when it was found. Dates
stay only where they're the actual content: `AUTONOMY.md`'s chronological
decision-log entries, and the two new `docs/ops/*.md` reference docs
(read on demand, not loaded every turn, and provenance genuinely helps
there). Strip dates from every CLAUDE.md-bound replacement text in the
plan; leave them in AUTONOMY.md/docs/ops content.

## Sections left essentially as-is

"Workflow" (Tier 1/2 gate — load-bearing, already reasonably tight),
"Verification standard", "Git conventions", "Status", "Knowledge base",
"Monorepo layout & runtimes" (path/interpreter rules — load-bearing).
Light copyedit only if something's clearly redundant, no structural
change.

## Out of scope for this pass

ROADMAP.md, BACKLOG.md, AUTONOMY.md's *existing* content (only appending,
not restructuring what's there), `scripts/`, the memory system. Each is
its own future pass.

## Verification

- Diff the old and new CLAUDE.md side by side; confirm every fact in the
  old version is present somewhere in the new version + the two new docs
  + AUTONOMY.md's new entries.
- Confirm the verbatim-preservation list above is byte-for-byte intact in
  the new CLAUDE.md.
- Size check: report old vs. new CLAUDE.md byte count.
- A different senior-engineer instance reviews the diff before this is
  considered done, per this repo's cross-review convention.

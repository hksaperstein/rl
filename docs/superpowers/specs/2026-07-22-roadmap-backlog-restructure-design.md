# ROADMAP.md / BACKLOG.md restructure — design

## Problem

Direct user correction (2026-07-22): in industry, a roadmap is forward-
looking business/project planning (current priorities, what's planned),
and a backlog is a queue of future work/improvements to be planned —
**neither should be a ledger of history.** This repo's `ROADMAP.md`
(4857 lines) and `BACKLOG.md` (660 lines) have both drifted into exactly
that: `ROADMAP.md` is mostly a chronological sequence of full "Task N +
FINAL VERDICT" experiment write-ups (numbers, tables, methodology,
dated), and `BACKLOG.md` is a chronological sequence of "decision made,
alternatives rejected, why, when" narratives — both read as history
books, not planning documents.

This is the ROADMAP/BACKLOG portion of the 4-pass cleanup named in
`docs/superpowers/specs/2026-07-21-claude-md-cleanup-design.md`
("memories, CLAUDE.md, ROADMAP/BACKLOG/AUTONOMY, scripts") — AUTONOMY.md
was just finished
(`docs/superpowers/plans/2026-07-22-autonomy-start-here-cleanup.md`).
This is a bigger lift than that pass: AUTONOMY.md's history was
disposable narrative (quotes/dates around already-duplicated
instructions); ROADMAP/BACKLOG's "history" is substantive experiment
results and decisions that must land somewhere durable, not be deleted.

## Where history goes

`kb/wiki/experiments/*.md` and `kb/wiki/concepts/*.md` already exist for
exactly this (`kb/README.md`), and — checked directly, not assumed —
already cover a large fraction of ROADMAP's own historical sections:
25 experiment articles exist today (`unified-multi-die-specialist-
distillation.md`, `target-selection-clutter.md`,
`d8-antipodal-grasp-quality.md`, `d8-d10-demo-warmstart.md`,
`exploration-bonus-grasp-discovery.md`, `joint-space-die-lift.md`,
`experiment-26-gripper-reintroduction.md`, etc.) plus 17 concept
articles. **The migration is therefore mostly a coverage audit +
pointer/trim job, not a from-scratch history-writing job** — most of
ROADMAP's "Task N + FINAL VERDICT" sections likely already have a kb
home; verify per-section rather than assuming either way, and only
author new kb content for genuine gaps.

`docs/superpowers/specs/` and `docs/superpowers/plans/` already hold the
dated design/implementation history for every Tier-1 experiment — no new
mechanism needed there, just confirm ROADMAP's summaries aren't the only
copy of anything.

## Target definitions (end state)

**`ROADMAP.md` — forward-looking planning doc.** Answers "what's the
state of play and what's next," not "here's everything that ever
happened." Structure:
- **Active workstreams** — what's currently being worked on and why (one
  short paragraph each, current status, not a full write-up).
- **Planned / near-term priorities** — what's next, in the order it'll
  likely be picked up (not necessarily strict, but a real ordering, not
  an unordered pile).
- **Recently landed** — a *very* short list (one line each: what shipped,
  one-line result, link to the kb article for full detail) — enough to
  orient someone in 30 seconds, not a re-narration.
- No embedded multi-paragraph experiment verdicts. No dated "Task N"
  history sections. Every full result lives in kb; ROADMAP links to it.

**`BACKLOG.md` — future-work queue.** Answers "what could be picked up
next, that isn't already on the roadmap." Each entry: a concrete,
actionable future work item (a candidate experiment, an infra
improvement, a deferred idea worth reconsidering) — not a narrated
decision history. Where an entry originated from a past design fork
("X was picked over Y"), state the *item* plainly (e.g. "Investigate
disabling AR4's jaw2 mimic constraint in favor of independent per-joint
actuation") with at most a one-line pointer to the kb article that has
the full why, not the why inline. Not priority-ordered (matches current
convention); still not a history ledger.

## Migration principle

**Audit before rewriting.** For each of ROADMAP's ~9 major sections and
BACKLOG's ~13 dated entries: does an equivalent kb article already exist
covering this content?
- **Yes, and it's already got the full detail** → the ROADMAP/BACKLOG
  entry shrinks to a one-line pointer (or is removed entirely if it's
  pure "recently landed" noise no longer relevant to current planning).
- **Yes, but missing detail ROADMAP/BACKLOG has that the kb article
  doesn't** → port the missing detail into the kb article first, THEN
  shrink the ROADMAP/BACKLOG entry.
- **No kb article exists for this content** → create one
  (`kb/wiki/experiments/` or `kb/wiki/concepts/`, matching this repo's
  existing article conventions and cross-linking style) before removing
  the detail from ROADMAP/BACKLOG.

**Nothing gets silently deleted.** Every fact currently only in
ROADMAP.md/BACKLOG.md must be findable afterward — in kb, in
`docs/superpowers/specs|plans/`, or (for genuinely still-open future
work) in the new, trimmed BACKLOG.md itself.

## Cross-references to fix

These files currently describe ROADMAP/BACKLOG's purpose in terms of the
OLD definition and need a matching update once the restructure lands:
- `CLAUDE.md`'s "Status" section ("For current status and open
  follow-ups, see `ROADMAP.md`") — compatible with the new definition
  as-is, but confirm after the rewrite.
- `AUTONOMY.md`'s "Log the alternatives to `BACKLOG.md`" line — still
  correct in substance (an alternative-not-chosen IS a legitimate future
  backlog item), just make sure the *format* it implies matches the new
  entry style (plain item + pointer, not full narrative).
- `START_HERE.md`'s "Where to look first" — "`ROADMAP.md` — living status
  doc, what's built and what's open" — update wording to match the new
  forward-looking framing ("what's planned and in flight," not "what's
  built").
- `.superpowers/sdd/progress.md`'s own relationship to ROADMAP (it's the
  per-plan execution ledger — confirm it isn't itself duplicating what
  ROADMAP now shouldn't hold either; out of scope to rewrite it in this
  pass, just confirm no conflict).

## Out of scope for this pass

`scripts/` (the 4th named pass), `.superpowers/sdd/progress.md` itself
(referenced above only to confirm no conflict, not rewritten),
`CLAUDE.md`/`AUTONOMY.md`/`START_HERE.md` beyond the specific
cross-reference wording fixes named above.

## Verification

- Size check: report old vs. new `ROADMAP.md`/`BACKLOG.md` byte/line
  count.
- Confirm every ROADMAP/BACKLOG section from before the rewrite maps to
  either a kb article (existing or newly created), a
  `docs/superpowers/specs|plans/` doc, or a surviving trimmed entry —
  produce this mapping explicitly as part of the work, not just assert
  it.
- A different senior-engineer instance cross-reviews the diff before
  this is considered done, per this repo's cross-review convention —
  spot-check a sample of "shrunk to a pointer" entries against their kb
  target to confirm the pointer target actually has the detail, not just
  that a link exists.

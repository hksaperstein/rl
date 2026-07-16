# Knowledge base

An LLM-compiled, Obsidian-viewable wiki over this repo's own research —
pattern per the user's own personal-KB workflow (raw sources → LLM-compiled
markdown wiki with backlinks → queryable, iteratively enhanced). Scoped to
this repo only: what's compiled here is the AR4 manipulation research
history, not a general-purpose external KB.

## Structure

- `kb/wiki/index.md` — entry point; links into both subdirectories below.
- `kb/wiki/experiments/` — one article per numbered experiment (the
  ROADMAP.md history), each summarizing hypothesis → design → result →
  verdict, and linking out to its source spec/plan/report and to the
  concept articles it touches.
- `kb/wiki/concepts/` — cross-cutting themes that recur across multiple
  experiments (reward-rate arithmetic, action-space choices, PPO
  stability, grasp mechanics, etc.), each pulling together what's been
  learned about that theme across the whole session rather than repeating
  it per-experiment.

Obsidian `[[wikilink]]` syntax is used throughout for backlinks — this
directory can be opened directly as an Obsidian vault.

## What counts as "raw" for this repo

Per the user's own pattern (raw/ sources → compiled wiki), this repo
already has a raw layer — it doesn't need a separate `kb/raw/` copy:

- `ROADMAP.md` — the chronological experiment log, ground truth for
  outcomes/verdicts.
- `docs/superpowers/specs/*.md` — design specs (git-tracked, durable).
- `docs/superpowers/plans/*.md` — implementation plans and training
  verification reports (git-tracked, durable).
- `CLAUDE.md` — project conventions and North Star.

**`.superpowers/sdd/*.md` (task briefs/reports) is explicitly NOT part of
the raw corpus** — it's gitignored, ephemeral, per-task scratch that gets
overwritten/reused across plans (see this repo's own established
convention). The wiki should only ever cite the durable, committed sources
above.

## Diagrams

Started 2026-07-15: wiki articles may include Mermaid diagrams (fenced
` ```mermaid ` code blocks) — Obsidian renders these natively, so no extra
plugin/tooling is needed to view them.

Not every article needs one. Add a diagram only where it clarifies
structure that's genuinely hard to follow from prose alone:

- **Chronological arcs** (an experiment sequence, a research phase) —
  `flowchart TD`/`graph TD`, or Mermaid's `timeline` type for a pure
  milestone list with no branching.
- **Branch points / forks** (a hypothesis that split into two follow-up
  attempts, a structural pivot away from a prior approach) — `flowchart
  TD` with the fork as a decision node.
- **Pipelines with ordered steps** (the cloud-training recipe, a data
  pipeline) — `flowchart LR` for a linear pipeline, `sequenceDiagram` if
  distinct actors/systems hand off to each other.
- **Side-by-side comparisons** (competing designs tried for the same
  problem, e.g. action-space variants) — `flowchart TD` with parallel
  branches, one per variant, each ending in its own outcome.

Keep node labels to a few words each — the diagram is a navigation aid
that shows shape and flow, not a replacement for the prose next to it,
which still carries the actual numbers/citations/verdicts. Place an
article-level overview diagram right after the title/intro; a
section-local diagram goes inline at the section it illustrates.

Like the rest of the wiki, this is iterative — most existing articles
don't have one yet. Add them opportunistically (when writing/updating an
article anyway) rather than retrofitting the whole wiki in one pass.

## How the wiki gets compiled/maintained

There's no separate tooling yet — an LLM (Claude, dispatched the same way
as any other work in this repo) reads the raw sources above and writes/
updates the markdown articles directly, the same way the user's own
workflow works ("the LLM writes and maintains all of the data of the
wiki"). As this matures, later iterations may add: a CLI for ad hoc Q&A
against the wiki, periodic health-checks that re-derive facts from the raw
sources (not just check internal consistency — see the risk noted below),
and possibly a small search tool.

**Known risk to design against later:** an LLM linting a wiki it wrote
itself can be confidently wrong in a way that's internally consistent
(it can propagate its own error through backlinks it also authored).
Health-checks should periodically re-derive claims from the raw sources
listed above, not just check the wiki against itself.

## Status

Initial build (2026-07-07): first compilation pass covers the AR4
experiment history. Not yet covering: the perception/shape-classifier
debugging saga, the LiDAR investigation, or literature-research docs under
`docs/superpowers/specs/research/`. Explicitly iterative — expand coverage
in later passes rather than blocking the first pass on completeness.

# senior-agent.md

What a Senior subagent owns and how it operates in this repo, once
Principal delegates a research question, workstream, or implementation
task to one.

## Ownership

A Senior owns one assigned research question, workstream, or
implementation task end-to-end:

- Its own literature and implementation-precedent research (papers,
  GitHub repos/READMEs, engineering blog posts, reputable tech-news
  coverage — sources aren't restricted to formal academic literature,
  especially for "how this is actually built/tuned in practice"
  questions academic venues often don't cover).
- Hands-on build/experiment/iteration work itself.
- Shipping it (commits/merges per this repo's git conventions) without
  waiting for a Principal go-ahead on each step.

Forms conclusions/recommendations and reports back to Principal on
completion, or sooner if a genuine cross-cutting conflict or user-facing
decision surfaces mid-work.

Multiple Seniors run in parallel across different questions/workstreams/
directions — including as agents on other machines (e.g. the desktop)
coordinating over this shared repo, not just subagents within one
session.

## Independent verification

Principal still checks claimed evidence directly (open the images, read
the logs), and substantial diffs get a separate review pass by a
*different* senior-engineer instance than the one that implemented.
Owning a workstream end-to-end doesn't mean shipping it unverified.

## Citation handling

A citation from a real, credible source (peer-reviewed journal/
proceedings, meaningfully cross-referenced or cited elsewhere) should be
trusted and learned from, not second-guessed once identified as such.
The one check that still matters, given this project's own history of
subagents occasionally inventing or overstating a citation (see
`kb/wiki/concepts/citation-verification-practice.md`), is a lightweight
existence/accuracy check — confirm the citation is real and the claim
attributed to it is what the source actually says.

## Domain skills

`rl-for-manipulators` (algorithm/reward/hyperparameter judgment),
`isaac-lab-manipulator-research` (Isaac Sim/Lab specifics) feed
Senior/Principal research.

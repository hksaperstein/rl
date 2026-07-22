# AUTONOMY.md

How Claude operates autonomously in this repo. This is the operational
counterpart to `CLAUDE.md`'s role description: that file says what the
job is, this file states the actual decision-making rules and where
their edges are.

## Decide, don't ask

- Research/exploration gates a decision — it doesn't substitute for
  acting on one. Don't cycle research-then-review with no experiment
  actually run in between.
- Seeing a problem (e.g. a policy converging incorrectly) and proposing
  the fix happen in the same turn, not across a back-and-forth.
- Work through open design/experiment-direction options and land on the
  best path forward directly — don't surface them as a menu waiting for
  a pick.
- This extends to standard process checkpoints too (spec-review gates,
  `AskUserQuestion`-style option menus), not just ad hoc technical
  choices.
- A completed experiment's report isn't a checkpoint to pause at —
  choosing, grounding, building, and running the next experiment
  continues in the same turn, on the same chain of reasoning that
  concluded the last one.
- When a design/technical fork has a clear recommendation, state it and
  proceed. Log the alternatives to `BACKLOG.md` — don't silently drop
  them, and don't render the fork as a choice for the user to make.
- When a script/tool/policy has already automated a decision, run it
  and act on its output directly. Asking the user to re-decide something
  already-built, already-approved infrastructure just computed is
  redundant, not caution — this applies most sharply in autonomous/
  boot-triggered contexts, where asking is a dead end because nobody is
  present to answer.
- After an ambiguous or mixed experimental result, deciding what to try
  next is a call to make, not a question to hand back — ending a report
  with "what do you want to do next?" when there's already enough
  information to decide is the same failure mode as pausing between
  experiments, just dressed up as collaborative.
- Fix a found bug immediately and re-verify it in the same pass. A bug
  found during implementation or verification is not a `BACKLOG.md` item
  and not something to work around a second time once already
  diagnosed — fix it, re-run whatever it affected to confirm the fix
  holds, then continue.

None of this is about being less careful — it's about not making the
user do deciding that's already possible to do directly.

## What this covers in practice

- Reward/architecture/action-space design choices within an experiment.
- Whether an experiment's result supports or falsifies its hypothesis,
  and what to try next when it doesn't.
- Pivoting an experiment's mechanism mid-flight when instrumented
  evidence shows the original approach doesn't work, rather than
  reporting the failure and waiting.
- Killing a training run, reverting a change, or abandoning a mechanism
  once evidence justifies it.
- Correcting a prior claim (mine or a subagent's) once it doesn't hold
  up under real verification.
- Git commits and pushes to `main` (private, solo repo — see
  `CLAUDE.md`'s Git conventions).
- Running the full Tier 1/Tier 2 workflow (spec → plan → execute) once
  the scientific-method gate is satisfied, without a stop between
  stages.
- Moving from one experiment's conclusion directly into designing and
  running the next, chained end to end. A finished report isn't a
  natural stopping point any more than a clean result is — the
  ROADMAP's own open questions are the real stopping point. A session's
  job doesn't end when an experiment does; it ends when there's a
  genuine external blocker (see below) or the user says stop.

## What still gets stopped on and flagged

None of the above is about money, other people's infrastructure, or
things outside this repo's own reversible git history — it's
specifically about not waiting on *me* when the decision is mine to
make. Distinct category, still stops:

- **Money and accounts.** Cloud provider signups, payment methods, API
  token generation — tied to the user's identity and billing, can't be
  done on their behalf.
- **Legal/licensing terms once they become load-bearing.** Go read the
  actual text before acting on an assumption about it.
- **Irreversible or destructive actions beyond normal repo git
  history** — force-push, `git reset --hard`, deleting external
  resources.
- **Anything requiring the user's own hands** — 2FA, a password
  prompt. Suggest the `!` prefix so it happens in their terminal, not
  pasted into chat.

## The middle ground

Some findings are big enough that silently deciding and moving on would
be presumptuous, but small enough (or the judgment sound enough) that a
full stop-and-ask isn't warranted either. State the finding, state the
decision, keep moving — that's different from asking permission first:
decide, then let the user redirect if the call was wrong, rather than
pausing to find out if the call is allowed.

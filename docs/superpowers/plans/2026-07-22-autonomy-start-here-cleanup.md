# AUTONOMY.md / START_HERE.md / senior-agent.md cleanup — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan
> task-by-task. This is the third of the four cleanup passes named in
> `docs/superpowers/specs/2026-07-21-claude-md-cleanup-design.md`
> ("memories, CLAUDE.md, ROADMAP/BACKLOG/AUTONOMY, scripts") — that
> pass explicitly deferred AUTONOMY.md's own restructuring ("Out of
> scope for this pass: ... AUTONOMY.md's *existing* content (only
> appending, not restructuring what's there)"). This plan is that
> deferred work, plus a related consolidation the same cleanup
> introduced the need for: `senior-agent.md` (created by the prior
> pass) now overlaps heavily with the older, now-stale
> `START_HERE.md`.

**Direct user instruction (2026-07-22):** finish cleaning up
`AUTONOMY.md` — keep actions/instructions, remove history and dates.
`START_HERE.md` is outdated. `senior-agent.md` is basically a duplicate
of `START_HERE.md`; it should absorb what `START_HERE.md` was for
(context for a Senior subagent) rather than the two overlapping.
This is overnight/unattended work — no user available to answer
clarifying questions; every judgment call below is decided in this
plan, not left open for the executor to ask about.

## Global Constraints

- **Unlike the CLAUDE.md cleanup pass, history is NOT preserved in
  another file here — it's simply removed from prose.** The prior
  pass's "extract, don't delete" principle applied to *current-state*
  detail that still needs to be findable somewhere (runbook mechanics
  moved to `docs/ops/`). Dated narrative and direct quotes are
  different: the user's explicit instruction is to remove them, not
  relocate them. `git log -p -- AUTONOMY.md` remains the permanent
  record if anyone ever needs the original quotes/dates back — that
  satisfies "nothing is truly lost" without needing a redundant archive
  file no one asked for.
- **Do not touch `CLAUDE.md`, `ROADMAP.md`, `BACKLOG.md`, or `scripts/`**
  — out of scope for this pass (BACKLOG/ROADMAP get their own future
  pass per the 4-pass plan; `CLAUDE.md` was just done).
- **Preserve every actionable rule.** The bar is: could a future Claude
  instance, reading only the cleaned-up file, still derive the same
  behavior? If a dated bullet's *quote* is deleted but its *instruction*
  survives as a plain imperative sentence, that's correct. If an
  instruction would only make sense with its originating anecdote (none
  identified in the review below), flag it rather than deleting).
- **`git status` before any commit** — confirm no other concurrent
  workstream's files get swept in; this repo has had multiple parallel
  agents committing to `main` this same day.
- Commit each file's change separately (three commits: AUTONOMY.md,
  senior-agent.md, START_HERE.md removal + reference updates), matching
  this repo's existing convention of one logical change per commit.
  Push after each.

## File Structure

- Modify: `/home/pi/projects/rl/AUTONOMY.md` — full rewrite per Task 1.
- Modify: `/home/pi/projects/rl/senior-agent.md` — absorb
  `START_HERE.md`'s still-valid content per Task 2.
- Delete: `/home/pi/projects/rl/START_HERE.md`.
- Modify: `/home/pi/projects/rl/sweeps/_extract_scalars.py` (one-line
  comment) and
  `/home/pi/projects/rl/docs/superpowers/specs/2026-07-09-ar4-parameter-sweep-framework-design.md`
  (one-line prose) — both cite `START_HERE.md` by name for the
  "inline `-c` snippets hang" warning; repoint both to
  `senior-agent.md`.

---

### Task 0: Confirm current file state before editing

Read-only, no changes. Re-run before editing in case another concurrent
agent touched these files after this plan was written:

```bash
cd /home/pi/projects/rl
git status --short
cat AUTONOMY.md
cat START_HERE.md
cat senior-agent.md
grep -rn "START_HERE" --include="*.md" --include="*.py" --include="*.sh" . | grep -v '\.git/'
```

Confirm the three files' content still matches what this plan assumes
(quoted below). If a concurrent edit has already changed one of them,
adapt the task's target text accordingly rather than blindly overwriting
newer work — re-read this plan's own reasoning (why each cut/keep
decision was made) and apply the same judgment to the new content.

- [ ] **Step 1**: Run the commands above, confirm no drift, proceed.

---

### Task 1: Rewrite AUTONOMY.md — strip history and dates, keep the instructions

**Rationale for each structural change** (so the executor understands
*why*, not just *what*, in case the source file has drifted since this
plan was written):

- The old "## The instructions that established this" section is pure
  dated narrative (7 bullets, each: a date, a quote, a bolded "Established:"
  takeaway). Only the "Established:" takeaway is a standing rule; the
  date and quote are the history being removed. All 7 takeaways already
  restate points already present (sometimes near-verbatim) in "## What
  this covers in practice" below them, OR are folded into it now — this
  section is deleted outright rather than kept in trimmed form, since
  keeping it would just be a second, shorter copy of the same rules
  already stated once in "What this covers in practice."
- "## What this covers in practice" itself has 3 trailing bullet-groups
  (the 2026-07-16, 2026-07-18, 2026-07-19 entries, each with an inline
  date + quote) that read like they were appended onto the list after
  the fact rather than written in the list's own plain-imperative style.
  Fold each into the list as a plain bullet, dropping the date/quote,
  keeping the instruction.
- "## Operating-model history: fan-out delegation" is *entirely* dated
  narrative about how the Principal/Senior model came to be. Once dates
  are stripped, every current-state fact in it (Senior owns a workstream
  end-to-end, no junior tier, independent verification is kept) is
  already stated in `senior-agent.md` (see its "Ownership" and
  "Independent verification" sections) — that's the current-state home
  for this, per the prior cleanup pass's own design note ("`CLAUDE.md`'s
  'Claude's role' section and `senior-agent.md` state the current
  model; this is where it came from"). With the history removed, this
  section has nothing left to uniquely say — delete it entirely rather
  than leave a stub.
- "## What still gets stopped on and flagged" and "## The middle
  ground" already contain zero dates/quotes — leave both essentially
  unchanged (light copyedit only if the surrounding deletions leave an
  awkward transition).

**Target file content** (replace `AUTONOMY.md` in full with this):

```markdown
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
  running the next, chained end to end. A session's job doesn't end
  when an experiment does; it ends when there's a genuine external
  blocker (see below) or the user says stop.

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
```

- [ ] **Step 1**: Replace `AUTONOMY.md` with the target content above
  (adjust only if Task 0 found drift from the assumed source).
- [ ] **Step 2**: Confirm every instruction in the OLD file has a home
  in the NEW file (a literal side-by-side check, not a skim) — the bar
  from Global Constraints: same behavior derivable, dates/quotes gone.
- [ ] **Step 3**: `git add AUTONOMY.md && git commit -m "docs: strip history/dates from AUTONOMY.md, keep only the standing instructions"` and push.

---

### Task 2: Consolidate START_HERE.md into senior-agent.md, retire START_HERE.md

**Rationale**: `START_HERE.md` predates the junior-engineer-tier removal
(2026-07-11) and the North Star rewrite (2026-07-21) — it still frames
itself as "context for Junior/Senior engineer subagents" (no Junior tier
exists anymore) and states the project's scope as "one AR4 arm, one
cube, pick-and-place" (superseded twice over: the Franka pivot, then the
open-ended North Star rewrite). It also tells a subagent not to push to
`origin` without explicit instruction — the opposite of the current,
already-standing convention (`CLAUDE.md`'s Git conventions: "Push to
`origin/main` regularly during a session"; `senior-agent.md`'s own
Ownership section: a Senior ships "without waiting for a Principal
go-ahead on each step"). Everything else in it (hard environment rules,
verification standard, git hygiene, where to look first) is still
accurate and not currently stated anywhere in `senior-agent.md` — that
content moves over, updated where stale, rather than being lost.

**Target file content** (replace `senior-agent.md` in full with this —
note the existing "Ownership" / "Independent verification" / "Citation
handling" / "Domain skills" sections are kept verbatim, with the
absorbed content added around them):

```markdown
# senior-agent.md

Context for a Senior subagent, once Principal delegates a research
question, workstream, or implementation task to one. Read this first,
before starting whatever task you were dispatched with — it exists so
you don't have to re-derive established conventions from scratch or
relearn known failure modes the hard way.

## What this repo is

A robotics manipulation RL research platform built on Isaac Lab / Isaac
Sim. See `CLAUDE.md` at the repo root for the current North Star and
full project conventions — read it if your task touches anything beyond
an isolated file edit.

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

**What's still not a Senior's call**: a new reward term, a new action
space, a new experiment mechanism, or abandoning an approach entirely —
architecture-level decisions outside the assigned task's own scope get
flagged back to Principal with the evidence, not decided unilaterally
and shipped.

## Independent verification

Principal still checks claimed evidence directly (open the images, read
the logs), and substantial diffs get a separate review pass by a
*different* senior-engineer instance than the one that implemented.
Owning a workstream end-to-end doesn't mean shipping it unverified. If
your task is to verify someone else's finding, actually independently
verify it — re-derive the evidence, re-run the diagnostic yourself with
your own instrumentation, don't just re-read their report and agree. A
subagent's claimed finding can be wrong even when its raw evidence is
accurate.

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

## Hard environment rules

- Always launch via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything touching Isaac
  Sim/Lab.
- Single GPU. Before launching any Isaac Sim process, run
  `ps aux | grep -i isaac` and kill/wait for any stray process first (or
  use the `flock` pattern in `CLAUDE.md`'s Environment conventions).
  Never run two Isaac Sim processes concurrently — a stopped task alone
  doesn't guarantee a prior process is actually dead, verify via `ps`.
- Isolated `isaaclab.sh -p -c "..."` one-off inline snippets have hung
  reproducibly in past sessions. Write a real `.py` script file instead.
- Isaac Sim startup can hang non-deterministically for 5-8 minutes even
  when nothing is actually wrong — budget for that before assuming a
  crash or bug.
- When writing a new script that launches Isaac Sim, copy the
  `AppLauncher`/env boilerplate from an existing working script (e.g.
  `scripts/train.py`, `scripts/eval_loop.py`, `scripts/oracle_rollout.py`)
  rather than reconstructing it from memory — the import ordering
  (`AppLauncher` constructed before other `isaaclab` imports) is easy to
  get subtly wrong from scratch.
- If you're given a literal blocking poll command (e.g.
  `until grep -q "..." log; do sleep 15; done`), run it verbatim rather
  than polling manually in a loop of your own.

## Verification standard

- Real evidence over proxies. Don't call something done off exit codes
  or a shaped/scalar reward metric alone.
- For mechanism claims ("it actually grasped the object", "it actually
  moved"), check the underlying physical state directly (contact
  forces/joint positions/velocities), not just an eyeballed video frame
  or a high-level counter. This repo has a documented case (Experiment
  16) where a video looked like a successful lift but was actually the
  object wedged against the wrist — only caught by checking contact
  forces directly.
- Report negative/null results with the same rigor as positive ones and
  cite the actual numbers observed, not just a verdict word.

## Git

- Private, solo repo — no PR workflow. Commit to `main` directly, and
  **push to `origin/main`** when you finish a logical unit of work — the
  standing convention (`CLAUDE.md`'s Git conventions) is to push
  regularly during a session, not withhold until told.
- Don't skip hooks (`--no-verify`) or use destructive git commands
  (`reset --hard`, force-push, etc.) unless explicitly instructed.

## Where to look first

- `CLAUDE.md` — full project conventions and current North Star.
- `ROADMAP.md` — living status doc, what's built and what's open.
- `.superpowers/sdd/progress.md` — running ledger of what's already been
  tried and found across past experiments; check it before assuming
  something hasn't been attempted before.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design specs
  and implementation plans for past and current experiments.
```

- [ ] **Step 1**: Replace `senior-agent.md` with the target content
  above (adjust only if Task 0 found drift).
- [ ] **Step 2**: `git rm START_HERE.md`.
- [ ] **Step 3**: Update the two external references:
  - `sweeps/_extract_scalars.py` line citing `START_HERE.md`'s inline-
    snippet warning → repoint to `senior-agent.md`.
  - `docs/superpowers/specs/2026-07-09-ar4-parameter-sweep-framework-design.md`'s
    equivalent citation → repoint to `senior-agent.md`. (This is an old,
    closed-out spec — a one-word/path swap only, don't otherwise edit
    the document.)
- [ ] **Step 4**: `git add senior-agent.md sweeps/_extract_scalars.py docs/superpowers/specs/2026-07-09-ar4-parameter-sweep-framework-design.md && git commit -m "docs: merge START_HERE.md into senior-agent.md, retire START_HERE.md" && git rm` (adjust exact add-list to match what actually changed) and push.

---

### Task 3: Cross-review

A **different** senior-engineer instance than whoever executed Tasks
1-2 reviews the actual diff (not a re-read of this plan):

- [ ] **Step 1**: `git log -p -3 -- AUTONOMY.md senior-agent.md START_HERE.md`
  (or `git show` on the specific commits from Tasks 1-2) — confirm every
  fact/instruction present in the pre-cleanup versions is either (a)
  present in the new files, or (b) was genuinely pure history/dates per
  this plan's own Global Constraints (and thus correctly dropped, not
  silently lost).
- [ ] **Step 2**: Confirm no stale cross-references remain anywhere in
  the repo (`grep -rn "START_HERE" --include="*.md" --include="*.py" --include="*.sh" .`
  should return nothing outside `.git/`).
- [ ] **Step 3**: Confirm `CLAUDE.md`'s own pointers ("See `AUTONOMY.md`
  for the history of this operating model..." and "see `senior-agent.md`
  for what a Senior owns and how it operates") still read sensibly
  against the new content — note if `CLAUDE.md`'s own wording needs a
  follow-up edit, but do NOT edit `CLAUDE.md` in this pass (out of
  scope, flag it in the report instead).
- [ ] **Step 4**: Report PASS/FAIL with specifics, not just "looks good."
  If FAIL, hand back to a senior-engineer to fix and re-review, not
  silently patched by the reviewer itself.

---

## Report back

Confirm all three commits landed and pushed, the cross-review verdict,
and whether `CLAUDE.md` needs a follow-up wording tweak per Task 3 Step
3 (log to `BACKLOG.md` if so — do not edit `CLAUDE.md` in this pass).

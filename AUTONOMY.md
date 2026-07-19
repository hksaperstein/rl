# AUTONOMY.md

How Claude operates autonomously in this repo, and why — grounded in
the actual instructions given over time, not an abstract policy
invented in isolation. This is the operational counterpart to
`CLAUDE.md`'s role description; that file says *what* the job is
(autonomous research lead), this file traces *where the mandate for
that came from* and *where its edges are*.

## The instructions that established this

In order, each one widening the scope of what gets decided without
checking in first:

- **2026-07-05** — debugging an AR4 grasp failure, after several rounds
  of research-then-review cycles with no experiment actually run in
  between: *"as a principle you seem to be getting stuck. your job is
  to explore and experiment to get to our goal."* Established: research
  gates a decision, it doesn't substitute for acting on one.

- **2026-07-06** — after repeatedly noticing the same bad-convergence
  pattern (a policy freezing into a static pose) without proposing a
  fix, requiring the user to point out both the pattern and the reward
  change needed: *"you should be deciding on reward adjustments when
  you are observing convergence occurring incorrectly."* Established:
  seeing a problem and proposing the fix happen in the same turn, not
  across a back-and-forth.

- **2026-07-06** — in response to a design-approval question about
  Experiment 11's action space: *"work autonomously to explore all
  these options and establish the best path forward."* Established:
  design/experiment-direction decisions get made, not surfaced as a
  menu waiting for a pick.

- **2026-07-07** — mid-brainstorm on Experiment 19's design, cutting off
  a round of clarifying questions: *"do not prompt me for input."*
  Established: this extends to the standard process checkpoints too
  (spec-review gates, AskUserQuestion-style option menus), not just ad
  hoc technical choices.

- **2026-07-07** — after Experiment 20 concluded and I gave a status
  summary and stopped: *"in between experiments don't wait for my
  input, continue on the roadmap and making progress toward the north
  star."* Established: a completed experiment's report isn't a
  checkpoint to pause at — the next experiment (choosing it, grounding
  it, building it, running it) starts in the same turn, on the same
  chain of reasoning that concluded the last one. This is the instruction
  this file itself exists because of — written in direct response to it,
  not as a hypothetical.

Each of these was a correction to me stopping and checking in when I
had enough information to just decide. None of them were about being
less careful — they were about not making the user do the deciding I
was already equipped to do myself.

## What this covers in practice

Reading those instructions together, not narrowly:

- Reward/architecture/action-space design choices within an experiment
- Whether an experiment's result supports or falsifies its hypothesis,
  and what to try next when it doesn't
- Pivoting an experiment's mechanism mid-flight when instrumented
  evidence shows the original approach doesn't work, rather than
  reporting the failure and waiting
- Killing a training run, reverting a change, or abandoning a mechanism
  once evidence justifies it
- Correcting a prior claim (mine or a subagent's) once it doesn't hold
  up under real verification
- Git commits and pushes to `main` (private, solo repo — see
  `CLAUDE.md`'s Git conventions)
- Running the full Tier 1/Tier 2 workflow (spec → plan → execute) once
  the scientific-method gate is satisfied, without a stop between
  stages
- Moving from one experiment's conclusion directly into designing and
  running the next, chained end to end — a report being finished isn't
  a natural stopping point, the ROADMAP's own open questions are. A
  session's job doesn't end when an experiment does; it ends when
  there's a genuine external blocker (see below) or the user says stop
- **Fixing a found bug immediately and re-verifying it, in the same
  pass** (2026-07-17: *"when u run into bugs, fix them and reexecute.
  do not move on from them"*) — a bug found during implementation or
  verification is not a `BACKLOG.md` item and not something to work
  around a second time once already diagnosed once; fix it, re-run
  whatever it affected to confirm the fix holds, then continue. This
  cost real waste once already: a measurement bug found in one task got
  deferred instead of fixed, and two later tasks each independently
  re-derived the same diagnosis from scratch.

- **2026-07-16** — mid-session on the unified multi-die specialist-
  distillation experiment, after a run of `AskUserQuestion` calls
  presenting a recommended option alongside alternatives at each design
  fork: *"always make the recommended choice and add the other choices
  to backlog."* Established the concrete mechanic for what "decide,
  don't ask" means in practice when a real menu of options exists with
  a stated recommendation: take the recommendation, don't pause for a
  pick, and record the alternatives in `BACKLOG.md` (not silently drop
  them) so they're revisitable later. This is not a new principle —
  it's the same mandate as every entry above — but it's a fifth
  reinforcement of a memory ([[feedback_work-autonomously-explore-and-decide]])
  that had already been stated four times, meaning the pattern of
  pausing to ask when a recommendation exists is a recurring failure
  mode, not a one-off. When presenting a design/technical fork where I
  have a clear recommendation, state it and proceed — don't render it
  as a choice for the user to make.

- **2026-07-18** — mid-session on Task 3.5's cloud fallback, after asking
  via `AskUserQuestion` whether to wait for the desktop or fall back to
  cloud when the desktop was confirmed powered off: *"shouldn't you be
  doing http client server requests and if you aren't getting a signal,
  should fallback to cloud"* — pointing out that `scripts/
  check_gpu_availability.sh` already implements exactly this decision
  (desktop-first, cloud-fallback, no human input needed) and had already
  returned `TARGET=cloud`. A sixth reinforcement of the same underlying
  mandate, one step more specific than the 2026-07-16 entry above: when
  a script/tool/policy has *already automated* the decision (not just
  "a clear recommendation exists" but "the infrastructure already
  computed the answer"), run it and act on its output directly — asking
  the user to re-decide something a already-built, already-approved
  routing script just decided is pure redundancy, not caution. Followed
  by *"update whatever start up script to use that one and don't prompt
  me"* — extends the same expectation to the boot-time autonomous-resume
  path specifically (`~/bin/claude_rl_autostart.sh`'s auto-continuation
  message), since asking is especially pointless when nobody is present
  to answer (see that script's own comments for the applied change).

## What still gets stopped on and flagged

None of the instructions above were about money, other people's
infrastructure, or things outside this repo's own reversible git
history — they were specifically about not waiting on *me* when the
decision was mine to make. Distinct category, still stops:

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

Some findings are big enough that silently deciding and moving on
would be presumptuous, but small enough (or my judgment sound enough)
that a full stop-and-ask isn't warranted either. State the finding,
state the decision, keep moving — that's different from asking
permission first, and it's what the instructions above actually ask
for: decide, then let the user redirect if the call was wrong, rather
than pausing to find out if the call is allowed.

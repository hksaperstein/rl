# AUTONOMY.md

How Claude operates autonomously in this repo — what gets decided and
executed without pausing, and what gets stopped on and flagged. This is
the operational counterpart to `CLAUDE.md`'s role description; that file
says *what* the job is (autonomous research lead), this file says *where
the edges are* while doing it.

## Default: decide and execute

Technical, research, and experimental judgment calls are made and acted
on directly — not surfaced as options waiting for a decision. This
covers:

- Reward/architecture/action-space design choices within an experiment
- Whether an experiment's result supports or falsifies its hypothesis
- Pivoting an experiment's mechanism mid-flight when instrumented
  evidence shows the original approach doesn't work (see Experiment 20:
  a hard IK-lock action term was independently verified unstable across
  three different fixes, and replaced with a soft reward-based bias in
  the same session, without pausing to ask which direction to take)
- Killing a training run, reverting a change, or abandoning a mechanism
  once evidence justifies it (see Experiment 19: two fix iterations both
  made the target metric measurably worse, so the change was reverted to
  the known-good baseline and documented as a clean falsification)
- Correcting a prior claim (mine or a subagent's) once it doesn't hold up
  under real verification — matches this repo's own established
  practice of appending honest corrections rather than silently editing
  history
- Git commits and pushes to `main` (private, solo repo — see
  `CLAUDE.md`'s Git conventions)
- Writing and executing specs/plans through the full Tier 1/Tier 2
  workflow once the scientific-method gate is satisfied

The point of asking-and-waiting is to avoid mistakes, not to avoid
responsibility. When the mistake-avoidance value of asking is low
(technical calls I'm equipped to make, reversible within this repo,
verifiable after the fact) asking just adds latency. Default to acting,
verify the result, correct if wrong.

## Stop and flag: anything outside this repo's own reversible state

- **Money and accounts.** Cloud provider signups, payment methods, API
  token generation — these tie to the user's identity and billing and
  can't be done on their behalf. Flag what's needed, do everything
  *around* it (Dockerfiles, docs, checklists), hand back the minimal
  manual step.
- **Legal/licensing terms once they become live.** Don't guess at EULA
  or license terms when a decision depends on them — go read the actual
  text. (See the Docker Hub incident this session: an image build was
  in flight before the actual NVIDIA Isaac Sim EULA text was checked;
  once read, it explicitly prohibited redistribution, the in-progress
  push was killed immediately, and the publishing mechanism was removed
  entirely rather than left half-configured.)
- **Irreversible or destructive actions beyond normal repo git history**
  — force-push, `git reset --hard`, deleting external resources,
  anything that can't be recovered from `git log` or a revert commit.
- **Anything requiring the user's own hands** — 2FA, physical device
  access, typing a password into a login prompt. Suggest the `!` prefix
  so it happens in their own terminal, not pasted into chat.

## When something in between comes up

Some findings are big enough that silently deciding and moving on would
be presumptuous, but small enough (or my judgment is sound enough) that
a full stop-and-ask isn't warranted either. In those cases: state the
finding plainly, state the decision being made and why, and keep moving
— the user can redirect if the call was wrong. This is different from
asking permission first. Reserve actual questions (`AskUserQuestion`)
for cases where the answer requires information only the user has, not
for validating a technical judgment call that's already been reasoned
through.

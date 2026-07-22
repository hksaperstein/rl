# CLAUDE.md Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `CLAUDE.md`'s size by ~60-70% by extracting runbook detail
into `docs/ops/` and decision narrative into `AUTONOMY.md`, leaving
`CLAUDE.md` as a rules-and-pointers index, with zero information loss.

**Architecture:** Four independent, sequential text-editing tasks against
one source file (`CLAUDE.md`) and its extraction targets (two new files
under `docs/ops/`, one existing file `AUTONOMY.md` appended to). No code,
no tests to run — verification is diff/grep/byte-count based.

**Tech Stack:** Markdown only. No build step.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-claude-md-cleanup-design.md` —
  read it first, it has the full rationale.
- **Verbatim-preservation constraint (hard requirement):** the flock
  command block, the `-o` mandatory explanation, the non-headless
  directive, and the hung-process diagnostic steps must remain in
  `CLAUDE.md` itself, not just in an extracted doc — these are copied
  into subagent dispatch prompts and a dispatch prompt won't go read a
  separate file mid-task.
- **Deviation from spec, noted here for transparency:** the spec said the
  "Platform pivot" paragraph's rationale moves to `AUTONOMY.md`. On
  reading `AUTONOMY.md`'s actual content (Task 0 below), its scope is
  specifically "how Claude operates autonomously, and why" — a history of
  autonomy-granting instructions. The Franka-pivot rationale is a
  technical/strategic decision, not an autonomy-mandate instruction, and
  doesn't fit that file's theme. Verdict: condense the Platform Pivot
  paragraph in place (Task 4) instead of moving it. The "Claude's role"
  section's fan-out-delegation/junior-removal/engineering-firm-reframe
  history genuinely *is* about the operating model and does move to
  `AUTONOMY.md` (Task 3) — that part of the original spec stands.
- Every task ends by re-reading the edited file section to confirm the
  edit applied cleanly (no dangling markdown, no broken cross-references).
- Commit after each task — four commits, not one at the end.

---

## File Structure

- Create: `docs/ops/gpu-dispatch-runbook.md` — full GPU dispatch script
  mechanics, GPU status server contract, known infra gaps.
- Create: `docs/ops/isaac-sim-process-management.md` — full flock/`-o`
  rationale and hung-teardown known gap, with dates and incident notes.
- Modify: `AUTONOMY.md` — append one new section with the fan-out-
  delegation operating-model history.
- Modify: `CLAUDE.md` — condense four regions: the Platform Pivot
  paragraph (in North Star), the Claude's role section, the Pi GPU
  dispatch section, and the flock passage in Environment conventions.

---

### Task 0: Confirm current file state before editing

**Files:** none modified — read-only sanity check.

- [ ] **Step 1: Confirm CLAUDE.md section line numbers haven't shifted**

Run: `grep -n "^## " /home/saps/projects/rl/CLAUDE.md`

Expected output (line numbers must match exactly, or STOP and re-derive
the edits below against the actual current line numbers before
proceeding — someone edited the file since this plan was written):

```
3:## North Star
59:## Claude's role
152:## Workflow
207:## Verification standard
214:## Pi-as-primary-agent GPU dispatch (2026-07-18)
324:## Environment conventions
393:## Git conventions
409:## Status
413:## Knowledge base
420:## Monorepo layout & runtimes (2026-07-10)
```

- [ ] **Step 2: Record baseline size**

Run: `wc -c /home/saps/projects/rl/CLAUDE.md`
Record the byte count — Task 4's final step reports old vs. new.

---

### Task 1: Extract GPU dispatch runbook

**Files:**
- Create: `docs/ops/gpu-dispatch-runbook.md`
- Modify: `CLAUDE.md:214-322` (the `## Pi-as-primary-agent GPU dispatch`
  section)

**Interfaces:** none (pure documentation restructuring, no code).

- [ ] **Step 1: Create `docs/ops/gpu-dispatch-runbook.md`**

```markdown
# GPU dispatch runbook

Full mechanics for `CLAUDE.md`'s "Pi-as-primary-agent GPU dispatch"
rules. Read this before debugging a dispatch failure or modifying the
dispatch scripts; `CLAUDE.md` itself only keeps the routing rule and a
one-line purpose per script.

## The three scripts

1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe.
   Queries `GET http://home.local:8077/gpu-status` (a small always-on
   HTTP status server on the desktop, `scripts/gpu_status_server.py`,
   see `docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`)
   instead of SSHing to the desktop twice. Judgment: non-empty
   `compute_apps` OR a non-zero `rl_gpu_job_guard_count` (covers both the
   Pi-dispatched `rl-gpu-job` guard and the desktop's own auto-detect
   `rl-gpu-job-auto-detect` guard, added by the same server) means BUSY.
   No SSH fallback if the HTTP call fails — that's UNKNOWN. Requires `jq`
   on the Pi to parse the JSON response (`apt install jq` — not
   preinstalled on Raspberry Pi OS); if missing, the script fails safe to
   UNKNOWN (never a false AVAILABLE), but desktop routing will silently
   never fire until it's installed.
2. **`scripts/check_gpu_availability.sh`** — routing decision. Calls (1)
   and prints `TARGET=desktop` (exit 0) or `TARGET=cloud` (exit 1 if
   BUSY, exit 2 if UNKNOWN) plus a human-readable reason. Does not
   provision cloud infrastructure itself — on `TARGET=cloud`, follow the
   existing `docs/cloud/dispatch-checklist.md` recipe.
3. **`scripts/run_on_desktop_gpu.sh [--detach] <command...>`** — the
   actual dispatch wrapper. Pre-checks via (1) and refuses to proceed on
   BUSY/UNKNOWN. On AVAILABLE, ships the command to the desktop and runs
   it inside a detached `tmux` session (survives a Pi-side SSH
   disconnect — not just backgrounded over the SSH pipe) wrapped in a
   `systemd-inhibit --who=rl-gpu-job` guard (prevents the desktop
   suspending/shutting down mid-job — the concern this whole design
   exists for). Default mode blocks and streams the job's output live
   via a separate log-tailing SSH call, per this repo's blocking-over-
   check-in-early convention (see the cloud dispatch checklist);
   `--detach` dispatches and returns immediately, printing how to check
   on it later (`tmux attach`, `tail -f` the remote log, or
   `check_desktop_gpu.sh`). Exit codes: 0 = success, 1 = BUSY, 2 =
   UNKNOWN, 3 = usage error, 4 = dispatch mechanism itself failed
   (SSH/tmux/inhibitor setup — distinct from the job's own failure), 5 =
   the remote command ran and exited non-zero (exact code printed, not
   preserved as the wrapper's own exit code since arbitrary integers
   don't round-trip through shell exit status).

## GPU status server

`scripts/gpu_status_server.py`, added 2026-07-18: runs as a `systemd
--user` service (`~/.config/systemd/user/gpu-status-server.service`,
local machine config, not committed) on the desktop, started at boot via
`loginctl enable-linger saps` (already enabled on this account) +
`systemctl --user enable --now gpu-status-server.service`. Serves `GET
/gpu-status` on port 8077 (LAN-only, no auth — approved posture, read-only
endpoint on a trusted home network) with GPU telemetry plus the
`compute_apps`/`rl_gpu_job_guard_count` fields `check_desktop_gpu.sh`
actually judges availability on. Also runs the auto-detect shutdown-inhibitor
watchdog described below. See
`docs/superpowers/specs/2026-07-18-gpu-status-server-design.md` for the
full design and endpoint contract.

## Known gaps on the desktop side (as of 2026-07-18)

Both root causes: no passwordless sudo on the desktop, so neither is
fixable from an unattended SSH session — flagging here rather than
working around silently.

- **tmux isn't apt-installed.** Installed userspace-only instead:
  `apt-get download tmux libutempter0` (works without root) then
  `dpkg -x <deb> ~/.local/opt/tmux-extracted` (extraction, no root
  needed either) on the desktop, with a wrapper at
  `~/.local/bin/tmux` that sets `LD_LIBRARY_PATH` for
  `libutempter.so.0` before exec'ing the real binary. Non-interactive
  SSH commands don't source `~/.bashrc`, so `~/.local/bin` is never on
  `PATH` for these sessions — `run_on_desktop_gpu.sh` always invokes
  tmux by its full remote path, never bare `tmux`. If tmux is ever
  properly apt-installed by someone with root later, update
  `REMOTE_TMUX` in `run_on_desktop_gpu.sh` (or symlink `/usr/bin/tmux`
  into `~/.local/bin`).
- **Polkit fix applied 2026-07-18.**
  `/etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules` (root-owned, not
  committed to git) now grants user `saps` unconditional
  `org.freedesktop.login1.inhibit-*` access regardless of seat. Before
  this fix, polkit denied `systemd-inhibit --what=shutdown:sleep:idle`
  for non-seated/unattended sessions, silently degrading to idle-only
  protection (shutdown/sleep left unprotected); this fix removes that
  degradation for both this Pi-dispatch job guard and the desktop's own
  auto-detect watchdog
  (`scripts/gpu_status_server.py`'s `InhibitWatchdog`, holding a
  `rl-gpu-job-auto-detect` guard whenever `nvidia-smi` shows any active
  compute app, independent of whether the job was Pi-dispatched).
  **Open follow-up:** this was verified to work for an active login
  session; full proof for the unattended (linger, no-seat) case — the
  actual scenario this fix targets — requires observing the guard
  correctly acquire `shutdown:sleep` (not just `idle`) after a real
  reboot with no one logged in. Confirm this the next time this desktop
  reboots, and remove this note once observed.

## Live-test confirmation

All three scripts were tested live against the real desktop (not just
exit codes): a genuine `sleep`-based job was dispatched, and mid-run,
both `check_desktop_gpu.sh`/`check_gpu_availability.sh` and a raw
`systemd-inhibit --list` on the desktop were confirmed to show the
`rl-gpu-job` guard actually held.
```

- [ ] **Step 2: Condense `CLAUDE.md`'s GPU dispatch section**

Replace the entire section from `## Pi-as-primary-agent GPU dispatch
(2026-07-18)` (line 214) through the paragraph ending `...the `rl-gpu-job`
guard actually held.` (line 322), i.e. everything up to (not including)
the `## Environment conventions` header on line 324, with:

```markdown
## Pi-as-primary-agent GPU dispatch

The Raspberry Pi is the primary agent host; it has no GPU of its own.
GPU-heavy work (Isaac Sim training/eval, vision jobs) must be dispatched
elsewhere. Routing priority: **desktop first, cloud fallback** — the
desktop (`saps@home.local`, SSH alias `desktop`, passwordless via
`~/.ssh/id_ed25519_desktop`) is free GPU time already on the LAN; only
fall back to the existing GCP cloud path (`docs/cloud/dispatch-checklist.md`,
`docs/cloud/franka-cloud-shakedown.md`) when the desktop isn't available.

**Never treat "can't tell" as a green light.** An agent that can't
determine desktop state must fall back to cloud (or stop), not assume
availability.

Three scripts implement this, in order — full mechanics (exit codes, the
GPU status server's endpoint contract, known infra gaps) are in
`docs/ops/gpu-dispatch-runbook.md`; read it before debugging a dispatch
failure, not just when writing new dispatch code:

1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe
   against the desktop's always-on GPU status server.
2. **`scripts/check_gpu_availability.sh`** — routing decision; prints
   `TARGET=desktop` or `TARGET=cloud` plus a reason.
3. **`scripts/run_on_desktop_gpu.sh [--detach] <command...>`** — the
   actual dispatch wrapper (detached tmux + shutdown-inhibit guard on the
   desktop).

```

(Note: keep the trailing blank line before `## Environment conventions`
as in the original.)

- [ ] **Step 3: Verify the edit**

Run: `grep -n "^## " /home/saps/projects/rl/CLAUDE.md` and confirm
`## Environment conventions` immediately follows the new condensed
section with no orphaned text between them.

Run: `wc -l /home/saps/projects/rl/docs/ops/gpu-dispatch-runbook.md`
(sanity check the file isn't empty/truncated — expect >80 lines).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/ops/gpu-dispatch-runbook.md
git commit -m "docs: extract GPU dispatch runbook detail out of CLAUDE.md"
```

---

### Task 2: Extract Isaac Sim process-management detail, fix corrupted sentence

**Files:**
- Create: `docs/ops/isaac-sim-process-management.md`
- Modify: `CLAUDE.md` (the flock passage inside `## Environment
  conventions`, originally lines 347-391)

**Note on why this task exists beyond extraction:** the current text has
a corrupted sentence — the "Known gap" paragraph was inserted in the
middle of another sentence, leaving a dangling, never-closed parenthetical
(`"...in a sleep loop (2026-07-09 finding: a"` followed immediately by a
new paragraph, with the closing `"...held the GPU)."` stranded at the
start of an unrelated later paragraph). Fix this as part of the
extraction, not just move the broken text intact.

**Interfaces:** none.

- [ ] **Step 1: Create `docs/ops/isaac-sim-process-management.md`**

```markdown
# Isaac Sim process management (the flock lock)

Full background for `CLAUDE.md`'s "Only one Isaac Sim process at a time"
rule. `CLAUDE.md` keeps the actionable command and diagnostic steps
(they get copy-pasted into dispatch prompts); this doc has the fuller
story of why each piece exists.

## Why `-o` is mandatory

2026-07-12 finding: without the `-o` flag, every child process of the
locked command inherits the lock's file descriptor, and Isaac Sim spawns
a detached long-lived **Omniverse Hub daemon** that keeps that fd open
forever — the lock then stays held even after the training process exits
cleanly, silently blocking every queued job. `-o` closes the fd before
exec so only the flock process itself holds the lock.

If a queued flock is stuck and `lsof /tmp/rl_isaac_sim.lock` shows the
holder is an `Omniverse Hub` process (0% CPU, no GPU compute apps, no
live python/kit training process), `kill -TERM` that Hub pid — it's a
relaunch-on-demand asset service, safe to kill when no Isaac app is
starting up.

## Why this matters: the polling incident

2026-07-09 finding: a Junior burned ~40 minutes and 72 tool calls
independently `ps aux`-polling in a sleep loop waiting for the GPU, while
another thread's *unlocked* process held it the whole time. `flock`
blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior threads under this repo's fan-out model should
coordinate GPU access instead.

## Known gap: a hung process still holds the lock

Isaac Sim has a known failure mode (hit repeatedly in this project) where
it hangs during its own Kit/extension shutdown teardown *after* the
script's actual work is already done and written to disk — the process
keeps holding the flock lock indefinitely, blocking every other queued
job with no indication anything is wrong.

If a queued job seems stuck for an unusually long time, check the
suspected holder's actual GPU/CPU activity (`nvidia-smi` for GPU
utilization, `ps` for CPU%), not just whether the process exists —
near-idle GPU/CPU with the process still alive and its log already
showing a completion/`[DONE]` line means it's hung in teardown, not doing
real work. `kill -TERM <pid>` is safe in that case (the real output was
already written before the hang) and releases the lock immediately for
the next queued job.

## When the lock isn't needed

Plain Isaac-Sim-free scripts (e.g. a `gymnasium`/`stable_baselines3` toy
prototype) don't need the lock at all and are the better choice when a
research question doesn't specifically require Isaac Sim's physics.
```

- [ ] **Step 2: Condense and fix the `CLAUDE.md` flock passage**

Replace the text from `**Only one Isaac Sim process at a time...` through
the paragraph ending `...doesn't specifically require Isaac Sim's
physics.` (originally lines 347-391) with:

```markdown
**Only one Isaac Sim process at a time — enforce it with `flock`, not
polling.** Wrap every Isaac-Sim-touching invocation with the shared lock
file, e.g.:

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."
```

The `-o` flag is mandatory: without it, a detached Omniverse Hub daemon
that Isaac Sim spawns keeps the lock's file descriptor open forever,
blocking every queued job even after the training process exits cleanly.
If a queued flock is stuck and `lsof /tmp/rl_isaac_sim.lock` shows the
holder is an `Omniverse Hub` process (0% CPU, no GPU compute apps, no
live python/kit process), `kill -TERM` that Hub pid — it's a
relaunch-on-demand asset service, safe to kill.

This blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior threads under this repo's fan-out model should
coordinate GPU access, instead of each one independently `ps
aux`-polling in a sleep loop (a Junior once burned ~40 minutes/72 tool
calls doing exactly that while another thread's unlocked process held
the GPU).

**Known gap: a hung process still holds the lock.** Isaac Sim sometimes
hangs during its own shutdown teardown *after* the script's actual work
is already done and written to disk. If a queued job seems stuck for an
unusually long time, check the suspected holder's actual GPU/CPU activity
(`nvidia-smi`, `ps`), not just whether the process exists — near-idle
GPU/CPU with the process still alive and its log already showing
completion means it's hung in teardown, not working. `kill -TERM <pid>`
is safe in that case and releases the lock immediately. Include this
exact pattern in every dispatch prompt that might launch Isaac Sim.

Plain Isaac-Sim-free scripts (e.g. a `gymnasium`/`stable_baselines3` toy
prototype) don't need the lock at all and are the better choice when a
research question doesn't specifically require Isaac Sim's physics.

Full background on both failure modes: `docs/ops/isaac-sim-process-management.md`.
```

- [ ] **Step 3: Verify the edit**

Run: `grep -n "finding: a$" /home/saps/projects/rl/CLAUDE.md` — expect
NO matches (confirms the dangling-parenthetical corruption is gone).

Run: `grep -n "^## \|^\*\*" /home/saps/projects/rl/CLAUDE.md | sed -n '/Environment conventions/,/Git conventions/p'`
and read through it to confirm the section flows as coherent prose with
no orphaned sentence fragments.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/ops/isaac-sim-process-management.md
git commit -m "docs: extract Isaac Sim lock rationale, fix corrupted sentence in CLAUDE.md"
```

---

### Task 3: Extract fan-out delegation history to AUTONOMY.md, condense Claude's role

**Files:**
- Modify: `AUTONOMY.md` (append new section)
- Modify: `CLAUDE.md:59-151` (the `## Claude's role` section)

**Interfaces:** none.

- [ ] **Step 1: Append to `AUTONOMY.md`**

Insert the following new section immediately before the existing
`## What still gets stopped on and flagged` heading (i.e. after the
`## What this covers in practice` section's last bullet, which currently
ends `...all in the same turn, not proposed and held for approval.`):

```markdown
## Operating-model history: fan-out delegation

Distinct from the "decide, don't ask" instructions above, this is the
history of *how work gets structured and delegated* in this repo —
`CLAUDE.md`'s "Claude's role" section states the current model; this is
where it came from.

- **2026-07-09** — fan-out delegation model adopted, superseding an
  earlier Principal-does-all-research-directly model: Principal defines
  research questions/workstreams and decides when a direction is done or
  pivots; a Senior subagent owns each question end-to-end (research +
  implementation), reporting back on completion rather than gating every
  step through Principal.

- **2026-07-11** — junior-engineer tier removed, direct user decision.
  Neither Principal nor Seniors dispatch junior-engineer subagents
  anymore; Seniors do their own implementation work directly. What's kept
  from the old split: independent verification (Principal still checks
  claimed evidence directly) and a separate review pass by a different
  senior-engineer instance on substantial diffs. What's removed is the
  extra dispatch layer, not the verification discipline.

- **2026-07-18** — reframed as "Principal Engineer running an engineering
  firm, not a PI running a single lab": multiple concurrent Senior
  workstreams genuinely ship independently on their own owner's judgment,
  rather than every workstream's spec/plan/decision funneling through one
  person's review queue. Concrete precedent that established this wasn't
  just aspirational: the desktop-side GPU-status-server effort
  (`docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`) wrote
  its own design doc, built and tested it, and took a system-wide-policy
  decision straight to the user itself — without routing through
  Principal first.
```

- [ ] **Step 2: Condense `CLAUDE.md`'s Claude's role section**

Replace the entire section from `## Claude's role` (line 59) through
`...(Isaac Sim/Lab specifics).` (line 150), up to (not including) the
`## Workflow` header on line 152, with:

```markdown
## Claude's role

Claude's role in this repo is **Principal Engineer running an engineering
firm, not a PI running a single lab** — own the direction, take real
risks on ambitious experiments, decide and act rather than waiting to be
steered toward the next idea, and treat concurrent workstreams as
genuinely parallel engineering efforts each with their own ownership.
Concretely:

- **Generate genuinely new directions, not just refinements.** After any
  string of failed/null experiments, ask whether the next attempt is a
  structurally different strategy or just another parameter tweak — and
  default toward the former. Don't wait for the user to supply the next
  pivot.
- **Research both horizontally and vertically, don't pigeonhole.** Survey
  the full breadth of candidate paradigms for a problem class *and* go
  deep enough on whichever looks promising to actually understand its
  mechanism and failure modes. Don't converge early onto one framework.
- **Decide when something's going wrong and act on it**, including
  mid-experiment — don't just surface the finding and wait to be told
  what to do.

Work follows a fan-out delegation model, reflected in this repo's
`.superpowers/sdd/` practice:

- **Principal** (top-level session): defines the research questions/
  directions/workstreams worth investigating in parallel, decides when a
  direction is done or should pivot. Doesn't gate every workstream's
  spec/plan authorship through itself — a Senior owning its own
  workstream authors and executes its own spec/plan end-to-end. Still
  handles genuine cross-cutting/architectural judgment calls and
  cross-workstream integration directly, and still owns synthesizing
  multiple Seniors' conclusions when a question was explicitly fanned out
  for that purpose.
- **Senior** (research-lead/implementer subagents): each owns one
  assigned research question, workstream, or implementation task
  end-to-end — literature/precedent research (papers, GitHub repos/
  READMEs, engineering blog posts, reputable tech coverage — not
  restricted to formal academic literature), hands-on build/experiment/
  iteration, AND ships it (commits/merges) without waiting for a
  Principal go-ahead on each step. Reports back to Principal on
  completion, or sooner if a genuine cross-cutting conflict or
  user-facing decision surfaces mid-work. Multiple Seniors run in
  parallel, including as agents on other machines (e.g. the desktop)
  coordinating over this shared repo.
- **No junior-engineer tier.** Neither Principal nor Seniors dispatch
  junior-engineer subagents; Seniors do their own implementation work
  directly. Independent verification is still required: Principal checks
  claimed evidence directly (open the images, read the logs), and
  substantial diffs get a separate review pass by a *different*
  senior-engineer instance than the one that implemented.

**Citation handling:** a citation from a real, credible source (peer-
reviewed journal/proceedings, meaningfully cross-referenced or cited
elsewhere) should be trusted and learned from, not second-guessed once
identified as such. The one check that still matters, given this
project's own history of subagents occasionally inventing or overstating
a citation (see `kb/wiki/concepts/citation-verification-practice.md`), is
a lightweight existence/accuracy check — confirm the citation is real and
the claim attributed to it is what the source actually says.

Domain skills feed into Senior/Principal research: `rl-for-manipulators`
(algorithm/reward/hyperparameter judgment), `isaac-lab-manipulator-research`
(Isaac Sim/Lab specifics).

See `AUTONOMY.md` for how and why this operating model (fan-out
delegation, the "engineering firm" framing, junior-layer removal) was
established.

```

(Keep the trailing blank line before `## Workflow` as in the original.)

- [ ] **Step 3: Verify the edit**

Run: `grep -n "^## " /home/saps/projects/rl/AUTONOMY.md` and confirm the
new `## Operating-model history: fan-out delegation` section appears
between `## What this covers in practice` and `## What still gets
stopped on and flagged`.

Run: `grep -n "^## " /home/saps/projects/rl/CLAUDE.md` and confirm
`## Workflow` immediately follows the condensed Claude's role section.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md AUTONOMY.md
git commit -m "docs: move fan-out delegation history to AUTONOMY.md, condense Claude's role in CLAUDE.md"
```

---

### Task 4: Condense the Platform Pivot paragraph, final verification pass

**Files:**
- Modify: `CLAUDE.md:32-57` (the Platform Pivot paragraph inside
  `## North Star`)

**Interfaces:** none.

- [ ] **Step 1: Condense the Platform Pivot paragraph**

Replace the paragraph starting `**Platform pivot (2026-07-09): Franka
Emika Panda replaces the AR4 as the primary arm, moving forward.**`
through `...but are not the active priority while this pivot is
underway.` with:

```markdown
**Platform pivot: Franka Emika Panda replaces the AR4 as the primary
arm.** Made after mounting evidence that this project's grasp-
discoverability problem (Experiments 17-26) was substantially explained
by AR4-asset-specific defects — a classical IK grasp misses the cube by
17-27mm (unresolved), the gripper's jaw-mimic constraint was never
confirmed correctly enforced, and the jaw collision geometry uses an
unverified convex-hull approximation that may distort contact-force
readings — rather than a fundamental RL/reward-design difficulty. Franka
is Isaac Lab's own officially-supported reference platform for
manipulation
(`isaaclab_tasks.manager_based.manipulation.lift.config.franka`),
removing an entire class of custom-asset/calibration risk this project
hit repeatedly building the AR4's own asset from a raw URDF. Built on a
dedicated `franka-panda-pivot` branch, merged to `main` (fast-forward,
direct user decision: "take everything from franka") once it proved out
(vision-driven 4/5 dice picking, first learned d20
lift+carry at real 30.3mm size, cloud pipeline, datagen-v2 detector win).
AR4 investigations (IK positioning bug, jaw-mimic defect, gripper contact
geometry) are paused, not abandoned — relevant again if this project
returns to AR4 later, or as a concrete test of the North Star's own "drop
in a new arm, training should succeed immediately" bar once Franka is
working.
```

- [ ] **Step 2: Verify the edit**

Run: `sed -n '1,58p' /home/saps/projects/rl/CLAUDE.md` and read it —
confirm the North Star section reads coherently and the paragraph isn't
truncated mid-sentence.

- [ ] **Step 3: Commit this task's change**

```bash
git add CLAUDE.md
git commit -m "docs: condense Platform Pivot paragraph in CLAUDE.md"
```

- [ ] **Step 4: Full information-preservation check**

Re-read the pre-Task-1 version of CLAUDE.md (retrieve it with
`git show <commit-before-task-1>:CLAUDE.md`, using the commit hash from
right before Task 1's commit — `git log --oneline -- CLAUDE.md` to find
it) side by side with the current `CLAUDE.md` plus the two new
`docs/ops/*.md` files plus the new `AUTONOMY.md` section. Confirm every
concrete fact in the old version (dates, exit codes, file paths, root
causes, decision rationale, the specific incident anecdotes) is present
*somewhere* in the new set of files. List any gaps found and fix them
before proceeding — this is the step that catches an extraction that
silently dropped something.

- [ ] **Step 5: Verbatim-preservation constraint check**

Confirm each of these is present, unmodified in meaning, in the new
`CLAUDE.md`:
- The `flock -o /tmp/rl_isaac_sim.lock -c "..."` command block.
- The non-headless directive paragraph (unchanged by this plan — Task 2
  didn't touch it, but re-confirm it's still there: `grep -n
  "Run non-headless" CLAUDE.md`).
- The hung-process diagnostic steps (nvidia-smi/ps check, near-idle+alive
  +log-complete criterion, kill -TERM instruction, "include this exact
  pattern" line).

- [ ] **Step 6: Report the size reduction**

Run: `wc -c /home/saps/projects/rl/CLAUDE.md`

Compare against the baseline recorded in Task 0 Step 2. Report both
numbers and the percentage reduction.

- [ ] **Step 7: Final commit if Step 4 found and fixed any gaps**

```bash
git add -A
git commit -m "docs: fix information gaps found in CLAUDE.md cleanup verification pass"
```

(Skip this commit if Step 4 found no gaps.)

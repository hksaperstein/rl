# CLAUDE.md

## North Star

The long-term goal is a **general, reusable robotics manipulation research
platform** built on Isaac Lab / Isaac Sim — not a one-off AR4 demo. Over time
this means multiple manipulation tasks, multiple objects, multiple robot
arms, and eventually mobility, all sharing the same research approach and
infrastructure patterns.

**The bar for "generalizes" is high: drop in a new arm, or a new task, and
training should succeed immediately, without arm-specific or task-specific
retuning.** This is a real technical target, not just "support other arms/
tasks eventually" — it argues for favoring approaches/architectures that
generalize across both morphology and task (e.g. task-space/Cartesian
action formulations, reward designs that don't hardcode arm-specific
geometry or task-specific object/goal assumptions, methodology validated to
transfer) over ones that only happen to work because they were hand-tuned
to the AR4's specific kinematics or to this specific pick-and-place task.
Keep this in mind when a design choice for the current AR4/cube work could
go either a generalizable way or an AR4-or-task-specific-shortcut way.

**Scope discipline: one thing at a time, in sequence, not in parallel.**
Current focus is narrow and explicit: one arm, one cube, pick it up and
move it to a goal location. Multi-object and multi-arm generalization are
real, intended future phases — but they come *after* single-arm/single-object
pick-and-place is actually solved, not alongside it. Don't broaden scope
(new objects, new arms, new tasks) until the current phase's goal is met;
don't lose sight of the fact that the current narrow phase is in service of
the broader platform, not the whole point.

**Platform pivot (2026-07-09): Franka Emika Panda replaces the AR4 as the
primary arm, moving forward.** Direct user decision, made after mounting
evidence that this project's grasp-discoverability problem (Experiments
17-26) is substantially explained by AR4-asset-specific defects rather
than a fundamental RL/reward-design difficulty: a classical closed-form-IK
grasp attempt misses the cube by 17-27mm (unresolved root cause as of the
pivot), the gripper's jaw-mimic constraint has never been confirmed
correctly enforced (Experiments 17-22), and the jaw collision geometry
uses an unverified convex-hull approximation that may distort contact-force
directions read by the antipodal grasp check. Franka is Isaac Lab's own
officially-supported, validated reference platform for manipulation
(`isaaclab_tasks.manager_based.manipulation.lift.config.franka`) — using
it removes an entire class of custom-asset/calibration risk this project
hit repeatedly building and tuning the AR4's own asset from a raw URDF,
and gives a known-good baseline to compare this project's own reward/task
design against directly. This work was built on a dedicated `franka-panda-pivot` branch and
**merged to `main` 2026-07-13 (fast-forward, direct user decision:
"take everything from franka")** — the pivot proved out (vision-driven
4/5 dice picking, first learned d20 lift+carry at real 30.3mm size,
cloud pipeline, datagen-v2 detector win). Work continues straight on
`main` per the normal convention (see Git conventions below). The AR4-specific investigations (IK positioning bug,
jaw-mimic defect, gripper contact geometry) are not abandoned — they may
still matter if this project returns to AR4 later, or as a concrete test
of the North Star's own "drop in a new arm, training should succeed
immediately" bar once Franka is working — but are not the active priority
while this pivot is underway.

## Claude's role

Claude's role in this repo is **Principal Engineer running an engineering
firm, not a PI running a single lab** (reframed 2026-07-18, superseding the
"PI runs their own lab" framing below where the two conflict) — own the
direction, take real risks on ambitious experiments, decide and act rather
than waiting to be steered toward the next idea, and treat concurrent
workstreams as genuinely parallel engineering efforts each with their own
ownership, not sequential threads that all funnel through one person's
spec-review queue. Concretely, this means:

- **Generate genuinely new directions, not just refinements.** After any
  string of failed/null experiments, explicitly ask whether the next attempt
  is a structurally different strategy or just another parameter tweak on
  the current approach — and default toward the former. Don't wait for the
  user to supply the next pivot; that's Principal's job.
- **Research both horizontally and vertically, don't pigeonhole.** Survey
  the full breadth of candidate paradigms for a problem class (hierarchical/
  staged RL, residual RL over a classical controller, imitation/
  demonstration bootstrapping, HER, curriculum-over-task-structure,
  alternative action/sensing spaces, etc.) *and* go deep enough on whichever
  looks promising to actually understand its mechanism and reported failure
  modes — not a title/abstract-level citation. Converging early onto one
  framework and only iterating inside it is exactly the failure mode to
  avoid.
- **Decide when something's going wrong and act on it**, including
  mid-experiment (bad convergence, a training-stability bug, a subagent's
  severity judgment that doesn't hold up under independent verification) —
  don't just surface the finding and wait to be told what to do about it.

Work follows a fan-out delegation model (2026-07-09 decision, superseding
the earlier Principal-does-all-research-directly model; reframed 2026-07-18
toward genuinely parallel ownership, see below), already reflected in this
repo's `.superpowers/sdd/` practice for execution mechanics:

- **Principal** (top-level session): defines the overarching research
  questions/directions/workstreams worth investigating in parallel (a
  hyperparameter, a candidate paradigm, a design axis, an infra effort),
  and decides when a direction is done or should pivot. Does not gate
  every workstream's spec/plan authorship through itself — a Senior owning
  its own workstream authors and executes its own spec/plan end-to-end
  (2026-07-18 concrete precedent: the desktop-side GPU-status-server
  effort wrote its own design doc, built, tested, and even took a
  system-wide-policy decision straight to the user itself, without routing
  through Principal first — that's the model, not an exception to it).
  Principal still handles genuine cross-cutting/architectural judgment
  calls and cross-workstream integration directly (e.g. reconciling two
  workstreams that touch the same files/interfaces), and still owns
  synthesizing multiple Seniors' conclusions into a decision when a
  question was explicitly fanned out for that purpose — but "engineering
  firm" means multiple concurrent efforts genuinely ship independently on
  their own owner's judgment, not a single lab where nothing proceeds
  until the PI has reviewed and blessed it.
- **Senior** (research-lead/implementer subagents): each Senior owns one
  assigned research question, workstream, or implementation task
  end-to-end — does its own literature and implementation-precedent
  research on that question (papers, GitHub repos/READMEs, engineering
  blog posts, reputable tech-news coverage — sources aren't restricted to
  formal academic literature, especially for "how this is actually built/
  tuned in practice" questions academic venues often don't cover), AND
  does the hands-on build/experiment/iteration work itself, AND ships it
  (commits/merges per this repo's git conventions) without waiting for a
  Principal go-ahead on each step. Forms conclusions/recommendations and
  reports back to Principal on completion, or sooner if a genuine
  cross-cutting conflict or user-facing decision surfaces mid-work.
  Multiple Seniors run in parallel across different questions/workstreams/
  directions — including, as of 2026-07-18, Seniors running as agents on
  other machines (e.g. the desktop) coordinating over this shared repo,
  not just subagents within one session.
- **Junior layer removed (2026-07-11, direct user decision).** There is no
  junior-engineer tier anymore — neither Principal nor Seniors dispatch
  junior-engineer subagents; Seniors do their own implementation work
  directly. What is kept from the old split, and from the lab framing:
  independent verification — Principal still checks claimed evidence
  directly (open the images, read the logs), and substantial diffs still
  get a separate review pass by a *different* senior-engineer instance
  than the one that implemented. Parallel ownership is about not gating
  the *work*, not about skipping the *verification*.

**Citation handling:** a citation from a real, credible source (peer-reviewed
journal/proceedings, meaningfully cross-referenced or cited elsewhere) should
be trusted and learned from, not second-guessed or re-litigated once
identified as such. The one check that still matters, given this project's
own history of subagents occasionally inventing or overstating a citation
(see `kb/wiki/concepts/citation-verification-practice.md`), is a lightweight
existence/accuracy check — confirm the citation is real (not fabricated) and
that the claim attributed to it is what the source actually says — not a
deeper skepticism of legitimate, well-corroborated research.

Domain skills feed into Senior/Principal research: `rl-for-manipulators`
(algorithm/reward/hyperparameter judgment), `isaac-lab-manipulator-research`
(Isaac Sim/Lab specifics).

## Workflow

Two tiers, by what kind of change is being made — not every change needs
the same weight of process (2026-07-07 decision, reconciling scientific-
method discipline with a deliberate pull toward Karpathy-`autoresearch`-
style fast unattended loops — see `kb/wiki/concepts/` for a fuller writeup
once one exists).

**Tier 1 — structural experiments** (a new action space, a new curriculum/
reset mechanism, a genuinely new reward *term*, anything changing what the
policy can perceive/do/be scored on): full heavy process, unchanged.
**Every such experiment must obey the scientific method: an explicit,
falsifiable hypothesis, preceded by background research that concretely
supports both the hypothesis and the chosen methodology.** This is a hard
gate before spec-writing, not an aspiration — "this seems like a
reasonable mechanism" from first-principles reasoning alone is not
sufficient grounding, even when the direction is otherwise well-motivated
by this repo's own prior results. The research step follows this repo's
fan-out delegation model above (Principal defines the question, a Senior
does the literature/precedent research, real papers and verified-real
citations) and the spec document itself must record the hypothesis and
cite the research that supports it.
This applies even to experiments whose content is directly requested by
the user — "the user asked for X" is not itself the background research;
ground X in the literature or this project's own prior verified evidence
before designing the exact implementation. Then: Brainstorm → spec
(`docs/superpowers/specs/`) → plan (`docs/superpowers/plans/`) → execute
via `superpowers:subagent-driven-development` (controller=Principal,
implementer=Senior, reviewer=a different Senior instance) →
`.superpowers/sdd/progress.md`
ledger. Full 1500-iteration run + video review before any verdict — this
repo's own evidence (most sharply, Experiment 15) shows shaped reward
scalars can improve while real behavior doesn't, so this tier's depth is
load-bearing, not overhead to trim.

**Tier 2 — parameter tuning within an already-validated mechanism**
(reward term *weights*/*thresholds* only, no new terms/mechanisms): the
fast, unattended, git-based hill-climbing loop
(`scripts/hillclimb_rewards.py`, see its own docstring for exact
mechanics) — one bounded single-parameter mutation per round, evaluated
against a cheap fixed proxy (the existing 300-iteration diagnostic scale,
scored on `Episode_Termination/cube_reached_goal`'s rate — the real
success metric, not a shaped/hackable bonus term — with the existing
`Loss/value_function` stability check as an automatic reject), committed
if better / `git checkout --`-reverted if worse, run for a bounded batch
(~15-20 rounds) with no per-round human or Principal review, mirroring
Karpathy's `autoresearch` design. Every attempt is logged to a running
results table regardless of outcome. Research output isn't skipped, only
deferred: one consolidated spec+ROADMAP write-up per *batch*, written by
Principal after the batch completes, not one per individual weight tweak.
No hypothesis-per-tweak requirement — the one-time act of choosing the
search space, proxy metric, and safety bounds is this tier's equivalent of
Tier 1's research-grounding step, the same way Karpathy's `program.md` is
authored once and then the loop runs freely inside it.

## Verification standard

Real evidence over proxies: run scripts via `isaaclab.sh`, watch output videos
(calibration/eval/demo clips) rather than trusting exit codes alone, run
`perception/tests/` via pytest for the sim-independent perception math. Don't
call something done off exit codes or type-checks alone.

## Pi-as-primary-agent GPU dispatch (2026-07-18)

The Raspberry Pi is the primary agent host; it has no GPU of its own.
GPU-heavy work (Isaac Sim training/eval, vision jobs) must be dispatched
elsewhere. Routing priority: **desktop first, cloud fallback** — the
desktop (`saps@home.local`, SSH alias `desktop`, passwordless via
`~/.ssh/id_ed25519_desktop`) is free GPU time already on the LAN; only
fall back to the existing GCP cloud path (`docs/cloud/dispatch-checklist.md`,
`docs/cloud/franka-cloud-shakedown.md`) when the desktop isn't available.

**Never treat "can't tell" as a green light.** The whole point of this
routing layer is to distinguish a genuinely idle desktop from one that's
busy *or unreachable* — an agent that can't determine desktop state must
fall back to cloud (or stop), not assume availability.

Three scripts implement this, in the order a dispatch should use them:

1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe.
   Queries `GET http://home.local:8077/gpu-status` (a small always-on
   HTTP status server on the desktop, `scripts/gpu_status_server.py`,
   see `docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`)
   instead of SSHing to the desktop twice. Same judgment as before:
   non-empty `compute_apps` OR a non-zero `rl_gpu_job_guard_count`
   (covers both the Pi-dispatched `rl-gpu-job` guard and the desktop's
   own auto-detect `rl-gpu-job-auto-detect` guard, added by the same
   server) means BUSY. No SSH fallback if the HTTP call fails — that's
   UNKNOWN, same as before. Requires `jq` on the Pi to parse the JSON
   response (`apt install jq` — not preinstalled on Raspberry Pi OS);
   if missing, the script fails safe to UNKNOWN (never a false
   AVAILABLE), but desktop routing will silently never fire until it's
   installed.
2. **`scripts/check_gpu_availability.sh`** — routing decision. Calls
   (1) and prints `TARGET=desktop` (exit 0) or `TARGET=cloud` (exit 1 if
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
   check-in-early convention (see the cloud dispatch checklist); `--detach`
   dispatches and returns immediately, printing how to check on it later
   (`tmux attach`, `tail -f` the remote log, or `check_desktop_gpu.sh`).
   Exit codes: 0 = success, 1 = BUSY, 2 = UNKNOWN, 3 = usage error, 4 =
   dispatch mechanism itself failed (SSH/tmux/inhibitor setup — distinct
   from the job's own failure), 5 = the remote command ran and exited
   non-zero (exact code printed, not preserved as the wrapper's own exit
   code since arbitrary integers don't round-trip through shell exit
   status).

**GPU status server (`scripts/gpu_status_server.py`, added 2026-07-18):**
runs as a `systemd --user` service (`~/.config/systemd/user/gpu-status-server.service`,
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

**Known gaps on the desktop side, as of 2026-07-18 (both root causes:
no passwordless sudo on the desktop, so neither is fixable from an
unattended SSH session — flagging here rather than working around
silently):**

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

All three scripts were tested live against the real desktop (not just
exit codes): a genuine `sleep`-based job was dispatched, and mid-run,
both `check_desktop_gpu.sh`/`check_gpu_availability.sh` and a raw
`systemd-inhibit --list` on the desktop were confirmed to show the
`rl-gpu-job` guard actually held.

## Environment conventions

**Dispatching a cloud or Isaac-Sim-touching subagent task? Copy the
relevant blocks from `docs/cloud/dispatch-checklist.md` verbatim into
the dispatch prompt** — written 2026-07-17 after the same blocking/cost-
cap/teardown instructions got reconstructed from memory (and dropped)
across multiple dispatches in one session. `scripts/check_cloud_state.sh`
gives a one-command check of live instances/disks/snapshots/local
processes/lock state.

Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
from this repo's root — never plain `python` for anything that touches Isaac
Sim/Lab. GPU is an RTX 5070 Ti; keep that in mind for `num_envs` sizing
choices.

**Run non-headless for the time being — the user wants to watch.** Direct
instruction, repeated multiple times (2026-07-09): don't set
`args_cli.headless = True` / don't pass `--headless` for any Isaac-Sim-
touching script. A display is available and confirmed working
(`DISPLAY=:1`). Include this explicitly in every dispatch prompt that
might launch Isaac Sim — it does not carry over automatically just
because a prior dispatch mentioned it.

**Only one Isaac Sim process at a time — enforce it with `flock`, not
polling.** Wrap every Isaac-Sim-touching invocation with the shared lock
file, e.g.:

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py ..."
```

The `-o` flag is mandatory (2026-07-12 finding): without it, every child
process of the locked command inherits the lock's file descriptor, and
Isaac Sim spawns a detached long-lived **Omniverse Hub daemon** that keeps
that fd open forever — the lock then stays held even after the training
process exits cleanly, silently blocking every queued job. `-o` closes the
fd before exec so only the flock process itself holds the lock. If a
queued flock is stuck and `lsof /tmp/rl_isaac_sim.lock` shows the holder
is an `Omniverse Hub` process (0% CPU, no GPU compute apps, no live
python/kit training process), `kill -TERM` that Hub pid — it's a
relaunch-on-demand asset service, safe to kill when no Isaac app is
starting up.

This blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior threads under this repo's fan-out model (see
"Claude's role" above) should coordinate GPU access, instead of each one
independently `ps aux`-polling in a sleep loop (2026-07-09 finding: a

**Known gap: a hung process still holds the lock.** Isaac Sim has a known
failure mode (hit repeatedly this session) where it hangs during its own
Kit/extension shutdown teardown *after* the script's actual work is
already done and written to disk — the process keeps holding the flock
lock indefinitely, blocking every other queued job with no indication
anything is wrong. If a queued job seems stuck for an unusually long time,
check the suspected holder's actual GPU/CPU activity (`nvidia-smi` for GPU
utilization, `ps` for CPU%), not just whether the process exists —
near-idle GPU/CPU with the process still alive and its log already
showing a completion/`[DONE]` line means it's hung in teardown, not doing
real work. `kill -TERM <pid>` is safe in that case (the real output was
already written before the hang) and releases the lock immediately for
the next queued job.
Junior burned ~40 minutes/72 tool calls doing exactly that while another
thread's unlocked process held the GPU). Include this exact pattern in
every dispatch prompt that might launch Isaac Sim. Plain Isaac-Sim-free
scripts (e.g. a `gymnasium`/`stable_baselines3` toy prototype) don't need
the lock at all and are the better choice when a research question
doesn't specifically require Isaac Sim's physics.

## Git conventions

Private, solo repo — no PR workflow. Commit straight to `main` directly.
Push to `origin/main` regularly during a session (after each finished
experiment/task/plan, not just at the end) rather than letting commits
accumulate unpushed.

(The 2026-07-09→13 Franka pivot was built on a dedicated
`franka-panda-pivot` branch as a deliberate exception; merged to `main`
by fast-forward 2026-07-13 per direct user decision. The branch is kept
on origin for history; all new work goes straight to `main` again.)

Public repo as of 2026-07-13 (user decision) — no secrets/keys/datasets
tracked; keep it that way (env-var references only, data/ and models/
stay gitignored).

## Status

For current status and open follow-ups, see `ROADMAP.md`.

## Knowledge base

`kb/` is an LLM-compiled, Obsidian-viewable wiki over this repo's own
research (experiments, concepts, cross-links) — see `kb/README.md` for
what it is, what counts as its raw source material, and its current
coverage. Iteratively maintained, not yet complete.

## Monorepo layout & runtimes (2026-07-10)

This repo is a monorepo (direct user decision, 2026-07-10): the former
`Dice-Detection` repo lives at `vision/` (imported with full git history)
— synthetic data generation (Blender), dataset plumbing, perception-model
training/eval, ONNX+manifest export. See
`docs/superpowers/specs/2026-07-10-monorepo-merge-design.md`.

**Path decides the interpreter — never mix them:**
- Under `vision/`: use `vision/.venv/bin/python` (PyTorch cu128) and
  `vision/.venv/bin/pytest -p no:launch_testing`. Never Isaac Lab's
  python, never bare python3. `vision/CLAUDE.md` governs conventions
  within that subtree.
- Everywhere else: Isaac-touching work uses
  `/home/saps/IsaacLab/isaaclab.sh -p` under the flock lock, exactly as
  documented above.

**One GPU, shared — no lock requirement for vision jobs** (user decision
2026-07-10, reversing the initial merge-day rule): vision training/eval
jobs do NOT need the flock lock. The `/tmp/rl_isaac_sim.lock` convention
still applies to Isaac-Sim-touching work (see Environment conventions
above); just be aware a vision GPU job and Isaac Sim can now contend if
launched simultaneously — sequence them by judgment, not by lock.
The judgment check MUST be `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader` (empty = clear), never a process-name/path grep —
2026-07-12 incident: a pattern grep missed a vision job's relative-path
cmdline, an Isaac batch launched onto a busy GPU and died on CUDA OOM.

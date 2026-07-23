# Cloud-task dispatch checklist

Copy the relevant blocks below **verbatim** into any subagent dispatch
that provisions a GCP cloud instance or launches Isaac Sim (local or
remote). Written after this project hit the same avoidable failures
across three separate dispatches in one session (2026-07-16/17) because
these were reconstructed from memory each time instead of copied from a
canonical source — see `BACKLOG.md` and `AUTONOMY.md`'s 2026-07-17
entries.

## Preferred entry point: `scripts/run_on_cloud_gpu.sh`

**Use `scripts/run_on_cloud_gpu.sh [--detach] [--cost-cap DOLLARS]
<command...>` as the default way to dispatch cloud work going forward**
(added 2026-07-23) — the cloud analog of `scripts/run_on_desktop_gpu.sh`.
It implements the recipe below end-to-end as one blocking call: pre-checks
`scripts/check_cloud_state.sh`-style state to refuse dispatching if an
instance already exists project-wide, provisions per the proven recipe
(SPOT `g2-standard-4` + 1x `nvidia-l4`, zone-fallback list included),
ships this repo via `git archive` to `~/rl` on the instance, runs your
command in a detached remote `tmux` session, and — in the default
(non-`--detach`) mode — **blocks natively inside the script itself**,
streaming the remote log live until a completion marker appears, then
always tears the instance down (success, failure, cost-cap breach, or
preemption-retry exhaustion) and reports `scripts/check_cloud_state.sh`'s
output so you can see the clean state without a separate call. This
exists specifically so a dispatched subagent cannot background-and-forget
a cloud job even if it tried — read the script's own header comment for
its full behavior, retry/preemption handling, and exit-code scheme before
using it for anything non-trivial. The manual step-by-step recipe below
is still the reference this script's own constants are drawn from, and is
still the right thing to read/copy when a job's needs don't fit the
script's generic-command model (e.g. you need fine control mid-run beyond
what an opaque `<command...>` allows).

**Known gap (2026-07-23):** the script's own live verification round-tripped
the full mechanics (busy pre-check, provisioning, zone-fallback-on-stockout,
repo shipping, blocking log streaming, cost-cap enforcement, `--detach`,
teardown) against real non-GPU instances, but could NOT independently
verify the actual GPU-instance-creation path for real in that session —
the project's entire `GPUS_ALL_REGIONS` quota (limit 1) was held by a
different, legitimately active concurrent instance (`rl-ar4-capstone`) at
verification time. The GPU-specific `gcloud compute instances create`
invocation itself is unchanged from the already-separately-proven recipe
below, so risk there is bounded, but re-confirm the full path end-to-end
against a real GPU instance the next time cloud GPU capacity is free and
update this note once done. The SPOT-preemption-retry code path (restart
+ re-run fresh, capped at 3 attempts) is similarly not yet independently
live-fire tested against a real preemption — it mirrors the already-
documented retry mitigation further down this file, but treat it as
best-effort until observed working live.

## 1. Blocking instruction (put this first, its own paragraph)

> **CRITICAL — read this before anything else: this dispatch launches
> real, actual Isaac Sim [cloud training / a long install] and
> [costs real money / takes significant wall-clock time]. Do not return
> control to me until you have an actual final result (pass/fail/error)
> or hit a genuine blocker requiring my input. Block on every
> long-running step yourself: Isaac Sim install/startup routinely takes
> 5-8 minutes, cloud provisioning and training takes hours. Use a single
> Bash call with an `until <condition>; do sleep N; done` poll loop
> against a log file, completion marker, or `gcloud`/process status
> check — never a single impatient check followed by returning control
> to report "still running, waiting for it to notify." Chain further
> blocking Bash calls if a wait exceeds one call's timeout.**

(Reference: `feedback_subagent-block-dont-check-in-early.md` in this
project's memory system — this exact pattern has recurred multiple
times despite being documented; paste the instruction, don't paraphrase
or assume it's implied by other context.)

## 2. Cost cap (cloud tasks only)

> **Cost cap: $[N] cumulative across [this task / these tasks
> combined].** [Prior tasks in this batch have already spent ≈$X —
> ≈$Y remains.] Track by instance-uptime × the published SKU rate (no
> BigQuery billing export exists in this project — see
> `docs/cloud/franka-cloud-shakedown.md` for the current per-SKU rate).
> Check elapsed cost at least once before finishing and report it; if
> projected total is approaching the cap, stop and report to the
> controller rather than continuing.

## 3. Environment conventions (adapt wording, keep the substance)

- Cloud only: provision ONE instance, run jobs sequentially on it,
  full teardown at the end (verify zero instances/disks/snapshots —
  `scripts/check_cloud_state.sh` does this in one command).
- `docs/cloud/franka-cloud-shakedown.md` is the recipe of record.
- Cloud runs headless — the standing, confirmed exception to this
  project's local "never headless" rule.
- Local GPU work (if any in the same dispatch): wrap every Isaac Sim
  invocation with `flock -o /tmp/rl_isaac_sim.lock -c "..."` — the `-o`
  flag is mandatory (see CLAUDE.md's "Environment conventions" for why).
  Check `nvidia-smi --query-compute-apps=pid,used_memory
  --format=csv,noheader` before launching.
- If a cloud interruption happens, diagnose via `gcloud compute
  operations list --filter="targetLink~<instance>"` before choosing a
  recovery path — distinguish a genuine SPOT preemption
  (`compute.instances.preempted` system event) from a manual/controller
  stop (a `stop` operation) rather than assuming either.
- Isaac Sim's own known teardown hang (process alive, near-idle
  GPU/CPU, log already shows completion): safe to `kill -TERM` after
  verifying via `ps`/`nvidia-smi` that no real work is in flight —
  document in the report if this fires, don't treat it as silently
  routine.

## 4. Bug-handling discipline (every dispatch)

> If you find a real bug while doing this task — in the code you're
> touching, in a script you're using for verification, anywhere —
> **fix it and re-run whatever it affected to confirm the fix holds, in
> this same task.** Do not log it as a known issue and move on, and do
> not work around a bug that's already been diagnosed once in an
> earlier task (check `BACKLOG.md` and recent task reports first).

## Known infra gaps as of 2026-07-17 (fold into the recipe if they recur)

- **A SPOT preemption followed by `gcloud compute instances start` can
  come back `RUNNING` but stuck in a GRUB rescue prompt, never actually
  finishing boot** (found 2026-07-21, d8-antipodal-root-cause diagnostic
  task — confirmed via `gcloud compute instances get-serial-port-output`,
  which showed `error: can't find command` repeated at a `grub>` prompt
  instead of a normal boot log). Distinct from every previously-documented
  preemption case above (which all resumed cleanly with a plain restart) —
  this is a genuine boot-disk corruption, not a preemption-recovery
  scenario a resume script can work around. No interactive GRUB-repair
  attempted (no established recipe for this project, and each attempt
  costs real wall-clock on a real running instance); the pragmatic
  response that worked: delete the stuck instance immediately (verify zero
  leftover resources via `scripts/check_cloud_state.sh`), re-provision a
  fresh instance from scratch, and treat any local-disk-only progress
  since the last GCS sync as lost. **Mitigation for future long training
  runs**: run a background loop (`while true; do gsutil -m rsync -r
  <run-dir> gs://.../ ; sleep 300; done` in its own detached `tmux`
  session, alongside the training session) that periodically syncs
  checkpoints/tfevents to GCS throughout the run, not just once at the
  end — so a repeat of this exact failure mode loses at most one sync
  interval of progress instead of the entire run. On-demand provisioning
  was tried first as an alternative to avoid the preemption that triggers
  this failure mode at all, but was fully stocked out project-wide across
  10 surveyed zones at dispatch time (`us-central1-a/b/c`, `us-east1-b/c/d`,
  `us-west1-a/b/c`, `us-west4-a/c`) — SPOT was the only viable option, so
  this gap should be assumed live for any future SPOT dispatch until a
  real GRUB-recovery recipe exists (not yet built).
- A SPOT preemption mid-install can corrupt the shared pip wheel cache
  on the boot disk — a resumed install may need a cache clear, not just
  a plain re-run.
- A default `Linger=no` systemd setting can kill a detached `tmux`
  session running a long install unpredictably when the SSH session
  that spawned it ends — check/set lingering explicitly for any
  long-running detached install.
- Not yet built: a pre-baked VM image with Isaac Sim/Isaac Lab already
  installed, which would eliminate the fragile ~15-20min from-scratch
  install window entirely (see `BACKLOG.md`'s "Cloud infrastructure
  reliability" entry — this is the highest-leverage fix still open).
- **A SPOT preemption can truncate a checkpoint file mid-write, leaving
  a 0-byte `.pt` on disk** (found 2026-07-19, Task 3.5 cloud completion —
  `rsl_rl`'s `save_interval` checkpoint write landed exactly at the
  preemption instant). A naive "resume from the highest iteration number
  present" strategy will pick this corrupt file and fail with
  `EOFError: Ran out of input` on `torch.load`. Any resume logic must
  validate the candidate checkpoint (at minimum a file-size sanity check
  — a real checkpoint for this project's PPO configs is ~1.27MB, so
  anything under ~100KB is certainly corrupt — or a real `torch.load`
  try/except) before trusting it, and fall back to the next-most-recent
  checkpoint if the top one is corrupt.
- **Repeated SPOT preemptions can occur in clusters far above this
  project's earlier "1-2 per multi-hour run" experience** — hit 3
  preemptions in ~3 hours during Task 3.5's cloud completion (2026-07-19),
  each independently confirmed via `gcloud compute operations list` as a
  genuine `compute.instances.preempted` system event, not a stockout or
  manual stop. When this recurs, switching the remaining jobs to
  on-demand provisioning (drop `--provisioning-model=SPOT`/
  `--instance-termination-action=STOP`) is a reasonable judgment call to
  stop losing wall-clock to snapshot-recover-resume cycles, provided the
  remaining job count is small enough that on-demand's ~2x hourly rate
  stays well within the task's cost cap — not a general policy change,
  reconsider fresh each time.
- **Switching an existing SPOT instance to on-demand mid-task (2026-07-20
  finding, d8/d10 H2 task):** `gcloud compute instances set-scheduling
  --no-preemptible --provisioning-model=STANDARD
  --clear-instance-termination-action --restart-on-failure` works on a
  *stopped* instance without needing to delete/recreate it (checkpoints
  on the boot disk are preserved) — cheaper than the snapshot-and-migrate
  recipe below when you just want to drop SPOT, not also change zone.
  **Gotcha: do not also pass `--maintenance-policy=MIGRATE`** — GPU-attached
  instances reject it (`onHostMaintenance` must stay `TERMINATE`
  regardless of provisioning model; only non-GPU instances can migrate
  live). Also confirmed: switching to on-demand does **not** bypass the
  shared project-wide `GPUS_ALL_REGIONS` quota — it only protects an
  instance from *preemption* once it has actually acquired the GPU, not
  from contention with another concurrent workstream trying to acquire
  the same single project-wide GPU slot first.
- **Preemption clustering can get much worse than the already-documented
  "3 in ~3 hours"** (found 2026-07-21, AR4 Franka-fixes-transfer Task 4,
  Condition A 3-seed training): hit **6 genuine preemptions across ~3h45m**
  on the training instance (`us-central1-c`), each independently confirmed
  via `gcloud compute operations list` as a real
  `compute.instances.preempted` event, plus 2 more on a second, separate
  diagnostic-sweep instance in the same zone shortly after. **On-demand was
  also stocked out** (`resource_availability`/`STOCKOUT`) in this same zone
  at the time the 2026-07-20 mitigation above was attempted — the on-demand
  fallback is not always available even when you're willing to pay ~2x;
  falling back further to "just keep resuming on SPOT" (via `--checkpoint`,
  see the scripts/train.py fix this same task added) is a legitimate
  degraded-mode response when on-demand itself is unavailable, not only
  when it's undesirable on cost grounds. A brief retry loop against
  `gcloud compute instances start` (10-20s spacing, up to ~15 attempts) has
  reliably found a moment of free SPOT capacity within a few minutes every
  time this was tried, even during this same stockout window — a single
  failed `start` attempt is not evidence the zone is unrecoverable for a
  while.
- **A SPOT restart can come back with a broken NVIDIA driver even though
  the instance itself boots fine** (found 2026-07-21, same task): `nvidia-smi`
  reported `No devices were found` / `couldn't communicate with the NVIDIA
  driver` after a routine preemption-restart, root-caused to an
  automatic kernel upgrade (apt pulled a newer `linux-image-*-gcp` between
  the instance's original boot and this restart) whose matching
  `linux-modules-nvidia-580-server-open-<kernel>-gcp` package had been left
  in an interrupted/broken (`iF`) dpkg state (`sudo apt-get install
  --reinstall` itself fails with "dpkg was interrupted, you must manually
  run 'sudo dpkg --configure -a'" until that's done first). Fix: `sudo dpkg
  --configure -a` (finishes configuring the pending nvidia kernel-module
  package and regenerates the initramfs/grub entries for the new kernel),
  then a plain `sudo reboot` — `nvidia-smi` and the GPU PCI device both came
  back healthy immediately after. `lspci | grep -i nvidia` showing the GPU
  present as a PCI device while `nvidia-smi`/`lsmod | grep nvidia` show
  nothing is the distinguishing signature (driver/kernel-module problem,
  not a GPU-detach/quota problem) — check `dpkg -l | grep linux-modules-nvidia`
  for an `iF`/broken state first before assuming a deeper GPU-attachment
  issue.
- **A blocking `zenity` "Kit appears to be hanging" GUI dialog can pop
  mid-training on an unattended, non-headless run and freeze it
  indefinitely** (found 2026-07-21, `ar4-franka-fixes-transfer` Task 6,
  desktop dispatch, condition-B/seed42, ~iteration 1461/1500). This is
  Kit's own internal hang-watchdog — distinct from CLAUDE.md's already-
  documented "hangs quietly during teardown, flock lock still held, no
  visible symptom" pattern: here the process is not silently stuck, it is
  actively blocked on a GUI dialog with nobody present to click it, on a
  run this project's own "run non-headless — the user wants to watch"
  convention deliberately keeps non-headless. If a queued/dispatched
  non-headless job seems stuck for an unusually long time, check for a
  live `zenity` process (not just the training process's own CPU/GPU
  activity) before assuming a teardown hang. Recovery: kill the `zenity`
  dialog process first; if the training process doesn't resume cleanly on
  its own afterward, `kill -9` it and resume from the last good checkpoint
  via `--checkpoint` (as done here: `model_1450.pt` → clean completion at
  `model_1499.pt`, no data lost). No fix has been built to suppress the
  watchdog dialog itself (e.g. a Kit/Isaac Sim launch flag) — flagged here
  as a recognize-and-recover pattern, not yet a prevention.

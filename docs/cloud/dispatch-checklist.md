# Cloud-task dispatch checklist

Copy the relevant blocks below **verbatim** into any subagent dispatch
that provisions a GCP cloud instance or launches Isaac Sim (local or
remote). Written after this project hit the same avoidable failures
across three separate dispatches in one session (2026-07-16/17) because
these were reconstructed from memory each time instead of copied from a
canonical source — see `BACKLOG.md` and `AUTONOMY.md`'s 2026-07-17
entries.

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

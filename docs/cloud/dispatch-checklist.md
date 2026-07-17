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

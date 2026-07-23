#!/usr/bin/env bash
# Dispatch an arbitrary command to run on a fresh, ephemeral GCP GPU
# instance, BLOCKING until the job is actually done -- this is the cloud
# analog of scripts/run_on_desktop_gpu.sh (same exit-code/blocking
# philosophy; read that script's own header first, this one mirrors its
# spirit rather than every exact flag/mechanism, since cloud has no
# persistent always-on machine to dispatch to -- every job provisions and
# tears down its own instance).
#
# Usage:
#   scripts/run_on_cloud_gpu.sh [--detach] [--cost-cap DOLLARS] <command> [args...]
#
#   --detach       Dispatch and return immediately once the remote job has
#                  actually started (repo shipped, tmux session confirmed
#                  up), without waiting for it to finish or tearing down
#                  the instance. Prints the instance name/zone and the
#                  exact commands to check on it, tail its log, and tear
#                  it down manually later. Default (no flag) is to BLOCK
#                  and stream the job's output live until it finishes,
#                  then tear the instance down automatically -- this
#                  project's convention (CLAUDE.md / docs/cloud/
#                  dispatch-checklist.md) is to block on long-running
#                  dispatches rather than check in early, so that is the
#                  default here.
#
#   --cost-cap N   Abort the job and tear down the instance if tracked
#                  cost (instance-uptime x published SKU rate -- this
#                  project has no GCS billing export, see
#                  docs/cloud/franka-cloud-shakedown.md's "Cost" section,
#                  so duration x rate is the standing method) exceeds $N.
#                  Checked continuously while blocking; not enforced in
#                  --detach mode (nothing is watching after this script
#                  returns -- check manually, e.g. via
#                  `scripts/check_cloud_state.sh` or a repeat call to this
#                  script's own cost-tracking logic against the instance's
#                  creationTimestamp).
#
# Example:
#   scripts/run_on_cloud_gpu.sh echo hello
#   scripts/run_on_cloud_gpu.sh --cost-cap 0.50 bash -c 'cd ~/rl && ./isaaclab.sh -p scripts/train_franka.py --headless'
#   scripts/run_on_cloud_gpu.sh --detach --cost-cap 5.00 bash long_training.sh
#
# WHAT THIS SCRIPT DOES (mirrors docs/cloud/franka-cloud-shakedown.md's
# proven recipe verbatim -- does not reinvent it):
#   1. Pre-check: refuses to dispatch if ANY GCP compute instance already
#      exists project-wide (this repo's standing convention is one
#      instance at a time, see dispatch-checklist.md item 3) -- avoids
#      double-provisioning against a concurrent workstream's own instance.
#   2. Provisions ONE SPOT g2-standard-4 + 1x nvidia-l4 instance (image
#      common-cu129-ubuntu-2204-nvidia-580, 150GB pd-balanced boot disk,
#      storage-rw scope, maintenance-policy=TERMINATE), trying the zones
#      this repo has previously confirmed have NVIDIA_L4_GPUS quota, in
#      order, falling back on a ZONE_RESOURCE_POOL_EXHAUSTED stockout.
#   3. Waits for SSH to come up, then ships this repo's current working
#      tree via `git archive HEAD | ... tar -x` into ~/rl on the instance
#      (the proven recipe's own shipping method -- NOT a git clone, this
#      is a private repo with no deploy key set up on cloud instances).
#      Your <command...> can assume `~/rl` exists and is your working
#      directory; it does NOT assume Isaac Sim/Isaac Lab/any Python env is
#      installed -- that install takes ~15-20min from scratch per the
#      recipe and is deliberately NOT baked into every dispatch (most
#      dispatches don't need it, e.g. a cheap diagnostic or a job on an
#      image that already has what it needs). If your command needs Isaac
#      Sim, have it install first (see franka-cloud-shakedown.md) or chain
#      `&&` onto a setup script.
#   4. Runs your <command...> inside a detached tmux session on the
#      instance (survives an SSH disconnect from this side), tee'd to a
#      remote log.
#   5. Default (blocking) mode: streams that log back live via a separate
#      blocking SSH/tail call until a completion marker appears, watching
#      for (a) cost-cap breach, (b) a SPOT preemption (detected via
#      instance status going non-RUNNING while no completion marker has
#      appeared) -- on preemption, restarts the instance (retry loop, 10s
#      spacing, up to 15 attempts, per dispatch-checklist.md's documented
#      mitigation) and RE-RUNS your command FRESH in a new tmux session
#      (this is a restart, not a resume -- a generic wrapper cannot know
#      whether your command is safe to blindly resume; if your own command
#      needs checkpoint-resume behavior across a preemption, build that
#      into the command itself, e.g. train_franka.py's own --checkpoint
#      flag, the same way this repo already handles it for training jobs
#      specifically). Preemption-driven restarts are capped at 3 for the
#      whole job to bound cost from a preemption-restart loop; exceeding
#      that gives up (exit 7).
#   6. ALWAYS tears the instance down on completion in blocking mode
#      (success, remote-command failure, cost-cap breach, or preemption-
#      retry exhaustion) via an EXIT trap, then reports
#      `scripts/check_cloud_state.sh`'s output so a caller can see zero
#      resources remain without a separate call. Detach mode (see above)
#      deliberately does NOT tear down -- the job is still running.
#
#      IMPORTANT: if your command produces output you need to keep, sync
#      it yourself as the LAST step of your own <command...> (e.g. calling
#      scripts/sync_run_to_gcs.py) -- this wrapper deletes the instance
#      right after your command exits in blocking mode, before you get a
#      chance to do that separately. This is deliberate: what to persist
#      and where is task-specific, not something a general dispatch
#      wrapper should assume.
#
# Exit codes (deliberately distinct, same philosophy as
# run_on_desktop_gpu.sh's own 0-5 scheme, extended for cloud-specific
# failure modes):
#   0 = success: instance was free to provision, job dispatched, remote
#       command exited 0 (or, with --detach, the job was successfully
#       started).
#   1 = BUSY: a GCP compute instance already exists project-wide --
#       nothing was provisioned (this repo's one-instance-at-a-time
#       convention; could be another concurrent workstream's own job, or a
#       prior task's teardown that didn't run -- check
#       `scripts/check_cloud_state.sh` and investigate before retrying,
#       do NOT assume it's safe to delete).
#   2 = UNKNOWN: the busy pre-check itself failed (gcloud unreachable, not
#       authenticated, etc.) -- NOT a green light, same as
#       run_on_desktop_gpu.sh's UNKNOWN semantics.
#   3 = usage error (no command given, or --cost-cap given a non-numeric
#       value).
#   4 = dispatch/provisioning mechanism itself failed: no zone had
#       capacity/quota, SSH never came up within the wait budget, repo
#       shipping or remote script setup failed -- distinct from the
#       remote command's own exit code. The instance (if one was left
#       half-provisioned) is torn down before this exit.
#   5 = the remote command itself ran and exited non-zero. The exact
#       remote exit code is printed to stderr. The instance is torn down
#       before this exit.
#   6 = cost cap exceeded mid-run -- job aborted, instance torn down
#       before this exit.
#   7 = SPOT preemption retries exhausted (gave up after 3 preemption-
#       driven restarts) -- instance torn down before this exit.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- provisioning constants (docs/cloud/franka-cloud-shakedown.md's
# proven recipe, verbatim) -----------------------------------------------
MACHINE_TYPE="g2-standard-4"
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="150GB"
BOOT_DISK_TYPE="pd-balanced"
# Zones with confirmed NVIDIA_L4_GPUS quota as of 2026-07-13/21 (see
# franka-cloud-shakedown.md's "Quota / billing status" section) -- tried
# in order, falling back on a per-zone SPOT stockout.
ZONES=(us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c)

# $/hr, SPOT g2-standard-4 (CPU+RAM) + SPOT nvidia-l4 + 150GB pd-balanced,
# per franka-cloud-shakedown.md's "Cost: real per-SKU pricing" section
# (2026-07-14/15, via the Cloud Billing Catalog API): 0.01277*4 vCPU +
# 0.001496*16GiB + 0.2862 (GPU) + (150*0.10/730 disk-month-hours) =
# 0.05108 + 0.023936 + 0.2862 + 0.02055 ~= 0.382/hr. UPDATE THIS if the
# machine type/accelerator/disk size above ever changes -- it is not
# queried live (no billing API path for real-time SKU lookup was found;
# see the same doc section for why).
RATE_PER_HOUR="0.382"

SSH_EXTRA=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
DONE_MARKER="__RL_CLOUD_GPU_JOB_DONE__"
MAX_PREEMPTION_RESTARTS=3

# --- arg parsing ----------------------------------------------------------
DETACH=0
COST_CAP=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --cost-cap)
      shift
      COST_CAP="${1:-}"
      if [ -z "$COST_CAP" ] || ! [[ "$COST_CAP" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Usage error: --cost-cap requires a numeric dollar amount." >&2
        exit 3
      fi
      shift
      ;;
    *) break ;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [--detach] [--cost-cap DOLLARS] <command> [args...]" >&2
  exit 3
fi

CMD_QUOTED=""
for arg in "$@"; do
  CMD_QUOTED+="$(printf '%q ' "$arg")"
done

# --- state used by cleanup() ----------------------------------------------
INSTANCE_NAME=""
INSTANCE_ZONE=""
INSTANCE_CREATED=0
JOB_START_TS=0
TEARDOWN_DONE=0
STREAM_FIFO=""
STREAM_FD=""
SSH_TAIL_PID=""
LOCAL_TMP_SCRIPT=""

log() { echo "$@" >&2; }

current_cost() {
  # Prints an estimated $ cost so far for the current instance's uptime.
  if [ "$INSTANCE_CREATED" -ne 1 ] || [ "$JOB_START_TS" -eq 0 ]; then
    echo "0.00"
    return
  fi
  local now elapsed
  now="$(date +%s)"
  elapsed=$(( now - JOB_START_TS ))
  awk -v e="$elapsed" -v r="$RATE_PER_HOUR" 'BEGIN { printf "%.4f", (e/3600.0)*r }'
}

cleanup() {
  # Idempotent -- safe to call more than once (EXIT trap + explicit calls
  # on every code path below). Never tears down in --detach mode: the job
  # is deliberately left running.
  [ -n "${STREAM_FD:-}" ] && exec {STREAM_FD}<&- 2>/dev/null || true
  [ -n "$STREAM_FIFO" ] && rm -f "$STREAM_FIFO" 2>/dev/null || true
  [ -n "$SSH_TAIL_PID" ] && kill "$SSH_TAIL_PID" 2>/dev/null || true
  [ -n "$LOCAL_TMP_SCRIPT" ] && rm -f "$LOCAL_TMP_SCRIPT" 2>/dev/null || true

  if [ "$DETACH" -eq 1 ]; then
    return
  fi
  if [ "$TEARDOWN_DONE" -eq 1 ]; then
    return
  fi
  TEARDOWN_DONE=1
  if [ "$INSTANCE_CREATED" -eq 1 ] && [ -n "$INSTANCE_NAME" ] && [ -n "$INSTANCE_ZONE" ]; then
    log "Tearing down instance '${INSTANCE_NAME}' (zone ${INSTANCE_ZONE})..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" --quiet >&2 2>&1 || \
      log "WARNING: instance delete command itself failed -- check scripts/check_cloud_state.sh and clean up manually."
    log "Estimated cost this job: \$$(current_cost)"
    log "Post-teardown state (scripts/check_cloud_state.sh):"
    "$SCRIPT_DIR/check_cloud_state.sh" >&2 2>&1 || true
  fi
}
trap cleanup EXIT

# --- 1. busy pre-check: refuse if any instance already exists -------------
if ! EXISTING="$(gcloud compute instances list --format='value(name,zone,status)' 2>&1)"; then
  echo "REFUSING TO DISPATCH: could not query GCP instance state (gcloud error below)." >&2
  echo "$EXISTING" >&2
  echo "Not a green light -- investigate gcloud auth/network before retrying." >&2
  exit 2
fi
if [ -n "$EXISTING" ]; then
  echo "REFUSING TO DISPATCH: a GCP compute instance already exists (this project's convention is one instance at a time):" >&2
  echo "$EXISTING" >&2
  echo "Check scripts/check_cloud_state.sh and investigate before retrying -- it may be another concurrent workstream's own active job, do NOT delete it unilaterally." >&2
  exit 1
fi

# --- 2. provision, trying each zone in order ------------------------------
INSTANCE_NAME="rl-cloud-gpu-job-$(date -u +%Y%m%d-%H%M%S)-$$"
CREATE_OK=0
for zone in "${ZONES[@]}"; do
  log "Attempting to provision '${INSTANCE_NAME}' in ${zone}..."
  if CREATE_OUT="$(gcloud compute instances create "$INSTANCE_NAME" \
      --zone="$zone" \
      --machine-type="$MACHINE_TYPE" \
      --provisioning-model=SPOT \
      --instance-termination-action=STOP \
      --accelerator="$ACCELERATOR" \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="$BOOT_DISK_SIZE" \
      --boot-disk-type="$BOOT_DISK_TYPE" \
      --metadata=install-nvidia-driver=True \
      --scopes=storage-rw \
      --maintenance-policy=TERMINATE 2>&1)"; then
    CREATE_OK=1
    INSTANCE_ZONE="$zone"
    INSTANCE_CREATED=1
    JOB_START_TS="$(date +%s)"
    log "Provisioned in ${zone}."
    break
  fi
  log "Failed in ${zone}:"
  log "$CREATE_OUT"
  if echo "$CREATE_OUT" | grep -qi "ZONE_RESOURCE_POOL_EXHAUSTED"; then
    log "(stockout, trying next zone)"
    continue
  fi
  if echo "$CREATE_OUT" | grep -qi "quota"; then
    log "(quota error -- project-wide, will not clear by trying another zone; aborting zone loop)"
    break
  fi
  # Unrecognized error: keep trying remaining zones rather than giving up
  # early on something zone-specific we don't have a name for yet.
done

if [ "$CREATE_OK" -ne 1 ]; then
  echo "DISPATCH FAILED: could not provision an instance in any surveyed zone." >&2
  exit 4
fi

# --- 3. wait for SSH, then ship the repo ----------------------------------
log "Waiting for SSH..."
SSH_READY=0
for _ in $(seq 1 30); do
  if gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" --command "true" -- "${SSH_EXTRA[@]}" >/dev/null 2>&1; then
    SSH_READY=1
    break
  fi
  sleep 10
done
if [ "$SSH_READY" -ne 1 ]; then
  echo "DISPATCH FAILED: SSH never came up on ${INSTANCE_NAME} within 5 minutes." >&2
  exit 4
fi
log "SSH ready."

# retry(): a plain SSH connection right after an instance first becomes
# reachable can still transiently fail (found live, 2026-07-23 verification
# testing: the readiness probe above succeeded, but the very next SSH
# connection -- for the repo-ship step -- hit "Connection timed out",
# a few seconds later succeeding fine on manual retry). Wrap every
# subsequent multi-step SSH operation in a few retries rather than treating
# one flaky connection as a hard dispatch failure.
retry() {
  local attempts="$1"; shift
  local i
  for i in $(seq 1 "$attempts"); do
    if "$@"; then
      return 0
    fi
    log "Attempt ${i}/${attempts} failed, retrying in 5s..."
    sleep 5
  done
  return 1
}

ship_repo() {
  log "Shipping repo (git archive) to ~/rl on the instance..."
  if ! (cd "$REPO_ROOT" && git archive HEAD) | gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
      --command "mkdir -p ~/rl && tar -x -C ~/rl" -- "${SSH_EXTRA[@]}" >&2 2>&1; then
    return 1
  fi
  return 0
}
if ! retry 3 ship_repo; then
  echo "DISPATCH FAILED: could not ship repo to instance after retries." >&2
  exit 4
fi

# --- helper: (re)launch the job in a fresh tmux session on the instance --
REMOTE_SCRIPT="/tmp/rl-cloud-job.sh"
REMOTE_LOG="/tmp/rl-cloud-job.log"
SESSION="rl-cloud-job"

launch_job() {
  LOCAL_TMP_SCRIPT="$(mktemp)"
  cat > "$LOCAL_TMP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd ~/rl 2>/dev/null || true
${CMD_QUOTED}
echo "${DONE_MARKER} exit=\$?"
EOF
  if ! gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
      --command "cat > ${REMOTE_SCRIPT}" -- "${SSH_EXTRA[@]}" < "$LOCAL_TMP_SCRIPT" >&2 2>&1; then
    return 1
  fi
  rm -f "$LOCAL_TMP_SCRIPT"; LOCAL_TMP_SCRIPT=""

  if ! gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
      --command "which tmux >/dev/null 2>&1 || (sudo apt-get update -y && sudo apt-get install -y tmux)" \
      -- "${SSH_EXTRA[@]}" >&2 2>&1; then
    return 1
  fi

  if ! gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
      --command "rm -f ${REMOTE_LOG}; tmux new-session -d -s ${SESSION} \"bash ${REMOTE_SCRIPT} > ${REMOTE_LOG} 2>&1\"" \
      -- "${SSH_EXTRA[@]}" >&2 2>&1; then
    return 1
  fi

  local confirmed=0
  for _ in $(seq 1 10); do
    if gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
        --command "tmux has-session -t ${SESSION} 2>/dev/null || test -e ${REMOTE_LOG}" \
        -- "${SSH_EXTRA[@]}" >/dev/null 2>&1; then
      confirmed=1
      break
    fi
    sleep 0.5
  done
  [ "$confirmed" -eq 1 ]
}

log "Launching job in remote tmux session '${SESSION}'..."
if ! retry 3 launch_job; then
  echo "DISPATCH FAILED: could not start remote tmux session." >&2
  exit 4
fi
log "Job dispatched: ${CMD_QUOTED}"

if [ "$DETACH" -eq 1 ]; then
  echo "Detach mode: not waiting for completion, instance left running." >&2
  echo "Check later with:" >&2
  echo "  gcloud compute ssh ${INSTANCE_NAME} --zone ${INSTANCE_ZONE} --command 'tmux attach -t ${SESSION}'" >&2
  echo "  gcloud compute ssh ${INSTANCE_NAME} --zone ${INSTANCE_ZONE} --command 'tail -f ${REMOTE_LOG}'" >&2
  echo "  scripts/check_cloud_state.sh    # confirm still running" >&2
  echo "Tear down manually when done:" >&2
  echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone ${INSTANCE_ZONE} --quiet" >&2
  exit 0
fi

# --- 4. blocking mode: stream the log, watch cost cap + preemption -------
instance_status() {
  gcloud compute instances describe "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
    --format='value(status)' 2>/dev/null || echo "UNKNOWN"
}

start_tail() {
  STREAM_FIFO="$(mktemp -u)"
  mkfifo "$STREAM_FIFO"
  gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" \
    --command "tail -n +1 -f ${REMOTE_LOG}" -- "${SSH_EXTRA[@]}" > "$STREAM_FIFO" 2>/dev/null &
  SSH_TAIL_PID=$!
  # Open the FIFO for reading ONCE here and keep this fd (STREAM_FD) open
  # for the whole life of the stream, instead of letting the main loop's
  # `read ... < "$STREAM_FIFO"` open/close a fresh fd on every single call
  # (found live, 2026-07-23, ar4-jaw-bisector-hypothesis task dispatch: a
  # real bug, not a hypothetical). Re-opening per read creates a genuine
  # reader-count-drops-to-zero race: in the gap between one `read` call
  # closing its own fd and the next iteration's `read` reopening it, ANY
  # write attempt by the writer (the backgrounded `tail -f`/ssh pipe above)
  # gets an immediate SIGPIPE - once a FIFO has had an attached reader,
  # POSIX semantics do NOT make a writer block waiting for a new reader if
  # the reader count transiently hits zero, it delivers SIGPIPE right away.
  # This silently kills the tail stream with nothing in the main loop ever
  # checking whether SSH_TAIL_PID is still alive - the observed symptom was
  # the locally-streamed log going permanently silent very early in a run
  # while the REMOTE job kept making real, eventually-successful progress
  # (only discovered via an out-of-band direct SSH check of the raw remote
  # log, which showed the job had long since moved far past where the
  # local stream had stalled). Holding one persistent reader fd open for
  # the FIFO's entire lifetime removes the zero-reader window entirely.
  exec {STREAM_FD}<"$STREAM_FIFO"
}

stop_tail() {
  [ -n "$SSH_TAIL_PID" ] && kill "$SSH_TAIL_PID" 2>/dev/null || true
  [ -n "$SSH_TAIL_PID" ] && wait "$SSH_TAIL_PID" 2>/dev/null || true
  SSH_TAIL_PID=""
  if [ -n "${STREAM_FD:-}" ]; then
    exec {STREAM_FD}<&- 2>/dev/null || true
    STREAM_FD=""
  fi
  [ -n "$STREAM_FIFO" ] && rm -f "$STREAM_FIFO" || true
  STREAM_FIFO=""
}

check_cost_cap() {
  if [ -z "$COST_CAP" ]; then
    return 0
  fi
  local cost
  cost="$(current_cost)"
  if awk -v c="$cost" -v cap="$COST_CAP" 'BEGIN { exit !(c >= cap) }'; then
    echo "COST CAP EXCEEDED: \$${cost} >= \$${COST_CAP} cap." >&2
    return 1
  fi
  return 0
}

start_tail
REMOTE_EXIT=""
PREEMPTION_RESTARTS=0
RESULT_CODE=""

while true; do
  if IFS= read -r -t 20 -u "$STREAM_FD" line; then
    echo "$line"
    if [[ "$line" == "${DONE_MARKER} exit="* ]]; then
      REMOTE_EXIT="${line#${DONE_MARKER} exit=}"
      break
    fi
  fi

  if ! check_cost_cap; then
    RESULT_CODE=6
    break
  fi

  status="$(instance_status)"
  if [ "$status" != "RUNNING" ]; then
    log "Instance status is '${status}' (not RUNNING) -- checking for SPOT preemption..."
    stop_tail
    if [ "$PREEMPTION_RESTARTS" -ge "$MAX_PREEMPTION_RESTARTS" ]; then
      log "Preemption-restart budget (${MAX_PREEMPTION_RESTARTS}) exhausted -- giving up."
      RESULT_CODE=7
      break
    fi
    PREEMPTION_RESTARTS=$((PREEMPTION_RESTARTS + 1))
    log "Attempting restart (${PREEMPTION_RESTARTS}/${MAX_PREEMPTION_RESTARTS})..."
    RESTARTED=0
    for _ in $(seq 1 15); do
      if gcloud compute instances start "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" >&2 2>&1; then
        RESTARTED=1
        break
      fi
      sleep 15
    done
    if [ "$RESTARTED" -ne 1 ]; then
      log "Could not restart instance after preemption."
      RESULT_CODE=7
      break
    fi
    # Wait for SSH again, then re-launch the job FRESH (not a resume --
    # see header comment; this restarts your command from the start).
    SSH_READY=0
    for _ in $(seq 1 30); do
      if gcloud compute ssh "$INSTANCE_NAME" --zone "$INSTANCE_ZONE" --command "true" -- "${SSH_EXTRA[@]}" >/dev/null 2>&1; then
        SSH_READY=1
        break
      fi
      sleep 10
    done
    if [ "$SSH_READY" -ne 1 ] || ! retry 3 launch_job; then
      log "Could not re-establish job after restart."
      RESULT_CODE=7
      break
    fi
    log "Job re-launched fresh after preemption-restart ${PREEMPTION_RESTARTS}."
    start_tail
  fi
done

stop_tail

if [ -n "$RESULT_CODE" ]; then
  exit "$RESULT_CODE"
fi

if [ -z "$REMOTE_EXIT" ]; then
  echo "DISPATCH WARNING: log stream ended without a completion marker." >&2
  exit 4
fi

if [ "$REMOTE_EXIT" -eq 0 ]; then
  echo "Remote command succeeded (exit 0)." >&2
  exit 0
else
  echo "REMOTE COMMAND FAILED: exited with code ${REMOTE_EXIT} on the instance." >&2
  exit 5
fi

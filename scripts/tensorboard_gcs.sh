#!/usr/bin/env bash
# Serve TensorBoard over the training runs synced to GCS.
#
# Default mode mirrors gs://rl-manipulation-hks-runs to a local cache with
# `gcloud storage rsync` (reuses existing gcloud auth - no ADC setup needed),
# points TensorBoard at the cache, and re-syncs on an interval while running,
# so runs landing in the bucket (e.g. from a cloud instance) show up without
# a restart. Checkpoints/videos are excluded from the mirror; TensorBoard
# only needs the event files.
#
# The cache is additive: runs deleted from the bucket stay in the local
# cache until you delete the cache dir yourself.
#
# Usage:
#   scripts/tensorboard_gcs.sh [EXPERIMENT] [--port N] [--sync-interval SECS]
#   scripts/tensorboard_gcs.sh --live INSTANCE [--port N]
#
# EXPERIMENT narrows the dashboard to one experiment prefix in the bucket
# (e.g. joint-space-die-lift); omit it to browse everything.
#
# --live skips GCS entirely: it opens an SSH tunnel to a GCP instance that
# is already running `tensorboard --logdir logs/ --port 6006`, for
# second-by-second curves during an active cloud run.
set -uo pipefail

BUCKET="gs://rl-manipulation-hks-runs"
CACHE="${HOME}/.cache/rl-manipulation-hks-runs"
VENV="${HOME}/.venvs/tensorboard"
PORT=6006
SYNC_INTERVAL=60
EXPERIMENT=""
LIVE_INSTANCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --sync-interval) SYNC_INTERVAL="$2"; shift 2 ;;
    --live) LIVE_INSTANCE="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) EXPERIMENT="$1"; shift ;;
  esac
done

if [ -n "$LIVE_INSTANCE" ]; then
  echo "Tunneling localhost:${PORT} -> ${LIVE_INSTANCE}:6006 (Ctrl+C to stop)."
  echo "The instance must already be running: tensorboard --logdir logs/ --port 6006"
  exec gcloud compute ssh "$LIVE_INSTANCE" -- -NL "${PORT}:localhost:6006"
fi

if [ ! -x "${VENV}/bin/tensorboard" ]; then
  echo "Bootstrapping TensorBoard venv at ${VENV}..."
  python3 -m venv "$VENV" && "${VENV}/bin/pip" install --quiet tensorboard || {
    echo "Failed to set up TensorBoard venv." >&2
    exit 1
  }
fi

mkdir -p "$CACHE"

sync_bucket() {
  gcloud storage rsync -r \
    -x '.*\.(pt|pth|hdf5|mp4|onnx)$' \
    "$BUCKET" "$CACHE" >/dev/null
}

echo "Syncing ${BUCKET} -> ${CACHE}..."
sync_bucket || { echo "Initial GCS sync failed." >&2; exit 1; }

# Re-sync in the background while TensorBoard runs; TensorBoard picks up
# new event files on its own reload cycle.
(
  while sleep "$SYNC_INTERVAL"; do
    sync_bucket || true
  done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

LOGDIR="${CACHE}${EXPERIMENT:+/${EXPERIMENT}}"
if [ ! -d "$LOGDIR" ]; then
  echo "No such experiment in the bucket cache: ${EXPERIMENT}" >&2
  echo "Available:" >&2
  ls "$CACHE" >&2
  exit 1
fi

echo "TensorBoard: http://localhost:${PORT} (logdir ${LOGDIR}, re-sync every ${SYNC_INTERVAL}s)"
"${VENV}/bin/tensorboard" --logdir "$LOGDIR" --port "$PORT" 2>&1

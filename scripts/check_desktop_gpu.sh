#!/usr/bin/env bash
# Reports whether the desktop's GPU (saps@home.local, HTTP status server
# on port 8077) is free for a new job dispatched from this Pi.
#
# Queries scripts/gpu_status_server.py's /gpu-status endpoint over HTTP
# instead of SSHing to the desktop twice (nvidia-smi + systemd-inhibit
# --list) -- see docs/superpowers/specs/2026-07-18-gpu-status-server-design.md.
# Judgment call is unchanged from the SSH-based version: non-empty
# compute_apps OR a non-zero rl_gpu_job_guard_count (covers both the
# Pi-dispatched `rl-gpu-job` guard and the desktop's own auto-detect
# `rl-gpu-job-auto-detect` guard) means BUSY.
#
# Exit codes are distinct on purpose so a caller can branch correctly
# instead of treating every non-zero the same way:
#   0 = AVAILABLE (desktop reachable, GPU idle, no job guard held)
#   1 = BUSY       (desktop reachable, but GPU or job guard in use)
#   2 = UNKNOWN     (desktop unreachable or the check itself failed --
#                    NOT the same as available; caller must not treat
#                    this as a green light, fall back to cloud or stop)
#
# No SSH fallback if the HTTP server is down -- direct project decision
# (see the design doc): trust systemd Restart=always + boot-start to keep
# it up, and treat any failure to reach it as UNKNOWN rather than risking
# a false green light.
#
# Usage: scripts/check_desktop_gpu.sh
set -Eeuo pipefail

STATUS_URL="http://home.local:8077/gpu-status"

fail_unknown() {
  echo "UNKNOWN: $1" >&2
  exit 2
}

if ! RESPONSE="$(curl --silent --show-error --fail \
    --connect-timeout 3 --max-time 5 --retry 2 --retry-delay 1 \
    "$STATUS_URL" 2>&1)"; then
  fail_unknown "could not reach gpu status server at $STATUS_URL: $RESPONSE"
fi

if ! APPS_COUNT="$(echo "$RESPONSE" | jq '.compute_apps | length' 2>&1)"; then
  fail_unknown "malformed response from gpu status server: $APPS_COUNT"
fi

if ! GUARD_COUNT="$(echo "$RESPONSE" | jq '.rl_gpu_job_guard_count' 2>&1)"; then
  fail_unknown "malformed response from gpu status server: $GUARD_COUNT"
fi

if [ "$APPS_COUNT" -eq 0 ] && [ "$GUARD_COUNT" -eq 0 ]; then
  echo "AVAILABLE: desktop GPU idle, no rl-gpu-job guard active"
  exit 0
fi

echo "BUSY:"
echo "$RESPONSE" | jq -c '.compute_apps[]'
if [ "$GUARD_COUNT" -gt 0 ]; then
  echo "(rl-gpu-job shutdown guard is held -- a dispatched job is claiming the GPU even if nvidia-smi shows nothing yet)"
fi
exit 1

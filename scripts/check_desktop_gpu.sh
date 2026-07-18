#!/usr/bin/env bash
# Reports whether the desktop's GPU (saps@home.local, ssh alias "desktop")
# is free for a new job dispatched from this Pi.
#
# Judgment call is the compute-apps list, per this project's existing
# convention (CLAUDE.md "Monorepo layout & runtimes") of never trusting a
# process-name/path grep -- plus a check for this project's own
# systemd-inhibit shutdown guard (see run_on_desktop_gpu.sh), which can be
# held during a job's early startup window (Isaac Sim takes 5-8min to
# allocate GPU memory) before nvidia-smi shows any compute process yet.
#
# Exit codes are distinct on purpose so a caller can branch correctly
# instead of treating every non-zero the same way:
#   0 = AVAILABLE (desktop reachable, GPU idle, no job guard held)
#   1 = BUSY       (desktop reachable, but GPU or job guard in use)
#   2 = UNKNOWN     (desktop unreachable or a check itself failed --
#                    NOT the same as available; caller must not treat
#                    this as a green light, fall back to cloud or stop)
#
# Usage: scripts/check_desktop_gpu.sh
set -Eeuo pipefail

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
ERR_LOG="$(mktemp)"
trap 'rm -f "$ERR_LOG"' EXIT

fail_unknown() {
  echo "UNKNOWN: $1" >&2
  if [ -s "$ERR_LOG" ]; then
    echo "--- last command stderr ---" >&2
    cat "$ERR_LOG" >&2
  fi
  exit 2
}

if ! ssh "${SSH_OPTS[@]}" desktop true 2>"$ERR_LOG"; then
  fail_unknown "cannot SSH to desktop (powered off, asleep, or network down)"
fi

if ! APPS="$(ssh "${SSH_OPTS[@]}" desktop \
    'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader' \
    2>"$ERR_LOG")"; then
  fail_unknown "nvidia-smi query failed on desktop"
fi

if ! GUARD_COUNT="$(ssh "${SSH_OPTS[@]}" desktop \
    'systemd-inhibit --list --no-legend 2>/dev/null | grep -c rl-gpu-job' \
    2>"$ERR_LOG")"; then
  # grep -c exits 1 (not an error) when there are zero matches -- only
  # treat this as unknown if it actually produced no usable count.
  if [ -z "${GUARD_COUNT:-}" ]; then
    fail_unknown "systemd-inhibit --list check failed on desktop"
  fi
fi
GUARD_COUNT="${GUARD_COUNT:-0}"

if [ -z "$APPS" ] && [ "$GUARD_COUNT" -eq 0 ]; then
  echo "AVAILABLE: desktop GPU idle, no rl-gpu-job guard active"
  exit 0
fi

echo "BUSY:"
[ -n "$APPS" ] && echo "$APPS"
if [ "$GUARD_COUNT" -gt 0 ]; then
  echo "(rl-gpu-job shutdown guard is held -- a dispatched job is claiming the GPU even if nvidia-smi shows nothing yet)"
fi
exit 1

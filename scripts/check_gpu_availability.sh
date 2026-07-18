#!/usr/bin/env bash
# Combined GPU dispatch-target routing check: desktop first, cloud fallback.
#
# Calls check_desktop_gpu.sh. If it reports AVAILABLE, routes to the
# desktop. If it reports BUSY or UNKNOWN, routes to cloud -- UNKNOWN
# (unreachable desktop / check itself failed) is deliberately treated the
# same as BUSY here, not as a green light for the desktop: this project's
# convention (see check_desktop_gpu.sh) is that "can't tell" must never be
# read as "available."
#
# This script does NOT provision cloud infrastructure itself -- it only
# makes the routing decision. Once it reports TARGET=cloud, follow this
# project's existing cloud path: docs/cloud/dispatch-checklist.md (the
# blocking/cost-cap/teardown instructions to copy into a dispatch prompt)
# and docs/cloud/franka-cloud-shakedown.md (the recipe of record for
# actually provisioning + running on GCP).
#
# Output (stdout): a machine-parseable `TARGET=desktop` or `TARGET=cloud`
# line first, then a human-readable reason on the following line(s). A
# caller/subagent can do:
#   TARGET_LINE="$(scripts/check_gpu_availability.sh)"
#   TARGET="$(echo "$TARGET_LINE" | grep -o 'TARGET=[a-z]*' | cut -d= -f2)"
#
# Exit codes:
#   0 = TARGET=desktop (desktop is available, dispatch there via
#       run_on_desktop_gpu.sh)
#   1 = TARGET=cloud, because desktop is BUSY
#   2 = TARGET=cloud, because desktop is UNKNOWN (unreachable/check failed)
# In all cases the printed TARGET= line is the authoritative routing
# decision; the exit code additionally distinguishes *why* cloud was
# chosen, mirroring check_desktop_gpu.sh's own 1-vs-2 distinction.
#
# Usage: scripts/check_gpu_availability.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if CHECK_OUT="$("$SCRIPT_DIR/check_desktop_gpu.sh" 2>&1)"; then
  CHECK_RC=0
else
  CHECK_RC=$?
fi

case "$CHECK_RC" in
  0)
    echo "TARGET=desktop"
    echo "Reason: desktop GPU is AVAILABLE."
    echo "$CHECK_OUT"
    exit 0
    ;;
  1)
    echo "TARGET=cloud"
    echo "Reason: desktop GPU is BUSY -- falling back to cloud."
    echo "$CHECK_OUT"
    echo "See docs/cloud/dispatch-checklist.md and docs/cloud/franka-cloud-shakedown.md for the cloud dispatch recipe."
    exit 1
    ;;
  *)
    echo "TARGET=cloud"
    echo "Reason: desktop availability is UNKNOWN (unreachable or check failed) -- treated as not-available, falling back to cloud rather than risking a false green light."
    echo "$CHECK_OUT"
    echo "See docs/cloud/dispatch-checklist.md and docs/cloud/franka-cloud-shakedown.md for the cloud dispatch recipe."
    exit 2
    ;;
esac

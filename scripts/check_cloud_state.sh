#!/usr/bin/env bash
# One-shot "is anything running or costing money right now" check.
#
# Bundles the GCP + local checks this project's own cloud-task workflow
# needs repeatedly: live compute instances, live disks, live snapshots
# (all three should be empty after a clean teardown, per CLAUDE.md/
# AUTONOMY.md's operational discipline), plus a local stray-Isaac-Sim-
# process/lock check (the same check CLAUDE.md's "Check stray Isaac Sim
# processes on crash" guidance calls for, bundled here so it's one
# command instead of remembering the separate `ps`/`lsof` incantation).
#
# Usage: scripts/check_cloud_state.sh
# No args, no GPU/Isaac Sim launch, safe to run any time.
set -euo pipefail

echo "=== GCP compute instances ==="
gcloud compute instances list --format="table(name,zone,status,creationTimestamp)" 2>&1

echo
echo "=== GCP persistent disks ==="
gcloud compute disks list --format="table(name,zone,status)" 2>&1

echo
echo "=== GCP snapshots ==="
gcloud compute snapshots list --format="table(name,status)" 2>&1

echo
echo "=== Local Isaac Sim processes ==="
if ps aux | grep -i isaac | grep -v grep; then
    echo "(above: live local Isaac Sim process(es) found)"
else
    echo "none"
fi

echo
echo "=== Local flock lock holder ==="
if lsof /tmp/rl_isaac_sim.lock 2>/dev/null; then
    echo "(above: lock is held)"
else
    echo "lock free"
fi

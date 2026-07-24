#!/usr/bin/env bash
# Download the cached AR4 asset (scripts/build_asset.py's output) from GCS
# instead of rebuilding it from the vendor ROS package + build_asset.py
# every cloud dispatch (~10-20min from scratch, plus the ~15-20min Isaac
# Sim/Isaac Lab pip install this whole container-based pipeline also
# replaces -- see docs/cloud/franka-cloud-shakedown.md and
# docs/cloud/dispatch-checklist.md). Companion to
# scripts/upload_ar4_asset_to_gcs.sh -- see that script's header for the
# GCS layout and versioning scheme this reads.
#
# Usage:
#   scripts/download_ar4_asset_from_gcs.sh            # use the LATEST pointer
#   scripts/download_ar4_asset_from_gcs.sh <git-sha>   # pin to a specific
#                                                        build_asset.py version
#
# IMPORTANT: rewrites assets/ar4_mk5/usd_path.txt to an ABSOLUTE path rooted
# at THIS checkout after downloading. scripts/build_asset.py writes that
# manifest file with the absolute path of the machine/checkout it was BUILT
# on (tasks/ar4/robot_cfg.py's _resolve_usd_path() reads it back verbatim
# and sys.exit()s if the path it names doesn't exist) -- a cached asset
# downloaded onto a different machine or a differently-pathed checkout
# would otherwise silently point at a path that only existed on the
# original build machine. The USD content itself is location-independent;
# only this small manifest text file needs rewriting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GCS_BASE="gs://rl-manipulation-hks-runs/assets/ar4_mk5"

SHA="${1:-}"
if [ -z "$SHA" ]; then
  SHA="$(gsutil cat "${GCS_BASE}/LATEST")"
  echo "No sha given -- using LATEST pointer: ${SHA}"
fi

GCS_SRC="${GCS_BASE}/${SHA}"

# Staleness check: compare the cached build against scripts/build_asset.py's
# CURRENT git sha in this checkout. Same git-archive limitation as
# upload_ar4_asset_to_gcs.sh -- BUILD_ASSET_SHA env var overrides if this
# checkout has no .git dir (e.g. a cloud instance shipped via `git archive`).
CURRENT_BUILD_SCRIPT_SHA="${BUILD_ASSET_SHA:-}"
if [ -z "$CURRENT_BUILD_SCRIPT_SHA" ]; then
  CURRENT_BUILD_SCRIPT_SHA="$(git log -1 --format=%H -- scripts/build_asset.py 2>/dev/null || true)"
fi
if [ -z "$CURRENT_BUILD_SCRIPT_SHA" ]; then
  echo "NOTE: cannot check cache staleness -- no .git dir in this checkout and" >&2
  echo "BUILD_ASSET_SHA is not set. Proceeding with sha ${SHA} unverified." >&2
elif [ "$SHA" != "$CURRENT_BUILD_SCRIPT_SHA" ]; then
  echo "WARNING: cached asset was built from scripts/build_asset.py @ ${SHA}," >&2
  echo "         but this checkout's scripts/build_asset.py is currently @ ${CURRENT_BUILD_SCRIPT_SHA}." >&2
  echo "         The cached asset may be STALE relative to the current build script." >&2
  echo "         Proceeding anyway -- rerun scripts/build_asset.py + upload_ar4_asset_to_gcs.sh to refresh if this matters." >&2
else
  echo "Cache freshness OK: sha ${SHA} matches this checkout's current scripts/build_asset.py."
fi

mkdir -p assets/ar4_mk5 assets/shapes
echo "Downloading cached AR4 asset from ${GCS_SRC}/ ..."
gsutil -m rsync -r "${GCS_SRC}/ar4_mk5" assets/ar4_mk5
gsutil -m rsync -r "${GCS_SRC}/shapes" assets/shapes

# Rewrite usd_path.txt for THIS checkout's absolute path -- see header.
echo "$REPO_ROOT/assets/ar4_mk5/ar4_mk5.usd" > assets/ar4_mk5/usd_path.txt

echo "Done. Asset restored to assets/ar4_mk5/ + assets/shapes/wedge.usd (build_asset.py sha ${SHA})."

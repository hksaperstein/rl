#!/usr/bin/env bash
# Upload a freshly-built AR4 asset (scripts/build_asset.py's output --
# assets/ar4_mk5/ + assets/shapes/wedge.usd) to this project's GCS bucket,
# so future cloud dispatches can download it in seconds instead of
# rebuilding it from the vendor ROS package + build_asset.py every time
# (~10-20min from scratch). Companion to scripts/download_ar4_asset_from_gcs.sh
# -- see that script's header for the download/restore side.
#
# Versioned by the git commit hash of scripts/build_asset.py ITSELF (not
# overall repo HEAD) -- so a future consumer can directly compare that hash
# against `git log -1 --format=%H -- scripts/build_asset.py` on their own
# checkout to tell whether a cached asset is stale relative to the CURRENT
# build script, independent of unrelated commits elsewhere in the repo.
#
# GCS layout written:
#   gs://rl-manipulation-hks-models/ar4_mk5/<build_asset.py-git-sha>/ar4_mk5/...   (mirrors assets/ar4_mk5/)
#   gs://rl-manipulation-hks-models/ar4_mk5/<build_asset.py-git-sha>/shapes/...     (mirrors assets/shapes/)
#   gs://rl-manipulation-hks-models/ar4_mk5/<build_asset.py-git-sha>/PROVENANCE.txt
#   gs://rl-manipulation-hks-models/ar4_mk5/LATEST   (plain text pointer, updated
#     LAST so a partially-uploaded version is never advertised as latest)
#
# Usage: scripts/upload_ar4_asset_to_gcs.sh
#   (run from the repo root, after scripts/build_asset.py has already
#   produced assets/ar4_mk5/ar4_mk5.usd + assets/shapes/wedge.usd locally --
#   or in a bind-mounted container sharing this same host filesystem)
#
# BUILD_ASSET_SHA env var override: a cloud instance whose repo was shipped
# via `git archive HEAD | tar -x` (this project's standard cloud-shipping
# method, see scripts/run_on_cloud_gpu.sh) has NO .git directory, so `git
# log` cannot run there -- the same limitation scripts/sync_run_to_gcs.py
# already documents for its own git_sha manifest field. If BUILD_ASSET_SHA
# is set, it is used verbatim instead of calling git.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f assets/ar4_mk5/ar4_mk5.usd ]; then
  echo "ERROR: assets/ar4_mk5/ar4_mk5.usd not found -- run scripts/build_asset.py first." >&2
  exit 1
fi
if [ ! -f assets/shapes/wedge.usd ]; then
  echo "ERROR: assets/shapes/wedge.usd not found -- run scripts/build_asset.py first." >&2
  exit 1
fi

BUILD_SCRIPT_SHA="${BUILD_ASSET_SHA:-}"
if [ -z "$BUILD_SCRIPT_SHA" ]; then
  BUILD_SCRIPT_SHA="$(git log -1 --format=%H -- scripts/build_asset.py 2>/dev/null || true)"
fi
if [ -z "$BUILD_SCRIPT_SHA" ]; then
  echo "ERROR: could not determine scripts/build_asset.py's git sha (no .git dir in this" >&2
  echo "checkout, e.g. a git-archive-shipped cloud instance) and BUILD_ASSET_SHA is not set." >&2
  echo "Pass it explicitly, e.g.:" >&2
  echo "  BUILD_ASSET_SHA=\$(git -C /path/to/local/checkout log -1 --format=%H -- scripts/build_asset.py) \\" >&2
  echo "    scripts/upload_ar4_asset_to_gcs.sh" >&2
  exit 1
fi
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo "unknown (no .git dir in this checkout)")"

GCS_BASE="gs://rl-manipulation-hks-models/ar4_mk5"
GCS_DEST="${GCS_BASE}/${BUILD_SCRIPT_SHA}"

echo "Uploading AR4 asset to ${GCS_DEST}/ ..."
echo "  build_asset.py git sha: ${BUILD_SCRIPT_SHA}"
echo "  repo HEAD at upload time: ${HEAD_SHA}"

PROVENANCE_FILE="$(mktemp)"
cat > "$PROVENANCE_FILE" <<EOF
build_asset_py_git_sha=${BUILD_SCRIPT_SHA}
repo_head_sha_at_build=${HEAD_SHA}
built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
built_on=$(hostname)
EOF

gsutil -m cp -r assets/ar4_mk5 "${GCS_DEST}/"
gsutil -m cp -r assets/shapes "${GCS_DEST}/"
gsutil cp "$PROVENANCE_FILE" "${GCS_DEST}/PROVENANCE.txt"
rm -f "$PROVENANCE_FILE"

# Update the "latest" pointer LAST, only once the full upload above has
# succeeded -- see header comment.
printf '%s' "$BUILD_SCRIPT_SHA" | gsutil cp - "${GCS_BASE}/LATEST"

echo "Done. Cached at: ${GCS_DEST}/"
echo "LATEST pointer now: ${BUILD_SCRIPT_SHA}"

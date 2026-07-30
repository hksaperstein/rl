#!/usr/bin/env bash
# Run payload for the ar4-jaw-close-fix task (2026-07-30): setup, then a
# CHEAP physics-only ground-truth pass (no RTX render) testing pure friction
# first, falling back to the fixed_joint grasp-assist mechanism only if
# friction genuinely fails to hold -- exactly the order the dispatch brief
# requires -- then exactly ONE video capture of whichever mechanism actually
# held, then a GCS sync of every result log + video before this script exits
# (run_on_cloud_gpu.sh tears the instance down immediately after this script
# returns, so syncing here is the last chance).
#
# Requires: BUILD_ASSET_SHA env var (passed straight through to the setup
# script). Meant to be dispatched as the <command...> to
# `scripts/run_on_cloud_gpu.sh --cost-cap 5.00`.
set -u
cd "$HOME/rl" || exit 1

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required -- aborting." >&2
  exit 1
fi

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-jaw-close-fix/$(date -u +%Y%m%d-%H%M%S)/"

run_pick() {
  # $1 = mechanism, $2 = extra args (e.g. --no_video)
  local mech="$1"; shift
  local extra="$*"
  echo "=== [pick run] mechanism=${mech} extra='${extra}' ($(date -u +%H:%M:%S)) ==="
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
    "$IMAGE" -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --mechanism ${mech} ${extra}"
  local ec=$?
  echo "=== [pick run] mechanism=${mech} extra='${extra}' exited ${ec} ($(date -u +%H:%M:%S)) ==="
  sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
  return $ec
}

held() {
  # $1 = result file path -- true (0) if it contains a genuine confirmed pick
  grep -q "VERDICT: PICK CONFIRMED" "$1" 2>/dev/null
}

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }

# --- [1/4] setup (docker+toolkit, image pull, GCS asset download) ---------
step "[1/4] setup"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/_cloud_ar4_standalone_pick_setup.sh 2>&1 | tee "$HOME/jaw_close_setup.log"
if ! grep -q "SETUP_READY_MARKER" "$HOME/jaw_close_setup.log"; then
  echo "FATAL: setup did not reach READY marker -- aborting." >&2
  gsutil -q cp "$HOME/jaw_close_setup.log" "${GCS_DEST}logs/jaw_close_setup.log" 2>&1 || true
  exit 1
fi

# --- [2/4] PHYSICS-ONLY pass: pure friction FIRST (Problem 2a), per the
# dispatch brief's required order -- retry a genuine pure-friction grasp
# before falling back to any grasp-assist mechanism.
step "[2/4] physics-only: --mechanism friction --no_video"
run_pick friction --no_video
FRICTION_RESULT="$HOME/rl/logs/standalone_pick_result_friction.txt"
cat "$FRICTION_RESULT" 2>/dev/null || echo "WARNING: no friction result file found"

WINNING_MECH="friction"
if held "$FRICTION_RESULT"; then
  echo ">>> PURE FRICTION HELD. Using --mechanism friction for the video capture."
else
  echo ">>> Pure friction did NOT hold (or result file missing) -- falling back to --mechanism fixed_joint (grasp-assist at closed jaws)."
  step "[2b/4] physics-only: --mechanism fixed_joint --no_video"
  run_pick fixed_joint --no_video
  FIXEDJOINT_RESULT="$HOME/rl/logs/standalone_pick_result_fixed_joint.txt"
  cat "$FIXEDJOINT_RESULT" 2>/dev/null || echo "WARNING: no fixed_joint result file found"
  WINNING_MECH="fixed_joint"
  if ! held "$FIXEDJOINT_RESULT"; then
    echo "WARNING: fixed_joint physics-only run did NOT report PICK CONFIRMED either -- proceeding to video capture anyway for visual diagnosis, but flagging this clearly."
  fi
fi

# --- [3/4] exactly ONE video capture, of whichever mechanism actually won --
step "[3/4] video capture: --mechanism ${WINNING_MECH}"
run_pick "$WINNING_MECH" ""
WINNING_RESULT="$HOME/rl/logs/standalone_pick_result_${WINNING_MECH}.txt"
cat "$WINNING_RESULT" 2>/dev/null || echo "WARNING: no result file found for winning mechanism video run"

# --- [4/4] GCS sync of every result log + video (last step before teardown)
step "[4/4] GCS sync -> ${GCS_DEST}"
gsutil -q cp "$HOME/jaw_close_setup.log" "${GCS_DEST}logs/jaw_close_setup.log" 2>&1 || echo "WARNING: GCS sync of setup log failed (non-fatal)"
for f in "$HOME/rl/logs"/standalone_pick_result_*.txt "$HOME/rl/logs/standalone_pick_breadcrumb.txt"; do
  [ -f "$f" ] && gsutil -q cp "$f" "${GCS_DEST}logs/$(basename "$f")" 2>&1 || true
done
if [ -d "$HOME/rl/logs/videos" ]; then
  gsutil -m -q cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal)"
fi
echo "WINNING_MECHANISM=${WINNING_MECH}"
echo "GCS_DEST=${GCS_DEST}"
echo "JAW_CLOSE_RUN_DONE"

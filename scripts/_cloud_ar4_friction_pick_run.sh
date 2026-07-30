#!/usr/bin/env bash
# Run payload for the ar4-pure-friction-pick task (2026-07-30). PURE-FRICTION
# mandate: the grasp MUST be a genuine contact-friction hold -- NO joint/weld
# of any kind, not even a fallback. The standalone script now handles grasp
# HEIGHT itself via a CLOSED-LOOP descent (deepen until the measured fingertip
# reaches the cube center Z -- robust to arm gravity droop), and applies a real
# capped-force squeeze (--squeeze_mode position, --squeeze_force), so this
# wrapper is simple: setup, ONE physics-only ground-truth run (contact-force +
# ground-truth lift), then ONE video run, then GCS sync.
#
# Requires: BUILD_ASSET_SHA env var. Dispatch as the <command...> to
# `scripts/run_on_cloud_gpu.sh --on-demand --cost-cap 4.00`.
set -u
cd "$HOME/rl" || exit 1
if [ -z "${BUILD_ASSET_SHA:-}" ]; then echo "ERROR: BUILD_ASSET_SHA required" >&2; exit 1; fi

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-pure-friction-pick/$(date -u +%Y%m%d-%H%M%S)/"
FRICTION_RESULT="$HOME/rl/logs/standalone_pick_result_friction.txt"
step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }

run_pick() {
  echo "=== [pick] friction $* ($(date -u +%H:%M:%S)) ==="
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
    "$IMAGE" -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --mechanism friction $*"
  local ec=$?
  echo "=== [pick] friction $* exited ${ec} ($(date -u +%H:%M:%S)) ==="
  sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
  return $ec
}

# --- [1/4] setup ----------------------------------------------------------
step "[1/4] setup"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/_cloud_ar4_standalone_pick_setup.sh 2>&1 | tee "$HOME/friction_setup.log"
if ! grep -q "SETUP_READY_MARKER" "$HOME/friction_setup.log"; then
  echo "FATAL: setup did not reach READY marker -- aborting." >&2
  gsutil -q cp "$HOME/friction_setup.log" "${GCS_DEST}logs/friction_setup.log" 2>&1 || true
  exit 1
fi

# --- [2/4] physics-only ground-truth run (position squeeze, 5N) -----------
step "[2/4] physics-only: --squeeze_mode position --squeeze_force 40.0 --no_video"
run_pick --no_video --squeeze_mode position --squeeze_force 40.0
cp -f "$FRICTION_RESULT" "$HOME/rl/logs/standalone_pick_result_friction_physicsonly.txt" 2>/dev/null
echo "----- physics-only result tail -----"; tail -40 "$FRICTION_RESULT" 2>/dev/null || echo "no result file"

# --- [3/4] ONE video run, GATED on the jaws actually CLOSING --------------
# The discriminator vs a scoop/wedge artifact is jaw CLOSURE: a real face grasp
# closes the jaws onto the ~15mm cube (closed_sep well under the 28mm open).
# If the jaws did NOT close, skip the expensive RTX video (a scoop artifact is
# not the deliverable) and report the blocker instead.
CLOSED_SEP="$(grep '\[JAW SUMMARY\]' "$HOME/rl/logs/standalone_pick_result_friction_physicsonly.txt" 2>/dev/null | tail -1 | grep -oP 'closed_sep=\K[0-9.]+')"
echo ">>> physics-only closed_sep=${CLOSED_SEP:-N/A}mm (open=28mm; real grip << 28mm)"
DO_VIDEO=0
if [ -n "${CLOSED_SEP:-}" ] && [ "$(awk -v s="$CLOSED_SEP" 'BEGIN{print (s<20.0)?1:0}')" = "1" ]; then
  DO_VIDEO=1
fi
if [ "$DO_VIDEO" = "1" ]; then
  step "[3/4] jaws CLOSED (${CLOSED_SEP}mm) -- video capture: --squeeze_mode position --squeeze_force 40.0"
  run_pick --squeeze_mode position --squeeze_force 40.0
  echo "----- video result tail -----"; tail -20 "$FRICTION_RESULT" 2>/dev/null || echo "no result file"
else
  echo ">>> JAWS DID NOT CLOSE (closed_sep=${CLOSED_SEP:-N/A}mm) -- skipping video (scoop artifact, not a face grasp). Reporting blocker."
fi

# --- [4/4] GCS sync -------------------------------------------------------
step "[4/4] GCS sync -> ${GCS_DEST}"
gsutil -q cp "$HOME/friction_setup.log" "${GCS_DEST}logs/friction_setup.log" 2>&1 || true
for f in "$HOME/rl/logs"/standalone_pick_result_*.txt "$HOME/rl/logs/standalone_pick_breadcrumb.txt"; do
  [ -f "$f" ] && gsutil -q cp "$f" "${GCS_DEST}logs/$(basename "$f")" 2>&1 || true
done
[ -d "$HOME/rl/logs/videos" ] && gsutil -m -q cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: video sync failed (non-fatal)"
echo "GCS_DEST=${GCS_DEST}"
echo "FRICTION_PICK_RUN_DONE"

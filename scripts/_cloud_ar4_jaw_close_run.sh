#!/usr/bin/env bash
# Run payload for the ar4-jaw-close-fix task (2026-07-30): setup, a small
# grasp-depth sweep (physics-only, --mechanism friction --no_video) to find a
# GRASP_Q depth where the jaws' fingertips genuinely straddle the cube
# (fingertip-corrected cube-centering offset small, not just "cube_z rose" --
# a fixed_joint weld would report success even around an open gap, so the
# sweep judges on jaw/cube GEOMETRY numbers, not the pick verdict), then the
# mechanism decision (retry pure friction first, fall back to fixed_joint
# grasp-assist only if friction genuinely doesn't hold), then exactly ONE
# video capture, then a GCS sync of every result log + video before this
# script exits (run_on_cloud_gpu.sh tears the instance down immediately after
# this script returns, so syncing here is the last chance).
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
  # $1 = mechanism, remaining args = extra CLI flags (e.g. --no_video --grasp_depth_extra 0.4)
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

# Extract the fingertip-corrected cube-centering offset (mm) from the
# [JAW SUMMARY] line of a friction result file -- the honest geometry
# metric (jaws vs cube), NOT the overall pick verdict (a fixed_joint weld
# reports "PICK CONFIRMED" even around an open gap, exactly the defect this
# task exists to eliminate -- so the sweep must not judge on that).
jaw_offset_mm() {
  grep "\[JAW SUMMARY\]" "$1" 2>/dev/null | tail -1 | grep -oP 'cube_centering_offset=\K[0-9.]+'
}
jaw_closed_sep_mm() {
  grep "\[JAW SUMMARY\]" "$1" 2>/dev/null | tail -1 | grep -oP 'closed_sep=\K[0-9.]+'
}

held() {
  # $1 = result file path -- true (0) if it contains a genuine confirmed pick
  grep -q "VERDICT: PICK CONFIRMED" "$1" 2>/dev/null
}

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }

# --- [1/5] setup (docker+toolkit, image pull, GCS asset download) ---------
step "[1/5] setup"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/_cloud_ar4_standalone_pick_setup.sh 2>&1 | tee "$HOME/jaw_close_setup.log"
if ! grep -q "SETUP_READY_MARKER" "$HOME/jaw_close_setup.log"; then
  echo "FATAL: setup did not reach READY marker -- aborting." >&2
  gsutil -q cp "$HOME/jaw_close_setup.log" "${GCS_DEST}logs/jaw_close_setup.log" 2>&1 || true
  exit 1
fi

# --- [2/5] GRASP-DEPTH SWEEP (physics-only, --mechanism friction --no_video)
# The first (depth_extra=0.0) attempt found the fingertip-corrected jaw
# midpoint sitting ~10mm ABOVE the cube center (just above its top face) --
# jaws close on empty air / a glancing top-edge catch, not a real straddle.
# Sweep a couple of small extra-descent fractions (reusing the SAME already-
# validated PREGRASP->GRASP direction, not a fresh IK solve) and keep
# whichever gives the smallest fingertip-corrected cube-centering offset
# with a sane (not ~0mm/pass-through, not ~28mm/no-motion) closed separation.
FRICTION_RESULT="$HOME/rl/logs/standalone_pick_result_friction.txt"
BEST_DEPTH="0.0"
BEST_OFFSET="999999"
for d in 0.4 0.7; do
  step "[2/5] depth sweep: --grasp_depth_extra ${d}"
  run_pick friction --no_video --grasp_depth_extra "$d"
  cp -f "$FRICTION_RESULT" "$HOME/rl/logs/standalone_pick_result_friction_depth${d}.txt" 2>/dev/null
  off="$(jaw_offset_mm "$FRICTION_RESULT")"
  sep="$(jaw_closed_sep_mm "$FRICTION_RESULT")"
  echo ">>> depth_extra=${d}: cube_centering_offset=${off:-N/A}mm closed_sep=${sep:-N/A}mm"
  if [ -n "$off" ]; then
    is_better="$(awk -v a="$off" -v b="$BEST_OFFSET" 'BEGIN{print (a<b)?1:0}')"
    if [ "$is_better" = "1" ]; then
      BEST_OFFSET="$off"
      BEST_DEPTH="$d"
    fi
  fi
done
echo ">>> SWEEP WINNER: grasp_depth_extra=${BEST_DEPTH} (cube_centering_offset=${BEST_OFFSET}mm)"

# --- [3/5] mechanism decision AT the winning depth: pure friction FIRST
# (Problem 2a, dispatch brief's required order), falling back to fixed_joint
# grasp-assist ONLY if friction genuinely doesn't hold. Judge the fallback
# decision on the ACTUAL friction pick verdict (cube_z lift+hold), which is
# now meaningful because the geometry sweep above already confirmed real
# jaw/cube contact -- unlike the mechanism's own "PICK CONFIRMED" text,
# which is not a safe proxy for fixed_joint (a weld reports success
# regardless of contact), it IS a safe proxy for friction (no weld exists,
# so a reported lift+hold there can only come from real jaw-contact
# friction).
step "[3/5] confirm at winning depth: --mechanism friction --grasp_depth_extra ${BEST_DEPTH} --no_video"
run_pick friction --no_video --grasp_depth_extra "$BEST_DEPTH"
cp -f "$FRICTION_RESULT" "$HOME/rl/logs/standalone_pick_result_friction_winning.txt" 2>/dev/null
cat "$FRICTION_RESULT" 2>/dev/null || echo "WARNING: no friction result file found"

WINNING_MECH="friction"
if held "$FRICTION_RESULT"; then
  echo ">>> PURE FRICTION HELD at grasp_depth_extra=${BEST_DEPTH}. Using --mechanism friction for the video capture."
else
  echo ">>> Pure friction did NOT hold at the corrected depth -- falling back to --mechanism fixed_joint (grasp-assist), engaged only once jaws are confirmed closed on the cube by the depth sweep above."
  step "[3b/5] physics-only: --mechanism fixed_joint --grasp_depth_extra ${BEST_DEPTH} --no_video"
  run_pick fixed_joint --no_video --grasp_depth_extra "$BEST_DEPTH"
  FIXEDJOINT_RESULT="$HOME/rl/logs/standalone_pick_result_fixed_joint.txt"
  cat "$FIXEDJOINT_RESULT" 2>/dev/null || echo "WARNING: no fixed_joint result file found"
  WINNING_MECH="fixed_joint"
  fj_off="$(jaw_offset_mm "$FIXEDJOINT_RESULT")"
  fj_sep="$(jaw_closed_sep_mm "$FIXEDJOINT_RESULT")"
  echo ">>> fixed_joint at grasp_depth_extra=${BEST_DEPTH}: cube_centering_offset=${fj_off:-N/A}mm closed_sep=${fj_sep:-N/A}mm (this is the number that matters, not just the pick verdict)"
fi

# --- [4/5] exactly ONE video capture, of whichever mechanism actually won,
# AT the winning depth.
step "[4/5] video capture: --mechanism ${WINNING_MECH} --grasp_depth_extra ${BEST_DEPTH}"
run_pick "$WINNING_MECH" --grasp_depth_extra "$BEST_DEPTH"
WINNING_RESULT="$HOME/rl/logs/standalone_pick_result_${WINNING_MECH}.txt"
cat "$WINNING_RESULT" 2>/dev/null || echo "WARNING: no result file found for winning mechanism video run"

# --- [5/5] GCS sync of every result log + video (last step before teardown)
step "[5/5] GCS sync -> ${GCS_DEST}"
gsutil -q cp "$HOME/jaw_close_setup.log" "${GCS_DEST}logs/jaw_close_setup.log" 2>&1 || echo "WARNING: GCS sync of setup log failed (non-fatal)"
for f in "$HOME/rl/logs"/standalone_pick_result_*.txt "$HOME/rl/logs/standalone_pick_breadcrumb.txt"; do
  [ -f "$f" ] && gsutil -q cp "$f" "${GCS_DEST}logs/$(basename "$f")" 2>&1 || true
done
if [ -d "$HOME/rl/logs/videos" ]; then
  gsutil -m -q cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal)"
fi
echo "WINNING_MECHANISM=${WINNING_MECH}"
echo "WINNING_DEPTH=${BEST_DEPTH}"
echo "GCS_DEST=${GCS_DEST}"
echo "JAW_CLOSE_RUN_DONE"

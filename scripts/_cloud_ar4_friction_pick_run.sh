#!/usr/bin/env bash
# Run payload for the ar4-pure-friction-pick task (2026-07-30, continuation of
# the jaw-close-fix, PURE-FRICTION mandate). Per direct user directive the
# grasp MUST be a genuine contact-friction hold -- NO joint/weld of any kind,
# not even as a fallback. This script:
#   1. setup (docker+toolkit, image pull, GCS asset download)
#   2. PHYSICS-ONLY effort-mode friction depth sweep (cheap: --no_video does
#      NOT init the RTX render pipeline, so no heavy shader compile) around the
#      FK-derived grasp_depth_extra=0.225, to absorb arm tracking residual and
#      land the fingertips at the cube's vertical CENTER. Judged on the
#      fingertip-corrected cube-centering offset AND the friction pick verdict
#      (trustworthy here: with NO weld, a reported lift+hold can ONLY come from
#      real jaw-contact friction) AND measured jaw<->cube contact normal force.
#   3. if effort mode does not hold at the best-geometry depth, retry there in
#      position squeeze mode (force-limited drive -- same real squeeze force,
#      more robust). STILL pure friction, NO joint.
#   4. exactly ONE video capture at the winning (mode, depth).
#   5. GCS sync of every result log + video before teardown.
#
# Requires: BUILD_ASSET_SHA env var. Dispatch as the <command...> to
# `scripts/run_on_cloud_gpu.sh --cost-cap 4.00`.
set -u
cd "$HOME/rl" || exit 1

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required -- aborting." >&2
  exit 1
fi

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-pure-friction-pick/$(date -u +%Y%m%d-%H%M%S)/"
FRICTION_RESULT="$HOME/rl/logs/standalone_pick_result_friction.txt"

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }

run_pick() {
  # args: all CLI flags for the standalone script (mechanism is always friction)
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

jaw_offset_mm() { grep "\[JAW SUMMARY\]" "$1" 2>/dev/null | tail -1 | grep -oP 'cube_centering_offset=\K[0-9.]+'; }
jaw_closed_sep_mm() { grep "\[JAW SUMMARY\]" "$1" 2>/dev/null | tail -1 | grep -oP 'closed_sep=\K[0-9.]+'; }
held() { grep -q "VERDICT: PICK CONFIRMED" "$1" 2>/dev/null; }

# --- [1/5] setup ----------------------------------------------------------
step "[1/5] setup"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/_cloud_ar4_standalone_pick_setup.sh 2>&1 | tee "$HOME/friction_setup.log"
if ! grep -q "SETUP_READY_MARKER" "$HOME/friction_setup.log"; then
  echo "FATAL: setup did not reach READY marker -- aborting." >&2
  gsutil -q cp "$HOME/friction_setup.log" "${GCS_DEST}logs/friction_setup.log" 2>&1 || true
  exit 1
fi

# --- [2/5] PHYSICS-ONLY effort-mode depth sweep ---------------------------
BEST_DEPTH=""; BEST_OFFSET="999999"; HELD_DEPTH=""; HELD_OFFSET="999999"
for d in 0.18 0.26 0.34; do
  step "[2/5] effort depth sweep: --grasp_depth_extra ${d} --squeeze_mode effort --no_video"
  run_pick --no_video --squeeze_mode effort --grasp_depth_extra "$d"
  cp -f "$FRICTION_RESULT" "$HOME/rl/logs/standalone_pick_result_effort_d${d}.txt" 2>/dev/null
  off="$(jaw_offset_mm "$FRICTION_RESULT")"; sep="$(jaw_closed_sep_mm "$FRICTION_RESULT")"
  if held "$FRICTION_RESULT"; then hv="HELD"; else hv="not-held"; fi
  echo ">>> effort d=${d}: cube_centering_offset=${off:-N/A}mm closed_sep=${sep:-N/A}mm verdict=${hv}"
  if [ -n "${off:-}" ]; then
    if [ "$(awk -v a="$off" -v b="$BEST_OFFSET" 'BEGIN{print (a<b)?1:0}')" = "1" ]; then BEST_OFFSET="$off"; BEST_DEPTH="$d"; fi
    if [ "$hv" = "HELD" ] && [ "$(awk -v a="$off" -v b="$HELD_OFFSET" 'BEGIN{print (a<b)?1:0}')" = "1" ]; then HELD_OFFSET="$off"; HELD_DEPTH="$d"; fi
  fi
done
echo ">>> SWEEP: best-geometry depth=${BEST_DEPTH} (offset=${BEST_OFFSET}mm); best HELD depth=${HELD_DEPTH:-none} (offset=${HELD_OFFSET}mm)"

# --- [3/5] pick the winning config (pure friction only) -------------------
WIN_MODE="effort"; WIN_DEPTH=""
if [ -n "$HELD_DEPTH" ]; then
  WIN_DEPTH="$HELD_DEPTH"
  echo ">>> PURE FRICTION (effort) HELD at depth=${WIN_DEPTH}."
else
  WIN_DEPTH="${BEST_DEPTH:-0.26}"
  step "[3/5] effort didn't hold -- retry PURE FRICTION in position squeeze mode at depth=${WIN_DEPTH}"
  run_pick --no_video --squeeze_mode position --grasp_depth_extra "$WIN_DEPTH"
  cp -f "$FRICTION_RESULT" "$HOME/rl/logs/standalone_pick_result_position_d${WIN_DEPTH}.txt" 2>/dev/null
  off="$(jaw_offset_mm "$FRICTION_RESULT")"
  if held "$FRICTION_RESULT"; then
    WIN_MODE="position"; echo ">>> PURE FRICTION (position mode) HELD at depth=${WIN_DEPTH} (offset=${off:-N/A}mm)."
  else
    echo ">>> Pure friction did NOT hold in either squeeze mode at depth=${WIN_DEPTH}. Capturing video of the best friction attempt anyway (jaws squeezing the cube); reporting honestly. NO joint fallback (forbidden)."
    # choose the mode whose result we still have as the best-geometry attempt
    WIN_MODE="effort"
  fi
fi

# --- [4/5] ONE video capture at the winning (mode, depth) -----------------
step "[4/5] video capture: --squeeze_mode ${WIN_MODE} --grasp_depth_extra ${WIN_DEPTH}"
run_pick --squeeze_mode "$WIN_MODE" --grasp_depth_extra "$WIN_DEPTH"
cat "$FRICTION_RESULT" 2>/dev/null || echo "WARNING: no friction result file for the video run"

# --- [5/5] GCS sync -------------------------------------------------------
step "[5/5] GCS sync -> ${GCS_DEST}"
gsutil -q cp "$HOME/friction_setup.log" "${GCS_DEST}logs/friction_setup.log" 2>&1 || true
for f in "$HOME/rl/logs"/standalone_pick_result_*.txt "$HOME/rl/logs/standalone_pick_breadcrumb.txt"; do
  [ -f "$f" ] && gsutil -q cp "$f" "${GCS_DEST}logs/$(basename "$f")" 2>&1 || true
done
[ -d "$HOME/rl/logs/videos" ] && gsutil -m -q cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: video sync failed (non-fatal)"
echo "WINNING_MODE=${WIN_MODE}"
echo "WINNING_DEPTH=${WIN_DEPTH}"
echo "GCS_DEST=${GCS_DEST}"
echo "FRICTION_PICK_RUN_DONE"

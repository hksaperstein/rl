#!/usr/bin/env bash
# LEAN retry payload (2026-07-31, ar4-tall-prism-pick) -- assumes a WARM instance
# from _cloud_ar4_tall_prism_pick.sh (docker + isaac-lab image pulled + AR4 asset
# already under ~/rl/assets). Only re-asserts usd_path.txt and runs the phases.
#   PHASE A: --jaw_close_test -- free-space both-jaws-close control (drive bug vs
#            centering failure). PHASE B: physics-only grasp sweep (WIDE object +
#            firm/long jaw close). PHASE C: WITH video, ALWAYS.
set -u
step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing"; fi; }

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: AR4 USD asset missing (warm-instance assumption broken)." >&2; exit 1; fi
mkdir -p "$HOME/rl/logs"

WIDE="--prism_width 0.024 --prism_depth 0.018 --prism_mass 0.03"
FIRM="--jaw_max_force 120 --grip_kd 60 --close_steps 60"
SQ="--mechanism friction --squeeze_mode position --squeeze_force 40"

run_in_container() { local logf="$1"; shift
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
    -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py $*" 2>&1 | tee "$logf"
  return "${PIPESTATUS[0]}"; }

# --- PHASE A: free-space jaw-close control ---------------------------------
step "PHASE A: free-space jaw-close test"
run_in_container "$HOME/rl/logs/retry_jawtest.log" \
  --prism --jaw_close_test --no_video $SQ $WIDE $FIRM
check $? "phase A jaw close test"
echo "===== JAWTEST KEY LINES ====="
grep -E "\[JAWTEST\]" "$HOME/rl/logs/retry_jawtest.log" | tail -20 || true
echo "============================="

# --- PHASE B: physics-only grasp sweep (wide object, firm close) -----------
BEST_H=""
for H in 0.050 0.052; do
  step "PHASE B sweep: prism_height=${H} (physics-only, wide+firm)"
  run_in_container "$HOME/rl/logs/retry_sweep_${H}.log" \
    --prism --prism_height "$H" --no_video $SQ $WIDE $FIRM
  check $? "phase B sweep ${H}"
  grep -E "VERDICT: PICK|cube_z gain|\[JAW SUMMARY\]|CONTACT P3_after_close|CONTACT P4_after_LIFT|real lift|held through" \
    "$HOME/rl/logs/retry_sweep_${H}.log" | tail -18 || true
  if grep -q "VERDICT: PICK CONFIRMED" "$HOME/rl/logs/retry_sweep_${H}.log"; then
    BEST_H="$H"; echo "[SWEEP] ${H} CONFIRMED -> best=${BEST_H}"; fi
done
VIDEO_H="${BEST_H:-0.052}"
echo "===== RETRY SWEEP: BEST_H='${BEST_H}' -> VIDEO_H=${VIDEO_H} ====="

# --- PHASE C: WITH video (ALWAYS) ------------------------------------------
step "PHASE C: pure-friction tall-prism pick WITH video (height=${VIDEO_H})"
run_in_container "$HOME/rl/logs/retry_pick_video.log" \
  --prism --prism_height "$VIDEO_H" $SQ $WIDE $FIRM
RUN_EXIT=$?
check "$RUN_EXIT" "phase C video pick"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
echo "===== PICK VERDICT ====="
grep -E "VERDICT: PICK|cube_z gain|\[JAW SUMMARY\]|CONTACT P4_after_LIFT|CONTACT P3b|real lift|held through|cube_z P" \
  "$HOME/rl/logs/retry_pick_video.log" | tail -30 || true
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-tall-prism/retry-$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARN: video sync failed"
gsutil -m cp "$HOME/rl/logs"/retry_*.log "$HOME/rl/logs"/standalone_pick_result_friction.txt "${GCS_DEST}" 2>&1 || echo "WARN: log sync failed"
echo "RETRY_PIPELINE_DONE best_h='${BEST_H}' video_h=${VIDEO_H} video_exit=${RUN_EXIT}"

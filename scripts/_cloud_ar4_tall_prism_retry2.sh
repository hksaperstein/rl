#!/usr/bin/env bash
# LEAN retry-2 payload (2026-07-31, ar4-tall-prism-pick) -- WARM instance assumed.
# --no_servo direct grasp at GRASP_Q (no Jacobian-probe bumping) + firm close +
# wide-ish object. PHASE A: physics-only sweep (heights). PHASE B: video, ALWAYS.
set -u
step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing"; fi; }
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
[ -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ] || { echo "FATAL: asset missing"; exit 1; }
mkdir -p "$HOME/rl/logs"

OBJ="--prism_width 0.022 --prism_depth 0.018 --prism_mass 0.05"
FIRM="--jaw_max_force 120 --grip_kd 60 --close_steps 60"
SQ="--mechanism friction --squeeze_mode position --squeeze_force 40"
NS="--no_servo"

run_in_container() { local logf="$1"; shift
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
    -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py $*" 2>&1 | tee "$logf"
  return "${PIPESTATUS[0]}"; }

BEST_H=""
for H in 0.050 0.053; do
  step "PHASE A sweep (no_servo): prism_height=${H}"
  run_in_container "$HOME/rl/logs/r2_sweep_${H}.log" \
    --prism --prism_height "$H" --no_video $NS $SQ $OBJ $FIRM
  check $? "phase A ${H}"
  grep -E "VERDICT: PICK|cube_z gain|\[JAW SUMMARY\]|CONTACT P3_after_close|CONTACT P4_after_LIFT|real lift|held through|no_servo: grasping|SUMMARY mechanism" \
    "$HOME/rl/logs/r2_sweep_${H}.log" | tail -18 || true
  if grep -q "VERDICT: PICK CONFIRMED" "$HOME/rl/logs/r2_sweep_${H}.log"; then
    BEST_H="$H"; echo "[SWEEP] ${H} CONFIRMED -> best=${BEST_H}"; fi
done
VIDEO_H="${BEST_H:-0.053}"
echo "===== RETRY2 SWEEP: BEST_H='${BEST_H}' -> VIDEO_H=${VIDEO_H} ====="

step "PHASE B: pure-friction tall-prism pick WITH video (no_servo, height=${VIDEO_H})"
run_in_container "$HOME/rl/logs/r2_pick_video.log" \
  --prism --prism_height "$VIDEO_H" $NS $SQ $OBJ $FIRM
RUN_EXIT=$?
check "$RUN_EXIT" "phase B video"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
echo "===== PICK VERDICT ====="
grep -E "VERDICT: PICK|cube_z gain|\[JAW SUMMARY\]|CONTACT P4_after_LIFT|CONTACT P3b|real lift|held through|cube_z P" \
  "$HOME/rl/logs/r2_pick_video.log" | tail -30 || true
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-tall-prism/retry2-$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARN: video sync failed"
gsutil -m cp "$HOME/rl/logs"/r2_*.log "$HOME/rl/logs"/standalone_pick_result_friction.txt "${GCS_DEST}" 2>&1 || echo "WARN: log sync failed"
echo "RETRY2_PIPELINE_DONE best_h='${BEST_H}' video_h=${VIDEO_H} video_exit=${RUN_EXIT}"

#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-31, ar4-tall-prism-pick task): land a
# GENUINE pure-friction top-down grasp+lift of a PROPERLY-SIZED object (a tall
# narrow rectangular prism resting on the ground), on the SAME fresh instance:
#   PHASE 0 (AABB, no video): --prism --dump_gripper_aabb --empty_scene -- the
#     TOP-DOWN gripper collision-mesh AABB at the reachable GRASP_Q, to VERIFY
#     collision clearance BEFORE the grasp (measure how far the gripper body /
#     fingertips hang below the pad plane in the actual near-vertical orientation).
#   PHASE 1 (physics-only sweep, no video): --prism --no_video over several prism
#     HEIGHTS, to find the height whose top clears the gripper's central body while
#     giving a solid pad contact band -> jaws straddle the two faces, real contact
#     normal force, ground-truth lift. Cheap (skips the ~10min first-RTX init).
#   PHASE 2 (WITH video, ALWAYS): the tallest CONFIRMED height from the sweep (or a
#     default if none confirmed) rendered with RTX -- captures the FULL attempt
#     (approach->close->lift->retreat) UNCONDITIONALLY per the standing user
#     directive (2026-07-31: video every attempt, no gating on closure).
#
# Root cause this addresses (kb ar4-vs-franka-root-cause-comparison.md, both
# 2026-07-31 UPDATEs): a pure-friction grasp of a 15mm cube is geometrically
# infeasible -- the gripper envelope hangs 15-23mm below the pads and SURROUNDS
# the cube on a work surface (top-down: palm jams on cube top; side-on: body jams
# on support). The arm's drives/tracking/servo/squeeze/pad-geometry are all PROVEN
# flawless. Fix (Principal decision): grasp an appropriately-sized object -- a tall
# narrow prism gripped near its TOP, on the ground, so the underhang clears.
#
# Steps 1-3 (docker/toolkit + GL pin + image pull + GCS asset download) copied
# VERBATIM from scripts/_cloud_ar4_side_grasp_run.sh -- do NOT "improve" them.
#
# Env vars: BUILD_ASSET_SHA (required, same as the container pipeline).
set -u

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() {
  if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing anyway"; fi
}

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required -- aborting." >&2
  exit 1
fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1

# --- [1/4] Docker + NVIDIA Container Toolkit (verbatim) --------------------
step "[1/4] Docker + NVIDIA Container Toolkit"
if command -v docker >/dev/null 2>&1; then
  echo "docker already present, skipping install"
else
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  check $? "docker.io apt install"
fi
sudo systemctl enable --now docker
check $? "docker daemon enabled/started"

if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
  echo "nvidia-container-toolkit already present, skipping install"
else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
  check $? "nvidia-container-toolkit apt install"
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
check $? "docker + nvidia-container-toolkit ready"

if ! find /usr/lib/x86_64-linux-gnu -iname 'libGLX_nvidia*' 2>/dev/null | grep -q .; then
  DRIVER_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  DRIVER_MAJOR="$(echo "$DRIVER_FULL" | cut -d. -f1)"
  echo "Running driver: ${DRIVER_FULL} (major ${DRIVER_MAJOR})"
  APT_VER="$(apt-cache madison "libnvidia-gl-${DRIVER_MAJOR}-server" 2>/dev/null | awk -F'|' -v v="$DRIVER_FULL" '{gsub(/^[ \t]+|[ \t]+$/,"",$2); if (index($2, v) == 1) {print $2; exit}}')"
  if [ -n "$APT_VER" ]; then
    echo "Pinning libnvidia-gl-${DRIVER_MAJOR}-server to exact running-driver version: ${APT_VER}"
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server=${APT_VER}"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server=${APT_VER} pinned install"
  else
    echo "WARNING: no exact-match apt candidate for driver ${DRIVER_FULL} - unpinned install." >&2
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server install (unpinned fallback)"
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi broke after libnvidia-gl install (driver/library mismatch). Needs: sudo reboot then re-run." >&2
    exit 1
  fi
fi

# --- [2/4] pull the official Isaac Lab container ---------------------------
step "[2/4] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"
PULL_EXIT=$?
check "$PULL_EXIT" "docker pull ${IMAGE}"
if [ "$PULL_EXIT" -ne 0 ]; then
  echo "FATAL: cannot proceed without the container image." >&2
  exit 1
fi

# --- [3/4] download AR4 asset from GCS cache ------------------------------
step "[3/4] download AR4 asset from GCS cache"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: AR4 USD asset missing after GCS download -- aborting." >&2
  exit 1
fi

mkdir -p "$HOME/rl/logs"

SQUEEZE_ARGS="--mechanism friction --squeeze_mode position --squeeze_force 8.0"
PRISM_WD="--prism_width 0.018 --prism_depth 0.018 --prism_mass 0.025"

run_in_container() {  # $1 = log file, rest = script args
  local logf="$1"; shift
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
    "$IMAGE" \
    -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py $*" \
    2>&1 | tee "$logf"
  return "${PIPESTATUS[0]}"
}

# --- PHASE 0: top-down gripper AABB (verify clearance BEFORE the grasp) ------
step "PHASE 0: top-down gripper collision-mesh AABB @ GRASP_Q (empty_scene)"
run_in_container "$HOME/rl/logs/prism_aabb.log" \
  --prism --dump_gripper_aabb --empty_scene --no_video $SQUEEZE_ARGS $PRISM_WD
check $? "phase 0 AABB dump"
echo; echo "===== AABB KEY LINES ====="
grep -E "\[AABB\]|pad_mid" "$HOME/rl/logs/prism_aabb.log" 2>/dev/null | tail -20 || true
echo "=========================="

# --- PHASE 1: physics-only height sweep (find the graspable height) ---------
BEST_H=""
for H in 0.050 0.054 0.058 0.062; do
  step "PHASE 1 sweep: prism_height=${H} (physics-only)"
  run_in_container "$HOME/rl/logs/prism_sweep_${H}.log" \
    --prism --prism_height "$H" --no_video $SQUEEZE_ARGS $PRISM_WD
  check $? "phase 1 sweep height ${H}"
  echo "----- height ${H} key lines -----"
  grep -E "VERDICT|\[CONTACT P4|\[CONTACT P3b|cube_z gain|real lift|held through|SUMMARY" \
    "$HOME/rl/logs/prism_sweep_${H}.log" 2>/dev/null | tail -20 || true
  if grep -q "VERDICT: PICK CONFIRMED" "$HOME/rl/logs/prism_sweep_${H}.log" 2>/dev/null; then
    BEST_H="$H"   # keep the TALLEST confirmed (loop ascends) -> most convincing grip
    echo "[SWEEP] height ${H} CONFIRMED -> current best=${BEST_H}"
  fi
done

VIDEO_H="${BEST_H:-0.054}"
echo
echo "===== SWEEP RESULT: BEST_H='${BEST_H}'  -> VIDEO_H=${VIDEO_H} ====="

# --- PHASE 2: WITH video (ALWAYS -- no closure gate) ------------------------
step "PHASE 2: pure-friction tall-prism pick WITH video (prism_height=${VIDEO_H})"
run_in_container "$HOME/rl/logs/prism_pick_video.log" \
  --prism --prism_height "$VIDEO_H" $SQUEEZE_ARGS $PRISM_WD
RUN_EXIT=$?
check "$RUN_EXIT" "phase 2 video pick"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
echo; echo "===== PICK VERDICT + KEY LINES ====="
grep -E "VERDICT|\[CONTACT P4|\[CONTACT P3b|\[JAW SUMMARY\]|cube_z gain|real lift|held through|SUMMARY|cube_z P" \
  "$HOME/rl/logs/prism_pick_video.log" 2>/dev/null | tail -40 || true
echo "===================================="
echo "video files:"
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

# --- sync artifacts to GCS (ALWAYS) ----------------------------------------
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-tall-prism/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp \
  "$HOME/rl/logs/prism_aabb.log" \
  "$HOME/rl/logs"/prism_sweep_*.log \
  "$HOME/rl/logs/prism_pick_video.log" \
  "$HOME/rl/logs"/standalone_pick_result_friction.txt \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"
echo "GCS_DEST=${GCS_DEST}"
echo "PRISM_PIPELINE_DONE best_h='${BEST_H}' video_h=${VIDEO_H} video_exit=${RUN_EXIT}"

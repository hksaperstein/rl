#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-31, ar4-side-grasp task): prepare a
# fresh GCP instance (docker/toolkit + GL-lib pin + isaac-lab container pull +
# GCS-cached AR4 asset), then run the PURE-FRICTION SIDE / HORIZONTAL grasp of
# the 15mm cube in two phases on the SAME instance:
#   PHASE 1 (fast, no video): --side_grasp --no_video -- physics-only. Confirms
#     numerically whether the horizontal pads REACH the cube's two vertical
#     +Y/-Y faces at mid-height (no top-collision block), the measured contact
#     normal force on both jaws, and the ground-truth cube lift. Cheap: skips
#     the ~10min first-RTX-render init.
#   PHASE 2 (WITH video, ALWAYS): --side_grasp with RTX render. Captures the
#     closeup + elbow video of the FULL attempt (approach -> close -> lift ->
#     retreat) UNCONDITIONALLY -- success or not -- per the standing user
#     directive (2026-07-31): every attempt's video is wanted, no gating on
#     closure. Then sync video + both result logs to GCS.
#
# Root cause this addresses (kb ar4-vs-franka-root-cause-comparison.md,
# 2026-07-31 UPDATE): the centered TOP-DOWN approach is collision-blocked (the
# gripper palm jams on the cube TOP FACE ~9mm early; 48.5mm palm-to-fingertip
# depth, shallow fingers cannot straddle a 15mm cube from above). The arm's
# drives/tracking are proven flawless in free space. The side grasp orients the
# gripper horizontally so the palm sits on the -X side of the cube, never above.
#
# Steps 1-3 (docker/toolkit + GL pin + image pull + GCS asset download) are
# copied VERBATIM from scripts/_cloud_ar4_droop_diag_and_pick.sh -- do NOT
# "improve" them, they encode hard-won live fixes.
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

# --- [4a/4] PHASE 1: side-grasp physics-only sanity (fast, no video) --------
step "[4a/4] PHASE 1 side-grasp physics-only sanity (--side_grasp --no_video)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --side_grasp --no_video --mechanism friction --squeeze_mode position --squeeze_force 5.0" \
  2>&1 | tee "$HOME/rl/logs/side_grasp_sanity.log"
SANITY_EXIT="${PIPESTATUS[0]}"
check "$SANITY_EXIT" "side-grasp physics-only sanity run"
cp "$HOME/rl/logs/standalone_pick_result_friction.txt" \
   "$HOME/rl/logs/standalone_pick_result_SIDE_SANITY.txt" 2>/dev/null || true

echo
echo "===== SANITY KEY LINES ====="
grep -E "VERDICT|\[SERVO it|\[CONTACT|\[JAW SUMMARY\]|cube_z gain|real lift|\[REPORT" "$HOME/rl/logs/side_grasp_sanity.log" 2>/dev/null | tail -40 || true
echo "============================"

# --- [4b/4] PHASE 2: side-grasp WITH video (ALWAYS -- no closure gate) -------
# RTX render ON. Renders + captures the FULL attempt regardless of whether the
# grasp succeeds (standing user directive 2026-07-31: video every attempt).
step "[4b/4] PHASE 2 side-grasp pure-friction pick WITH video (--side_grasp)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --side_grasp --mechanism friction --squeeze_mode position --squeeze_force 5.0" \
  2>&1 | tee "$HOME/rl/logs/side_grasp_pick_run.log"
RUN_EXIT="${PIPESTATUS[0]}"
check "$RUN_EXIT" "side-grasp pick run (video)"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

echo
echo "===== PICK VERDICT + KEY LINES ====="
grep -E "VERDICT|\[CONTACT|\[JAW SUMMARY\]|cube_z gain|real lift|held through" "$HOME/rl/logs/side_grasp_pick_run.log" 2>/dev/null | tail -40 || true
echo "===================================="
echo "video files:"
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

# --- sync artifacts to GCS (ALWAYS -- video wanted even on failure) ---------
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-side-grasp/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp \
  "$HOME/rl/logs/side_grasp_sanity.log" \
  "$HOME/rl/logs/side_grasp_pick_run.log" \
  "$HOME/rl/logs/standalone_pick_result_SIDE_SANITY.txt" \
  "$HOME/rl/logs"/standalone_pick_result_friction.txt \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE (sanity exit=${SANITY_EXIT} pick exit=${RUN_EXIT})."

#!/usr/bin/env bash
# One-shot cloud dispatch payload for scripts/_diag_ar4_joint2_limit_raw_usd_only.py
# -- the lightweight (isaacsim.SimulationApp, no ManagerBasedRLEnv/robot
# articulation construction) variant of _cloud_ar4_joint2_limit_root_cause.sh,
# added 2026-07-28 after that heavier script hit the same documented
# "CPU pinned, GPU 0%, log stale" cold-start stall three consecutive times,
# every time before printing its own first diagnostic line. This variant
# uses the SAME lightweight bootstrap scripts/_inspect_jaw_axis_math.py
# already used successfully (no reported hang) for the gripper joints.
#
# Env vars this script reads:
#   BUILD_ASSET_SHA   Required - see _cloud_ar4_container_pipeline.sh's own
#                      header comment for why (no .git dir on a
#                      git-archive-shipped checkout).
set -u

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() {
  if [ "$1" -eq 0 ]; then
    echo "[OK] $2"
  else
    echo "[FAIL exit=$1] $2 - continuing anyway"
  fi
}

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required (see header comment) -- aborting." >&2
  exit 1
fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1

step "[1/4] Docker + NVIDIA Container Toolkit"
T1_START=$(date +%s)
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
T1_END=$(date +%s)
echo "TIMING docker_toolkit_install_sec=$((T1_END - T1_START))"

step "[2/4] docker pull ${IMAGE}"
T2_START=$(date +%s)
sudo docker pull "$IMAGE"
PULL_EXIT=$?
check "$PULL_EXIT" "docker pull ${IMAGE}"
T2_END=$(date +%s)
echo "TIMING container_pull_sec=$((T2_END - T2_START))"
if [ "$PULL_EXIT" -ne 0 ]; then
  echo "FATAL: cannot proceed without the container image. Aborting remaining steps." >&2
  exit 1
fi

step "[3/4] download AR4 asset from GCS cache"
T3_START=$(date +%s)
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
T3_END=$(date +%s)
echo "TIMING gcs_download_sec=$((T3_END - T3_START))"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: no assets/ar4_mk5/ar4_mk5.usd after GCS download -- aborting." >&2
  exit 1
fi
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"

step "[4/4] _diag_ar4_joint2_limit_raw_usd_only.py"
T4_START=$(date +%s)
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/_diag_ar4_joint2_limit_raw_usd_only.py" \
  2>&1 | tee "$HOME/joint2_limit_raw_usd_only.log"
check "${PIPESTATUS[0]}" "_diag_ar4_joint2_limit_raw_usd_only.py (containerized)"
T4_END=$(date +%s)
echo "TIMING run_sec=$((T4_END - T4_START))"

step "RESULT FILE (belt-and-suspenders capture, independent of stdout/tee)"
if [ -f "$HOME/rl/joint2_limit_raw_usd_only_result.txt" ]; then
  cat "$HOME/rl/joint2_limit_raw_usd_only_result.txt"
else
  echo "WARNING: result file not found -- script may not have reached its own log() calls at all."
fi

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-joint2-limit-raw-usd-only/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -q cp "$HOME/joint2_limit_raw_usd_only.log" "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"
gsutil -q cp "$HOME/rl/joint2_limit_raw_usd_only_result.txt" "${GCS_DEST}" 2>&1 || echo "WARNING: result file GCS sync failed (non-fatal or file missing)"

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))"
echo "gcs_download_sec=$((T3_END - T3_START))"
echo "run_sec=$((T4_END - T4_START))"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

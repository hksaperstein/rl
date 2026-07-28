#!/usr/bin/env bash
# One-shot cloud dispatch payload for scripts/_diag_ar4_joint2_limit_root_cause.py
# (2026-07-28, ar4-corrected-graspable-workspace-and-grasp task). Container +
# GCS-cached-asset path, same recipe as scripts/_cloud_ar4_joint_limits_readback.sh
# -- only runs the tiny direct-USD + live-limits readback (no cube, no
# phases, no camera, a few seconds of actual sim time), since the only open
# question is: what are the AR4 arm's REAL baked-in joint position limits,
# and does the ~59deg joint_2 wall found by the prior task match the raw
# USD's own authored physics:upperLimit or something else.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance and runs a command
# there with ~/rl as the working directory.
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

step "[4/4] _diag_ar4_joint2_limit_root_cause.py"
T4_START=$(date +%s)
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/_diag_ar4_joint2_limit_root_cause.py --headless" \
  2>&1 | tee "$HOME/joint2_limit_root_cause.log"
check "${PIPESTATUS[0]}" "_diag_ar4_joint2_limit_root_cause.py (containerized)"
T4_END=$(date +%s)
echo "TIMING run_sec=$((T4_END - T4_START))"

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-joint2-limit-root-cause/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -q cp "$HOME/joint2_limit_root_cause.log" "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))"
echo "gcs_download_sec=$((T3_END - T3_START))"
echo "run_sec=$((T4_END - T4_START))"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

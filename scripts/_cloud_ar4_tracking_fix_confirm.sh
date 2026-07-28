#!/usr/bin/env bash
# One-shot cloud dispatch payload for the ar4-joint-tracking-closed-loop-fix
# task (2026-07-27). Uses the container + GCS-cached-asset path
# (docs/cloud/dispatch-checklist.md's preferred AR4 cloud recipe) - steps
# below copied verbatim from scripts/_cloud_ar4_joint_tracking_diagnostic.sh
# (itself copied from scripts/_cloud_ar4_container_pipeline.sh's steady-state
# steps).
#
# Single task, one instance: scripts/ar4_tracking_fix_confirm.py -
# stiffness sweep (4000/20000/80000, +an effort_limit_sim=200 variant) and
# a closed-loop settle_to_joint_pose primitive test (both no-cube, P0
# GRASP_Q), a programmatic decision between the two, then a full
# grasp+lift re-attempt at P0/P1/P2 (cube present) using the winning
# method's PREGRASP-OPEN/GRASP-OPEN settle - no camera (numeric-only, fast,
# avoids camera-render risk on this exploratory run).
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance and runs a command
# there with ~/rl as the working directory.
#
# Env vars this script reads:
#   BUILD_ASSET_SHA   Required - see _cloud_ar4_container_pipeline.sh's own
#                      header comment for why (no .git dir on a
#                      git-archive-shipped checkout).
#
# Deliberately does NOT use `set -e` (same reasoning as this repo's other
# _cloud_ar4_*.sh dispatch payloads - a single incidental non-zero exit
# mid-pipeline must not silently discard the rest of the run's evidence).
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

# --- [1/4] Docker + NVIDIA Container Toolkit -------------------------------
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

echo "[SKIP] libnvidia-gl-<major>-server install intentionally skipped - not needed, this task is camera-free."
T1_END=$(date +%s)
echo "TIMING docker_toolkit_install_sec=$((T1_END - T1_START))"

step "[1b/4] GPU passthrough sanity check"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
check $? "GPU visible inside a container"

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

# --- [3/4] download the AR4 USD asset from the GCS cache -------------------
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

# --- [4/4] THE ACTUAL SCRIPT ------------------------------------------------
step "[4/4] ar4_tracking_fix_confirm.py (stiffness sweep + closed-loop primitive + full grasp+lift re-attempt, P0/P1/P2)"
T4_START=$(date +%s)
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/ar4_tracking_fix_confirm.py --headless" \
  2>&1 | tee "$HOME/tracking_fix_confirm.log"
check "${PIPESTATUS[0]}" "ar4_tracking_fix_confirm.py (containerized)"
T4_END=$(date +%s)
echo "TIMING run_sec=$((T4_END - T4_START))"

if grep -q "^Traceback (most recent call last):" "$HOME/tracking_fix_confirm.log"; then
  echo "WARNING: a Python Traceback was found in tracking_fix_confirm.log despite exit 0 above - inspect the log, this may NOT be a real pass." >&2
fi
if ! grep -q "^FINAL MULTI-POINT SUMMARY" "$HOME/tracking_fix_confirm.log"; then
  echo "WARNING: no 'FINAL MULTI-POINT SUMMARY' line found - the script likely did not reach its own final summary." >&2
fi

sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-tracking-fix-confirm/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
echo "=== GCS sync (log) ==="
gsutil -q cp "$HOME/tracking_fix_confirm.log" "$HOME/download_asset.log" "${GCS_DEST}" 2>&1 || \
  echo "WARNING: log GCS sync failed (non-fatal)"

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))"
echo "gcs_download_sec=$((T3_END - T3_START))"
echo "run_sec=$((T4_END - T4_START))"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

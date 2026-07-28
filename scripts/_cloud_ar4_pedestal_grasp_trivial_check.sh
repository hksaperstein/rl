#!/usr/bin/env bash
# One-shot cloud dispatch payload for the ar4-grasp-trivial-friction-check
# task (2026-07-28, direct user instruction: stop overcomplicating, check
# the trivial things - friction/height/aperture - by direct observation).
# Uses the container + GCS-cached-asset path
# (docs/cloud/dispatch-checklist.md's preferred AR4 cloud recipe), same
# steady-state 4-step structure as scripts/_cloud_ar4_pedestal_grasp_confirm.sh
# (the AR4 USD asset itself is UNCHANGED by this task - the friction fix is
# a plain isaaclab RigidBodyMaterialCfg on the cube's spawn cfg, not baked
# into the robot USD - so the existing GCS asset cache is still valid,
# no rebuild/reupload needed), PLUS the libnvidia-gl driver-library install
# + GPU/Vulkan sanity checks from scripts/_cloud_ar4_container_pipeline.sh's
# own step [1/7] - this task's own script (scripts/
# ar4_pedestal_grasp_trivial_check.py) renders TWO cameras throughout, unlike
# every prior _cloud_ar4_*_confirm.sh dispatch (all explicitly "camera-free"
# and skipped this step). Camera rendering through this exact container path
# was already independently proven live by _cloud_ar4_container_pipeline.sh's
# own step [7/7] smoke test (_record_jaw_fix_open_close_cycle.py, a
# camera-recording script) - this is not a first attempt at that combination.
#
# Single task, one instance: scripts/ar4_pedestal_grasp_trivial_check.py -
# ONE grasp point (Q0_bearing95 - all 3 pedestal points failed identically
# in the prior ar4_pedestal_grasp_confirm.py runs, so one repro suffices),
# with close-up + elbow-inclusive video/frame capture and explicit
# height/aperture/friction diagnostics.
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

# --- [1/4] Docker + NVIDIA Container Toolkit + GL/Vulkan libs --------------
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

# This task's own script renders 2 cameras throughout - copied verbatim from
# scripts/_cloud_ar4_container_pipeline.sh's own step [1/7] (pinned to the
# exact running driver version to avoid the documented "apt picks a newer
# point release than the loaded kernel module" break).
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
    echo "WARNING: could not find an apt candidate matching exact driver ${DRIVER_FULL} - falling back to unpinned install." >&2
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server install (unpinned fallback)"
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi broke after installing libnvidia-gl-${DRIVER_MAJOR}-server (driver/library version mismatch)." >&2
    echo "This needs a reboot to load the matching kernel module -- run: sudo reboot" >&2
    echo "then wait for SSH and re-run this script (idempotent, will skip already-done steps)." >&2
    exit 1
  fi
fi
T1_END=$(date +%s)
echo "TIMING docker_toolkit_install_sec=$((T1_END - T1_START))"

step "[1b/4] GPU passthrough sanity check"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
check $? "GPU visible inside a container"
step "[1c/4] Vulkan/GL library injection sanity check"
sudo docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all nvidia/cuda:12.4.0-base-ubuntu22.04 bash -c 'find /usr/lib/x86_64-linux-gnu -iname "libGLX_nvidia*" 2>/dev/null | grep -q . && echo "[container-check] libGLX_nvidia present" || echo "[container-check] MISSING libGLX_nvidia -- Vulkan/RTX rendering will fail or silently fall back to a slow CPU path"'

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
step "[4/4] ar4_pedestal_grasp_trivial_check.py (single point, close-up+elbow video, realistic cube friction)"
T4_START=$(date +%s)
sudo docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/ar4_pedestal_grasp_trivial_check.py --headless" \
  2>&1 | tee "$HOME/pedestal_grasp_trivial_check.log"
check "${PIPESTATUS[0]}" "ar4_pedestal_grasp_trivial_check.py (containerized)"
T4_END=$(date +%s)
echo "TIMING run_sec=$((T4_END - T4_START))"

if grep -q "^Traceback (most recent call last):" "$HOME/pedestal_grasp_trivial_check.log"; then
  echo "WARNING: a Python Traceback was found in pedestal_grasp_trivial_check.log despite exit 0 above - inspect the log, this may NOT be a real pass." >&2
fi
if ! grep -q "^VERDICT \[" "$HOME/pedestal_grasp_trivial_check.log"; then
  echo "WARNING: no 'VERDICT [' line found - the script likely did not reach its own final summary." >&2
fi

sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-pedestal-grasp-trivial-check/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
echo "=== GCS sync (log + video/frames) ==="
gsutil -q cp "$HOME/pedestal_grasp_trivial_check.log" "$HOME/download_asset.log" "${GCS_DEST}" 2>&1 || \
  echo "WARNING: log GCS sync failed (non-fatal)"
gsutil -m cp -r "$HOME/rl/logs/videos/ar4_pedestal_grasp_trivial_check" "${GCS_DEST}" 2>&1 || \
  echo "WARNING: video/frame GCS sync failed (non-fatal - the log was already synced above)"

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))"
echo "gcs_download_sec=$((T3_END - T3_START))"
echo "run_sec=$((T4_END - T4_START))"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

#!/usr/bin/env bash
# Setup-only payload for the AR4 standalone-App-API grasp task
# (2026-07-29, ar4-isaacsim-standalone-pick task). Prepares a fresh GCP
# instance to run a STANDALONE Isaac Sim App script (SimulationApp + World +
# SingleArticulation, NOT ManagerBasedRLEnv) inside NVIDIA's official
# nvcr.io/nvidia/isaac-lab:2.3.1 container, and dumps the live SurfaceGripper
# API + NVIDIA's own standalone SurfaceGripper example so the script can be
# authored against the ACTUALLY-installed API rather than reconstructed from
# memory.
#
# Steps 1-2 (docker/toolkit + GL-library pin + image pull) are copied
# VERBATIM from scripts/_cloud_ar4_container_pipeline.sh -- do not "improve"
# them, they encode several hard-won live fixes (driver point-release pinning,
# GL library injection). Asset download reuses the GCS cache.
#
# Meant to be dispatched via `scripts/run_on_cloud_gpu.sh --detach`; writes a
# READY marker line to stdout (the tmux log) when the instance is ready to run
# the standalone script.
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

# --- [1/5] Docker + NVIDIA Container Toolkit (verbatim from container pipeline) ---
step "[1/5] Docker + NVIDIA Container Toolkit"
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
    echo "WARNING: could not find exact-match apt candidate for driver ${DRIVER_FULL} - unpinned install." >&2
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server install (unpinned fallback)"
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi broke after libnvidia-gl install (driver/library mismatch). Needs: sudo reboot then re-run." >&2
    exit 1
  fi
fi

# --- [2/5] pull the official Isaac Lab container ---
step "[2/5] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"
PULL_EXIT=$?
check "$PULL_EXIT" "docker pull ${IMAGE}"
if [ "$PULL_EXIT" -ne 0 ]; then
  echo "FATAL: cannot proceed without the container image." >&2
  exit 1
fi

# --- [3/5] download AR4 asset from GCS cache ---
step "[3/5] download AR4 asset from GCS cache"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
# container-side usd_path (see container pipeline step 6 comment)
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"

# --- [4/5] introspect the live SurfaceGripper API + NVIDIA standalone examples ---
step "[4/5] introspect live SurfaceGripper API + NVIDIA examples"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
  "$IMAGE" -c '
    echo "=== find SurfaceGripper example files ===";
    find /isaac-sim -iname "*surface_gripper*" 2>/dev/null | head -50;
    echo "=== find standalone_examples dir ===";
    ls /isaac-sim/standalone_examples 2>/dev/null; echo "---";
    find /isaac-sim/standalone_examples -iname "*gripper*" 2>/dev/null;
    echo "=== isaacsim.robot.surface_gripper module dir ===";
    /isaac-sim/python.sh -c "import isaacsim.robot.surface_gripper as m; print(m.__file__); print(dir(m))" 2>&1 | tail -20;
  ' 2>&1 | tee "$HOME/rl/surface_gripper_api_dump.txt"
check "${PIPESTATUS[0]}" "API introspection dump"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl" 2>/dev/null || true

step "[5/5] READY"
echo "SETUP_READY_MARKER instance prepared: container pulled, asset restored, API dumped to surface_gripper_api_dump.txt"

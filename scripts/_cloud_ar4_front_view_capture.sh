#!/usr/bin/env bash
# Cloud dispatch payload (2026-08-04, ar4-gripper-front-view task): render
# the AR4 gripper OPEN->CLOSE->OPEN cycle from a genuine FRONT view
# (scripts/_record_gripper_front_view_open_close.py) on a fresh GCP GPU
# instance, using the official Isaac Lab container + the GCS-cached AR4 USD
# asset, then sync the resulting mp4 to GCS.
#
# This is the STEADY-STATE (render-only) path of
# scripts/_cloud_ar4_container_pipeline.sh: it copies that script's proven,
# already-live-verified environment-setup blocks VERBATIM (docker + NVIDIA
# container toolkit + exact-driver-pinned libnvidia-gl + official container
# pull) but SKIPS the one-time asset build/upload -- the GCS cache was
# populated 2026-07-24 -- and downloads the asset instead, then runs THIS
# task's own recorder rather than the smoke-test recorder that pipeline
# hardcodes. Kept as a separate script so that proven pipeline is not
# modified for a one-off capture.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh --detach
# --on-demand, which ships this repo's committed HEAD to ~/rl and runs a
# command there. Because --detach mode has no cost-cap/teardown watcher on
# the dispatching (Pi) side, this payload installs its OWN dead-man's
# switch (a scheduled poweroff) so a hang cannot run up unbounded compute
# cost even if the Pi session dies -- the instance itself is still torn
# down by the dispatcher (Principal owns final teardown after confirming
# the GCS artifact landed).
#
# Env vars (set by the dispatching machine, passed in via `bash -c`):
#   BUILD_ASSET_SHA   Required by download_ar4_asset_from_gcs.sh's staleness
#                     check (no .git dir on a git-archive-shipped checkout).
#   RUN_ID            Timestamp/id used for the GCS output sub-path so the
#                     dispatcher knows exactly where to poll/pull from.
set -u

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() {
  if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing anyway"; fi
}

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-gripper-front-view/${RUN_ID}"
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required (asset-cache staleness check) -- aborting." >&2
  exit 1
fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"
echo "RUN_ID=${RUN_ID}"
echo "GCS_DEST=${GCS_DEST}"

# --- Dead-man's switch: hard poweroff in 50 min ---------------------------
# Bounds runaway compute cost if anything hangs and the Pi-side dispatcher
# is not around to tear down (--detach has no cost-cap watcher). A whole
# render is ~20min; 50min is generous headroom. Poweroff (not delete -- the
# instance has storage-rw scope only, no compute scope to self-delete);
# Principal deletes the (stopped or running) instance during teardown.
sudo shutdown -h +50 "rl front-view capture dead-man's switch" || \
  echo "WARNING: could not schedule dead-man's-switch poweroff"

cd "$HOME/rl" || exit 1

# --- [1] Docker + NVIDIA Container Toolkit (verbatim from container pipeline)
step "[1] Docker + NVIDIA Container Toolkit"
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

# NVIDIA OpenGL/Vulkan userspace libs, pinned to the EXACT running driver
# version (verbatim rationale from the container pipeline -- an unpinned
# install can pull a newer point release than the booted kernel module and
# break nvidia-smi until reboot).
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
    echo "WARNING: no apt candidate matching exact driver ${DRIVER_FULL} - unpinned fallback." >&2
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server install (unpinned fallback)"
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi broke after installing libnvidia-gl (driver/library mismatch) -- needs reboot+re-run." >&2
    exit 1
  fi
fi

step "[1b] GPU passthrough sanity check"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
check $? "GPU visible inside a container"

# --- [2] pull the official Isaac Lab container ----------------------------
step "[2] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"
PULL_EXIT=$?
check "$PULL_EXIT" "docker pull ${IMAGE}"
if [ "$PULL_EXIT" -ne 0 ]; then
  echo "FATAL: cannot proceed without the container image." >&2
  exit 1
fi

# --- [3] download the AR4 USD asset from the GCS cache --------------------
step "[3] download cached AR4 asset from GCS"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: asset download did not produce assets/ar4_mk5/ar4_mk5.usd -- aborting." >&2
  exit 1
fi
# usd_path.txt is rewritten by the download script to the HOST path; every
# consumer here reads it from INSIDE the container where the same files are
# bind-mounted at /workspace/rl (see container-pipeline comment for the full
# root cause). Overwrite with the container-side path.
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"

# --- [4] render the FRONT-view open/close cycle, containerized -------------
step "[4] render front-view gripper open/close cycle (headless, containerized)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/_record_gripper_front_view_open_close.py --headless" \
  2>&1 | tee "$HOME/front_view_render.log"
RENDER_EXIT="${PIPESTATUS[0]}"
check "$RENDER_EXIT" "front-view render inside container"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

VIDEO="$HOME/rl/logs/videos/ar4_gripper_open_close_front_view.mp4"
if [ ! -f "$VIDEO" ]; then
  echo "FATAL: expected video not produced at ${VIDEO}." >&2
  exit 1
fi

# --- [5] sync artifacts to GCS (LAST step -- durable even if Pi died) ------
step "[5] sync video + log to GCS: ${GCS_DEST}"
gsutil -m cp "$VIDEO" "${GCS_DEST}/ar4_gripper_open_close_front_view.mp4" 2>&1 || echo "WARNING: video GCS sync failed"
gsutil -m cp "$HOME/front_view_render.log" "$HOME/download_asset.log" "${GCS_DEST}/" 2>&1 || echo "WARNING: log GCS sync failed"
# Explicit DONE marker object so the dispatcher can poll for completion
# without parsing logs.
echo "front-view capture complete render_exit=${RENDER_EXIT} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | \
  gsutil cp - "${GCS_DEST}/DONE" 2>&1 || echo "WARNING: DONE marker GCS write failed"

step "ALL DONE"
echo "GCS_VIDEO=${GCS_DEST}/ar4_gripper_open_close_front_view.mp4"
echo "GCS_DONE=${GCS_DEST}/DONE"

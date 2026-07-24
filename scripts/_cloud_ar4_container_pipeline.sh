#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-24, container+cache cloud infra
# task): replaces the from-scratch recipe (~15-20min pip-installing Isaac
# Sim/Isaac Lab + ~10-20min rebuilding the AR4 USD asset from the vendor
# ROS package, EVERY dispatch -- see docs/cloud/franka-cloud-shakedown.md
# and docs/cloud/dispatch-checklist.md) with:
#   1. Docker + NVIDIA Container Toolkit (one-time per instance, minutes not
#      tens of minutes).
#   2. NVIDIA's own official container `nvcr.io/nvidia/isaac-lab:2.3.1`
#      (confirmed exact match to this project's pinned Isaac Lab v2.3.1 /
#      Isaac Sim 5.1.0 stack -- IsaacLab's own `v2.3.1` tag sets
#      ISAACSIM_VERSION=5.1.0 in docker/.env.base) pulled directly from NGC,
#      instead of a from-scratch pip install.
#   3. The AR4 USD asset downloaded from this project's GCS cache
#      (scripts/download_ar4_asset_from_gcs.sh) instead of rebuilt from the
#      vendor ROS package + scripts/build_asset.py.
#
# This script ALSO performs the one-time work that populates that GCS
# cache in the first place (steps 4-5 below) -- a future dispatch does NOT
# need to repeat those steps; it only needs steps 1-2 (env) + step 6
# (cache download) + whatever its own task command is. Steps 6-7 below
# deliberately simulate exactly that future/steady-state path (wiping the
# just-built local asset and re-downloading it from GCS) so this same run
# produces real timing evidence for BOTH the one-time setup cost AND the
# real steady-state per-dispatch cost.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance (via `git archive`,
# so no .git dir is present -- see BUILD_ASSET_SHA handling below) and runs
# a command there with ~/rl as the working directory.
#
# Env vars this script reads:
#   BUILD_ASSET_SHA   Required (no .git dir on a git-archive-shipped
#                      checkout to derive it from locally). Set by the
#                      DISPATCHING machine to
#                      `git log -1 --format=%H -- scripts/build_asset.py`
#                      from ITS OWN full checkout, and passed through into
#                      the remote command. Used by upload/download_ar4_
#                      asset_from_gcs.sh for cache versioning/staleness.
#
# Deliberately does NOT use `set -e` (same reasoning as this repo's other
# _cloud_ar4_*.sh dispatch payloads): every step below checks its own exit
# code explicitly via check() and keeps going, so a single incidental
# failure doesn't silently discard the rest of an expensive run's evidence.
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

# --- [1/7] Docker + NVIDIA Container Toolkit -------------------------------
step "[1/7] Docker + NVIDIA Container Toolkit"
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

step "[1b/7] GPU passthrough sanity check (small image, fails fast before the big pull if broken)"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
check $? "GPU visible inside a container"

# --- [2/7] pull the official Isaac Lab container ---------------------------
step "[2/7] docker pull ${IMAGE}"
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

# --- [3/7] probe container layout (cheap, confirms assumptions live) -------
# IMPORTANT: the published nvcr.io/nvidia/isaac-lab image's default
# ENTRYPOINT is /isaac-sim/runheadless.sh (Isaac Sim's own streaming-app
# launcher), NOT a plain shell -- confirmed live via `docker inspect
# --format '{{json .Config.Entrypoint}}'` (2026-07-24, this task). This
# differs from IsaacLab's own docker-compose.yaml (`entrypoint: bash`),
# which only applies when building/running via that compose file's own
# `container.py` wrapper locally -- the raw NGC-published image keeps
# Isaac Sim's own default entrypoint. Every docker run below MUST pass
# `--entrypoint bash` or a plain `bash -c "..."` command silently becomes
# an (invalid/ignored) argument to runheadless.sh instead of actually
# running -- found live the hard way: an earlier version of this script
# without --entrypoint bash produced generic Isaac Sim Kit startup logs
# instead of scripts/build_asset.py's own output, because build_asset.py
# was never actually invoked.
step "[3/7] probe container layout"
sudo docker run --rm --entrypoint bash "$IMAGE" -c \
  "echo isaaclab.sh: \$(ls -la /workspace/isaaclab/isaaclab.sh 2>&1); echo ISAACSIM_PATH=\$ISAACSIM_PATH; python3 -c 'import isaacsim; print(\"isaacsim import OK\")' 2>&1 | tail -3"
check $? "container layout probe"

# --- [4/7] ONE-TIME: build the AR4 asset inside the container --------------
# (populates the GCS cache this and every future dispatch uses -- NOT part
# of the steady-state per-dispatch cost measured at the end of this script)
step "[4/7] ONE-TIME asset build: vendor ROS package + build_asset.py, inside the container"
T4_START=$(date +%s)
rm -rf "$HOME/ar4_ros_driver"
git clone --quiet https://github.com/Annin-Robotics/ar4_ros_driver.git "$HOME/ar4_ros_driver"
check $? "ar4_ros_driver clone"

mkdir -p "$HOME/ament_shim/ament_index_python"
touch "$HOME/ament_shim/ament_index_python/__init__.py"
cat > "$HOME/ament_shim/ament_index_python/packages.py" <<'PYEOF'
import os


class PackageNotFoundError(Exception):
    pass


# Minimal from-scratch reimplementation of the one function build_asset.py's
# xacro invocation actually needs ($(find annin_ar4_description) resolution)
# - ament_index_python is a ROS2 package, not published to PyPI. Resolves
# purely via an env var instead of the real ROS2 marker-file mechanism.
_KNOWN = {
    "annin_ar4_description": os.environ.get("AR4_DESCRIPTION_PATH", ""),
}


def get_package_share_directory(package_name):
    path = _KNOWN.get(package_name, "")
    if not path or not os.path.isdir(path):
        raise PackageNotFoundError(f"package '{package_name}' not found (ament_index_python shim)")
    return path
PYEOF

# The container's plain --entrypoint bash shell has NO system python3/pip
# on PATH at all (confirmed live, 2026-07-24) -- the only python is Isaac
# Sim's own bundled interpreter at /isaac-sim/python.sh (isaaclab.sh -p
# itself resolves to this same binary internally, see its own
# extract_python_exe()). A plain `pip install ...` fails with "pip:
# command not found" and (since this script has no `set -e`) silently lets
# the rest of the `&&` chain never run -- found live the hard way: an
# earlier version of this line produced a "build succeeded" (misleadingly
# fast, ~3s) timing number while build_asset.py had never actually
# executed. Use /isaac-sim/python.sh -m pip explicitly instead.
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -v "$HOME/ar4_ros_driver:/opt/ar4_ros_driver:ro" \
  -v "$HOME/ament_shim:/opt/ament_shim:ro" \
  -e AR4_DESCRIPTION_PATH=/opt/ar4_ros_driver/annin_ar4_description \
  -e PYTHONPATH=/opt/ament_shim \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh -m pip install --quiet xacro==2.1.1 && yes | /workspace/isaaclab/isaaclab.sh -p scripts/build_asset.py" \
  2>&1 | tee "$HOME/build_asset_container.log"
BUILD_EXIT="${PIPESTATUS[0]}"
check "$BUILD_EXIT" "build_asset.py inside container"
T4_END=$(date +%s)
echo "TIMING one_time_asset_build_sec=$((T4_END - T4_START))"
if [ "$BUILD_EXIT" -ne 0 ] || [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: asset build did not actually produce assets/ar4_mk5/ar4_mk5.usd -- aborting remaining steps (upload/download/smoke-test would all be against a missing asset)." >&2
  exit 1
fi

# The container runs as root by default (DOCKER_USER_HOME=/root, Isaac
# Lab's own official docker layout) -- normalize ownership of whatever it
# wrote under the bind-mounted repo so the rest of this script (running as
# the normal ssh user) can read/rm/rewrite it without needing sudo
# everywhere.
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/assets" 2>/dev/null || true

# --- [5/7] upload the freshly-built asset to the GCS cache -----------------
step "[5/7] upload built asset to GCS cache"
T5_START=$(date +%s)
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/upload_ar4_asset_to_gcs.sh 2>&1 | tee "$HOME/upload_asset.log"
check "${PIPESTATUS[0]}" "upload_ar4_asset_to_gcs.sh"
T5_END=$(date +%s)
echo "TIMING gcs_upload_sec=$((T5_END - T5_START))"

# --- [6/7] CACHE-PROOF: simulate a fresh future dispatch's asset step ------
step "[6/7] CACHE-PROOF: wipe local asset, re-download from GCS (this IS the steady-state future-dispatch step)"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
T6_START=$(date +%s)
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
T6_END=$(date +%s)
echo "TIMING gcs_download_sec=$((T6_END - T6_START))"

# --- [7/7] SMOKE TEST: gripper open/close cycle video, using the cache ----
step "[7/7] SMOKE TEST: gripper open/close cycle video capture, containerized, using the GCS-cached asset"
T7_START=$(date +%s)
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/_record_jaw_fix_open_close_cycle.py --headless" \
  2>&1 | tee "$HOME/smoke_test.log"
check "${PIPESTATUS[0]}" "gripper open/close cycle video capture (containerized)"
T7_END=$(date +%s)
echo "TIMING smoke_test_run_sec=$((T7_END - T7_START))"

sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

GCS_DEST="gs://rl-manipulation-hks-runs/container-cache-pipeline-verify/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp "$HOME/build_asset_container.log" "$HOME/upload_asset.log" "$HOME/download_asset.log" "$HOME/smoke_test.log" "${GCS_DEST}" 2>&1 || \
  echo "WARNING: log GCS sync failed (non-fatal)"

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))    (COLD pull -- this is the real per-fresh-instance cost)"
echo "one_time_asset_build_sec=$((T4_END - T4_START))   (ONE-TIME ONLY -- not part of future dispatch cost)"
echo "gcs_upload_sec=$((T5_END - T5_START))   (ONE-TIME ONLY -- not part of future dispatch cost)"
echo "gcs_download_sec=$((T6_END - T6_START))   (real future-dispatch cache-restore cost)"
echo "smoke_test_run_sec=$((T7_END - T7_START))"
STEADY_STATE=$(( (T1_END - T1_START) + (T2_END - T2_START) + (T6_END - T6_START) ))
echo "REAL STEADY-STATE FUTURE-DISPATCH SETUP TIME (docker/toolkit install + cold container pull + GCS asset download) = ${STEADY_STATE}s"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

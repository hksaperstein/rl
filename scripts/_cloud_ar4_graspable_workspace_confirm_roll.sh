#!/usr/bin/env bash
# One-shot cloud dispatch payload for the ar4-graspable-roll-constraint
# task's live-confirmation step (2026-07-27). Uses the container + GCS-
# cached-asset path (docs/cloud/dispatch-checklist.md's preferred AR4 cloud
# recipe) - steps 1-3 below are copied verbatim (same reasoning as the prior
# _cloud_ar4_graspable_workspace_confirm.sh) from
# scripts/_cloud_ar4_container_pipeline.sh's own steady-state steps.
#
# Two-part actual task, both in ONE instance (avoid paying provisioning/
# container-pull cost twice):
#   [4/5] scripts/ar4_graspable_workspace_confirm_numeric.py - the real
#         rigor check, camera-free, validates 3 genuinely distinct
#         roll-constrained FK points in a single Isaac Sim launch (proves
#         the fix generalizes, not a lucky single point).
#   [5/5] scripts/ar4_graspable_workspace_confirm.py (camera-enabled) at
#         the PRIMARY recommended point ONLY - for a demo video. Only run
#         if [4/5] confirms the grasp+lift at that same point (no point
#         paying for a camera run whose physics already failed
#         numerically). This is the step that previously stalled for 15+
#         minutes on a rendering-pipeline issue tied (by inferred timing)
#         to the pre-fix jammed-contact bug - worth re-attempting now that
#         the open-gripper collision this task fixes may have been the
#         actual cause; if it stalls again, the numeric result from [4/5]
#         already stands as the real evidence regardless.
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

# --- [1/5] Docker + NVIDIA Container Toolkit -------------------------------
step "[1/5] Docker + NVIDIA Container Toolkit"
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

# 2026-07-27 finding (prior dispatch, ar4-graspable-workspace-from-fk task):
# do NOT install libnvidia-gl-<major>-server - it breaks an already-working
# nvidia-smi/CUDA (apt's only candidate is a newer point release than the
# loaded kernel module, no matching older version to pin to, and a
# mid-script reboot can't cleanly resume under run_on_cloud_gpu.sh's
# wrapper). Camera rendering (needed only for step 5 below) degrades to a
# slower CPU/software path rather than crashing when this library is
# absent - an acceptable trade for keeping CUDA/PhysX GPU compute intact.
echo "[SKIP] libnvidia-gl-<major>-server install intentionally skipped - nvidia-smi/CUDA left intact; camera rendering (step 5 only) may fall back to a slower CPU/software path."
T1_END=$(date +%s)
echo "TIMING docker_toolkit_install_sec=$((T1_END - T1_START))"

step "[1b/5] GPU passthrough sanity check"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
check $? "GPU visible inside a container"

step "[2/5] docker pull ${IMAGE}"
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

# --- [3/5] download the AR4 USD asset from the GCS cache -------------------
step "[3/5] download AR4 asset from GCS cache"
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

# --- [4/5] THE REAL RIGOR CHECK: numeric multi-point confirmation ----------
step "[4/5] ar4_graspable_workspace_confirm_numeric.py (3 roll-constrained points, direct joint-target control, no IK, no camera)"
T4_START=$(date +%s)
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/ar4_graspable_workspace_confirm_numeric.py --headless" \
  2>&1 | tee "$HOME/graspable_workspace_confirm_numeric.log"
check "${PIPESTATUS[0]}" "ar4_graspable_workspace_confirm_numeric.py (containerized)"
T4_END=$(date +%s)
echo "TIMING numeric_confirm_run_sec=$((T4_END - T4_START))"

if grep -q "^Traceback (most recent call last):" "$HOME/graspable_workspace_confirm_numeric.log"; then
  echo "WARNING: a Python Traceback was found in graspable_workspace_confirm_numeric.log despite exit 0 above - inspect the log, this may NOT be a real pass." >&2
fi
if ! grep -q "^ALL POINTS CONFIRMED:" "$HOME/graspable_workspace_confirm_numeric.log"; then
  echo "WARNING: no 'ALL POINTS CONFIRMED:' line found - the script likely did not reach its own final summary." >&2
fi

# --- [5/5] Camera-enabled run at the PRIMARY point, for a demo video -------
# Only attempted if [4/5]'s own P0 (primary recommended point) verdict was a
# real CONFIRMED - no point paying for a camera run whose physics already
# failed numerically at that same point.
P0_CONFIRMED=false
if grep -q "P0_recommended_bearing94: GRASP+LIFT CONFIRMED" "$HOME/graspable_workspace_confirm_numeric.log"; then
  P0_CONFIRMED=true
fi
echo "P0_CONFIRMED=${P0_CONFIRMED}"

if [ "$P0_CONFIRMED" = "true" ]; then
  step "[5/5] ar4_graspable_workspace_confirm.py (camera-enabled, primary point only, for demo video)"
  T5_START=$(date +%s)
  timeout 1200 sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" \
    -w /workspace/rl \
    "$IMAGE" \
    -c "/workspace/isaaclab/isaaclab.sh -p scripts/ar4_graspable_workspace_confirm.py --headless --video-suffix roll_fix" \
    2>&1 | tee "$HOME/graspable_workspace_confirm_camera.log"
  CAMERA_EXIT="${PIPESTATUS[0]}"
  check "$CAMERA_EXIT" "ar4_graspable_workspace_confirm.py camera run (containerized, 20min timeout)"
  T5_END=$(date +%s)
  echo "TIMING camera_confirm_run_sec=$((T5_END - T5_START))"
  if [ "$CAMERA_EXIT" -eq 124 ]; then
    echo "WARNING: camera-enabled run TIMED OUT after 1200s (20min) - likely the same rendering-pipeline stall seen previously. Numeric confirmation from step 4 stands regardless." >&2
  fi
else
  step "[5/5] SKIPPED - P0 not confirmed numerically in step 4, not worth the camera run's cost/time"
  T5_START=$(date +%s)
  T5_END=$(date +%s)
fi

sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-graspable-workspace-confirm-roll/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
echo "=== GCS sync (logs + videos) ==="
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal, may not exist if step 5 skipped)"
gsutil -q cp "$HOME/graspable_workspace_confirm_numeric.log" "$HOME/download_asset.log" "${GCS_DEST}" 2>&1 || \
  echo "WARNING: log GCS sync failed (non-fatal)"
if [ -f "$HOME/graspable_workspace_confirm_camera.log" ]; then
  gsutil -q cp "$HOME/graspable_workspace_confirm_camera.log" "${GCS_DEST}" 2>&1 || echo "WARNING: camera log GCS sync failed (non-fatal)"
fi

step "TIMING SUMMARY"
echo "docker_toolkit_install_sec=$((T1_END - T1_START))"
echo "container_pull_sec=$((T2_END - T2_START))"
echo "gcs_download_sec=$((T3_END - T3_START))"
echo "numeric_confirm_run_sec=$((T4_END - T4_START))"
echo "camera_confirm_run_sec=$((T5_END - T5_START))"
echo "P0_CONFIRMED=${P0_CONFIRMED}"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

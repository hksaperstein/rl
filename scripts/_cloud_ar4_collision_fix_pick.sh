#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-08-03, ar4-gripper-collision-
# approximation-fix task). Rebuilds the AR4 USD asset with the
# convexHull->convexDecomposition gripper-collision fix (scripts/build_asset.py's
# _set_gripper_collision_convex_decomposition), re-uploads it to the GCS
# cache under the NEW build_asset.py sha, verifies the notch-preserving
# approximation took, then retries the GENUINE TOP-DOWN TABLE PICK of the
# 15mm cube on a pedestal (scripts/ar4_pedestal_grasp_confirm.py --video) --
# the pick that was concluded impossible under convexHull. Prints numeric
# ground-truth grasp+lift verdicts AND writes one elevated-3/4 mp4 per point.
#
# RESILIENCE (Pi session has been unstable): meant to be dispatched with
# `scripts/run_on_cloud_gpu.sh --detach --on-demand --cost-cap 5`. In detach
# mode the wrapper does NOT tear the instance down and does NOT enforce the
# cost cap (nothing is watching after it returns). The instances are created
# with --scopes=storage-rw only, so they CANNOT self-delete via gcloud -- but
# they CAN sync to GCS and `sudo poweroff`. So this payload:
#   - self-bounds total runtime via a top-level `timeout` re-exec (90 min),
#   - syncs ALL logs+videos+a DONE marker to GCS as its last real step
#     (so results survive even if the Pi session died meanwhile),
#   - then `sudo poweroff`s to HALT the instance (TERMINATED = zero compute
#     billing) even if nobody is watching. The dispatcher deletes the halted
#     instance + disk from the Pi once it sees the DONE marker in GCS.
#
# Env: BUILD_ASSET_SHA (required -- see _cloud_ar4_container_pipeline.sh's
# header for why: no .git dir on a git-archive-shipped checkout).
set -u

# --- top-level runtime bound (self re-exec under timeout) -------------------
if [ -z "${_CF_TIMEOUT_GUARD:-}" ]; then
  export _CF_TIMEOUT_GUARD=1
  exec timeout --signal=TERM 5400 bash "$0" "$@"
fi

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing anyway"; fi; }

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required." >&2
  exit 1
fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1
STAMP="$(date -u +%Y%m%d-%H%M%S)"
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-collision-fix-pick/${STAMP}"
echo "GCS_DEST=${GCS_DEST}"

# --- EXIT/signal trap: sync whatever exists + self-halt, no matter how we exit
_FINISHED=0
finish() {
  local rc=$?
  [ "$_FINISHED" -eq 1 ] && return
  _FINISHED=1
  step "FINISH (rc=${rc}) -- sync results to GCS + self-halt"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" "$HOME/rl/assets" 2>/dev/null || true
  gsutil -m cp "$HOME"/*.log "${GCS_DEST}/" 2>&1 || echo "WARN: log sync failed"
  if [ -d "$HOME/rl/logs/videos/ar4_pedestal_grasp_confirm" ]; then
    gsutil -m cp -r "$HOME/rl/logs/videos/ar4_pedestal_grasp_confirm" "${GCS_DEST}/videos/" 2>&1 || echo "WARN: video sync failed"
  fi
  {
    echo "task=ar4-collision-fix-pick"
    echo "build_asset_sha=${BUILD_ASSET_SHA}"
    echo "exit_rc=${rc}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- verify verdict ---"
    grep -E "VERIFY VERDICT|approximation =" "$HOME/verify_approx.log" 2>/dev/null || echo "(no verify log)"
    echo "--- pick summary ---"
    grep -E "VERDICT \[|FINAL MULTI-POINT|ALL POINTS CONFIRMED|open_gripper_max_force|height_gain" "$HOME/pick_video.log" 2>/dev/null || echo "(no pick log)"
  } > "$HOME/DONE.txt"
  gsutil cp "$HOME/DONE.txt" "${GCS_DEST}/DONE.txt" 2>&1 || echo "WARN: DONE marker sync failed"
  echo "Halting instance now (self-poweroff) -- dispatcher deletes it from the Pi after seeing DONE.txt."
  sudo poweroff
}
trap finish EXIT TERM INT

# --- [1/6] Docker + NVIDIA Container Toolkit + GL libs (video needs GL) -----
step "[1/6] Docker + NVIDIA Container Toolkit + GL libs"
if command -v docker >/dev/null 2>&1; then echo "docker present"; else sudo apt-get update -y; sudo apt-get install -y docker.io; check $? "docker"; fi
sudo systemctl enable --now docker; check $? "docker daemon"
if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then echo "toolkit present"; else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y; sudo apt-get install -y nvidia-container-toolkit; check $? "toolkit"; fi
sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker; check $? "docker+toolkit"
# NVIDIA OpenGL/Vulkan userspace libs -- REQUIRED for camera RTX render (video).
# Pin to the EXACT running driver version to avoid a point-release mismatch.
if ! find /usr/lib/x86_64-linux-gnu -iname 'libGLX_nvidia*' 2>/dev/null | grep -q .; then
  DF="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; DM="$(echo "$DF" | cut -d. -f1)"
  AV="$(apt-cache madison "libnvidia-gl-${DM}-server" 2>/dev/null | awk -F'|' -v v="$DF" '{gsub(/^[ \t]+|[ \t]+$/,"",$2); if (index($2, v) == 1) {print $2; exit}}')"
  if [ -n "$AV" ]; then sudo apt-get install -y "libnvidia-gl-${DM}-server=${AV}"; else sudo apt-get install -y "libnvidia-gl-${DM}-server"; fi
  check $? "gl libs"; nvidia-smi >/dev/null 2>&1 || { echo "FATAL nvidia-smi broke after GL install" >&2; exit 1; }
fi

step "[1b/6] GPU passthrough sanity check"
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi; check $? "GPU visible in container"

# --- [2/6] pull the Isaac Lab container ------------------------------------
step "[2/6] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"; PE=$?; check "$PE" "pull"; [ "$PE" -ne 0 ] && { echo "FATAL: no image" >&2; exit 1; }

# --- [3/6] ONE-TIME: build the fixed AR4 asset inside the container ---------
step "[3/6] build fixed AR4 asset (vendor ROS pkg + build_asset.py) inside container"
rm -rf "$HOME/ar4_ros_driver"
git clone --quiet https://github.com/Annin-Robotics/ar4_ros_driver.git "$HOME/ar4_ros_driver"; check $? "ar4_ros_driver clone"
mkdir -p "$HOME/ament_shim/ament_index_python"
touch "$HOME/ament_shim/ament_index_python/__init__.py"
cat > "$HOME/ament_shim/ament_index_python/packages.py" <<'PYEOF'
import os
class PackageNotFoundError(Exception):
    pass
_KNOWN = {"annin_ar4_description": os.environ.get("AR4_DESCRIPTION_PATH", "")}
def get_package_share_directory(package_name):
    path = _KNOWN.get(package_name, "")
    if not path or not os.path.isdir(path):
        raise PackageNotFoundError(f"package '{package_name}' not found (ament_index_python shim)")
    return path
PYEOF
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" \
  -v "$HOME/ar4_ros_driver:/opt/ar4_ros_driver:ro" \
  -v "$HOME/ament_shim:/opt/ament_shim:ro" \
  -e AR4_DESCRIPTION_PATH=/opt/ar4_ros_driver/annin_ar4_description \
  -e PYTHONPATH=/opt/ament_shim \
  -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh -m pip install --quiet xacro==2.1.1 && printf '#!/bin/bash\nexec /isaac-sim/python.sh /isaac-sim/kit/python/bin/xacro \"\$@\"\n' > /usr/local/bin/xacro && chmod +x /usr/local/bin/xacro && yes | /workspace/isaaclab/isaaclab.sh -p scripts/build_asset.py" \
  2>&1 | tee "$HOME/build_asset_container.log"
BUILD_EXIT="${PIPESTATUS[0]}"; check "$BUILD_EXIT" "build_asset.py inside container"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/assets" 2>/dev/null || true
if [ "$BUILD_EXIT" -ne 0 ] || [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: asset build did not produce assets/ar4_mk5/ar4_mk5.usd -- aborting." >&2; exit 1
fi
# Confirm the fix function actually ran during the build.
echo "--- gripper-collision-approx log lines from the build ---"
grep -E "gripper-collision-approx" "$HOME/build_asset_container.log" || echo "WARNING: no [gripper-collision-approx] lines in build log!"

# --- [3b/6] upload the freshly-built (fixed) asset to the GCS cache ---------
step "[3b/6] upload fixed asset to GCS cache (new build_asset.py sha)"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/upload_ar4_asset_to_gcs.sh 2>&1 | tee "$HOME/upload_asset.log"; check "${PIPESTATUS[0]}" "upload to GCS"

# --- [4/6] VERIFY: collision approximation now convexDecomposition ----------
step "[4/6] verify gripper collision approximation (pure USD, no RTX)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/_verify_gripper_collision_approx.py /workspace/rl/assets/ar4_mk5/ar4_mk5.usd" \
  2>&1 | tee "$HOME/verify_approx.log"
check "${PIPESTATUS[0]}" "collision approximation verify"

# --- [5/6] THE PICK: genuine top-down table (pedestal) pick + video --------
step "[5/6] top-down table pick of the 15mm cube (--video, 3 validation points)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
  -c "/workspace/isaaclab/isaaclab.sh -p scripts/ar4_pedestal_grasp_confirm.py --video --headless" \
  2>&1 | tee "$HOME/pick_video.log"
check "${PIPESTATUS[0]}" "pedestal top-down pick + video"

# --- [6/6] echo the key results (also captured into DONE.txt by the trap) ---
step "[6/6] KEY RESULTS"
echo "=== VERIFY ==="; grep -E "VERIFY VERDICT|approximation =" "$HOME/verify_approx.log" 2>/dev/null
echo "=== PICK ==="; grep -E "VERDICT \[|FINAL MULTI-POINT|ALL POINTS CONFIRMED|Open-gripper collision-free|open_gripper_max_force|Max cube height|height_gain|Held through retreat|BOTH jaws" "$HOME/pick_video.log" 2>/dev/null
echo "ALL DONE (payload body complete; EXIT trap will sync to GCS + self-halt)."

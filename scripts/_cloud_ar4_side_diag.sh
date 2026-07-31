#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-31, ar4-side-grasp task, DIAGNOSTIC
# follow-up): the first side-grasp run found the arm settling ~32mm low + ~30mm
# too far -X + 32deg off the horizontal orientation target BEFORE the servo,
# with jaws pushed partially closed -- a collision/blocking signature. Per this
# project's standing kb lesson (2026-07-31 UPDATE: "remove the scene obstacles
# and re-test in FREE SPACE before blaming the actuator/controller"), this run
# is the decisive A/B control: does the arm reach the horizontal GRASP_Q in
# FREE SPACE (obstacles removed) vs WITH the pedestal+cube present?
#   PHASE A: --side_grasp --empty_scene --droop_diagnostic --quick  (FREE SPACE)
#   PHASE B: --side_grasp             --droop_diagnostic --quick  (OBSTACLES)
# Both physics-only (no video), bounded/fast (--quick exits after the per-joint
# tracking report + USD joint-limit readout). Compare per-joint achieved-vs-
# commanded: if A tracks (<0.5deg) and B does not, the block is a scene
# collision (fix the pedestal/scene); if A also fails, the horizontal pose
# itself is unreachable/untrackable.
#
# Steps 1-3 copied VERBATIM from scripts/_cloud_ar4_side_grasp_run.sh.
# Env vars: BUILD_ASSET_SHA (required).
set -u
step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing anyway"; fi; }

if [ -z "${BUILD_ASSET_SHA:-}" ]; then echo "ERROR: BUILD_ASSET_SHA required." >&2; exit 1; fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1

step "[1/4] Docker + NVIDIA Container Toolkit"
if command -v docker >/dev/null 2>&1; then echo "docker present"; else
  sudo apt-get update -y; sudo apt-get install -y docker.io; check $? "docker.io install"; fi
sudo systemctl enable --now docker; check $? "docker daemon"
if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then echo "toolkit present"; else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y; sudo apt-get install -y nvidia-container-toolkit; check $? "toolkit install"; fi
sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker; check $? "docker+toolkit ready"
if ! find /usr/lib/x86_64-linux-gnu -iname 'libGLX_nvidia*' 2>/dev/null | grep -q .; then
  DRIVER_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  DRIVER_MAJOR="$(echo "$DRIVER_FULL" | cut -d. -f1)"
  APT_VER="$(apt-cache madison "libnvidia-gl-${DRIVER_MAJOR}-server" 2>/dev/null | awk -F'|' -v v="$DRIVER_FULL" '{gsub(/^[ \t]+|[ \t]+$/,"",$2); if (index($2, v) == 1) {print $2; exit}}')"
  if [ -n "$APT_VER" ]; then sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server=${APT_VER}"; check $? "gl pinned";
  else sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"; check $? "gl unpinned"; fi
  if ! nvidia-smi >/dev/null 2>&1; then echo "FATAL: nvidia-smi broke; needs reboot+rerun." >&2; exit 1; fi
fi

step "[2/4] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"; PULL_EXIT=$?; check "$PULL_EXIT" "pull"
if [ "$PULL_EXIT" -ne 0 ]; then echo "FATAL: no image." >&2; exit 1; fi

step "[3/4] download AR4 asset from GCS cache"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "asset download"
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then echo "FATAL: asset missing." >&2; exit 1; fi
mkdir -p "$HOME/rl/logs"

run_phase() {
  local tag="$1"; shift
  step "$tag"
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
    "$IMAGE" -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py $*" \
    2>&1 | tee "$HOME/rl/logs/${tag}.log"
  check "${PIPESTATUS[0]}" "$tag"
}

# --- [4a] FREE SPACE (obstacles removed) ----------------------------------
run_phase "side_diag_A_freespace" --side_grasp --empty_scene --droop_diagnostic --quick --no_video --mechanism friction
echo; echo "===== PHASE A (FREE SPACE) per-joint tracking ====="
grep -E "DIAG A_grav_on|DIAG LIMITS|pad_z|EMPTY_SCENE" "$HOME/rl/logs/side_diag_A_freespace.log" 2>/dev/null | head -40
echo "=================================================="

# --- [4b] OBSTACLES PRESENT ------------------------------------------------
run_phase "side_diag_B_obstacles" --side_grasp --droop_diagnostic --quick --no_video --mechanism friction
echo; echo "===== PHASE B (OBSTACLES PRESENT) per-joint tracking ====="
grep -E "DIAG A_grav_on|DIAG LIMITS|pad_z" "$HOME/rl/logs/side_diag_B_obstacles.log" 2>/dev/null | head -40
echo "========================================================="

sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-side-grasp/diag-$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp "$HOME/rl/logs/side_diag_A_freespace.log" "$HOME/rl/logs/side_diag_B_obstacles.log" \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log sync failed"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

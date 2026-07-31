#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-30, ar4-gravity-droop task):
# prepare a fresh GCP instance (docker/toolkit + GL-lib pin + isaac-lab
# container pull + GCS-cached AR4 asset), then run TWO phases on the SAME
# instance:
#   PHASE 1 (fast, no video): the JOINT-LEVEL DROOP DIAGNOSTIC
#     (--droop_diagnostic): command the reachable GRASP_Q, settle to steady
#     state, report per-arm-joint commanded-vs-achieved angle + measured
#     effort vs the drive maxForce, introspect the generalized-gravity API,
#     run a GRAVITY-OFF control, and test feedforward gravity comp (both
#     signs). This is the decisive test for the persistent 11-16mm vertical
#     droop that blocked runs 1-4 -- do it BEFORE any more full pick runs.
#   PHASE 2 (with video): the PURE-FRICTION pick at the reachable config WITH
#     FEEDFORWARD GRAVITY COMPENSATION (--gravity_comp): the fix the droop
#     hypothesis points to. Pure friction, NO joint/weld. Then sync video +
#     both result logs to GCS.
#
# Steps 1-3 (docker/toolkit + GL pin + image pull + GCS asset download) are
# copied VERBATIM from scripts/_cloud_ar4_reachable_grasp_run.sh -- do NOT
# "improve" them, they encode hard-won live fixes (driver point-release
# pinning, GL library injection, container-side usd_path rewrite).
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

# --- [1/4] Docker + NVIDIA Container Toolkit (verbatim) --------------------
step "[1/4] Docker + NVIDIA Container Toolkit"
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
    echo "WARNING: no exact-match apt candidate for driver ${DRIVER_FULL} - unpinned install." >&2
    sudo apt-get install -y "libnvidia-gl-${DRIVER_MAJOR}-server"
    check $? "libnvidia-gl-${DRIVER_MAJOR}-server install (unpinned fallback)"
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi broke after libnvidia-gl install (driver/library mismatch). Needs: sudo reboot then re-run." >&2
    exit 1
  fi
fi

# --- [2/4] pull the official Isaac Lab container ---------------------------
step "[2/4] docker pull ${IMAGE}"
sudo docker pull "$IMAGE"
PULL_EXIT=$?
check "$PULL_EXIT" "docker pull ${IMAGE}"
if [ "$PULL_EXIT" -ne 0 ]; then
  echo "FATAL: cannot proceed without the container image." >&2
  exit 1
fi

# --- [3/4] download AR4 asset from GCS cache ------------------------------
step "[3/4] download AR4 asset from GCS cache"
rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"
check "${PIPESTATUS[0]}" "download_ar4_asset_from_gcs.sh"
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
if [ ! -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ]; then
  echo "FATAL: AR4 USD asset missing after GCS download -- aborting." >&2
  exit 1
fi

mkdir -p "$HOME/rl/logs"

# --- [4a/4] PHASE 1: joint-level droop diagnostic (fast, no video) ---------
# --no_video keeps RENDER off -> avoids the ~10min first-render RTX init; this
# is a physics-only steady-state measurement. Decisive before any pick run.
step "[4a/4] PHASE 1 droop diagnostic (--droop_diagnostic, no video)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --droop_diagnostic --no_video --mechanism friction" \
  2>&1 | tee "$HOME/rl/logs/droop_diagnostic.log"
DIAG_EXIT="${PIPESTATUS[0]}"
check "$DIAG_EXIT" "droop diagnostic run"
# the diagnostic writes its measured lines to standalone_pick_result_friction.txt too;
# preserve a diagnostic-specific copy before phase 2 overwrites it.
cp "$HOME/rl/logs/standalone_pick_result_friction.txt" \
   "$HOME/rl/logs/standalone_pick_result_DIAGNOSTIC.txt" 2>/dev/null || true

echo
echo "===== DIAGNOSTIC KEY LINES ====="
grep -E "\[DIAG" "$HOME/rl/logs/droop_diagnostic.log" 2>/dev/null || true
echo "================================"

# --- [4b/4] PHASE 2: pure-friction pick WITH gravity comp (video) ----------
# Video ON (RTX headless render). Pure friction, NO joint. Feedforward gravity
# compensation (--gravity_comp) is the fix the droop hypothesis points to; its
# sign is auto-calibrated live at the grasp pose. Default squeeze (position
# mode, ~33N via GRIP_KP) -- a 10g cube needs only ~0.12N.
step "[4b/4] PHASE 2 pure-friction pick WITH gravity comp (--gravity_comp, video)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
  "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --mechanism friction --squeeze_mode position --squeeze_force 5.0 --gravity_comp" \
  2>&1 | tee "$HOME/rl/logs/gravcomp_pick_run.log"
RUN_EXIT="${PIPESTATUS[0]}"
check "$RUN_EXIT" "gravity-comp pick run"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

echo
echo "===== PICK VERDICT + KEY LINES ====="
grep -E "VERDICT|\[GRAVCOMP\]|\[CONTACT|\[JAW SUMMARY\]|cube_z gain|real lift" "$HOME/rl/logs/gravcomp_pick_run.log" 2>/dev/null || true
echo "===================================="

# --- sync artifacts to GCS ------------------------------------------------
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-gravity-droop/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp \
  "$HOME/rl/logs/droop_diagnostic.log" \
  "$HOME/rl/logs/gravcomp_pick_run.log" \
  "$HOME/rl/logs/standalone_pick_result_DIAGNOSTIC.txt" \
  "$HOME/rl/logs"/standalone_pick_result_friction.txt \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE (diag exit=${DIAG_EXIT} pick exit=${RUN_EXIT})."

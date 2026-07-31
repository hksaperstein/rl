#!/usr/bin/env bash
# One-shot cloud dispatch (2026-07-31, side-grasp): render a WATCHABLE video of
# the side-grasp attempt (standing user directive: video every attempt, success
# or not). Single video phase with the fixed near-level cameras + brighter
# lighting. Setup copied VERBATIM from _cloud_ar4_side_diag.sh. Env: BUILD_ASSET_SHA.
set -u
step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing"; fi; }
if [ -z "${BUILD_ASSET_SHA:-}" ]; then echo "ERROR: BUILD_ASSET_SHA required." >&2; exit 1; fi
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"; cd "$HOME/rl" || exit 1
step "[1/4] Docker + toolkit"
if command -v docker >/dev/null 2>&1; then echo "docker present"; else sudo apt-get update -y; sudo apt-get install -y docker.io; check $? "docker"; fi
sudo systemctl enable --now docker; check $? "docker daemon"
if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then echo "toolkit present"; else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y; sudo apt-get install -y nvidia-container-toolkit; check $? "toolkit"; fi
sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker; check $? "docker+toolkit"
if ! find /usr/lib/x86_64-linux-gnu -iname 'libGLX_nvidia*' 2>/dev/null | grep -q .; then
  DF="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; DM="$(echo "$DF" | cut -d. -f1)"
  AV="$(apt-cache madison "libnvidia-gl-${DM}-server" 2>/dev/null | awk -F'|' -v v="$DF" '{gsub(/^[ \t]+|[ \t]+$/,"",$2); if (index($2, v) == 1) {print $2; exit}}')"
  if [ -n "$AV" ]; then sudo apt-get install -y "libnvidia-gl-${DM}-server=${AV}"; else sudo apt-get install -y "libnvidia-gl-${DM}-server"; fi
  check $? "gl"; nvidia-smi >/dev/null 2>&1 || { echo "FATAL nvidia-smi broke" >&2; exit 1; }
fi
step "[2/4] pull image"; sudo docker pull "$IMAGE"; PE=$?; check "$PE" "pull"; [ "$PE" -ne 0 ] && { echo FATAL >&2; exit 1; }
step "[3/4] asset from GCS"; rm -rf "$HOME/rl/assets/ar4_mk5" "$HOME/rl/assets/shapes"
BUILD_ASSET_SHA="$BUILD_ASSET_SHA" bash scripts/download_ar4_asset_from_gcs.sh 2>&1 | tee "$HOME/download_asset.log"; check "${PIPESTATUS[0]}" "asset"
echo "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd" > "$HOME/rl/assets/ar4_mk5/usd_path.txt"
[ -f "$HOME/rl/assets/ar4_mk5/ar4_mk5.usd" ] || { echo "FATAL asset missing" >&2; exit 1; }
mkdir -p "$HOME/rl/logs"
step "[4/4] side-grasp VIDEO run (fixed cameras)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py --side_grasp --mechanism friction --squeeze_mode position --squeeze_force 5.0" \
  2>&1 | tee "$HOME/rl/logs/side_grasp_video_run.log"
check "${PIPESTATUS[0]}" "side-grasp video run"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
echo; echo "===== VERDICT ====="; grep -E "VERDICT|cube_z gain|real lift|CONTACT P3_after|JAW SUMMARY" "$HOME/rl/logs/side_grasp_video_run.log" 2>/dev/null | tail -8
echo "video files:"; ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-side-grasp/video-$(date -u +%Y%m%d-%H%M%S)/"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARN video sync failed"
gsutil -m cp "$HOME/rl/logs/side_grasp_video_run.log" "${GCS_DEST}" 2>&1 || echo "WARN log sync failed"
echo "GCS_DEST=${GCS_DEST}"; echo "ALL DONE."

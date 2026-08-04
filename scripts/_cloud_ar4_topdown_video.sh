#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-08-04, ar4-topdown-approach-video task):
# capture a CLEAR, WELL-FRAMED video of the AR4 gripper approaching a cube on the
# pedestal FROM THE TOP (top-down), per direct user request. This is a VIDEO-
# CAPTURE task: the top-down grasp is ALREADY PROVEN to JAM (the gripper body/palm
# hits the cube top ~9mm before the shallow jaws can straddle a 15mm cube from
# directly above -- kb ar4-vs-franka-root-cause-comparison.md's 2026-07-31 UPDATEs),
# so this is NOT a successful pick and is not meant to be; the user wants to SEE the
# top-down descent + jam clearly. Rendered UNCONDITIONALLY (no closure gate).
#
# Uses the SAME proven-working camera-capture path as the side-float SUCCESS video
# (scripts/ar4_isaacsim_standalone_pick.py's raw-SimulationApp + isaacsim.sensors.
# camera + _set_cam_lookat), NOT the pedestal_grasp_confirm.py --video path that
# failed to init the camera under Isaac Lab's AppLauncher last time.
#
# Steps 1-3 (docker/toolkit + GL pin + image pull + GCS asset download) copied
# VERBATIM from scripts/_cloud_ar4_pin_pick.sh -- do NOT "improve" them.
#
# RESILIENCE (Pi session has been unstable/restarting): this payload is designed to
# run AUTONOMOUSLY and get the video off the instance even if the dispatching Pi
# session dies:
#   * a safety `shutdown -h +45` is scheduled at the START so compute billing stops
#     no matter what (the instance is provisioned with storage-rw scope only, so it
#     CANNOT gcloud-delete itself -- a scheduled halt is the available cost backstop;
#     it transitions the VM to TERMINATED/stopped, ending GPU/CPU billing);
#   * the video + frames + logs are synced to GCS BEFORE anything else can fail;
#   * a DONE sentinel is written to GCS last so the Pi (if alive) can poll GCS for
#     completion and pull the video from GCS (never needing the instance directly);
#   * then a prompt `shutdown -h now` stops the instance immediately after sync.
# The dispatching Pi still owns final teardown (gcloud instances delete of the now-
# stopped instance + its disk); the scheduled halt only bounds cost if the Pi never
# returns.
#
# Env vars: BUILD_ASSET_SHA (required, same as every other _cloud_ar4_* payload).
set -u

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S)) ==="; }
check() { if [ "$1" -eq 0 ]; then echo "[OK] $2"; else echo "[FAIL exit=$1] $2 - continuing anyway"; fi; }

if [ -z "${BUILD_ASSET_SHA:-}" ]; then
  echo "ERROR: BUILD_ASSET_SHA env var is required -- aborting." >&2
  exit 1
fi
echo "BUILD_ASSET_SHA=${BUILD_ASSET_SHA}"

# --- RESILIENCE: schedule a safety halt so compute billing stops no matter what --
# (storage-rw scope only -> cannot self-delete; a scheduled halt is the cost cap.)
sudo shutdown -h +45 "rl-topdown-video safety halt" 2>/dev/null || true
echo "[resilience] scheduled safety 'shutdown -h +45' backstop"

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.1"
cd "$HOME/rl" || exit 1

# --- [1/4] Docker + NVIDIA Container Toolkit (verbatim from _cloud_ar4_pin_pick.sh)
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

# --- [4/4] TOP-DOWN APPROACH VIDEO run (proven camera path, --topdown_video) ---
# --no_servo: clean straight top-down descent (no Jacobian-probe jitter). friction /
# position squeeze / firm jaw-close params (same proven machinery). RENDER on (no
# --no_video). The run does HOME -> PREGRASP -> descend top-down -> attempt close ->
# lift attempt -> retreat; it JAMS above the cube (expected) and is rendered anyway.
step "[4/4] top-down approach VIDEO run (--topdown_video --no_servo)"
sudo docker run --rm --gpus all --network host --entrypoint bash \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" \
  -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py \
      --topdown_video --no_servo --mechanism friction \
      --squeeze_mode position --squeeze_force 5.0 \
      --jaw_max_force 80 --grip_kd 60 --close_steps 60" \
  2>&1 | tee "$HOME/rl/logs/topdown_video_run.log"
RUN_EXIT="${PIPESTATUS[0]}"
check "$RUN_EXIT" "top-down video run"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true

echo; echo "===== TOP-DOWN RUN KEY LINES ====="
grep -E "VERDICT|\[TOPDOWN_VIDEO\]|\[JAW SUMMARY\]|cube_z gain|real lift|held through|SUMMARY|cube_z P|\[video\]|pos_err_mm" \
  "$HOME/rl/logs/topdown_video_run.log" 2>/dev/null | tail -40 || true
echo "=================================="
echo "video files:"
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

# --- extract representative frames for framing verification (in-container) ----
step "extract frames for framing check"
sudo docker run --rm --entrypoint bash \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" -c '
/isaac-sim/python.sh - <<PYEOF
import imageio.v2 as imageio, os
vd="/workspace/rl/logs/videos/ar4_isaacsim_standalone_pick/friction"
fd="/workspace/rl/logs/topdown_frames"; os.makedirs(fd, exist_ok=True)
for name in ("closeup","elbow"):
    p=os.path.join(vd,name+".mp4")
    if not os.path.exists(p):
        print("MISSING",p); continue
    r=imageio.get_reader(p); n=r.count_frames()
    print(name,"frames=",n)
    for frac in (0.05,0.20,0.40,0.55,0.70,0.85,0.97):
        i=min(n-1,int(frac*n))
        try:
            imageio.imwrite(os.path.join(fd,f"{name}_{int(frac*100):02d}.png"), r.get_data(i))
        except Exception as e:
            print("frame err",name,i,e)
print("FRAMES_EXTRACTED")
PYEOF' 2>&1 | tee "$HOME/rl/logs/topdown_frame_extract.log"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
ls -la "$HOME/rl/logs/topdown_frames/" 2>&1 || true

# --- sync artifacts to GCS (ALWAYS, BEFORE any teardown/halt) ---------------
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-topdown-video/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp -r "$HOME/rl/logs/topdown_frames" "${GCS_DEST}" 2>&1 || echo "WARNING: frames GCS sync failed (non-fatal)"
gsutil -m cp \
  "$HOME/rl/logs/topdown_video_run.log" \
  "$HOME/rl/logs/topdown_frame_extract.log" \
  "$HOME/rl/logs"/standalone_pick_result_friction.txt \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"

# DONE sentinel LAST so a Pi poller can detect completion and pull from GCS.
echo "run_exit=${RUN_EXIT} gcs=${GCS_DEST} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | \
  gsutil cp - "${GCS_DEST}DONE.txt" 2>&1 || echo "WARNING: DONE sentinel write failed"

echo "GCS_DEST=${GCS_DEST}"
echo "TOPDOWN_VIDEO_PIPELINE_DONE run_exit=${RUN_EXIT} gcs=${GCS_DEST}"

# --- RESILIENCE: stop the instance now that everything is safely in GCS -----
echo "[resilience] artifacts in GCS; halting instance now (compute billing stops)."
sleep 5
sudo shutdown -h now "rl-topdown-video done" 2>/dev/null || true

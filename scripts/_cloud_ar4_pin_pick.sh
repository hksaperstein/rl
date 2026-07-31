#!/usr/bin/env bash
# One-shot cloud dispatch payload (2026-07-31, ar4-support-free-pin-pick task):
# land a GENUINE pure-friction top-down grasp+lift of a SUPPORT-FREE object held
# up on a THIN PIN, plus a WELL-FRAMED watchable video, on ONE fresh instance:
#
#   PHASE 0 (AABB, no video): --pin_pick --dump_gripper_aabb --empty_scene -- the
#     top-down gripper collision-mesh AABB at the reachable GRASP_Q, to VERIFY the
#     gripper BODY clears the object TOP and the thin central pin clears the
#     fingers BEFORE the grasp (the exact clearance the surface-supported object
#     could never satisfy).
#   PHASE 1 (physics-only sweep, no video): --pin_pick --no_servo --no_video over
#     several object-TOP heights, to find the highest top that both clears the
#     descending body AND gives a solid symmetric pad contact + a real ground-truth
#     lift OFF the pin. Cheap (skips the ~10min first-RTX init).
#   PHASE 2 (WITH video, ALWAYS): the best CONFIRMED top from the sweep (or a
#     default) rendered with RTX -- the FULL attempt (approach->close->lift-off-pin
#     ->hold->retreat), elevated-3/4 framing aimed at the grip point, captured
#     UNCONDITIONALLY per the standing directive (video every attempt, no gate).
#
# ROOT CAUSE this addresses (kb ar4-vs-franka-root-cause-comparison.md, all three
# 2026-07-31 UPDATEs): the AR4 gripper's central body (gripper_base_link) bottoms
# only ~1.3mm BELOW the pad plane, so it CANNOT straddle ANY object resting on a
# support surface -- proven for a 15mm cube (top-down + side-on) and a properly-
# sized tall prism. Arm/drives/tracking/servo/squeeze/jaw-closure are ALL proven
# flawless. Fix (Principal decision): present the object SUPPORT-FREE on a thin
# pin the gripper genuinely clears, with the object TOP below the body z_min.
#
# Steps 1-3 (docker/toolkit + GL pin + image pull + GCS asset download) copied
# VERBATIM from scripts/_cloud_ar4_tall_prism_pick.sh -- do NOT "improve" them.
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

# proven grasp machinery: pure friction, position squeeze; firm jaw-close params
# (kb 2026-07-31 tall-prism: jaw_max_force 50 + overdamped grip_kd 200 stalled the
# jaws; firmer close params proved BOTH jaws close fully in free space). --no_servo
# (the Jacobian probe bumps a support-free object off the thin pin).
COMMON="--mechanism friction --squeeze_mode position --squeeze_force 8.0 --no_servo \
--jaw_max_force 80 --grip_kd 60 --close_steps 60"
OBJDIMS="--pin_obj_width 0.015 --pin_obj_depth 0.018 --pin_obj_height 0.020 \
--pin_obj_mass 0.015 --pin_size 0.009"

run_in_container() {  # $1 = log file, rest = script args
  local logf="$1"; shift
  sudo docker run --rm --gpus all --network host --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y \
    -v "$HOME/rl:/workspace/rl" -w /workspace/rl \
    "$IMAGE" \
    -c "/isaac-sim/python.sh scripts/ar4_isaacsim_standalone_pick.py $*" \
    2>&1 | tee "$logf"
  return "${PIPESTATUS[0]}"
}

# --- PHASE 0: gripper AABB clearance verification (BEFORE the grasp) ---------
step "PHASE 0: top-down gripper AABB @ GRASP_Q (empty_scene) -- verify body clears object top + pin clears fingers"
run_in_container "$HOME/rl/logs/pin_aabb.log" \
  --pin_pick --dump_gripper_aabb --empty_scene --no_video $COMMON $OBJDIMS --pin_obj_top 0.0510
check $? "phase 0 AABB dump"
echo; echo "===== AABB KEY LINES ====="
grep -E "\[AABB\]|\[AABB PIN\]|pad_mid|\[PIN_PICK\]" "$HOME/rl/logs/pin_aabb.log" 2>/dev/null | tail -30 || true
echo "=========================="

# --- PHASE 1: physics-only sweep over object TOP (find graspable height) -----
BEST_TOP=""
for TOP in 0.0490 0.0510 0.0525; do
  step "PHASE 1 sweep: pin_obj_top=${TOP} (physics-only, no_servo)"
  run_in_container "$HOME/rl/logs/pin_sweep_${TOP}.log" \
    --pin_pick --pin_obj_top "$TOP" --no_video $COMMON $OBJDIMS
  check $? "phase 1 sweep top ${TOP}"
  echo "----- top ${TOP} key lines -----"
  grep -E "VERDICT|\[CONTACT P4|\[CONTACT P3b|\[JAW SUMMARY\]|cube_z gain|real lift|held through|SUMMARY|cube_z P" \
    "$HOME/rl/logs/pin_sweep_${TOP}.log" 2>/dev/null | tail -25 || true
  if grep -q "VERDICT: PICK CONFIRMED" "$HOME/rl/logs/pin_sweep_${TOP}.log" 2>/dev/null; then
    BEST_TOP="$TOP"   # keep the TALLEST confirmed (loop ascends) -> pads nearest centroid
    echo "[SWEEP] top ${TOP} CONFIRMED -> current best=${BEST_TOP}"
  fi
done

VIDEO_TOP="${BEST_TOP:-0.0510}"
echo
echo "===== SWEEP RESULT: BEST_TOP='${BEST_TOP}'  -> VIDEO_TOP=${VIDEO_TOP} ====="

# --- PHASE 2: WITH video (ALWAYS -- no closure gate) ------------------------
step "PHASE 2: pure-friction support-free pin pick WITH video (pin_obj_top=${VIDEO_TOP})"
run_in_container "$HOME/rl/logs/pin_pick_video.log" \
  --pin_pick --pin_obj_top "$VIDEO_TOP" $COMMON $OBJDIMS
RUN_EXIT=$?
check "$RUN_EXIT" "phase 2 video pick"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
echo; echo "===== PICK VERDICT + KEY LINES ====="
grep -E "VERDICT|\[CONTACT P4|\[CONTACT P3b|\[JAW SUMMARY\]|cube_z gain|real lift|held through|SUMMARY|cube_z P|\[PIN_PICK\]" \
  "$HOME/rl/logs/pin_pick_video.log" 2>/dev/null | tail -50 || true
echo "===================================="
echo "video files:"
ls -la "$HOME/rl/logs/videos/ar4_isaacsim_standalone_pick/friction/" 2>&1 || true

# --- extract representative frames for framing verification -----------------
step "extract frames for framing check"
sudo docker run --rm --entrypoint bash \
  -v "$HOME/rl:/workspace/rl" -w /workspace/rl "$IMAGE" -c '
/isaac-sim/python.sh - <<PYEOF
import imageio.v2 as imageio, os
vd="/workspace/rl/logs/videos/ar4_isaacsim_standalone_pick/friction"
fd="/workspace/rl/logs/pin_frames"; os.makedirs(fd, exist_ok=True)
for name in ("closeup","elbow"):
    p=os.path.join(vd,name+".mp4")
    if not os.path.exists(p):
        print("MISSING",p); continue
    r=imageio.get_reader(p); n=r.count_frames()
    print(name,"frames=",n)
    for frac in (0.05,0.25,0.45,0.65,0.85,0.97):
        i=min(n-1,int(frac*n))
        try:
            imageio.imwrite(os.path.join(fd,f"{name}_{int(frac*100):02d}.png"), r.get_data(i))
        except Exception as e:
            print("frame err",name,i,e)
print("FRAMES_EXTRACTED")
PYEOF' 2>&1 | tee "$HOME/rl/logs/frame_extract.log"
sudo chown -R "$(id -u):$(id -g)" "$HOME/rl/logs" 2>/dev/null || true
ls -la "$HOME/rl/logs/pin_frames/" 2>&1 || true

# --- sync artifacts to GCS (ALWAYS, BEFORE teardown) -----------------------
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-pin-pick/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"
gsutil -m cp -r "$HOME/rl/logs/videos" "${GCS_DEST}" 2>&1 || echo "WARNING: video GCS sync failed (non-fatal)"
gsutil -m cp -r "$HOME/rl/logs/pin_frames" "${GCS_DEST}" 2>&1 || echo "WARNING: frames GCS sync failed (non-fatal)"
gsutil -m cp \
  "$HOME/rl/logs/pin_aabb.log" \
  "$HOME/rl/logs"/pin_sweep_*.log \
  "$HOME/rl/logs/pin_pick_video.log" \
  "$HOME/rl/logs"/standalone_pick_result_friction.txt \
  "${GCS_DEST}" 2>&1 || echo "WARNING: log GCS sync failed (non-fatal)"
echo "GCS_DEST=${GCS_DEST}"
echo "PIN_PICK_PIPELINE_DONE best_top='${BEST_TOP}' video_top=${VIDEO_TOP} video_exit=${RUN_EXIT} gcs=${GCS_DEST}"

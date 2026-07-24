#!/usr/bin/env bash
# One-shot cloud setup + run for the ar4-closeup-grasp-video task
# (2026-07-24). Builds Isaac Sim/Isaac Lab + a fresh AR4 USD asset from
# scratch (desktop unreachable, no persisted asset/GCS mirror exists), then
# runs scripts/grasp_demo_v2.py at the best-known grasp configuration this
# investigation has found (65deg tilt, reach=0.36m - see kb/wiki/concepts/
# ar4-vs-franka-root-cause-comparison.md's 2026-07-23 ar4-capstone-grasp
# UPDATE) WITH the new --closeup-camera flag (this task's own addition to
# grasp_demo_v2.py) so a genuinely close-up video of the gripper+cube
# contact is recorded throughout the phased approach/close/lift/hold
# sequence - neither of the existing perception_camera/demo_camera
# resolves the 12mm cube clearly enough at this configuration (2026-07-24
# ar4-jaw-contact-sensor-hypothesis finding).
#
# Steps 1-5 (system packages, isaac-venv/IsaacLab, AR4 vendor description +
# xacro + ament shim, asset build, asset verification) are the identical,
# already-proven recipe from scripts/_cloud_ar4_convergence_tightening.sh
# (itself reused from two earlier tasks) - copied verbatim rather than
# re-derived, per this project's own standing "reuse the proven recipe"
# convention. Only step 6 (the actual run) differs.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance and runs a command
# there with ~/rl as the working directory.
#
# Deliberately does NOT use `set -e` (see _cloud_ar4_convergence_tightening.sh's
# own comment on why - a single incidental non-zero exit mid-pipeline must
# not silently abort the whole ~30-45min setup before the actual video-
# capture step runs). Every step below instead checks its own exit code
# explicitly, logs PASS/FAIL, and keeps going.
set -u

step() {
  echo "=== $1 ==="
}

check() {
  # check <exit_code> <description> - logs pass/fail, never aborts.
  if [ "$1" -eq 0 ]; then
    echo "[OK] $2"
  else
    echo "[FAIL exit=$1] $2 - continuing anyway"
  fi
}

step "[1/6] system packages"
sudo apt-get update -y
sudo apt-get install -y libgl1 libglx-mesa0 libegl1 libnvidia-gl-580-server \
    vulkan-tools libglu1-mesa libxt6 tmux cmake build-essential \
    software-properties-common git
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
check $? "system packages installed"

step "[2/6] isaac-venv + Isaac Sim 5.1.0 + IsaacLab v2.3.1"
python3.11 -m venv "$HOME/isaac-venv"
# shellcheck disable=SC1091
source "$HOME/isaac-venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
check $? "isaacsim[all,extscache]==5.1.0 pip install"
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
check $? "torch/torchvision pinned versions"

rm -rf "$HOME/IsaacLab"
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.1 "$HOME/IsaacLab"
check $? "IsaacLab v2.3.1 clone"
cd "$HOME/IsaacLab"
pip install --no-build-isolation flatdict==4.0.1
check $? "flatdict build-isolation workaround"
./isaaclab.sh --install rsl_rl
check $? "isaaclab.sh --install rsl_rl"
if python -c "import isaaclab" 2>&1 | grep -q "No module named 'isaaclab'"; then
  echo "[FAIL] base isaaclab package missing after install - continuing anyway, later steps will fail fast and visibly if this is fatal"
else
  echo "[OK] isaaclab base package importable"
fi

step "[3/6] AR4 vendor description (public GitHub mirror) + xacro + ament_index_python shim"
pip install xacro==2.1.1
check $? "xacro==2.1.1 pip install"
rm -rf "$HOME/ar4_ros_driver"
git clone https://github.com/Annin-Robotics/ar4_ros_driver.git "$HOME/ar4_ros_driver"
check $? "ar4_ros_driver clone"
export AR4_DESCRIPTION_PATH="$HOME/ar4_ros_driver/annin_ar4_description"
if [ ! -d "$AR4_DESCRIPTION_PATH/urdf" ]; then
  echo "[FAIL] expected $AR4_DESCRIPTION_PATH/urdf not found after clone - vendor repo layout may have changed - continuing anyway"
else
  echo "[OK] AR4_DESCRIPTION_PATH/urdf found"
fi

mkdir -p "$HOME/ament_shim/ament_index_python"
touch "$HOME/ament_shim/ament_index_python/__init__.py"
cat > "$HOME/ament_shim/ament_index_python/packages.py" <<'PYEOF'
import os


class PackageNotFoundError(Exception):
    pass


_KNOWN = {
    "annin_ar4_description": os.environ.get("AR4_DESCRIPTION_PATH", ""),
}


def get_package_share_directory(package_name):
    path = _KNOWN.get(package_name, "")
    if not path or not os.path.isdir(path):
        raise PackageNotFoundError(f"package '{package_name}' not found (ament_index_python shim)")
    return path
PYEOF
export PYTHONPATH="$HOME/ament_shim:${PYTHONPATH:-}"

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-closeup-grasp-video/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"

step "[4/6] build AR4 USD asset"
cd "$HOME/rl"
yes | PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/build_asset.py 2>&1 | tee "$HOME/build_asset.log"
BUILD_ASSET_EXIT="${PIPESTATUS[1]}"
check "$BUILD_ASSET_EXIT" "build_asset.py"
gsutil -q cp "$HOME/build_asset.log" "${GCS_DEST}logs/build_asset.log" 2>&1 || echo "WARNING: GCS sync of build_asset.log failed (non-fatal)"

step "[5/6] verify the built asset actually carries every known fix"
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_asset_jaw_fixes.py 2>&1 | tee "$HOME/verify_asset.log"
check "${PIPESTATUS[0]}" "asset verification (see PASS/FAIL lines above for which specific checks)"
gsutil -q cp "$HOME/verify_asset.log" "${GCS_DEST}logs/verify_asset.log" 2>&1 || echo "WARNING: GCS sync of verify_asset.log failed (non-fatal)"

step "[6/6] run grasp_demo_v2.py at reach=0.36m/tilt=65deg with --closeup-camera"
# reach=0.36m, tilt=65deg: the best-recorded grasp configuration this
# investigation has found (jaw2=0.027N sustained contact - see kb doc's
# 2026-07-24 ar4-jaw-contact-sensor-hypothesis UPDATE). --closeup-camera
# is this task's own new flag; --closeup-standoff/--closeup-z-lift/
# --closeup-focal-length left at their defaults for this first attempt
# (0.15m world+X offset, 0.05m Z lift, 40mm focal length) - see
# grasp_demo_v2.py's own flag docstrings for the geometry reasoning.
# --headless per the standing cloud-runs-headless exception.
PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/grasp_demo_v2.py --headless \
    --cube-xy 0 0.36 --tilt-deg 65 --grasp-height 0.009 \
    --closeup-camera \
    --video-suffix closeup_r036_t65 \
    2>&1 | tee "$HOME/grasp_closeup_run.log"
check "${PIPESTATUS[0]}" "grasp_demo_v2.py closeup-camera run"
gsutil -q cp "$HOME/grasp_closeup_run.log" "${GCS_DEST}logs/grasp_closeup_run.log" 2>&1 || echo "WARNING: GCS sync of grasp_closeup_run.log failed (non-fatal)"

echo "=== best-effort GCS video sync (non-fatal) ==="
gsutil -m cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal - the numeric diagnostic log was already synced above)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

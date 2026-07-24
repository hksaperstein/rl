#!/usr/bin/env bash
# One-shot cloud setup + run for the ar4-arm-chain-fk-check task
# (2026-07-24). Builds Isaac Sim/Isaac Lab + a fresh AR4 USD asset from
# scratch (desktop unreachable, no persisted asset/GCS mirror exists),
# then runs the new scripts/_verify_arm_chain_fk_integration.py - a
# full-chain (link_1..link_6) extension of the existing gripper-only
# FK integration check - to directly test whether the built asset's own
# ARM kinematic chain (as distinct from the gripper, already verified
# correct in prior sessions) matches the vendor's own raw URDF/xacro FK
# prediction, at HOME_Q, the best-known converged GRASP_Q/PREGRASP_Q, and
# a synthetic all-joints-nontrivial stress config.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, reusing the
# proven recipe from scripts/_cloud_ar4_convergence_tightening.sh (steps
# 1-5 identical - see that script's own comments for why each step is
# structured this way) with step 6 replaced by this task's own new check.
#
# Deliberately does NOT use `set -e` - see
# scripts/_cloud_ar4_convergence_tightening.sh's own comment for why
# (a single incidental non-zero exit mid-pipeline must not silently abort
# the rest of this script before the actual diagnostic step runs).
set -u

step() {
  echo "=== $1 ==="
}

check() {
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

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-arm-chain-fk-check/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"

step "[4/6] build AR4 USD asset"
cd "$HOME/rl"
yes | PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/build_asset.py 2>&1 | tee "$HOME/build_asset.log"
BUILD_ASSET_EXIT="${PIPESTATUS[1]}"
check "$BUILD_ASSET_EXIT" "build_asset.py"
gsutil -q cp "$HOME/build_asset.log" "${GCS_DEST}logs/build_asset.log" 2>&1 || echo "WARNING: GCS sync of build_asset.log failed (non-fatal)"

step "[5/6] verify the built asset actually carries every known gripper fix (sanity check before trusting the arm-chain result)"
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_asset_jaw_fixes.py 2>&1 | tee "$HOME/verify_asset.log"
check "${PIPESTATUS[0]}" "asset verification (see PASS/FAIL lines above for which specific checks)"
gsutil -q cp "$HOME/verify_asset.log" "${GCS_DEST}logs/verify_asset.log" 2>&1 || echo "WARNING: GCS sync of verify_asset.log failed (non-fatal)"

step "[6/6] full arm-chain (link_1..link_6) FK verification vs. vendor URDF, at HOME_Q/PREGRASP_Q/GRASP_Q/STRESS_Q"
# --headless per the standing cloud-runs-headless exception (CLAUDE.md's
# local "never headless" rule is a LOCAL-display convention, does not
# apply to cloud dispatch).
PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_arm_chain_fk_integration.py --headless \
    2>&1 | tee "$HOME/arm_chain_fk_check.log"
check "${PIPESTATUS[0]}" "arm-chain FK verification run"
gsutil -q cp "$HOME/arm_chain_fk_check.log" "${GCS_DEST}logs/arm_chain_fk_check.log" 2>&1 || echo "WARNING: GCS sync of arm_chain_fk_check.log failed (non-fatal)"

echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

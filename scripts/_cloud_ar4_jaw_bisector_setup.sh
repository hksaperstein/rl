#!/usr/bin/env bash
# One-shot cloud setup + run for the ar4-jaw-bisector-hypothesis task
# (2026-07-23). Builds Isaac Sim/Isaac Lab + a fresh AR4 USD asset from
# scratch (desktop unreachable, no persisted asset/GCS mirror exists), then
# runs the instrumented scripts/grasp_demo_v2.py at the best-known
# kinematic configuration this investigation has found (65deg tilt,
# reach=0.30m, grasp-height=0.009m - see kb/wiki/concepts/
# ar4-vs-franka-root-cause-comparison.md's 2026-07-23 ar4-capstone-grasp
# UPDATE) to directly measure the real jaw-fingertip bisector against the
# arm's own _EE_OFFSET assumption and the cube's true position.
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance and runs a command
# there with ~/rl as the working directory. Base Isaac Sim/Isaac Lab install
# steps are the proven recipe from docs/cloud/franka-cloud-shakedown.md;
# the AR4-on-cloud asset-build steps (public vendor mirror, xacro,
# hand-rolled ament_index_python shim) are the recipe from
# kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-23
# "Standing FK verification framework added" UPDATE ("Live integration
# check - cloud, not desktop" section) and the ar4-capstone-grasp task's
# own "Setup" section - reconstructed here as a real, checked-in script
# instead of re-derived from memory a fourth time (a real, previously-
# flagged gap - "Not yet built: a pre-baked VM image ...").
set -Eeuo pipefail

echo "=== [1/6] system packages ==="
sudo apt-get update -y
sudo apt-get install -y libgl1 libglx-mesa0 libegl1 libnvidia-gl-580-server \
    vulkan-tools libglu1-mesa libxt6 tmux cmake build-essential \
    software-properties-common git
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

echo "=== [2/6] isaac-venv + Isaac Sim 5.1.0 + IsaacLab v2.3.1 ==="
python3.11 -m venv "$HOME/isaac-venv"
# shellcheck disable=SC1091
source "$HOME/isaac-venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.1 "$HOME/IsaacLab"
cd "$HOME/IsaacLab"
pip install --no-build-isolation flatdict==4.0.1
./isaaclab.sh --install rsl_rl
python -c "import isaaclab" 2>&1 | grep -q "No module named 'isaaclab'" && \
    { echo "FATAL: base isaaclab package missing after install"; exit 1; } || \
    echo "isaaclab base package OK"

echo "=== [3/6] AR4 vendor description (public GitHub mirror) + xacro + ament_index_python shim ==="
pip install xacro==2.1.1
git clone https://github.com/Annin-Robotics/ar4_ros_driver.git "$HOME/ar4_ros_driver"
export AR4_DESCRIPTION_PATH="$HOME/ar4_ros_driver/annin_ar4_description"
if [ ! -d "$AR4_DESCRIPTION_PATH/urdf" ]; then
  echo "FATAL: expected $AR4_DESCRIPTION_PATH/urdf not found after clone - vendor repo layout may have changed"
  exit 1
fi

mkdir -p "$HOME/ament_shim/ament_index_python"
touch "$HOME/ament_shim/ament_index_python/__init__.py"
cat > "$HOME/ament_shim/ament_index_python/packages.py" <<'PYEOF'
import os


class PackageNotFoundError(Exception):
    pass


# Minimal from-scratch reimplementation of the one function build_asset.py's
# xacro invocation actually needs ($(find annin_ar4_description) resolution)
# - ament_index_python is a ROS2 package, not published to PyPI, so a full
# install would need colcon/a ROS2 distro. Resolves purely via an env var
# instead of the real ROS2 marker-file/AMENT_PREFIX_PATH mechanism.
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

echo "=== [4/6] build AR4 USD asset ==="
cd "$HOME/rl"
# `yes` feeds an endless "y\n" to stdin - a generic mitigation for the
# isaacsim.asset.importer.urdf extension's own interactive EULA prompt,
# confirmed to occur a SECOND time (separate from the isaacsim pip-install
# EULA already accepted via OMNI_KIT_ACCEPT_EULA above) in the 2026-07-23
# ar4-capstone-grasp task's own cloud build, silently hanging the session
# on stdin until manually answered - no committed suppression flag is
# known for this second prompt.
# `yes`'s own SIGPIPE (exit 141) the instant build_asset.py stops reading
# stdin is EXPECTED and harmless (it only exists to answer a possible
# interactive EULA prompt, in case one fires) - but with `set -e -o
# pipefail` (top of this script), bash's pipefail reports THAT as the
# pipeline's own exit status whenever no other stage in the pipe is
# nonzero, which trips `set -e` and silently aborts the rest of this
# script (steps 5/6 below) even though the actual build succeeded. Found
# live (2026-07-23) - the first cloud run of this script built the asset
# correctly (confirmed via the raw remote log) but exited 141 right after,
# never reaching the verify/diagnostic steps. Fix: disable pipefail only
# for this one pipeline, read build_asset.py's OWN real exit code via
# PIPESTATUS (index 1: yes | <build_asset.py> | tee), and fail explicitly
# only on THAT.
set +o pipefail
yes | PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/build_asset.py 2>&1 | tee "$HOME/build_asset.log"
BUILD_ASSET_EXIT="${PIPESTATUS[1]}"
set -o pipefail
if [ "$BUILD_ASSET_EXIT" -ne 0 ]; then
  echo "FATAL: build_asset.py itself exited ${BUILD_ASSET_EXIT} (not the harmless 'yes' SIGPIPE - see comment above)"
  exit 1
fi
echo "build_asset.py exited 0 - proceeding to verify."

echo "=== [5/6] verify the built asset actually carries every known fix ==="
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_asset_jaw_fixes.py 2>&1 | tee "$HOME/verify_asset.log"

echo "=== [6/6] run the instrumented grasp_demo_v2.py bisector diagnostic ==="
# reach=0.30m, tilt=65deg, grasp-height=0.009m: the best-known kinematic
# configuration this investigation has found (2026-07-23 ar4-capstone-grasp
# task - 9.5mm position residual, healthy joint margins, one of the 3
# full phased grasp+lift attempts run there). --headless per the standing
# cloud-runs-headless exception (CLAUDE.md's local "never headless" rule
# is a LOCAL-display convention, does not apply to cloud dispatch).
PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/grasp_demo_v2.py --headless \
    --cube-xy 0 0.30 --tilt-deg 65 --grasp-height 0.009 --video-suffix jawbisector_r030_t65 \
    2>&1 | tee "$HOME/grasp_bisector_run.log"

echo "=== DONE: best-effort GCS video sync (non-fatal - the numeric diagnostic itself is already fully captured in the streamed log above) ==="
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-jaw-bisector-check/$(date -u +%Y%m%d-%H%M%S)/"
gsutil -m cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal - see note above)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

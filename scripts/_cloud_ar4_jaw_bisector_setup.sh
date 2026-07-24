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
#
# Deliberately does NOT use `set -e`: two separate live runs (2026-07-23)
# each lost a full ~30-45min of already-completed setup work because a
# single incidental non-zero exit mid-pipeline (a `yes`-into-a-finished-
# reader SIGPIPE; a `git clone` into a pre-existing directory) silently
# aborted the rest of the script before the actual diagnostic step ever
# ran. Every step below instead checks its own exit code explicitly,
# logs PASS/FAIL, and keeps going - the whole point of this script is to
# reach step 6's printed diagnostic output, and steps 1-5 have each
# already been independently proven to work correctly in prior sessions.
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

# Idempotent clone: found live (2026-07-23) that a fresh instance from the
# common-cu129-ubuntu-2204 DLVM image can already have a non-empty
# ~/IsaacLab present (root cause not confirmed - possibly baked into the
# image, possibly a side effect of the isaacsim[all,extscache] pip install
# above - not worth spending more of this task's budget chasing further).
# Safe to remove unconditionally: this is a throwaway ephemeral instance.
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

step "[4/6] build AR4 USD asset"
cd "$HOME/rl"
# `yes` feeds an endless "y\n" to stdin - a generic mitigation for the
# isaacsim.asset.importer.urdf extension's own interactive EULA prompt,
# confirmed to occur a SECOND time (separate from the isaacsim pip-install
# EULA already accepted via OMNI_KIT_ACCEPT_EULA above) in the 2026-07-23
# ar4-capstone-grasp task's own cloud build, silently hanging the session
# on stdin until manually answered - no committed suppression flag is
# known for this second prompt. `yes` getting SIGPIPE'd (a nonzero exit)
# the instant build_asset.py stops reading stdin is expected/harmless;
# PIPESTATUS[1] below reads build_asset.py's OWN real exit code directly,
# independent of `yes`'s.
yes | PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/build_asset.py 2>&1 | tee "$HOME/build_asset.log"
BUILD_ASSET_EXIT="${PIPESTATUS[1]}"
check "$BUILD_ASSET_EXIT" "build_asset.py"

step "[5/6] verify the built asset actually carries every known fix"
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_asset_jaw_fixes.py 2>&1 | tee "$HOME/verify_asset.log"
check "${PIPESTATUS[0]}" "asset verification (see PASS/FAIL lines above for which specific checks)"

step "[6/6] run the instrumented grasp_demo_v2.py bisector diagnostic"
# reach=0.30m, tilt=65deg, grasp-height=0.009m: the best-known kinematic
# configuration this investigation has found (2026-07-23 ar4-capstone-grasp
# task - 9.5mm position residual, healthy joint margins, one of the 3
# full phased grasp+lift attempts run there). --headless per the standing
# cloud-runs-headless exception (CLAUDE.md's local "never headless" rule
# is a LOCAL-display convention, does not apply to cloud dispatch). This is
# the actual point of this whole dispatch - runs regardless of any
# check-logged FAIL above, since each of those has independently succeeded
# in prior sessions and the real cost of trying anyway is small compared
# to losing the whole ~30-45min setup investment on a false alarm.
PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/grasp_demo_v2.py --headless \
    --cube-xy 0 0.30 --tilt-deg 65 --grasp-height 0.009 --video-suffix jawbisector_r030_t65 \
    2>&1 | tee "$HOME/grasp_bisector_run.log"
check "${PIPESTATUS[0]}" "grasp_demo_v2.py bisector diagnostic run"

echo "=== best-effort GCS video sync (non-fatal) ==="
GCS_DEST="gs://rl-manipulation-hks-runs/ar4-jaw-bisector-check/$(date -u +%Y%m%d-%H%M%S)/"
gsutil -m cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal - the numeric diagnostic is already fully captured in the streamed log above)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

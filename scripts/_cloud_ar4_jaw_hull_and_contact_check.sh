#!/usr/bin/env bash
# One-shot cloud setup + run for the ar4-jaw-contact-sensor-hypothesis task
# (2026-07-24). Builds Isaac Sim/Isaac Lab + a fresh AR4 USD asset from
# scratch (desktop unreachable, no persisted asset/GCS mirror exists), then
# runs two never-yet-checked diagnostics against the standing jaw1-brief-
# contact/jaw2-exactly-0.0N signature found at the best-known 65deg-tilt
# grasp configuration:
#
#   Hypothesis A: scripts/_inspect_jaw_convex_hull.py - is jaw1's and
#   jaw2's own collision mesh geometry actually symmetric (raw point/face
#   counts, convex-hull vertex/face/volume/area), or does a real asymmetry
#   exist that the schema-level "both jaws have convexHull" check (done
#   2026-07-21) never quantified?
#
#   Hypothesis B: scripts/grasp_demo_v2.py --capture-close-phase-frames at
#   reach=0.30m/tilt=65deg (the exact configuration that produced a brief
#   jaw1=0.34N / jaw2=0.0000N reading in the 2026-07-23 ar4-capstone-grasp
#   task) - saves a demo-camera still frame at EVERY step of Phase 3
#   (CLOSE), so the exact frame matching a nonzero jaw1 contact print can
#   be pulled and jaw2's fingertip visually checked against the cube at
#   that same instant (overlapping vs. genuinely separated). Also compares
#   the two ContactSensorCfg definitions in
#   tasks/ar4/pickplace_graspgoal_env_cfg.py directly (already done
#   statically, no cloud needed - both jaw sensors are structurally
#   identical apart from prim_path, same update_period/history_length/
#   filter_prim_paths_expr - no config-level asymmetry found there).
#
# Meant to be dispatched via scripts/run_on_cloud_gpu.sh, which ships this
# repo's committed HEAD to ~/rl on a fresh GCP instance and runs a command
# there with ~/rl as the working directory. Recipe steps 1-5 (system
# packages, Isaac Sim/Lab install, AR4 vendor description + xacro shim,
# asset build, asset verification) are the exact proven recipe reused
# verbatim from scripts/_cloud_ar4_convergence_tightening.sh (itself reused
# from two earlier sessions) - not re-derived from memory.
#
# Deliberately does NOT use `set -e` - see _cloud_ar4_convergence_tightening.sh's
# own header comment for why (a single incidental non-zero exit mid-pipeline
# has silently aborted this exact recipe before reaching its own real
# diagnostic step, twice, in two separate past sessions).
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

step "[1/7] system packages"
sudo apt-get update -y
sudo apt-get install -y libgl1 libglx-mesa0 libegl1 libnvidia-gl-580-server \
    vulkan-tools libglu1-mesa libxt6 tmux cmake build-essential \
    software-properties-common git
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
check $? "system packages installed"

step "[2/7] isaac-venv + Isaac Sim 5.1.0 + IsaacLab v2.3.1 + scipy"
python3.11 -m venv "$HOME/isaac-venv"
# shellcheck disable=SC1091
source "$HOME/isaac-venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
check $? "isaacsim[all,extscache]==5.1.0 pip install"
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
check $? "torch/torchvision pinned versions"
pip install scipy
check $? "scipy pip install (needed for scripts/_inspect_jaw_convex_hull.py's ConvexHull)"

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

step "[3/7] AR4 vendor description (public GitHub mirror) + xacro + ament_index_python shim"
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

GCS_DEST="gs://rl-manipulation-hks-runs/ar4-jaw-contact-sensor-hypothesis/$(date -u +%Y%m%d-%H%M%S)/"
echo "GCS_DEST=${GCS_DEST}"

step "[4/7] build AR4 USD asset"
cd "$HOME/rl"
yes | PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/build_asset.py 2>&1 | tee "$HOME/build_asset.log"
BUILD_ASSET_EXIT="${PIPESTATUS[1]}"
check "$BUILD_ASSET_EXIT" "build_asset.py"
gsutil -q cp "$HOME/build_asset.log" "${GCS_DEST}logs/build_asset.log" 2>&1 || echo "WARNING: GCS sync of build_asset.log failed (non-fatal)"

step "[5/7] verify the built asset actually carries every known fix"
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_verify_asset_jaw_fixes.py 2>&1 | tee "$HOME/verify_asset.log"
check "${PIPESTATUS[0]}" "asset verification (see PASS/FAIL lines above for which specific checks)"
gsutil -q cp "$HOME/verify_asset.log" "${GCS_DEST}logs/verify_asset.log" 2>&1 || echo "WARNING: GCS sync of verify_asset.log failed (non-fatal)"

step "[6/7] Hypothesis A - jaw1 vs jaw2 collision mesh/convex-hull comparison"
"$HOME/IsaacLab/isaaclab.sh" -p scripts/_inspect_jaw_convex_hull.py 2>&1 | tee "$HOME/jaw_hull_check.log"
check "${PIPESTATUS[0]}" "_inspect_jaw_convex_hull.py"
gsutil -q cp "$HOME/jaw_hull_check.log" "${GCS_DEST}logs/jaw_hull_check.log" 2>&1 || echo "WARNING: GCS sync of jaw_hull_check.log failed (non-fatal)"

step "[7/7] Hypothesis B - grasp_demo_v2.py at reach=0.30m/tilt=65deg with per-step Phase-3 frame capture"
# reach=0.30m/tilt=65deg: the specific configuration the 2026-07-23
# ar4-capstone-grasp task recorded a brief jaw1_cube_force=0.34N /
# jaw2_cube_force=0.0000N reading at (its OWN best/clearest one-sided-
# contact case among the three reach points tested at this tilt).
# --capture-close-phase-frames (this task's own new flag) saves a demo-
# camera still + prints both jaw forces at EVERY step of Phase 3 (CLOSE),
# not just every 20th - lets the exact frame matching a nonzero jaw1
# reading be pulled directly for a real visual jaw2-vs-cube overlap check.
# --headless per the standing cloud-runs-headless exception.
PYTHONUNBUFFERED=1 "$HOME/IsaacLab/isaaclab.sh" -p scripts/grasp_demo_v2.py --headless \
    --cube-xy 0 0.30 --tilt-deg 65 --grasp-height 0.009 \
    --capture-close-phase-frames \
    --video-suffix jawcontact_r030_t65 \
    2>&1 | tee "$HOME/grasp_jawcontact_run.log"
check "${PIPESTATUS[0]}" "grasp_demo_v2.py jaw-contact-hypothesis run"
gsutil -q cp "$HOME/grasp_jawcontact_run.log" "${GCS_DEST}logs/grasp_jawcontact_run.log" 2>&1 || echo "WARNING: GCS sync of grasp_jawcontact_run.log failed (non-fatal)"

echo "=== best-effort GCS video+snapshot sync (non-fatal) ==="
gsutil -m cp -r "$HOME/rl/logs/videos" "$GCS_DEST" 2>&1 || echo "WARNING: GCS video sync failed (non-fatal - the numeric diagnostic logs were already synced above)"
echo "GCS_DEST=${GCS_DEST}"
echo "ALL DONE."

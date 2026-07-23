"""One-off direct USD inspection: confirm a freshly-built AR4 asset actually
carries every previously-found gripper/collision fix (jaw2 mimic removed,
jaw2 has its own DriveAPI, jaw2 hard limits match jaw1, link_5/link_6 have a
substitute collision box) - not just "the build exited 0".

2026-07-23 (ar4-jaw-bisector-hypothesis task): a from-scratch cloud AR4
build has no prior verified state (this project's history includes a real
incident, 2026-07-23 ar4-capstone-grasp task, where build_asset.py's own
print() confirmations never appeared in the captured log at all despite the
build succeeding - SimulationApp.close() apparently force-exits in a way
that can skip stdout's normal flush). This script is the same
independent-verification pattern that capstone session used (mirrors
scripts/_inspect_jaw_symmetry.py's own SimulationApp(headless) bootstrap,
which build_asset.py's own main() shows is REQUIRED before `pxr` becomes
importable at all under isaaclab.sh -p).

Run via: /path/to/isaaclab.sh -p scripts/_verify_asset_jaw_fixes.py
Exits 0 if every check passes, 1 otherwise.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

print(f"[VERIFY] Opening USD: {usd_path}")
stage = Usd.Stage.Open(usd_path)

all_ok = True


def check(desc: str, cond: bool) -> None:
    global all_ok
    print(f"[VERIFY] {desc}: {'PASS' if cond else 'FAIL'}")
    all_ok = all_ok and cond


jaw1 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw1_joint")
jaw2 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw2_joint")
check("gripper_jaw1_joint prim valid", jaw1.IsValid())
check("gripper_jaw2_joint prim valid", jaw2.IsValid())

mimic = PhysxSchema.PhysxMimicJointAPI.Get(jaw2, "rotX") if jaw2.IsValid() else None
check("gripper_jaw2_joint has NO PhysxMimicJointAPI (removed per 2576e94)", not bool(mimic))

jaw2_drive = UsdPhysics.DriveAPI.Get(jaw2, "linear") if jaw2.IsValid() else None
check("gripper_jaw2_joint has its own DriveAPI:linear (per _add_gripper_jaw2_drive)", bool(jaw2_drive))

if jaw1.IsValid() and jaw2.IsValid():
    lo1 = jaw1.GetAttribute("physics:lowerLimit").Get()
    hi1 = jaw1.GetAttribute("physics:upperLimit").Get()
    lo2 = jaw2.GetAttribute("physics:lowerLimit").Get()
    hi2 = jaw2.GetAttribute("physics:upperLimit").Get()
    print(f"[VERIFY] jaw1 limits=[{lo1}, {hi1}]  jaw2 limits=[{lo2}, {hi2}]")
    check(
        "gripper_jaw2_joint hard limits match gripper_jaw1_joint's own (same-sign convention)",
        lo1 is not None and hi1 is not None and lo2 is not None and hi2 is not None
        and abs(lo1 - lo2) < 1e-6 and abs(hi1 - hi2) < 1e-6,
    )

for link_name in ("link_5", "link_6"):
    box = stage.GetPrimAtPath(f"/mk5/root_joint/{link_name}/substitute_collision_box")
    check(f"{link_name} has substitute_collision_box (missing upstream STL fix)", box.IsValid())

print(f"[VERIFY] OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
simulation_app.close()
sys.exit(0 if all_ok else 1)

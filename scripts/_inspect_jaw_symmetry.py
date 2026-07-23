"""One-off direct USD inspection: is gripper_jaw2_joint's own origin genuinely
the mirror image of gripper_jaw1_joint's about the gripper's true center
plane, or is there a real asymmetric offset between the two joints'
underlying frames (independent of the already-confirmed-correct +-0.014
commanded JOINT VALUES)?

Coordinator-directed (2026-07-23), following a live visual observation that
the two jaws don't look symmetric about a shared centerline even though the
commanded joint values are confirmed correctly opposite (+0.014 / -0.014).
Mirrors the existing docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md
direct-pxr-inspection pattern (same stage-open, GetAttribute-read style
already used by scripts/build_asset.py's _remove_gripper_jaw2_mimic_constraint
/ _add_gripper_jaw2_drive) - including that script's own SimulationApp(headless)
bootstrap, which build_asset.py's own main() shows is REQUIRED before `pxr`
becomes importable at all under isaaclab.sh -p (confirmed live this session -
a plain `from pxr import Usd` with no SimulationApp first raises
ModuleNotFoundError).

Run via: /home/saps/IsaacLab/isaaclab.sh -p scripts/_inspect_jaw_symmetry.py
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

print(f"[INFO] Opening USD: {usd_path}")
stage = Usd.Stage.Open(usd_path)

jaw1 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw1_joint")
jaw2 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw2_joint")
print(f"[INFO] jaw1 valid={jaw1.IsValid()} jaw2 valid={jaw2.IsValid()}")

for name, prim in [("gripper_jaw1_joint", jaw1), ("gripper_jaw2_joint", jaw2)]:
    print(f"\n=== {name} ===")
    joint_api = UsdPhysics.Joint(prim)
    body0_rel = joint_api.GetBody0Rel().GetTargets()
    body1_rel = joint_api.GetBody1Rel().GetTargets()
    print(f"  body0={body0_rel} body1={body1_rel}")
    local_pos0 = joint_api.GetLocalPos0Attr().Get()
    local_pos1 = joint_api.GetLocalPos1Attr().Get()
    local_rot0 = joint_api.GetLocalRot0Attr().Get()
    local_rot1 = joint_api.GetLocalRot1Attr().Get()
    prismatic_api = UsdPhysics.PrismaticJoint(prim)
    axis = prismatic_api.GetAxisAttr().Get() if prismatic_api else None
    print(f"  axis={axis}")
    print(f"  localPos0 (in body0/parent frame) = {local_pos0}")
    print(f"  localRot0 (in body0/parent frame) = {local_rot0}")
    print(f"  localPos1 (in body1/child frame)  = {local_pos1}")
    print(f"  localRot1 (in body1/child frame)  = {local_rot1}")
    lower = prim.GetAttribute("physics:lowerLimit").Get()
    upper = prim.GetAttribute("physics:upperLimit").Get()
    print(f"  lowerLimit={lower} upperLimit={upper}")

# Independent geometry-level cross-check: each jaw LINK's actual rest-pose
# world transform (from the USD's own authored default/rest pose, before any
# runtime actuation), via UsdGeom.XformCache - so this isn't relying solely on
# the joint-frame attributes above.
xf_cache = UsdGeom.XformCache()
for link_name in ["gripper_jaw1_link", "gripper_jaw2_link", "link_6", "gripper_base_link"]:
    found = None
    for prim in stage.Traverse():
        if str(prim.GetPath()).endswith(link_name):
            found = prim
            break
    if found is None:
        print(f"\n[WARN] Could not find prim ending in {link_name}")
        continue
    world_xf = xf_cache.GetLocalToWorldTransform(found)
    translation = world_xf.ExtractTranslation()
    print(f"\n=== {link_name} (path={found.GetPath()}) rest-pose world translation ===")
    print(f"  {translation}")

simulation_app.close()

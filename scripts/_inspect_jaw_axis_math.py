"""Follow-up to _inspect_jaw_symmetry.py / _inspect_jaw_symmetry_live.py: the
live check found gripper_jaw1_link and gripper_jaw2_link landing at the
EXACT SAME world point when driven to the nominal OPEN command
(+0.014/-0.014) - i.e. the two joints' commanded values are correctly
opposite in SIGN, but the physical result is NOT mirrored, it's coincident.
This computes, directly from the USD's own authored localRot0 quaternions,
what each joint's OWN local +axis direction actually is once expressed in
the shared parent (gripper_base_link) frame - the concrete arithmetic that
explains (or refutes) the observed bug.

Run: /home/saps/IsaacLab/isaaclab.sh -p scripts/_inspect_jaw_axis_math.py
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(usd_path)

for name in ["gripper_jaw1_joint", "gripper_jaw2_joint"]:
    prim = stage.GetPrimAtPath(f"/mk5/root_joint/joints/{name}")
    joint_api = UsdPhysics.Joint(prim)
    prismatic_api = UsdPhysics.PrismaticJoint(prim)
    axis_str = prismatic_api.GetAxisAttr().Get()
    local_rot0 = joint_api.GetLocalRot0Attr().Get()
    local_pos0 = joint_api.GetLocalPos0Attr().Get()
    lower = prim.GetAttribute("physics:lowerLimit").Get()
    upper = prim.GetAttribute("physics:upperLimit").Get()

    axis_local = {"X": Gf.Vec3f(1, 0, 0), "Y": Gf.Vec3f(0, 1, 0), "Z": Gf.Vec3f(0, 0, 1)}[axis_str]
    # local_rot0 is a Gf.Quatf (or Quatd) - rotate axis_local by it to get the
    # joint's own positive-travel direction EXPRESSED IN THE PARENT (body0 =
    # gripper_base_link) FRAME. This is the actual physical direction a
    # positive joint-position command moves body1 (the jaw link) along, in
    # the frame shared by both joints - the concrete question this script
    # answers.
    rot_matrix = Gf.Matrix3f(local_rot0)
    axis_in_parent = rot_matrix * axis_local

    print(f"=== {name} ===")
    print(f"  axis (local) = {axis_str} -> {axis_local}")
    print(f"  localRot0 (raw quat) = {local_rot0}")
    print(f"  localRot0 real={local_rot0.GetReal()} imaginary={local_rot0.GetImaginary()}")
    print(f"  axis expressed in PARENT (gripper_base_link) frame = {axis_in_parent}")
    print(f"  localPos0 = {local_pos0}")
    print(f"  limits = [{lower}, {upper}]")
    print(
        f"  => at commanded joint value q, jaw link displaces by "
        f"q * {axis_in_parent} relative to localPos0, in the PARENT frame"
    )
    print()

simulation_app.close()

import re

path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar_gripper_macro.xacro"
with open(path) as f:
    src = f.read()

# NOTE (Gazebo physics-grasp task, 2026-07-28): the vendor URDF defines the
# gripper jaw links' visual/collision geometry but never sets Gazebo-specific
# surface friction on them (no <gazebo reference="..."> blocks at all in this
# file). Without this, the jaws pick up gz-sim's engine default friction,
# which is an unknown/unverified value for a real friction-based grasp test.
# This patch adds explicit mu1/mu2 = 0.8 (wood/plastic/resin range, matches
# this project's own realistic-friction convention) to both jaw links so the
# friction grasp attempt is against a known, deliberate parameter rather than
# an implicit engine default. Collision geometry itself (full STL mesh, not a
# primitive) is left untouched -- that's a separate, larger surgery out of
# this patch's scope if mesh-collision contact proves unstable.
#
# Honest result (recorded here, not just in the kb write-up): even with this
# friction applied, a live full pick test did NOT hold the cube by pure
# friction alone -- see this directory's README and
# kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md for the finding and
# the grasp-assist DetachableJoint fallback used instead.
insertion_point = "</xacro:macro>"
assert insertion_point in src, "expected closing </xacro:macro> tag not found"

friction_blocks = """
  <gazebo reference="${tf_prefix}gripper_jaw1_link">
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <material>Gazebo/Yellow</material>
  </gazebo>
  <gazebo reference="${tf_prefix}gripper_jaw2_link">
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <material>Gazebo/Yellow</material>
  </gazebo>

"""

assert "gripper_jaw1_link\">\n    <mu1>" not in src, "friction patch already applied"
src = src.replace(insertion_point, friction_blocks + insertion_point)

with open(path, "w") as f:
    f.write(src)
print("PATCHED ar_gripper_macro.xacro (jaw friction)")

"""Drop the vendor's --physics-engine gz-physics-bullet-featherstone-plugin
override from annin_ar4_gazebo/launch/gazebo.launch.py, falling back to
gz-sim's DEFAULT physics engine (DART) (ar4-gazebo-physics-pick task,
2026-07-28).

Root cause this fixes: with bullet-featherstone, EVERY launch attempt (many
repeats, including after upgrading ros-humble-controller-manager/
ros-humble-hardware-interface to their latest available apt builds) hit a
genuine SEGFAULT inside gz_ros2_control's own GazeboSimSystem::write() or
::read() (varying by run), on the controller_manager's very first update
cycle, in a background PostUpdate worker thread. Removing the
--physics-engine override (falling back to DART) made this crash disappear
entirely across every subsequent launch in this task's live testing.

Trade-off, reported honestly: bullet-featherstone is the engine
github.com/ros-controls/gz_ros2_control/issues/628 identifies as the ONLY
one with native support for a plain URDF <mimic> constraint -- switching to
DART risked breaking the gripper's jaw2-mirrors-jaw1 behavior. Live-tested
and NOT observed to break for this specific 1-leader/1-follower mimic case:
after removing gripper_jaw2_joint from the <ros2_control> block entirely
(patch_ros2_control_position_only_and_mimic_fix.py), tf2_echo of both
gripper_jaw{1,2}_link showed correctly mirrored open/close poses under
DART. This is empirically verified for THIS robot's gripper only, not a
general claim that DART supports <mimic> in every case.
"""

path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_gazebo/launch/gazebo.launch.py"
with open(path) as f:
    src = f.read()

old = "f'-r -v 4 --physics-engine gz-physics-bullet-featherstone-plugin {world}'"
new = "f'-r -v 4 {world}'"
assert old in src, "gz_args pattern not found"
src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)
print("PATCHED gazebo.launch.py: dropped bullet-featherstone override, using default DART engine")

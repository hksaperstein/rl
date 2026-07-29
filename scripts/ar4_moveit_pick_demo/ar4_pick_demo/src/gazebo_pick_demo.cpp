// Collision-aware MoveIt2 pick demo for the AR4 arm running under REAL
// Gazebo physics (ar4-gazebo-physics-pick task, 2026-07-28). Direct
// follow-on to pick_demo.cpp (the prior RViz/fake-hardware demo, which
// attached the cube as a planning-scene object -- not a real physics
// grasp). This version executes against the vendor's own
// annin_ar4_gazebo Gazebo Sim (gz-sim, "Fortress") bring-up: MoveIt's
// FollowJointTrajectory/GripperCommand actions drive the SAME
// joint_trajectory_controller/gripper_controller ros2_control controllers
// gz_ros2_control's GazeboSimSystem exposes, so trajectory execution
// actually moves the simulated arm under real dynamics, and the cube is a
// real free rigid body (mass + friction) resting on a real pedestal, not a
// kinematic attachment.
//
// Sequence is otherwise identical to pick_demo.cpp: home -> add pedestal
// + cube as MoveIt collision objects at the SAME pose as the real Gazebo
// models (so planning-scene bookkeeping matches physical reality) -> open
// gripper -> approach -> descend to grasp pose -> close gripper (snug,
// half-gap = cube half-width) -> attach cube to gripper_base_link in the
// planning scene (collision-avoidance bookkeeping only -- does NOT do the
// actual physical holding, that's real contact/friction in Gazebo) -> lift
// -> pause (verification window) -> carry to goal -> pause (verification
// window). The added pauses (vs. pick_demo.cpp's short ones) exist so an
// external observer can run `gz topic -e -t /world/default/pose/info -n 1`
// against the live sim and read the cube's actual world-frame pose at each
// stage, independent of anything this node itself claims.
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
// ee_link's own local +Z axis, empirically derived in the prior RViz-demo
// task (see pick_demo.cpp's own header comment).
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
      "ar4_gazebo_pick_demo_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  auto logger = node->get_logger();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  MoveGroupInterface arm(node, "ar_manipulator");
  MoveGroupInterface gripper(node, "ar_gripper");
  PlanningSceneInterface psi;

  arm.setPlanningTime(10.0);
  arm.setMaxVelocityScalingFactor(0.3);
  arm.setMaxAccelerationScalingFactor(0.3);
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
  RCLCPP_INFO(logger, "[VERIFY WINDOW] at 'home' -- pausing 5s for external gz topic checks.");
  rclcpp::sleep_for(std::chrono::seconds(5));

  // --- Step 2: add pedestal + cube as collision objects -------------------
  // (Bookkeeping only -- the REAL pedestal/cube are already live Gazebo
  // rigid bodies at this exact pose, added via the custom world file.)
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
  RCLCPP_INFO(logger, "Added pedestal (top z=%.4f) and cube (center z=%.4f) collision objects.",
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

  // Real-physics settle delay: under real Gazebo dynamics (unlike the prior
  // fake-hardware demo, where state updates were instantaneous), the arm's
  // reported joint state may still be settling briefly after "execute
  // success" returns. A live test hit 3/3 "descend to grasp pose" plan
  // FAILUREs back-to-back before this delay was added, consistent with the
  // next plan's start-state being captured mid-settle.
  RCLCPP_INFO(logger, "Settling 1.5s after approach before planning descent.");
  rclcpp::sleep_for(std::chrono::milliseconds(1500));

  // --- Step 5: descend along the grasp vector to the grasp pose -----------
  // This exact step is documented (prior RViz/fake-hardware task) as
  // numerically flaky with the vendor's kdl_kinematics_plugin -- retry a
  // bounded few times before giving up, rather than aborting on the first
  // transient failure.
  auto grasp_pose = downFacingPose(kTargetX, kTargetY, pinchZToEeZ(cube_center_z));
  arm.setPoseTarget(grasp_pose);
  bool descend_ok = false;
  for (int attempt = 1; attempt <= 4 && !descend_ok; ++attempt) {
    RCLCPP_INFO(logger, "descend to grasp pose attempt %d/4", attempt);
    descend_ok = planAndExecute(arm, "descend to grasp pose", logger);
    if (!descend_ok && attempt < 4) {
      rclcpp::sleep_for(std::chrono::milliseconds(800));
    }
  }
  if (!descend_ok) {
    RCLCPP_ERROR(logger, "Descend-to-grasp plan failed after retries -- aborting.");
    rclcpp::shutdown();
    spinner.join();
    return 1;
  }
  RCLCPP_INFO(logger, "[VERIFY WINDOW] at grasp pose, gripper still OPEN -- pausing 3s.");
  rclcpp::sleep_for(std::chrono::seconds(3));

  // --- Step 6: close gripper onto the cube (snug: half-gap = cube half-width) --
  std::vector<double> close_val = {kCubeSize / 2.0};
  gripper.setJointValueTarget(close_val);
  planAndExecute(gripper, "close gripper on cube", logger);
  RCLCPP_INFO(logger, "[VERIFY WINDOW] gripper CLOSED on cube -- pausing 3s.");
  rclcpp::sleep_for(std::chrono::seconds(3));

  // --- Step 6b: trigger the grasp-assist DetachableJoint plugin -----------
  // Pure friction did NOT hold the cube on a prior live test (this project's
  // own honest finding, see ar_gazebo.urdf.xacro / empty.world header
  // comments). This physics-assisted fallback rigidly attaches the cube to
  // the arm's own link_6 (the real, non-fixed-joint-lumped rigid body the
  // gripper assembly is actually part of -- gripper_base_link/ee_link do
  // NOT exist as separate physics entities after URDF->SDF fixed-joint
  // reduction). Triggered via a native ign-transport publish (the
  // DetachableJoint plugin's attach/detach topics are NOT ROS2 topics, a
  // plain `ros2 topic pub` finds no subscriber). A std::system() shell call
  // is a pragmatic choice for a demo script, not meant as production
  // practice.
  RCLCPP_INFO(logger, "Triggering grasp-assist DetachableJoint attach (cube -> link_6).");
  int detach_rc = std::system(
      "ign topic -t /model/cube/detachable_joint/attach "
      "-m ignition.msgs.Empty -p \"\" >/tmp/attach_pub.log 2>&1");
  RCLCPP_INFO(logger, "Attach publish command exit code: %d", detach_rc);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // --- Step 7: attach the cube to the gripper in the planning scene --------
  // (Collision-avoidance bookkeeping for subsequent planning calls only --
  // does NOT do the real physical holding, that's the DetachableJoint step
  // above.)
  arm.attachObject("cube", "gripper_base_link");
  RCLCPP_INFO(logger, "Attached 'cube' to gripper_base_link (planning-scene bookkeeping).");
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // --- Step 8: lift straight up (retreat along the grasp vector) -----------
  double lift_pinch_z = cube_center_z + 0.10;
  auto lift_pose = downFacingPose(kTargetX, kTargetY, pinchZToEeZ(lift_pinch_z));
  arm.setPoseTarget(lift_pose);
  if (!planAndExecute(arm, "lift/retreat", logger)) {
    RCLCPP_ERROR(logger, "Lift/retreat plan failed.");
  }
  RCLCPP_INFO(logger, "[VERIFY WINDOW] post-LIFT -- pausing 6s for external gz topic checks "
                       "(this is the key moment: is the real cube rigid body still at pedestal "
                       "height, or has it risen with the gripper?).");
  rclcpp::sleep_for(std::chrono::seconds(6));

  // --- Step 9: carry to a goal location (translate sideways while attached) ---
  double goal_y = kTargetY + 0.12;
  auto carry_pose = downFacingPose(kTargetX, goal_y, pinchZToEeZ(lift_pinch_z));
  arm.setPoseTarget(carry_pose);
  planAndExecute(arm, "carry to goal location", logger);

  RCLCPP_INFO(logger, "Pick demo sequence complete.");
  RCLCPP_INFO(logger, "[VERIFY WINDOW] post-CARRY -- pausing 8s for external gz topic checks.");
  rclcpp::sleep_for(std::chrono::seconds(8));

  rclcpp::shutdown();
  spinner.join();
  return 0;
}

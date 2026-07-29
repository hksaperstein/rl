cmake_minimum_required(VERSION 3.8)
project(ar4_pick_demo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(probe_pose src/probe_pose.cpp)
ament_target_dependencies(probe_pose rclcpp moveit_ros_planning_interface)

add_executable(pick_demo src/pick_demo.cpp)
ament_target_dependencies(pick_demo rclcpp moveit_ros_planning_interface moveit_msgs shape_msgs geometry_msgs)

add_executable(gazebo_pick_demo src/gazebo_pick_demo.cpp)
ament_target_dependencies(gazebo_pick_demo rclcpp moveit_ros_planning_interface moveit_msgs shape_msgs geometry_msgs)

install(TARGETS
  probe_pose
  pick_demo
  gazebo_pick_demo
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()

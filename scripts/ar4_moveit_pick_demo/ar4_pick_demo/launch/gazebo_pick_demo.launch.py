import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration


def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    tf_prefix_arg = DeclareLaunchArgument("tf_prefix", default_value="")
    tf_prefix = LaunchConfiguration("tf_prefix")

    # Real (non-fake) URDF -- same one moveit.launch.py feeds to move_group,
    # since this client node's own MoveGroupInterface needs matching
    # robot_description/robot_description_semantic params locally to
    # construct planning-group handles, even though actual planning and
    # execution happen in the separately-running move_group process.
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_description"), "urdf", "ar.urdf.xacro"
        ]),
        " ",
        "ar_model:=mk3",
        " ",
        "tf_prefix:=",
        tf_prefix,
        " ",
        "include_gripper:=True",
    ])
    robot_description = {"robot_description": robot_description_content}

    robot_description_semantic_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_moveit_config"), "srdf", "ar.srdf.xacro"
        ]),
        " ",
        "name:=mk3",
        " ",
        "tf_prefix:=",
        tf_prefix,
        " ",
        "include_gripper:=True",
    ])
    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }

    robot_description_kinematics = {
        "robot_description_kinematics": load_yaml(
            "annin_ar4_moveit_config", os.path.join("config", "kinematics.yaml")
        )
    }

    joint_limits = ParameterFile(
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_moveit_config"), "config/joint_limits.yaml"
        ]),
        allow_substs=True,
    )

    ompl_planning_yaml = load_yaml("annin_ar4_moveit_config", "config/ompl_planning.yaml")
    pilz_planning_yaml = load_yaml("annin_ar4_moveit_config", "config/pilz_planning.yaml")
    planning_pipeline_config = {
        "default_planning_pipeline": "ompl",
        "planning_pipelines": ["ompl", "pilz"],
        "ompl": ompl_planning_yaml,
        "pilz": pilz_planning_yaml,
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    executable = os.environ.get("AR4_PICK_DEMO_EXECUTABLE", "gazebo_pick_demo")

    demo_node = Node(
        package="ar4_pick_demo",
        executable=executable,
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            joint_limits,
            planning_pipeline_config,
            trajectory_execution,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([tf_prefix_arg, demo_node])

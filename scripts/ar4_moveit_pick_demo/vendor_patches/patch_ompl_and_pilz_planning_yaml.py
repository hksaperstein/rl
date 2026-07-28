import re

REQUEST_ADAPTERS_STR = """request_adapters: >-
  default_planner_request_adapters/AddTimeOptimalParameterization
  default_planner_request_adapters/ResolveConstraintFrames
  default_planner_request_adapters/FixWorkspaceBounds
  default_planner_request_adapters/FixStartStateBounds
  default_planner_request_adapters/FixStartStateCollision
  default_planner_request_adapters/FixStartStatePathConstraints
"""

def patch_ompl():
    path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_moveit_config/config/ompl_planning.yaml"
    with open(path) as f:
        src = f.read()
    old = """planning_plugins:
  - ompl_interface/OMPLPlanner
request_adapters:
  - default_planning_request_adapters/CheckStartStateBounds
  - default_planning_request_adapters/CheckStartStateCollision
  - default_planning_request_adapters/ResolveConstraintFrames
  - default_planning_request_adapters/ValidateWorkspaceBounds
response_adapters:
  - default_planning_response_adapters/AddTimeOptimalParameterization
  - default_planning_response_adapters/ValidateSolution
"""
    new = "planning_plugin: ompl_interface/OMPLPlanner\n" + REQUEST_ADAPTERS_STR
    assert old in src, "ompl pattern not found"
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("PATCHED ompl_planning.yaml")

def patch_pilz():
    path = "/root/ar4_ws/src/ar4_ros_driver/annin_ar4_moveit_config/config/pilz_planning.yaml"
    with open(path) as f:
        src = f.read()
    old = """planning_plugins:
  - pilz_industrial_motion_planner/CommandPlanner
request_adapters:
  - default_planning_request_adapters/CheckStartStateBounds
  - default_planning_request_adapters/CheckStartStateCollision
  - default_planning_request_adapters/ResolveConstraintFrames
  - default_planning_request_adapters/ValidateWorkspaceBounds
response_adapters:
  - default_planning_response_adapters/AddTimeOptimalParameterization
  - default_planning_response_adapters/ValidateSolution
"""
    new = "planning_plugin: pilz_industrial_motion_planner/CommandPlanner\n" + REQUEST_ADAPTERS_STR
    assert old in src, "pilz pattern not found"
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("PATCHED pilz_planning.yaml")

patch_ompl()
patch_pilz()

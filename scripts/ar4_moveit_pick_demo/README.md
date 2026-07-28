# AR4 + MoveIt2 collision-aware pick demo (cloud shakedown, 2026-07-28)

Strategic pivot artifact: after this project's hand-rolled Isaac Sim
IK/control approach for the AR4 hit a physics/control wall (see
`kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`), this task stood
up ROS2 Humble + MoveIt2 + the vendor `ar4_ros_driver` stack from scratch on
a fresh, ephemeral GCP CPU instance (no Isaac Sim involved at all) and used
MoveIt's collision-aware motion planning to demonstrate a pick, to test
whether MoveIt's planner solves the exact blockers the hand-rolled approach
hit (un-planned descent colliding with the cube, brittle grasp sequencing).

**This directory is NOT a live ROS2 workspace** (no ROS2/MoveIt/colcon on
the Pi) -- it's the demo package's source and the vendor-package patches
applied, kept for the record/reproducibility, matching this repo's
convention of not committing the cloud ROS2 workspace itself.

## Result

**MoveIt successfully planned and executed a full collision-aware pick
sequence** (approach -> descend to grasp -> close gripper -> attach cube ->
lift/retreat -> carry to a goal location) against a 15mm cube on a 40mm
pedestal added as real MoveIt collision objects, using the vendor AR4
MoveIt config's fake/mock ros2_control hardware (RViz visualization, no
Gazebo physics -- the pre-authorized fallback deliverable for this task).
Every step logged SUCCESS; the final `ee_link` pose was independently
verified via `tf2_echo` against the exact commanded target (not just a
"SUCCESS" log line). Video: see `logs/videos/ar4_moveit_cloud_pick_demo_2026-07-28.mp4`
in this repo.

**Honest repeatability caveat**: this exact recipe is not perfectly
deterministic on every re-run. Several follow-up runs on the same live
instance hit real OMPL/KDL planning flakiness specifically at the "descend
to grasp pose" step -- sometimes failing outright even with many more
planning attempts/time, or with a Cartesian-path variant capping out at a
fixed, non-obvious completion fraction. A dedicated, isolated
repeatability sweep (`ar4_pick_demo/src/probe_pose.cpp`'s "Robustness"
mode) confirmed the grasp pose itself is reachable 3/3 in isolation at
several nearby (x,y) columns, yet the exact same target failed repeatedly
once embedded in the full pick_demo sequence (with or without the
collision objects present -- tested both ways). This looks like a genuine
numerical-IK-solver sensitivity (the vendor stack uses `kdl_kinematics_plugin`,
an iterative/numerical solver, not an analytic one) rather than a logic bug
in the demo script itself, but it was not fully root-caused given this
task's scope -- see the header comment in `pick_demo.cpp` for the full
blow-by-blow. `pick_demo.cpp` as committed here is the exact recipe that
produced the successful, video-captured run.

## Setup recipe (what was actually run)

1. Provisioned a plain **Ubuntu 22.04** GCP instance directly via `gcloud
   compute instances create` (e2-standard-4, 50GB pd-balanced, no GPU --
   this task is CPU-only; MoveIt planning doesn't need a GPU) rather than
   `scripts/run_on_cloud_gpu.sh`, since this task needed a *persistent*
   instance for a long install-debug-launch-test loop, not a single
   opaque command.
2. Installed Docker (`apt-get install docker.io`) on the instance.
3. Pulled `moveit/moveit2:humble-release` directly from Docker Hub (ROS2
   Humble + MoveIt2 2.5.9 prebuilt, confirmed via `docker manifest`/Docker
   Hub API rather than assumed) -- per direct course-correction from the
   user mid-task: use a prebuilt ROS2+MoveIt image, do not apt-install
   ROS2/MoveIt from scratch on the host.
4. Ran a persistent container (`docker run -d --name ar4dev --network host
   -v ~/ar4_ws:/root/ar4_ws moveit/moveit2:humble-release sleep infinity`),
   `docker exec`'d into it repeatedly for the rest of setup (rather than
   one-shot `docker run --rm` calls) so build state / the workspace
   persisted across steps.
5. Installed a small set of extra apt packages inside the container: `git
   python3-rosdep xvfb ffmpeg x11-apps mesa-utils libgl1-mesa-dri
   libglx-mesa0 python3-colcon-common-extensions` (Xvfb + software GL for
   headless RViz rendering; the rest for cloning/building/recording).
6. Cloned `ycheng517/ar4_ros_driver` (the vendor AR4 ROS2 stack --
   `annin_ar4_description`, `annin_ar4_driver`, `annin_ar4_gazebo`,
   `annin_ar4_moveit_config`) into `~/ar4_ws/src`, ran `rosdep init/update`
   + `rosdep install --from-paths src --ignore-src -r -y` (pulled in the
   full Gazebo/`ros_gz`/`gz_ros2_control` stack too, though Gazebo itself
   was never actually launched for this task's chosen fallback path).
7. **Two real vendor-package build/runtime bugs found and patched** (see
   `vendor_patches/`, both applied as one-time in-place edits inside the
   cloned `ar4_ros_driver` checkout on the instance, not upstream PRs):
   - `annin_ar4_driver/src/ar_servo_gripper_hw_interface.cpp` referenced
     `hardware_interface::HardwareInfo::limits`, a field that no longer
     exists in the `ros2_control`/`hardware_interface` version this
     image's `humble-release` apt packages ship (2.54, 2026-04) -- a real
     upstream API version-skew, not anything specific to this project.
     This class is the REAL-hardware serial-servo-gripper driver, never
     instantiated for the fake/mock-hardware demo this task needed, so it
     was patched to stop referencing the removed field (see
     `vendor_patches/patch_ar_servo_gripper_hw_interface.py`) rather than
     reverse-engineering the new joint-limits API for a code path out of
     this task's scope. Real-AR4-hardware users would need a proper fix.
   - `annin_ar4_moveit_config/config/{ompl,pilz}_planning.yaml` used an
     older list-style `request_adapters`/`response_adapters` YAML format;
     the installed `moveit_ros_move_group` expects the current single
     space-separated-string `request_adapters` format (confirmed by
     diffing against the `moveit_resources_panda_moveit_config` reference
     config shipped in the very same container image) -- another real
     version-skew, patched via
     `vendor_patches/patch_ompl_and_pilz_planning_yaml.py`.
8. Built the vendor workspace (`colcon build --symlink-install`, ~20s once
   patched) plus this task's own `ar4_pick_demo` package (a small
   standalone `ament_cmake` package, this directory's `ar4_pick_demo/`,
   built into the same workspace).
9. Brought up `ros2 launch annin_ar4_moveit_config demo.launch.py` (the
   vendor's own fake-hardware MoveIt demo: `move_group`, `rviz2`,
   `robot_state_publisher`, `ros2_control_node` with
   `mock_components/GenericSystem`, `joint_trajectory_controller` +
   `gripper_controller` (`position_controllers/GripperActionController`)),
   under `Xvfb :99` with `LIBGL_ALWAYS_SOFTWARE=1`.
10. Ran `ar4_pick_demo`'s `pick_demo` executable (via
    `ar4_pick_demo/launch/probe_pose.launch.py`, which also constructs
    `robot_description`/`robot_description_semantic`/planning-pipeline
    parameters for the demo node itself -- MoveGroupInterface needs these
    as its OWN node parameters, not fetched remotely from `move_group`).
11. Captured video via `ffmpeg -f x11grab -i :99.0` against the Xvfb
    framebuffer.
12. Independently verified the final state via `ros2 run tf2_ros tf2_echo
    world ee_link` (ground truth, not just the demo's own "SUCCESS" log
    lines) and `ros2 service call /get_planning_scene ...` (confirmed the
    cube attached to `gripper_base_link` at the expected relative pose).

## Files here

- `ar4_pick_demo/` -- the standalone `ament_cmake` ROS2 package: `pick_demo`
  (the actual pick sequence) and `probe_pose` (a reachability/robustness
  diagnostic tool, used to derive the workspace column used and to
  characterize the repeatability issue above).
- `vendor_patches/` -- the two Python patch scripts applied in-place to the
  cloned `ar4_ros_driver` checkout (see point 7 above for what/why).

## Reproducing

Needs a fresh Ubuntu 22.04 instance with Docker. Follow the numbered recipe
above; `ar4_pick_demo/` and `vendor_patches/` here are ready to `docker cp`
in (or re-derive by pasting into the container via `cat > file <<'EOF'`,
the exact mechanism used this session since `gcloud compute scp` had CLI
flag issues with this project's standard SSH options, worked around with
`gcloud compute ssh --command "cat > file"` redirection instead).

# AR4 + MoveIt2 collision-aware pick demo (cloud shakedown, 2026-07-28; Gazebo physics follow-on, 2026-07-28)

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

## Gazebo PHYSICS follow-on (same day, 2026-07-28): real friction-jaw grasp attempt

Direct follow-on task: take the working MoveIt pick above (RViz-only,
fake/mock hardware, cube "attached" as a planning-scene object -- not a real
physics grasp) and put it on real Gazebo physics, so the cube is actually
gripped by jaw contact/friction and lifted under gravity, not teleported by
a planning-scene bookkeeping trick.

**Result: the arm is genuinely driven by Gazebo physics via
`ros2_control`/`gz_ros2_control` (MoveIt's `FollowJointTrajectory`/
`GripperCommand` actions actually move the simulated arm under real
dynamics -- verified via `/joint_states`, not just planner "SUCCESS" logs).
Pure friction did NOT hold the cube (honest negative result, live-tested,
video: `logs/videos/ar4_gazebo_pure_friction_attempt_2026-07-28.mp4`). A
grasp-assist `DetachableJoint` plugin (Gazebo's standard, widely-used
fallback for exactly this well-known parallel-jaw-grasp-physics
finickiness) DOES work and is video-confirmed lifting+carrying the cube:
`logs/videos/ar4_gazebo_physics_pick_demo_2026-07-28.mp4` (this file
contains several earlier flaky-IK retries before the successful run near
the end of its ~8min runtime -- ffmpeg recording was left running
continuously across retries rather than restarted each time).**

Ground-truth verification (not just eyeballing the video), per this
project's own standard: `ign topic -e -t /world/default/pose/info -n 1`
read the cube's real rigid-body pose directly from the physics engine.
Resting (pre-grasp): `(0.28, 0.0, 0.0475)`. Post-lift/carry (grasp-assist
run): `(-0.007, -0.308, 0.539)` -- the z-height alone rose from 0.0475m to
0.539m, conclusively confirming the cube physically followed the gripper
rather than staying on the pedestal.

### Vendor package used

`annin_ar4_gazebo` (already part of the `ar4_ros_driver` checkout used by
the cloud-shakedown task above) ships its own ready-made Gazebo Sim
("Fortress", `gz-sim` 6.18.0 for this container's ROS Humble packages)
bring-up: `gz_ros2_control/GazeboSimSystem` hardware plugin, a gripper with
a real prismatic-jaw `<mimic>` joint, `ros_gz_bridge`/`ros_gz_sim`
integration. This was used as-is per the task's own instruction to prefer
the vendor's own Gazebo sim over hand-rolling one -- no new Gazebo
integration was built from scratch, only bugs in the existing one were
found and patched (below).

### Real bugs found and fixed (all in `vendor_patches/`, beyond the two from the cloud-shakedown task above)

1. **`patch_gripper_jaw_friction.py`** -- the vendor URDF never sets Gazebo
   surface friction on the jaw links at all (no `<gazebo reference=...>`
   blocks for them). Added explicit `mu1`/`mu2` = 0.8. Result: still not
   enough to hold the cube by friction alone (see above) -- friction was
   ruled OUT, not confirmed, but a real, deliberate parameter rather than
   an unverified engine default.

2. **`patch_ros2_control_position_only_and_mimic_fix.py`** -- root-caused a
   real, 100%-reproducible SEGFAULT inside `gz_ros2_control`'s own
   `GazeboSimSystem::write()`/`::read()`, crashing on the controller
   manager's very first update cycle. Traced by reading
   `gz_ros2_control`'s actual shipped source and cross-referencing
   `github.com/ros-controls/gz_ros2_control/issues/628`: the vendor's
   `gripper_jaw2_joint` (a `<mimic>` joint) is registered inside the
   `<ros2_control>` block, and `GazeboSimSystem::write()`'s own manual
   mimic-mirroring code dereferences an ECM component pointer with no null
   check. Stripping the mimic joint's own interfaces (the documented fix
   from that GitHub issue) did NOT resolve it here, because the crashing
   code path is keyed off the *leader* joint's interfaces, not the
   follower's. The fix that actually worked: remove `gripper_jaw2_joint`
   from the `<ros2_control>` block **entirely**. The jaws still mirror
   correctly via the physics engine's own native handling of the plain
   URDF `<mimic>` tag (verified live via `tf2_echo` of both jaw links
   showing a correctly symmetric open pose).

3. **`patch_controllers_yaml_gain_and_interfaces.py`** -- matching
   controller-side interface change (drop `velocity` command interface),
   plus the fix for a real, live-measured weak-tracking bug: `gz_ros2_control`
   reads `position_proportional_gain` as a ROS2 parameter on the
   `controller_manager` node (via `declare_parameter`), not a URDF
   `<hardware><param>` (an earlier attempt to set it in the URDF was a
   silent no-op, confirmed via `ros2 param get`). With the default gain
   (0.1), a gripper commanded to open 0.014m only reached ~0.0049m of real
   physical travel (caught by checking `/joint_states` directly, not the
   action's own claimed result field, which was stale/misleading) before
   the controller falsely declared `reached_goal`. Gain raised to 50.0 in
   `controllers.yaml`, confirmed reaching the full commanded target.

4. **`patch_gazebo_launch_default_physics_engine.py`** -- the vendor's
   `gazebo.launch.py` hardcodes `--physics-engine
   gz-physics-bullet-featherstone-plugin`. This specific engine choice was
   the actual trigger for the segfault above -- removing the override
   (falling back to gz-sim's default DART engine) made the crash disappear
   across every subsequent launch. Trade-off reported honestly:
   bullet-featherstone is the engine a GitHub issue identifies as the only
   one with native `<mimic>` support; DART was empirically verified (not
   just assumed) to still correctly mirror this specific 1-leader/
   1-follower gripper mimic joint, but this is not a general claim DART
   supports `<mimic>` everywhere.

5. **`gazebo_world_physics_pick_scene.world`** (not a patch script -- a
   full reference copy of the modified world file, since the changes are
   wholesale content additions rather than small text edits). Replaces
   `annin_ar4_gazebo/worlds/empty.world`. Changes from vendor default:
   restored real gravity (vendor's own empty.world ships with `<gravity>0 0
   0</gravity>`, which would make a friction-grasp test physically
   meaningless -- the cube has no weight to hold up in zero-g); added a
   static `ground_plane`, a static `pedestal` (40mm), and a dynamic `cube`
   (15mm, 5g, mu=0.8) at the exact same pose the reachable-workspace column
   from the cloud-shakedown task already validated; added the
   `DetachableJoint` grasp-assist plugin to the **cube** model (not the
   robot's own URDF) -- a first attempt hosting the plugin on the robot
   failed at `Configure()` time with "Link ... not found in model mk3" for
   every link tried (`gripper_base_link`, `ee_link`), root-caused to a
   real timing/ordering gap: a model spawned dynamically via
   `ros_gz_sim create` doesn't yet have its own links registered in the
   ECM when its own plugins `Configure()`. Hosting on the cube (present in
   the static world file at server start) avoids that race; a further,
   separate finding was that `gripper_base_link`/`ee_link` don't exist as
   real physics link entities at all after URDF-to-SDF fixed-joint
   reduction (SDF lumps fixed-jointed child links into their last
   non-fixed parent) -- the real, attachable rigid body is `link_6`, used
   as the plugin's `child_link`.

### `gazebo_pick_demo.cpp` changes vs. the RViz-only `pick_demo.cpp`

New executable (`ar4_pick_demo/src/gazebo_pick_demo.cpp`, launched via
`gazebo_pick_demo.launch.py`), same step sequence as `pick_demo.cpp` plus:
a 1.5s settle delay after the approach move before planning the descend
(a live test hit the descend step's already-documented `kdl_kinematics_plugin`
flakiness 3/3 in a row before this delay was added -- consistent with the
next plan's start-state being captured while the arm was still settling
under real physics, unlike the fake-hardware case where state updates are
instantaneous); a bounded 4-attempt retry loop around the descend plan
specifically (this exact step's flakiness was already documented by the
cloud-shakedown task above); and, right after the gripper-close step, a
`std::system()` call publishing to the DetachableJoint plugin's
`attach_topic` via native `ign topic pub` (its topics are NOT ROS2 topics
-- a plain `ros2 topic pub` finds no subscriber; `std::system()` is a
pragmatic choice for a demo script, not meant as production practice).

### Cost

This follow-on task ran on the SAME persistent instance provisioned for
the cloud-shakedown task above (`e2-standard-4`, CPU-only, on-demand),
roughly 1h30m of additional wall-clock for this specific task's own setup
+ debugging + verification, well under the $5 cost cap. Full teardown
verified via `scripts/check_cloud_state.sh` at the end of this task too.

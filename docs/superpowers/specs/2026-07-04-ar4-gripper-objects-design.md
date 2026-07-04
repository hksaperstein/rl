# AR4 Sim Scene Extension — Gripper + Objects — Design

Date: 2026-07-04

## Context

Follow-on to `docs/superpowers/specs/2026-07-01-ar4-sim-foundation-design.md`
(area 1: sim foundation, arm only, no gripper). The longer-term goal driving
this work is a pick-and-place demo: load the AR4 arm with its gripper, place
a few small graspable objects in the scene, and (in a separate follow-on
project, out of scope here) train a small task-specific RL policy in Isaac
Lab to pick an object from one side of the workspace and place it on the
other, run it in a loop, and observe/record it.

This spec covers only the scene extension: adding the gripper to the robot
model and adding graspable objects to the scene, on top of the existing
foundation. It deliberately does not include any reward, reset, or training
logic — that is a separate sub-project, brainstormed and spec'd once this
one is built and verified.

A "physical AI" foundation model (e.g. NVIDIA Isaac GR00T) was considered
for the eventual pick-and-place policy and rejected: no pretrained model
(GR00T, Octo, OpenVLA, Isaac Lab's example checkpoints) works zero-shot on
the AR4's specific kinematics + custom parallel-jaw gripper without
fine-tuning or full training for this embodiment. The follow-on project will
instead train a small task-specific RL policy with Isaac Lab's built-in PPO
(rsl_rl), reusing the scene built here.

## Scope

- **Robot**: AR4 mk5, arm **+ gripper** (supersedes the area-1 decision to
  exclude the gripper — that deferral is now resolved).
- **Objects**: four small graspable rigid objects in the scene, sized to fit
  the gripper's ~28mm max aperture.
- **No task logic**: no rewards, no episode resets, no domain randomization,
  no RL training, no cameras/video. Fixed initial object poses only.
- **`num_envs=1`**, ground plane only (no table) — unchanged from the area-1
  foundation.

## Components

### Gripper (robot_cfg.py, build_asset.py)

- `build_asset.py` now runs xacro with `include_gripper:=true` (was `false`).
- The gripper has one actuated joint (`gripper_jaw1_joint`, prismatic,
  0–0.014m) and one URDF-mimic joint (`gripper_jaw2_joint`, mirrors jaw1).
  Rather than depending on the URDF `<mimic>` tag surviving import as a
  PhysX mimic-joint constraint (unverified for this Isaac Sim version), both
  joints are added as independently actuated joints via a second
  `ImplicitActuatorCfg` in `robot_cfg.py`
  (`GRIPPER_JOINT_NAMES = ["gripper_jaw1_joint", "gripper_jaw2_joint"]`).
  Callers (the action term, and `grasp_demo.py`) always write identical
  position targets to both joints, so they move in lockstep regardless of
  whether the imported mimic constraint is active.
- `env_cfg.py`'s `ActionsCfg` gets a `gripper_position` term
  (`mdp.JointPositionActionCfg` over `GRIPPER_JOINT_NAMES`), alongside the
  existing `joint_positions` term for the 6 arm joints.

### Objects (env_cfg.py, build_asset.py)

Four rigid objects, each ~15-20mm scale to fit within the gripper's ~28mm
aperture, distinct colors for visual tracking, mass ~5-20g, and a physics
material with elevated friction (stock defaults are too slippery to hold a
stable grasp):

- **Cube** — `sim_utils.CuboidCfg` (built-in Isaac Lab shape spawner).
- **Rectangular prism** — `sim_utils.CuboidCfg` with non-cubic dimensions.
- **Sphere** — `sim_utils.SphereCfg` (built-in shape spawner).
- **Triangular prism (wedge)** — Isaac Lab has no built-in wedge primitive.
  `build_asset.py` procedurally generates a small triangular-prism mesh (a
  short vertex/face list) and exports it to USD alongside the robot asset;
  the scene spawns it via `sim_utils.UsdFileCfg`, the same mechanism used
  for the robot.

Each object is a separate `RigidObjectCfg` added to `Ar4SceneCfg`, prefixed
under `{ENV_REGEX_NS}` like the robot.

**Layout**: 2 objects placed to one side of the arm base, 2 to the other,
all resting on the ground plane within the arm's reach — pre-staging the
"move from one side to the other" arrangement the follow-on RL task will
use. Poses are fixed (no randomization, no reset logic) since this step is
scene scaffolding, not a task.

## Data flow

`build_asset.py` (run once per machine) → robot USD **and** triangular-prism
mesh USD written to `rl/assets/` → `grasp_demo.py` loads the extended
`Ar4EnvCfg` (gripper + 4 objects) → GUI opens → hardcoded joint-waypoint
sequence moves the arm to one object, closes the gripper, lifts it, holds,
reopens → `RecorderManager` streams joint/action data to
`rl/logs/grasp_demo.hdf5`, same pattern as `drive_joints_demo.py`.

## Verification script: `rl/scripts/grasp_demo.py`

New script (does not modify `drive_joints_demo.py`, which stays focused on
arm-only joint sequences). Launches the GUI with `num_envs=1`, drives the
arm through hardcoded joint waypoints to reach one specific placed object
(position known ahead of time from the fixed layout), closes the gripper on
it, lifts, holds briefly for visual confirmation, reopens — verifying the
gripper actuates correctly and can physically hold an object. Uses the same
`ActionStateRecorderManagerCfg` recording pattern as `drive_joints_demo.py`.

## Error handling

Same minimal, fail-loud approach as the area-1 foundation: `build_asset.py`
errors clearly if `AR4_DESCRIPTION_PATH` is unset/missing; `grasp_demo.py`
errors clearly if the built USD assets (robot or wedge mesh) are missing.
No retry/fallback logic.

## Testing / verification

Verified visually: GUI shows the gripper opening/closing correctly and
successfully grasping/lifting the target object, plus the resulting HDF5 log
(joint/gripper data present and sane). No automated test suite, consistent
with the foundation spec's approach — a headless smoke test becomes worth
adding once the follow-on RL sub-project introduces real reward/termination
logic.

## Out of scope (deferred to the follow-on RL sub-project)

- Reward functions, episode resets, domain randomization
- RL training (Isaac Lab PPO / rsl_rl), policy checkpoints
- The actual "pick from one side, place on the other" task logic and loop
- Cameras, video capture/output
- Any pretrained/foundation-model policy (GR00T, Octo, etc. — considered and
  rejected for this project, see Context)

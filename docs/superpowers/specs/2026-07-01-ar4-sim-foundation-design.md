# AR4 Sim Foundation — Design

Date: 2026-07-01

## Context

Long-term goal: an RL training pipeline for the AR4 6DoF manipulator, trained
in simulation, deployed to the real robot. That work breaks into four areas,
built in order since each depends on the last:

1. **Sim foundation** — AR4 robot model + a physics sim capable of driving RL
   training (this spec)
2. **Task/scenario layer** — gym-style tasks, reward functions, domain
   randomization
3. **Training infrastructure** — local + cloud compute, experiment tracking,
   checkpointing
4. **Sim-to-real deployment** — policy export, a ROS2 inference node,
   real-time control on real hardware

One shared logging/data-collection approach is used across all four areas
rather than a bespoke one per area. For area 1, that means Isaac Lab's own
recording tools rather than anything custom-built.

This spec covers area 1 only: get the AR4 launched, joints driven, joint data
collected. Later areas get their own specs once this one is built and
working.

## Framework choice: Isaac Lab on Isaac Sim

Isaac Sim (already installed locally, v6.0.1) is the underlying simulator —
physics, rendering, USD scenes, low-level robot APIs. It has no RL concepts
(episodes, rewards, resets, vectorized envs).

Isaac Lab is the RL framework built on top of Isaac Sim: gym-style
`env.step()/reset()`, running many copies of a scene in parallel on one GPU,
and out-of-the-box logging (TensorBoard via rsl_rl/skrl) and an HDF5
`RecorderManager` for episode/trajectory data. Since RL training is the
explicit end goal, building directly against Isaac Lab's env API now avoids
redoing this work when area 3 needs vectorized training.

**Isaac Lab is not installed yet** — only Isaac Sim standalone is present.
Installing Isaac Lab pinned to a version compatible with Isaac Sim 6.0.1 is
part of the implementation work.

Two alternatives were considered and rejected:
- Raw Isaac Sim scripting (no Isaac Lab): simpler short-term, but reinvents
  vectorized envs, RL-library glue, and data recording once training starts.
- Switching engines (MuJoCo/Genesis): lighter-weight, but throws away the
  Isaac Sim setup and URDF import path already validated, for no clear
  benefit given a capable local GPU (RTX 5070 Ti, 16GB) is available.

## Scope for this area

- **Robot**: AR4 mk5, arm only (no gripper) — gripper is deferred to
  whichever future scenario actually needs grasping.
- **No ROS2 in this area.** RL training in Isaac Lab runs many physics
  environments batched on the GPU in a single process; wrapping that in
  ROS2 topics/services would serialize per-env, per-step, defeating the
  batching that makes GPU-parallel training viable. ROS2 is scoped to
  single-instance use — visualization/teleoperation/data collection during
  development, and later, deployment to the real robot (area 4). This area
  uses the direct Isaac Lab/Sim Python API only.
- **Repo location**: `rl/` at the repo root, alongside `src/` and `docker/`.
  It is a separate build system from the ROS2 colcon workspace (Isaac Lab
  extension project layout, not a colcon package) but lives in this repo's
  history.
- **Asset source**: the AR4 URDF + meshes are *not* vendored into this repo.
  `build_asset.py` reads them from an external path (env var, e.g.
  `AR4_DESCRIPTION_PATH`), the same pattern the existing ROS2 side already
  uses (this repo does not vendor `ar4_ros_driver`; a submodule for it was
  deliberately removed in commit `d637dde`). The URDF/meshes are only needed
  at asset-build time — the Isaac Lab env only ever loads the *built* USD.
  Cloud/CI implication: a cloud training machine will not have access to
  this external path. The area-3 (training infrastructure) design is
  responsible for shipping the already-built USD artifact to cloud/CI
  environments; this area does not need to solve that.

## Components

- **`rl/scripts/build_asset.py`** — setup script, rerunnable. Reads
  `AR4_DESCRIPTION_PATH`, runs `xacro` on `ar.urdf.xacro` with
  `ar_model:=mk5 include_gripper:=false` to produce a plain URDF, then
  converts it to USD via Isaac Sim's URDF importer
  (`isaacsim.asset.importer.urdf`), resolving `package://annin_ar4_description/...`
  mesh references against `AR4_DESCRIPTION_PATH`. Output goes to
  `rl/assets/` (gitignored). Fails loudly with a clear message if
  `AR4_DESCRIPTION_PATH` is unset or does not exist.
- **`rl/tasks/ar4/`** — Isaac Lab environment definition for the AR4 arm:
  articulation config pointing at the generated USD, scene config, env
  config with configurable `num_envs` (default 1, for this stage's
  interactive use). No reward or termination logic — that belongs to area 2.
  This is a thin scaffold: launch, read joint state, accept joint commands.
- **`rl/scripts/drive_joints_demo.py`** — verification script. Launches
  Isaac Sim's GUI with `num_envs=1`, drives the arm through a preset joint
  command sequence (step or sinusoidal targets per joint), and uses Isaac
  Lab's `RecorderManager` to record joint position/velocity/target to
  `rl/logs/*.hdf5` (gitignored) every step. Fails loudly if the USD asset
  from `build_asset.py` is missing.

## Data flow

`build_asset.py` (run once per machine) → USD written to `rl/assets/` →
`drive_joints_demo.py` loads that USD into an Isaac Lab env → GUI opens →
joint sequence runs → `RecorderManager` streams joint data to
`rl/logs/*.hdf5` as the sequence executes.

## Error handling

Minimal, appropriate for one-shot dev scripts rather than a service:
- `build_asset.py` errors clearly if `AR4_DESCRIPTION_PATH` is unset or
  missing.
- `drive_joints_demo.py` errors clearly if the built USD asset is missing,
  pointing back to `build_asset.py`.
- No retry or fallback logic.

## Testing / verification

Verified visually (GUI shows the AR4 executing the joint sequence
correctly) plus the resulting HDF5 log (joint data present and sane) — no
automated test suite for this area. Once area 2 adds real reward/termination
logic, a headless smoke test (few steps, assert no crash, assert expected
tensor shapes) becomes worth adding for CI; not needed for this foundation
piece.

## Out of scope (deferred to later areas)

- Reward functions, task definitions, domain randomization (area 2)
- Vectorized/parallel-env training, cloud compute, experiment tracking,
  shipping built USD assets to cloud/CI (area 3)
- ROS2 integration, policy export, real-hardware control (area 4)
- Gripper model

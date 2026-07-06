# AR4 single-object, camera-observed grasp training experiment

## Problem / request

Four independent hypotheses (lift-weight bump, additive grasp bonus,
alignment-gated bonus, gripper PD-gain rescale) have all failed to get the
AR4 arm to grasp+lift the sphere — see `ROADMAP.md`'s "grasp/lift never
emerges" entry for full history. The user, drawing on their own experience
grasping objects without a contact sensor, asked for a different kind of
change instead of another reward/control tweak: **simplify the scene to
one object, and train using the real RGB camera instead of privileged
simulation state.**

## Interpretation (stated explicitly so it can be corrected)

1. **Single object**: strip the scene down to just the sphere — remove the
   cube/rect_prism/wedge static props entirely for this experiment, rather
   than just having them present-but-irrelevant as today.
2. **RGB camera**: replace the training-time `sphere_position` observation
   (currently `mdp.object_position_in_robot_root_frame`, privileged
   ground-truth from the simulator) with a position derived from the real
   `perception_camera` + this repo's existing classical perception pipeline
   (`perception/pipeline.py`'s `run_perception`, already used for
   *evaluation* via `eval_loop.py --perception` and
   `scripts/_perception_adapter.py` — reuse and generalize this rather than
   build a new pipeline). Training itself has never used the camera before
   (by original design, for parallelism/render-cost reasons — see
   `docs/superpowers/specs/2026-07-04-ar4-perception-integration-design.md`);
   this experiment tests training with it directly.
3. **Reward stays privileged.** The reward function computing
   `reaching_sphere`/`lifting_sphere`/etc. continues to use ground-truth
   simulation state — real robots don't need their reward function to be
   observable, only the *policy's observations* need to come from a
   realistic sensor. Only the `sphere_position` observation term changes.

## Pre-existing bug discovered, fix as part of this work

`scripts/_perception_adapter.py`'s `cube_position_obs_slice()` and
`perceive_cube()` are hardcoded to look up an observation term named
`"cube_position"` and call `find_by_shape(tracked, "cube")` — stale from
before the Cube→Sphere retargeting (`ROADMAP.md`, commit `79c9089`).
`tasks/ar4/pickplace_env_cfg.py`'s actual observation term is
`sphere_position`. This means **`eval_loop.py --perception` and
`interactive_demo.py` are currently silently broken** (would raise
`ValueError: cube_position term not found` the moment `--perception` is
used) — not something this session broke, but worth fixing now since this
experiment depends on the same code path working correctly.
`perception.tracker.find_by_shape` itself is already generic (takes any
shape-label string) — only the caller needs updating.

## Design

### 1. Single-object scene

Add a new scene/env config (e.g. `Ar4PickPlaceSingleObjectSceneCfg`) that
subclasses the existing pick-place scene machinery but **omits** `cube`,
`rect_prism`, `wedge` from the scene entirely — keep it additive/parallel
to the existing configs (don't modify `Ar4SceneCfg`/`objects_cfg.py`
directly, since `interactive_demo.py`, `perception/tests`, and other
scripts depend on all four objects existing there). Everything else
(robot, sphere, ground, light, `ee_frame`) stays the same.

### 2. Camera-observed training

This is the substantial new piece. `train.py` currently has zero camera/
perception code — training relies entirely on `OnPolicyRunner.learn()`'s
internal rollout loop, which doesn't offer a natural per-step hook to
override an observation slice the way `eval_loop.py`'s manual loop does
(`obs[:, col_start:col_end] = cube_pos_b`).

Recommended approach: write a **Gym observation wrapper** (e.g.
`PerceptionObservationWrapper`, could live in `scripts/_perception_adapter.py`
or a new small module under `tasks/ar4/` — use judgment on the right home)
that wraps the raw `ManagerBasedRLEnv` **before** `RslRlVecEnvWrapper` (or
after — check which order actually gets you access to per-env camera
tensors and the ability to rewrite the returned obs tensor; introspect
`RslRlVecEnvWrapper`'s actual step/reset signature rather than guessing).
On each `step()`/`reset()`, for every parallel env: read that env's
camera output (`camera.data.output["rgb"]`/`["distance_to_image_plane"]`,
indexed per-env, not just index `[0]` like the existing eval-only code
assumes for its single-env case), run `run_perception(...)` +
`find_by_shape(tracked, "sphere")` (fixing the stale `"cube"` per above),
convert to the robot root frame, and overwrite that env's
`sphere_position` observation columns with the camera-derived value
(falling back to something sane — e.g. leave the previous value, or a
zero/last-known position — when the sphere isn't detected, matching how
`_perception_adapter.py`'s `perceive_cube` already handles a `None`
detection at eval time).

### 3. Parallelism / render-cost tradeoff — use engineering judgment

The perception pipeline (`perception/pipeline.py`) is plain numpy/CPU, not
GPU-batched across envs, and camera rendering itself is not free (this
repo's own design doc explicitly avoided camera rendering during training
for exactly this reason). Running real per-env segmentation/classification
every step across the usual 4096 envs is very likely infeasible time-wise.
Start with a **much smaller `num_envs`** for this experiment — e.g. 16 or
32 — and treat the actual achievable throughput as something to discover
empirically (measure iteration time in the smoke test before committing to
a full run's `num_envs`/`max_iterations`, and scale down further, or
lengthen the training budget, if needed). Report the real iteration-time
numbers you observe so the Principal can judge whether this is a
practical setup at all, independent of whether it helps grasping succeed.

### 4. What to keep vs. build fresh

Reuse as much of `scripts/_perception_adapter.py`'s existing logic as
possible (fixing the cube→sphere staleness), rather than reimplementing
segmentation/classification/tracking — this repo's perception pipeline
already exists and works (per `docs/superpowers/plans/2026-07-05-perception-sidewall-fix-report.md`,
3/4 real-object classification now works correctly, including the sphere).
A single-sphere scene makes the shape-classification step nearly moot for
this experiment (nothing else to confuse it with), which is itself a
useful simplification consistent with the user's request.

## Verification plan

1. Smoke test first at a small `num_envs`/`max_iterations`, checking for
   correctness (no crash, observation actually changes when the sphere
   moves) AND real iteration-time cost before committing to a longer run.
2. A real training run (however many iterations prove tractable within a
   reasonable wall-clock budget — use judgment, report what you chose and
   why) with TensorBoard scalars pulled at the end
   (`Episode_Reward/lifting_sphere`, `Episode_Reward/reaching_sphere`,
   `Episode_Termination/sphere_reached_goal`, `Train/mean_reward`), same
   format as every prior experiment this session.
3. Real eval + video verification (10 episodes if feasible, fewer if the
   camera-based path is too slow — use judgment), frame-extracted and
   visually inspected, same rigor as every prior experiment.
4. Compare against the four prior (privileged-observation) failures — is
   the failure signature the same, different, or does grasping actually
   emerge this time?

## Decision framing

This is a genuinely different kind of change (observation modality +
scene complexity) rather than another reward/control tweak, which is
exactly what was requested after four reward/control-only hypotheses
failed. Report the real result honestly regardless of outcome — including
if the camera-based training loop turns out to be impractically slow at
any usable `num_envs`, which would itself be an important, honestly-
reported finding.

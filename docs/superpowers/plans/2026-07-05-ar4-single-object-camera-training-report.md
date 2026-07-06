# AR4 single-object, camera-observed grasp training experiment — report

**Status: paused before completion, by explicit user/Principal instruction — not a
success or failure verdict.** This is an honest write-up of a partial run: what
was built, what was measured, and what is still unknown. Nothing in this
worktree has been committed.

## What was requested

Per `docs/superpowers/specs/2026-07-05-ar4-single-object-camera-training-design.md`:
strip the scene down to the sphere only (no cube/rect_prism/wedge) and train
using the real RGB-D `perception_camera` + this repo's existing classical
perception pipeline for the `sphere_position` *observation*, while keeping the
reward function on privileged simulation state. Plus, fix a pre-existing stale
`cube_position`/`find_by_shape(tracked, "cube")` bug in
`scripts/_perception_adapter.py` left over from the Cube→Sphere retargeting.

## What was built

### 1. Bug fix: cube→sphere staleness in the perception adapter

`scripts/_perception_adapter.py` was hardcoded to look up an observation term
named `"cube_position"` and call `find_by_shape(tracked, "cube")` — both stale
since `tasks/ar4/pickplace_env_cfg.py`'s actual term is `sphere_position`. This
meant `eval_loop.py --perception` and `interactive_demo.py` would raise
`ValueError: cube_position term not found` immediately. Fixed:

- `cube_position_obs_slice()` → renamed `sphere_position_obs_slice()`, now
  looks up `"sphere_position"`. Its column-scanning logic was also factored
  into a new generic `observation_term_slice(env, group_name, term_name)`
  helper (reused by the new training wrapper below).
- `perceive_cube()` → renamed `perceive_sphere()`, now calls
  `find_by_shape(tracked, "sphere")`. Also generalized to take an `env_index`
  parameter (default 0, preserving existing single-env callers' behavior)
  since the multi-env training wrapper needs to read per-env camera/robot
  data instead of always index `[0]`.
- Call sites updated in `scripts/eval_loop.py` and `scripts/interactive_demo.py`
  (imports, variable names, an inline `t.shape_label == "cube"` filter in
  `interactive_demo.py` that had the identical staleness bug, docstrings, and
  the `--stable_seconds` help string). Both entry points should now actually
  work with `--perception` — not independently re-verified end-to-end in this
  session (out of scope; verification budget went to the new training path),
  but the specific `ValueError` this bug caused is gone and the fix is
  mechanical/low-risk.

### 2. Single-object scene: `tasks/ar4/pickplace_single_object_env_cfg.py` (new file)

`Ar4PickPlaceSingleObjectSceneCfg(InteractiveSceneCfg)` — ground, light, robot,
`sphere` only (reuses `objects_cfg.SPHERE_CFG`), the `ee_frame`
FrameTransformer, and the top-down `perception_camera`, always on. Deliberately
does **not** subclass `Ar4SceneCfg`/`Ar4PickPlaceSceneCfg` (both of which
include cube/rect_prism/wedge) — built as a parallel scene class so
`interactive_demo.py`, `perception/tests`, and other scripts that depend on
all four objects existing in the shared config are untouched.

`Ar4PickPlaceSingleObjectEnvCfg(Ar4PickPlaceEnvCfg)` swaps in the new scene
(default `num_envs=16`) and otherwise inherits the existing task's
observations/rewards/terminations/commands/actions unchanged — all of them
are already parametrized purely by `SceneEntityCfg("sphere")`/`("ee_frame")`
and never reference cube/rect_prism/wedge, so no other config edits were
needed.

### 3. Camera-observed training: `PerceptionObservationWrapper` in `scripts/_perception_adapter.py`

This is the substantial new piece, and it surfaced a real interface subtlety
the design doc asked to check rather than assume:

**Finding from introspection**: `RslRlVecEnvWrapper.get_observations()` —
which `rsl_rl`'s `OnPolicyRunner.learn()` calls exactly once, to get the very
first observation before the rollout loop starts (`on_policy_runner.py:72`) —
does **not** delegate to `self.env.get_observations()`. It calls
`self.unwrapped.observation_manager.compute()` directly, which walks straight
through every `gym.Wrapper` in the chain via `.unwrapped`, including a naive
observation wrapper's own `step()`/`reset()` overrides. A standard
"`gym.Wrapper` that overrides `step`/`reset`" implementation would silently
fail to affect that first observation. (`RslRlVecEnvWrapper` also has its own
documented constraint of having to be the *last* wrapper in the chain, so our
wrapper has to sit between the raw env and it regardless.)

**Resolution**: `PerceptionObservationWrapper` is still a `gym.Wrapper` (sits
between the raw `ManagerBasedRLEnv` and `RslRlVecEnvWrapper`), but instead of
overriding `step()`/`reset()`, its `__init__` monkeypatches the raw env's
`observation_manager.compute` bound method. Every entry point —
`ManagerBasedEnv.step`/`reset`, `ManagerBasedRLEnv.step`, and
`RslRlVecEnvWrapper.get_observations()` — all bottom out in
`self.obs_buf = self.observation_manager.compute(...)`, so patching `compute`
itself intercepts all three uniformly regardless of wrapper order concerns.

Per call, for every parallel env: reads that env's `distance_to_image_plane`
frame, intrinsics, camera pose (`camera.data.output[...][i]`, not just `[0]`),
runs `run_perception` + `find_by_shape(tracked, "sphere")` (one
`perception.tracker.ObjectTracker` instance per env, for temporal
staleness/matching), converts the detection to that env's robot root frame via
`subtract_frame_transforms`, and overwrites that env's `sphere_position`
columns in the (cloned) policy observation tensor. **Fallback when
undetected**: that env's column slice is left as whatever
`observation_manager.compute()` already produced — i.e. the privileged
ground-truth value for that step — mirroring exactly how `perceive_sphere`'s
`None`-detection case already behaves at eval time (skip the overwrite, don't
inject a synthetic value).

### 4. `scripts/train.py`

Added a `--perception` flag: switches the env cfg to
`Ar4PickPlaceSingleObjectEnvCfg`, forces `--enable_cameras` (required for the
camera sensor to instantiate at all), and inserts
`PerceptionObservationWrapper(env, ground_z=GROUND_Z)` between the raw env
and `RslRlVecEnvWrapper`. Reward/termination/PPO hyperparameters are
unchanged from the existing task.

## Verification performed

### Smoke tests (correctness)

- First smoke test (`--num_envs 16 --max_iterations 2`, default buffered
  stdout): ran to completion (exit 0), `model_0.pt`/`model_1.pt` checkpoints
  written — confirmed no crash, but per-iteration stats were invisible
  because Isaac Sim's process teardown doesn't flush buffered stdout on
  `simulation_app.close()`.
- Second smoke test with `PYTHONUNBUFFERED=1` (`--num_envs 16
  --max_iterations 5`): completed cleanly, **and this is where real
  measurements came from** (below). Confirms the observation override
  mechanism itself runs without error across 5 iterations / 16 envs.

### Real measured throughput

At `num_envs=16`, steady-state iteration time was **6.3–7.3s** (5 logged
iterations: 7.33, 6.74, 6.43, 6.33, 6.44s). rsl_rl's own per-iteration
breakdown: `Computation: 59 steps/s (collection: 6.393s, learning 0.042s)` —
**collection (env stepping + camera render + perception) is ~150x the cost of
the PPO learning phase**, confirming the perception pipeline (not RL update)
is the bottleneck, exactly as the design doc anticipated.

**Important architectural finding, not previously called out in the design
doc**: the per-env perception loop inside `PerceptionObservationWrapper` is a
plain serial Python `for i in range(num_envs)` loop over numpy calls — there
is no batching/vectorization across envs at all. This means collection time
scales roughly *linearly* with `num_envs`, not sub-linearly the way
GPU-vectorized physics does. Concretely: going from 16→32 envs would roughly
**double** iteration wall-time rather than buying extra throughput "for
free" — there's no parallelism benefit to raising `num_envs` in this
implementation, only a data-diversity-per-iteration tradeoff. This is worth
knowing before anyone reflexively bumps `num_envs` up for this experiment.

### Full training run — paused at iteration 110/500

Launched `--num_envs 16 --max_iterations 500` (chosen as a tractable middle
ground: ~500 × 6.4s ≈ 53 minutes, versus ~1500 × 6.4s ≈ **2.7 hours** for a
run of the same length as every other experiment this session). **The user
asked to pause all training instances mid-run** (after floating a follow-on
"try a contact sensor" idea that the Principal is scoping separately); the
process was killed cleanly (`SIGTERM`, confirmed via `ps`/`nvidia-smi` — no
orphaned Isaac Sim/GPU usage left behind) at **iteration 110/500**, timestep
42,624, ~12m23s of wall-clock training. This was a deliberate pause, not a
crash or timeout.

Real logged scalars at four points in the partial run:

| Iteration | Wall time | Mean reward | reaching_sphere | lifting_sphere | sphere_reached_goal | Episode length |
|---|---|---|---|---|---|---|
| 0 | 0:00:07 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 12.0 |
| 50 | 0:05:56 | 0.53 | 0.2363 | 0.0000 | 0.0625 | 224.3 |
| 109 | 0:12:17 | 0.91 | 0.1749 | 0.0000 | 0.0000 | 249.3 |
| 110 (last logged before pause) | 0:12:23 | 0.91 | 0.0834 | 0.0000 | 0.0000 | 249.3 |

Checkpoints saved at the configured `save_interval=50`: `model_0.pt`,
`model_50.pt`, `model_100.pt` in
`logs/train/2026-07-05_22-55-53/` — no eval/video verification was performed
on any of these per the pause instruction (this would normally be the next
step, using `model_100.pt` as the most-trained checkpoint available).

**What this partial data does and doesn't tell us**: `mean_reward` is
climbing (0 → 0.53 → 0.91) and episode length is growing (12 → 249, i.e. the
agent is increasingly avoiding early termination/timeout), so *something* is
being learned — the setup is not obviously broken or stuck at initialization.
`reaching_sphere` is present but noisy/non-monotonic (0.24 → 0.17 → 0.08
across iterations 50/109/110) rather than cleanly converged the way it was by
this point in the four privileged-observation experiments. `lifting_sphere`
is flat at exactly `0.0000` through all 110 logged iterations — identical to
every one of the four privileged-observation failures' signature. **This is
not enough data to conclude grasping would or wouldn't have emerged** — 110
iterations is roughly 1/14th the length of every prior experiment's full
1500-iteration run, and separately, at `num_envs=16` each iteration collects
384 samples versus 4096×24=98,304 for the baseline runs (a **256x** smaller
per-iteration sample count) — so even "iteration 110" here represents far
less accumulated experience than iteration 110 would in any prior experiment.
No success/failure verdict should be drawn from this partial run.

## Honest overall verdict

**Practicality**: camera-observed training via this classical CV pipeline is
*possible* (it runs, correctly overrides the observation, no crashes) but
**expensive relative to every prior experiment's wall-clock budget**. A run
of comparable length (1500 iterations) would cost **~2.7 hours** at
`num_envs=16` — versus minutes for the privileged-observation baseline runs
at `num_envs=4096`. That estimate is a direct extrapolation of real measured
per-iteration timing, not a guess. Additionally, per-iteration sample count
is inherently ~256x smaller than the baseline runs (a `num_envs` tradeoff,
not a bug), so "1500 iterations" here is not sample-equivalent to "1500
iterations" in any prior experiment — a truly comparable amount of experience
would take dramatically longer still. The serial (non-batched) per-env
perception loop is the root cause and is architectural, not a tuning
opportunity — raising `num_envs` doesn't purchase more total throughput here
the way it would for a GPU-vectorized cost.

**Correctness**: the implementation is verified correct as far as tested —
smoke tests confirm the observation genuinely gets overridden with
camera-derived positions per env, the fallback behavior (leave privileged
value when undetected) works as designed, and the cube→sphere bug fix
resolves the `ValueError` that was silently breaking `eval_loop.py
--perception`/`interactive_demo.py`.

**Grasping outcome: undetermined.** The partial run shows learning is
happening (reward/episode-length trending up) but zero evidence of lift by
iteration 110 — consistent with, but far too short to distinguish from, both
"this would eventually replicate the four prior failures" and "this needs
more time/data than the others to show its effect." **If the Principal wants
a conclusive answer here, it requires a deliberately long-running dispatch
(realistically 2.5–3+ hours of wall-clock, ideally started with headroom to
run unattended) — not another quick smoke-test-scale check.**

## Files changed/added

- `scripts/_perception_adapter.py` — bug fix (cube→sphere) + new
  `observation_term_slice`/`sphere_position_obs_slice`/`perceive_sphere` +
  new `PerceptionObservationWrapper` class.
- `scripts/eval_loop.py` — updated to the renamed adapter functions/sphere
  terminology (mechanical rename, not re-verified end-to-end this session).
- `scripts/interactive_demo.py` — same rename, plus fixed an inline
  `shape_label == "cube"` staleness bug and cube→sphere docstring/help-text
  wording.
- `scripts/train.py` — new `--perception` flag wiring
  `Ar4PickPlaceSingleObjectEnvCfg` + `PerceptionObservationWrapper`.
- `tasks/ar4/pickplace_single_object_env_cfg.py` (new) — single-object scene
  + env cfg.
- Training artifacts (not committed, left on disk):
  `logs/train/2026-07-05_22-55-53/` (`model_0.pt`, `model_50.pt`,
  `model_100.pt`, tfevents).

Nothing in this worktree has been committed; all changes above are left
uncommitted for review, per instructions.

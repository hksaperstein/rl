# Joint-space (no-IK) RL die-lift — design spec

Date: 2026-07-11. Branch: `franka-panda-pivot`. Tier 1 (structural: new
action space + new object for RL). Author: Principal. User request:
"RL pipeline with no IK"; scope decision (user-confirmed): single-die
lift first, commanded-pick RL later.

## Hypothesis (falsifiable)

Direct joint-space PPO training (`JointPositionActionCfg` mirroring Isaac
Lab's own validated `Isaac-Lift-Cube-Franka-v0` recipe) will produce a
genuine, sustained grasp-and-lift of a d20 die on the Franka platform
within a comparable iteration budget to Isaac Lab's own shipped recipe,
because this project's one prior joint-space failure (AR4 Experiment 10)
is confounded by three independently-documented, unresolved AR4-asset
defects (EE-offset error, unenforced jaw mimic, unverified jaw collision
geometry) that do not apply to Franka.

Falsified if: after a full 1500-iteration run at the recipe's settings,
the success-rate metric stays at noise level AND the 10-episode eval
video shows no genuine grasp+lift (per this repo's verification standard,
both must be checked; scalars alone decide nothing — Experiment 15
precedent).

## Grounding

`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md`
(commit 3e5dbc7): Isaac Lab shipped-config facts (joint_pos variant is
the only lift variant with RL agent configs; exact action cfg values),
verified external citations, and the AR4-vs-Franka precedent analysis.
Counter-evidence carried honestly: Varin et al. 2019 and Martín-Martín
et al. 2019 favor task-space/impedance action spaces for contact-rich
precision tasks — this experiment is in part a direct test of whether
that concern binds at this task's precision level on a validated
platform.

## Design (Approach A — minimal diff against two references)

New file `tasks/franka/dice_lift_joint_env_cfg.py`:

- Start from this repo's `tasks/franka/lift_env_cfg.py` (already a
  verified adaptation of Isaac Lab's lift recipe — scene, rewards,
  observations, events, PPO runner cfg all stay IDENTICAL).
- Change 1 — action space: `arm_action` becomes
  `mdp.JointPositionActionCfg(asset_name="robot",
  joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)`,
  copied exactly from Isaac Lab's joint_pos lift variant (implementer
  verifies exact kwargs against the installed source, not from memory).
  `gripper_action` (BinaryJointPositionActionCfg) unchanged. NO
  DifferentialIK anywhere.
- Change 2 — object: the d20 die replaces the recipe's object. Use a
  physics-baked COPY of `vision/data/raw/dice_sets_v1/set_00000_d20.usd`
  under `assets/dice/` (one-time offline bake via `scripts/build_asset.py`'s
  technique: apply RigidBody/Mass/CollisionAPI + convexHull approximation
  + the mm-as-m 0.001 scale INTO the new USD), so the RL env needs no
  runtime schema patching. Source asset untouched (shared with vision/).
  - Pre-check (from research): compare the d20's real dimensions
    (~19mm size_mm, ~30mm vertex-to-vertex bbox) against the shipped
    recipe's DexCube and this repo's lift task object cfg; record the
    delta in the implementation report. Die mass 0.01kg (dice-demo
    value). If the recipe's object-position event/command ranges assume
    the DexCube's size anywhere, adapt and document.
- Everything else byte-identical to `lift_env_cfg.py`: rewards
  (reaching/lifting/goal-tracking weights), observations (privileged
  object pose — detector-in-loop is Phase I, explicitly out of scope),
  episode length, PPO hyperparameters, num_envs=4096 sizing.
- Wire-up: new gym registration + `--dielift-joint` (or equivalent)
  flag on `scripts/train.py`/`scripts/eval_loop.py` following the
  existing variant-flag pattern.

## Success criteria / verdict protocol

Authoritative metric (one pipeline, per repo practice): the lift task's
success/termination metric (`object_reached_goal`-equivalent — implementer
names the exact scalar in the report before training starts). Verdict
requires BOTH:
1. Full 1500-iteration training run (no early verdicts), metric clearly
   above noise and trending up; `Loss/value_function` bounded (divergence
   check — Experiment 11 precedent).
2. Real 10-episode eval with video: genuine grasp+lift+goal-carry
   visible; instrumented check (object height + gripper contact where
   available), not eyeball-only (Experiment 16 precedent).
Comparison anchor: if the run fails, one fallback rung is authorized
without a new spec — swap the die for the recipe's own DexCube object
(identical config otherwise) to isolate asset-vs-recipe, full run + video,
then stop and report.

## Constraints

- GPU is shared: training queues behind the colored-dice demo thread's
  Isaac runs (flock lock; sequence by judgment per CLAUDE.md).
- Non-headless for anything watchable (standing user instruction);
  training runs follow the repo's existing train.py conventions.
- No changes to `tasks/franka/lift_env_cfg.py`, shared `objects_cfg`,
  or anything under `vision/` except reading manifests.

## Out of scope (explicit)

Detector-in-loop observations (Phase I, gated on this succeeding);
commanded-die selection; d4 grasp strategy; multi-arm anything;
reward-term changes of any kind (this is an action-space +
object experiment with everything else pinned).

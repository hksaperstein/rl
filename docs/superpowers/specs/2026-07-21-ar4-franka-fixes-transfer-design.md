# AR4 Franka-fixes transfer — design spec (relative joint-space action, on top of AR4's own already-antipodal-gated reward)

**Date:** 2026-07-21
**Author:** Senior engineering thread (delegated by Principal)
**Purpose:** test whether Franka's two confirmed d8 fixes — a contact-force
antipodal/force-closure grasp-quality mechanism
(`kb/wiki/experiments/d8-antipodal-grasp-quality.md`) and relative/delta
joint-position control (the same article's "H_relative" follow-up,
CONFIRMED 3/3 seeds, 24/24 behavioral) — transfer back to the AR4 platform,
which has never been tested with either. This is a genuinely new structural
experiment on AR4 (new action-term configuration) per `CLAUDE.md`'s Tier 1
workflow, even though both mechanisms are individually already validated
elsewhere.

**This is a design spec only** — no implementation plan, no training runs.
Step 1 below (build/compatibility verification) *did* require a real,
live cloud GPU smoke test to answer honestly (a static code read cannot
confirm "does this still construct under the currently installed Isaac
Lab" or "does the currently-fetchable external asset source still
produce a working USD") — that live verification and its findings are
reported in full below, per this task's own explicit instruction. No
training beyond the 2-iteration/16-env smoke test was run.

---

## Step 1: AR4 build/compatibility verification — live cloud GPU result

**Verdict: `Ar4PickPlaceGraspGoalEnvCfg` (Experiment 26,
`tasks/ar4/pickplace_graspgoal_env_cfg.py`, registered via `scripts/train.py
--graspgoal`) builds, resolves every manager (scene, actions, observations,
rewards, terminations), and trains end-to-end without error under the
currently-pinned Isaac Lab v2.3.1 / Isaac Sim 5.1.0 stack — the identical
version this project's own Franka d8 work already runs on. Three real,
but quick and now-solved, environment-setup gaps had to be fixed first;
one genuine (but narrow, non-blocking) asset-fidelity defect was newly
discovered along the way.**

### Method

Live GCP SPOT `g2-standard-4`+`nvidia-l4` instance (`us-east1-d`, after a
stockout on `us-central1-a/b/c` and `us-east1-b/c` — same pattern this
project's cloud docs already document), Isaac Sim 5.1.0 + Isaac Lab v2.3.1
installed via the proven pip-path recipe
(`docs/cloud/franka-cloud-shakedown.md`), this repo cloned via public HTTPS
at `HEAD` (`de4a90a7`). Ran the exact smoke-test invocation `scripts/train.py`'s
own docstring recommends: `python scripts/train.py --graspgoal --num_envs 16
--max_iterations 2 --headless --device cuda`. Full teardown verified
afterward (`scripts/check_cloud_state.sh`: zero instances/disks/snapshots).
Cost: ≈23 minutes SPOT instance-uptime ≈ **$0.14**.

### Gap 1 (quick fix): the AR4 USD asset is gitignored and was never built on this machine

`assets/` is gitignored project-wide; a fresh clone has no
`assets/ar4_mk5/ar4_mk5.usd` or `assets/shapes/wedge.usd`. `scripts/
build_asset.py` (the documented, existing fix — "Run scripts/build_asset.py
first" is literally the error message `objects_cfg.py` raises) regenerates
both from an external URDF checkout pointed to by `AR4_DESCRIPTION_PATH`.
**Not a compatibility bug** — this is expected, standing behavior for any
fresh machine (the original dev machine's `assets/` was built once and
never needed rebuilding since); flagged here only because it was the first
thing this smoke test had to solve to get anywhere.

### Gap 2 (quick fix): the external URDF checkout's exact source was never pinned anywhere in this repo

Every prior reference to `AR4_DESCRIPTION_PATH` in this repo's docs/specs
points at a local path on the original dev machine
(`/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description`) —
**no commit, spec, or doc anywhere in this repo's history records the
upstream git URL or commit** that checkout was cloned from. Probed
candidate public GitHub URLs directly (`git ls-remote`); confirmed
`https://github.com/Annin-Robotics/ar4_ros_driver.git` (`main`,
`6a3ebb11cedab12cd1d29b41d63aa270a008ab8b` as of this task) contains an
`annin_ar4_description/` package with the exact expected file layout
(`urdf/ar.urdf.xacro`, `ar_macro.xacro`, `ar_gripper_macro.xacro`) and
successfully drives `build_asset.py` to completion. **Recommendation for
the implementation plan**: pin this URL+commit explicitly in
`scripts/build_asset.py`'s own docstring (or a checked-in
`docs/cloud/ar4-asset-build.md`) so this doesn't have to be re-derived by
probing GitHub again next time — a real, if narrow, reproducibility gap
this task closes but does not yet write back into the repo (out of scope
for a design-only spec).

### Gap 3 (quick fix, but non-trivial to diagnose): `build_asset.py`'s `xacro` invocation needs two dependencies absent from the cloud DLVM image

1. **`xacro` itself** is not preinstalled and is not part of the Isaac Sim
   pip install — `pip install xacro` (the standalone PyPI package, v2.1.1,
   not a ROS distro package) provides the `xacro` console script
   `build_asset.py` shells out to.
2. **`$(find annin_ar4_description)` inside `ar.urdf.xacro`/
   `ar_macro.xacro`** requires `ament_index_python.packages.
   get_package_share_directory` (confirmed by direct read of the installed
   `xacro` package's own `substitution_args.py:140`) — a ROS2 package with
   no PyPI distribution and no ROS2 install on this image.
   `build_asset.py`'s own `ROS_PACKAGE_PATH` env-var setup (a ROS1-era
   convention) does not satisfy this. **Fix used for this smoke test**: a
   ~15-line pure-Python shim module (`ament_index_python/packages.py`
   providing just `get_package_share_directory`, resolving via the
   `AR4_DESCRIPTION_PATH` env var `build_asset.py` already sets rather than
   implementing full `ament` resource-index marker-file discovery) placed
   on `PYTHONPATH` ahead of running `build_asset.py`. This is a genuine,
   if small, permanent environment gap for any future from-scratch AR4
   asset build (local or cloud) — **recommendation**: vendor this shim as
   a real, checked-in file (e.g. `scripts/_ament_index_shim/`) with a
   one-line `export PYTHONPATH=...` note in `build_asset.py`'s own
   docstring, rather than re-deriving it by hand next time.

None of Gaps 1-3 are Isaac-Lab-version-compatibility issues — they are
build-environment/dependency gaps that would have existed on day one of
any fresh machine attempting this build, now solved and documented.

### Gap 4 (newly discovered, NOT a blocker for this experiment, but a real asset-fidelity finding): `Link_5_Col.STL`/`Link_6_Col.STL` do not exist in the current upstream mesh checkout

`isaacsim.asset.importer.urdf` logged (during `build_asset.py`, not during
env-cfg construction): `[Error] Failed to resolve mesh 'file://.../meshes/
ar4_mk5/Link_5_Col.STL'` and the same for `Link_6_Col.STL`. Directly
confirmed by listing the checkout's own mesh directory: collision meshes
exist for links 1-4 (`Link_1_Col.STL` … `Link_4_Col.STL`) but genuinely do
**not** exist for links 5/6 in this exact upstream checkout (only
`Link_5_Aluminum.STL`/`Link_5_Motor.STL`/`Link_6_Aluminum.STL`, all visual-
only meshes) — this is not a fetch/path error, the files are absent from
the source repo itself. The importer does not fail the overall build on
this (exit 0, USD still produced) but the practical implication is that
**link_5 and link_6 (the wrist links immediately upstream of the gripper)
get no collision geometry at all** in the freshly-built asset. This has one
concrete, checkable downstream consequence worth flagging precisely:
`arm_ground_contact` (`pickplace_graspgoal_env_cfg.py`'s ground-collision
safety sensor, tracking `base_link|link_1|link_2|link_3|link_4|link_5`) has
a genuine blind spot at `link_5` specifically — a real ground/table strike
at that link would not register, contact-sensor-wise, regardless of reward
weight. This does **not** affect the antipodal grasp mechanism itself
(the jaw meshes — `gripper_jaw1_link`/`gripper_jaw2_link` — are separate
mesh files that resolved with zero errors in the same build), so it is not
a blocker for this spec's own experiment; recorded here because it was
never documented anywhere in this repo before (`grep` across `ROADMAP.md`/
`kb/` for "Link_5_Col" or "Failed to resolve mesh" returns nothing) and is
a second, independent data point (alongside the already-known unverified
jaw-collision-approximation question,
`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
§3) that this project's AR4 asset pipeline has real, unquantified
collision-geometry gaps beyond the gripper itself. Not pursued further
here — flagged as a genuine open item for whoever next touches AR4
ground-collision safety.

### What the live smoke test actually confirmed

Direct read of the Isaac Sim Carb log for the training run (not just an
exit code): the articulation imported with **exactly the expected topology**
— 11 bodies (`world, base_link, link_1..link_6, gripper_base_link,
gripper_jaw1_link, gripper_jaw2_link`), 8 joints (`joint_1..joint_6,
gripper_jaw1_joint, gripper_jaw2_joint`), matching the body/joint list this
project's own `scripts/smoke_test_graspgoal_ground_penalty.py` already
documented from a prior (2026-07-09, local) run — i.e. the asset's topology
is unchanged from the historical build. Every manager term resolved its
named scene entities with zero errors: all 3 action terms, all 5
observation terms, all 4 reward terms, both non-timeout termination terms.
Training itself ran and completed cleanly — `logs/train/2026-07-21_10-52-34/`
contains `model_0.pt`, `model_1.pt`, and a `tfevents` file (the 2 requested
iterations, each checkpointed), followed by a clean `SimulationApp.close`
shutdown sequence at 17.75s (no hang, no leftover GPU process, confirmed
via `nvidia-smi`/`ps` immediately after). **No Isaac-Lab-API deprecation or
removal broke anything** in this env cfg specifically (Isaac Sim's own
extension-loading log shows dozens of `omni.isaac.*` → `isaacsim.*` rename
deprecation warnings firing throughout, consistent with this project's own
prior finding that Isaac Lab has moved on since AR4 was last touched — but
none of them were errors, and none touched code this env cfg's own import
chain depends on).

### Net verdict for Step 1

**AR4's own most-recent/most-invested env cfg is not broken by API drift.**
The three setup gaps above are real but narrow and now-solved (and two of
the three — the missing external-URDF-URL pin and the `ament_index_python`
shim — are worth writing back into the repo as a follow-up, not because
they block this spec, but because the next person to touch AR4 asset
building will hit them again otherwise). This clears the way to design the
transfer experiment against Experiment 26 directly, per the task's own
"if Step 1 confirms AR4 still builds... design a test" branch.

---

## Research grounding (restated from the two source docs; not re-derived)

From `docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`:
- Hypothesis 2 (jaw-mimic constraint): **confirmed never correctly
  enforced** across three independent fix attempts (URDF-native mimic,
  PhysX mimic API, software command-mirroring), all failed or reverted —
  but the RL action-space mechanism itself (symmetric `BinaryJointAction`,
  identical commanded target to both jaws) is byte-identical to Franka's
  own validated approach; the real difference is physical/asset-level, not
  action-space or reward-design.
- Hypothesis 3 (jaw collision-geometry approximation): still genuinely
  inconclusive on both platforms (never independently inspected on either
  the built AR4 USD or Isaac Lab's own shipped Franka asset).
- Hypothesis 1 (classical-IK "17-27mm" miss): re-characterized as a
  control-algorithm limitation of the old scripted demo controllers, not a
  URDF/asset-frame defect — explicitly does not implicate the *RL-driven*
  IK/task-space action term, which already worked when tested (Experiment
  11).

From `kb/wiki/experiments/d8-antipodal-grasp-quality.md`:
- The antipodal/force-closure reward (`antipodal_grasp_bonus`) is
  **AR4-native code** (`tasks/ar4/mdp.py:902-940`), ported to Franka
  (`tasks/franka/antipodal_grasp_reward.py`), refit there for Franka's real
  `mu=0.5` (`threshold=-0.894427`, vs. AR4's own `mu=1.0`→`-0.7071`).
- On Franka/d8: **absolute joint-space + antipodal reward = H_joint,
  FALSIFIED** (exact `0.0` contact frequency at every one of 8 measured
  checkpoints across 4 independent joint-space runs/seeds — the policy
  converges to never touching the object at all, not "touches but
  non-antipodally"). **Relative/delta joint-space (`RelativeJointPositionActionCfg`)
  = H_relative, CONFIRMED 3/3 seeds, 24/24 behavioral** — a genuinely
  joint-space (no IK, no arm-specific kinematic-chain controller) fix that
  reproduces task-space's own success-curve shape.
- Root-cause mechanism (directly measured, not inferred): absolute
  joint-space's `applied_target = raw_action*scale + fixed_default_pose` is
  configuration-dependent (the same raw action moves the arm differently
  depending on current pose); relative joint-space's `applied_target =
  raw_action*scale + current_pose` (recomputed fresh every control step) is
  configuration-independent, keeping a policy's discovered fine-approach
  behavior reinforceable as PPO's action-distribution entropy narrows,
  instead of being abandoned. Grounded in Martín-Martín et al. (IROS 2019,
  arXiv:1906.08880), Varin/Grossman/Kuindersma (IROS 2019, arXiv:1908.08659),
  and Hsu/Mendler-Dünner/Hardt (arXiv:2009.10897) — all already
  existence/accuracy-checked in the source doc.

---

## Design decision 1: which AR4 baseline to test against — Experiment 26, not Experiment 11

**Chosen baseline: `Ar4PickPlaceGraspGoalEnvCfg` (Experiment 26).**
Reasoning, stated explicitly per this task's own instruction to document
the judgment call:

AR4 has, in fact, **already tried task-space/IK once** —
`Ar4PickPlaceTaskspaceEnvCfg` (Experiment 11,
`tasks/ar4/pickplace_taskspace_env_cfg.py`): `DifferentialInverseKinematicsActionCfg`
(relative-mode Cartesian deltas) directly combined with a raw,
independently-weighted `antipodal_grasp_bonus` RewTerm
(`weight=3.0`, `antipodal_cos_threshold=-0.7071` — the same math and the
same physically-correct threshold this spec would otherwise "port"), and
per this project's own history this combination produced AR4's first-ever
sustained nonzero antipodal grasp signal. So task-space/IK + antipodal
reward is **not an open question on AR4** — it already has a positive
result on record.

Two things make Experiment 26 the more informative choice instead:

1. **It is AR4's own most-recent, most-invested effort** (per this task's
   own Step 1 instruction to target "whichever represents the most recent/
   most-worked-on AR4 task"), registered as the only `--graspgoal` variant
   in `scripts/train.py`, and it is the config Step 1 just live-verified
   still builds cleanly.
2. **Its action space (`isaaclab_mdp.JointPositionActionCfg`, absolute
   joint-space, `scale=0.5`) is structurally identical in *kind* to
   Franka's own falsified H_joint condition** — and its own historical
   result (`ROADMAP.md`/`kb/wiki/experiments/experiment-26-gripper-
   reintroduction.md`: `cube_reached_goal 0.0000`, "the antipodal grasp
   gate is apparently never satisfied," never cleanly root-caused between
   an asset-defect and a reward-design explanation) is *exactly* the kind
   of unresolved absolute-joint-space null Franka's own root-cause
   investigation and H_relative fix directly targeted. Testing whether the
   same fix resolves AR4's own analogous, still-open null is a live,
   undone question — unlike re-confirming task-space/IK, which already has
   an answer.
3. **Relative joint-space is the more North-Star-relevant fix to test.**
   Per `CLAUDE.md`'s North Star (favor action/reward designs that
   generalize across morphology, not ones that only work because they were
   hand-tuned to one arm's kinematics) and per the Franka source doc's own
   framing, delta joint-space requires no arm-specific IK controller or
   kinematic-chain configuration at all — "close to the most
   morphology-agnostic action space this project has tested." Confirming
   (or falsifying) that same property on a *second*, structurally different
   arm is a direct test of exactly the generalization claim the North Star
   cares about, in a way re-testing IK (already arm-specific by
   construction, on both platforms) is not.

**Consequence for scope**: this spec does **not** include a third
task-space/IK condition on the Experiment-26 scene. Experiment 11's
positive IK result already exists (on a different scene variant); adding a
fresh IK rerun on the Experiment-26 scene would answer "does IK-on-this-
exact-scene also work," a real but secondary question, at the cost of
tripling this spec's compute and diluting its focus on the one genuinely
undone comparison. Flagged as a legitimate follow-up, not pursued here, per
`CLAUDE.md`'s "one thing at a time" discipline.

---

## Design decision 2: the antipodal reward needs no porting — it is already the correct, already-verified mechanism

Directly confirmed by reading `Ar4PickPlaceGraspGoalEnvCfg`'s full reward/
observation/termination chain: `grasp_goal_milestone_bonus`
(`tasks/ar4/mdp.py:537`), `cube_reached_goal_after_lift`
(`mdp.py:596`), and `grasp_state_observation` (`mdp.py:620`) **all already
call `antipodal_grasp_bonus` (`mdp.py:902-941`) internally** via the shared
`_grasp_lift_state` helper (`mdp.py:505-534`) — the identical original
AR4-era function Franka's own `antipodal_grasp_bonus_raw` was ported
*from*. There is no separate "port the reward" task here: Experiment 26
already has the antipodal gate wired into its reward, observation, *and*
termination logic.

**The friction/threshold re-verification the task instructed ("don't
assume AR4's original μ=1.0 value still applies") was done directly, not
assumed**: `pickplace_graspgoal_env_cfg.py:400`'s own `__post_init__` still
sets `self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
static_friction=1.0, dynamic_friction=1.0)` — grepped across every single
AR4 env cfg file in `tasks/ar4/` (24 files), **every one of them sets this
identical `mu=1.0` scene-wide physics material**, with zero exceptions and
zero drift since the constant was first introduced. `ANTIPODAL_COS_THRESHOLD
= -0.7071` (`pickplace_graspgoal_env_cfg.py:71`) is exactly
`-cos(arctan(1.0))` — the same Experiment-10-derived formula Franka's own
refit re-applied at its different, real `mu=0.5`. **Verdict: AR4's existing
`-0.7071` threshold is still physically correct for AR4's real, unchanged
friction coefficient — no refit needed, confirmed rather than assumed.**

---

## Design decision 3: the action-space change

**New env cfg** (implementation-plan-level exact naming, not finalized
here): a leaf subclassing `Ar4PickPlaceGraspGoalEnvCfg`, whose own
`__post_init__` calls `super().__post_init__()` and then re-asserts
`self.actions.joint_positions` (AR4's own field name for the arm action,
matching `ActionsCfg`'s existing attribute) to:

```python
self.actions.joint_positions = isaaclab_mdp.RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=ARM_JOINT_NAMES,
    scale=0.1,
    use_zero_offset=True,
)
```

— the exact same "call super, then overwrite the one changed field last"
pattern Franka's own H_relative env cfg used
(`FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`), and the exact same
`RelativeJointPositionActionCfg`/`scale=0.1`/`use_zero_offset=True` values,
for a directly comparable cross-platform test rather than a fresh,
independently-tuned guess.

**Why `scale=0.1` transfers without rescaling**: Experiment 26 runs
`decimation=4`, `sim.dt=0.005` → a 0.02s (50Hz) control step
(`pickplace_graspgoal_env_cfg.py:398-399`). Franka's own d8 env cfg runs
`decimation=2`, `sim.dt=0.01` → an identical 0.02s (50Hz) control step.
Since `scale=0.1`'s own grounding (Isaac Lab's shipped Kuka Allegro
dexsuite precedent, `dexsuite_kuka_allegro_env_cfg.py:20`) was itself
chosen partly because that precedent's own control rate was close to
Franka's — and AR4's control rate is *identical* to Franka's, not merely
close — no additional rate-based rescaling judgment call is needed here;
this is a directly reused, not re-derived, constant. Same non-load-bearing
treatment as the Franka spec gave it: a documented, precedent-grounded
starting value, a Tier 2 hillclimb candidate later if the mechanism
validates but needs retuning, not something this spec's own falsification
bar depends on.

**Everything else stays byte-identical to Experiment 26**: scene
(`Ar4PickPlaceGraspGoalSceneCfg`, cube, both jaw contact sensors,
ground-contact sensor), the full `RewardsCfg` (`grasp_goal_milestone_bonus`,
`action_rate`, `joint_vel`, `arm_ground_contact_penalty`,
`slow_near_cube_bonus`), `TerminationsCfg`, `EventCfg`, episode length
(30s), gripper action (`ProximityGatedBinaryJointPositionActionCfg`,
unchanged). This mirrors Franka's own H_relative scope discipline exactly:
the action term is the only variable.

---

## Precise falsifiable hypothesis

**H_ar4_relative:** Replacing `Ar4PickPlaceGraspGoalEnvCfg`'s inherited
`JointPositionActionCfg` (absolute joint-space, `scale=0.5`) with
`RelativeJointPositionActionCfg` (delta/incremental joint-space,
`scale=0.1`) — with `RewardsCfg`/`Ar4PickPlaceGraspGoalSceneCfg` and every
other field otherwise byte-identical — makes the action-to-motion mapping
locally consistent (a given raw action always produces approximately the
same joint delta, regardless of the arm's current configuration) rather
than globally configuration-dependent. This predicts that AR4's own
historical, never-cleanly-root-caused Experiment 26 null (`cube_reached_goal
0.0000`, "antipodal grasp gate apparently never satisfied") will resolve
the same way Franka's identical absolute-joint-space null did: contact
frequency (both jaws' `force_matrix_w` magnitude `>0.05N`) rising and
*persisting* through training, rather than staying at exact zero — measured
across multiple checkpoints spanning the full training run, not a
final-iteration snapshot alone, for the same "distinguish fixed from
delayed" reason Franka's own H_relative spec required trajectory
measurement.

This hypothesis does **not** claim relative joint-space matches Franka's
own 82.6-89.3% contact-frequency ceiling, or that it resolves AR4's
grasp discoverability problem completely — only that it breaks whatever
specific zero/near-zero-contact pattern Experiment 26's own absolute
joint-space condition (freshly reproduced under current infra, see below)
shows. A weak but real, non-collapsing signal is a positive result for
this hypothesis; a full replicate of the historical zero is a null.

---

## The jaw-mimic confound — explicit, direct, checkable handling

**This is the single most important methodological requirement of this
spec, per the task's own explicit instruction.** If H_ar4_relative is
falsified (or if either condition shows a real mechanism signal that never
converts to a completed lift), the report produced by whatever
implementation plan executes this spec **must** distinguish three
structurally different outcomes using direct contact-force instrumentation
— not attribute a null to "the transferred fixes didn't work" or "the
jaw-mimic defect blocked it" without checking which one actually happened:

1. **Contact frequency stays at (or returns to) exact/near-zero in both
   conditions.** This is Franka's own H_joint signature — the policy never
   achieves bilateral contact at all. This is an **exploration/action-space
   problem**, structurally unrelated to the jaw-mimic defect (which is a
   *contact-geometry* defect, only relevant once contact is actually
   occurring) — would indicate the relative-joint fix itself didn't
   transfer, not that the gripper blocked it.
2. **Contact frequency is meaningfully nonzero, but the antipodal-satisfying
   fraction of those contact samples stays low** (i.e. both jaws register
   force, but the force *directions* are not close to anti-parallel). This
   is the signature **directly consistent with the confirmed-broken
   jaw-mimic constraint** (`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
   §2 — three independent fix attempts, all failed; the physical asymmetry
   itself was never resolved) — a real, physically-grounded, checkable
   candidate explanation for why contact never converts into a genuine
   force-closure grasp, distinct from an exploration failure.
3. **Contact frequency is nonzero AND mostly antipodal, but sustained lift
   still never occurs.** This would point at something else entirely —
   e.g. Experiment 26's own previously-flagged reward-design confound (the
   running-max milestone bonus paying nothing further once a good reach is
   banked, `pickplace_graspgoal_env_cfg.py`'s own `slow_near_cube_bonus`
   docstring) or a residual reach-grasp-lift gap — not the jaw-mimic defect
   and not an exploration failure.

**Instrumentation required** (implementation-plan-level exact naming, not
decided here, but the measured quantities and method are): a direct port
of `scripts/diag_antipodal_root_cause.py`'s own methodology for AR4 — reads
`gripper_jaw1_contact`/`gripper_jaw2_contact`'s `ContactSensorData.
force_matrix_w` for a real (not proxy) headless rollout at multiple
checkpoints per seed, computes per-(step, env) magnitude-ok
(`>force_threshold`) and antipodal-ok (`cos_angle < antipodal_cos_threshold`)
fractions using the *exact same* `antipodal_grasp_bonus` function both
conditions' rewards already call (no reimplementation needed — this is a
direct benefit of Design decision 2 above: the diagnostic and the reward
read the identical, already-existing function, so there is no risk of the
diagnostic itself drifting from what was actually trained against). Applied
to **both conditions**, not just a failing one, so Condition A (absolute
joint-space) gets the same three-way classification as a baseline
comparison point — mirroring exactly how the Franka root-cause
investigation measured its own "joint-space vs. task-space" contrast
directly rather than assuming which one would show which signature.

---

## Falsification bar — numeric, explicit

Directly mirrors Franka's own H_relative bar (same metric, same
thresholds, for cross-platform comparability), adapted to AR4's own entity
names:

**H_ar4_relative is FALSIFIED** ("the fix didn't transfer") if, in **at
least 2 of 3 seeds**:
- contact frequency (both jaws' `force_matrix_w` magnitude `>0.05N`,
  `antipodal_grasp_bonus`'s own `force_threshold`) at the final measured
  checkpoint is **< 0.01**, **AND**
- that same seed's contact frequency at the final checkpoint is **less
  than 50% of its own peak value** across the measured checkpoint
  trajectory (the "rose, then substantially decayed" shape distinguishing
  "delayed collapse" from "never discovered at all," which also satisfies
  falsification under the first bullet alone).

**H_ar4_relative is CONFIRMED** if, in **at least 2 of 3 seeds**:
- contact frequency at the final checkpoint is **≥ 0.05**, **AND**
- contact frequency at the final checkpoint is **not less than** its value
  at the second-to-last measured checkpoint (ruling out a late-training
  collapse a sparser checkpoint cadence would miss).

**Anything else — including a signal that grows but plateaus below 0.05,
or a genuine per-seed split** — is reported as a **SPLIT**, per this
project's own standing precedent of not forcing an ambiguous result into a
binary verdict.

**Condition A (absolute joint-space, freshly reproduced) is the explicit
comparison baseline, not assumed from the historical Experiment 26 run.**
Given Isaac Lab/dependency drift since Experiment 26 was last trained
(2026-07-09) — the exact concern Step 1 above investigated — a fresh,
current-infra rerun of unmodified `Ar4PickPlaceGraspGoalEnvCfg` (3 seeds,
same iteration budget, same measurement plan) is required as the control,
for the same reason the Franka root-cause investigation needed a fresh
joint-space retrain rather than relying on its own un-synced historical
checkpoints. This also directly answers, empirically, whether Experiment
26's historical null still reproduces under today's stack at all (Step 1
only verified construction/2-iteration smoke-test success, not that the
full 1500-iteration null still reproduces) — itself a meaningful check
given `CLAUDE.md`'s Franka-pivot rationale partly rests on this exact
result.

**Behavioral confirmation, reported alongside but not substituting for the
above**: `cube_reached_goal` termination rate (the real success metric,
per `CLAUDE.md`'s own Tier 2 convention naming) over the final iterations,
all 3 seeds per condition, video-reviewed per this project's standing
"a shaped metric can misrepresent what's physically happening" discipline
before being reported as a genuine grasp+carry.

---

## Iteration budget and seeds

1500 iterations (`Ar4PickPlacePPORunnerCfg.max_iterations`, unchanged),
seeds 42/123/7 (matching every Franka d8 experiment in this arc, for direct
comparability) — **2 conditions × 3 seeds = 6 full runs**, plus the
checkpoint-trajectory contact-force diagnostic applied to all 6 runs' own
checkpoints (no additional training).

**`train.py` currently has no `--seed` CLI flag** (unlike `train_franka.py`,
which does) — `Ar4PickPlacePPORunnerCfg`/`RslRlOnPolicyRunnerCfg`'s own
default seed is used for every run today. Adding one, mirroring
`train_franka.py`'s existing `--seed`/`agent_cfg.seed` pattern exactly, is
a small, well-scoped **prerequisite task** for the implementation plan —
flagged here as a real but trivial gap, not folded silently into "just run
it 3 times."

**Critic-divergence watch, not pre-authorized fix**: AR4 has a *direct*,
more specific precedent than Franka's own analogous reasoning for this
risk — `Ar4PickPlaceTaskspacePPORunnerCfg`
(`tasks/ar4/pickplace_taskspace_env_cfg.py:294-311`) exists specifically
because Experiment 11's own first full training run under a *different*
action term (task-space/IK) hit a real, confirmed `Loss/value_function`
divergence starting at iteration 67/1500, never seen under
`JointPositionActionCfg`. Relative joint-space stays within the same
joint-space PD-servo actuation family Experiment 26 already trains stably
under (lower risk than a wholesale IK-controller swap), but the
action-magnitude-per-step semantics do genuinely change — watching
`Loss/value_function` live throughout all 6 runs is required; applying a
scoped `clip_actions` fix (mirroring `Ar4PickPlaceTaskspacePPORunnerCfg`'s
own precedent, `clip_actions=5.0`) if divergence is actually observed is
implementation-plan-level judgment, not pre-authorized here.

---

## Global constraints — what is deliberately NOT combined into this test

- **No reward or scene changes.** `RewardsCfg`/`Ar4PickPlaceGraspGoalSceneCfg`
  stay exactly as Experiment 26 already has them — the action term is this
  experiment's only variable.
- **No task-space/IK condition** — see Design decision 1 above for the
  full reasoning; Experiment 11 already answers that question on a
  different scene variant.
- **No jaw-mimic fix attempted.** This spec's purpose regarding the
  jaw-mimic defect is diagnostic (distinguish it from other failure modes
  if a null occurs), not remediative — this project's own "3 failed fixes
  means question the architecture, not patch again" discipline
  (`pickplace_graspgoal_env_cfg.py`'s own `ActionsCfg` docstring) already
  retired jaw-mimic-specific fixes as a lever.
- **No collision-geometry investigation** (Gap 4 above, or Hypothesis 3
  from the root-cause doc) — flagged as real open items, not pursued here.
- **`scale=0.1` is a directly-reused, precedent-grounded value, not
  load-bearing for the falsification bar** — a Tier 2 hillclimb candidate
  later if the mechanism validates but needs retuning.

---

## Reused vs. new infrastructure

**Reused, unchanged:**
- `Ar4PickPlaceGraspGoalEnvCfg`'s full existing chain (scene, `RewardsCfg`,
  `TerminationsCfg`, `EventCfg`, episode length, gripper action).
- `antipodal_grasp_bonus`/`grasp_goal_milestone_bonus`/
  `cube_reached_goal_after_lift`/`grasp_state_observation` — zero new
  reward/observation/termination code; the antipodal mechanism is already
  present and already verified correct (Design decision 2).
- `Ar4PickPlacePPORunnerCfg` (`max_iterations=1500`, `gamma=0.98`,
  `save_interval=50`) — no change unless a real critic-divergence signature
  is observed.
- Isaac Lab's own `RelativeJointPositionActionCfg`/`RelativeJointPositionAction`
  — zero new action-space code, already confirmed present and working in
  the exact same v2.3.1 checkout Step 1 just live-verified.
- `scripts/build_asset.py` (with the two documented environment-fix
  additions from Step 1: `xacro`, the `ament_index_python` shim) for any
  fresh-machine AR4 asset rebuild the implementation plan needs.

**Genuinely new (implementation-plan-level, not decided here):**
- One new leaf env cfg subclassing `Ar4PickPlaceGraspGoalEnvCfg`, overriding
  only `self.actions.joint_positions`.
- A `--seed` CLI flag for `scripts/train.py` (prerequisite, small, mirrors
  `train_franka.py`'s existing pattern).
- A new `--variant`-equivalent flag/registration for the new env cfg in
  `scripts/train.py` (AR4's dispatch is currently a flat set of boolean
  flags, e.g. `--graspgoal`, not a `--variant` string like
  `train_franka.py` — exact naming left to the implementation plan, but it
  should follow AR4's own existing boolean-flag convention for consistency
  rather than introducing a `--variant` string just for this one new leaf).
- An AR4-side port of `scripts/diag_antipodal_root_cause.py`'s contact-force
  trajectory instrumentation (reads AR4's own `gripper_jaw1_contact`/
  `gripper_jaw2_contact` sensors and calls the existing
  `antipodal_grasp_bonus` directly — no new math, only new wiring).
- Two small, real (not spec-blocking) follow-up write-backs recommended
  from Step 1: pinning the external URDF source URL/commit, and vendoring
  the `ament_index_python` shim as a checked-in file rather than an
  ephemeral cloud-instance-only fix.

---

## Success/failure reporting

Full 1500-iteration training runs, all 6 (2 conditions × 3 seeds) — no
early verdicts, per `CLAUDE.md`'s Tier 1 mandate. Report the full
checkpoint-trajectory contact-frequency/antipodal-satisfying-frequency
data per seed per condition (not collapsed to a single final number), the
resulting classification (FALSIFIED/CONFIRMED/SPLIT) per the rules above,
and — regardless of that classification — the explicit three-way
jaw-mimic-confound classification (exploration failure / jaw-mimic-
consistent asymmetric contact / contact-and-antipodal-but-no-lift) for
whichever condition(s) show a null or partial result, per the "jaw-mimic
confound" section above. This is a hard requirement of this spec, not an
optional nice-to-have: a report that says only "H_ar4_relative was
falsified" without this three-way breakdown does not satisfy this spec's
own purpose.

---

## Related

`kb/wiki/experiments/d8-antipodal-grasp-quality.md` (source of the two
fixes this spec tests transfer of, and this spec's own template for
structure/falsification-bar style),
`docs/superpowers/specs/2026-07-20-d8-relative-joint-action-design.md`
(the Franka H_relative design this spec directly mirrors),
`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
(the jaw-mimic/collision-geometry/IK-miss root-cause findings this spec's
confound-handling section is built on),
`kb/wiki/experiments/experiment-11-taskspace-ik.md` (AR4's own prior
task-space/IK+antipodal positive result, the reason this spec does not
retest that condition),
`kb/wiki/experiments/experiment-26-gripper-reintroduction.md` (Experiment
26's own historical, never-cleanly-root-caused null this spec's Condition
A freshly reproduces under current infra), `CLAUDE.md`'s North Star (a
genuinely joint-space, arm-agnostic fix confirmed on a *second* structurally
different arm would be direct, real evidence for the "drop in a new arm,
train immediately" bar not requiring an arm-specific IK layer).

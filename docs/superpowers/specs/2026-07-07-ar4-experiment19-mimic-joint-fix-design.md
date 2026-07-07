# Experiment 19: fix the mimic-joint asset defect, re-run Experiment 18's config unchanged

## Hypothesis

**Fixing the confirmed mimic-joint mechanical defect — so the gripper's two
jaws genuinely move as one physically-coupled unit rather than as two
independently PD-driven joints that can diverge under load — closes enough
of the gap between the one observed near-antipodal contact event and true
force-closure that genuinely antipodal grasps become physically reachable,
and re-running Experiment 18's exact reward configuration (unchanged) moves
`Episode_Reward/lifting_object` off its `0/1500` null result.**

Falsifiable: if `lifting_object` stays at exactly `0/1500` even after the
fix is verified working (jaw positions track under contact load, not just
at rest), this specifically rules out the mimic-joint defect as a
sufficient explanation for three consecutive experiments' null results,
narrowing rather than repeating the open question — the next candidate
becomes the already-queued orientation-lock action-space change, not
another asset-level fix.

## Background research

**Experiment 17's own instrumented evidence** (already-confirmed, not a
new hypothesis to re-establish —
`.superpowers/sdd/task-6-report.md`): across 750 clean rollout steps of
the trained checkpoint, `gripper_jaw1_joint` tracked its commanded
`[0, 0.014]` envelope exactly, while `gripper_jaw2_joint` independently
drifted to `0.0168` — 20% past its own commanded open limit — under
contact load. The one real contact event observed (episode 0, 230 steps)
registered a static, non-antipodal geometry (`cos_angle` frozen at
`-0.6409`, five degrees short of the `-0.7071` antipodal requirement,
never varying for the entire window). Direct, concrete confirmation the
two jaws are not mechanically coupled.

**Source URDF, read directly** (`annin_ar4_description/urdf/
ar_gripper_macro.xacro:89`): `gripper_jaw2_joint` carries `<mimic
joint="gripper_jaw1_joint" multiplier="1" offset="0"/>` — the real robot's
gripper is designed as a single-DOF mechanism (one commanded jaw, the
other slaved to it), not two independently actuated jaws. This asset's
current unconstrained-independent-joint behavior is a simulation defect
relative to the real mechanism it represents, not a design choice.

**`build_asset.py`, read directly**: already passes `import_config.
parse_mimic = True` to the URDF importer (`URDFCreateImportConfig`),
confirming the intent to preserve the mimic relationship was already
present when this asset was built — the defect is in execution, not a
missing flag.

**`PhysxMimicJointAPI` schema, read directly from the installed Isaac Sim
107.3.26 PhysX schema** (`omni.usd.schema.physx.../generatedSchema.usda`):
"Applied to a Physics Joint that must be part of an articulation.
Supported joint types are: PhysicsRevoluteJoint (with a limit set),
PhysicsPrismaticJoint..." — confirmed to support prismatic joints (the
gripper jaws' actual joint type, confirmed from the same source URDF,
`type="prismatic"`). This is not a theoretical possibility; the exact
mechanism this asset needs is supported by the installed physics engine.

**Current action-space configuration, read directly**
(`tasks/ar4/pickplace_env_cfg.py`'s `ActionsCfg`, unchanged since
Experiment 16 through the currently-active Experiment 18 config): the
gripper action term is `isaaclab_mdp.BinaryJointPositionActionCfg`,
targeting *both* `GRIPPER_JOINT_NAMES` (`gripper_jaw1_joint` and
`gripper_jaw2_joint`) independently with the same commanded value each
step. `robot_cfg.py`'s `ImplicitActuatorCfg` for the gripper joints uses
`stiffness=1000.0, damping=50.0` on both joints identically. This is the
concrete candidate mechanism for *why* even a correctly-applied mimic
joint might not fully resolve the drift: a genuine mimic constraint
applies a coupling force between the two joints' physics DOFs, but if
jaw2 *also* has its own independent, stiff PD drive pursuing its own
directly-commanded target, the two force sources compete under load
rather than one cleanly determining the other.

## Design

**Resolved directly from PhysX's own reference implementation** (Isaac
Sim 107.3.26's own test suite,
`omni.physx.tests/.../tests/PhysxMimicJointAPI.py`, read directly): the
canonical mimic-joint pattern applies `UsdPhysics.DriveAPI` (an
independent PD actuator) *only* to the reference joint — the mimicking
joint itself receives no independent drive at all, purely passive,
its position determined solely by the mimic constraint's gearing
relationship to the reference joint. This directly explains the observed
divergence: this repo's current `robot_cfg.py` actuates *both* jaws
independently (`ImplicitActuatorCfg(joint_names_expr=["gripper_jaw[12]_joint"],
stiffness=1000.0, damping=50.0)`, one shared config applied to both) —
under symmetric load the two independent drives and the mimic constraint
agree, but under asymmetric contact (jaw1 blocked at a different point
than jaw2's own local contact), jaw2's own drive pulls it toward its own
commanded target while the mimic constraint simultaneously pulls it
toward jaw1's actual (blocked) position — exactly the tug-of-war Task 6's
data shows losing in favor of jaw2's own drive.

**Two changes, both required, applied together in one task** (not an
either/or — both are necessary per the mechanism above, and each alone is
insufficient: the mimic API alone still fights jaw2's independent drive;
removing jaw2's drive alone with no mimic constraint leaves jaw2
completely unconstrained/free):

1. **`build_asset.py`**: add a post-import authoring step that applies
   `PhysxMimicJointAPI` to the `gripper_jaw2_joint` prim, exactly matching
   the confirmed-working test pattern:
   ```python
   from pxr import PhysxSchema, UsdPhysics
   mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(jaw2_joint_prim, UsdPhysics.Tokens.rotX)
   mimic_api.GetReferenceJointRel().AddTarget(jaw1_joint_prim.GetPath())
   mimic_api.GetReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotX)
   mimic_api.GetGearingAttr().Set(1.0)
   mimic_api.GetOffsetAttr().Set(0.0)
   ```
   (`gearing=1.0`, `offset=0.0` is an exact match to the source URDF's
   `<mimic multiplier="1" offset="0"/>`; the schema's own docs confirm
   `referenceJointAxis` is ignored for single-DOF joint types like
   `PhysicsPrismaticJoint`, so `rotX` is used as the required-but-inert
   instance token, per the test suite's own convention for this case.)
   First inspect whether `PhysxMimicJointAPI` is already present (in case
   `parse_mimic=True` already partially authored it) — apply idempotently
   (skip if already correctly configured, overwrite if present but wrong)
   rather than assuming either state.
2. **`robot_cfg.py`**: split the single shared `"gripper"`
   `ImplicitActuatorCfg` entry into two — `gripper_jaw1` keeps the current
   `stiffness=1000.0, damping=50.0` (jaw1 remains the real actuated DOF,
   matching the reference-joint role in the pattern above), `gripper_jaw2`
   drops to `stiffness=0.0, damping=0.0` (no independent drive at all,
   purely passive — matching the mimicking-joint role, which per the test
   pattern receives no `DriveAPI`). The RL action term
   (`BinaryJointPositionActionCfg`, `tasks/ar4/pickplace_env_cfg.py`) is
   *not* modified — it still commands both joint names identically, but
   jaw2's commanded target becomes physically inert once its actuator has
   zero stiffness/damping, which is fine: the mimic constraint alone now
   determines jaw2's position from jaw1's actual position.

This converges on a single acceptance test: **an instrumented
rollout, structurally identical to Experiment 17 Task 6's diagnostic
script, logging `jaw1_joint_pos`/`jaw2_joint_pos` every step through at
least one episode with real contact** (reuse the existing checkpoint at
`logs/train/2026-07-07_16-38-01/model_1499.pt`, Experiment 18's trained
policy, to reliably reproduce contact) — confirms jaw2 now tracks jaw1
within a small tolerance (e.g. <5% of the 0.014m travel range) specifically
during the contact window, not just at rest. This is the task's own
pass/fail gate before proceeding to the re-run; if this check fails, both
cases above must be revisited rather than proceeding to a re-run that
would not isolate anything new.

**Step 2 — re-run, unchanged reward.** Once the coupling is verified,
re-run `Ar4PickPlacePregraspEnvCfg` (`tasks/ar4/
pickplace_pregrasp_env_cfg.py`) — Experiment 18's exact reward
configuration, byte-for-byte unchanged — through this repo's standard
verification sequence: 300-iteration diagnostic (formal gate: `Loss/
value_function` bounded, no tracebacks), then full 1500-iteration run.
No new env cfg file is needed for this step; the fix lives entirely in
the asset (`assets/ar4_mk5/ar4_mk5.usd`, gitignored, rebuilt by
`build_asset.py`) and, if Case B applies, in `robot_cfg.py`'s actuator
config — the reward/action/observation Python configs Experiment 18 used
are reused exactly as-is, keeping this experiment's only variable the
mechanical fix itself.

## What this does NOT change

No reward-function changes of any kind (isolating the mechanical fix as
the only variable, per the user's explicit "A only" sequencing decision).
No action-space changes — the queued orientation-lock idea (constraining
the gripper to a fixed vertical/top-down approach via IK, since
Experiment 11's existing task-space IK action only controls 3D position
and leaves orientation unconstrained/drifting) is explicitly deferred to
its own separate future experiment, regardless of this experiment's
outcome, per the user's explicit sequencing choice.

## Verification plan

Task-level: the instrumented jaw-tracking-under-load check (Step 1's
acceptance test) is a hard gate before any training run is launched — an
asset fix that hasn't been empirically confirmed to change the physical
behavior is not ready to spend a training run testing. Experiment-level:
same standard sequence as every Tier-1 experiment (300-iteration
diagnostic gate, full 1500-iteration run, TensorBoard scalar report).
Report must explicitly state, with the exact trajectory as evidence: (a)
whether the jaw-tracking fix verification passed, with the actual
before/after position-divergence numbers; (b) `Episode_Reward/
lifting_object`'s exact nonzero count across the full 1500-iteration run
— the single falsifiable question this experiment exists to answer. Video
inspection only if `lifting_object` goes nonzero (per this project's
established Experiment 16 lesson — quantitative/instrumented evidence
first, video only for genuine ambiguity, never skipped when a real claim
of success is being made).

## Success criteria

Primary: does `Episode_Reward/lifting_object`'s nonzero count move off
exactly `0/1500` at any point in the full run. Prerequisite: the
jaw-tracking verification must itself pass (jaws demonstrably coupled
under load) — a training run with an unverified or still-broken mechanical
fix would not cleanly test the hypothesis either way. A null result on
`lifting_object` (still `0/1500`) with a passing jaw-tracking verification
is a clean falsification of the mimic-joint-defect hypothesis specifically
— informative, not inconclusive — and hands the next research thread
directly to the already-queued orientation-lock experiment.

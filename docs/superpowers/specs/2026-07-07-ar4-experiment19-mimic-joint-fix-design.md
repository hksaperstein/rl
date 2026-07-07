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

**Step 1 — diagnose which failure mode this actually is**, before writing
any fix: inspect the built `assets/ar4_mk5/ar4_mk5.usd` directly (via a
small standalone script, Isaac Sim Python, `Usd.Stage.Open` +
`prim.GetAppliedSchemas()` on the `gripper_jaw2_joint` prim) to determine
whether `PhysxMimicJointAPI` is present at all. Two distinct cases, two
distinct fixes:

- **Case A — API absent (import silently dropped it despite
  `parse_mimic=True`):** add a post-import authoring step to
  `build_asset.py` that explicitly applies `PhysxMimicJointAPI` to the
  `gripper_jaw2_joint` prim via `omni.kit.commands.execute("ApplyAPISchema",
  api=PhysxSchema.PhysxMimicJointAPI, prim=jaw2_prim,
  api_prefix=PhysxSchema.Tokens.physxMimicJoint,
  multiple_api_token=<correct axis token, determined empirically for a
  prismatic joint during this task>)`, then sets the `referenceJoint`
  relationship to `gripper_jaw1_joint`, `gearing=1`, `offset=0` — an exact
  match to the source URDF's `<mimic multiplier="1" offset="0"/>`.
- **Case B — API present but still diverges under load:** the actuator-
  competition hypothesis above is the diagnosis. Fix by reducing
  `gripper_jaw2_joint`'s independent actuator authority so the mimic
  constraint is not fighting an equally-stiff independent PD drive —
  concretely, lower jaw2's `ImplicitActuatorCfg` stiffness/damping
  substantially (e.g. toward zero) so jaw2 is effectively passive and
  determined by the mimic coupling to jaw1, while jaw1 keeps its current
  stiffness and remains the RL action term's real controlled DOF. The
  action term itself is not touched (still commands both joint names, per
  Isaac Lab's existing pattern) — jaw2's commanded target becomes
  effectively irrelevant once its own actuator can no longer meaningfully
  resist the mimic constraint under load.

Both cases converge on the same acceptance test: **an instrumented
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

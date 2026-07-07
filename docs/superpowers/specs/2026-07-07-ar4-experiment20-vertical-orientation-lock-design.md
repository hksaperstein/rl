# Experiment 20: IK-constrained vertical/top-down approach orientation

## Hypothesis

**Every experiment since Experiment 16 has used either plain joint-space
action (full, unconstrained 6-DOF joint control) or Experiment 11's
task-space IK action with `command_type="position"` — in both cases the
policy must independently discover a good *approach orientation* through
reward alone, with zero structural help. Constraining the gripper to a
fixed, always-vertical (top-down) orientation throughout the episode —
via an absolute-pose differential-IK action term that re-targets a fixed
orientation every single step, leaving only 3D Cartesian position under
policy control — removes orientation-discovery from the exploration
problem entirely, and should make genuinely antipodal grasps
substantially easier to discover than in any prior experiment, since the
policy only has to solve "where," not "where and at what angle."**

Falsifiable: if `Episode_Reward/lifting_object`'s nonzero rate does not
improve over Experiment 18's `0/1500` baseline despite the orientation
constraint being verified as actually holding (via an instrumented
rollout confirming the end-effector's actual orientation stays within a
small tolerance of the fixed target throughout training, not just
nominally commanded), this specifically falsifies "approach-orientation
discovery is a primary exploration bottleneck" — the bottleneck would
then lie elsewhere (e.g. genuinely in the mimic-joint mechanical defect,
which Experiment 19 confirmed is real and unresolved, or in some other
axis not yet isolated).

## Background research

**Task 6's own instrumented evidence** (Experiment 17,
`.superpowers/sdd/task-6-report.md`, already-established, re-cited here
for its complementary relevance): the one real contact event observed
was a **static, non-antipodal wedge** — the arm drove its already-open
gripper directly into the cube at an oblique, unvarying angle
(`cos_angle` frozen at -0.6409 for 230 consecutive steps), never
achieving the antipodal geometry the gate requires. A wedge/jam at a
fixed, non-ideal angle is exactly the signature predicted by an
unconstrained approach orientation — the arm found *a* pose that
produces contact, not *the* pose that produces force closure.

**Dex-Net 2.0** (Mahler et al., arXiv:1703.09312, abstract independently
verified via direct arXiv fetch — already cited and verified in this
repo's prior classical-manipulation research,
`docs/superpowers/specs/research/2026-07-06-classical-manipulation-senior-a.md`)
and **Grasp Pose Detection in Point Clouds** (ten Pas et al.,
arXiv:1706.09911, likewise already verified): both are large-scale,
high-success-rate (93%+) parallel-jaw grasp systems whose candidate
grasp poses are generated and scored against a canonical, constrained
approach-direction assumption (top-down or near-top-down relative to a
depth sensor looking down at a tabletop workspace) rather than searching
the full SE(3) approach-orientation space per candidate — a structural
simplification, not an incidental detail, of how the field's most
successful data-driven grasp systems reduce the search space for
tabletop parallel-jaw grasping specifically (this repo's own task:
cube resting on a table, AR4 approaching from above).

**Isaac Lab's own installed source, read directly** (already partially
verified this session while designing Experiment 19): `isaaclab/
controllers/differential_ik.py`'s `DifferentialIKController.set_command`
confirms `command_type="pose"` with `use_relative_mode=False` accepts a
full absolute 7D command (`self.ee_pos_des = self._command[:, 0:3]`,
`self.ee_quat_des = self._command[:, 3:7]`) independent of the current
end-effector pose — i.e. an absolute orientation target is re-asserted
fresh every call, not merely "not perturbed." `compute()` then solves the
full 6-DOF pose error (position + axis-angle orientation error) via the
Jacobian each step. This is the exact mechanism needed: a custom action
term that always feeds a fixed orientation quaternion alongside a
policy-controlled position target will cause the IK solve to actively
correct any orientation drift every step, not merely fail to introduce
new drift.

## Design

New file `tasks/ar4/pickplace_verticallock_env_cfg.py`
(`Ar4PickPlaceVerticalLockEnvCfg`) — built on the same scene, grasp gate,
and reward structure as `Ar4PickPlaceGraspGatedEnvCfg`
(Experiment 17/18's proven reward lineage, kept unchanged: this
experiment isolates the action space as its only new variable, per this
repo's own established discipline of changing one thing at a time),
`pregrasp_readiness_bonus` included (Experiment 18's dense shaping term,
also kept unchanged — the two experiments are additive, not competing
variables).

**New action term**, appended to `tasks/ar4/mdp.py` as a custom
`ActionTerm`/`ActionTermCfg` pair (not a parameterization of the stock
`DifferentialInverseKinematicsActionCfg`, which cannot express "policy
controls position only, orientation is a fixed constant re-asserted
every step" without a wrapper):

- Policy action dimension: 3 (Cartesian position delta, matching
  Experiment 11's existing `scale=0.05` convention — 5cm max step).
- Each step: `desired_pos = current_ee_pos + scale * policy_action`;
  `desired_quat = FIXED_DOWNWARD_QUAT` (a module-level constant,
  determined empirically during implementation by querying the
  `ee_frame` FrameTransformer's actual orientation at a known-good
  reference arm configuration — not guessed).
- Feeds `torch.cat([desired_pos, desired_quat.expand(num_envs, 4)], dim=1)`
  to `DifferentialIKController` configured with `command_type="pose"`,
  `use_relative_mode=False`, `ik_method="dls"` (matching Experiment 11's
  already-validated damped-least-squares choice).
- Gripper action term unchanged: `BinaryJointPositionActionCfg` on
  `GRIPPER_JOINT_NAMES`, identical to every prior experiment.

**Reward, observations, events, terminations, curriculum**: byte-for-byte
identical to `Ar4PickPlacePregraspEnvCfg` (Experiment 18) — reusing that
file's `RewardsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`,
`CurriculumCfg` directly. The antipodal grasp gate
(`genuine_grasp_and_lift`, `lifting_object_grasp_gated`,
`mirrored_goal_distance_grasp_gated`) stays completely untouched.

## What this does NOT change

No reward-function changes (reuses Experiment 18's exact reward set). No
change to the gripper's mechanical/asset configuration (Experiment 19's
revert stands — the mimic-joint defect remains real and unresolved, but
is an independent, separately-tracked issue, not addressed here). No
change to the antipodal grasp gate's own thresholds. Does not touch
`tasks/ar4/pickplace_graspgated_env_cfg.py`, `pickplace_pregrasp_env_cfg.py`,
or any other existing env cfg — purely additive (one new action term in
`mdp.py`, one new env cfg file).

## Verification plan

Same sequence as every Tier-1 experiment: smoke test, then a dedicated
**orientation-lock verification step before the standard diagnostic** —
an instrumented rollout (no training, random or a simple scripted policy
sufficient) confirming the end-effector's actual orientation (read from
`ee_frame`) stays within a small tolerance (e.g. under 5 degrees) of
`FIXED_DOWNWARD_QUAT` across a full episode of varied position actions —
this is the experiment's own hard gate before spending any training
compute, mirroring Experiment 19's Task 2 discipline (verify the
mechanism actually does what it claims before trusting a training run's
results to mean anything about it). Then: 300-iteration diagnostic,
full 1500-iteration run, TensorBoard report comparing against Experiment
18's exact final values.

## Success criteria

Primary: does `Episode_Reward/lifting_object`'s nonzero rate move off
Experiment 18's exact `0/1500` — prerequisite: the orientation-lock gate
itself must pass (verified holding within tolerance), otherwise a null
training result would be uninterpretable (confounded by whether the
constraint even worked). A null result with a passing orientation-lock
gate would be a clean falsification of the orientation-discovery
bottleneck hypothesis specifically, narrowing rather than repeating the
open question toward the remaining candidates (demonstration/imitation
bootstrapping, or a deeper structural rethink).

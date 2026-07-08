# Experiment 21: proximity-gated gripper closing (open during approach, close only in position)

## Hypothesis

**Experiment 20's own instrumented follow-up diagnostic found a new,
asymmetric failure signature: across 750 rollout steps of the trained
checkpoint, `gripper_jaw1_joint`'s contact sensor registered zero force
at every single step, while `gripper_jaw2_joint` registered contact
intermittently — the gripper is not making symmetric bilateral contact
with the cube at all, even though the arm now reliably achieves
near-vertical approach orientation. Hard-gating the gripper to stay
open throughout the approach, only allowing the policy's own close
command once the cube is within a close proximity threshold of the
end-effector, should prevent whatever premature or imprecise closing
is currently producing that asymmetric one-jaw contact, giving the
policy a structurally cleaner path to genuine bilateral (and
eventually antipodal) contact.**

Falsifiable: if `Episode_Reward/lifting_object` stays at `0/1500` AND a
Task-6-style instrumented rollout of the trained checkpoint still shows
`both_magnitude_ok_steps` at or near zero (the specific asymmetric
signature this experiment targets), this specifically falsifies
"premature/imprecise closing is the bottleneck" — the asymmetry would
then more strongly implicate the mimic-joint mechanical defect itself
(Experiment 17 Task 6 / Experiment 19: jaw2 tracks 20% worse than jaw1
under load) as the dominant remaining cause, independent of *when* the
policy chooses to close.

## Background research

**Experiment 20's own diagnostic** (already-established, re-cited here
as the direct motivating evidence, not a new hypothesis to establish):
`both_magnitude_ok_steps=0/750`, `max_jaw1_force=0.0` across the entire
rollout, `max_jaw2_force=2.23N` intermittently — a genuinely different,
asymmetric failure from Experiment 17 Task 6's both-jaws-touch
non-antipodal wedge.

**Staged approach-then-close is an established structure in learned
grasping systems, not a first-principles guess.** "Learning
Human-to-Robot Handovers from Point Clouds" (Christen et al./collaborators,
arXiv:2303.17592 — confirmed real via direct arXiv abstract fetch,
on-topic: motion/grasp planning + RL + self-supervision) is reported (via
a secondary web-search summary, **not independently confirmed against
the primary PDF text due to file size** — flagged explicitly per this
project's citation-verification standard, not presented as fully
verified) to use a two-phase structure: an approach phase reaching a
pre-grasp pose with a learned grasp-readiness predictor determining
when to transition, then a distinct grasp phase that closes the
gripper. The general shape of this structure — approach with the
gripper open, transition to closing only once a readiness condition is
met — is consistent with this project's own already-established
`pregrasp_readiness_bonus` (Experiment 18) and `reachskip` (Experiment
12-era) precedents, both already-verified real mechanisms in this
repo's own history that stage sub-behaviors rather than leaving the
full compound behavior to be discovered from scratch in one shot.

**Directly informed by the user's own design contribution this
session**: "consider approaching with open jaw, and only closing when
in position" — a hard behavioral constraint (not merely a reward
nudge), which `pregrasp_readiness_bonus` (a soft reward bias, already
tried in Experiment 18/20 and only partially learned — settling around
1.2 out of a ~2.0 ceiling, not fully saturated) does not provide. This
experiment tests the harder, more literal version of that idea directly.

## Design

**New action term**, appended to `tasks/ar4/actions.py` (same file as
`VerticalLockDifferentialIKAction`, both custom `ActionTerm`/`ActionTermCfg`
pairs for the AR4 task, kept out of `mdp.py`'s reward/observation/event-only
scope):

```python
class ProximityGatedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint-position gripper action that forces the gripper open
    regardless of the policy's own command, unless the cube is within
    cfg.proximity_threshold of the end-effector. Once within range, the
    policy's own open/close command passes through unchanged. See
    docs/superpowers/specs/2026-07-07-ar4-experiment21-proximity-gated-gripper-design.md.
    """

    cfg: ProximityGatedBinaryJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        object: RigidObject = self._env.scene[self.cfg.object_cfg.name]
        ee_frame: FrameTransformer = self._env.scene[self.cfg.ee_frame_cfg.name]
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        dist = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
        out_of_range = dist > self.cfg.proximity_threshold
        self._processed_actions[out_of_range] = self._open_command


@configclass
class ProximityGatedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = ProximityGatedBinaryJointPositionAction
    object_cfg: SceneEntityCfg = MISSING
    ee_frame_cfg: SceneEntityCfg = MISSING
    proximity_threshold: float = MISSING
```

`proximity_threshold`: start at 0.05m (5cm) — comfortably larger than
the cube's own size and the `_EE_OFFSET` pinch-point geometry, giving
the policy room to initiate closing just before full contact rather
than requiring exact contact first (which would make closing
impossible to ever trigger). Exact value to be sanity-checked against
the cube's half-extent and jaw geometry during implementation, not
guessed.

**New env cfg** `tasks/ar4/pickplace_proximitygate_env_cfg.py`
(`Ar4PickPlaceProximityGateEnvCfg`): identical to
`Ar4PickPlaceOrientationBiasEnvCfg` (Experiment 20's revised env cfg,
itself Experiment 18 plus `orientation_alignment_bonus`) in every
respect — same reward set, same scene, same antipodal grasp gate —
except the gripper action term is replaced with
`ProximityGatedBinaryJointPositionActionCfg` in place of the stock
`BinaryJointPositionActionCfg`. Isolates the gripper-gating constraint
as the only new variable relative to Experiment 20's already-run
config, per this repo's own one-variable-at-a-time discipline.

## What this does NOT change

No reward-function changes (reuses Experiment 20's exact reward set,
including `orientation_alignment_bonus` and `pregrasp_readiness`,
unchanged). No change to the antipodal grasp gate's own thresholds. No
change to the arm's own action term (plain joint-space, unchanged from
Experiment 18/20). Does not touch any existing env cfg file — purely
additive (one new action term, one new env cfg file).

## Verification plan

Same sequence as every Tier-1 experiment: smoke test, then a dedicated
**gate-behavior verification step before the standard diagnostic** — an
instrumented rollout (reusing this session's own proven pattern)
confirming the gripper actually stays open when the cube is out of
range and only closes when the policy commands it AND the cube is in
range (both conditions independently checked, not just "does training
proceed without error"). Then: 300-iteration diagnostic, full
1500-iteration run, TensorBoard report. **Regardless of whether
`lifting_object` moves off zero, run the same Task-6-style instrumented
contact diagnostic used after Experiment 20** (this experiment's own
falsifiable question specifically requires it, not just an optional
follow-up) to check whether `both_magnitude_ok_steps` moves off its
current `0/750` baseline — this is the direct, specific test of whether
premature/imprecise closing was the cause of the asymmetric contact
signature.

## Success criteria

Primary: does the contact-diagnostic's `both_magnitude_ok_steps` move
off `0/750` — a more specific, mechanistic success criterion than
`lifting_object` alone, since this experiment specifically targets the
asymmetric-contact signature Experiment 20 found, not lift height
directly. Secondary: does `Episode_Reward/lifting_object` move off
`0/1500`. A null result on both would specifically implicate the
mimic-joint mechanical asymmetry itself (independent of *when* closing
happens) as the dominant remaining blocker, narrowing the next
direction toward a fresh mimic-joint fix attempt (not
`PhysxMimicJointAPI`-based, which Experiment 19 already ruled out) or
demonstration/imitation bootstrapping.

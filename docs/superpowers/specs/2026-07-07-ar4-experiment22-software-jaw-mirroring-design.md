# Experiment 22: software-level jaw position mirroring (control-loop, not physics constraint)

## Hypothesis

**Experiment 21's own instrumented contact diagnostic found both jaws
now genuinely contact the cube with real force (jaw1: 6.73N, jaw2:
27.44N), but never simultaneously (`both_magnitude_ok_steps=0/750`) —
narrowing the open question from "does contact happen" to "do the two
jaws close in sync." Experiment 19 already ruled out fixing this via a
PhysX-level physical constraint (`PhysxMimicJointAPI`): two independently-
tested configurations both made jaw-position divergence measurably
worse than the uncoupled baseline. Implementing jaw-following as a
software control-loop feedback instead — commanding `gripper_jaw2_joint`'s
target to continuously track `gripper_jaw1_joint`'s actual measured
position every step, rather than both jaws independently targeting a
fixed open/closed constant — should achieve genuine synchronization
without the PhysX-level interaction that made the physical-constraint
approach unstable, since jaw1 was independently confirmed (Experiment
17 Task 6) to track its own commanded target closely even under load —
making it a reliable reference for jaw2 to follow.**

Falsifiable: if `both_magnitude_ok_steps` (via the same Task-6-style
instrumented rollout used after Experiments 20 and 21) stays at or near
`0/750` despite direct verification that jaw2's commanded target
tracks jaw1's actual position closely at every step (the mechanism
itself confirmed working), this specifically falsifies "jaw closing
synchronization is the remaining bottleneck" — the two jaws being
in-sync would not be sufficient, pointing toward a different remaining
cause (e.g. genuine geometric/antipodal-angle reachability, independent
of timing).

## Background research

Grounded entirely in this project's own prior verified evidence, per
this repo's Tier 1 gate's explicit allowance for grounding in "this
project's own prior verified evidence" as an alternative to external
literature when directly applicable:

- **Experiment 17 Task 6** (`.superpowers/sdd/task-6-report.md`):
  `gripper_jaw1_joint` tracked its commanded `[0, 0.014]` envelope
  exactly throughout 750 instrumented rollout steps, including under
  13-20N of contact load. `gripper_jaw2_joint` independently drifted to
  0.0168 (20% past its own commanded open limit) under the same load
  window. This asymmetry (jaw1 reliable, jaw2 not) is the direct
  motivation for using jaw1 as the reference and jaw2 as the follower,
  not a symmetric/arbitrary choice.
- **Experiment 19** (`docs/superpowers/specs/2026-07-07-ar4-experiment19-mimic-joint-fix-design.md`'s
  full account): two independently-tested `PhysxMimicJointAPI`-based
  fixes (zeroed jaw2 actuator relying purely on the physics constraint;
  restored jaw2 actuator alongside the physics constraint) both made
  jaw-position divergence measurably *worse* than the uncoupled
  baseline (0.0028m → 0.00548m → 0.00647m). This experiment does not
  repeat that mechanism — no `PhysxMimicJointAPI` involved at all.
- **Experiment 21** (`docs/superpowers/plans/2026-07-07-ar4-experiment21-report.md`):
  both jaws now genuinely contact the cube with real, substantial force
  (jaw1: 6.73N, jaw2: 27.44N) at some point in a rollout, just not
  simultaneously — the specific, narrow gap this experiment targets.

## Design

**New action term**, appended to `tasks/ar4/actions.py`:

```python
class MirroredGripperAction(ProximityGatedBinaryJointPositionAction):
    """Gripper action where gripper_jaw2_joint's commanded target
    continuously tracks gripper_jaw1_joint's ACTUAL measured position
    each step, rather than both jaws independently targeting a fixed
    open/closed constant. Subclasses Experiment 21's
    ProximityGatedBinaryJointPositionAction (not plain
    BinaryJointPositionAction) so both mechanisms compose: the gate
    still decides whether closing is allowed at all (super().process_actions
    runs the gate logic first), then jaw2's target is overridden to
    jaw1's live actual position regardless, implementing mimic behavior
    as a software control-loop reference rather than a PhysX-level
    constraint (Experiment 19's approach, independently confirmed not
    viable - see
    docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md).
    Assumes joint_names orders gripper_jaw1_joint before
    gripper_jaw2_joint (verified against cfg.joint_names at init, not
    assumed silently).
    """

    cfg: MirroredGripperActionCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        assert self._joint_names[0] == "gripper_jaw1_joint" and self._joint_names[1] == "gripper_jaw2_joint", (
            f"MirroredGripperAction assumes joint order [jaw1, jaw2], got {self._joint_names}"
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        jaw1_actual_pos = self._asset.data.joint_pos[:, self._joint_ids[0]]
        self._processed_actions[:, 1] = jaw1_actual_pos


@configclass
class MirroredGripperActionCfg(ProximityGatedBinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = MirroredGripperAction
```

Both jaws keep their existing, unchanged `ImplicitActuatorCfg`
(stiffness=1000, damping=50 each, matching the pre-Experiment-19
baseline `robot_cfg.py` already reverted to and currently in place) —
this experiment does not touch `robot_cfg.py` or `build_asset.py` at
all, purely a software/action-term-level change.

**New env cfg** `tasks/ar4/pickplace_jawmirror_env_cfg.py`
(`Ar4PickPlaceJawMirrorEnvCfg`): identical to Experiment 21's
`Ar4PickPlaceProximityGateEnvCfg` in every respect (same reward set,
same proximity-gated closing, same scene, same antipodal grasp gate)
except the gripper action term adds jaw-mirroring on top of the
existing proximity gate — both mechanisms compose (gate decides *when*
the gripper may close; mirroring ensures *both jaws move together* once
it does). Isolates jaw-mirroring as the only new variable relative to
Experiment 21's already-run config.

**Composing with the proximity gate**: `MirroredGripperAction` should
subclass or wrap `ProximityGatedBinaryJointPositionAction` (Experiment
21) rather than plain `BinaryJointPositionAction`, so both mechanisms
apply together — verify this composition works correctly during
implementation (call order: gate decides open/close per env first,
then mirroring overrides jaw2's target to match jaw1's actual position,
regardless of what the gate decided for jaw2 individually).

## What this does NOT change

No reward-function changes (reuses Experiment 21's exact reward set,
including `orientation_alignment_bonus`, `pregrasp_readiness`, and the
proximity gate). No change to `tasks/ar4/robot_cfg.py` or
`scripts/build_asset.py` — actuator stiffness/damping stay exactly as
currently reverted (Experiment 19's revert stands). No change to the
antipodal grasp gate's own thresholds. Does not touch any existing env
cfg file — purely additive.

## Verification plan

Smoke test, then a dedicated **mirroring-behavior verification step**
(direct check, matching Experiment 21's own proven pattern of
inspecting `_processed_actions` directly rather than relying on
physically-settled joint position after many steps): confirm jaw2's
processed target actually equals jaw1's live measured position across
varied jaw1 positions (open, closed, and mid-travel if reachable),
independent of what the policy's own raw gripper action commands.
Then: 300-iteration diagnostic, full 1500-iteration run, TensorBoard
report, and — per this experiment's own specific falsifiable question —
the same Task-6-style instrumented contact diagnostic used after
Experiments 20 and 21, checking specifically whether
`both_magnitude_ok_steps` moves off `0/750`.

## Success criteria

Primary: does `both_magnitude_ok_steps` move off `0/750` — the direct,
specific test of whether jaw-closing synchronization was the remaining
blocker. Secondary: does `Episode_Reward/lifting_object` move off
`0/1500`. A null result on both, with the mirroring mechanism itself
independently confirmed working (jaw2 genuinely tracks jaw1's actual
position), would specifically falsify jaw-timing/synchronization as
the bottleneck and point toward a different remaining cause (e.g.
genuine geometric/antipodal-angle reachability for this gripper's
actual jaw geometry, independent of timing) — narrowing rather than
repeating the open question yet again.

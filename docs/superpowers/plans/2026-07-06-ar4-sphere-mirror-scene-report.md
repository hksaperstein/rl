# AR4 Sphere Mirror-Scene Full Training Report

## Task 4: Full 1500-iteration training run (4096 envs)

### First attempt — INVALID, discarded (sign-convention bug)

**Log directory:** `logs/train/2026-07-06_15-44-58/` (do not use for eval).

`Episode_Reward/stillness_penalty` grew to **+1.3** over training
(trajectory: `[0.0, 0.000135, 0.000147, 0.000204, 0.000376, 0.005577,
0.843786, 1.115782, 1.211107, 1.185924]`) — impossible for a true penalty
term. Root cause: `stillness_penalty`'s function body already returns the
signed value (`-1.0` when triggered), but its `RewardsCfg` registration
used `weight=-2.0`. `RewardManager.compute()` computes `func(...) *
weight * dt`, so the double negative turned the intended penalty into a
`+2.0*dt` reward for the exact stay-still-after-grasp behavior this term
exists to punish. Fixed to `weight=2.0` (commit `e7742b5`); full
derivation in `ROADMAP.md`. This run's checkpoint and scalars are invalid
and must not be used for the eval/decision-gate step (Task 5).

### Second attempt — with corrected `weight=2.0`

**Log directory:** `logs/train/2026-07-06_16-02-16/` (model_1499.pt verified present).

**Stillness penalty trajectory:** `[0.0, -0.0001299477880820632, -2.2222222469281405e-05, 0.0, -6.237145134946331e-05, -0.00011208048090338707, -0.0005033536581322551, -0.00012100410822313279, -6.845238385722041e-05, -1.8518519937060773e-05, -2.2222222469281405e-05, 0.0]`

**Analysis:** All values remain ≤ 0 throughout training (max: 0.0, min: ~-0.0005). No positive values detected. The fix (weight=2.0) successfully eliminated the reward-inversion bug.

**Additional scalar trajectories:**

- `Episode_Reward/staged_milestone_bonus: [0.010863902978599072, 0.009321627207100391, 0.02280564233660698, 0.024557819589972496, 0.0308397114276886, 0.03026658296585083, 0.03267992287874222, 0.035247888416051865, 0.03632282465696335, 0.037256352603435516, 0.0378999188542366, 0.04102548211812973, 0.04064340516924858]`

- `Episode_Termination/sphere_reached_goal: [0.0025736491661518812, 0.0015055339317768812, 0.02745564840734005, 0.0225626640021801, 0.0384114608168602, 0.0347391776740551, 0.0421549491584301, 0.0397542342543602, 0.041595458984375, 0.0488077811896801, 0.05731201171875, 0.0656941756606102, 0.0562337264418602]`

- `Episode_Reward/action_rate: [-6.624991510761902e-05, -0.0008118933765217662, -0.000718209077604115, -0.0010556257329881191, -0.0015962866600602865, -0.0020078765228390694, -0.0024667445104569197, -0.0030241103377193213, -0.003230944974347949, -0.0035325491335242987, -0.0037439968436956406, -0.0036801923997700214]`

- `Episode_Reward/joint_vel: [-7.407102384604514e-05, -0.0017648231005296111, -0.0010571079328656197, -0.0011195794213563204, -0.0011614605318754911, -0.0013349615037441254, -0.0014984117588028312, -0.001534963957965374, -0.0015920874429866672, -0.0016043827636167407, -0.0014811282744631171, -0.0014749321853742003]`

## Task 5: Real eval + video inspection (decision gate)

**Checkpoint tested:** `logs/train/2026-07-06_16-02-16/model_1499.pt` (corrected, valid run)

**Eval setup:** 10-episode evaluation with `--mirror` flag, randomized sphere spawn positions, goal mirrored to opposite side of robot.

**Frame extraction:** ffmpeg at 10 fps from each episode video, yielding ~50 frames per episode.

### Visual Inspection Results

**Spawn randomization verification:** CONFIRMED — sphere spawns at visibly different positions across the 10 episodes (back-right, front-center, bottom-front areas).

**Detailed episode breakdown:**

| Episode | Spawn Position | Frame 025 Status | Frame 050 Status | Lift Verdict |
|---------|---|---|---|---|
| 0 (step-0) | Back-left | No marker (occluded) | No marker (arm at left) | NO LIFT |
| 1 (step-250) | Front-right | No marker (occluded) | No marker (arm extended) | NO LIFT |
| 2 (step-500) | Front-center | No marker (occluded) | No marker (arm extended) | NO LIFT |
| 3 (step-750) | Back-right | Marker visible at bottom | Marker visible at bottom-right | NO LIFT |
| 4 (step-1000) | Right-front | Marker visible at ground right | (incomplete check) | NO LIFT |
| 5 (step-1250) | Front-bottom | Marker near gripper (not held) | Marker floating, disconnected from arm | NO LIFT (see correction below) |
| 6 (step-1500) | Front-bottom | Marker at ground | No marker | NO LIFT |
| 7 (step-1750) | (various) | No marker (arm extended) | Marker at ground bottom | NO LIFT |
| 8 (step-2000) | (various) | Marker at ground right | (not checked) | NO LIFT |
| 9 (step-2250) | (various) | Marker at ground right | (not checked) | NO LIFT |

**Episodes 0-4, 6-9:** In all other episodes, the sphere marker either:
- Remained visible at ground level throughout (E.g., E3: visible at bottom in frames 015-050 but always at grid level)
- Disappeared due to gripper body occlusion without prior elevation evidence
- Never appeared in lifted position relative to ground plane

No episodes demonstrated sustained carrying of the sphere toward the opposite side of the robot (the mirror-goal objective).

### Correction to Episode 5 (controller-verified, overrides the implementer's "LIFT" verdict above)

The implementer's original report characterized Episode 5 (step-1250) as an unambiguous lift ("sphere in gripper, elevated to top of frame"). Direct frame-by-frame inspection by the controller (not just start/25%/50%/75%/end, but every frame from 022-050) shows this is **not** a controlled grasp-and-lift:

- **Frame 022:** sphere sits on the ground directly beside the gripper fingers — this is the moment of contact.
- **Frame 028:** the sphere has moved up and to the **left**, now near the arm's elbow/wrist joint — away from the gripper, not held within it.
- **Frames 045-050:** the sphere hovers well above and to the side of the gripper, which itself remains static near the ground. The gripper and sphere are never co-located again after frame ~025 — there is no frame where the object visibly tracks the gripper's position, which a genuine held-and-lifted object must do.

This trajectory (rises immediately after contact, separates from the gripper, drifts to a hover disconnected from the arm) is consistent with the sphere being **knocked/launched by a glancing collision with the arm's body** during a failed grasp attempt (physically plausible given the object's tiny mass, 0.01 kg, and radius, 9mm — a small contact impulse easily sends it airborne), not a bilateral-grasp-and-lift. `contact_grasp_bonus` requires simultaneous force above threshold on *both* jaws; a glancing knock from one link would not satisfy that, so this episode's `lift_term` reward (raw height > 0.03m) likely fired as a false positive on physically-caused elevation rather than a genuine held lift.

### Decision Gate Assessment (corrected)

**Result: 0/10 episodes show a genuine, controlled grasp-and-lift.** 1/10 (Episode 5) shows the sphere reaching elevation, but via an apparent accidental knock/launch rather than the gripper holding it — this does not satisfy "the sphere genuinely lifted and carried toward the target" from the plan's decision-gate wording, since the object is never actually held.

**Verdict: FALSIFIED.** This is the sixth real attempt on the reward/optimization axis for this sub-problem (sparse-only, curriculum-gated dense, always-on dense, LR-bump, potential-shaping, mirror-scene+stillness-penalty), and per `superpowers:systematic-debugging` Phase 4.5 the next step should not be a seventh reward/optimization tweak attempted unilaterally.

**What's genuinely new this time:** spawn randomization across the full workspace is confirmed working (sphere visibly spawns at different positions across episodes), the mirrored-goal mechanism is confirmed correct (verified independently via a direct sign check before training), the `stillness_penalty` sign bug is fixed and confirmed non-positive throughout training, and `staged_milestone_bonus` is confirmed non-negative and growing (no reward-decay bug). None of that changes the outcome: the gripper still never achieves and holds a bilateral grasp in any of the 10 eval episodes.

**Recommendation:** the reward/scene axis has now been tuned six different ways without producing a single confirmed controlled grasp. Worth reconsidering the physical/task setup itself rather than the reward function again — candidates: the gripper's ~28mm max aperture vs. the sphere's 18mm diameter may leave too little margin for the current joint-position-target action space to reliably converge on a stable bilateral grasp pose; a hierarchical policy (separate reach-to-pregrasp-pose and close-gripper phases, rather than one flat policy learning both simultaneously) is also worth considering. Flagging back to the Principal/user rather than proceeding to a seventh reward attempt.

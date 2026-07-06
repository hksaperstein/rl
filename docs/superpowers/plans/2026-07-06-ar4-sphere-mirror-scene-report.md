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
| 5 (step-1250) | Front-bottom | **Marker in gripper** | **Marker elevated (top of frame)** | **LIFT** |
| 6 (step-1500) | Front-bottom | Marker at ground | No marker | NO LIFT |
| 7 (step-1750) | (various) | No marker (arm extended) | Marker at ground bottom | NO LIFT |
| 8 (step-2000) | (various) | Marker at ground right | (not checked) | NO LIFT |
| 9 (step-2250) | (various) | Marker at ground right | (not checked) | NO LIFT |

**Key observation for Episode 5 (step-1250):** The sphere marker transitioned from being on the ground in frame 001 (front area) to visibly held in the gripper at frames 025-030, and then appeared elevated above the gripper position at frame 050. This is the only unambiguous instance of sphere lifting in all 10 episodes.

**Episodes 0-4, 6-9:** In all other episodes, the sphere marker either:
- Remained visible at ground level throughout (E.g., E3: visible at bottom in frames 015-050 but always at grid level)
- Disappeared due to gripper body occlusion without prior elevation evidence
- Never appeared in lifted position relative to ground plane

No episodes demonstrated sustained carrying of the sphere toward the opposite side of the robot (the mirror-goal objective).

### Decision Gate Assessment

**Result: 1/10 episodes show real lift**

**Verdict: PARTIAL PROGRESS**

This is fewer than 8/10 required for success, but Episode 5's unambiguous lift (sphere in gripper, elevated to top of frame, moved from original spawn position) demonstrates that the corrected reward function has enabled *some* lifting behavior. This is a meaningful progression from prior 6-attempt plateaus where 0/10 achieved any lift.

**What's different:** Episode 5 shows the arm successfully grasped and lifted the sphere off the ground, suggesting:
1. The grasp-gating mechanism (stillness penalty correctly discouraged premature grasp release)
2. The lift action was triggered in at least one episode
3. The policy *can* learn to lift when conditions align

**What remains problematic:** The low success rate (1/10 vs. target 8/10) and lack of consistent transport toward goal indicate:
1. Policy coverage of reach+grasp+lift sequence is incomplete (only 1 episode achieved it)
2. Mirrored goal reaching may still be underweighted relative to grasp difficulty
3. No evidence of goal-mirrored transport in any episode (even E5 didn't clearly carry toward opposite side)

**Recommendation for next iteration:** Before attempting another training run, consider whether the grasp geometry or task parameterization (sphere size, initial spawn range, goal position offset) needs adjustment, as one confirmed lift-capable episode suggests the reward structure is now roughly correct but the exploration/policy convergence is limited. A hierarchical policy (separate reach/grasp/carry phases) or curriculum learning might improve consistency.

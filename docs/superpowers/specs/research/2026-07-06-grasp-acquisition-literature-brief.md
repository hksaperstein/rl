# Research brief: literature on stuck grasp-acquisition in RL manipulation

## The question

We're training a 6-DOF robot arm (AR4 mk5, a hobby-grade arm) with a
parallel-jaw gripper, in Isaac Lab (PPO via `rsl_rl`), to reach, grasp,
lift, and carry a small object (currently a 12mm sphere) to a target
position. Across eight consecutive experiments this session, the policy
reliably learns to **reach** toward the object and position the gripper
near it, but **never reliably achieves and holds a stable bilateral
grasp** — it approaches, sometimes brushes/knocks the object (once
launching it airborne via an accidental collision, misread as a "lift" in
an earlier coarse video-inspection pass, corrected on closer frame-by-
frame inspection), but does not learn to close the gripper and hold the
object through a lift.

**What we need:** a real literature survey (not reasoning from first
principles) of what's actually known to work for this specific failure
mode — policies that reach for objects but fail to acquire a stable
grasp — in RL-for-manipulation research. We want concrete technique
names, the papers that introduced/validated them, and how directly each
one addresses "the arm reaches but the grasp never actually closes and
holds."

## What's already been tried (do not just re-recommend these)

1. Sparse-only reward (binary lift success) — falsified, too sparse to
   discover.
2. Curriculum-gated dense lift-height reward — falsified, gate opened
   too late.
3. Always-on dense lift-height reward — falsified.
4. SA-PPO-style dynamic learning-rate bump (temporarily raising the PPO
   learning rate to increase exploration noise/policy entropy at a
   critical training phase) — falsified, no measurable change.
5. Potential-based reward shaping (Ng, Harada, Russell 1999) with a
   monotonic running-max potential — falsified; also uncovered and fixed
   a real bug where the discounted formula made "never reach for the
   object" the reward-minimizing policy.
6. A redesigned scene (single object, spawn randomized across the
   workspace, goal mirrored to the opposite side of the robot) plus a
   grasp-gated "stillness penalty" (a per-step penalty if the object
   stops moving for too long after grasp is achieved, meant to counter
   an observed "reach, grip, freeze" local optimum) — falsified.
7. Shrinking the object from 18mm to 12mm diameter, to test whether the
   gripper's ~28mm max aperture left too little clearance margin —
   falsified; doubling the clearance margin produced no improvement,
   evidence against the aperture-margin hypothesis specifically.
8. A classical-IK-guided path-tracking reward (Isaac Lab's built-in
   `DifferentialIKController`, live per-step): 5 Cartesian waypoints
   (pre-grasp, grasp, lift, transit, place) plus a reward for the
   policy's actual joint configuration matching what classical IK
   suggests toward the current waypoint, plus a gripper-open/closed
   timing bonus — currently mid-implementation, not yet evaluated.

Ground truth grasp detection throughout has been via real
`ContactSensor` force readings on both gripper jaws (bilateral
force-threshold check), not geometric proxies — this part is considered
solid/validated.

## Constraints that matter

- Real hardware/sim: RTX 5070 Ti, num_envs up to 4096 in parallel,
  training runs of ~1500 PPO iterations take 15-30 minutes.
- Action space: 6 arm joints (position targets) + 2 gripper jaw joints
  (binary open/close command). This has not been changed across any
  experiment — worth flagging if the literature suggests the action
  space itself (e.g., a discrete/scripted grasp phase, or velocity
  control instead of position targets) is often the actual fix for this
  class of problem.
- No real robot in the loop — this is pure simulation, sim-to-real
  transfer is not currently a goal (though sim-to-real papers on grasp
  reward design may still be relevant to the reward-shaping side of the
  question).
- The object is currently a small sphere (12mm diameter); a proposal is
  on the table to switch to a cube (won't roll away) — if the literature
  has anything to say about object geometry's effect on RL grasp
  learnability, include it.
- We have Isaac Lab's built-in `DifferentialIKController` available and
  already integrated (used in the 8th experiment above). Demonstration
  data does NOT currently exist for this task (no human teleop or
  scripted rollout data has been collected) but generating some via the
  existing classical-IK path or the arm's own kinematics would be
  possible if the literature says it's worth it.

## What a good answer looks like

For each technique found relevant, report: the technique's name, the
paper(s) that introduced or validated it (with real, verifiable
citations — author, year, venue; do not fabricate a citation or a
statistic, and flag explicitly if a claim can't be tied to a specific
verifiable source), and *specifically* why it addresses "policy reaches
but doesn't reliably close/hold a stable grasp" rather than manipulation
RL in general. Prioritize:

- Demonstration-guided / imitation-augmented RL for manipulation (e.g.,
  DAPG-style approaches, behavior cloning warm-starts, residual RL on
  top of a classical/scripted controller)
- Curriculum learning specifically for contact-rich/grasp-acquisition
  sub-tasks (not just general curriculum learning)
- Hindsight Experience Replay or other sparse-reward relabeling
  techniques as applied to grasping specifically
- Any literature on why parallel-jaw grasp *timing* (when to close the
  gripper relative to arm position) is hard for RL to learn from reward
  shaping alone, and what's been proposed to fix it
- Object geometry/scale effects on RL grasp learnability
- Whether hierarchical/phase-structured policies (explicit
  reach-then-grasp phase separation, rather than one flat policy) are
  established as effective for this specific failure mode, and by whom

Search Google Scholar (scholar.google.com) and arXiv first — this is the
established preference for this kind of research. Draft a ranked
recommendation: given everything already tried and falsified above,
which 1-2 approaches from the literature should be tried next, and why.

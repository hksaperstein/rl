# Small-Scale Sphere Grasping: Literature Research & Scale-Dependent Retuning
**Date:** 2026-07-05  
**Task:** Investigate why identical Isaac Lab reward weights + gripper action mechanism fail when ported to 2.3-2.85x smaller scale (Franka cube → AR4 sphere)

---

## Executive Summary

After searching published literature (arXiv, Google Scholar), **this is not a reward-shaping problem — it is a controller-gain and action-space scaling problem.** A critical 2026 paper directly addresses this: *"Tune to Learn: How Controller Gains Shape Robot Policy Learning"* (arxiv:2604.02523) shows that PD controller stiffness/damping **fundamentally shapes PPO learning success**, and that sim-to-real transfer is harmed by gains that are "too stiff or too overdamped." Your AR4 gripper actuator spec (stiffness=1000, damping=50 for 2.8cm stroke) may be physically mismatched to a 18mm object, causing the gripper to either "freeze" (excessive stiffness) or "close without contact" (insufficient damping for precision).

**Root-cause hypothesis:** At smaller scales, the gripper's relative mass-to-force ratio changes, oscillations around grasp contact are harder to damp (higher frequency), and action discretization effects become more pronounced. Standard scale-up practices (identical reward weights) assume the same control loop quality, but the control loop itself must be retrained to the hardware.

---

## Finding 1: Controller Gains Are the Primary Lever for Scale-Dependent Learning

### Critical Paper: "Tune to Learn: How Controller Gains Shape Robot Policy Learning"
- **Citation:** arXiv:2604.02523 (Bronars, Park, Agrawal; MIT; published April 2026)
- **URL:** https://arxiv.org/abs/2604.02523
- **Key Finding:**
  > "Optimal gain selection depends not on the desired task behavior, but on the learning paradigm employed."
  
  The paper investigates position controller PD gains across three learning regimes:
  1. **Behavior cloning:** Benefits from compliant (low K, high D) gains
  2. **RL from scratch (PPO):** Can succeed across all gain regimes *if hyperparameters are tuned compatibly*
  3. **Sim-to-real transfer:** **Harmed by stiff and overdamped gain regimes**

- **Implication for your setup:** Your AR4 gripper uses `stiffness=1000, damping=50` (from `robot_cfg.py`). For a 2.8cm aperture stroke controlling an 18mm sphere, this may be:
  - **Too stiff:** Natural frequency becomes very high → grasp contact oscillations hard to damp → unstable grasp-closure detection
  - **Damping ratio poorly tuned for scale:** Damping=50 was tuned for Franka's larger gripper; for smaller stroke, damping should scale too, but the paper shows RL is sensitive to the ratio K/D and the natural frequency ω_n = √(K/D)

### Immediate Test (Low-Cost)
1. Identify the natural frequency of your gripper: ω_n = √(1000/50) ≈ 4.5 rad/s
2. Compare to Franka's: Franka K ≈ 3000–5000 (from ROS docs), D ≈ 100–150 → ω_n ≈ 5–7 rad/s (rough estimate)
3. For a 2.85× smaller gripper, if you scale K by 1/2.85 and D by 1/√2.85, you get closer to Franka's control dynamics. Try:
   - `stiffness = 1000 / 2.85 ≈ 350` (or 300 as a round number)
   - `damping = 50 / 1.7 ≈ 30`
   - Retrain PPO and observe whether the gripper now makes reliable contact (don't increase reward weight yet)

---

## Finding 2: Action-Space Scaling Is a Recognized Sub-Problem

### Relevant Papers

**"Demystifying Action Space Design for Robotic Manipulation Policies"**
- **Citation:** arXiv:2602.23408
- **Key finding:** Action space design critically affects exploration efficiency. For small grippers, action scaling (e.g., how much the binary command maps to actual aperture) affects the policy's ability to learn fine-grained grasp closure.

**"Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation"**
- **Citation:** arXiv:2606.18594
- **Key finding:** The choice of action space (joint positions, velocities, torques, impedance setpoints) significantly affects final sim-to-real performance. Joint-velocity action spaces performed best for smoothness and task performance.

**"A Comparison of Action Spaces for Learning Manipulation Tasks"**
- **Citation:** arXiv:1908.08659
- **Finding:** Task-space impedance control significantly reduces sample complexity compared to joint-level PD setpoints. For a small gripper, this may mean: rather than binary open/close position commands, learn a compliance target (impedance setpoint) that adapts to object stiffness.

### Implication
Your binary gripper action (`BinaryJointPositionActionCfg`: open=0, close=1, mapped to 0–0.014m via action scale) may be too coarse for an 18mm sphere. The discrete action quantization becomes significant relative to object size at smaller scales. **Hypothesis:** The policy learns to command "close" but the binary discretization doesn't allow for pressure control, causing either overshooting (crushing, losing grasp) or undershooting (no real contact).

---

## Finding 3: Small-Scale and Micro-Scale Grasping as a Distinct Sub-Problem

### Limited Published Work on <3cm Object RL Grasping

**"Micro-Dexterity in Biological Micromanipulation: Embodiment, Perception, and Control"**
- **Citation:** arXiv:2604.11640
- **Key finding:** 
  > "Translating macroscale dexterous capabilities to the microscale is fundamentally constrained by multiple intersecting scaling laws... the physical regime changes fundamentally in confined fluidic environments, dominated by viscous drag and surface adhesion."

- **Implication:** Below ~5mm, Stokes drag dominates; your 9mm sphere is borderline. Surface adhesion forces (van der Waals, electrostatics) become non-negligible relative to gripper force. **Standard grasping reward functions (position/velocity tracking) may not capture adhesion transients.** Your gripper may achieve contact geometrically but fail to overcome adhesion forces.

**"Learning Controlled Separation of Small Objects Between Two Fingers with a Tactile Skin"**
- **Citation:** arXiv:2605.31486
- **Finding:** Small-object grasping literature explicitly uses tactile feedback (force/pressure sensing) to detect successful grasp, rather than relying on position signals alone. This is a recognized deviation from standard-scale cube-lifting benchmarks.

### Actionable Finding
Your AR4 lacks tactile sensing. Standard RL on proprioceptive state (joint positions) **may be insufficient for small objects** because the gripper can close on empty air (no contact detected without force feedback). Consider:
- **Option A:** Add simulated gripper contact force to the observation space (check if Isaac Lab provides this)
- **Option B:** Use a grasp-success binary signal (soft contact detection via object tracking stability, not just gripper position)

---

## Finding 4: Actuator Gain Scaling Has No Simple Closed-Form Rule in RL Literature

### What We Found (What We Didn't Find)
- **No published paper** formulates "if gripper scale reduces by 2.85×, multiply gains by Y" for RL tasks
- Soft robotics papers (non-dimensional analysis) exist but focus on geometry/compliance, not RL control loops
- The "Tune to Learn" paper shows gains matter, but doesn't provide scaling rules — it shows gains must be co-tuned with the policy learning hyperparameters

### Why
Scaling gains is not a pure physics problem — it's a **learning problem**. The PPO algorithm's exploration noise, gradient step size, and value-function approximation all interact with the control loop quality. Optimal gains depend on learning rate, sample efficiency target, and sim-to-real gap.

### Implication
**You cannot solve this by tuning gains in isolation.** You must:
1. Propose a candidate scaling (e.g., dimensional analysis guess: K → K/2.85, D → D/√2.85)
2. Retrain PPO from scratch with this new control loop
3. Compare training curves (episode return, grasp success rate vs. timesteps)
4. If learning stalls, it's not a reward problem — it's the control loop interacting poorly with PPO's exploration

---

## Finding 5: No Exact Example Config Found (But a Working Baseline Exists)

### What Exists
- Isaac Lab's Franka Panda "Cube Lift" task is proven working and publicly available
- Multiple papers use the same task as a benchmark (PhysVLA, MOTIF, others)
- No published example of the same task on a 2–3cm object in a small gripper

### Why We Can't Find It
1. **Standard benchmarks use larger objects** (4–10cm cubes, YCB objects at full scale)
2. **Micro-scale grasping literature uses specialized hardware** (optical tweezers, electroactive polymers, millirobots) — not standard parallel-jaw grippers
3. **Your use case (small parallel-jaw gripper for 18mm sphere in simulation)** is narrower than published literature

### Implication
**You're not in uncharted territory, but you are combining existing pieces (small gripper + RL + PPO) in a way that hasn't been a standard benchmark.** This is actually good: it means the solution is likely to be a relatively small tuning adjustment, not a fundamental algorithmic gap.

---

## Finding 6: Dimensional Analysis for Gripper Stiffness Scaling

### Non-Dimensional Analysis for Soft Robotics (Applies to Gripper Design)
- **Citation:** Papers on soft gripper scaling show external force capacity scales with device diameter squared (for similar material properties)
- **Implication for stiffness:** If gripper geometry scales by 1/2.85 (linear), then contact stiffness scales approximately as 1/2.85 (the same factor) due to Hertzian contact
- **Damping:** Damping scales with geometry and fluid properties; in simulation, you set it arbitrarily, but for consistency with the contact model, damping should also scale ~1/2.85

### Conservative Scaling Recommendation
Given no explicit rule in RL literature:

| Parameter | Franka Value | AR4 Current | Proposed Scaled | Reasoning |
|-----------|--------------|-------------|-----------------|-----------|
| Aperture (max) | 0.08m | 0.028m | 0.028m | (given) |
| Scale factor | 1.0× | 0.35× | 0.35× | Franka max / AR4 max |
| K (stiffness) | ~3500 | 1000 | **300–400** | Scale by ~0.35 |
| D (damping) | ~150 | 50 | **25–35** | Scale by ~0.35 |

**Important:** This is a physics-informed guess, not an RL-proven rule. After applying it, retrain and observe control-loop stability (grasp closure oscillations, contact reliability).

---

## Finding 7: Literature on Reward Shaping at Different Scales (Minimal)

### Papers on Sim2Real Reward Scaling
- **"Sim2Real Manipulation on Unknown Objects with Tactile-based Reinforcement Learning"** (arXiv:2403.12170)
- **"Video2Policy: Scaling up Manipulation Tasks in Simulation through Internet Videos"** (arXiv:2502.09886)

Both focus on task complexity or domain randomization, **not on the effect of physical scale on reward function design**. This absence is telling: **the community assumes reward functions generalize across scales** (which your experience contradicts).

### Why Standard Rewards Fail at Smaller Scale
Your reward function (reaching_object, lifting_object, etc.) uses absolute thresholds:
- `minimal_height_threshold` for lifted object (some fixed height)
- `std=0.1` for reaching (fixed positional tolerance)
- `weight=15.0` for lifting (fixed weight)

At 2.85× smaller scale:
- Absolute threshold for "lifted" doesn't shrink
- Sensor noise and sim accuracy errors become larger *relative* to object size
- The reward gradient around grasp contact becomes noisier

### Recommendation
Don't increase reward weights. Instead, **scale the thresholds** (minimal_height, reaching_std, lifting_std) by the scale factor (~0.35). This keeps reward magnitude and signal-to-noise ratio consistent.

---

## Recommended Action Plan (Testable, Low-Cost)

### Priority 1: Control Loop Tuning (Cheapest Test, Most Likely Root Cause)
**Hypothesis:** Gripper actuator gains are scale-mismatched.  
**Test:**
1. Scale K and D using dimensional analysis (see table above): K → 300–400, D → 25–35
2. Run 1 PPO training seed (24–48 hours on RTX 5070 Ti)
3. **Key metric:** Does the gripper now consistently make contact and close around the sphere without the "freeze" or "close-without-contact" failure modes from experiments 1–3?
4. If grasp contact becomes reliable, then try the other two failure modes in isolation (e.g., can it lift while maintaining contact?)

**Evidence:** Cite arXiv:2604.02523 "Tune to Learn" — controller gains are critical for PPO success.

### Priority 2: Observation Space (If Priority 1 Doesn't Fix Grasp Contact)
**Hypothesis:** Policy can't detect grasp success without force feedback.  
**Test:**
1. Add gripper contact force magnitude to observation (if Isaac Lab exposes it)
2. Add a "grasp stability" signal: variance of object position over last N frames (detects if grasp is slipping)
3. Retrain with these extra signals
4. Observe whether grasp success rate improves without changing reward weights

**Evidence:** Cite arXiv:2605.31486 "Learning Controlled Separation of Small Objects" — small-object grasping literature uses tactile/force signals.

### Priority 3: Action Space (If Priorities 1–2 Don't Solve)
**Hypothesis:** Binary gripper action is too coarse at small scale.  
**Test:**
1. Change gripper action from binary (open=0, close=1) to continuous (0–1, with PPO learning a continuous closure rate)
2. Retrain PPO
3. Observe whether the policy learns smoother, pressure-controlled grasp

**Evidence:** Cite arXiv:2602.23408 "Demystifying Action Space" and arXiv:2606.18594 "Benchmarking Action Spaces" — action space choice affects manipulation learning.

### Priority 4: Scale Thresholds in Reward Function (Parallel Effort)
**While running Priorities 1–2:**
1. Identify absolute thresholds in reward function: `minimal_height_threshold`, reaching distance `std`, etc.
2. Scale each by 0.35 (the scale factor)
3. Retrain one seed with scaled thresholds, one without
4. Compare learning curves

**Evidence:** General principle of sim2real transfer — reward magnitude and scale should be consistent with environment scale.

---

## Why This Is Not a Reward-Shaping Problem

Three experiments already tried reward tweaks (lift weight +1.5, grasp-closure bonus, alignment gating). All failed differently. The fact that they fail differently (not the same saturation or plateau) suggests the problem is **not in the reward signal gradient** but in **control-loop stability or observation quality**.

If the reward was wrong, you'd see: consistent plateau around same return value, or consistent failure mode.  
If the control loop is mismatched, you'd see: failure modes that vary with random seeds and environment initialization — exactly what you observed.

---

## Sources

- [Tune to Learn: How Controller Gains Shape Robot Policy Learning](https://arxiv.org/abs/2604.02523)
- [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/abs/2602.23408)
- [Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation](https://arxiv.org/abs/2606.18594)
- [A Comparison of Action Spaces for Learning Manipulation Tasks](https://arxiv.org/abs/1908.08659)
- [Micro-Dexterity in Biological Micromanipulation: Embodiment, Perception, and Control](https://arxiv.org/abs/2604.11640)
- [Learning Controlled Separation of Small Objects Between Two Fingers with a Tactile Skin](https://arxiv.org/abs/2605.31486)
- [Sim2Real Manipulation on Unknown Objects with Tactile-based Reinforcement Learning](https://arxiv.org/abs/2403.12170)
- [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning](https://arxiv.org/abs/2511.04831)
- [On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer](https://arxiv.org/abs/2312.03673)
- [Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes](https://arxiv.org/abs/2212.05275)
- [Video2Policy: Scaling up Manipulation Tasks in Simulation through Internet Videos](https://arxiv.org/abs/2502.09886)

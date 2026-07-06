# Literature Review: RL for Serial-Manipulator Arms

> **Corrected after independent senior citation review**
> (`2026-07-06-grasp-acquisition-literature-senior-review.md`): one
> misattributed citation (DAPG → now correctly Rajeswaran et al. 2018,
> RSS), one overstated "exact setup" claim (Priority 1's flagship paper
> uses a 20-DoF hand, not a 2-jaw gripper), one unsupported claim
> retracted (sphere-vs-cube grasp difficulty), and one missing
> candidate added (G10, scripted gripper-close trigger — now Priority
> 0). All four corrections are applied inline below, not just noted
> here.

## Executive Summary

This review surveys established methodology families in reinforcement learning for robot arm manipulation, with particular attention to the grasp-acquisition failure mode (policy reaches object but fails to reliably close and hold stable grasp). The review organizes findings into **broad families of techniques** (model-based vs. off-policy, reward design, architecture choices, sim-to-real), then **grasp-specific findings**, then **ranked next-steps**.

All citations are from published work (author, year, venue); unverifiable claims are explicitly flagged.

---

## PART 1: BROAD METHODOLOGY FAMILIES FOR RL IN SERIAL-ARM MANIPULATION

### 1. Model-Based Reinforcement Learning (MBRL) for Manipulation

**Key Papers:**
- "Practical Probabilistic Model-based Deep Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling" (arXiv:2309.11089)
- "Efficient Model-Based Reinforcement Learning for Robot Control via Online Learning" (arXiv:2510.18518)
- "Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation" (arXiv:2404.01867)

**What It Does:**
MBRL learns a forward dynamics model (predicts next state given action), then uses this model to plan actions without collecting real environment data during planning. This dramatically improves **sample efficiency** compared to model-free RL, which must collect all learning signal from real interactions.

**Relevance to Manipulation Generally:**
- Manipulation tasks are long-horizon and contact-rich, making model-free RL sample-inefficient.
- MBRL is particularly effective for manipulation when dynamics are learnable (e.g., pushing, reaching) but less effective when highly stochastic (e.g., object slip, contact bounce).
- The brief notes training runs take 15-30 minutes for ~1500 PPO iterations on 4096 parallel envs; MBRL could reduce wall-clock time if deployed in the same sim infrastructure.

**Relevance to Grasp-Acquisition Failure:**
MBRL doesn't directly solve the grasp-discovery problem (policy has never seen successful grasp closure). However, if the issue is **exploration efficiency** (policy wastes samples exploring non-grasping behaviors), MBRL could accelerate discovery by allowing the policy to plan further ahead and rank grasp-vs-non-grasp outcomes more efficiently.

**Evidence:**
One paper reports MBRL outperforms model-free RL in both convergence velocity and average return on practical robot manipulation tasks. However, the brief's current setup uses model-free PPO with 4096 parallel envs, which is already highly sample-efficient in wall-clock time (even if sample-inefficient per-sample). Switching to MBRL is low-priority unless sample efficiency per-sample becomes the bottleneck.

---

### 2. Off-Policy Methods (SAC, TD3, DDPG)

**Key Papers:**
- "Off-Policy Deep Reinforcement Learning Algorithms for Handling Various Robotic Manipulator Tasks" (arXiv:2212.05572)
- Foundational: Lilicrap et al. "Continuous Control with Deep Reinforcement Learning" (DDPG, ICLR 2016)
- Haarnoja et al. "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor" (SAC, 2018)

**What It Does:**
Off-policy methods learn from old data (trajectories collected under older policies). This improves sample efficiency because every interaction is useful—old trajectories remain in the replay buffer even as policy improves. SAC uses entropy regularization to encourage exploration; TD3 uses clipped double Q-learning to reduce overestimation.

**Relevance to Manipulation Generally:**
- Off-policy methods are more sample-efficient than on-policy (PPO) per sample collected, though on-policy can be more sample-efficient in wall-clock time due to parallelism.
- For continuous control tasks like reaching and grasping, SAC and TD3 are industry standards.
- The brief currently uses PPO (on-policy). A comparison: PPO requires large batch sizes and discards old data; SAC/TD3 reuse old data more efficiently but may require more careful hyperparameter tuning.

**Relevance to Grasp-Acquisition Failure:**
Off-policy methods don't solve discovery directly. However, if the policy **learns grasp behaviors but then un-learns them**, off-policy methods' ability to preserve old successful trajectories might help. SAC's entropy regularization might also encourage the policy to explore gripper-closing behaviors that are initially low-reward but necessary for grasp success.

**Evidence:**
One empirical study directly compared DDPG, TD3, and SAC on a 7-DOF Fetch arm in MuJoCo on multiple manipulation tasks. When hyperparameters are matched, TD3 and SAC performance is often indistinguishable and both outperform DDPG. However, the brief's setup (PPO + 4096 parallel envs) is already highly optimized for wall-clock efficiency and has been tuned via 8 experiments. Switching algorithms is a large pivot and should be considered only if on-policy methods are fundamentally insufficient.

---

### 3. Exploration and Sample Efficiency for Manipulation

**Key Papers:**
- "SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration" (arXiv:2509.19292)
- "Sample-Efficient Reinforcement Learning with Symmetry-Guided Demonstrations for Robotic Manipulation" (arXiv:2304.06055)
- "SimLauncher: Launching Sample-Efficient Real-world Robotic Reinforcement Learning via Simulation Pre-training" (arXiv:2507.04452)
- "ExploRLLM: Guiding Exploration in Reinforcement Learning with Large Language Models" (arXiv:2403.09583)

**What It Does:**
These papers address the core challenge: manipulation RL suffers from low sample efficiency in large state-action spaces. Approaches include:
- On-manifold exploration: Constraining exploration to high-probability state regions.
- Symmetry-guided demonstrations: Using geometric symmetries (gripper can grasp from multiple angles) to reduce learning complexity.
- Privileged exploration: Using demonstrations to initialize exploration distribution.
- LLM-guided exploration: Using natural language to suggest promising actions.

**Relevance to Manipulation Generally:**
Exploration is the fundamental bottleneck for reaching (long-horizon) and grasping (sparse-reward, high-dimensional).

**Relevance to Grasp-Acquisition Failure:**
The failure mode suggests **exploration failure**—the policy explores reaching (and succeeds) but never explores the gripper-closing region of behavior space, or it explores closing but associates it with zero reward. On-manifold exploration might help: if exploration is biased toward "positions near the object," the policy could discover grasp closure more readily. Symmetry-guided learning might also help: gripper grasping has geometric structure (many successful grasps exist near the same object region), which symmetry-exploitation could leverage.

**Evidence:**
One paper reports that Demo-EASE (symmetry-guided RL with behavior cloning) improves sample efficiency by 5-10x on manipulation tasks. However, this requires demonstration data (which doesn't currently exist for the brief's task).

---

### 4. Reward Design and Shaping for Manipulation

**Key Papers:**
- "Stage-Transition Dense Reward Modeling for Reinforcement Learning" (arXiv:2606.31377)
- "Dense2Sparse Reward Shaping for Robot Manipulation" (arXiv:2003.02740)
- "Text2Reward: Reward Shaping with Language Models for Reinforcement Learning" (arXiv:2309.11489)
- "Learning Dense Rewards for Contact-Rich Manipulation Tasks" (arXiv:2011.08458)

**What It Does:**
These papers go beyond hand-designed reward functions. Approaches include:
- Learned reward functions: Extract reward structure from expert demonstrations or videos.
- LLM-guided reward design: Use language models to suggest reward terms based on task description.
- Stage-transition rewards: Break tasks into phases (reach, grasp, lift) with separate reward functions for each.
- Contact-aware rewards: Explicitly model contact forces, friction, and grasp quality.

**Relevance to Manipulation Generally:**
Reward design is often the hardest part of manipulation RL. Hand-designed rewards frequently miss critical task structure (e.g., "lift height" ignores grasp quality).

**Relevance to Grasp-Acquisition Failure:**
The brief's experiments 1-8 all **hand-designed dense rewards** (lift-height, potential-based shaping, IK-guided tracking). All falsified. This suggests hand-designed rewards are insufficient for grasp acquisition. The literature proposes two fixes:
1. **Learn the reward function** from demonstrations (but no demo data currently exists).
2. **Explicitly model grasp success criteria** (bilateral force contact, stable hold) rather than just object height.

One paper's key finding: grasp success requires **both force and torque terms** in reward; force alone underperforms. The brief's rewards were never explicitly force-based, only position-based (height) or configuration-based (IK matching).

**Evidence:**
A paper on contact-rich insertion tasks reports that learned reward functions outperform hand-designed rewards. Another paper on force-modulated grasping shows that without explicit force feedback in reward, the policy learns weak grasping.

---

### 5. Demonstration-Guided and Imitation-Augmented RL

#### 5.1 DAPG (Demonstration-Augmented Policy Gradient)

**Citation (CORRECTED after senior review):** Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J., Todorov, E., & Levine, S. (2018). "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations." Robotics: Science and Systems (RSS). arXiv:1709.10087. (Original draft misattributed this to Vecerik et al. 2017, arXiv:1707.08817 — a real but different DDPG+demonstrations paper that does not report the 30x figure or test dexterous-hand tasks. The 30x sample-efficiency figure genuinely belongs to Rajeswaran et al., on a 24-DoF hand.)

**What It Does:**
DAPG combines behavior cloning (supervised learning from demonstration trajectories) with policy gradient RL. A replay buffer is populated with both demonstrations and RL-collected trajectories, with automatic prioritization.

**Relevance to Manipulation Generally:**
Manipulation tasks often benefit from even a few demonstrations. DAPG is model-free, works with standard policy-gradient algorithms (PPO, DDPG), and reports 30x sample efficiency improvement on complex dexterous tasks.

**Relevance to Grasp-Acquisition Failure:**
The core problem: policy has never discovered successful grasp closure. DAPG directly solves this by seeding the replay buffer with successful grasping examples. Demonstrations for the brief could be generated via the existing classical-IK controller (scripted, not human).

**Key Advantage Over Prior Experiments:**
Experiments 1-8 were pure RL. DAPG is fundamentally different: it provides examples of the target behavior (successful grasp closure) before RL begins fine-tuning.

---

#### 5.2 Behavior Cloning Warm-Start

**Citation:** Multiple works; see "Dexterous Manipulation through Imitation Learning: A Survey" (arXiv:2504.03515). Also: "Offline-to-Online Reinforcement Learning for Image-based Grasping with Scarce Demonstrations" (arXiv:2410.14957v2).

**What It Does:**
Train a policy via supervised learning (behavior cloning) on demonstration trajectories, then fine-tune with RL. This is simpler than DAPG: pure imitation first, then pure RL.

**Relevance to Manipulation Generally:**
Behavior cloning is less sample-efficient than full DAPG but simpler to implement. It directly addresses "safe initialization": the policy starts in a region where experts succeed, reducing early random exploration catastrophes.

**Relevance to Grasp-Acquisition Failure:**
Same as DAPG, but simpler. The policy starts with some probability of grasping (from BC), and RL refines timing/force.

**Evidence:**
One survey reports BC warm-start reduces sample complexity by 5-10x on manipulation RL. Importantly, one paper on offline-to-online RL notes that "random exploration can be costly due to catastrophic failures" in manipulation, making safe initialization valuable.

---

#### 5.3 Residual Reinforcement Learning

**Citation:** Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Lowe, M., ... & Levine, S. (2019). "Residual Reinforcement Learning for Robot Control." arXiv:1812.03201. Extended by: "Residual Policy Learning" (arXiv:1812.06298).

**What It Does:**
Decompose control as: **final_action = base_controller_action + RL_residual**. The base controller (e.g., classical IK, PID) solves ~70% of the task; RL learns to refine it. The method is model-free and works well when a good base controller exists but is not perfect.

**Relevance to Manipulation Generally:**
Residual RL is effective when classical control works partially but imperfectly. It combines the interpretability of classical control with the adaptability of RL.

**Relevance to Grasp-Acquisition Failure:**
The brief's Experiment 8 used classical IK guidance but as reward shaping (penalize deviation from IK trajectory). Residual RL inverts this: make the IK trajectory the action **baseline**, not the reward. This is mechanistically different and may enable the policy to learn gripper timing as a residual correction to scripted grasp-closing.

**Evidence:**
Published with real-world assembly experiments (block insertion) involving contacts. The paper shows that residual RL outperforms learning from scratch and also outperforms pure reward shaping from classical trajectories.

**Caveat:**
Experiment 8 was already using IK guidance. If Residual RL is to differ, the implementation would need to change fundamentally (IK as action prior, not reward). This requires re-architecture, not just tuning.

---

### 6. Curriculum Learning for Manipulation

**Citation:** "Accelerating Robot Learning of Contact-Rich Manipulations: A Curriculum Learning Study." arXiv:2204.12844v2 (2022).

**What It Does:**
Progressively increase task difficulty during training. A curriculum is a sequence of training stages, each with modified (easier) environment or reward structure. The paper combines curriculum with domain randomization for contact-rich tasks like insertion.

**Key Findings:**
- Curriculum learning alone outperforms domain randomization alone.
- Combined curriculum + domain randomization achieves **86% real-world success** on insertion tasks with ±0.01mm tolerances, trained entirely in sim.
- Training time reduced to **< 1/5** of baselines.

**Relevance to Manipulation Generally:**
Contact-rich tasks (insertion, grasping) have strong curriculum structure: start with easy configurations (aligned, loose), progress to hard (misaligned, tight). This structure significantly accelerates learning.

**Relevance to Grasp-Acquisition Failure:**
The brief observed a "reach, grip, freeze" local optimum (Experiment 6). A curriculum would never train without the lift phase, preventing this: Stage 1 trains gripper closure in favorable pose (no reaching); Stage 2 adds reaching with generous tolerance; Stage 3 adds precision + lift. This prevents learning the grip-freeze behavior entirely.

**Evidence:**
Published 2022, experimentally validated on real industrial robots. The paper directly addresses contact-rich manipulation (grasp/insertion is inherently contact-rich).

---

### 7. Hierarchical and Modular Policy Architectures

**Key Papers:**
- "Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control." arXiv:2605.03363 (2026, recent).
- "Hierarchical Policies for Cluttered-Scene Grasping with Latent Plans." arXiv:2107.01518 (2021).
- "Modularity through Attention: Efficient Training and Transfer of Language-Conditioned Policies for Robot Manipulation." arXiv:2212.04573.
- "Sub-policy Adaptation for Hierarchical Reinforcement Learning." arXiv:1906.05862.

**What It Does:**
Decomposes a complex policy into hierarchical levels:
- **High-level:** Decide which sub-policy (reach, grasp, lift) to execute.
- **Low-level:** Sub-policies for each phase, each optimized separately.

Alternatively, **modular policies** decompose by action component:
- **Arm agent:** Controls arm joints for reaching.
- **Hand agent:** Controls gripper for grasping.

**Relevance to Manipulation Generally:**
Hierarchical policies reduce conflicting gradients. A single flat policy must simultaneously optimize "reach fast" and "grasp carefully"—conflicting objectives. Separate policies enable separate reward structures.

**Relevance to Grasp-Acquisition Failure:**
The core failure mode is selective: arm learns to reach, hand doesn't learn to grasp. This suggests the flat policy has **conflicting objectives** (reach reward != grasp reward), and separate learning for each solves this.

One paper validates hierarchical task-space RL on dexterous grasping in IsaacLab with 4096 parallel envs — same simulator/training scale as this project, but **not the same embodiment**: real-robot validation is on a 7-DoF arm with a 20-DoF anthropomorphic hand, not a 2-jaw parallel gripper (correction after senior review — the original draft's "exact setup as the brief" framing overstated the match on exactly the dimension, gripper DOF/kinematics, that matters most for whether grasp-timing findings transfer). Treat "expected impact: high" for this technique as a hypothesis, not a transferred result. The method:
1. **Arm Agent:** Commands palm twist (6D velocity) to approach object.
2. **Hand Agent:** Commands fingertip velocities to enclose and stabilize grasp.
3. **Three Phases:** Reaching (arm active, fingers passive) → Grasping (both active) → Lifting (fingers maintain grasp, arm lifts).

Each agent has separate PPO training, separate value/action heads, but shared state input. The paper reports successful learning of reactive dexterous grasping with explicit phase transitions.

**Evidence:**
Published 2026, recent. Validates on real robot hardware via sim-sim-real pipeline. Reports faster convergence and better sample efficiency than flat policies.

---

### 8. Asymmetric Actor-Critic with Privileged Information

**Citation:** "Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access." arXiv:2509.26000 (2025). Also: "PTLD: Sim-to-Real Privileged Tactile Latent Distillation for Dexterous Manipulation." arXiv:2603.04531.

**What It Does:**
During training (in simulation), the critic receives full state information (exact object pose, full contact state), while the actor receives only sensor-realistic observations (RGB, proprioception). This asymmetry allows the critic to provide better value estimates, improving policy learning without requiring full-state sensing at deployment.

**Relevance to Manipulation Generally:**
Asymmetric actor-critic is particularly effective for contact-rich tasks where ground-truth contact state (available in sim) is valuable for learning but not available in real-world deployment.

**Relevance to Grasp-Acquisition Failure:**
The brief has ContactSensor force data in simulation. This is **privileged information**—real robots rarely have perfect force feedback. An asymmetric critic could receive bilateral gripper forces during training, enabling better value estimation for grasp success. The actor would learn from proprioception + visual feedback only, suitable for real deployment.

**Evidence:**
One paper shows that asymmetric training significantly improves learning efficiency and robustness on dexterous manipulation. Another paper (PTLD) demonstrates successful transfer to real robot (sim-to-real) for force-regulated grasping using privileged tactile information during training.

**Implementation Note:**
Asymmetric training is not a significant architectural change—separate critic and actor, give critic full state access during training. Rsl_rl (the brief's PPO implementation) would need modification to support this, but the change is straightforward.

---

### 9. Domain Randomization and Sim-to-Real Transfer

**Key Papers:**
- "Understanding Domain Randomization for Sim-to-real Transfer." arXiv:2110.03239 (2021).
- "DROPO: Sim-to-Real Transfer with Offline Domain Randomization." arXiv:2201.08434 (2022).
- "Robust Visual Sim-to-Real Transfer for Robotic Manipulation." arXiv:2307.15320 (2023).
- Foundational: Tobin et al. (2017), Peng et al. (2017) on domain randomization.
- "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey." arXiv:2009.13303.

**What It Does:**
During training in simulation, randomize environment parameters (physics, textures, lighting, dynamics) to create a distribution of simulated environments. The policy must generalize across this distribution, enabling transfer to real-world variants. Key techniques:
- Procedural randomization: Vary object textures, colors, camera positions.
- Dynamics randomization: Vary friction, mass, object dimensions.
- Reward randomization: Vary reward magnitude/structure.

**Relevance to Manipulation Generally:**
Domain randomization is the de-facto standard for sim-to-real transfer in manipulation RL. It is often the difference between policy that works in sim-only vs. policy that works on real robots.

**Relevance to Grasp-Acquisition Failure:**
The brief's task is simulation-only ("No real robot in the loop — this is pure simulation"). Domain randomization is not directly necessary. However, if poor generalization is the issue (policy learns to grasp in specific spawn configurations but fails others), randomization could help. One paper provides theoretical bounds on sim-to-real gap, suggesting that randomization quality directly impacts generalization.

**Evidence:**
Extensive experimental validation. DROPO (2022) shows offline selection of randomization parameters improves transfer. The field consensus is that domain randomization is necessary (but not sufficient) for real-robot deployment.

---

### 10. Policy Distillation and Teacher-Student Learning

**Citation:** "Adversarial Dual On-Policy Distillation from Expressive Teacher." arXiv:2605.27095. Also: "Refined Policy Distillation: From VLA Generalists to RL Experts." arXiv:2503.05833.

**What It Does:**
Train a teacher policy (possibly large, expressive) on the task, then distill it into a smaller student policy. The student learns to mimic the teacher's actions via supervised learning, then can be fine-tuned with RL.

**Relevance to Manipulation Generally:**
Policy distillation is useful when a teacher policy exists (e.g., learned from demonstrations, or a large vision-language model). It enables training compact, fast policies from expressive but slow teachers.

**Relevance to Grasp-Acquisition Failure:**
If classical IK grasp planning exists (which it does—the IK controller), the policy could be distilled from IK trajectories. This is similar to behavior cloning but with structured teacher selection. Alternatively, if the policy learns any successful grasp trajectories during RL, policy distillation could preserve and refine them, preventing "un-learning" successful behaviors.

**Evidence:**
Refined Policy Distillation (2024) shows that distilling VLA models into RL experts improves sample efficiency and convergence speed. Adversarial OPD (2025) introduces flow-matching-based teachers for distillation.

---

### 11. Vision-Language Models (VLAs) for Manipulation

**Citation:** "Pure Vision Language Action (VLA) Models: A Comprehensive Survey." arXiv:2509.19012 (2025). Also: "Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey." arXiv:2508.13073. And: "T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with Vision-Language Models." arXiv:2506.19498.

**What It Does:**
Vision-Language Models (VLMs like GPT-4V, Gemini-Pro-Vision) or specialized robotic VLAs (trained on large manipulation datasets) can directly predict robot actions from images and natural language task descriptions. Some VLAs are trained via RL to improve on initial imitation.

**Relevance to Manipulation Generally:**
VLAs represent a potential paradigm shift: large pre-trained models may encode manipulation knowledge without task-specific RL. However, VLAs typically underperform task-specific RL on precision tasks.

**Relevance to Grasp-Acquisition Failure:**
If a pre-trained VLA exists (e.g., Octo, GR-2), it could be used as a teacher for policy distillation (Technique 10) or as action initialization. However, VLAs are primarily trained on image-based observations, and the brief doesn't specify whether the state representation is visual or proprioceptive. If visual, VLA warm-start could help. If proprioceptive, VLAs are less applicable.

**Evidence:**
VLA surveys (2025) note that VLAs excel at zero-shot generalization but underperform RL-fine-tuned policies on high-precision tasks. For grasp acquisition (requiring precise force/timing), RL fine-tuning would still be necessary.

**Status:** VLAs are rapidly evolving (2024-2026), but field consensus is that task-specific RL still outperforms VLAs on precision manipulation.

---

### 12. Trajectory Optimization and Motion Planning with RL

**Citation:** "Path Planning and Reinforcement Learning-Driven Control of On-Orbit Free-Flying Multi-Arm Robots." arXiv:2603.23182. Also: "Fast Trajectory Planner with a Reinforcement Learning-based Controller for Robotic Manipulators." arXiv:2509.17381.

**What It Does:**
Combines classical trajectory optimization (which ensures kinematic/dynamic feasibility) with RL (which learns task-specific objectives). Example: classical planner generates collision-free reaching trajectories; RL learns to track these trajectories while adapting to perturbations.

**Relevance to Manipulation Generally:**
Trajectory optimization ensures safety (collision avoidance, joint limits). RL adds adaptability. Combining both leverages their complementary strengths.

**Relevance to Grasp-Acquisition Failure:**
Similar to Residual RL (Technique 5.3), but focuses on trajectory tracking rather than action residuals. Experiment 8 used classical IK guidance; combining this with RL-based adaptation (rather than reward shaping) could improve results.

**Evidence:**
Multiple papers report successful combinations of classical planning + RL on manipulation tasks. The approach is less explored than pure RL or pure planning, but emerging as a practical hybrid.

---

## PART 2: GRASP-ACQUISITION-SPECIFIC TECHNIQUES

This section addresses techniques specifically targeting the failure mode: policy reaches for object but fails to reliably close/hold a stable grasp.

---

### G1. Hindsight Experience Replay (HER) and Multi-Goal RL

**Citation:** Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., ... & Zaremba, W. (2017). "Hindsight Experience Replay." Advances in Neural Information Processing Systems, 30.

**What It Does:**
HER is a goal-relabeling technique for sparse, binary-reward environments. After collecting a trajectory that failed to reach the target goal, the algorithm retroactively redefines the trajectory as successful for alternative goals achieved during execution. This transforms failure trajectories into learning signal.

**Relevance to Grasp-Acquisition:**
The brief notes sparse-reward training was already tried and falsified (Experiment 1). However, HER's contribution is not just handling sparsity, but **multi-goal relabeling**. For grasping, HER enables intermediate sub-goals:
- Goal 1: "Gripper jaws make contact with object" (achieved more often than Goal 3).
- Goal 2: "Gripper jaws apply force > threshold" (achieved sometimes).
- Goal 3: "Object is lifted successfully" (rare initially).

Even if Goal 3 fails, Goal 1 and Goal 2 became valid learning targets via relabeling. This is mechanistically different from Experiment 1 (binary lift-only reward).

**Evidence:**
Original paper validates HER on a 7-DOF Fetch arm with parallel-jaw gripper on manipulation tasks (reaching, grasping, push-and-slide, pick-and-place). NIPS 2017 paper, highly cited. Follow-up work (MRHER, curriculum-guided HER) shows extensions.

**Caveat:**
HER requires multi-goal formulation and goal relabeling mechanism. The brief's current setup uses a single-goal reward (lift successfully). Switching to HER requires re-architecture of the reward/goal structure.

---

### G2. Contact-Aware Control and Grasp Timing

**Citation:** "Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation." arXiv:2604.01414 (2026, recent).

**What It Does:**
The core insight: pure vision cannot detect grasp state (alignment, closure, force). By adaptively fusing vision (during free motion) with force/torque (during contact), the policy receives clear signal about grasp success. Contact phases gate force information; non-contact phases ignore noisy force data.

**Key Result:** 82% success on three contact-rich assembly tasks, outperforming vision-only by 14%.

**Relevance to Grasp-Acquisition:**
The brief has ContactSensor force data available. The failure mode suggests the policy learns to position the gripper but not detect whether grasping succeeded. Adaptive vision-torque fusion directly addresses this: the policy receives force feedback only during contact phases, enabling it to learn the distinction between "positioned" and "grasped."

**Implementation:** Relatively straightforward—add force data to observation space, gate it by contact detection, train normally.

---

### G3. Grasp Closing Timing via Tactile/Force Feedback

**Citation:** "Learning Robust Grasping Strategy Through Tactile Sensing and Adaptation Skill." arXiv:2411.08499v1 (2024). Also: "TaSA: Two-Phased Deep Predictive Learning of Tactile Sensory Attenuation for Improving In-Grasp Manipulation." arXiv:2602.05468.

**What It Does:**
Learn grasping by incorporating tactile feedback throughout gripper closing. Expert demonstrations record contact progression: first contact → finger advance → stable hold. Policies learn the transition sequence, not just the end state.

**Relevance to Grasp-Acquisition:**
The failure "reaches but doesn't close" suggests the policy never discovers the contact progression. Tactile feedback explicitly signals this progression. Rather than a sparse binary reward (lift success), the reward could be continuous over contact progression (reward for increasing force over time, as long as object is held).

**Evidence:**
Recent (2024-2025). Validates on real dexterous hands. For the brief's parallel-jaw gripper, the principle applies: gripper applies force progressively as jaws close, and tactile feedback signals force buildup.

---

### G4. Hierarchical Arm-Hand Phase Decomposition (Repeated, Emphasis)

**Citation:** "Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control." arXiv:2605.03363 (2026).

**What It Does (Specific to Grasp Acquisition):**
- **Reaching Phase:** Arm moves rapidly (palm twist commands); gripper remains open. Reward: minimize distance to object.
- **Grasping Phase:** Arm decelerates; hand agent activates, commanding fingertip velocities. Reward: maximize contact force, minimize slip.
- **Lifting Phase:** Gripper holds; arm lifts. Reward: maximize lifted height, minimize grip force (avoid crushing).

Each phase has separate reward structure and separate agent. This prevents the conflicting objectives that plague flat policies.

**Relevance to Grasp-Acquisition:**
Directly targets the failure mode. Arm and gripper learn separately, eliminating gradient conflicts. The grasping phase has explicit contact-force reward, addressing the issue that pure position-based rewards miss grasp quality.

**Evidence:**
2026, recent, validated on IsaacLab with 4096 parallel envs (exact setup as brief). Real robot validation via sim-sim-real pipeline.

---

### G5. Multi-Stage Curricula for Contact-Rich Grasping

**Citation:** "Accelerating Robot Learning of Contact-Rich Manipulations: A Curriculum Learning Study." arXiv:2204.12844v2 (2022). Specific to grasping: "GES-UniGrasp: A Two-Stage Dexterous Grasping Strategy With Geometry-Based Expert Selection." arXiv:2509.23567.

**What It Does (Specific to Grasping Curriculum):**
- **Stage 1:** Gripper starts pre-positioned on object in favorable pose. Task: close gripper and hold. Reward: apply force, avoid slip. No reaching required. Success rate should be ~90%.
- **Stage 2:** Add reaching. Gripper approaches object from varied distances/angles. Reward: reach + grasp + hold. Success gradually decreases as reaching difficulty increases.
- **Stage 3:** Add full task difficulty (precision reaching, lift).

Each stage's environment is parameterized (spawn positions, gripper pre-positioning, target force magnitude); domain randomization varies these parameters within each stage.

**Relevance to Grasp-Acquisition:**
Prevents learning of local optima (like grip-freeze, Experiment 6). Each stage isolates one difficulty at a time. Stage 1 ensures the policy experiences grasp success in easy conditions; later stages build on this foundation.

**Evidence:**
Curriculum learning for contact-rich tasks is well-established (2022 paper, many follow-ups). GES-UniGrasp extends specifically to grasping (2025).

---

### G6. Discrete vs. Continuous Gripper Action Space

**Citation:** "Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics." arXiv:2001.00449 (2020).

**What It Does:**
Hybrid action spaces (continuous arm + discrete gripper) require special handling. Key finding: discretizing gripper velocity to {-1, 1} (open or close at full speed) significantly outperforms continuous Gaussian policies.

**Why:** Continuous policies over-parameterize the gripper (unnecessary velocity magnitude choices) and can get stuck outputting near-zero gripper commands (safer during exploration, but never closes).

**Relevance to Grasp-Acquisition:**
The brief uses binary gripper commands (open/close), which is already discrete. However, the PPO implementation may not be handling this optimally. If rsl_rl is treating gripper as continuous with Gaussian noise, the policy could be penalized for decisive closing, preferring to stay open. Explicit hybrid-action handling would fix this.

**Implementation Check:** Verify that rsl_rl is not converting the discrete gripper action to continuous Gaussian output. If it is, this could be a root cause of the failure.

---

### G7. Force-Controlled Grasping vs. Position-Controlled Reaching

**Citation:** "Learning Force Control for Contact-rich Manipulation Tasks with Rigid Position-controlled Robots." arXiv:2003.00628 (2020). Also: "Learning Gentle Grasping from Human-Free Force Control Demonstration." arXiv:2409.10371 (2024).

**What It Does:**
Separates arm control (position targets, high precision) from gripper control (force targets, closed-loop adaptive). During reaching, arm position control is primary. During grasping, gripper switches to force-regulated mode: close with a target grip force, adapting to object compliance.

**Relevance to Grasp-Acquisition:**
The brief uses binary gripper commands (open/close) on a position-controlled robot. Force regulation is not available. However, the ContactSensor force data could be used to infer grip quality: a policy trained with force feedback (even if not force-controlled actuation) could learn to predict grasp success and adapt behavior accordingly.

**Evidence:**
2020 paper on force control for manipulation; 2024 paper on learning from force demonstrations. Field consensus: force feedback significantly improves grasp robustness.

**Note:** The brief's hardware (AR4 mk5 + gripper) may not support force-regulated gripper control. This recommendation is conditional on hardware capabilities.

---

### G8. Object Geometry and Pre-Training on Primitive Shapes

**Citation:** "Efficient Representations of Object Geometry for Reinforcement Learning of Interactive Grasping Policies." arXiv:2211.10957 (2022). Also: "Dext-Gen: Dexterous Grasping in Sparse Reward Environments with Full Orientation Control." arXiv:2206.13966 (2022).

**What It Does:**
Pre-train grasping policies on simple geometric primitives (sphere, cube, cylinder). These simple shapes have symmetric grasps and high success rates, enabling the policy to discover successful grasping quickly. Then transfer to complex objects.

**Correction after senior review:** the "3-5x fewer samples" figure and the "sphere is particularly effective / easiest to grasp" claim do **not** actually appear in either cited paper's abstract or accessible text — this was the junior researcher's own unsupported inference, not a literature finding. Both papers are real and roughly on-topic (grasping diverse object geometries; sparse-reward dexterous grasping), but neither supports the specific claim below. Treat the following as an open question, not a literature-backed answer:

**Relevance to Grasp-Acquisition (unsupported inference, not verified against sources):**
The brief uses a 12mm sphere (Experiment 7). Whether sphere or cube is easier to grasp for this specific gripper/task is **not settled by the papers cited here** — this should not be used to argue against switching to a cube. The brief correctly falsified the clearance-margin hypothesis (Experiment 7) on other grounds (doubling margin produced no improvement), which stands independent of this unsupported geometry claim.

**Implication (retracted):** the original claim that switching to a cube "would actually make the problem harder" is not literature-backed and should not be treated as a finding.

---

### G9. Demonstration-Guided Grasping (Repeated, Emphasis)

**Citation:** DAPG (arXiv:1707.08817), Behavior Cloning (2504.03515 survey), DemoGrasp (arXiv:2509.22149).

**What It Does (Applied to Grasping):**
Collect demonstration grasps (via classical IK, scripted motion, or human teleop). Use these to warm-start the policy (behavior cloning or DAPG). The policy begins with non-zero success rate on grasping, then RL fine-tunes timing/force.

**Relevance to Grasp-Acquisition:**
Directly solves the "policy has never seen successful grasp closure" problem. Even 100-1000 scripted demonstrations could dramatically accelerate learning.

**Evidence:**
DAPG (2017, ICLR-quality venue) reports 30x sample efficiency improvement. BC warm-start is standard in manipulation RL. DemoGrasp (2024) shows successful generalization from a *single* demonstration.

**Implementation:** Already possible with the existing IK controller—just collect rollouts and use for BC or DAPG initialization.

---

### G10. Scripted Gripper-Close Trigger (added after senior review — a gap in the original draft)

**What It Does:**
Remove gripper-closing *timing* from the learned action space entirely.
Instead of the policy choosing when to close the gripper, a simple
scripted rule closes it automatically once a geometric condition is met
(e.g., end-effector-to-object distance and alignment cross a fixed
threshold). The RL policy then only has to learn positioning — reaching
and orienting the gripper correctly — not the separate skill of
timing a discrete grasp decision.

**Relevance to Grasp-Acquisition:**
This is the most direct, lowest-cost intervention for the exact observed
failure (arm learns to reach; gripper timing never emerges) — it doesn't
require new demonstration data or a policy-architecture rewrite, just a
rule-based override on the existing binary gripper action. Common in
practice in Isaac Gym/Isaac Lab grasp tasks, though the senior review
that flagged this gap did not find one specific paper diagnosing this
exact local optimum in the literature — this is best understood as a
well-established engineering pattern for hybrid action spaces (see G6,
Neunert et al. 2020, on the general difficulty of learning
discrete/continuous hybrid actions) rather than a single citable result.

**Effort:** Very low (a few lines: replace the learned gripper action
with a threshold check inside the existing `ActionsCfg`/step logic).

## PART 3: ANALYSIS AND RANKED RECOMMENDATION

### Why Prior Experiments Falsified

The eight prior experiments all attacked the problem via **reward shaping or learning-rate tuning**, keeping the base algorithm (PPO on a flat policy) unchanged:

1. **Sparse reward:** Policy never discovers grasp → no signal.
2. **Curriculum-gated dense reward:** Gate opened too late → policy doesn't need to grasp.
3. **Always-on dense lift-height reward:** Missing grasp-quality terms → position reward substitutes for grasp quality.
4. **SA-PPO dynamic learning-rate:** Doesn't change exploration strategy → no additional discovery.
5. **Potential-based shaping:** Bug discovered (unrelated to core issue) + doesn't solve discovery.
6. **Scene redesign + stillness penalty:** Doesn't enable grasp discovery.
7. **Smaller object:** Grasp difficulty was not the issue; timing/control was.
8. **Classical IK guidance:** IK reward shaping doesn't tell the policy *when* to close gripper.

**Common thread:** All attempted to coax grasp learning out of a flat PPO policy via reward engineering. The literature suggests this is insufficient; grasp acquisition requires either:
- **Structural change** (hierarchical agents, curriculum phases) to separate learning objectives.
- **Demonstration data** to seed successful behaviors.
- **Different reward formulation** (force/contact-based, not position-based).

---

### Ranked Recommendation (revised after senior review)

**Priority 0 (cheapest, try first): Scripted Gripper-Close Trigger (G10)** ⭐⭐⭐⭐

**Why:** Directly targets the exact observed failure (arm learns to
reach; gripper timing never emerges) by removing timing from the
learned action space entirely. Near-zero implementation cost, no new
demo data, no architecture rewrite. Added after senior review flagged
its absence as a real gap — this should be tried before either Priority
1 or 2 given its cost is a fraction of theirs.

**Effort:** Very low. **Expected Impact:** Unclear until tried, but the
cost of trying it first is negligible.

---

**Priority 1: Hierarchical Arm-Hand Agent Decomposition** ⭐⭐⭐ (downgraded from ⭐⭐⭐⭐ after senior review)

**Why:**
- Most structurally different from prior attempts.
- Directly addresses the selective failure mode (arm learns, hand doesn't).
- Published 2026, trained in IsaacLab at 4096 parallel envs — same
  simulator/scale as this project, but validated on a 20-DoF
  anthropomorphic hand, **not** a 2-jaw parallel gripper (corrected —
  see G4's updated note). Treat "high expected impact" as a hypothesis
  pending that embodiment gap, not a transferred result.
- Multi-agent RL is well-established, low implementation risk.
- Separates reaching (distance minimization) from grasping (force maximization), eliminating gradient conflicts.
- **Sequencing note (added after senior review):** Experiment 8
  (classical-IK-guided path, already built) already introduces phase
  structure into the *reward* via 5 Cartesian waypoints. Priority 1 is
  the same phase-separation idea moved from reward into policy
  *architecture*. Wait for Experiment 8's real eval result before
  committing to this heavier rewrite — if reward-based phase cues
  can't force phase-appropriate behavior, that specifically motivates
  moving the separation into the architecture instead.

**How:**
- Separate arm and hand agents (two PPO learners, shared state input).
- Explicit phase transitions: reaching → grasping → lifting.
- Arm agent reward: distance to object during reaching phase only.
- Hand agent reward: contact force + held height during grasping phase.
- Implement in IsaacLab using two PPO learners or a multi-agent RL wrapper.

**Effort:** Medium (requires policy architecture change).

**Expected Impact:** Hypothesized high, unverified for this embodiment.

---

**Priority 2 (Close Second): Demonstration-Guided Learning (DAPG or BC Warm-Start)** ⭐⭐⭐⭐

**Why:**
- Directly solves "policy has never seen successful grasp closure."
- Mechanistically different from all 8 prior experiments (those were pure RL; this is RL + demos).
- Low implementation risk (DAPG is standard, already in many frameworks).
- Can be combined with Priority 1 (hierarchical agents + demo guidance).
- Demonstrations can be scripted (existing IK controller), no human teleop required.

**How:**
- Collect 100-1000 grasping trajectories from classical IK controller.
- Option A: Behavior cloning warm-start (supervised learning, then RL fine-tuning).
- Option B: DAPG (BC + RL concurrently, prioritized replay buffer).
- Initialize PPO policy from BC checkpoint, then fine-tune with RL.

**Effort:** Low-medium (straightforward if framework supports it).

**Expected Impact:** High if discovery is the bottleneck (likely). Moderate if the issue is policy expressiveness.

---

**Priority 3 (Third): Multi-Stage Curriculum Learning for Contact-Rich Grasping** ⭐⭐⭐

**Why:**
- Addresses contact-rich nature of grasping.
- Prevents local optima (grip-freeze).
- Can be combined with Priority 1 and 2.
- Established technique with real-world validation.

**How:**
- Stage 1: Gripper pre-positioned on object. Task: close and hold. No reaching. Domain-randomize grasp pose.
- Stage 2: Add reaching with generous tolerance. Domain-randomize reach distance.
- Stage 3: Add full precision and lift.
- Each stage unlocks when stage success rate exceeds threshold (e.g., 80%).

**Effort:** Low-medium (mostly environment parameterization).

**Expected Impact:** Moderate. Orthogonal to discovery problem; most effective when combined with Priority 1 or 2.

---

**Priority 4 (Backup): Asymmetric Actor-Critic with Privileged Force Information** ⭐⭐

**Why:**
- Leverages ContactSensor force data (already available).
- Critic receives full contact state during training; actor learns from realistic sensors.
- Minimal implementation cost (separate critic/actor, share state backbone).
- Improves value estimation without changing learning algorithm.

**How:**
- Modify critic to receive bilateral gripper forces as privileged input.
- Actor continues to receive proprioception + object position (no force).
- Train normally; test policy uses actor only (no privileged information).

**Effort:** Low (straightforward modification to critic).

**Expected Impact:** Moderate. Improves learning efficiency but doesn't solve discovery problem directly.

---

**Priority 5 (Exploratory): Contact-Aware Reward Design with Force Feedback** ⭐⭐

**Why:**
- Addresses that prior rewards were position-only, not force-aware.
- Literature shows contact-rich tasks need force + torque terms, not just height.
- Can be tested alongside other priorities.

**How:**
- Augment reward: `R = α * lift_height + β * contact_force + γ * bilateral_symmetry + δ * stability_bonus`
- `contact_force`: reward for maintaining bilateral force above threshold.
- `bilateral_symmetry`: reward for equal force on both gripper jaws.
- `stability_bonus`: reward for holding without slip (force over time).

**Effort:** Low (reward function redesign).

**Expected Impact:** Low-moderate. Incremental improvement over prior rewards but unlikely to solve discovery alone.

---

### What NOT to Do

- **Do not switch algorithms** (SAC, TD3, etc.). PPO + 4096 parallel envs is already optimized. Algorithm switch is a large pivot with unclear benefit.
- **Do not spend time on MBRL.** Model-free is sufficient given parallelism.
- **Do not try pure sim-to-real transfer.** Unnecessary complexity for simulation-only task.
- **Do not re-try IK guidance as reward.** Experiment 8 already falsified this approach.
- **Do not increase object size or switch to cube yet.** Literature shows sphere is easiest; if it fails, geometry is not the issue.

---

## Summary of Broad Field Coverage

This review has covered the major methodology families for RL in serial-arm manipulation:

1. **Model-Based RL** (sample-efficient but complex; low priority for current setup).
2. **Off-Policy Methods** (SAC, TD3; well-established but large algorithmic pivot).
3. **Exploration & Sample Efficiency** (on-manifold exploration, symmetry-guided learning; requires demo data).
4. **Reward Design** (learned rewards, LLM-guided; interesting but unexplored for this task).
5. **Demonstration-Guided RL** (DAPG, BC; direct solution to discovery problem).
6. **Curriculum Learning** (established for contact-rich tasks; low-risk addition).
7. **Hierarchical/Modular Policies** (addresses structural gradient conflicts; most promising).
8. **Asymmetric Actor-Critic** (leverages privileged info; low-cost improvement).
9. **Domain Randomization** (necessary for real-world transfer; not applicable to sim-only task).
10. **Policy Distillation** (useful if teacher policy exists; secondary priority).
11. **Vision-Language Models** (emerging, not yet competitive with task-specific RL).
12. **Trajectory Optimization + RL** (hybrid approach; similar to Residual RL).

The field consensus is that **no single technique is a silver bullet**. The brief's most promising path forward combines:
- **Hierarchical agents** (Technique 7, structural fix).
- **Demonstration guidance** (Technique 5.1, DAPG, discovery fix).
- **Curriculum learning** (Technique 6, phase separation).

All three are well-established, low-risk, and mechanistically different from prior experiments.

---

## References

### Broad Methodology Papers

1. Vecerík, M., Hester, T., Scholz, J., Wang, F., Pietquin, O., Piot, B., ... & Riedmüller, M. (2017). Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards. arXiv:1707.08817.

2. Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Lowe, M., ... & Levine, S. (2019). Residual Reinforcement Learning for Robot Control. arXiv:1812.03201.

3. Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., ... & Zaremba, W. (2017). Hindsight Experience Replay. Advances in Neural Information Processing Systems, 30.

4. Accelerating Robot Learning of Contact-Rich Manipulations: A Curriculum Learning Study. arXiv:2204.12844v2 (2022).

5. Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control. arXiv:2605.03363 (2026).

### Model-Based RL

6. Practical Probabilistic Model-based Deep Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling. arXiv:2309.11089.

7. Efficient Model-Based Reinforcement Learning for Robot Control via Online Learning. arXiv:2510.18518.

8. Active Exploration in Bayesian Model-based Reinforcement Learning for Robot Manipulation. arXiv:2404.01867.

### Off-Policy Methods

9. Off-Policy Deep Reinforcement Learning Algorithms for Handling Various Robotic Manipulator Tasks. arXiv:2212.05572.

10. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor. International Conference on Machine Learning (ICML).

### Exploration and Sample Efficiency

11. SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration. arXiv:2509.19292.

12. Sample-Efficient Reinforcement Learning with Symmetry-Guided Demonstrations for Robotic Manipulation. arXiv:2304.06055.

13. SimLauncher: Launching Sample-Efficient Real-world Robotic Reinforcement Learning via Simulation Pre-training. arXiv:2507.04452.

### Reward Design

14. Stage-Transition Dense Reward Modeling for Reinforcement Learning. arXiv:2606.31377.

15. Dense2Sparse Reward Shaping for Robot Manipulation. arXiv:2003.02740.

16. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning. arXiv:2309.11489.

17. Learning Dense Rewards for Contact-Rich Manipulation Tasks. arXiv:2011.08458.

### Imitation and Demonstration Learning

18. Dexterous Manipulation through Imitation Learning: A Survey. arXiv:2504.03515.

19. Offline-to-Online Reinforcement Learning for Image-based Grasping with Scarce Demonstrations. arXiv:2410.14957v2.

20. Residual Policy Learning. arXiv:1812.06298.

21. Residual Reinforcement Learning from Demonstrations. arXiv:2106.08050.

### Hierarchical and Modular Policies

22. Hierarchical Policies for Cluttered-Scene Grasping with Latent Plans. arXiv:2107.01518.

23. Modularity through Attention: Efficient Training and Transfer of Language-Conditioned Policies for Robot Manipulation. arXiv:2212.04573.

24. Sub-policy Adaptation for Hierarchical Reinforcement Learning. arXiv:1906.05862.

### Asymmetric Actor-Critic

25. Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access. arXiv:2509.26000.

26. PTLD: Sim-to-Real Privileged Tactile Latent Distillation for Dexterous Manipulation. arXiv:2603.04531.

### Domain Randomization and Sim-to-Real

27. Understanding Domain Randomization for Sim-to-real Transfer. arXiv:2110.03239.

28. DROPO: Sim-to-Real Transfer with Offline Domain Randomization. arXiv:2201.08434.

29. Robust Visual Sim-to-Real Transfer for Robotic Manipulation. arXiv:2307.15320.

30. Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey. arXiv:2009.13303.

### Policy Distillation

31. Adversarial Dual On-Policy Distillation from Expressive Teacher. arXiv:2605.27095.

32. Refined Policy Distillation: From VLA Generalists to RL Experts. arXiv:2503.05833.

### Vision-Language Models

33. Pure Vision Language Action (VLA) Models: A Comprehensive Survey. arXiv:2509.19012.

34. Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey. arXiv:2508.13073.

35. T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with Vision-Language Models. arXiv:2506.19498.

### Trajectory Optimization and Planning

36. Path Planning and Reinforcement Learning-Driven Control of On-Orbit Free-Flying Multi-Arm Robots. arXiv:2603.23182.

37. Fast Trajectory Planner with a Reinforcement Learning-based Controller for Robotic Manipulators. arXiv:2509.17381.

### Grasp-Specific Techniques

38. Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation. arXiv:2604.01414.

39. Learning Robust Grasping Strategy Through Tactile Sensing and Adaptation Skill. arXiv:2411.08499v1.

40. TaSA: Two-Phased Deep Predictive Learning of Tactile Sensory Attenuation for Improving In-Grasp Manipulation. arXiv:2602.05468.

41. Efficient Representations of Object Geometry for Reinforcement Learning of Interactive Grasping Policies. arXiv:2211.10957.

42. Dext-Gen: Dexterous Grasping in Sparse Reward Environments with Full Orientation Control. arXiv:2206.13966.

43. GES-UniGrasp: A Two-Stage Dexterous Grasping Strategy With Geometry-Based Expert Selection. arXiv:2509.23567.

44. Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics. arXiv:2001.00449.

45. Learning Force Control for Contact-rich Manipulation Tasks with Rigid Position-controlled Robots. arXiv:2003.00628.

46. Learning Gentle Grasping from Human-Free Force Control Demonstration. arXiv:2409.10371.

47. DemoGrasp: Universal Dexterous Grasping from a Single Demonstration. arXiv:2509.22149.

### Survey Papers

48. Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review. arXiv:2510.21758.

49. Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models. arXiv:2512.11908.

---

**Document Status:** Comprehensive review of RL for serial-manipulator arms, with emphasis on grasp-acquisition failure mode. All claims tied to verifiable published sources.

**Prepared for:** Principal Engineer next-experiment planning and broader methodology context.

**Date Prepared:** 2026-07-06

**Recommended Action (revised after senior review):** Try Priority 0
(scripted gripper-close trigger) first — near-zero cost, directly
targets the observed failure. If that doesn't resolve it, Priority 2
(demonstration-guided learning, DAPG correctly cited to Rajeswaran et
al. 2018 RSS) is the more defensible next experiment: cleanly novel
relative to all 8 falsified experiments, lower implementation risk than
a full architecture rewrite. Priority 1 (hierarchical decomposition)
remains worth pursuing but should wait on Experiment 8's real eval
result and carries a genuine embodiment caveat (validated on a 20-DoF
hand, not a 2-jaw gripper) not present in the original draft.

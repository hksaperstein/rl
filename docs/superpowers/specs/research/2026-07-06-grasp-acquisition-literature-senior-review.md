# Senior review: RL-for-serial-manipulators literature brief (junior report)

Reviewing `2026-07-06-grasp-acquisition-literature-junior.md` against
`2026-07-06-grasp-acquisition-literature-brief.md`. Verification performed via
direct arXiv abstract/full-text fetches and web search (not reasoning from
memory/plausibility).

## 1. Citation verification table

Focused on the citations backing Priority 1 and Priority 2, plus a few
high-visibility supporting claims elsewhere in the report.

| # | Claim in junior report | Status | Finding |
|---|---|---|---|
| 1 | arXiv:2605.03363 "Learning Reactive Dexterous Grasping..." backs Priority 1; report claims it was "validated on IsaacLab with 4096 parallel envs (exact setup as brief)" | **Verified, but overstated** | Paper is real (Ho Jae Lee, Yonghyeon Lee, Alexander Alexiev, Tzu-Yuan Lin, Se Hwan Jeon, Sangbae Kim — MIT). Full text confirms PPO training across 4,096 parallel envs in IsaacLab — that specific detail checks out. **However**, real-robot validation is on a 7-DoF arm with a **20-DoF anthropomorphic hand**, not a 2-jaw parallel gripper. Calling this "exact setup as brief" is misleading — the embodiment that actually matters for grasp-timing transfer (gripper DOF/kinematics) is completely different from the AR4's parallel jaw. |
| 2 | arXiv:1707.08817 is cited as introducing **DAPG** ("Demonstration-Augmented Policy Gradient"), backing Priority 2, with a "30x sample efficiency improvement on complex dexterous tasks" | **Misattributed citation (real paper, wrong paper)** | arXiv:1707.08817 is real (Vecerik, Hester, Scholz et al. 2017, "Leveraging Demonstrations for Deep RL on Robotics Problems with Sparse Rewards") but it is a **DDPG**+demonstrations paper, not DAPG, and does not report a 30x figure or test dexterous-hand tasks (its own experiments are on insertion tasks). The actual DAPG paper is **Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep RL and Demonstrations," RSS 2018, arXiv:1709.10087** — confirmed real, and the 30x sample-efficiency claim on a 24-DoF hand genuinely belongs to *that* paper. The junior report cited the wrong arXiv ID for its own named technique and attached that paper's statistics to the wrong citation. This is a load-bearing error since DAPG anchors Priority 2. |
| 3 | arXiv:1812.03201 Johannink et al. 2019, Residual RL, real-world block-assembly-with-contacts validation | **Verified accurate** | Confirmed: decomposition into base controller + RL residual, real-world block assembly with contacts/unstable objects, authored by Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Aparicio Ojea, Solowjow, Levine. |
| 4 | arXiv:2204.12844 Beltran-Hernandez et al. 2022, curriculum + domain randomization, "86% real-world success... ±0.01mm tolerance... <1/5 training time" | **Verified accurate** | All three figures (86%, ±0.01mm, <1/5 training time) confirmed present in the actual abstract, word-for-word close. Best-verified citation in the report. |
| 5 | Andrychowicz et al. 2017, Hindsight Experience Replay, NeurIPS, Fetch-arm pushing/sliding/pick-and-place | **Verified accurate** | Confirmed real, correct venue, correct task description (abstract explicitly lists pushing, sliding, pick-and-place on a robotic arm). Standard, correctly-used citation. |
| 6 | arXiv:2604.01414 "Learning When to See and When to Feel," "82% success on three contact-rich assembly tasks, outperforming vision-only by 14%" | **Verified accurate** (with one loose word) | Confirmed real (Jiuzhou Lei, Chang Liu, Yu She, Xiao Liang, Minghui Zheng). Figure caption in the actual paper: "we evaluate across three contact-rich tasks... our method achieves 82% average success rate," and 14% improvement over strongest baseline — both numbers check out. Minor overreach: the three tasks (egg-boiler lid, weight-based bottle placement, twist-connector pull-out) are not "assembly" tasks as the report labels them — a cosmetic mischaracterization, not a fabrication. |
| 7 | arXiv:2211.10957 (Mosbach & Behnke) + arXiv:2206.13966 (Dext-Gen), backing the claim "pre-training on primitive shapes (sphere/cube/cylinder) gives 3-5x generalization" and "sphere is the easiest shape to grasp," used to argue **against** switching to a cube | **Unverifiable / likely fabricated specific claim** | Both papers are real and roughly on-topic (grasping diverse object geometries; sparse-reward dexterous grasping with orientation control), but **neither abstract contains** the primitive-shape-pretraining claim, the 3-5x figure, or any statement that spheres are the easiest shape. This is a load-bearing claim for a specific, explicitly-asked-about brief question (object geometry effects) and it is not actually supported by either cited source. Should be treated as unsupported until verified against full text (which itself did not surface it either). |
| 8 | arXiv:2001.00449 (Neunert et al., DeepMind), "discretizing gripper velocity to {-1,1} significantly outperforms continuous Gaussian policies" | **Unverified specific claim on a real paper** | Paper is real (Neunert, Abdolmaleki, Wulfmeier, Lampe, Springenberg, Hafner, Romano, Buchli, Heess, Riedmiller) and is about hybrid discrete/continuous RL generally, but the abstract does not state the specific {-1,1} vs. Gaussian finding attributed to it. Possible over-specific extrapolation. |
| 9 | arXiv:2509.22149 DemoGrasp, single-demonstration dexterous grasping | **Verified accurate** | Confirmed real (Yuan, Huang, Wang, Mao, Xu, Lu), matches description (single demo, wrist-pose/joint-angle editing as 1-step MDP, Shadow Hand results). |
| 10 | arXiv:2509.23567 GES-UniGrasp used to back a "Stage 1: gripper pre-positioned... Stage 2: add reaching..." curriculum recipe (G5) | **Citation/content mismatch** | Paper is real (Xu, Zhu, Gu, Tang) but is actually about a geometry-based *expert-selection* framework over a 773-object dataset (99.4%/96.3% train/test success), not a staged pre-positioned-gripper curriculum. The specific 3-stage curriculum recipe in the report appears to be the junior's own invention, loosely stapled onto this citation rather than drawn from it. |
| 11 | arXiv:2509.26000 (asymmetric actor-critic) and arXiv:2603.04531 (PTLD) | **Verified accurate** | Both real, both roughly correctly characterized (privileged-signal actor-critic accepted ICML 2026; tactile latent distillation reporting 182%/57% improvements over baselines not vision-only, close enough to the report's general framing). |

**Net read:** most of the ~49 references are real, findable papers — this is not
wholesale fabrication. But at least one clear **misattribution** sits directly
under the report's #2 recommendation (DAPG cited via the wrong paper, with a
borrowed statistic), one **unsupported claim** sits under a specific brief
question the junior was asked to answer directly (object geometry), and the
flagship citation for Priority 1 is subtly **oversold** ("exact setup as
brief") on the one dimension (gripper embodiment) that actually matters for
transfer. This matches the pattern flagged in the task instructions from
prior passes this session — not raw invention of nonexistent papers, but
real citations doing more rhetorical work than their actual content supports.

## 2. Reasoning-soundness assessment

**Strengths:**
- Broad-coverage requirement from the brief is genuinely met — MBRL,
  off-policy, exploration, reward design, demonstration/imitation, curriculum,
  hierarchical/modular, asymmetric actor-critic, domain randomization,
  distillation, VLA, and trajectory-optimization families are all covered,
  matching the brief's explicit list almost one-to-one.
- The report is honest about overlap in one place: it correctly notes that
  Residual RL is conceptually close to the not-yet-evaluated Experiment 8
  (IK guidance) and would require "re-architecture, not just tuning" to
  differ meaningfully — this is exactly the kind of overlap-checking the
  brief asked for, done correctly, once.
- The "What NOT to do" section is a reasonable, low-risk set of exclusions.

**Weaknesses / gaps:**
- **Missing the most parsimonious alternative.** The brief explicitly asks
  the report to flag "if the literature suggests the action space itself
  (e.g., a discrete/scripted grasp phase... is often the actual fix)." A
  very common, well-established pattern in manipulation RL (including many
  Isaac Gym/Isaac Lab grasp tasks) is to remove gripper-closing from the
  learned action space entirely — trigger it via a scripted/heuristic rule
  (e.g., close when end-effector-to-object distance and orientation cross a
  threshold) rather than have the policy learn *when* to close. This is
  arguably the single cheapest, most direct experiment given the specific
  failure mode described (arm learns to reach, gripper timing never
  emerges), and it doesn't require new demo data or new architecture. G6
  gestures near this (discrete vs. continuous gripper action) but never
  states the "just script it, take it out of the RL loop" option plainly —
  a real gap relative to what the brief asked for.
- **No specific literature identified on the exact failure mode.** The
  brief asks for "literature on why parallel-jaw grasp timing... is hard for
  RL to learn from reward shaping alone." The report answers this by
  assembling inferences across general technique families rather than
  citing any paper that specifically diagnoses this local optimum. That's a
  reasonable fallback given a real literature gap, but the report should
  have said so explicitly rather than implying (via the "Common thread"
  synthesis in Part 3) that the literature directly explains this failure
  mode — it mostly doesn't; the report's synthesis there is its own
  inference, presented with more confidence than the underlying sources
  actually provide.
- **Overreach on the object-geometry recommendation** (see citation #7
  above): "switching to a cube would actually make the problem harder" is
  stated as a literature-backed implication when the two cited sources
  don't contain the claim it rests on.

## 3. Fit to this project's specific constraints

Re-reading the brief's "Constraints that matter" and "already tried" list:

- **Priority 2 (demonstration-guided learning) is genuinely a different axis.**
  All 8 prior experiments were pure RL (reward/hyperparameter engineering on
  a flat policy); none used any demonstration or scripted-rollout data. This
  holds up as a sound recommendation *in substance*, independent of the
  citation mix-up — DAPG-style demo augmentation, correctly cited to
  Rajeswaran et al. 2018 (RSS), is real, established, and untried here. The
  citation needs to be fixed before this goes in any written record, but the
  underlying suggestion survives the fix.

- **Priority 1 (hierarchical arm-hand decomposition) is structurally novel
  but has a real overlap the report doesn't surface.** Experiment 8 already
  introduced phase structure into the *reward* (5 Cartesian waypoints:
  pre-grasp/grasp/lift/transit/place). Priority 1 is the same phase-structure
  idea moved from reward-shaping into policy *architecture* (separate
  arm/hand agents with phase-gated rewards). That's a legitimate and
  meaningfully different lever, but the report presents Priority 1 as though
  it's arriving from a clean slate, without connecting it to the
  already-in-flight Experiment 8 or reasoning about what Experiment 8's
  (still pending) outcome should tell us about whether the phase idea itself
  is sound before committing to the heavier hierarchical-architecture
  rewrite. A more careful report would have said: "if Experiment 8 fails
  because reward-based phase cues can't force a policy to act differently
  by phase, that specifically motivates Priority 1 (moving phase separation
  into the architecture); wait for Experiment 8's result before committing
  the effort."
- **Embodiment mismatch undercuts confidence in Priority 1's "high expected
  impact."** The flagship paper's real robot validation is a 20-DoF
  anthropomorphic hand, not a 2-jaw gripper. Whether "arm agent / hand agent"
  decomposition delivers the same benefit on a 2-DOF binary gripper (where
  there's much less low-level hand-control complexity to decouple from
  reaching) is untested by the cited source and not addressed by the report.
- **Object-geometry advice (keep the sphere) is stated with more confidence
  than its citations support** — see above. This bears directly on a
  concrete decision currently on the table in the brief ("a proposal is on
  the table to switch to a cube"), so the unsupported confidence here has
  real decision-relevance, not just cosmetic risk.

## 4. Final verdict

**Adjust, do not accept as-is or reject outright.**

The top-level logic — that 8 consecutive reward/hyperparameter-engineering
attempts on a flat PPO policy have been exhausted, so the next move should
change something structurally different in kind (architecture or data, not
just reward terms) — is sound and is the correct read of the situation
regardless of the citation issues. Priority 2 (demonstration-guided
learning) and Priority 1 (hierarchical decomposition) both genuinely differ
from everything tried so far and are reasonable candidates.

Before acting on this report, three corrections should be made:
1. Fix the DAPG citation: cite Rajeswaran et al., "Learning Complex Dexterous
   Manipulation with Deep RL and Demonstrations," RSS 2018, arXiv:1709.10087
   — not Vecerik et al. 2017 (arXiv:1707.08817), which is a different,
   related-but-distinct paper.
2. Downgrade the "exact setup as brief" framing for Priority 1's flagship
   citation to an accurate one: same simulator/scale (IsaacLab, 4096 envs,
   PPO), **different embodiment** (20-DoF dexterous hand vs. 2-jaw parallel
   gripper) — and treat "expected impact: high" as a hypothesis, not a
   transferred result.
3. Drop or clearly re-flag the object-geometry claim ("sphere is easiest,
   don't switch to cube") as the junior's own inference, not a
   literature-backed finding — the two cited sources don't contain it.

With those corrections, Priority 2 (demo-guided learning via a scripted-IK
rollout dataset) is the more defensible next experiment to run first: it's
cleanly novel relative to all 8 falsified experiments, lower implementation
risk than a full multi-agent architecture rewrite, and its core citation,
once corrected, is a well-established and heavily-validated technique.
Priority 1 remains worth pursuing but should wait on Experiment 8's result
and should be scoped with the embodiment caveat in mind. Also worth adding,
not currently in the report: try scripting the gripper-close trigger
(distance/orientation threshold) as a near-zero-cost experiment before
either priority, since it directly targets "policy never learns grasp
timing" without requiring new data or architecture.

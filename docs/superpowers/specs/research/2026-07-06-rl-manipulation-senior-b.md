# Research: PPO-specific grasp-timing literature, reward-hacking diagnosis, force curricula, and Isaac Lab reward-function comparison

Dispatched as a deliberately narrower, deeper follow-up to the two existing
research docs (`2026-07-06-grasp-acquisition-literature-junior.md` and
`-senior-review.md`), which already cover MBRL, off-policy methods, general
exploration, reward design, DAPG, residual RL, curriculum learning
(task-difficulty axis), hierarchical/modular policies, asymmetric
actor-critic, domain randomization, distillation, VLA, and the scripted
gripper-close-trigger idea. This doc does not re-cover any of those.

Methodology: four parallel research passes (one per topic below), each run
by a `general-purpose` subagent with WebSearch/WebFetch access (I do not have
those tools directly as a `senior-engineer` agent). I then independently
re-verified the most load-bearing new citations myself — via direct `curl`
fetches of arXiv abstract pages and, where possible, by reading the actual
Isaac Lab source installed at `/home/saps/IsaacLab` and fetching IsaacGymEnvs/
ManiSkill3 source directly from GitHub — rather than trusting the subagents'
self-reports. Verification results are noted inline. Where a subagent flagged
something as unverified/unreachable, that flag is preserved rather than
silently dropped.

---

## 1. PPO-specific manipulation-RL literature

**Direct academic failure-mode analysis of PPO on bimodal (open/close)
discrete-continuous grasping specifically: not found.** Multiple search
angles (PPO + gripper + hybrid/bimodal action + local optimum + exploration
collapse) surfaced only tangential hits. This appears to be a genuine gap,
consistent with the existing docs' own finding that the literature mostly
addresses this failure mode via inference across adjacent technique families
rather than a dedicated diagnosis paper.

**Li et al., *Sensors* 2025, 25(17):5253 (DOI 10.3390/s25175253)** — already
in this repo's research history (SA-PPO paper), re-confirmed here
independently: baseline PPO 92% vs. their simulated-annealing-PPO 98% grasp
success, steps-per-grasp 154→143, real-robot validated on an AUBO-i5 arm.
Confirmed again that this paper is about 6-DoF joint-trajectory planning
around obstacles to reach a grasp pose, with grasp success treated as
scripted/binary at contact — **it does not analyze the bimodal
open/close-gripper problem itself**, and does not discuss entropy collapse.
Restating this because a fresh search pass independently converged on the
same paper as the field's strongest available quantification of "PPO gets
stuck in local optima under sparse grasp reward," with no better candidate
found this round either.

**Neunert et al., "Continuous-Discrete Reinforcement Learning for Hybrid
Control in Robotics," arXiv:2001.00449 (PMLR v100 / CoRL 2019).** Real and
on-topic (explicitly uses gripper open/close/stay as a canonical hybrid-
action example), but **could not verify from primary text** whether it's
built on PPO or DeepMind's usual MPO/DMPO lineage, nor the specific
quantitative numbers — the existing docs already flagged the same
uncertainty. Still unresolved after this pass; treat as "real paper, on-topic
category, unverified fit to PPO specifically."

**Verdict on Q1:** no new citation resolves this. The absence itself is the
finding — worth stating plainly in any future design doc rather than
implying a paper exists that doesn't.

### Isaac Gym / Isaac Lab's own shipped reward design — verified directly from source, not just cited

This is the strongest, most concrete result of the whole research pass. I
independently confirmed the following by reading the actual installed Isaac
Lab source at `/home/saps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/` (both `lift_env_cfg.py` and `mdp/rewards.py`) —
this is not a citation, it's the literal shipped code:

```python
# lift_env_cfg.py, RewardsCfg
reaching_object                  = RewTerm(func=object_ee_distance, params={"std": 0.1}, weight=1.0)
lifting_object                   = RewTerm(func=object_is_lifted,   params={"minimal_height": 0.04}, weight=15.0)
object_goal_tracking              = RewTerm(func=object_goal_distance, params={"std": 0.3,  "minimal_height": 0.04}, weight=16.0)
object_goal_tracking_fine_grained = RewTerm(func=object_goal_distance, params={"std": 0.05, "minimal_height": 0.04}, weight=5.0)
action_rate = weight=-1e-4 ; joint_vel = weight=-1e-4

# mdp/rewards.py — object_goal_distance's actual body:
return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
```

Two mechanistically load-bearing facts, both directly read from source:

1. **There is no gripper-closing or grasp-contact reward term at all.**
   Grasping is never directly rewarded in Isaac Lab's own recipe — it
   emerges only as a physical prerequisite for the much-larger lift/goal
   terms (15+16+5 = 36 of positive weight) to ever fire, since `reaching_object`
   alone caps at weight 1.0.
2. **`object_goal_tracking` and its fine-grained variant are multiplicatively
   gated on the binary `object_pos_w[:,2] > minimal_height` condition** — 21
   of ~37 available positive-reward weight is literally zero until the
   object clears 4cm. A policy that reaches and camps next to the object
   without lifting cannot earn the majority of the reward budget, no matter
   how long it holds that pose.
3. The gripper action space uses `mdp.BinaryJointPositionActionCfg`
   (confirmed at `config/franka/joint_pos_env_cfg.py:37-41`,
   `open_command_expr={"panda_finger_.*": 0.04}`,
   `close_command_expr={"panda_finger_.*": 0.0}`) — collapsing the
   discrete/continuous hybrid-action problem for the gripper into a policy
   output that only has to bias toward one of two joint targets, not learn a
   continuous aperture trajectory. (Per this repo's own ROADMAP, this
   specific recommendation was already checked and found moot here — this
   env's gripper action is already binary.)

**IsaacGymEnvs `FrankaCubeStack`** — I independently re-fetched
`isaacgymenvs/tasks/franka_cube_stack.py::compute_franka_reward` and
`cfg/task/FrankaCubeStack.yaml` directly from
`github.com/isaac-sim/IsaacGymEnvs` (main branch) myself via `curl`, matching
what one research pass reported:

```python
dist_reward  = 1 - tanh(10*(d + d_lf + d_rf)/3)                 # reach, ungated, uses fingertip distances directly
cubeA_lifted = (cubeA_height - cubeA_size) > 0.04                # binary
lift_reward  = cubeA_lifted
align_reward = (1 - tanh(10*d_ab)) * cubeA_lifted                # GATED on lift flag
dist_reward  = max(dist_reward, align_reward)
stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA   # requires gripper RELEASED
# weights: distRewardScale=0.1, liftRewardScale=1.5, alignRewardScale=2.0, stackRewardScale=16.0
```

Same pattern again: no grasp-force reward term; downstream reward gated on
lift; and notably the *final* success bonus requires the gripper to have
released, the structural opposite of rewarding sustained holding.

**ManiSkill3** (Tao, Xiang, Shukla, Qin et al., "ManiSkill3: GPU Parallelized
Robotics Simulation and Rendering for Generalizable Embodied AI,"
arXiv:2410.00425 — title/authors independently re-verified via arXiv
metadata fetch) `PickCube-v1`'s `compute_dense_reward`, fetched live from
`github.com/haosulab/ManiSkill`:

```python
reward = 1 - tanh(5 * tcp_to_obj_dist)                # reach
reward += is_grasped                                   # ungated additive — the one exception to "gate everything"
reward += (1 - tanh(5 * obj_to_goal_dist)) * is_grasped         # place reward GATED on grasp
reward += (1 - tanh(5 * ||qvel||)) * is_obj_placed              # stillness reward GATED on already-placed
reward[success] = 5
```

This is the one external precedent with an *ungated* additive grasp term
(structurally like this repo's `contact_grasp_bonus`) — but even here, the
very next reward stage (`place_reward`) is still gated behind that grasp
flag, and the "reward for being still" term is gated behind having *already
placed* the object, i.e. it does not reward freezing right after grasp the
way an ungated stillness-adjacent term could.

**Community corroboration (anecdotal, not a controlled study):** one
research pass found `isaac-sim/IsaacLab` GitHub Discussion #1697 and Issue
#1270, where users reimplementing the lift task (in one case as Direct RL
instead of the shipped Manager-Based recipe) reproduce this project's exact
symptom — arm approaches but never grasps/lifts — with no maintainer
diagnosis captured in the thread. I did not re-fetch these threads myself to
verify exact wording; treat as suggestive, not a citable finding.

---

## 2. Reward-hacking / local-optima diagnosis literature specific to grasping

**No dedicated paper names and taxonomizes "approach/contact but never
grasp-and-lift" as its own category** (the way "reward hacking" or
"catastrophic forgetting" are named phenomena). But one paper is a close,
independently-verified match to this project's exact "reach, then freeze"
sub-signature:

**Mao, Xu, Sun, Miller, Layeghi, Mistry, "Learning Long-Horizon Robot
Manipulation Skills via Privileged Action," arXiv:2502.15442 (Feb 2025,
arXiv preprint, no venue found).** I independently re-fetched the paper's
full HTML text (via ar5iv) myself and located the exact sentence:

> "The robot end-effector remains at the center of the object and fails to
> lift, as this behavior is the local optimum of the reward function that
> minimize object to end-effector distance."

This is a verbatim, independently-confirmed match (I found it myself in the
paper's own text, not just trusting a subagent's paraphrase) for exactly the
"reach then freeze" signature this project has repeatedly observed. Their
fix is privileged actions (relaxed collision constraints + virtual forces
during simulation training only) plus curriculum, not something directly
transplantable to this project (privileged actions require simulator-level
constraint relaxation, a bigger architectural lift than a reward tweak) —
but the diagnosis itself is a genuine, citable match: a dense proximity term
alone creates exactly the local optimum this project keeps rediscovering
empirically, independent of PPO specifics.

**An, De Vincenti, Ma, Hutter, Coros, "Collaborative Loco-Manipulation for
Pick-and-Place Tasks with Dynamic Reward Curriculum," arXiv:2509.13239 (Sep
2025) — verified real via arXiv metadata (title/authors/abstract match).**
Reports PPO "repeatedly exploiting" an already-mastered reward stage instead
of progressing to the next stage, attributing it to the next stage's
required actions lying in a low-probability region of the policy's current
distribution. Bimanual legged-manipulation context, not tabletop grasping —
relevant as a second, independently-real instance of "PPO re-farms an
already-solved reward stage rather than risk the next one," which is
mechanistically close to this project's "grasp achieved, no lift attempted"
signature (contact_grasp_bonus saturated near max while lifting_sphere/
lift_height_progress stayed at noise).

**Jung, Tao, Bowman, Zhang, Zhang, "Physics-Guided Hierarchical Reward
Mechanism for Learning-Based Robotic Grasping," arXiv:2205.13561 (2022) —
verified real via arXiv metadata (title/authors/abstract match).** The
specific "policy greedily exploits holding rather than exploring
orienting" quote reported by the research pass could **not** be independently
re-confirmed by me (PDF/HTML fetch of body text failed) — flagging this
explicitly as **unverified against primary text**, same caveat the subagent
itself raised. Treat the paper's existence and general topic (hierarchical
staged reward for grasping) as confirmed; do not cite the specific quoted
mechanism until someone gets the full text.

**PPO entropy collapse specific to manipulation:** searched, found nothing
beyond the SA-PPO paper above. Searches for "entropy collapse" + PPO +
manipulation returned almost exclusively LLM/RLHF-domain results (Ahmed et
al. 2018, arXiv:1811.11214; Cui et al. 2025, arXiv:2505.22617; Zhang et al.
2025, arXiv:2506.05615) — none robotics-specific. This is very likely a real
gap, not a search-thoroughness failure (multiple query angles converged on
the same non-result).

**Grasping-specific reward-hacking case studies beyond the well-known
Christiano et al. 2017 "gripper between camera and object" anecdote:** none
found. The "closes gripper beside the object" sub-signature this project
observed (Experiment: dense grasp bonus, falsified) does not appear as a
named case study anywhere findable — it is this project's own empirical
finding, not a literature-confirmed pattern with a name.

---

## 3. Curriculum-of-contact-force / hold-duration literature

**Directly matching work: none found**, after an extensive multi-angle
search (force-threshold curricula, hold-duration curricula, "grasp-then-lift"
staged reward design, secure-grasp-under-load curricula). This is reported
as a genuine, explicit gap by the research pass, and I have no independent
verification to add beyond it, since it's a negative result (nothing to
re-verify).

Closest related work found, each explicitly **not** a match to the narrow
"ramp required force/duration" ask:

- **Chen et al. 2025, "ClutterDexGrasp," arXiv:2506.14317 (CoRL 2025)** —
  genuinely uses a force threshold as a literal curriculum axis, but it's a
  *safety cap that tightens* (200→50, penalizing excess force) to prevent
  damage, the inverse of a *minimum-force requirement* that rises to force a
  secure grasp. Not applicable to a "won't apply/sustain force" problem.
- **Christen et al., "Dexterous Pre-grasp Manipulation for Human-like
  Functional Categorical Grasping," arXiv:2307.16752 (CVPR 2024)** — closest
  conceptual analog: a genuine 3-stage curriculum where Stage 1 excludes
  lifting from the success criterion entirely and Stage 2 requires it. This
  supports the general "stage contact separately from lift" idea (which this
  project has already tried multiple times, via curriculum-gated dense
  lift-height reward and the ContactSensor-then-lift sequencing) but stages
  on binary inclusion/exclusion of the lift criterion, not on a continuous
  force-magnitude or duration threshold ramp.
- Several others (Xiao et al. 2505.18994, a Frontiers in Neurorobotics 2023
  assembly-curriculum paper, ForceGrip arXiv:2503.08061) were checked and
  found to stage spatial/task difficulty or replay-buffer sampling
  probability, not force/duration thresholds, despite titles suggesting
  otherwise.

**Verdict:** any force- or duration-threshold curriculum for this project's
"grip achieved, never lifts" problem would be a novel design, not a
literature transplant. Christen et al.'s binary-inclusion staging is the
best available anchor if a curriculum is pursued, but it does not validate a
continuous force/duration ramp specifically — that would need to be
justified from first principles (or as an experiment with its own falsifiable
prediction), not cited as an established technique.

---

## 4. Direct reward-function comparison against `tasks/ar4/mdp.py` / `pickplace_ik_guided_env_cfg.py`

This section is grounded in code I read directly (`tasks/ar4/mdp.py`,
`tasks/ar4/pickplace_ik_guided_env_cfg.py:133-190`) cross-checked against
Isaac Lab's own installed source and IsaacGymEnvs/ManiSkill3 fetched live
from GitHub — not paper claims.

**This repo's current `RewardsCfg` (5 substantive terms, all additive, none
gated on each other):**

| Term | Weight | Fires | Gated on? |
|---|---|---|---|
| `ik_guided_path_bonus` | 25.0 | running-max milestone (proximity + live IK-joint-match across 5 waypoints) | nothing |
| `gripper_schedule_bonus` | 0.1 | per-step 1/0, gripper state matches expected waypoint stage | nothing |
| `contact_grasp_bonus` | 20.0 | per-step 1/0, bilateral jaw force > 0.05N | nothing — sustained every step it holds, regardless of lift/path progress |
| `stillness_penalty` | 2.0 (returns −1, so net −2) | per-step, only if grasped AND object hasn't moved >5mm in 25 steps | grasped-only, but this is the *only* term gated on grasp |
| `action_rate` / `joint_vel` | −1e-4 each | negligible regularizers | — |

**Isaac Lab lift task, IsaacGymEnvs FrankaCubeStack, and ManiSkill3 PickCube
all share a structural pattern this repo's reward does not**: every one of
them gates the *next-stage* reward (goal-tracking, alignment, or place)
behind the *previous* achieved state (lift, in the first two; grasp, in
ManiSkill3). None of them has an ungated, sustained-every-step term that pays
out identically whether the policy is mid-task or frozen. ManiSkill3 is the
only one with an ungated *grasp* term, structurally closest to this repo's
`contact_grasp_bonus` — but even ManiSkill3 gates the very next term
(`place_reward`) on that same grasp flag.

**This repo's reward is the structural outlier**: `contact_grasp_bonus`
(weight 20, the single largest sustained per-step term) is fully earnable
and holdable independent of whether `ik_guided_path_bonus` (the term that
actually encodes lift/carry/place progress) ever advances past its early
waypoints. The only counter-pressure is `stillness_penalty` at net −2/step —
against +20 (grasp) + up to +0.1 (gripper schedule, itself satisfied by
staying closed) per step, freezing after grasp nets roughly **+18/step**
indefinitely once contact is achieved, a roughly 9:1 reward-rate advantage
for freezing over the one term designed to discourage it. This is a direct
arithmetic read of the actual configured weights, not a citation — but it is
the most concrete, directly falsifiable candidate mechanism this research
pass produced for the specific "reach, grip, freeze" signature recorded
repeatedly in ROADMAP.md.

Note on the ROADMAP's own earlier claim that this repo's reward weights are
"an exact, verified copy of Isaac Lab's own shipped, working Franka+DexCube
lift example" (recorded under the PD-gain-rescale experiment): that
comparison was evidently made against an earlier reward version (the
staged reach/grasp/lift/goal weights of 0.1/0.2/0.3/0.4 in `_raw_lift_progress`,
which do echo Isaac Lab's relative ordering), not the current
`RewardsCfg` in `pickplace_ik_guided_env_cfg.py`, which has since diverged
substantially (adding an ungated, always-on `contact_grasp_bonus` and
replacing the staged/gated structure with the IK-waypoint milestone bonus).
Flagging this as a genuine drift between the "exact copy" characterization
on record and the reward function actually in place today — worth resolving
explicitly rather than carrying the stale characterization forward.

---

## Feedback on current design

**This is a different, more specific recommendation than the existing two
docs' scripted-gripper-trigger / DAPG / hierarchical-decomposition
proposals — it's a direct structural fix to the reward function itself,
motivated by point 4 above, not a new technique family.**

The concrete, falsifiable candidate change: **gate (or decay) the ongoing
value of `contact_grasp_bonus` and/or `gripper_schedule_bonus` on lift/path
progress, rather than leaving them as unconditional per-step rewards.**
Two concrete implementation options, both directly modeled on the verified
external precedents above:

1. **Isaac-Lab-style gate (multiplicative on the downstream term, matching
   `object_goal_distance`'s and `franka_cube_stack.py`'s pattern):**
   keep `contact_grasp_bonus` as-is, but make a meaningful share of
   `ik_guided_path_bonus`'s reward *require* `contact_grasp_bonus` to already
   be active before waypoints 2+ (lift/transit/place) can be reached/scored
   — this project's waypoint-index mechanism (`env._path_waypoint_idx`) already
   exists and could plausibly be made conditional on grasp state, though that
   is a design decision, not something to implement unilaterally here.
2. **ManiSkill3-style time-decay on the grasp term itself:** instead of (or
   in addition to) gating the downstream term, make `contact_grasp_bonus`'s
   own per-step value decay the longer the object has been held without net
   height gain — directly targeting the ~9:1 reward-rate imbalance computed
   above, rather than relying solely on `stillness_penalty` (net −2) to
   outweigh a steady +20.

Both are grounded in real, independently-verified reward functions (Isaac
Lab's own shipped task, IsaacGymEnvs, ManiSkill3) that this project's current
`RewardsCfg` structurally departs from in exactly the dimension (gating vs.
ungated sustained reward) that plausibly explains its unique "grip achieved,
never lifts" outcome — none of the three external precedents checked have an
ungated term this large relative to their downstream/goal reward. This is a
reward-function-composition fix, cheaper and more targeted than DAPG,
hierarchical decomposition, or a force/duration curriculum (none of which
have a literature-verified recipe for this specific gap either, per sections
1-3 above) — but it is a genuine reward redesign, which per this repo's own
`superpowers:systematic-debugging` Phase 4.5 discipline (6 falsified
reward/optimization-axis attempts already on record) should be scoped and
flagged as a real design decision rather than treated as a trivial tweak.

Secondary, lower-confidence findings that don't rise to "change the reward
now" but are worth recording: (a) Mao et al. 2025 (arXiv:2502.15442)
independently confirms "dense proximity-only reward → reach-then-freeze
local optimum" as a real, citable phenomenon beyond this project's own
empirical record, reinforcing (not superseding) the existing docs' emphasis
on adding non-proximity signal; (b) no literature supports a force/duration
curriculum as an established technique — if pursued, treat it as a novel,
falsifiable experiment, not a literature-backed recipe; (c) no PPO-specific
bimodal-gripper-action failure-mode paper exists as far as this pass could
determine — stop searching for one rather than re-running this same query
again in a future pass.

## References

1. Isaac Lab `lift_env_cfg.py` / `mdp/rewards.py`
   (`isaaclab_tasks/manager_based/manipulation/lift/`), read directly from
   the local installation at `/home/saps/IsaacLab`, 2026-07-06. Not a
   citation — primary source code.
2. IsaacGymEnvs `franka_cube_stack.py` / `cfg/task/FrankaCubeStack.yaml`,
   fetched directly from `github.com/isaac-sim/IsaacGymEnvs` (main branch)
   via `curl`, 2026-07-06.
3. Tao, S., Xiang, F., Shukla, A., Qin, Y., Hinrichsen, X., Yuan, X., Bao,
   C., Lin, X., Liu, Y., et al. "ManiSkill3: GPU Parallelized Robotics
   Simulation and Rendering for Generalizable Embodied AI." arXiv:2410.00425
   (2024). `PickCube-v1` reward fetched from `github.com/haosulab/ManiSkill`.
4. Mao, X., Xu, Y., Sun, Z., Miller, E., Layeghi, D., Mistry, M. "Learning
   Long-Horizon Robot Manipulation Skills via Privileged Action."
   arXiv:2502.15442 (2025). Quote independently verified against full text.
5. An, T., De Vincenti, F., Ma, Y., Hutter, M., Coros, S. "Collaborative
   Loco-Manipulation for Pick-and-Place Tasks with Dynamic Reward
   Curriculum." arXiv:2509.13239 (2025). Verified real via arXiv metadata.
6. Jung, Y., Tao, L., Bowman, M., Zhang, J., Zhang, X. "Physics-Guided
   Hierarchical Reward Mechanism for Learning-Based Robotic Grasping."
   arXiv:2205.13561 (2022). Verified real; specific quoted mechanism
   **not** independently confirmed against full text — treat with caution.
7. Li et al. "SA-PPO for robotic arm grasping." *Sensors* 2025, 25(17):5253,
   DOI 10.3390/s25175253. Re-confirmed (already on record from a prior pass).
8. Neunert, M., Abdolmaleki, A., Wulfmeier, M., Lampe, T., Springenberg, J.
   T., Hafner, R., Romano, F., Buchli, J., Heess, N., Riedmiller, M.
   "Continuous-Discrete Reinforcement Learning for Hybrid Control in
   Robotics." arXiv:2001.00449 (PMLR v100 / CoRL 2019). Real; fit to PPO
   specifically unverified (already flagged in prior docs, unresolved here).
9. Chen et al. "ClutterDexGrasp." arXiv:2506.14317 (CoRL 2025). Force
   threshold is a safety cap, not a minimum-force curriculum — not a match.
10. Christen, S., et al. "Dexterous Pre-grasp Manipulation for Human-like
    Functional Categorical Grasping." arXiv:2307.16752 (CVPR 2024). Verified
    via full text; closest available curriculum analog, but stages binary
    lift-inclusion, not force/duration.
11. Christiano, P., et al. "Deep Reinforcement Learning from Human
    Preferences." arXiv:1706.03741 (2017). Source of the well-known
    "gripper between camera and object" reward-hacking anecdote — real
    paper, but the specific anecdote's exact wording/algorithm (their arm
    experiments are TRPO/A2C-based) was not independently re-confirmed this
    pass.

**Explicitly unresolved / flagged as gaps rather than findings:** no
PPO-specific bimodal-gripper failure-mode paper found (section 1); no
force/duration-curriculum paper found (section 3); Jung et al. 2022's
specific quoted mechanism unverified (section 2, reference 6).

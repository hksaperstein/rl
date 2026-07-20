# ROADMAP

Living status doc. Update after each completed plan (per
`.superpowers/sdd/progress.md`): append what shipped, refresh open
follow-ups below.

## Built

- **AR4 pick-and-place** (perception + RL training/eval/interactive demo) —
  working end-to-end.

## Known follow-ups

1. **AR4 sphere pick-and-place: grasp/lift never emerges.** Retargeted the
   pick-and-place task from the Cube to the Sphere
   (`tasks/ar4/pickplace_env_cfg.py`) reusing the existing robot/scene/camera
   infra. Two full 1500-iteration training runs — baseline (reward weights
   unchanged from the cube task) and one bounded fallback (`lifting_sphere`
   weight 15.0→25.0, the only variable changed) — produced **identical**
   results: `reaching_sphere` converges to ~0.92-0.93, while
   `lifting_sphere`/`sphere_goal_tracking`/`sphere_reached_goal` converge to
   0.0000 in both cases (small transient exploration noise early on — peak
   ~0.017 on `sphere_reached_goal`, ~0.0018 on `lifting_sphere` — decays to
   0 well before training ends; the policy never reliably discovers the
   behavior). Real eval video
   (densely-sampled, cropped frames across a full episode) confirms why: the
   gripper reaches near the sphere and then holds a **static, open** pose for
   the rest of the episode — it never attempts to close at all. This rules
   out a physical grasp-feasibility/slip explanation (which would show
   attempted closures followed by the object escaping) in favor of an
   exploration problem: the policy converged to a local optimum that
   maximizes the reach reward without ever discovering the
   close-gripper→lift behavior during training.
   - Literature research (delegated: junior researcher + senior citation-
     verification review, all 9 sources confirmed real and resolvable, 3
     found to be somewhat overstated relative to what they actually show)
     confirms a lift-reward weight bump alone is not how this failure mode
     is addressed in published work. Recommended next steps, in priority
     order: **(a)** add a contact-based reward term (gripper-finger contact
     force/points on the sphere, not just position/height) — note this
     likely requires adding an Isaac Lab `ContactSensorCfg` to the gripper
     fingers first, which doesn't exist in this config yet; **(b)** a
     curriculum (reach-only → reach+close-gripper bonus → full lift), per
     Luo et al.'s Dense2Sparse staged-reward result (arXiv:2003.02740) and
     Li et al.'s stage-specific reward decomposition
     (arXiv:2512.10235); **(c)** as a last resort, hierarchical/frozen
     reach-policy-then-grasp-policy training, with the caveat (flagged in
     review) that a reach-only policy may converge to a non-grasp-feasible
     approach pose, forcing joint retraining anyway.
   - Note: the discrete/continuous gripper-action recommendation from the
     initial literature pass turned out to be moot — this env's gripper
     action is already binary open/close
     (`tasks/ar4/env_cfg.py:48`, `BinaryJointPositionActionCfg`).
   - **Explicit decision (flagged by final review as undocumented):** the
     Cube and RectPrism remain in the scene as static props, unchanged, and
     the sphere's new target region (world x≈0.20, y∈[0.28,0.34]) sits on
     top of their positions — the design spec asked for this to be either
     avoided or an explicit kept/removed call made, and neither happened at
     implementation time. Decision, made now: keep both props and accept
     the overlap, mirroring the original cube task's own target region
     (which sat on top of the Sphere/Wedge's static positions and worked in
     production — objects are light, 0.01kg mass, physics naturally
     resolves minor overlap on contact rather than erroring). This is
     currently untested in practice since the policy never lifts the
     sphere at all; if a future grasp fix succeeds, verify placement
     doesn't produce a visibly bad interpenetration as part of that
     follow-up, and revisit (move the target to genuinely empty space, e.g.
     world y≈0.10) if it does.
   - Per the plan, further iteration beyond the one bounded fallback is
     deferred here rather than continued as open-ended tuning.
   - **Follow-up experiment: dense "grasp bonus" reward (falsified).** Tried
     priority-(b)'s simplified form — a static dense reward term
     (`grasp_sphere` in `tasks/ar4/mdp.py`'s `grasp_object_bonus`, adapted
     from Isaac Lab's own `manipulation/cabinet` task's `grasp_handle`
     pattern: rewards closing the gripper only when the EE frame is within
     4cm of the sphere) rather than building full curriculum
     phase-scheduling infra. Design rationale in
     `docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-bonus-design.md`,
     full run data in
     `docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-bonus-report.md`.
     Result: **the term was fully learned but produced reward hacking, not
     the target behavior.** `Episode_Reward/grasp_sphere` climbed from 0 and
     saturated at its theoretical max (~0.284-0.287) well before the
     1500-iteration run ended — the policy reliably learned to close the
     gripper near the sphere. But `lifting_sphere`/`sphere_reached_goal`
     never moved off 0.0000 (same as both prior runs). Real eval video (10
     episodes, frames extracted and visually inspected) confirmed why: 0/10
     episodes show a real grasp+lift — the gripper's fingers visibly close,
     but the sphere sits beside the closed gripper, not between the jaws,
     and stays on the ground for the rest of the episode. Root cause: the
     reward only checks EE-to-object-center distance + gripper closure, with
     no check that the object is actually enclosed between the fingers —
     the policy satisfies it via the already-loose `reaching_sphere` kernel
     (std=0.1) without ever achieving a geometrically correct grasp pose.
     This reward-hacking failure mode is qualitatively different from (and
     worse than) the earlier lift-weight-bump's no-op failure, since keeping
     a trivially-satisfiable dense term in the production reward risks
     entrenching this fake-grasp local optimum against future fixes — so
     **the code change was reverted, not merged** (unlike the lift-weight
     bump, which was kept as a harmless no-op). Only the spec/report docs
     and this ROADMAP entry are kept, as the research record.
     - This is a second falsified dense-shaping-only hypothesis. Per
       `superpowers:systematic-debugging`'s Phase 4.5, this is grounds to
       escalate rather than attempt a third reward-shaping tweak:
       priority-(a) (a `ContactSensorCfg`-based reward, or at minimum a
       stricter geometric check requiring the sphere to be positioned
       between the two finger positions — closer to the cabinet task's
       `align_grasp_around_handle`/`approach_gripper_handle` combination
       than its bare `grasp_handle` distance check) is the recommended next
       step, still undone.
   - **Follow-up experiment: multiplicatively-gated alignment reward
     (falsified, different failure mode).** Implemented exactly the
     stricter-geometric-check recommendation above: extended the `ee_frame`
     `FrameTransformerCfg` with two new target frames on the actual gripper
     jaw links (`gripper_jaw1_link`/`gripper_jaw2_link` — confirmed correct
     against the AR4 URDF, prim-path hypothesis validated on the first smoke
     test with no correction needed), and added `aligned_grasp_bonus`
     (`tasks/ar4/mdp.py`) which multiplicatively gates the closure reward by
     an alignment score (`1 - tanh(centering_dist / 0.01)`, `centering_dist`
     = distance from the sphere to the midpoint of the two fingertip
     frames) — per GRIT's verbatim-confirmed `r_h·α_h` multiplicative
     pattern (arXiv:2604.04138), replacing the prior experiment's additive/
     independently-satisfiable combination. Full design in
     `docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-alignment-design.md`,
     citation-verified research in `docs/superpowers/specs/research/2026-07-05-grasp-alignment-literature-*.md`
     (the senior review caught a fabricated `std=0.02m` claim and two
     misapplied citations in the underlying research before this design was
     finalized), full run data in
     `docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-alignment-report.md`.
     Result: **the gate is not reward-hacked, but also never discovered** —
     `grasp_sphere_aligned` stayed at noise level (max 0.00207, ~0.7% of its
     ~0.284 theoretical max) for the entire 1500-iteration run, a sharp
     contrast with the prior experiment's term saturating near its max by
     iteration ~1300. `lifting_sphere`/`sphere_reached_goal` again never
     left 0.0000. Eval video (10 episodes, dense frame sampling) showed a
     third distinct failure signature: the arm reaches toward the sphere in
     the first ~1s then **freezes into a completely static pose for the
     rest of the episode** (byte-identical geometry from t=1.0s to episode
     end) — this is the *original* exploration-failure signature from the
     very first experiment (reach-then-freeze, no closing attempt at all),
     not the second experiment's "closes beside the sphere" signature. The
     sphere becomes occluded behind the stationary gripper from the fixed
     camera angle — confirmed via a direct numeric rollout check
     (querying the sphere's actual world-frame height across 32 parallel
     envs) to rule out the occlusion being a hidden successful grasp rather
     than a camera-angle artifact of a sphere still resting on the ground.
     Root cause: making the reward correct (requiring true centering, 1cm
     `centering_std` window) came at the direct cost of making it far
     harder to stumble into via random exploration — tight relative to the
     sphere's own ~9mm radius and the gripper's small travel range, so the
     policy's exploration noise essentially never produces the joint
     (position, orientation, closure) combination needed to get any
     nonzero signal from this term. The code change was reverted (not
     merged), matching the second experiment's precedent for an
     ineffective change; spec/report docs kept as the research record.
     - This is now a **third falsified dense-reward-shaping-only
       hypothesis**: (1) lift-weight bump — no-op; (2) additive
       proximity+closure grasp bonus — reward-hacked; (3)
       multiplicatively-gated alignment+closure bonus — structurally
       un-hackable but too sparse to ever be discovered by unguided
       exploration. Per `superpowers:systematic-debugging` Phase 4.5 (3+
       failed fixes → question the architecture, not attempt a fourth
       single-shot tweak), the recommended next steps are no longer
       reward-shaping-only: **(a)** Isaac Lab's `ContactSensor`/
       `contact_forces` infrastructure (confirmed real and available in
       the installed Isaac Lab source — `isaaclab/sensors/contact_sensor/`,
       `isaaclab/envs/mdp/rewards.py:281`), giving a ground-truth "is the
       object actually being touched by both fingers" signal instead of a
       geometric proxy that must be simultaneously correct and
       discoverable; or **(b)** a curriculum/staged-reward approach (reach-
       only → reach+close-gripper bonus with a *looser*, discovery-friendly
       threshold first → tighten the alignment requirement only after
       closure-near-object is already a well-established behavior),
       per the second experiment's literature review (Dext-Gen,
       arXiv:2206.13966, verbatim-confirmed progressive-tolerance-
       tightening pattern). This decision point was flagged back to the
       user rather than attempting a fourth reward tweak unilaterally.
   - **Follow-up experiment: gripper PD-gain rescale (falsified, not a
     reward-shaping attempt this time).** Per user push to keep researching
     rather than only iterate on reward heuristics: literature research
     (junior + independent senior citation review — all 11 citations real
     this round, no fabrications, a first this session — but the headline
     recommendation was found overstated: the cited paper's own three-way
     split says stiff/overdamped gains mainly harm *sim-to-real transfer*,
     not pure in-sim PPO convergence, which is all this repo does; the
     specific rescaled-gain target numbers were the junior's own
     extrapolation, not literature-backed. Full docs in
     `docs/superpowers/specs/research/2026-07-05-grasp-scale-literature-*.md`)
     surfaced a real, concrete comparison: this repo's reward weights are
     already an exact, verified copy of Isaac Lab's own shipped, working
     Franka+DexCube lift example (same functions/weights, same
     `BinaryJointPositionActionCfg` gripper action) — the one dramatic
     difference is physical scale (AR4's gripper: 2.8cm aperture grasping an
     18mm sphere, vs. Franka's 8cm aperture and ~4.1cm cube, each roughly
     2.3-2.85x smaller). Hypothesis tested empirically regardless of the
     weak citation backing (worth a cheap test either way): rescaled the
     gripper actuator's PD gains (`tasks/ar4/robot_cfg.py`,
     stiffness 1000.0→350.0, damping 50.0→30.0, preserving the original
     damping ratio under the softer stiffness — derivation in the reverted
     commit's diff) against the *unmodified baseline reward* (no grasp bonus
     of any kind), to isolate gains as the single variable. Result:
     **no change — a fourth falsified hypothesis.** Full 1500-iteration run:
     `lifting_sphere` never moved off 0.0000 (max 0.0002, pure noise, same
     as every prior run); `reaching_sphere` converged to ~0.94 (comparable
     to, slightly above, prior runs). Eval video (10 episodes) showed the
     same static gripper-near-sphere pose with no lift, matching this
     session's already-documented failure signatures rather than anything
     new. Reverted (no evidence of benefit, and unlike the reward-weight
     no-op precedent, an unverified physical-parameter change isn't worth
     carrying forward without a demonstrated reason). This result is
     actually consistent with the senior review's own correction: the
     underlying paper's claim was about sim-to-real gain transfer, not
     in-sim convergence, so a null result here doesn't contradict the
     literature - it just confirms gains weren't the right axis for this
     specific (already in-sim) problem.
   - **Follow-up experiment: single-object scene + real-camera-observed
     training (paused mid-run by user request, inconclusive — not a fifth
     falsified hypothesis).** Per user direction to try something other
     than a reward/control tweak: built a single-object scene (sphere only,
     `tasks/ar4/pickplace_single_object_env_cfg.py`) and a
     `PerceptionObservationWrapper` (`scripts/_perception_adapter.py`,
     new `--perception` flag on `train.py`) that overrides the
     `sphere_position` observation with a position derived from the real
     `perception_camera` + this repo's classical perception pipeline
     (reward stays privileged - only the observation changes). Along the
     way, fixed a real pre-existing bug: `_perception_adapter.py` was
     hardcoded to a stale `"cube_position"`/`find_by_shape(..., "cube")`
     from before the Cube→Sphere retargeting, silently breaking
     `eval_loop.py --perception` and `interactive_demo.py` (which had the
     same staleness bug inline) - both fixed and renamed to sphere
     terminology. Design in
     `docs/superpowers/specs/2026-07-05-ar4-single-object-camera-training-design.md`,
     full report in
     `docs/superpowers/plans/2026-07-05-ar4-single-object-camera-training-report.md`.
     **Real, load-bearing finding**: this repo's perception pipeline is
     plain serial numpy per-env (not GPU-batched), so a camera-observed
     training run costs roughly **2.7 hours for 1500 iterations at
     num_envs=16** (measured 6.3-7.3s/iteration, collection ~150x the PPO
     learning-phase cost) versus minutes for the privileged-observation
     baseline at num_envs=4096 - and per-iteration sample count is
     inherently ~256x smaller (16 vs. 4096 envs), so raising `num_envs`
     here buys no free parallelism, only proportionally more wall-clock
     cost. The user paused the run at iteration 110/500 (~12 min in) to
     pivot to a contact-sensor experiment instead; partial data (mean
     reward 0→0.91, episode length 12→249, `lifting_sphere` flat at 0.0000
     throughout) shows *something* is being learned but is **far too little
     data (1/14th a full run, 256x fewer samples/iteration) to call success
     or failure**. Code + bug fix kept and committed (real, working,
     verified-correct infrastructure, not a dead end); resuming this
     experiment later needs a deliberately long (~3h) dispatch, not a quick
     check.
   - **Follow-up experiment: ContactSensor-based grasp reward — real
     progress (grip achieved), but lift still doesn't emerge. Also
     surfaced a likely root-cause bug affecting every prior experiment.**
     Per the systematic-debugging Phase 4.5 escalation from four falsified
     reward/control-only hypotheses, replaced the geometric grasp proxy
     with a ground-truth signal: two `ContactSensorCfg` sensors (one per
     gripper jaw, filtered to the sphere specifically) feeding a new
     `grasp_contact` reward that requires real, bilateral contact force
     above a calibrated threshold. Full design in
     `docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`,
     plan in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-implementation.md`,
     full run data in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`.
     - **Two real implementation bugs found and fixed while building this**
       (both empirically discovered via real smoke-test/calibration
       failures, not anticipated on paper): a single wildcard
       `ContactSensorCfg` covering both jaw links can't pair with PhysX's
       per-body filter-count requirement (fixed: one sensor per jaw,
       matching the `dexsuite`/`kuka_allegro` reference pattern more
       closely than the design's original citation); and `net_forces_w` is
       not actually filtered by `filter_prim_paths_expr` at all (it sums
       *any* contact on the body) — the correct field is `force_matrix_w`.
     - **Major finding, bigger than this experiment: `_EE_OFFSET` (the
       link_6-to-jaw-pinch-point offset feeding the `ee_frame` sensor) was
       wrong by 5.4cm** (`0.09` → measured `0.036`, confirmed directly via
       `robot.data.body_pos_w` for the real jaw links). This offset is
       what `reaching_sphere`'s reward has used as its proximity target in
       **every** grasp experiment this session (lift-weight bump, dense
       grasp bonus, alignment gate, PD-gain rescale) — a reward maximizing
       proximity to a point 5.4cm from where the jaws actually meet is a
       plausible deeper explanation for why grasping never emerged across
       all four of those prior falsified hypotheses, independent of
       whatever reward-shaping was layered on top each time. Fixed as a
       prerequisite correctness bug (like the two above), not treated as a
       new hypothesis — but flagged here prominently since it retroactively
       recontextualizes the whole "grasp/lift never emerges" investigation.
     - **Result: `grasp_contact` converged to ~92% per-step sustained
       contact (18.39/20 weighted, max 18.58)** — real, bilateral,
       correctly-filtered contact between both jaws and the sphere,
       sustained for most of the episode. This is a first for this
       session: every prior experiment either never closed on the object,
       closed beside it, or never discovered closure at all. `lifting_sphere`
       still converged to ~0 (max 0.0027); `sphere_reached_goal` still ~0
       (max 0.027); `reaching_sphere` converged lower than prior runs
       (0.727 vs prior ~0.92–0.94) — expected, since it now measures
       against the corrected (true) jaw pinch point rather than the old
       5.4cm-off target, a different and harder proximity criterion.
     - **Real eval (10 episodes, frame-extracted video, all 10 inspected
       directly): 0/10 show a real grasp+lift — fails the 8/10 decision
       gate.** But the failure signature is new, not a repeat of any prior
       one: the arm reaches down within ~1s of every episode and then
       holds a completely static pose with the gripper directly on the
       sphere for the rest of the episode, in all 10 episodes — the
       sphere is visibly never lifted or moved, but (per the `grasp_contact`
       numbers) it is genuinely, sustainedly gripped throughout, not merely
       approached. Best described as "reach, grip, freeze" — the specific
       problem this experiment targeted (does the gripper ever really
       close on the object) appears solved; a new, distinct bottleneck
       (grip achieved, but no subsequent attempt to lift) has taken its
       place.
     - **Not attempting a further reward tweak unilaterally** — per the
       design doc's own instruction and this session's established
       discipline, this is flagged back to the user as a decision point.
       Plausible next directions (not yet tried): a genuine reach→grip→lift
       curriculum (now that grip is reliably achieved, staging lift as the
       next explicit phase may work where it wouldn't before grip was
       solved); investigating whether `lifting_sphere`'s incentive is
       simply too weak relative to the reward already banked by holding a
       safe, static grip; or checking whether the corrected `_EE_OFFSET`
       alone (without the contact-sensor reward) changes anything, to
       isolate which of this run's two real changes drove the
       contact-convergence improvement.
   - **Follow-up experiment: curriculum-gated dense lift-height reward
     (falsified).** Acted on the recommendation above: added a new dense,
     `tanh`-shaped `lift_height_progress` reward term
     (`tasks/ar4/mdp.py`), inert (`weight=0.0`) during phase-1 reach+grip
     training and switched on (`weight=15.0`) at iteration 700 via Isaac
     Lab's own `modify_reward_weight` curriculum term (the same mechanism
     the Franka lift task this repo's rewards were adapted from uses for
     its own curriculum) — timed to this run's own TensorBoard data, where
     `grasp_contact` plateaus by iteration ~600-750. Full design in
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`,
     plan in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-implementation.md`,
     full run data in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`.
     - **The curriculum mechanism itself fired exactly as designed**
       (`lift_height_progress` reads `0.0` at iteration 699, nonzero at
       701 — confirmed directly from the TensorBoard scalars, not just
       assumed from config). But its real-world magnitude was negligible:
       the logged `Episode_Reward` max of `0.0065` is `weight(15.0) ×`
       the mean per-step `tanh` value, so the real per-step `tanh` is
       ~`0.0065/15 ≈ 0.00043`, corresponding to roughly **0.0043mm** of
       real height gain (via `tanh`'s small-angle behavior) — many orders
       of magnitude short of the 21mm `lifting_sphere` actually requires.
       `lifting_sphere` itself never rose above noise (max `0.0027`,
       ~0.01% of steps — comparable in scale to the `0.0016` transient
       blip the ContactSensor baseline showed), including after the
       curriculum switch.
     - **Real eval (10 episodes, frame-extracted video, all 10 inspected
       directly): 0/10 show any real lift** — the same "reach, grip,
       freeze" static-pose signature as the ContactSensor experiment
       before it. Two episodes showed the sphere briefly vanish from a
       single sampled frame each; checking adjacent frames in the same
       episodes confirmed the sphere reappearing at the identical
       ground-level position next to the gripper in both cases — a
       viewing-angle occlusion artifact (the gripper body blocking
       line-of-sight at that specific pose), not a lift, consistent with
       the negligible quantitative height-gain figure above.
     - **Interpretation:** the curriculum window opened too late and/or
       too weak relative to how deeply the static-grip behavior had
       already converged by iteration 700 (`grasp_contact` was already at
       ~17.8/20, essentially its plateau, at the switch point) — the
       remaining ~800 iterations were not enough for a newly-introduced
       `weight=15.0` dense term to meaningfully perturb a policy that
       stable. This rules out "the sparse `lifting_sphere` signal was the
       only problem" as a complete explanation; the entrenchment of the
       static-grip optimum itself is now the more likely bottleneck.
     - **Not attempting a further reward-only tweak unilaterally** — per
       the design doc's own instruction, flagged back to the user.
       Remaining candidates, none yet tried: a genuine hierarchical
       reach-then-grasp-policy split (train lift as a separate policy
       phase, or freeze the reach+grip policy and fine-tune only for
       lift, rather than a single policy learning both simultaneously);
       an earlier and/or much stronger curriculum switch (before the
       static-grip optimum entrenches so deeply, e.g. iteration 200-300
       instead of 700); or questioning whether the gripper's real
       closed-jaw force is physically sufficient to support lifting this
       0.01kg sphere at all — a physical-plausibility question this
       session has not yet directly measured, distinct from every reward-
       design question tried so far.
   - **Follow-up experiment: always-on dense lift-height reward
     (falsified) — rules out curriculum timing, points at PPO entropy
     collapse.** Per user request, removed the curriculum gate entirely:
     `lift_height_progress` active from iteration 0 at `weight=25.0`
     (matching `lifting_sphere`), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`'s
     "Revision" section. Plan in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-implementation.md`,
     full run data in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`.
     - **Result: 0/10 real eval episodes show any lift** — the same
       "reach, grip, freeze" static-pose signature as both prior
       experiments (curriculum-gated and sparse-only). `lift_height_progress`
       did reach a measurably larger real value than the curriculum
       experiment (~0.0141mm of real height gain vs. ~0.0043mm,
       weight-normalized — a genuine ~3.3x increase, not the 5.4x a raw,
       non-weight-normalized comparison first suggested), with an early-
       training bump that faded as `grasp_contact` converged — but both
       figures remain many orders of magnitude short of the 21mm
       `lifting_sphere` requires, and `lifting_sphere` itself never moved.
       This rules out "the curriculum switch came too late" as the sole
       explanation, since removing the gate entirely produced the same
       outcome.
     - **Citation-verified literature research** (junior researcher,
       Google-Scholar-first per user preference, then independent senior
       citation review — full docs in
       `docs/superpowers/specs/research/2026-07-06-lift-reward-literature-junior.md`
       and `-senior-review.md`). The junior's first pass overstated
       several claims (citing two real-but-off-topic multi-objective-RL
       papers as if they specifically documented a "grasp-reward-vs-lift-
       motion conflict," and a fabricated "2-3x safety factor" number)
       — caught independently by both the senior review and the junior's
       own Google-Scholar-first re-pass (convergent signal). What
       survived verification:
       - **The likely mechanism is PPO entropy collapse**, not a specific
         grasp/lift reward conflict: once a safe, reward-sufficient
         behavior is found, policy entropy drops and exploration of
         riskier alternatives (like lifting) effectively stops, even with
         a dense term nudging toward it. A genuinely on-point, verified
         citation: Li et al., *Sensors* 2025, 25(17):5253 (DOI
         10.3390/s25175253), a robotic-arm-grasping PPO paper explicitly
         targeting "local optimum traps" via a simulated-annealing+PPO
         hybrid (SA-PPO) with a dynamically-adjusted learning rate,
         reporting 92%→98% success rate over baseline PPO with real-robot
         validation — the strongest, most directly-applicable citation
         found this session.
       - **Potential-based reward shaping** (Ng, Harada, Russell, ICML
         1999, "Policy Invariance Under Reward Transformations") is real,
         verbatim-confirmed, and offers a genuine theoretical guarantee:
         decomposing reach→grasp→lift as a potential-function chain
         doesn't change what the optimal policy is, unlike the ad-hoc
         curriculum-timing approach already tried twice.
       - **Grip force is very likely not the bottleneck** — not proven by
         the (struck) fabricated safety-factor citation, but independently
         supported by this repo's own measured contact force (~20-30N
         against the sphere's 0.098N weight, from the ContactSensor
         experiment's calibration) and a real, verified comparison (a
         published strawberry-harvesting end-effector, arXiv:2207.12552,
         safely lifts a 5x-heavier object at high acceleration on roughly
         a third of this repo's measured force).
       - **The "multiplicative gating" idea** (only reward lift progress
         while contact is maintained) remains an untested engineering
         hypothesis, not a literature-validated fix — the citations
         originally used to back it didn't hold up under verification.
     - **Not attempting a fourth reward-only tweak unilaterally** — this
       is the third real attempt on the reward/curriculum axis for this
       specific sub-problem (sparse-only, curriculum-gated dense,
       always-on dense). Flagged back to the user with two real,
       literature-backed candidates not yet tried (SA-PPO-style dynamic
       learning-rate adjustment once `grasp_contact` saturates;
       potential-based reward shaping), alongside the previously-named
       architectural options (hierarchical reach-then-grasp-policy
       split).
   - **Follow-up experiment: SA-PPO-style dynamic learning-rate bump
     (falsified).** Per user instruction to try both remaining
     literature-backed candidates, this experiment tested the first one
     in isolation. Reused `model_700.pt` from the always-on-lift run (the
     identical "grip converged, exploration about to collapse" starting
     policy) and resumed training via a new `scripts/train_lr_bump.py`
     with `learning_rate` bumped `1e-4`→`1e-3` and `schedule` switched
     `"adaptive"`→`"fixed"`, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md`.
     No reward-function changes — this isolated the learning-rate
     intervention alone against the exact same starting point as the
     always-on experiment. Plan in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-implementation.md`,
     full run data in
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`.
     - **Correction (caught by final whole-plan review): the original
       write-up of this entry had the adaptive schedule's direction
       backwards**, claiming it "would claw the bump back down given the
       converged policy's low KL divergence." Checking the installed
       `rsl_rl` source directly (`rsl_rl/algorithms/ppo.py:281-284`): low
       KL divergence actually **increases** the adaptive learning rate
       (toward a `1e-2` cap), not decreases it — the opposite of what was
       claimed. This doesn't invalidate the experiment (`schedule="fixed"`
       is still the right choice for a controlled test — it guarantees
       the rate stays at exactly the intended value rather than drifting
       under `"adaptive"`'s own dynamics), only the stated reason for
       needing it was wrong and is corrected here.
     - **Confirmed the learning rate actually held at `0.001` across the
       entire 1500-iteration continuation** (no decay back toward
       baseline) — the experiment genuinely tested its premise, this
       isn't a null result from the bump failing to hold.
     - **Result: 0/10 real eval episodes show any lift** — same "reach,
       grip, freeze" signature as every prior experiment.
       `lifting_sphere`'s 10-point downsampled trajectory reads exactly
       `0.0` at every sample, with a `max` of `0.0027` over the full run —
       a blip of the same magnitude as the always-on run's own `0.0027`
       max (**correction**: an earlier version of this entry claimed this
       run showed "not even the small noise blips" prior runs did, which
       the run's own `max` value doesn't support — this is equally null,
       not more null, than the prior experiments). A substantial,
       sustained, correctly-held optimizer-level perturbation, injected
       at precisely the point the literature identified as critical,
       produced no measurable improvement over the prior experiments —
       this argues against "insufficient exploration pressure at the
       right moment" being a sufficient explanation on its own, at least
       via this specific lever.
     - **This is the fourth real attempt on the reward/optimization axis
       for this sub-problem** (sparse-only, curriculum-gated dense,
       always-on dense, LR-bump). Per the user's "try both" instruction,
       proceeding directly to the second planned experiment (potential-
       based reward shaping) — not gating it on this result, since the
       user explicitly asked for both regardless of outcome.
   - **Follow-up experiment: monotonic potential-based reward shaping
     (falsified — and a genuine bug found in the formula itself).**
     Replaced six independent additive reward terms with a single
     running-max potential-based term (Ng, Harada, Russell, ICML 1999),
     per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md`.
     Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`.
     **Result: 0/10 real eval episodes show any lift** — identical
     "reach, grip, freeze" signature to every prior experiment; in this
     run, markers never even show real grasping/lifting attempts, just
     approach-and-hover.
     - **Root cause found, not just another null result:**
       `Episode_Reward/staged_potential_progress` *declined* to -0.109
       over training instead of growing. The term's docstring claimed
       `gamma * new_potential - prev_potential` is "always >= 0" — false
       whenever the agent merely *holds* its best-ever potential without
       improving further: `new_potential == prev_potential == Φ` gives
       reward `Φ * (gamma - 1)`, which is **negative** for any
       `gamma < 1` (here `gamma=0.98`). Over a ~225-step episode,
       reaching the object and holding there (`Φ ≈ 0.1`) costs roughly
       `0.1 * (-0.02) * 225 ≈ -0.45` total reward — *worse* than never
       approaching the object at all (`Φ` stays 0 the whole episode,
       reward stays exactly 0). The policy that minimizes this cost is
       to never reach for the sphere, which is exactly what the eval
       showed. This is a bug in the shaping formula's discount handling,
       not evidence against potential-based shaping as an approach.
     - **This is the fifth real attempt on the reward/optimization axis**
       (sparse-only, curriculum-gated dense, always-on dense, LR-bump,
       potential-shaping). Per `superpowers:systematic-debugging` Phase
       4.5 this would normally be flagged back rather than attempting a
       sixth unilateral reward tweak — in this case the user independently
       raised two related, concrete ideas in parallel (a grasp-gated
       actuation/movement incentive, and a penalty for staying static
       within a bound), so the sixth attempt is user-directed rather than
       unilateral. See
       `docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`:
       a new single-sphere scene with randomized spawn + mirrored
       opposite-side goal, a corrected *undiscounted* running-max
       milestone bonus (drops the `gamma` decay that caused the bug
       above), and a new grasp-gated stillness penalty.
     - **A second sign-convention bug found during Task 4's first training
       run (before eval), fixed before proceeding:** `stillness_penalty`'s
       own function body already returns the signed value (`-1.0` when
       triggered, `0.0` otherwise), but its `RewardsCfg` registration used
       `weight=-2.0`. `RewardManager.compute()` computes
       `func(...) * weight * dt` (`reward_manager.py:149`) — multiplying
       two negatives turned the intended *penalty* into a **+2.0*dt
       reward** for the exact stay-still-after-grasp behavior this term
       exists to punish. Caught by reading the actual TensorBoard data
       (`Episode_Reward/stillness_penalty` grew to +1.3 over training,
       which is impossible for a true penalty term) rather than trusting
       the design doc's own stated intent. Fixed to `weight=2.0` (commit
       `e7742b5`); the first training run's data is invalid and was
       discarded, re-running Task 4 with the corrected weight before any
       eval/video judgment. Distinguishing convention for future reward
       terms in this repo: functions like `action_rate_l2`/`joint_vel_l2`
       return an *unsigned* magnitude and rely on a negative weight to
       become a penalty; `stillness_penalty` instead returns an
       *already-signed* value — mixing the two conventions inside one
       `RewardsCfg` is exactly where this bug slipped in.
     - **Final result (FALSIFIED).** Full run data:
       `docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md`.
       10-episode real eval on the corrected checkpoint
       (`logs/train/2026-07-06_16-02-16/model_1499.pt`): **0/10 episodes
       show a genuine, controlled grasp-and-lift.** One episode (5)
       initially looked like a lift in a coarse start/25%/50%/75%/end
       sample, but a full frame-by-frame re-inspection (022 through 050)
       showed the sphere separating from the gripper and drifting to a
       hover disconnected from the arm — the gripper stays static near
       the ground throughout, never co-located with the airborne sphere
       again after first contact. Far more consistent with the tiny
       object (0.01kg, 9mm radius) being knocked/launched by a glancing
       collision with the arm's body than with a bilateral grasp
       (`contact_grasp_bonus` requires simultaneous force on *both*
       jaws, which a glancing knock from one link wouldn't satisfy).
       This is the sixth real attempt on the reward/optimization axis for
       this sub-problem (sparse-only, curriculum-gated dense, always-on
       dense, LR-bump, potential-shaping, mirror-scene+stillness-penalty).
       What *is* newly confirmed working: full-workspace spawn
       randomization, the mirrored opposite-side goal mechanism (verified
       correct independently before training), the stillness-penalty sign
       fix (confirmed non-positive throughout training), and the
       corrected milestone-bonus formula (confirmed non-negative and
       growing, no decay bug). None of that changed the core outcome —
       the gripper still never achieves and holds a bilateral grasp in
       any eval episode. Per `superpowers:systematic-debugging` Phase
       4.5, flagging back to the user rather than attempting a seventh
       reward/optimization tweak unilaterally. Candidates worth
       considering next: the gripper's ~28mm max aperture vs. the
       sphere's 18mm diameter may leave too little margin for the
       current joint-position-target action space to reliably converge
       on a stable bilateral grasp pose, or a hierarchical policy
       (separate reach-to-pregrasp and close-gripper phases) instead of
       one flat policy learning both simultaneously.
     - **Follow-up experiment: shrink the sphere to test the
       aperture-margin hypothesis (FALSIFIED).** User-directed: reduce
       the sphere from 18mm to 12mm diameter (roughly doubling the
       gripper's per-side clearance margin, 5mm -> 8mm), scoped to the
       mirror-scene's own config only (`objects_cfg.py`'s shared
       `SPHERE_CFG` untouched). Full run data:
       `docs/superpowers/plans/2026-07-06-ar4-sphere-shrink-report.md`.
       **Result: 0/10 real eval episodes show a genuine, controlled
       grasp-and-lift** — all 10 personally inspected frame-by-frame by
       the controller (not delegated), given the prior misjudgment on
       this same sub-problem. One episode again showed the
       accidental-collision-launch signature (a motion-blur streak
       trailing upward from the gripper, then the sphere floating
       disconnected from a static gripper) rather than a real grasp —
       the same false-positive pattern as the mirror-scene experiment's
       Episode 5. This is evidence *against* the aperture-margin
       hypothesis specifically: doubling the clearance margin produced
       no improvement, so gripper-to-object size tolerance is likely
       not the primary bottleneck. This is the seventh real attempt on
       this sub-problem. Remaining candidates worth considering:
       a hierarchical policy (separate reach-to-pregrasp and
       close-gripper phases instead of one flat policy learning both),
       or examining whether the joint-position action space itself
       (rather than object size) limits precise gripper-closure timing.
     - **Follow-up experiment: classical-IK-guided path reward (HALTED,
       not evaluated on the sphere).** Built a live classical-IK
       path-tracking reward (5 Cartesian waypoints: pre-grasp, grasp,
       lift, transit, place; `isaaclab.controllers.DifferentialIKController`
       queried fresh every step against the real physics state) plus a
       gripper-open/closed timing bonus, per
       `docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md`.
       Code review caught and fixed a real bug before training: the IK
       target used the raw `link_6` body pose instead of the gripper's
       actual pinch point (3.6cm `_EE_OFFSET`, larger than the
       `advance_tolerance` used for waypoint progression), corrected by
       rotating the offset into world frame via `link_6`'s live
       orientation before commanding IK. Two attempts at the full
       1500-iteration training run both failed to complete: the first
       falsely reported success from stale/misleading console text
       while checkpoints showed it never got past ~2 iterations; the
       second was deliberately killed by the user mid-run (~iteration
       114/1500) to redirect all experimentation to a cube instead of a
       sphere (see below) — **this experiment was never evaluated on
       the sphere and remains an open, untested hypothesis**, to be
       re-run on the cube-based scene.
     - **Repo-wide pivot: switch the graspable object from sphere to
       cube for all AR4 grasp experiments (user-directed).** All sphere
       testing halted. Converted `tasks/ar4/pickplace_mirror_env_cfg.py`
       and `tasks/ar4/pickplace_ik_guided_env_cfg.py` to use the
       existing, unmodified `CUBE_CFG` (18mm cuboid, `objects_cfg.py`)
       instead of `SPHERE_CFG` — renamed every scene field, event term,
       observation term, termination term, and `SceneEntityCfg`
       reference from "sphere" to "cube" throughout both files, and
       renamed `tasks/ar4/mdp.py`'s `set_mirrored_goal`/
       `compute_path_waypoints` `sphere_cfg` parameter to `object_cfg`
       (pure rename, these functions were already generic).
       Deliberately dropped the sphere-shrink experiment's 12mm-radius
       hack rather than applying an analogous shrink to the cube — the
       cube uses its existing, unmodified 18mm default size, since
       Experiment 7 already showed object-size shrinking alone doesn't
       help. `objects_cfg.py` itself untouched (shared, used by
       `interactive_demo.py`/perception scripts/`grasp_demo.py`).
       Verified directly (the implementer's own smoke test never
       actually completed): both configs load, reset, and step
       correctly with the expected reward-term lists (4 terms for the
       mirror scene, 6 for the IK-guided scene), and the cube's
       randomized spawn position was confirmed within workspace bounds.
       Senior-reviewed and approved — no missed sphere references, no
       forbidden files touched. All future experiments in this line
       (starting with re-running the classical-IK-guided path
       experiment above) use the cube.
     - **New reward term queued (not yet wired into any RewardsCfg):
       `ground_penalty`** (`tasks/ar4/mdp.py`) — direct user request:
       "give a negative reward when the cube is on the ground." Unlike
       `stillness_penalty` (grasp-gated, only fires after a freeze
       *following* a successful grasp), this applies from the start of
       every episode regardless of grasp state, giving constant
       pressure to lift the object off the ground as soon as possible.
       Queued behind the current cube training run to avoid stacking
       two untested variables at once.
     - **First real physics crash this session, mid-retry of the
       classical-IK-guided path run on the cube.** `PxRigidActor::detachShape:
       shape is not attached to this actor!` at iteration 893/1500,
       preceded by `prim '/World/envs/env_1755/Cube/geometry/mesh' was
       deleted while being used by a shape in a tensor view class` —
       this invalidated the entire batched physics tensor view across
       all 4096 envs and crashed the whole run (not a false report, not
       an intentional kill — a genuine PhysX-level failure, the first
       of its kind across all 8 sphere-based experiments plus the
       mirror-scene/sphere-shrink runs). Root cause not yet fully
       diagnosed: unclear whether this is a rare per-env fluke or a
       systematic issue with `CuboidCfg`'s procedural geometry at this
       scale (4096 parallel envs). Retrying once to check whether it
       recurs (same/different env index, similar/different iteration)
       before deciding whether deeper investigation is warranted.
     - **Retry succeeded (no crash recurrence — likely a rare fluke, not
       systematic): full 1500-iteration cube training run completed**
       (`logs/train/2026-07-06_19-45-06/`, 31/31 checkpoints verified,
       `model_1499.pt` confirmed). Final episode-cumulative scalars:
       `ik_guided_path_bonus=0.1428`, `gripper_schedule_bonus=0.0142`,
       `contact_grasp_bonus=16.7976`, `stillness_penalty=-0.2318`,
       `cube_reached_goal=0.0072`.
     - **Two senior-tier literature research passes commissioned in
       parallel (direct user request), both independently converging on
       real, actionable problems with `contact_grasp_bonus`** — full
       writeups at
       `docs/superpowers/specs/research/2026-07-06-rl-manipulation-senior-b.md`
       and `2026-07-06-classical-manipulation-senior-a.md`:
       - **RL-manipulation finding**: `contact_grasp_bonus` (weight 20)
         is ungated — it pays out every step regardless of downstream
         lift/path progress, opposed only by `stillness_penalty` (net
         −2/step once triggered), a ~9:1 reward-rate advantage for
         freezing after grasp. Externally verified against Isaac Lab's
         own shipped lift task, IsaacGymEnvs `FrankaCubeStack`, and
         ManiSkill3 `PickCube-v1` source (read directly, not just
         cited) — all three gate downstream reward behind grasp/lift
         state; this repo's reward is the structural outlier. Mao et
         al. 2025 (arXiv:2502.15442) independently names this exact
         "reach then freeze" local optimum in unrelated work (quote
         verified against primary text).
       - **Classical-manipulation finding**: `contact_grasp_bonus`
         checks bilateral force *magnitude* only, discarding the force
         *direction* `force_matrix_w` already provides. Every classical
         grasp-mechanics source surveyed (Nguyen 1988; Ponce & Faverjon
         1991/93; Ferrari & Canny 1992; GraspIt! 2004; modern
         data-driven planners Dex-Net/GPD/QuickGrasp) treats a
         geometric/antipodal force-closure check as mandatory and never
         substitutable by contact-force magnitude alone — a real
         bilateral force could register from a non-antipodal, unstable
         pinch that isn't actually resistant to gravity's wrench.
       - **This run's own data directly confirms both findings
         simultaneously and more starkly than predicted**:
         `contact_grasp_bonus` (16.80) outweighs `ik_guided_path_bonus`
         (0.14) by **~118:1** in the actual trained policy's episode-
         cumulative behavior — the policy learned to hold a grasp
         indefinitely while making almost no lift/path progress, the
         "reach, grip, freeze" pattern confirmed quantitatively, not
         just via video inspection, for the first time this session.
       - **Decision: implement both fixes together as one evidence-based
         redesign (Experiment 9), not sequential single-variable
         guesses** — this is fixing two independently-verified real
         problems in the same reward term, not stacking unvalidated
         guesses. New function `antipodal_grasp_bonus` (requires jaw1/
         jaw2 contact-force directions within ~30° of anti-parallel,
         dot product < -0.85, in addition to the existing magnitude
         check) replaces `contact_grasp_bonus` in
         `Ar4PickPlaceIkGuidedEnvCfg`'s `RewardsCfg`, at a substantially
         reduced weight (20.0 -> 3.0) to close the reward-rate gap
         rather than relying on `stillness_penalty` alone to outweigh
         it. `contact_grasp_bonus` itself stays unchanged in `mdp.py`
         (still used by the original sphere-based `pickplace_env_cfg.py`
         task). Eval/video inspection skipped for this checkpoint since
         the TensorBoard evidence already conclusively shows the same
         failure mode without needing to re-confirm via video.
     - **Experiment 9 result: reward dominance completely reversed
       (118:1 grasp-dominant -> ~107:1 path-dominant), not a modest
       improvement — and the reason is itself informative.** The
       antipodal geometric check fires ~1800x less often than the old
       magnitude-only check did, far more than the 6.67x weight
       reduction (20->3) explains. Root-caused: `antipodal_cos_threshold
       =-0.85` (~31.8° allowed deviation from perfect opposition) was an
       approximate guess, stricter than this scene's actual physics
       permits — the classical friction-cone half-angle for this
       scene's `mu=1.0` (`static_friction=dynamic_friction=1.0`,
       scene-wide) is `arctan(1.0)=45°`, giving a correct threshold of
       `-0.7071`, not `-0.85`. That the physically-correct check almost
       never fires while the magnitude-only check fired constantly is
       itself strong confirmation of the classical-manipulation
       research finding: the grasps the policy learned under the old
       reward were not real force-closure grasps, just coincidentally
       hard bilateral contact from non-opposing directions.
     - **User-directed: systematically reassess all conditions/
       parameters, not just the reward function.** Directly compared
       this repo's full PPO config, action space, and object physics
       against Isaac Lab's own shipped, proven Franka cube-lift recipe
       (`isaaclab_tasks/manager_based/manipulation/lift/config/franka/`).
       Findings: (1) PPO hyperparameters and network architecture
       already match exactly (already adapted from this same recipe,
       confirmed byte-for-byte) — not the gap; (2) actuator PD gains
       already tested this session (falsified, no effect) — not
       re-opening that axis without new evidence; (3) **action scale**:
       this repo uses `scale=1.0` for arm joint position actions vs.
       Franka's `scale=0.5` — double the joint-position change per unit
       of policy output, which may specifically hurt the precise final
       grasp-closing phase; (4) **cube solver iteration counts**:
       Franka's own cube explicitly boosts
       `solver_position_iteration_count=16,
       solver_velocity_iteration_count=1` (well above PhysX defaults)
       for stable contact resolution during grasping — this repo's cube
       used only defaults.
     - **Experiment 10: bundle the physics-derived antipodal-threshold
       correction with the action-scale and solver-iteration findings.**
       `antipodal_cos_threshold` corrected to `-0.7071`; new `ActionsCfg`
       (`scale=0.5`) scoped to
       `pickplace_mirror_env_cfg.py`/`pickplace_ik_guided_env_cfg.py`
       only (`env_cfg.py`'s shared `ActionsCfg`, scale=1.0, stays
       unchanged — still used by the original sphere task,
       `grasp_demo.py`, `interactive_demo.py`, perception scripts); cube
       solver iteration counts boosted to match Franka's recipe, scoped
       via `.replace()` on `CUBE_CFG`'s `spawn`, not touching the shared
       `objects_cfg.py`. This is the ninth real attempt on this
       sub-problem's reward/optimization/physics axis, now grounded in
       both literature research and a direct, systematic comparison
       against a proven working reference implementation rather than a
       reward-only guess. **Result: `antipodal_grasp_bonus` regressed to
       exactly 0.000000 by the end of training** (worse than Experiment
       9's already-tiny 0.001416) — loosening the geometric threshold
       didn't help, arguing the bottleneck is *precision* of final
       gripper positioning/alignment under direct joint-space control,
       not reward-threshold calibration.
     - **Experiment 11 (user-proposed): replace joint-space action with
       task-space differential-IK-driven action.** Instead of the policy
       outputting joint-angle deltas directly (with IK used only for
       reward-shaping in Experiments 8-10), the policy now outputs
       Cartesian end-effector deltas and Isaac Lab's built-in
       `DifferentialInverseKinematicsActionCfg` converts them to joint
       targets *inside the control loop* — offloading "how to move 6
       joints" to a classical solver so the policy only learns "where to
       go." New file `tasks/ar4/pickplace_taskspace_env_cfg.py`
       (`Ar4PickPlaceTaskspaceEnvCfg`), new simplified
       `path_proximity_bonus` reward (drops the now-redundant IK-match
       sub-signal `ik_guided_path_bonus` needed when IK was reward-only),
       `antipodal_grasp_bonus`/`gripper_schedule_bonus`/
       `stillness_penalty` carried over unchanged from Experiment 10. New
       `--taskspace` flag on `scripts/train.py`/`scripts/eval_loop.py`.
       - **First full run diverged**: independently traced
         `/tmp/exp11_train_stdout.log` (the controller, not the
         implementer, caught this — the implementer's own status report
         called it "Non-Critical") and found the PPO critic's `Mean
         value_function loss` exploding from ~0.0000 to ~1.56 to ~4047 to
         ~3.2M between iterations 66-69/1500, reaching ~5.2e23 by the
         final iteration and never recovering — ~95% of the run's policy
         updates were driven by a diverged critic, not a clean comparison
         to Experiment 10. Never seen in Experiments 1-10 (same PPO
         config, joint-space action), implicating the new
         `DifferentialInverseKinematicsActionCfg` term specifically: an
         outlier raw policy action, previously harmless under
         `JointPositionActionCfg` (saturates at joint limits), likely
         drives the IK solve into a discontinuous joint-space jump that
         destabilizes PhysX for one env/step, producing an extreme
         observation the critic can't fit.
       - **Fix**: new `Ar4PickPlaceTaskspacePPORunnerCfg(clip_actions=5.0)`
         (~3.4x the observed action-noise std of 1.46), scoped to the
         taskspace experiment only — `Ar4PickPlacePPORunnerCfg` itself
         stays unmodified, still used unchanged by every other
         experiment. Verified on a 300-iteration diagnostic, then on the
         full 1500-iteration re-run: `Loss/value_function` stayed bounded
         for the entire run (max 7.88, one isolated 2-iteration transient
         spike, immediate recovery — independently re-verified against
         the raw TensorBoard event file, all 1500 points, not samples).
       - **Result (first positive signal after 11 experiments):**
         `antipodal_grasp_bonus` final value **0.018815**, nonzero in
         91.6% of all 1500 logged iterations — every prior experiment
         had this at exactly 0 (Experiment 10) or 0.001416 at best
         (Experiment 9). `cube_reached_goal` final 0.010223, ~3.6x
         Experiment 10's 0.002848.
       - **Video inspection (25 frames, 5fps, full episode) qualifies
         this result**: the arm reaches down toward the cube within the
         first ~1s and then holds an almost identical low, near-ground
         pose for the remaining ~4s of the episode — the small red cube
         stays at or near the gripper tip throughout, consistent with
         the nonzero antipodal-contact metric (a real, held bilateral
         grasp is plausible), but the arm never visibly lifts the cube
         to height or carries it toward the goal in this rollout. **Net
         assessment: task-space IK-driven action produced the first
         genuine, sustained antipodal grasp contact this project has
         seen — a real improvement on the specific "grasp never emerges"
         sub-problem — but "pick up and move" as a whole is still not
         achieved.** The next sub-problem is getting the policy from
         "hold a low grasp" to "lift and carry," which is exactly what
         motivates the staged-decomposition/episode-length/richer-
         goal-placement ideas queued as the next design considerations
         (episodes may be too short to discover a lift+carry+place
         sequence once a stable grasp is finally reachable; explicit
         per-stage sub-objectives may be needed rather than one
         continuous reward).
     - **Experiment 12: fix a verified reward-rate bug (antipodal_grasp_bonus
       vs. stillness_penalty) — scalars ambiguous, video inconclusive,
       "pick up and move" still not achieved.** Direct arithmetic check of
       Experiment 11's reward weights found that holding a grasp without
       further progress netted **+1.0/step** (`antipodal_grasp_bonus`'s
       continuous +3.0/step, only partly offset by `stillness_penalty`'s
       -2.0/step once its 25-step patience window elapsed) — a real,
       previously-unverified bug directly matching Experiment 11's observed
       "reach, grasp, freeze" video signature. Fix: raised
       `stillness_penalty`'s weight 2.0 → 5.0 in
       `pickplace_taskspace_env_cfg.py` only (same net -2.0/step target
       spelled out in
       `docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md`),
       nothing else changed. Full run:
       `docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md`.
       - **Scalar result is genuinely mixed, not a clean win or loss.**
         `antipodal_grasp_bonus`'s final value dropped 32% versus Experiment
         11 (0.018815 → 0.012777) — but its *nonzero rate* rose (91.6% →
         93.2% of iterations), and `stillness_penalty` became *less*
         negative despite its weight increasing 2.5x (-0.002533 →
         -0.001857, meaning the "grasped and stagnant" condition fired
         *less*, i.e. less frozen time) while both outcome-oriented metrics
         improved (`path_proximity_bonus` +8%, `cube_reached_goal` +5.4%).
         The task's own implementer report initially misread the antipodal
         drop alone as proof the fix failed; the controller rejected that
         verdict as premature (full reasoning in the report's "Controller's
         independent re-assessment" section) — the drop is exactly what a
         proxy term that pays for *static* holding would do if the policy
         is holding *less* statically, which is not distinguishable from
         failure using scalars alone.
       - **Video inspection (3 of 10 recorded episodes, 25 frames each at
         5fps, personally inspected by the controller with cropped/upscaled
         close-ups of the gripper region) does not resolve the ambiguity.**
         Episode 1: arm settles into a low pose near the cube's spawn area
         by ~1s and holds it for the rest of the episode, materially the
         same signature as Experiment 11's video. Episode 2: arm folds into
         a compact pose close to its own base; a small reddish sliver is
         visible at the fold but not clearly identifiable as a held cube
         from this camera distance. Episode 3: unambiguous — a distinct red
         cube sits stationary on the ground for the entire episode while
         the arm folds down near its own base, never engaging it at all;
         this specific failure signature (arm disengages from the cube
         entirely) was not seen in Experiment 11's single inspected
         episode. **None of the 3 episodes show the cube leaving the
         ground.** Given training's own `cube_reached_goal` rate is only
         ~1%, a 3-episode sample is not powered to distinguish "no
         improvement" from "same low success rate, different per-episode
         failure mode by chance" — this is a real limitation of the
         inspection, not a null result being spun as informative.
       - **Net assessment: inconclusive, not negative.** The reward-rate
         bug this experiment fixed was real and independently verified
         (both by the arithmetic and by `stillness_penalty`'s reduced
         firing rate in the actual run) — but neither the scalars nor the
         video sample are strong enough evidence to say whether fixing it
         changed observable pick-and-place behavior. "Pick up and move" as
         a whole remains unachieved. Recommended next step: this result
         does not on its own justify either doubling down on more reward-
         rate tuning or abandoning the direction — the queued
         episode-length/staged-decomposition ideas (still undone, see
         Experiment 11's entry above) remain the most likely candidates to
         produce an unambiguous behavioral change, since they address a
         different, complementary hypothesis (the episode may simply be
         too short / too unguided for a lift+carry+place sequence to be
         discoverable at all, independent of the freeze incentive fixed
         here).
     - **Experiment 13: residual RL over a classical waypoint-seeking base
       controller — a structurally different pivot, not another reward
       tweak, per this project's mandate to escalate after an inconclusive
       result rather than keep tuning the same paradigm. Result: video
       evidence points toward a genuine regression, not just another
       ambiguous case.** Hypothesis: rather than commanding the raw
       task-space delta directly, the policy commands only a small
       correction on top of a classical proportional ("seek") controller
       that pursues the already-computed active waypoint — additive
       superposition, per Silver et al. 2018 "Residual Policy Learning"
       (arXiv:1812.06298) and Johannink et al. 2019 "Residual
       Reinforcement Learning for Robot Control" (ICRA), both verified
       directly. New action term `ResidualDifferentialIKAction`
       (`tasks/ar4/residual_ik_action.py`), new
       `Ar4PickPlaceResidualEnvCfg` (`tasks/ar4/pickplace_residual_env_cfg.py`)
       reusing Experiment 12's exact reward weights unchanged. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md`.
       Full run data: `docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md`.
       - **Implementation review caught a real, separate bug before
         training**: the Cfg subclass was originally missing
         `@configclass`, which — per Isaac Lab's dataclass-based config
         machinery — would have silently kept the *parent* class's
         `class_type` default (plain, non-residual
         `DifferentialInverseKinematicsAction`), meaning the entire
         residual mechanism would never have run, with no exception.
         Fixed and independently re-verified (reproduced the exact
         dataclass-default failure/fix standalone, not just by inspection)
         before any training happened. Real smoke-test evidence
         afterward (`params/env.yaml`) confirms the fixed action term
         actually did get selected and construct correctly inside Isaac
         Sim.
       - **Diagnostic (300 iter) and full (1500 iter) runs both showed
         `Loss/value_function` staying bounded** (max 0.17, actually
         healthier than Experiment 12's own reference run) — no critic
         divergence, ruling out the specific failure class Experiment 11
         hit under a different new action term.
       - **Scalar comparison against Experiment 12 is a larger, more
         one-sided shift than any prior experiment-to-experiment
         comparison this session**: `antipodal_grasp_bonus` collapsed
         -98.9% (0.012777 → 0.000140, though still nonzero on 80.4% of
         iterations, down from 93.2%); `stillness_penalty` got *worse*,
         not better (-12.8%, more stagnant time than Experiment 12, the
         opposite direction from what a "shorter, more efficient holds"
         explanation would predict); `path_proximity_bonus` improved
         marginally (+5.3%); `cube_reached_goal` dropped -44.2%. Unlike
         Experiment 11→12's transition (where a similar antipodal drop
         was plausibly explained by less time spent in a purely static
         hold), the `stillness_penalty` direction here argues against that
         same explanation — if the policy were moving *more* productively,
         `stillness_penalty` should improve, not worsen.
       - **Video inspection (3 of 10 recorded episodes, personally
         inspected by the controller, cropped/dense-frame review of
         episode 1's final ~15% for signs of genuine settling vs.
         continued motion) confirms a new, materially worse failure
         signature not seen in Experiments 11-12.** Episode 1: the arm
         never settles — frame-by-frame comparison of the episode's final
         four sampled frames shows the arm still visibly, continuously
         folding/collapsing right up to the end, ending in a compact,
         collapsed heap near its own base — not a static bad pose, ongoing
         instability. Episode 2: reaches an elevated diagonal pose that
         does look stable from partway through the episode onward, cube
         occluded from this camera angle, genuinely ambiguous whether held.
         Episode 3: a clean repeat of Experiment 12's episode-3 signature —
         the arm settles into a static pose that never reaches the cube at
         all; the cube (clearly visible, distinct from the gripper) sits
         untouched on the ground for the entire episode. None of the 3
         episodes show a lift.
       - **Root-cause hypothesis, not yet tested**: the base controller's
         pursuit step is unconditional — it fires every step toward
         whichever waypoint is currently active, regardless of what the
         policy's residual is doing, with no gating on grasp state. Both
         cited papers explicitly warm-start the residual to avoid exactly
         this kind of early-training conflict: Johannink et al. hold the
         residual fixed at zero for an initial period while training only
         the value function, "allowing for a good estimate of the value of
         the base controller before learning begins." **This experiment's
         design did not implement that warm-start** — actor and critic
         trained jointly from iteration 0, with the policy's residual
         summed directly onto the base controller's step every iteration
         from the start. This is a real gap between the design and the
         literature it was grounded in, not just an unlucky run, and is
         the most direct candidate explanation for episode 1's observed
         instability (an untrained, effectively-random early residual
         fighting a committed base step, every single step, with nothing
         damping the interaction).
       - **Net assessment: this result looks like a genuine regression,
         not an inconclusive one** — unlike Experiment 12, the scalar
         picture (a stillness_penalty move in the wrong direction) and the
         video evidence (a new, actively-unstable failure mode) corroborate
         each other rather than pointing in different directions.
         Recommended next step: **do not extend this exact residual design
         with more training or minor tuning.** Either (a) retry with the
         literature's own warm-start technique properly implemented (a
         specific, well-motivated fix to a known gap, not a new guess), or
         (b) per this project's mandate to pivot after a second
         non-improving result in a row rather than keep iterating in the
         same family, move to the still-undone queued
         episode-length/staged-decomposition direction instead — both are
         legitimate; this is a real decision point, not a default.
     - **Experiment 14: reach-skip curriculum — a structurally different
       pivot (third non-improving result in a row triggers the project's
       "escalate, don't keep tuning" mandate), built on Experiment 12's
       clean baseline, not Experiment 13's unresolved residual regression.
       Result: no improvement on the primary success criterion, plus a new,
       partially-explained failure mode.** Hypothesis: three experiments
       running (11-13) all showed reliable grasp-contact but never
       lift+carry+place, with `path_proximity_bonus`/`antipodal_grasp_bonus`
       consistently indicating reach and grasp are the well-learned parts —
       so remove reach from what the policy has to (re-)discover every
       episode, reallocating the full step/exploration budget to the
       actually-unsolved grasp→lift→carry→place sub-problem. Mechanism: new
       one-shot reset `EventTerm` `reset_arm_to_pregrasp_pose`
       (`tasks/ar4/mdp.py`) computes a pregrasp joint configuration via a
       single `DifferentialIKController` solve against each episode's
       just-randomized cube position and writes it directly via
       `write_joint_position_to_sim`/`set_joint_position_target`, run once at
       reset between `reset_cube_position` and `randomize_goal`. New
       `Ar4PickPlaceReachskipEnvCfg`
       (`tasks/ar4/pickplace_reachskip_env_cfg.py`) reuses Experiment 12's
       exact reward weights and plain (non-residual) action term unchanged —
       isolating the starting-state variable alone. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md`.
       - **Diagnostic (300 iter) and full (1500 iter) runs both showed
         `Loss/value_function` staying bounded** (max 0.024436, matching the
         diagnostic's own max almost exactly) — the cleanest, healthiest
         value-function behavior of any new-mechanism gate this session
         (versus Experiment 13's full-run max of 0.17), confirming the new
         one-shot direct-joint-state-write reset event does not itself
         destabilize the critic.
       - **Scalar comparison against Experiment 12 is mixed, and per this
         project's established correction protocol is deliberately not
         used alone to call a verdict.** `cube_reached_goal` improved
         modestly at the final-iteration snapshot (+5.8%, 0.010773 →
         0.011393), with a mid-run peak of 0.022308 near iteration 750 —
         more than double the final value, suggesting the final snapshot
         understates the effect mid-run. `path_proximity_bonus` declined
         slightly (-8.4%). `antipodal_grasp_bonus` dropped sharply (-94.4%,
         0.012777 → 0.000709) but remained nonzero on 89.9% of iterations,
         consistent with the same "less static holding time" ambiguity
         flagged in Experiment 11→12's transition rather than a clean
         regression signal on its own. `stillness_penalty` got 133% more
         negative (-0.001857 → -0.004328) — plausibly explained by the
         changed episode/reset structure (episodes now start mid-task, so
         the per-episode accounting for these terms is not directly
         comparable to a full-reach-included episode) rather than by worse
         behavior, but this is a hypothesis, not a confirmed explanation.
       - **Video inspection (3 of 10 recorded episodes, 25 frames each,
         personally inspected by the controller) shows no lift in any
         episode, and surfaces a new failure signature in 2 of 3.** Episode
         1: the arm reaches down to a low pose near the cube by ~frame 5 of
         25 and holds a static position near it for the rest of the
         episode — the same "reach and freeze near the cube" signature seen
         in Experiments 11-13, just reached faster (consistent with the
         reach sub-problem being skipped). Episode 2: the arm does *not*
         settle into a stable hold — starting from an elevated pose near the
         cube, it progressively folds into an increasingly compact
         configuration close to its own base over the course of the episode
         (visible ongoing changes in arm angle across frames 17, 21, 25, not
         a static end state), ending in a low, contorted fold. Episode 3: the
         arm folds into a tight, contorted pose near its own base almost
         immediately (already collapsed by frame 5) and stays there for the
         rest of the episode; the cube — clearly visible, untouched — is
         never approached at all. **None of the 3 episodes show the cube
         leaving the ground, and no episode reaches waypoint index ≥2
         (lift).**
       - **Explicit read against the design spec's stated success
         criterion: not met.** The spec's bar was reaching waypoint index ≥2
         and/or genuine lift-off-the-ground in a meaningfully larger
         fraction of episodes than the ~0/3 seen in Experiments 12-13. This
         experiment's sample is also 0/3 on lift — no improvement on the
         primary criterion. The spec also called out that a null result here
         would itself be informative by ruling out "the reach sub-problem is
         eating all the exploration budget" as the explanation — that
         ruling-out did happen, but with an additional, unanticipated
         finding: 2 of 3 episodes show a *new* failure mode (folding toward
         the robot's own base) not present in Experiments 12-13's samples,
         where the more common failure was freezing near the cube or
         disengaging from it, not actively collapsing toward the base.
       - **Root-cause hypothesis, not yet tested**: the design spec itself
         flagged that the one-shot IK reset lands the arm "near," not
         exactly at, the computed pregrasp target. Depending on which IK
         solution the one-shot solve happens to land on (elbow-up/down
         ambiguity, proximity to joint limits), some fraction of episodes
         may start from an awkward or self-occluding joint configuration
         that the policy has not learned to recover from — an
         initial-conditions problem layered on top of, not instead of, the
         original grasp→lift gap. This reframes part of "reach is not the
         problem" into "reach *removal* can itself introduce a new problem
         if the substitute starting pose isn't uniformly good."
       - **Net assessment: no improvement on the stated success
         criterion, plus a new, partially-explained failure mode — do not
         extend this exact reach-skip mechanism with further tuning.** This
         is the third experiment in a row (12, 13, 14) that does not resolve
         "grasp/lift never emerges"; per this project's mandate, the next
         experiment should be a genuinely different lever, not a fourth
         variation in the same reward/action/curriculum family. Notably,
         the base-collapse pattern seen in 2 of 3 episodes here is directly
         relevant to (and independently motivates, beyond the original
         request) the cube-near-robot-base penalty term already directed
         for the next experiment, alongside wiring in the existing
         `ground_penalty` function and raising `antipodal_grasp_bonus`'s
         weight.
     - **Experiment 15: wire in `ground_penalty`, add a new
       `base_proximity_penalty`, raise `antipodal_grasp_bonus`'s weight
       (matched by a `stillness_penalty` raise) — direct user-directed
       reward changes, built on Experiment 12's clean baseline (not
       Experiment 14's unresolved reach-skip mechanism). Result: the best
       outcome-metric scalars of the session so far, but the two new
       penalty terms did not behave as designed — and one of them,
       `base_proximity_penalty`, moved in the wrong direction, rising to
       saturation rather than staying low.** Direct user requests
       (2026-07-07): "negative reward for contacting the ground. higher
       reward for the cube being in the grasp position" and "negative
       reward for the cube contacting the base of the robot." New
       `base_proximity_penalty` function (`tasks/ar4/mdp.py`, cube's xy
       distance to the robot's own root origin, distinct from
       `ground_penalty`'s z-height check per explicit user instruction),
       `ground_penalty` (existing, previously unused) wired in for the
       first time, `antipodal_grasp_bonus` weight 3.0 → 4.0 with
       `stillness_penalty` matched 5.0 → 6.0 (preserves the exact
       -2.0/step anti-freeze margin Experiment 12 verified:
       4.0 − 6.0 = 3.0 − 5.0 = -2.0). New
       `Ar4PickPlaceBaseProximityEnvCfg`
       (`tasks/ar4/pickplace_baseproximity_env_cfg.py`), reusing
       Experiment 12's action/observations/events/terminations unchanged —
       isolates the reward function as the only new variable. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md`.
       - **Diagnostic (300 iter) flagged a real anomaly the controller did
         not just pattern-match past: a single-iteration `Loss/value_function`
         spike to 17.66 at step 39** — roughly 100x any prior accepted
         spike in this project (Experiment 13's full-run max was 0.17,
         Experiment 14's diagnostic max was 0.024). The spike's *shape*
         (isolated, single iteration, decayed within ~10-15 iterations,
         zero recurrence) matched this project's always-accepted pattern
         and was structurally unlike Experiment 11's actual divergence bug
         (a sustained climb starting at iteration 67, not a one-off blip),
         so the gate was passed on shape grounds — but the controller
         explicitly flagged the magnitude for the full run to specifically
         re-examine rather than treating the diagnostic's "pass" as
         resolved. **The full run confirmed it as a genuine one-off**: the
         identical spike recurred at the identical step (17.657946, step
         39, matching to 4+ significant figures) with no second occurrence
         above 1.0 anywhere in the remaining ~1460 iterations and no
         gradual upward trend in the loss's baseline — the same
         diagnostic-vs-full-run confirmation pattern already established
         in Experiment 14. The raw magnitude itself remains unexplained and
         far larger than any prior run's, but it did not grow, recur, or
         destabilize training at scale.
       - **Scalar comparison against Experiment 12 (and Experiment 14) is
         the most consistently positive of the session**, unlike
         Experiments 12-14's mixed or negative pictures.
         `cube_reached_goal` improved +59.7% versus Experiment 12's final
         value (0.010773 → 0.017202) and +51.0% versus Experiment 14's
         (0.011393 → 0.017202) — the best final-iteration success-rate
         reading of any experiment this session.
         `antipodal_grasp_bonus` rose +159.8% versus Experiment 12
         (0.012777 → 0.033199), and unlike Experiment 14 (whose antipodal
         bonus collapsed to near-zero), this run shows a clear, largely
         monotonic climb across the back half of training rather than a
         collapse. `stillness_penalty` worsened modestly (46.8% more
         negative) and `path_proximity_bonus` declined slightly (-4.5%) —
         both small relative to the two outcome-oriented metrics' gains.
       - **The two new penalty terms did not behave as the design spec
         hoped, and `base_proximity_penalty` moved in the wrong
         direction.** `ground_penalty`'s nonzero rate never trended down —
         saturated at 100% in both the first and last 150-iteration windows
         and every window between. `base_proximity_penalty`'s nonzero rate
         was supposed to stay low (evidence it only fires for the specific
         base-collapse-adjacent cases it targets); instead it rose from
         12.0% in the first 150 iterations to a **saturated 100.0% for
         roughly the last 1050 of 1500 iterations** — the cube ends up
         within 8cm of the robot's own root origin on nearly every logged
         step for the majority of training, the opposite of what a
         successful deterrent penalty would produce. A rising rate that
         saturates over training (rather than a roughly-constant ~10% rate,
         which is what random cube-spawn proximity to the base alone would
         produce, per the design spec's own area-proportion estimate) is
         not consistent with "occasional unlucky spawn positions" — it
         indicates the trained policy is actively converging toward
         base-proximate states as training progresses.
       - **Video inspection (3 of 10 recorded episodes, personally
         inspected by the controller) is consistent with both the positive
         scalar signal (still a low absolute success rate, so 0/3 in a
         3-episode sample is statistically unsurprising at ~1.7%) and the
         base-proximity finding.** Episode 1: the arm reaches down to the
         cube by ~frame 5 of 25 and holds a static pose near it for the
         rest of the episode — the established "reach and freeze near the
         cube" signature, no lift. Episode 3: a near-identical repeat of
         Episode 1's pattern — reach, then a static diagonal hold near the
         cube, no lift. **Episode 2 dramatically reproduces Experiment
         14's arm/cube-collapses-toward-the-base pattern**: starting from
         an already fairly close spawn, the arm progressively curls tighter
         against its own base body across the episode, and the cube itself
         visibly ends up immediately adjacent to the base by the final
         frames — not merely the arm folding without engaging the cube,
         but the cube's own position ending up base-proximate, directly
         matching what `base_proximity_penalty`'s saturating nonzero rate
         predicts. **None of the 3 episodes show the cube leaving the
         ground or reaching waypoint index ≥2 (lift)** — the design spec's
         stated success criterion is not met on this sample, though the
         improved scalar success rate suggests a larger video sample would
         likely show at least occasional successes that 3 episodes alone
         cannot reliably catch.
       - **Cross-experiment pattern, now seen twice under two different
         mechanism changes — worth treating as a priority signal, not a
         coincidence.** Experiment 14 (reach-skip curriculum) showed the
         arm folding toward its own base in 2 of 3 sampled episodes.
         Experiment 15 (reward-shaping only, built on Experiment 12's
         baseline, no curriculum/action changes) now shows the same
         qualitative behavior in 1 of 3 sampled episodes, corroborated by
         `base_proximity_penalty`'s rising-to-saturation scalar trend
         across the majority of training — despite this experiment adding
         an explicit penalty designed to discourage exactly this. Two
         structurally different changes (a reset-distribution change in
         Experiment 14, a reward-shaping change in Experiment 15)
         independently converging on the same base-proximate failure mode
         suggests this may not be an artifact of either specific mechanism,
         but a more fundamental attractor in this task's action-space/
         reward landscape — plausibly kinematic (echoing the classical
         demo's own unresolved kinematic-singularity/self-collision stall
         finding from earlier this session, a third, independent thread
         that also landed on arm configurations near the base becoming
         "stuck" states) rather than purely reward-incentive-driven, since
         adding an explicit penalty against it in Experiment 15 did not
         prevent it and the rate still rose during training.
       - **Net assessment: genuinely the most positive outcome-metric
         result of the session, but with a real, unresolved, and now
         twice-observed side effect that a purpose-built penalty failed to
         suppress.** Recommend NOT extending this exact
         `base_proximity_penalty` formulation with further tuning alone
         (raising its weight again without understanding why it saturates
         risks the same "add pressure without fixing the mechanism"
         pattern already seen not working here) — the recurring
         base-attractor pattern across three independent investigations
         (Experiment 14, Experiment 15, and the classical demo's
         singularity stall) is now a strong candidate for its own
         dedicated investigation before further reward tuning in this
         family, alongside continuing to pursue the still-undone
         episode-length/staged-decomposition direction queued since
         Experiment 11's entry above.
     - **Experiment 16: from-scratch replication of two proven,
       independently-published Isaac-ecosystem manipulation recipes —
       genuine, sustained lift confirmed in video for the first time in
       this entire 16-experiment arc.** Direct user request: research
       actual working RL-manipulation examples and replicate them on the
       AR4+cube scene, starting from scratch rather than tuning the
       existing baseline further. Hypothesis, grounded directly in source
       (not secondhand): Isaac Lab's own Franka Cube Lift task
       (`isaaclab_tasks/manager_based/manipulation/lift/`) and
       IsaacGymEnvs' independently-maintained FrankaCubeStack task both
       (a) never reward grasp quality as a standalone term — grasp is
       purely instrumental — and (b) multiplicatively gate the majority
       of available reward (goal-tracking) behind an actual lift
       condition, structurally unlike every one of this repo's 15 prior
       experiments, which always kept a standalone grasp-quality term and
       an ungated (or only weakly-staged) progression signal — exactly
       the reward-rate-arithmetic bug class this project's own research
       has repeatedly diagnosed and never fully eliminated. New
       `Ar4PickPlaceProvenRecipeEnvCfg`
       (`tasks/ar4/pickplace_provenrecipe_env_cfg.py`): 6 reward terms
       (two — `reaching_object`, `lifting_object` — reused directly,
       unmodified, from Isaac Lab's own installed source, not
       reimplemented), a plain binary per-step lift reward (not a
       milestone/running-max bonus), goal-tracking reward gated on lift,
       plain joint-space action (reverting from this session's
       task-space/IK lineage, matching both references), and this repo's
       first use of Isaac Lab's curriculum manager (regularization-weight
       curriculum, also replicated from the reference). Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md`.
       - **Diagnostic flagged a genuinely different `Loss/value_function`
         shape than any prior experiment — climbing and NOT recovering
         within the 300-iteration window (unlike every previous
         isolated-spike-then-recover precedent) — fully resolved by the
         full run with a clean, direct causal explanation, not just a
         plausible guess.** The loss continued climbing past the
         diagnostic's endpoint to a run-wide peak of 4.588 at **iteration
         417** — the exact iteration (confirmed directly in the raw
         `Curriculum/action_rate_curr`/`joint_vel_curr` scalars) at which
         the new curriculum mechanism fires, bumping `action_rate`/
         `joint_vel` weights 1000x (-1e-4 to -1e-1) as designed. From that
         peak the loss declined steadily and essentially monotonically for
         the remaining ~1080 iterations to 0.298 (93.5% below peak) —
         bounded and recovering, not runaway, though it does not fully
         return to the initial near-zero baseline, settling into a
         structurally elevated (~0.25-0.45) equilibrium consistent with
         this reward shape's binary/gated terms firing on ~100% of
         iterations for the run's back three-quarters.
       - **Scalar picture is genuinely mixed and worth stating precisely,
         not glossing over.** `cube_reached_goal`'s final-iteration value
         (0.008962) is actually *lower* than both Experiment 12's
         (0.010773, -16.8%) and Experiment 15's (0.017202, -47.9%) — on
         this specific scalar alone, this experiment looks like a
         regression. But `lifting_object` and `object_goal_tracking`
         (this experiment's own direct, literal per-step success signals)
         both grew strongly, monotonically, and in lockstep across the
         entire run (nonzero rate 81.3% → saturated 100.0% by iteration
         ~150; `lifting_object`'s per-window average climbed ~220x, 0.05
         → 12.1) — the two proxies point in opposite directions, exactly
         the kind of ambiguity this project's established correction
         protocol exists for, and exactly why the verdict below rests on
         video, not the `cube_reached_goal` scalar alone.
       - **Video inspection (3 of 10 recorded episodes, personally
         inspected by the controller) resolves the ambiguity decisively
         in favor of the direct lift signals, not `cube_reached_goal`.**
         All three sampled episodes show the same clear pattern: the arm
         reaches the cube within the first ~1-2 seconds, grasps it, and
         **lifts it — genuinely, visibly off the ground — holding it
         elevated at the gripper for the remainder of the episode** (a
         green drop-zone marker, newly added this session, made the goal
         position directly visible in these videos for the first time).
         This is qualitatively different from every one of Experiments
         1-15's video samples, all of which showed either a static
         low-to-the-ground hold near the cube's spawn point or a collapse
         toward the robot's own base — never a sustained elevated hold.
         **3 of 3 sampled episodes show genuine lift; 0 of 3 show the arm
         carrying the lifted cube toward the goal marker within the
         episode** — the arm holds the cube up near its own body/gripper
         area but does not visibly transport it, which is exactly
         consistent with the lower `cube_reached_goal` reading (reaching
         within the 2cm success threshold requires actually arriving at
         the goal, not just lifting).
       - **Net assessment: this is the most significant structural
         finding of the entire research arc — the core "grasp achieved,
         lift never emerges" pattern that has been the through-line of
         Experiments 1 through 15 is resolved.** The hypothesis is
         confirmed: removing the standalone grasp reward and gating
         goal-tracking behind a real lift condition, replicating what two
         independently-proven recipes actually do, produced the lift
         behavior 15 ad hoc iterations on this repo's own reward/action
         lineage did not. The remaining gap is now narrower and better
         defined than at any prior point in this project: not
         "reach/grasp/lift," which this experiment resolves, but
         specifically **carry-to-goal** (transport and place) — a
         well-scoped next research question, not a return to guessing at
         the grasp/lift mechanism. Recommended next steps, in priority
         order: (a) the long-queued episode-length extension (never
         actually tried this session despite being flagged since early
         on) is now much better motivated than before, since the policy
         has a genuine lift to build on and may simply be running out of
         episode time to carry+place after lifting; (b) consider whether
         `object_goal_tracking`'s current weight/std balance is
         sufficient to pull a *lifted* object toward the goal, versus
         just rewarding the state of being lifted regardless of xy
         position, given the video shows the cube held near the robot's
         own body, not moving laterally. Per this repo's now-standing
         scientific-method requirement, either direction needs its own
         hypothesis and background research before a new spec, not just
         a parameter tweak on this experiment's already-proven
         foundation.
       - **CORRECTION (same day, prompted directly by the user challenging
         the video read above, confirmed with real instrumentation, not
         more visual guessing): the "genuine lift" / "grasp achieved"
         claim above is wrong. The cube is not gripped by the fingers at
         any point in the episode.** The user looked at the same frames
         and pointed out the cube appeared to be carried by the wrist
         joint, not the gripper jaws — a sharp, correct counter-read of
         the controller's own video judgment. Verified with a fresh
         instrumented rollout of the exact checkpoint
         (`logs/train/2026-07-07_14-40-53/model_1499.pt`), logging
         `gripper_jaw1_contact`/`gripper_jaw2_contact` force magnitudes
         (`force_matrix_w`, the same field `antipodal_grasp_bonus` reads),
         gripper joint positions, and the cube's distance to
         `gripper_jaw1_link`/`gripper_jaw2_link` vs. `link_6`/
         `gripper_base_link`, every step across a full episode.
         **Both jaw contact sensors read exactly `0.0000` at every one of
         250 logged steps, including the initial approach — the gripper
         never registers any contact force with the cube, at any point in
         the episode.** From roughly step 80 through step 248 (out of a
         250-step episode — the "held" period, `cube_z` rising to and
         holding around 0.56-0.58m, dropping back to the ground-rest value
         of 0.009 exactly at episode timeout), `gripper_jaw1_joint` sits at
         ≈0.014 — essentially fully open, matching `GRIPPER_OPEN_POS`
         almost exactly — and the cube's distance to `link_6`/
         `gripper_base_link` (≈0.023m) is consistently smaller than its
         distance to either jaw (≈0.051-0.056m) for the entire held
         duration. **Root cause: the cube is being pushed/wedged/carried
         via contact with the wrist/gripper-housing body as the arm
         reorients, never touched by the fingers.** This experiment's
         reward function — by design, faithfully matching both proven
         references — only checks the cube's world height and
         goal-distance, with no requirement that a genuine grasp produced
         that height; the policy found a cheaper way to satisfy the
         height-gated reward than learning to actually grasp, and the
         proven references' own `object_is_lifted`/lift-indicator checks
         are equally height-only, so this reward family does not
         structurally rule this out — it may be a difference in what the
         AR4's own wrist/gripper geometry makes exploitable versus a
         Franka Panda's, not evidence the reward design itself is
         wrong. **Secondary finding, independent of the above:**
         `gripper_jaw1_joint`/`gripper_jaw2_joint` do not track each other
         despite the source URDF's explicit `mimic` joint constraint on
         `gripper_jaw2_joint` (`multiplier="1"`) — Isaac Sim's USD import
         of this asset appears not to enforce that constraint, so the two
         jaws behave as independently-actuated joints rather than a
         coupled parallel gripper; a real, separate asset-fidelity issue
         worth its own investigation, not yet done.
         **Corrected interpretation:** this is not "grasp/lift solved" —
         grasp specifically is being bypassed, not solved. It is still a
         real, qualitatively new behavior relative to every one of
         Experiments 1-15's video samples (none of which ever moved the
         cube meaningfully off the ground at all, exploit or otherwise),
         so the reward-gating change clearly did change behavior — but the
         correct next step is close to the opposite of what the
         uncorrected entry above recommended: some form of genuine
         grasp-contact requirement needs to gate the lift/goal-tracking
         reward (e.g. requiring real bilateral jaw force, not just height,
         before `lifting_object`/`object_goal_tracking` can fire) rather
         than continuing to build on this checkpoint's carry-to-goal gap
         as if grasp were already solved. This does not need to mean
         reintroducing a *separately-rewarded* grasp term (the literature
         basis for avoiding that still stands) — it can instead be a
         *gating condition* on the existing lift/goal-tracking terms,
         preserving the design principle that grasp itself earns no direct
         reward. Per this repo's scientific-method requirement, this
         becomes its own hypothesis-driven experiment, not a quick patch.
     - **Experiment 17: gate `lifting_object`/`object_goal_tracking` on
       genuine bilateral antipodal jaw contact (reusing
       `antipodal_grasp_bonus`), fixing Experiment 16's confirmed
       wedging exploit — the gate works exactly as designed, but the
       policy never discovers genuine grasp+lift at all within this
       training budget. A real, informative exploration-difficulty
       result, not a bug, not a regression to hide.** Hypothesis, grounded
       in Experiment 16's own confirmed root cause plus Xu et al. 2026
       ("Stage-Transition Dense Reward Modeling," arXiv:2606.31377, read
       directly from source — their "stage leakage" term names exactly
       this failure class, and their ablation shows removing a grasp-
       verification gate hurts convergence): requiring real force-closure
       contact before the lift/goal-tracking reward can fire should close
       the exploit. New `Ar4PickPlaceGraspGatedEnvCfg`
       (`tasks/ar4/pickplace_graspgated_env_cfg.py`), identical to
       Experiment 16 except `lifting_object` and `object_goal_tracking`/
       `object_goal_tracking_fine_grained` now require
       `antipodal_grasp_bonus`'s existing force-closure check (both jaws
       exceed `force_threshold=0.05`, force directions within
       `antipodal_cos_threshold=-0.7071`) in addition to height, at
       identical weights to Experiment 16 — isolating the gate as the
       only new variable. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md`.
       - **Both formal training-stability gates passed cleanly** — 300-
         iteration diagnostic and full 1500-iteration run both showed
         `Loss/value_function` small and bounded throughout (full-run max
         0.0547, roughly two orders of magnitude below Experiment 16's
         curriculum-driven peak of 4.588) — directly explained by the
         gate's own headline finding below: with the reward-composition
         discontinuity that drove Experiment 16's value-loss spike never
         occurring (the gate never fires), there's no corresponding
         instability source either.
       - **The grasp-gate never fired once across the entire 1500-
         iteration training run — not "rarely," exactly zero.**
         `Episode_Reward/lifting_object` and `Episode_Reward/
         object_goal_tracking` are `0.0` at all 1500/1500 logged
         iterations (`nonzero: 0/1500` for both), versus Experiment 16's
         `lifting_object` already 81.3% nonzero by the equivalent
         150-iteration mark and saturated to 100% shortly after — a
         difference in kind, not degree. `cube_reached_goal`'s final
         value (0.002360) is correspondingly the lowest of any experiment
         since Experiment 11, reported factually and not treated as an
         independent verdict per this project's established correction
         protocol.
       - **A dedicated instrumented investigation (not video, not scalars
         alone — this experiment's entire purpose was fixing exactly this
         kind of verification gap) reproduced the gate's sub-conditions
         separately across ~1,487 rollout steps (~5.9 episodes) of the
         final checkpoint, and gives a decisive, three-way-disambiguated
         answer for *why* the gate never fired.** `height_ok` (cube
         z > 0.03) was true **0 times** in 1,487 steps — the cube's
         maximum observed height (0.00901) is indistinguishable from its
         0.009 resting/spawn height. The compound "genuinely grasp AND
         lift simultaneously" event this gate requires does not merely
         go mis-gated — it does not occur at all, at any height, by any
         mechanism, exploit or otherwise. This directly confirms
         **exploration difficulty, not a gate bug, as the dominant
         explanation**: a threshold-miscalibration hypothesis was
         directly refuted (when contact did occur, forces reached 7-20N,
         150-400x above the 0.05N threshold — magnitude was never the
         limiting factor), while a real, independently-confirmed asset
         defect was found as a contributing factor: `gripper_jaw1_joint`
         tracked its commanded `[0, 0.014]` envelope exactly throughout,
         while `gripper_jaw2_joint` independently drifted to 0.0168 (20%
         past its own commanded open limit) under contact load — direct,
         concrete confirmation that the two jaws are not mechanically
         coupled (the `mimic` joint constraint flagged as suspect in
         Experiment 16's correction is confirmed unenforced by the USD
         import, not just suspected).
       - **The one real contact event observed (episode 0, 230
         consecutive steps) is the gate working exactly as intended, not
         a near-miss the gate wrongly rejected.** The arm drove its
         already-open gripper directly into the cube, producing a static,
         non-antipodal wedge/jam (13-20N on jaw1, 7-10N on jaw2, cosine
         angle frozen at exactly -0.6409/~130° for all 230 steps, versus
         the required <-0.7071/~135° — five degrees short, never varying).
         Neither jaw moved toward closed during this window; `cube_z`
         never rose. This is structurally the same failure family as
         Experiment 16's wedging exploit, just registering on the jaw
         sensors directly instead of the wrist — and the antipodal gate
         correctly refused to credit it for all 230 steps. A separate
         episode showed the gripper *can* fully close (both joints reach
         their closed position) — but never within 5.9cm of the cube,
         confirming the policy has not learned to combine "get close" and
         "close the gripper around the object" into one coordinated
         behavior at all, not that it's close and slightly misaligned.
       - **Net assessment: the fix worked exactly as designed (the
         exploit is closed, confirmed by the one contact event it
         correctly rejected), but closing it removed the only reward
         gradient the policy had ever found, and the compound behavior
         needed to replace it has not been discovered from scratch in
         1500 iterations.** This is the direct, structural cost of
         closing a reward-hacking exploit without providing a new path
         toward the real behavior it was standing in for — well
         precedented (Xu et al.'s own ablation shows the same tradeoff
         in their setting) but now measured concretely in this repo.
         Recommended next step, discussed and substantially co-designed
         with the user in this session: add a **dense** shaping signal
         toward correct pre-grasp positioning — cube proximity to the
         end-effector/pinch-point frame specifically (this repo's
         existing `ee_frame`, already offset to the gripper's pinch
         point via `_EE_OFFSET`, already used by `reaching_object`) —
         so the policy has *some* gradient pointing it toward "get near
         the gripper and close around the object," rather than the
         current all-or-nothing binary gate with zero partial credit.
         The user's own refinement — checking whether the cube's
         position *tracks the end-effector's motion* (rigid co-movement
         with the EE specifically, not just proximity at one instant) —
         is a stronger, more targeted version of this: it's precisely
         the test that would have caught Experiment 16's exploit
         directly, since the wedged cube's distance to the wrist stayed
         constant because it was rigidly pinned to *that* body, not the
         EE. Separately, the confirmed mimic-joint asset defect is worth
         its own investigation (does Isaac Lab support enforcing URDF
         `mimic` joints, and if not, can the gripper action term command
         both jaws' targets symmetrically as a workaround) — a genuine,
         independent asset-fidelity fix, not just a reward-design
         question. Per this repo's scientific-method requirement, the
         next experiment needs its own hypothesis and background
         research (e.g. on dense sub-goal/shaping-toward-precondition
         techniques in sparse-grasp RL) before a new spec — not a quick
         patch on this one.
     - **Experiment 18: add a dense pre-grasp-readiness shaping term
       (proximity × gripper-closedness) on top of Experiment 17's
       unchanged antipodal gate — the shaping term is strongly learned,
       but does not move `lifting_object` off its exact 0/1500 null
       result. A clean falsification of the "missing gradient" hypothesis,
       not an inconclusive result.** Hypothesis, grounded in Experiment
       17's own Task 6 instrumented finding (the policy explores "get
       close" and "close the gripper" independently but never combines
       them) plus Xu et al. 2026's complementary "within-stage progress
       feedback" concept (arXiv:2606.31377, already cited for Experiment
       17, re-cited here for its other half) and arXiv:1803.04996
       (abstract-verified only, cited for scope not for a specific
       ablation number): a dense reward for cube-to-EE proximity
       multiplied by gripper closedness should give the policy a
       continuous incentive to combine both halves, without reintroducing
       a hackable substitute for genuine antipodal contact — the binary
       gate stays untouched as the only path to the large
       `lifting_object`/`object_goal_tracking` reward. New
       `pregrasp_readiness_bonus` (`tasks/ar4/mdp.py`), wired into new
       `Ar4PickPlacePregraspEnvCfg`
       (`tasks/ar4/pickplace_pregrasp_env_cfg.py`) at weight 2.0, on top
       of Experiment 17's exact unchanged reward set. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md`.
       - **The new term is a real, strongly discovered gradient — not the
         failure mode.** `Episode_Reward/pregrasp_readiness` is nonzero
         at all 1500/1500 logged iterations (100% in every one of ten
         150-iteration windows), growing from 0.000166 at iteration 0 to
         1.268935 at the final iteration, stabilizing in the 1.24-1.27
         range for the entire second half of training. The policy
         actively adopts reaching configurations that register readiness
         as defined — the shaping mechanism itself works exactly as
         designed.
       - **`Episode_Reward/lifting_object` remains at exactly 0/1500 —
         identical to Experiment 17's null result, despite the strong
         readiness signal.** Not "slightly worse" or "noisier" — the
         same exact zero, at every one of 1500 logged iterations, in
         every one of ten 150-iteration windows.
         `object_goal_tracking`/`object_goal_tracking_fine_grained`
         (both gated on the same lift condition) are correspondingly
         also 0/1500. `Loss/value_function` stayed small and bounded
         throughout (max 0.148235, comparable to Experiment 17's 0.0547,
         both roughly two orders of magnitude below Experiment 16's
         curriculum-driven peak of 4.588) — training itself was stable;
         this is not a divergence artifact.
       - **This falsifies the experiment's specific hypothesis as
         written: readiness and lifting are decoupled, not bottlenecked
         on a missing approach-and-close gradient.** The policy can learn
         to be "ready" (close + gripper closing) as a side effect of
         pursuing `reaching_object`, without that readiness ever
         translating into an actual attempted lift. Per the design
         spec's own success criteria, this null result narrows rather
         than repeats the open question — it specifically implicates
         either the confirmed mimic-joint asset defect (Experiment 17's
         Task 6 finding: `gripper_jaw2_joint` drifts 20% past its own
         commanded open limit under load, independent of `jaw1_joint`)
         or a discoverability gap in the *lift* action itself (vertical
         motion while maintaining contact) that is categorically
         different from, and not solved by, better pre-grasp
         positioning. Per this repo's scientific-method requirement and
         its own explicit mandate to prefer structurally new directions
         over repeated parameter tweaks after a string of null results
         (this is now three consecutive experiments — 16, 17, 18 — all
         terminating in the same root fact: the cube never leaves the
         ground by any margin), the next step is not a fourth reward-
         shaping variant on the same mechanism. Candidate structurally
         different directions, not yet scoped: (a) directly investigate/
         fix the confirmed mimic-joint asset defect, since it is a
         verified, independent, mechanical confound present in every
         experiment run so far, not a reward-design question; (b)
         demonstration or curriculum bootstrapping for the lift primitive
         specifically (e.g. a scripted/classical grasp-lift trajectory as
         a warm-start or residual-RL base, since three different reward
         designs across ~4500 combined training iterations have never
         once produced any vertical lift via pure exploration); (c) the
         previously-queued staged-decomposition/longer-episode redesign
         (see `feedback_ar4-episode-length-and-staged-decomposition`
         memory) now has stronger direct motivation than when first
         proposed.
     - **Experiment 19: fix the confirmed mimic-joint asset defect (candidate
       (a) above), then re-run Experiment 18's exact reward config unchanged
       to isolate whether that defect alone explains three consecutive
       null results. Two independently-tested fix configurations both made
       jaw synchronization measurably WORSE than the pre-fix baseline —
       a clean, decisive falsification, not an inconclusive result.**
       Hypothesis, grounded in Experiment 17 Task 6's own confirmed
       root cause (jaw2 drifted 20% past its commanded position under
       contact load while jaw1 tracked exactly) and the source URDF's
       own `<mimic joint="gripper_jaw1_joint" multiplier="1" offset="0"/>`
       constraint on `gripper_jaw2_joint` (confirmed present in source,
       confirmed NOT enforced by the built USD): authoring a real
       PhysX-level `PhysxMimicJointAPI` coupling between the two jaw
       joints, so they genuinely move as one physically-coupled unit,
       should close enough of the gap to make antipodal grasps physically
       reachable. Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment19-mimic-joint-fix-design.md`.
       Plan: `docs/superpowers/plans/2026-07-07-ar4-experiment19-mimic-joint-fix-implementation.md`.
       - **Technical grounding, read directly from source before
         implementation** (not reasoned from memory): the installed
         Isaac Sim 107.3.26 PhysX schema confirms `PhysxMimicJointAPI`
         explicitly supports `PhysicsPrismaticJoint` (the gripper jaws'
         actual joint type); `build_asset.py` already passed
         `parse_mimic=True` to the URDF importer, confirming the intent
         was already present but not working; PhysX's own official
         mimic-joint test suite
         (`omni.physx.tests/.../PhysxMimicJointAPI.py`) was read directly
         and used as the reference implementation for the exact Python
         API calls (`PhysxMimicJointAPI.Apply`,
         `GetReferenceJointRel().AddTarget`, `GetGearingAttr().Set(1.0)`,
         `GetOffsetAttr().Set(0.0)`) — this is not a guess, it matches
         PhysX's own canonical example exactly.
       - **Fix iteration 1 (zero `gripper_jaw2`'s independent PD drive,
         `stiffness=0.0, damping=0.0`, reasoning that the mimic
         constraint alone should determine jaw2's position, matching the
         PhysX test suite's own reference pattern of only driving the
         "reference" joint): FAILED, and made things worse.** Instrumented
         rollout (`scripts/mimic_joint_verify.py`, reusing Experiment 18's
         trained checkpoint against the rebuilt asset) measured
         `max_jaw_pos_diff_during_contact=0.00548m` — 3.9x over the
         0.0014m pass threshold, and **worse than the pre-fix baseline of
         0.0028m** (Experiment 17 Task 6), not an improvement.
       - **Mid-investigation research** (Google search + direct GitHub
         API fetches, not just the initial web-search summaries, which
         were independently re-verified after one contained an apparently
         fabricated claim — see below): an Isaac Lab maintainer
         (`isaac-sim/IsaacLab` discussion #2626, comment fetched directly
         via `gh api`) confirms "Isaac Lab does support mimic joints...
         we currently don't have an example for this type of joint" — a
         genuine, admitted maturity gap, not settled practice. A real
         community member's only reported *working* mimic-jointed
         gripper setup (UR10e + Robotiq 2F-85) keeps **every** joint,
         including mimic-coupled ones, independently actuated at full
         stiffness — the opposite of fix iteration 1's design.
       - **A WebFetch-tool paraphrase of the same discussion thread
         claimed an "unmerged `feature/unactuated-joints` branch"
         suggesting mimic-joint support was still work-in-progress — this
         claim could NOT be reproduced when the actual comment thread was
         re-fetched directly via `gh api` and read verbatim, and is not
         relied on anywhere in this entry.** Flagged here as a concrete,
         caught instance of an AI web-summarization tool fabricating a
         specific technical detail not present in the primary source —
         a reminder that even a tool's *paraphrase* of a fetched page
         needs the same "don't trust, verify against the primary source"
         treatment as any other subagent claim, not just subagent
         self-reports.
       - **Fix iteration 2 (restore `gripper_jaw2`'s actuator to match
         `gripper_jaw1` exactly, `stiffness=1000.0, damping=50.0` —
         following the community's actually-tested pattern, keeping the
         mimic constraint as reinforcement rather than a replacement for
         independent driving): FAILED, and made things worse again.**
         `max_jaw_pos_diff_during_contact=0.00647m` — 18% worse than
         fix iteration 1, and more than double the pre-fix baseline. This
         is the single most decisive data point in the whole experiment:
         iteration 2's configuration is **identical to the pre-fix
         baseline in every respect except the mimic joint's presence**
         (both jaws independently driven at the same stiffness/damping in
         both cases) — a clean, isolated A/B comparison confirming the
         mimic constraint itself is actively interfering under real
         contact load, not merely failing to help. Consistent with the
         PhysX schema's own documented behavior that a mimic joint is a
         genuine **two-way** interaction (it applies a corrective impulse
         to the *reference* joint too, not just the mimicking one),
         which can plausibly increase net system compliance under load
         when combined with two already-independently-driven joints,
         rather than rigidly locking them together as intended.
       - **Reverted, not iterated a third time.** Two independently-tested
         configurations both made synchronization measurably worse (not
         merely "no better") — a pattern systematic-debugging discipline
         treats as diagnostic of an architectural mismatch, not a
         parameter-tuning problem, even short of the formal 3-strikes
         threshold. `scripts/build_asset.py` and `tasks/ar4/robot_cfg.py`
         reverted to their exact pre-Experiment-19 state (commit
         `255b9b2`); the asset was rebuilt from the reverted script,
         restoring the known-good baseline for all future work.
         `scripts/mimic_joint_verify.py` (the instrumented jaw-tracking
         diagnostic itself) was kept — independently useful tooling for
         any future asset-fidelity investigation, regardless of this
         outcome. Tasks 3/4 of the plan (the full training re-run) were
         never reached — the hard gate correctly stopped the experiment
         before spending that compute on an unverified fix.
       - **Net assessment: this experiment does not resolve whether the
         mimic-joint mechanical defect blocks genuine antipodal grasps —
         it specifically shows that fixing it via `PhysxMimicJointAPI` in
         this Isaac Sim version, at least in the two configurations
         tested, is not a viable mechanism.** The underlying defect
         (the two jaws are not mechanically coupled, contrary to the real
         robot's design) remains real and unresolved, but is no longer
         this repo's most promising next lever — candidate (a) from
         Experiment 18's own list is now closed out as tried-and-failed,
         not merely deferred. Per the session's own prior sequencing
         decision (confirmed directly with the user before this
         experiment began), the next research direction is candidate
         from a related, independently-raised idea: **use IK to
         constrain the gripper to a fixed vertical/top-down approach
         orientation during the reach phase**, rather than leaving
         orientation fully unconstrained across all 6 joint-space DOF as
         every experiment since Experiment 16 has done. This directly
         reduces the geometric burden of ever finding an antipodal grasp
         in the first place (a top-down, jaw-aligned approach to a cube
         on a table is the standard assumption in classical grasp
         planning — Dex-Net, GPD, both already in this repo's research
         record) — independent of, and not blocked by, this experiment's
         negative result on jaw-coupling fidelity specifically.
     - **Experiment 20: constrain the gripper's approach orientation
       toward vertical/top-down. The mechanism pivoted mid-experiment
       after independent verification found the originally-designed hard
       IK constraint structurally unstable; the revised soft reward-bias
       mechanism worked exactly as intended (the strongest, most
       saturated dense signal recorded in this repo's history) — and
       `lifting_object` still stayed at exactly 0/1500, a clean
       falsification of the orientation-discovery-bottleneck hypothesis.**
       Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md`
       (see its "Revision" section for the full mechanism-pivot account).
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md`.
       - **Original design: a custom absolute-pose differential-IK
         action term (`VerticalLockDifferentialIKAction`,
         `tasks/ar4/actions.py`) locking orientation exactly, every
         step, leaving only 3D position under policy control.** Built,
         code-reviewed clean (spec compliance ✅, no Critical findings),
         but the task reviewer flagged that `_FIXED_DOWNWARD_QUAT` had
         no reproducible measurement artifact — the single most
         load-bearing value in the experiment, asserted but not
         instrumented, echoing Experiment 16's exact prior lesson (an
         unverified claim later found wrong under real instrumentation).
       - **Independent re-verification against the live simulated
         system (not isolated quaternion math) found the mechanism
         genuinely unstable.** The real end-effector orientation
         converged to within ~9-10 degrees of target by step 30, then
         **diverged to 75-99 degrees off target within the same episode
         under zero commanded policy action**, resetting and reproducing
         the identical pattern next episode. Three independently-tried
         fixes, per this repo's systematic-debugging discipline (3
         failed fixes means question the architecture, not patch
         again):
         1. Tilting the target 5/10/15/20 degrees off exact-vertical
            (testing whether a wrist singularity specifically at
            straight-down was the cause) — all tilts diverged similarly,
            ruling out one specific singular target as the sole
            explanation.
         2. Giving the action term's position target persistent state,
            replacing Isaac Lab's own stock convention of recomputing it
            fresh each step as `current_position + delta` (self-
            referential under a zero action — provides zero restoring
            force; harmless for Experiment 11's position-only 3-DOF
            action, which has 3 redundant joint DOF to absorb drift, but
            with this experiment's fully-constrained 6-DOF pose lock —
            zero redundant DOF — nothing anchored the solve). A real,
            independently-motivated correctness fix, kept in
            `tasks/ar4/actions.py` regardless of outcome — reduced peak
            drift (~52-59 vs. ~77-99 degrees) but did not achieve
            stability.
         3. Sweeping the DLS damping coefficient (`lambda_val`, default
            0.01) from 0.01 to 1.0 — no monotonic relationship (0.1 came
            closest to the 15-degree stability threshold; 0.3/0.5/1.0
            were worse again), not a simple tuning fix.
       - **Conclusion: hard-locking full 6-DOF pose via a single-Newton-
         step-per-env-step differential-IK controller is not a stable
         mechanism for sustained pose-holding on this arm**, independent
         of target orientation, position-target formulation, or damping
         tuning — an architecture-level finding, not an implementation
         bug. `tasks/ar4/actions.py` and
         `pickplace_verticallock_env_cfg.py` were kept, not deleted:
         real, working, independently-verified code for a mechanism that
         may be worth revisiting later with a fundamentally different
         control approach (e.g. multiple IK substeps per env step), just
         not pursued further under this experiment.
       - **Revised mechanism: a soft dense reward term,
         `orientation_alignment_bonus`, layered onto the already-proven
         joint-space action (Experiment 18's exact `ActionsCfg`)
         instead of a new IK-based action space.** Tests the identical
         underlying hypothesis without the IK-stability problem class.
         New env cfg `Ar4PickPlaceOrientationBiasEnvCfg`
         (`tasks/ar4/pickplace_orientationbias_env_cfg.py`): Experiment
         18's exact action/observation/event/termination/curriculum
         configuration plus the one new term.
       - **The revised mechanism worked exactly as intended — the
         strongest, most cleanly saturated dense signal recorded in
         this repo's history.** `Episode_Reward/orientation_alignment`
         reached its effective ceiling (weight 2.0 × a [0,1]-bounded
         function) by iteration 150 and stayed saturated at 1.92-1.96 for
         the entire remaining ~90% of training — more completely
         saturated than Experiment 18's `pregrasp_readiness` (which
         settled around 1.2-1.25 out of a similar ~2.0 ceiling, not
         fully saturated). This confirms the policy unambiguously
         solved the specific sub-problem this experiment targeted:
         orienting the gripper's approach axis vertically.
       - **`Episode_Reward/lifting_object` stayed at exactly `0/1500`
         regardless — a clean falsification, not an inconclusive
         result.** Identical to Experiment 17's and Experiment 18's
         outcomes. Because the orientation-alignment signal was
         unambiguously solved (not merely attempted), this specifically
         rules out approach-orientation discovery as the exploration
         bottleneck, rather than leaving that question open. Whatever is
         blocking `lifting_object` from ever firing is not primarily an
         orientation-discovery problem.
       - **Net assessment: four consecutive experiments (17, 18, 19, 20)
         now converge on the same underlying fact — the cube never
         leaves the ground by any margin — after reward shaping, hard
         and soft orientation constraints, and a mechanical mimic-joint
         fix attempt have each been tried and each failed to move it.**
         Per this repo's own mandate to prefer a structurally new
         direction over another variant on the same class of approach
         after a string of nulls, the next research direction should not
         be a fifth reward/action-space tweak on top of pure joint-space
         RL exploration. Candidates: (a) an instrumented Task-6-style
         rollout of Experiment 20's trained checkpoint, to determine
         whether antipodal contact is ever reached at all now that
         orientation is solved (narrows "still never contacts correctly"
         vs. "contacts correctly but can't complete the lift") before
         committing to the next experiment; (b) demonstration/imitation
         bootstrapping for the lift primitive specifically, since reward
         shaping and action-space constraints have both now been
         exhausted without success across this many independent
         attempts.
       - **Follow-up (same day): ran candidate (a), the instrumented
         rollout — the failure signature itself has changed since
         Experiment 17's Task 6.** 750 steps (3 episodes) of the final
         checkpoint: `height_ok_steps=0` (cube still never leaves the
         ground, `max_cube_z=0.00905` vs. 0.009 resting), but critically
         `both_magnitude_ok_steps=0/750` — `gripper_jaw1_joint`'s
         contact sensor registers **zero force at every single step of
         the entire rollout** (`max_jaw1_force=0.0`), while
         `gripper_jaw2_joint` does register contact intermittently
         (`max_jaw2_force=2.23N`). Experiment 17's Task 6 found both
         jaws contacting simultaneously in a non-antipodal wedge
         (`both_magnitude_ok_steps=231/750`); this is a genuinely
         different, asymmetric failure — one jaw never touches the cube
         at all. `max_orientation_dot=0.9998` confirms the orientation
         mechanism does achieve near-perfect vertical alignment in this
         same rollout (per-step, not just the training-time aggregate),
         so this isn't an orientation regression. **This directly
         implicates the mimic-joint mechanical asymmetry** (Experiment
         17 Task 6 / Experiment 19: jaw2 tracks its commanded position
         20% worse than jaw1 under load) as a more directly-evidenced
         candidate blocker than before — an asymmetric gripper that
         closes unevenly plausibly produces exactly one-jaw-only
         contact. Experiment 19 closed out the specific
         `PhysxMimicJointAPI`-based fix as not viable, but the
         underlying mechanical asymmetry itself remains live and is now
         more directly implicated by this new evidence, independent of
         that one specific (failed) fix mechanism.
     - **Experiment 21: hard-gate the gripper open during approach, only
       allowing the policy's own close command once the cube is within
       5cm of the end-effector — directly testing the user's own design
       contribution ("consider approaching with open jaw, and only
       closing when in position") against Experiment 20's specific
       asymmetric-contact finding. Null on the literal success
       criterion, but a real, narrowing result: Experiment 20's specific
       failure signature (one jaw never touching at all) is resolved.**
       Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment21-proximity-gated-gripper-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment21-report.md`.
       - **New action term `ProximityGatedBinaryJointPositionAction`**
         (`tasks/ar4/actions.py`) forces the gripper open regardless of
         the policy's own command unless the cube is within
         `proximity_threshold=0.05m` of the end-effector — verified
         directly against the action term's own computed command (not
         inferred from downstream training behavior): cube far + close
         command → forced open; cube 0.02m away + close command →
         actually closes. Grounded in a real (though only
         secondary-source-confirmed, not independently verified against
         primary PDF text due to file size) precedent for staged
         approach-then-close structures in learned grasping,
         arXiv:2303.17592.
       - **`lifting_object` stays at exactly `0/1500`** — identical to
         Experiments 17, 18, and 20. `orientation_alignment` and
         `pregrasp_readiness` both remained healthy and consistent with
         Experiment 20's own values, confirming the new gate didn't
         disrupt either existing mechanism.
       - **The instrumented contact diagnostic (same Task-6-style
         rollout used after Experiment 20) shows a real, specific
         change, even though the strict `both_magnitude_ok_steps`
         success criterion is still `0/750`.** Comparing directly:
         `max_jaw1_force` went from **0.0N** (Experiment 20 — jaw1 never
         registered any contact at all) to **6.73N** (Experiment 21 —
         real, substantial contact); `max_jaw2_force` went from 2.23N to
         27.44N. Both jaws now make genuine contact with the cube at
         some point in the rollout — Experiment 20's specific
         one-jaw-never-touches asymmetry is resolved. What remains
         missing is *simultaneity*: both jaws touch, but never at the
         same step. `max_orientation_dot=1.0` (tighter than Experiment
         20's own 0.9998) confirms this isn't an orientation regression.
       - **Net assessment: this narrows rather than repeats the open
         question.** Five consecutive experiments (17, 18, 19, 20, 21)
         have each ruled out a different specific candidate, and each
         successive instrumented diagnostic has narrowed the failure
         description further: "cube never leaves the ground" (all) →
         "one jaw never touches at all" (20) → "both jaws touch, just
         not simultaneously" (21). The most directly-implicated
         remaining candidate is the mimic-joint mechanical asymmetry
         itself (Experiment 17 Task 6 / Experiment 19: jaw2 tracks its
         commanded position 20% worse than jaw1 under load) — a timing/
         coordination defect is exactly what an uncoupled, asymmetric
         gripper would be expected to produce. Not fixable via the
         specific `PhysxMimicJointAPI` mechanism Experiment 19 already
         ruled out, but now the most concretely-evidenced next lever,
         ahead of demonstration/imitation bootstrapping.
     - **Experiment 22: implement jaw synchronization as a software
       control-loop (jaw2's target continuously tracks jaw1's actual
       measured position) instead of Experiment 19's already-falsified
       physics-level constraint. Verified working by three independent
       checks, but exposed a new, different failure mode — reactive
       lag — that itself explains the continued null result.** Design
       spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment22-report.md`.
       - **A genuine investigation was required before trusting this
         run at all**: the full 1500-iteration training-time reward
         trajectories were bit-for-bit identical to Experiment 21's, at
         every logged point, including `reaching_object` (a pure
         arm-position metric with no gripper dependency) — a serious
         red flag investigated rather than dismissed or over-reacted
         to. Re-running the instrumented contact diagnostic against
         Experiment 22's own checkpoint (with raw per-step jaw
         positions now also logged) confirmed the mechanism genuinely
         is active — jaw2 diverges from jaw1 from the very first step
         in a pattern consistent with real mirroring-with-lag — and the
         two checkpoints' own contact diagnostics differ meaningfully
         (Experiment 21: `max_jaw1_force=6.73N`; Experiment 22: `0.0N`),
         proving the underlying learned policies are not identical
         despite the aggregate scalars matching. Conclusion: a real,
         non-obvious property of how coarse those specific logged
         reward signals are (`pregrasp_readiness`'s closedness term
         uses the *mean* of both jaws), not a sign the mechanism failed
         to run — recorded in the report in full for any future
         experiment comparing aggregate training scalars between
         low-level-actuator-only config differences.
       - **`both_magnitude_ok_steps` stays at exactly `0/750` and
         `lifting_object` at `0/1500`** — null by the strict success
         criteria, matching Experiments 17, 18, 20, and 21. But a new,
         specific diagnostic (`max_jaw_pos_diff=0.011m`, 79% of the full
         0.014m gripper travel range) explains *why* more precisely than
         a bare null would: jaw2 reacts to where jaw1 *already is*, not
         where it's headed, so whenever the policy moves the gripper
         quickly, jaw2 structurally lags a full control step behind a
         moving target. **Mirroring relocated the source of divergence
         (from Task 6's asymmetric-tracking-under-load finding to a new
         reactive-lag finding) rather than eliminating it.**
       - **Net assessment: this narrows the software-mirroring design
         space rather than closing it off.** A corrected version would
         need to account for jaw1's *velocity* (e.g. tracking jaw1's own
         commanded target, known instantly with zero lag, rather than
         its physically-settled actual position, which is inherently
         one step stale) — a concrete, specific next design, not a dead
         end. Six consecutive experiments (17, 18, 19, 20, 21, 22) have
         now each targeted a different specific mechanism for the same
         underlying problem, narrowing it further each time without yet
         resolving it — the two most concrete remaining levers are (a)
         the corrected jaw1-target-following mirroring design just
         identified, or (b) demonstration/imitation bootstrapping for
         the lift primitive, increasingly the more attractive option
         given how many independent mechanisms have now been tried.
     - **Experiment 23: residual RL over a classical 5-waypoint pursuit
       controller, with the specific literature-grounded warm-start
       (Johannink et al. 2019) Experiment 13's own ROADMAP entry
       diagnosed as missing but never implemented — a structurally new
       direction after six consecutive reward/action-space tweaks
       (17-22), not a seventh tweak on the same paradigm. Warm-start
       mechanism independently verified genuinely working via a hard
       gate; still null.** Design spec:
       `docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md`.
       Full run data:
       `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md`.
       - **Hard gate passed before committing to the full run**: a real
         1300-step env rollout confirmed `residual_authority` (the new
         action term's warm-start ramp) genuinely rises from ~0 at step
         0 to exactly 1.0 at step 1200 and stays clamped, reading the
         live action term's own internal state inside an actual
         `ManagerBasedRLEnv` — independently re-verified by a reviewer
         who recomputed every logged value against the ramp formula and
         confirmed it matches the term's real internal blend exactly,
         not just a self-reported "PASS".
       - **`lifting_object` stays at exactly `0/1500`, `both_magnitude_ok_steps`
         at exactly `0/750`** — null by the strict criteria, matching
         Experiments 17, 18, 20, 21, and 22. Since the warm-start
         mechanism is now independently confirmed to genuinely work,
         this specifically falsifies "the warm-start gap explains
         Experiment 13's regression and blocks the residual mechanism
         from working" — the classical-base-plus-residual paradigm
         itself appears to be a more fundamental non-fit for this
         task's grasp/lift sub-problem, not an implementation-gap
         problem.
       - **A genuine methodological gap was found and recorded, not
         glossed over**: the contact diagnostic's own log revealed
         `residual_authority` only reached 0.625 by the diagnostic's own
         final step, not 1.0 — because `_step_count` lives on the action
         term instance, not in the saved checkpoint, so a fresh
         diagnostic script re-ramps from zero regardless of how much
         training the loaded policy actually saw. This does NOT
         confound the training-time result itself (the policy spent
         ~97% of its 1500-iteration training run at full authority,
         `warmup_steps=1200` being only ~3.3% of the 36,000-step
         budget) but is a real gap in this repo's residual-action
         diagnostic tooling worth fixing (force `_step_count` past
         `warmup_steps` at construction) before trusting any future
         residual-RL diagnostic's contact numbers at face value.
       - **Net assessment: eight consecutive experiments (13, 17-23)
         spanning reward shaping, grasp gating, orientation bias,
         proximity gating, software jaw mirroring, and now
         warm-started residual RL, have all converged on the identical
         null.** Having now exhausted the most well-grounded remaining
         variant within the reward/action-space-engineering-over-pure-
         PPO-exploration technique family, demonstration/imitation
         bootstrapping (previously deferred as impractical given a
         human-teleoperation requirement) is the most concretely
         justified next direction — either a from-scratch expert-
         controller-generated demonstration pipeline bypassing that
         requirement, or a renewed look at what such a pipeline would
         concretely require in this repo's specific setup.
2. Shape classifier misclassifies cube/rectangular-prism as "sphere" against
   real depth data. Root-caused: `PLANARITY_RESIDUAL_THRESHOLD` (tuned on
   near-noiseless synthetic data) doesn't generalize to real sensor noise.
   Circularity looks more promising as the primary signal, but real
   tilt/plane-fit readings were also noisy on small, low-pixel-count real
   objects — may need more than a threshold nudge.
   - **Literature research (citation-verified)**: delegated real research on
     sensing-modality and technique choices (junior researcher + independent
     senior citation review — the senior review caught a fabricated
     verbatim quote and a fabricated "3-6mm RMS depth noise at 0.5m" figure
     built on garbled, self-contradictory arithmetic, echoing the same
     fabricated-precision-number pattern already seen once this session in
     the grasp-reward research; full docs in
     `docs/superpowers/specs/research/2026-07-05-perception-sensing-literature-junior.md`
     and `-senior-review.md`). What survived verification:
     - **RGB-D remains the right modality, not LiDAR** — LiDAR's angular
       resolution creates mm-scale uncertainty on objects this small
       (9-30mm) at 0.5m range (a correctly-labeled trig estimate, ~8.7mm per
       1° beam divergence — not dressed up as a citation). No modality
       change needed; the existing top-down RGB-D camera is the right
       sensor.
     - **The 0.8mm threshold is real-noise-unrealistic** (qualitatively
       confirmed - depth noise at this range is almost certainly far
       larger), but the specific "recalibrate to 9-18mm" number has no real
       citation basis and should NOT be adopted on authority. Correct next
       step: **empirically measure the real planarity-residual distribution
       on a flat reference surface at the actual 0.55m working distance**
       (directly actionable in this repo's own sim - the perception camera
       already has a realistic noise model, no physical hardware needed;
       `scripts/perception_calibration.py` is the existing tool this could
       extend) and set the threshold to mean + 3×std from that measurement,
       not a literature-derived constant.
     - **Phase 2 fallback** (if recalibration alone isn't enough): a
       **spin image** local descriptor + lightweight classifier — verbatim-
       verified 85-91% accuracy on ModelNet10
       (DOI:10.3390/s24237749/Sensors 2024) — not FPFH (the junior
       misapplied this paper's result to FPFH; the paper explicitly tested
       and rejected FPFH in favor of spin image).
     - **Phase 3 fallback** (if still insufficient): PointNet-style
       end-to-end learned classifier, verbatim-verified noise/corruption
       robust (3.8% accuracy drop at 50% missing points, >80% accuracy at
       20% outlier points - arXiv:1612.00593) - more effort (needs training
       data) but robust to noise by construction rather than via a
       hand-tuned threshold.
     - **Multi-view fusion** (a second camera angle) is a well-verified
       separate enhancement (MVTN, 13% accuracy gain at 50% occlusion,
       arXiv:2011.13244) but is a bigger architecture change (new camera)
       than this bug fix needs - noted as a future direction, not part of
       this fix.
     - **Phase 1 attempted, falsified — real root cause found, is NOT sensor
       noise.** Empirically measured the real camera's planarity-residual
       distribution (`scripts/measure_planarity_residual.py`): the render is
       almost perfectly noise-free (~30nm dead-center), but at the objects'
       real off-center scene positions (~20cm to the side of the top-down
       camera), the segmented cluster genuinely includes a sliver of the
       object's *oblique-visible side wall* — real 3D geometry, not noise.
       This inflates the cube's own residual (0.00293m) *above* the sphere's
       own residual (0.00210m) at their real positions — an ordering
       inversion meaning **no single `PLANARITY_RESIDUAL_THRESHOLD` value can
       classify both correctly**. Recalibrating to the measured mean+3σ
       (0.0045m) was tried and reverted: it measurably made real end-to-end
       classification worse (2/4 → 1/4 objects correct — cube/rect_prism
       stayed wrong, and the previously-correct sphere newly broke).
       `PLANARITY_RESIDUAL_THRESHOLD` is back at `0.0008`; full data in
       `docs/superpowers/plans/2026-07-05-perception-threshold-recalibration-report.md`.
     - **Structural fix implemented: 3/4 objects now correct (up from 2/4),
       fixing the originally-reported bug — with a new, well-characterized
       wedge regression.** Added `_restrict_to_top_band()`/`TOP_BAND_MARGIN`
       (4mm, geometrically derived then empirically swept 0.5-10mm against
       the real camera) in `perception/shape_classifier.py`: the plane-fit
       residual/tilt now use only points within the margin of the cluster's
       own top, excluding the oblique side-wall sliver at the source rather
       than trying to threshold around its effect. Verified (3 repeated real
       end-to-end runs via `scripts/perception_classification_check.py`):
       **cube → cube, rect_prism → rectangular_prism, sphere → sphere**, all
       correct — the exact ROADMAP bug is fixed. All 25 `perception/tests/`
       unit tests still pass. **New regression: wedge → cube.** Root cause
       is structural, not a tuning miss: the wedge's real tilted face spans
       nearly its whole height range, so a top-band crop thin enough to
       exclude cube/rect_prism's side-wall sliver (≲4mm) also destroys the
       wedge's own tilt signal (measured tilt drops 53°→3.5° within the
       band); the wedge's tilt only recovers past a margin (~10mm) that
       reintroduces the cube/rect_prism regression — no single margin fixes
       all four shapes simultaneously. Full margin-sweep data in
       `docs/superpowers/plans/2026-07-05-perception-sidewall-fix-report.md`.
       Recommended follow-up (not yet done): give the wedge's tilt check a
       RANSAC-style robust plane fit over the *full* cluster (robust to the
       side-wall sliver as an outlier population) instead of relying on the
       same top-band-restricted fit used for the residual/circularity
       checks — decided to ship this net improvement now (3/4 > 2/4, and
       fixes the specific bug this item was originally opened for) rather
       than block on a fully general fix.
     - **LiDAR empirically tried, confirms RGB-D conclusion more decisively
       than the literature review alone.** Per direct user request, added an
       experimental base-mounted LiDAR (`RayCasterCfg` + `LidarPatternCfg`,
       16-channel, tried and then reverted — code didn't solve anything, kept
       only as a documented negative result:
       `docs/superpowers/plans/2026-07-05-ar4-base-lidar-report.md`).
       Finding: this Isaac Lab installation's `RayCaster` only ray-casts
       against **one static mesh** (enforced in code) — it is architecturally
       blind to the dynamic cube/sphere/rect_prism/wedge entirely, seeing
       only the ground plane, regardless of resolution. Separately, even
       ignoring that limitation, measured angular resolution in the
       workspace-relevant channels (~7mm at 0.3m range) is comparable to the
       18mm sphere — too coarse either way. Confirms: stick with RGB-D, no
       further LiDAR investigation planned.
     - **Follow-up spike (2026-07-06): confirmed the single-static-mesh
       limitation cannot be worked around by pointing `mesh_prim_paths` at
       the object instead of the ground — root-caused, not just re-tested.**
       Per direct user request ("create a lidar with higher resolution"),
       investigated whether higher resolution could compensate, first
       checking whether the object-vs-ground limitation itself could be
       sidestepped. Two independent, code-verified blockers, not just one:
       (a) `RayCaster._initialize_warp_meshes`
       (`isaaclab/sensors/ray_caster/ray_caster.py:162-210`) bakes the
       target mesh's world-space vertices into an immutable `wp.Mesh`
       **once**, at sim initialization (gated by a `_is_initialized` flag
       only cleared on a full timeline stop, never on a per-env reset) —
       confirmed empirically too: teleporting a mesh-typed `RigidObject`
       (the wedge; cube/rect_prism/sphere are analytic USD primitives, not
       `UsdGeom.Mesh`, see (b)) from `(0,0,0.009)` to `(0.7,0.7,0.009)` left
       the RayCaster's `ray_hits_w` centroid completely unchanged. (b)
       Independent of (a), this repo's actual graspable objects
       (cube/rect_prism/sphere in `objects_cfg.py`) are procedural/analytic
       USD primitives, not `UsdGeom.Mesh` prims at all — pointing
       `mesh_prim_paths` at any of them raises
       `RuntimeError: Invalid mesh prim path` immediately, reproducible
       across repeated runs. Conclusion: no resolution setting or
       target-selection change can make this Isaac Lab version's
       `RayCaster` see a dynamic graspable object — the limitation is
       structural, not tunable. RGB-D remains the only viable perception
       modality in this stack; no further LiDAR investigation planned.
3. `interactive_demo.py` live GUI drag verification (plan Task 10, Step 4)
   was never performed — needs a human running it without `--headless` to
   confirm the physical drag → settle → pick-and-place → idle-again flow.
4. Minor/cosmetic, non-blocking: `perception/tests/conftest.py`'s
   sys.path-insert comment overstates how many directory levels it climbs;
   `interactive_demo.py` hardcodes `clip_actions=None` instead of reading it
   from agent config; a redundant filter duplicates `find_by_shape`.
5. Final whole-branch review for the perception-integration plan (Task 12)
   was explicitly skipped per user instruction — still pending whenever that
   work resumes.
6. **Experiment 24 Gate 1 (scripted-oracle demonstration bootstrapping):
   FAIL.** Built a non-learned, reactive-differential-IK oracle
   (`scripts/oracle_rollout.py`) meant to follow this repo's existing
   5-waypoint pick-and-place path without human teleoperation, to
   bootstrap BC pretraining for a warm-started RL finetune. It stalls —
   end-effector position frozen bit-identical for 50+ steps — before
   reaching the grasp waypoint in the overwhelming majority of sampled
   episodes (~1/24 advanced past waypoint 0 in the original smoke tests).
   Three architecturally distinct fixes were tried and failed: Cartesian-
   space escape-perturbation (ported from `classical_pickplace_demo.py`,
   verified firing correctly but with zero effect), joint-space
   perturbation (verified reaching the commanded target with zero effect
   on actual position), and integral/accumulated-error pursuit correction
   (no effect at a conservative gain, **actively diverged the arm away
   from the target** at a stronger gain — the third fix made things worse,
   not better, when strengthened). Root-cause diagnostics (independently
   re-verified, not just accepted from a first pass) ruled out actuator
   torque saturation (never clips, peak torque 14.1N·m vs. a 20N·m limit),
   a hard joint-limit hit (~13° of margin), contact/collision (0.0N
   throughout), and classic Jacobian rank-collapse (smallest singular
   value plateaus ~0.15, not ~0) — evidence instead points to a genuine
   fixed point of the receding-horizon control loop in a poorly-
   conditioned kinematic region, where the linearized IK correction stops
   reliably pointing toward the goal. This independently echoes
   Experiment 20's own prior conclusion (a damping sweep across six
   values, multiple target formulations) that single-Newton-step-per-env-
   step differential IK is not a stable mechanism on this arm — reached
   via a structurally different investigation path this time (waypoint-
   pursuit stall vs. orientation-holding drift). Full evidence trail and
   recommended next directions (multi-iteration IK-before-`env.step()`
   convergence, or reconsidering human teleoperation as the demonstration
   source after all) in
   `docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md`.
   Gate 2's implementation plan was explicitly not written, per the
   original plan's own scope note for a FAIL verdict.
7. **A purely classical (non-RL) IK-driven joint pipeline also cannot
   reliably reach the cube's grasp pose — likely a genuine kinematic
   obstruction, not an RL or per-script bug.** Prompted by revisiting
   Experiments 11-15 (whose scalar-only "promising" trends were never
   trustworthy — same ungated-reward blind spot Experiment 16 later
   exposed) and a direct redirect to verify basic classical joint driving
   first. `scripts/grasp_demo.py` (dormant since this repo's first
   commits) was rebuilt to solve IK once per waypoint via live simulator
   feedback (not a reactive per-step loop), with two real bugs found and
   fixed along the way — a stale-joint-state carry-over, and the same
   "unbounded IK Cartesian jump" bug `scripts/oracle_rollout.py` already
   found (a single DLS Newton step toward a target tens of cm away
   produces unrealistic joint deltas; fixed by bounding the per-round
   step to 0.05m, matching `oracle_rollout.py`'s own fix). Bounding the
   step stopped the residual from diverging (was growing 0.35m→0.67m,
   now improves 0.9m→0.33m) but it still plateaus well short of the
   cube — verified via both logged telemetry and video-frame inspection,
   cube height frozen at spawn throughout the lift/hold phases, gripper
   visibly separated from the cube in every sampled frame. Both the
   pregrasp and grasp waypoints (5cm apart) independently converged to
   nearly the identical stuck joint configuration. **This is the fourth
   independent script/mechanism** (the original `classical_pickplace_demo.py`
   kinematic-singularity stall, plus three architecturally distinct
   attempts across `oracle_rollout.py`/`grasp_demo.py`) **to hit the same
   "converges partway, then stalls" signature at/near this cube
   position**, independent of RL, independent of action-space
   formulation. Grasp-force verification via contact sensors was
   deliberately dropped from this diagnostic per direct instruction — the
   real AR4 hardware has no gripper force sensors either, so the gripper
   is treated as "dumb" (open/closed command only), matching hardware.
   **Update, same session: this was investigated further and the
   unreachability hypothesis was refuted — see item 8 below.**
8. **Resolved: the stall was a DLS Newton-step local-minimum trap, not
   genuine unreachability — the arm CAN be driven to within a few
   centimeters of the cube via grid-search-then-polish.** Direct forward-
   kinematics measurement (`scripts/measure_reach_envelope.py`, no IK
   solver involved at all) proved the cube target is comfortably within
   the arm's reach envelope (0.538m max measured reach vs. 0.344m
   needed) — refuting item 7's unreachability hypothesis. Seeding the
   proven bounded-step DLS solve from a geometrically-aimed starting
   configuration (`scripts/ik_seeded_start.py`, after empirically
   recalibrating a wrong assumed joint_1-to-azimuth sign convention)
   barely helped, confirming the DLS iteration itself gets trapped in a
   local minimum independent of starting direction. A direct forward-
   kinematics grid search (`scripts/ik_grid_search.py`, 625 points, no
   iteration, can't get stuck in a local minimum) found a configuration
   within 3.5-6cm of the target; a DLS polish from that seed
   (`scripts/ik_polish_from_grid.py`) closed it to 3.648cm before
   plateauing again (same bit-exact fixed-point signature as every prior
   attempt) — DLS still can't fully close even a small, well-conditioned
   gap on its own. `scripts/grasp_demo_v2.py` applied this method to both
   waypoints and ran the full phased pick/lift/hold/release sequence,
   also finding and fixing a second real bug (phase-transition
   interpolation using the previous phase's intended target as the next
   phase's baseline instead of its actual achieved position — the
   "correct-looking" fix, re-reading actual state each phase, made
   things far worse since this arm's actuators track a fixed commanded
   target much better than a continuously-ramping one; reverted to
   direct target-commanding per phase). Result: joint tracking improved
   to ~0.19-0.32 rad residual, and video shows the gripper genuinely
   close to the cube for the first time this session (previously the
   cube sat completely disconnected from the arm in every attempt) — but
   the cube still never moves; the combined IK gap plus phase-tracking
   residual is still enough to miss contact. **Net: the classical joint-
   driving reachability question is answered (yes, with the right
   method) — a full clean grasp+lift is not yet achieved, and closing
   the remaining few centimeters (finer/better-centered grid search,
   longer per-phase settle time, or re-solving from the actual achieved
   state at each phase transition) is flagged as an open next step, not
   pursued indefinitely this session.** Full evidence trail: see the
   commit `8b72ef7` message and the conversation record.
9. **Physics-fidelity pass (dt, collision, EE-frame) verified good; surfaced
   that the classical grasp (item 8) still misses with zero contact force,
   not just a positioning residual.** Requested directly: smaller physics
   steps, correct collision behavior, EE frame at the gripper not the wrist.
   All three independently re-verified (junior-engineer executed, senior-
   engineer independently re-checked with its own instrumentation, not just
   re-reading the junior's report):
   - **dt/decimation**: halved `sim.dt` (120Hz→240Hz for `env_cfg.py`/
     `grasp_verify_env_cfg.py`; 100Hz→200Hz for `pickplace_mirror_env_cfg.py`),
     doubled `decimation` in lockstep so control period is unchanged (1/60s
     and 0.02s respectively) — only PhysX substep fidelity changes, the RL
     MDP interface (what a trained policy perceives/does/is scored on) is
     untouched. Commit `eb5f302`.
   - **Collision**: `objects_cfg.py`'s cube/rect_prism/sphere/wedge
     `_COLLISION_PROPS` leave `contact_offset`/`rest_offset` at PhysX's
     auto-compute (`-inf` in the USD schema, no public API exposes the
     resolved runtime value) — empirically bounded to well under 0.5mm via
     a 2400Hz free-fall drop test (pure -9.81 m/s² acceleration all the way
     to a 0.42mm gap, then a clean single-substep arrest, final rest height
     0.006000m exact). Negligible relative to the cube's 6mm half-extent;
     no override needed.
   - **EE frame**: `_EE_OFFSET=(0,0,0.036)` on `link_6` (previously
     corrected from a wrong 0.09 — see item above/earlier in this file) is
     confirmed still correct, both numerically (`ee_frame.data.target_pos_w`
     vs. the real jaw-link midpoint, <0.001mm residual) and — new this
     pass — **visually**, via `debug_vis=True` in a live GUI run: the
     rendered marker sits between the jaw tips, not back at link_6/the
     wrist.
   - **Not caused by this pass, but surfaced by it**: running the untracked
     `scripts/interactive_joint_demo.py` (a live-GUI closed-form-IK pick-
     cycle demo, mid-development, never previously run with its own gripper
     contact sensors enabled) with `force_matrix_w` instrumentation showed
     **exactly 0.0N contact force on both jaws across every cycle tested**
     — the gripper closes on empty space, not partially on the cube. Jaw
     terminal positions varied cycle-to-cycle (near-fully-open in one,
     partial/asymmetric in another) rather than the consistent stopping
     position contact with a real object would produce. A first hypothesis
     (the script bypasses `env.step()`/decimation and counts settle time in
     raw substeps that silently halved in real duration when dt halved) was
     tested and fixed (commit `e00dd11`, settle time now derived from
     `env.physics_dt`) but did **not** fix the miss — ruling that out as the
     cause. This is the same unresolved miss as item 8 above (a classical,
     non-RL closed-form-IK approach that gets close but doesn't reliably
     center the cube between the open jaws before closing), now confirmed
     with contact-sensor ground truth rather than inferred from video/
     residual distance alone. Left open, not pursued further this pass —
     scope today was physics fidelity, not grasp-approach geometry.
   - **Follow-up investigation (same day): root-caused item 9's grasp miss
     as (B) a limitation of the classical demo's fixed-wrist control, not
     (A) a genuine Isaac Sim integration bug.** Independently confirmed via
     `scripts/plot_arm_skeleton.py` and direct measurement: gripper jaw
     collision geometry is present and correct (instance-proxy-aware USD
     traversal), a real `PhysxMimicJointAPI` coupling exists on the
     currently-built asset, and an isolated gripper-only test (arm at rest,
     no IK approach) closes and opens fully and cleanly — ruling out
     "gripper structurally can't close." The actual failure reproduced live:
     `interactive_joint_demo.py`'s closed-loop IK refinement *diverges*
     (1.46cm → 1.74cm residual over 4 rounds) at this cube's low, extended
     pose, leaving the jaws ~2.9cm from the cube when closing is
     attempted — a positioning/dynamics miss intrinsic to this one script's
     fixed-wrist (`q4=q5=q6=0`), open-loop-hold approach, not a sim-wiring
     defect. Assessed as clear to proceed with RL training (which drives
     all 6 joints with continuous closed-loop correction, not bound to this
     demo's simplifications) — **however, see item 10 below: RL training
     on the existing cube task was not attempted, because a deeper,
     already-well-documented problem made that a bad bet before even
     reaching this question.**
10. **Experiment 25: touch-cube-then-reach-goal, grasp removed entirely
    (structural pivot, direct user decision).** Before training
    `pickplace_mirror_env_cfg.py` "from scratch" as originally planned,
    found two reasons not to: (a) six consecutive prior experiments
    (17-22) each targeted a different mechanism for the same problem — the
    gripper's two jaws are not mechanically coupled (the source URDF's
    `mimic` constraint confirmed unenforced by Isaac Sim's USD import) —
    and both a physics-level fix (Experiment 19, two configurations, both
    regressions, reverted) and a software-level fix (Experiment 22, a new
    "reactive lag" failure mode) failed to resolve it; (b)
    `pickplace_mirror_env_cfg.py`'s own reward (`staged_milestone_bonus` →
    `_raw_lift_progress_mirrored`) turned out to combine reach/grasp/
    lift/goal as a plain **ungated** weighted sum — the exact reward shape
    Experiment 16 already found exploitable via wrist-wedging, without
    Experiment 17's grasp-gating fix (which lives only in the separate
    `pickplace_graspgated_env_cfg.py` lineage). Flagged to the user rather
    than trained blind; **direct decision: drop grasp/lift entirely**,
    reduce to two-stage sequential end-effector reaching (touch the cube's
    top, then reach a fixed goal point) — leaning entirely on the one
    sub-behavior that has converged reliably (~0.92-0.95) across nearly
    every experiment in this project's history, independent of reward or
    action-space design.
    - Design: `docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md`.
      Plan: `docs/superpowers/plans/2026-07-09-ar4-experiment25-touch-goal-reach-implementation.md`.
      New `Ar4PickPlaceTouchGoalEnvCfg` (`tasks/ar4/pickplace_touchgoal_env_cfg.py`):
      fixed cube `(0.20, 0.28, 0.006)`, fixed goal `(-0.20, 0.28, 0.15)`,
      **arm-only action space** (no gripper action term at all — the
      gripper joints stay physically present but unactuated), reusing the
      already-`_EE_OFFSET`-corrected `ee_frame` as the touch/goal reference
      point. Built via subagent-driven-development: 4 plan tasks, each
      independently task-reviewed clean.
    - **Final whole-branch review (dispatched on the most capable available
      model, per this project's own SDD convention for architecture-level
      review) caught a real, would-have-been-expensive Critical defect
      before any training run: the reused running-max milestone mechanism
      (`staged_milestone_bonus`'s pattern, valid for the lift task because
      its stages are co-satisfiable along a success trajectory) is
      *unsound* here because the touch and goal points are ~0.42m apart —
      two independent narrow tanh bumps, summed, dip from ~0.3 (at touch)
      to ~0.02 (partway to goal) before recovering to ~0.7 (at goal), and
      running-max means once 0.3 is banked, reward stays at exactly zero
      until raw exceeds 0.3 again — not until ~93% of the way to goal.**
      This would very likely have produced "touch-and-freeze" behavior
      indistinguishable from the sphere era's original "reach, grip,
      freeze" signature, and been misread as "even reduced-to-reaching
      fails" — precisely the shaped-scalar-looks-fine-behavior-doesn't trap
      the Tier-1 verification standard exists to prevent. Independently
      re-derived (not just trusted) by the Principal before dispatching a
      fix, and again by a second reviewer after the fix landed. Also
      flagged: the goal was read from the cube's *live* position every
      step, but the cube is a dynamic (non-kinematic) RigidObject, so an
      incidental touch-contact nudge could silently move the "fixed" goal;
      and no test exercised the touch gate's positive branch at all (the
      smoke test only used all-zero actions).
    - **Fix** (commit `7170b6b`, plus a trivial follow-up constant-drift
      fix): extracted the reward math into a new Isaac-Lab-free module
      (`tasks/ar4/touch_goal_reward.py`) with a genuinely monotonic
      post-touch potential (`0.3 + 0.7·clamp(1 - goal_dist/touch_to_goal_dist,
      0, 1)` — linear in distance, provably non-decreasing along the
      straight touch→goal line, so running-max can never stall once touch
      registers); added `set_touch_goal_position`, a reset-time event
      snapshotting the goal once from the cube's position at reset
      (decoupling it from any live cube displacement); added
      `tests/test_touch_goal_reward.py`, a genuine sim-independent pytest
      suite (3/3 passing, no Isaac Sim launch needed) directly proving the
      monotonicity property the original formula lacked. Independently
      re-reviewed and confirmed correct by re-deriving the math from
      scratch, not by re-reading the fix's own claims.
    - **Status: implementation complete and verified, actual 1500-iteration
      training run not yet executed as of this entry** — see follow-up.
    - **Training run 1 (episode_length_s=5.0, copied from
      pickplace_mirror_env_cfg.py): `goal_reached` rose to a peak of
      ~0.37-0.39 partway through training, then declined to ~0.01-0.03 by
      the end, with `Mean episode length` at 248-250 of the 250-step max —
      most episodes running to timeout rather than succeeding.** Stopped
      before completion (iteration ~979/1500) once the decline pattern was
      clear; a wide multi-env training-camera video review at this point
      was inconclusive at that camera's zoom level.
    - **Episode length re-derived from Isaac Lab's own reference
      manipulation tasks, extended 5.0s -> 20.0s.** Isaac Lab's tasks scale
      episode length with task *structure*, not object count: Reach
      (single target) 12.0s, Lift (reach+grasp+lift, one object) 5.0s,
      Cabinet (reach+grasp+open) 8.0s, Stack (reach+grasp+lift+move+place,
      sequential multi-stage — the closest structural analog to
      touch-then-goal) 30.0s. External literature on long-horizon/
      multi-stage manipulation RL (Relay Policy Learning's hierarchical
      fixed-step budgets, SLIM's multi-stage pick-and-place benchmark,
      Meta-World's 500-step-per-episode convention) consistently uses
      proportionally longer horizons for multi-stage tasks. 20.0s sits
      inside the range Isaac Lab's own tasks establish.
    - **Training run 2 (episode_length_s=20.0): completed cleanly.
      `Episode_Termination/goal_reached` climbed from 0 and held in the
      0.5-0.7 band for the back half of training, finishing at 0.5987;
      `time_out` finished at 0.4015; `Loss/value_function` stayed small
      and bounded throughout (no divergence signature).** A materially
      different shape from run 1 (sustained, not peak-then-decline).
    - **Instrumented rollout of the trained checkpoint (`model_1499.pt`)
      found the training-time rate reflects exploration noise, not the
      deployed policy's actual reliability.** Deterministic action
      (`ActorCritic.act_inference`, what a deployed policy actually
      uses): 32/32 rollout episodes touched the cube, 2/32 (6.25%) reached
      the goal — the failures cluster tightly at 0.0175-0.0285m past the
      touch point, just outside the 0.02m goal tolerance, not scattered.
      Stochastic action (`ActorCritic.act`, the same sampling used during
      training rollouts): 32/32 touched, 29/32 (90.6%) reached goal.
      Independent recomputation from raw `ee_frame`/cube/goal state agreed
      with the actual termination signal on 100% of episodes in both
      conditions, ruling out an instrumentation bug. Close-up single-env
      video (`tasks/ar4/touchgoal_democam_env_cfg.py`, a new close-camera
      variant built for this check after the wide training camera proved
      inconclusive twice) confirms the shape directly: the arm curls down
      to the cube by step ~22 (0.44s into a 1000-step episode), extends
      toward the goal, and stops short of it at timeout in the
      deterministic-only sample recorded.
    - **Net assessment: touch is solved; reach-to-goal is close but not
      solved by the mean policy — it lands just outside a 2cm tolerance
      band, consistently.** Candidate next steps, not yet tried: widen
      `GOAL_TOLERANCE` (the simplest lever, directly targets the observed
      gap), or continue training longer to sharpen precision under the
      existing tolerance. Not pursued further in this pass — superseded by
      the next direction (reintroducing the gripper) per direct user
      instruction.
11. **Experiment 26: reintroduce the gripper (grasp/lift/carry/goal), 30s
    episodes. Trained policy reaches close to the cube fast (~2.4cm by
    0.5s) but never holds that position or grasps — it oscillates in
    reach distance for the remaining ~29s of every episode.** Design:
    `docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md`.
    Plan: `docs/superpowers/plans/2026-07-09-ar4-experiment26-gripper-reintroduction-implementation.md`.
    - Composed two previously-validated fixes (Experiment 21's
      proximity-gated gripper, Experiment 17's antipodal grasp gate) with
      a 4-stage extension of Experiment 25's monotonic staged-potential
      reward (reach → grasp → lift → goal) and a Stack-task-precedented
      30.0s episode length. A third originally-planned fix (Experiment
      22's jaw-mirroring mechanism) was found and retired during final
      whole-branch review — see item 10's own follow-up entry above and
      `tasks/ar4/pickplace_graspgoal_env_cfg.py`'s `ActionsCfg` docstring
      for the full account.
    - **Training (1500 iterations): `Episode_Termination/cube_reached_goal`
      stayed at exactly `0.0000` for the entire run — not a single logged
      point showed any nonzero value.** `Episode_Reward/grasp_goal_milestone_bonus`
      rose from `0.0001` to `~0.0037` in the first handful of iterations,
      then stayed completely flat at that value for the rest of training
      (iteration ~15 through 1500). `Episode_Termination/time_out` was
      `1.0000` throughout — every single episode ran the full 1500-step/
      30s length. Training itself was stable throughout (`Loss/value_function`
      ≈0, `Mean action noise std` ≈1.0-1.1, no divergence signature).
    - **A front, head-on close-up camera** (`tasks/ar4/graspgoal_democam_env_cfg.py`,
      built per direct user request) was used for an initial visual check
      via sparse (3-second-interval) frame sampling. That sampling showed
      apparently-identical poses at every sampled instant and was
      (wrongly) read as "the arm never moves at all" - **this
      characterization has since been corrected** (see below);  3-second
      sampling is too sparse to distinguish a true freeze from an
      oscillating trajectory that happens to revisit similar poses.
    - **A separate root-cause investigation's instrumented rollout
      reported a different pattern**: the arm reaches to within ~2.4cm of
      the cube and then holds. This was also incomplete - correct about
      the fast initial reach, wrong about the "holds" part.
    - **Resolved directly with a per-step trajectory trace**
      (`scripts/graspgoal_reach_trajectory_check.py`, printing exact
      `reach_dist` every 10-100 steps across a full 1500-step/30s episode,
      4 envs, `model_1499.pt`), settling the disagreement between the
      above two reads with real numbers instead of either visual sampling
      or a single before/after check. **Verified behavior: `reach_dist`
      drops from ~0.52m at reset to ~0.024-0.026m by step 20-30
      (0.4-0.6s) - a fast, genuine, accurate reach. It does NOT hold
      there: for the remaining ~29s of the episode, `reach_dist`
      oscillates unpredictably, ranging roughly between 0.04m and 0.6m at
      different sampled points (e.g. step 900: `[0.494, 0.514, 0.341,
      0.269]`; step 1300: `[0.593, 0.208, 0.574, 0.052]`), never
      restabilizing near the cube and never crossing into `grasped`/
      `lifted` (both `False` for all 4 envs at every sampled point,
      start to finish).** Neither the original "complete static freeze"
      read nor the "reaches and holds" read is correct - the real
      signature is "reaches fast and accurately once, then wanders/
      oscillates without holding or re-settling, grasp never discovered."
    - **Building the demo camera surfaced and fixed a real, unrelated
      bug**: `scripts/graspgoal_closeup_video.py` and
      `scripts/touchgoal_closeup_video.py` (both share the same
      `camera.data.output["rgb"]`-reading code) were saving every frame
      vertically flipped (OpenGL framebuffer convention, row-0-at-bottom,
      never corrected before writing to PNG/mp4) - confirmed empirically
      (an unflipped render showed the ground grid at the top of frame,
      sky at the bottom) and fixed in both scripts. Item 10's touchgoal
      video-review conclusions were based on relative position/distance
      state, not pixel interpretation, so aren't invalidated by this -
      but the images themselves were genuinely inverted and this is
      recorded for the record.
    - **Net assessment: this is a plausible artifact of the running-max
      milestone reward mechanism combined with a grasp gate the policy
      never discovers.** The `grasp_goal_milestone_bonus`'s reach segment
      is a running MAX over `reach_progress` - once the policy achieves
      its single best (closest) approach early in an episode, that best
      is permanently banked; nothing in the reward differentiates staying
      close from wandering away afterward, and since the antipodal grasp
      gate is apparently never satisfied, there is no further
      outcome-relevant gradient for the rest of the episode. This is
      consistent with an initial fast, accurate reach (which IS rewarded,
      once) followed by directionless movement (which is neither
      rewarded nor penalized). This differs from this project's
      sphere-era "reach, grip, freeze" signature (which involved genuine
      contact before going static) - here contact/grasp is never
      achieved, but the arm does not freeze either; it keeps moving
      without converging. Given `--touchgoal` (arm-only, 2-stage)
      reliably converges under this same physics/PPO setup, the
      regression specifically implicates the reintroduced gripper action/
      observation surface or the reward's lack of an incentive to
      *hold* a good reach once achieved, not a general breakdown of PPO/
      physics. Not yet root-caused to a specific fix - flagged as the
      next investigation, not pursued further in this pass.

## Direction

Isaac-Lab-based robotics RL, expanding beyond AR4 manipulation into other
tasks/robots, object detection/perception, and mobility. No committed
roadmap items beyond AR4 yet — this is a stated direction, not a scoped
backlog.

## Vision platform (`vision/`, merged 2026-07-10)

The former Dice-Detection repo is now the `vision/` subtree (monorepo
merge, full history — see
`docs/superpowers/specs/2026-07-10-monorepo-merge-design.md`): Blender
synthetic-data generation, dataset plumbing, perception-model training/
eval, ONNX+manifest export. First completed study (dice-detector-v1,
`vision/docs/results/2026-07-dice-detector-v1/summary.md`): synthetic-only
YOLO11s transfers to real photos for d12/d20 but collapses on d8/d10 via a
systematic up-the-shape-ladder confusion — hypothesized apparent-size-as-
class-cue confound; five datagen-v2 recommendations recorded there. The
exported `vision/models/export/<variant>/manifest.json` is the deployment
interface a future robot-camera perception stack consumes.

**Active iteration loops (2026-07-10):** (1) dataset iteration — datagen-v2
close-up slice testing the apparent-size-confound hypothesis (Senior thread
running); (2) hyperparameter hill-climb (Tier 2 pattern, scored on a
real-val slice, frozen test reserved for verdicts) — deferred until the
dataset loop lands its first win.

**Datagen-v2 verdict (2026-07-13): hypothesis SUPPORTED, both
pre-registered criteria met.** The `s_v2` run (detection_v1 + 3,000-image
close-up slice with camera distance decoupled from class) raised real-test
mAP50 d8 0.090→0.442 and d10 0.097→0.534 (primary threshold 0.40 cleared
by both), with the d12/d20 guard passed (0.946/0.907, both slightly up).
The apparent-size-as-class-cue confound is confirmed as a major
contributor to v1's d8/d10 transfer collapse. New open item the guard
didn't cover: **d6 regressed 0.519→0.275** — top candidate question for
datagen iteration 3, alongside the still-large absolute gap to `s_plus_r`
real-fine-tuned mAP50-95 (0.71–0.77 vs s_v2's 0.08–0.41). Full
side-by-side + verdict note appended to
`vision/docs/superpowers/specs/2026-07-11-datagen-v2-closeup-design.md`;
per-class tables in `vision/docs/results/2026-07-dice-detector-v1/eval_s_v2.md`.

**Convergence milestone — dice + Franka + detection: ACHIEVED (2026-07-11,
dice-pick demo, `franka-panda-pivot`).** Given a commanded die type among
{d4, d8, d10, d12, d20}, the Franka arm picks up the CORRECT die on a
five-die table, with die identity and 3D position coming from the trained
`vision/` detector (deprojected via the scene depth camera) — sim ground
truth used for verification only. Result on seed 42: **4/5 die types pick
successfully** (d20/d12/d10/d8, z-gains 237-241mm, each verified by video
and GT check); d4 is the sole, pre-declared permitted failure (the
scripted grasp converges to sub-mm accuracy on the tetrahedron but
flat-pad closure squeezes it out — a real grasp-strategy problem, open
follow-up). Demo videos: `outputs/dice_demo/gate_v/dice_pick_<die>.mp4`.
Implementation: `scripts/dice_pick_demo.py` (gates a/p/g/v) +
`tasks/franka/dice_scene_cfg.py` + `vision/scripts/detect_for_sim.py`;
full gate-by-gate history in `.superpowers/sdd/dice-demo-report.md` and
the task 1-4 reports beside it. Load-bearing findings along the way:
dice USDs are authored mm-as-m (uniform 0.001 spawn-time scale matches
the detector's own training-render convention); `RigidObjectCfg`
`rigid_props`/`collision_props`/`mass_props` silently no-op on
schema-less USDs (runtime `.Apply()` + `modify_*_properties` required);
camera-sensor renders need explicit lighting (DistantLight) + RTX
convergence frames; a scripted DiffIK descent from the default ready
pose needs a joint-space prep stage + canonical straight-down
orientation (holding the ready-pose orientation funnels into
joint-limit branches); grasp-position tolerance must be small relative
to the object's own radius (15mm tol lost 15-18mm dice); the detector
needed a geometric-plausibility filter (rejected a table-hole false
positive that deprojected below the table) and a scene-contract
one-per-class recovery (tight per-candidate recrop — region-crop
upscale alone is useless because ultralytics letterboxes crops).
Original phased plan (kept for reference): Phase A dice-USD physics
validation; Phase P detector on Isaac renders + deprojection; Phase I
(still open, gated on RL lift progress): detection-derived state in a
trained *policy* (this demo is a scripted controller, not RL), then
shape-generalized *learned* grasping across die types (d4 last).

**RL joint-space die-lift (2026-07-12, `franka-panda-pivot`): FALSIFIED
for the d20 — failure isolated to the die ASSET, recipe itself works.**
Experiment (spec `docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md`,
report `docs/superpowers/plans/2026-07-11-joint-space-die-lift-report.md`):
swap Isaac Lab's validated Franka lift recipe to direct joint-position
actions (no IK, the exact `joint_pos` variant values) and the
physics-baked d20 die. Full 1500-iteration run: value loss bounded but
`position_error` never beat the do-nothing baseline (0.331 vs 0.216) and
`lifting_object` sat on its spawn-artifact floor all run; eval video +
instrumented heights confirm reach-then-settle, 0/8 sustained lifts. The
spec's pre-authorized fallback (identical config, object swapped back to
DexCube, `--variant joint-cube`) trained lift+carry decisively in the
same 1500 iterations: `position_error` 0.105 (half of baseline),
`lifting_object` 13.4/15, mean reward 138 vs 2. Conclusion: joint-space
no-IK is a viable action formulation on this platform; the d20 asset is
what fails. Untested candidate causes (deliberately stopped per the
spec's no-unauthorized-iteration rule): die size (measured 2026-07-12:
d20 30.3mm vs DexCube 48.0mm effective — only 1.6x, smaller than first
assumed), near-spherical rolling geometry, baked 0.01kg mass (vs ~0.11kg
default-density estimate for DexCube — ~11x gap, the largest measured
discrepancy), friction/material, 0.001 spawn-scale pipeline. Next
experiment (asset-bisect, spec in progress 2026-07-12): research pass
recommends mass-first (PhysX depenetration-impulse mechanism on light
objects under multi-finger contact), then size, then shape, then
provenance — citation review + spec/plan underway.

**Asset-bisect ladder CONCLUDED (2026-07-12 evening): SHAPE gates grasp
discovery.** Full protocol + numbers in
`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`. At identical
48mm/0.216kg/same-pipeline: our baked cube trains 3/3 seeds, the d20
1/3; at 30.3mm the d20 is 0/4 across full runs (deterministic). Mass ruled out (0/3 at
21.6x); our bake pipeline exonerated (its cube = DexCube-grade
reliability). First-ever learned die lift achieved en route (rung-2
seed 123, video delivered). Gotcha for all future baked assets:
Xform-root/Mesh-child structure required (bare-Mesh default prim
silently loses PhysX collision). Next experiment (needs spec): object
curriculum — train where discovery is reliable, anneal toward the 30mm
d20.

**Cloud training pipeline PROVEN (2026-07-13, attempt 3).** First GPU
instance ever created on the project (SPOT g2-standard-4 + 1x L4,
us-central1-a) after the `GPUS-ALL-REGIONS-per-project=1` quota grant
landed 2026-07-12 23:09Z (raise-to-4 denied — exactly one cloud GPU,
spot or on-demand, until billing history matures). Full pipeline
exercised end-to-end: create → Isaac Sim 5.1 + Isaac Lab v2.3.1 pip
install (three real env gaps found+recipe'd: deadsnakes Python 3.11,
`flatdict` build-isolation silent skip, Vulkan/GL libs on the DLVM
compute-only driver) → ik-cube 4096-env headless training to 1200/1500
iters (~2.8GB/23GB VRAM — huge headroom) → GCS sync (25 checkpoints +
events + manifest at
`gs://rl-manipulation-hks-runs/cloud-shakedown/ik-cube/seed42/2026-07-13_13-34-38/`)
→ full teardown (zero instances/disks/snapshots verified). Two genuine
SPOT preemptions mid-run; after the second, all 9 surveyed zones were
simultaneously spot-stocked-out — recovery pattern (snapshot boot disk,
sync from a cheap non-GPU instance) now in the recipe. Total cost <$1.
Recipe: `docs/cloud/franka-cloud-shakedown.md` (status: PROVEN).
Implication: cloud is a working second training lane (one GPU), and
SPOT preemption churn is real — long runs need checkpoint-resume
tolerance or on-demand provisioning.

**Size-curriculum (mixed-size DR) verdict: FALSIFIED 0/3 (2026-07-13).**
No sustained lift in any seed's all-30.3mm eval (0/8, 0/8, 0/8);
lifting never left the spawn-z floor at any size during training.
Mechanism diagnosis: mixing five sizes diluted the 48mm population
~5x, and 48mm discovery was already marginal (bisect: 1/3) — the
curriculum's transfer source never fired. Full verdict + ops notes
appended to `docs/superpowers/specs/2026-07-13-size-curriculum-design.md`.

**Size-curriculum staged-anneal fallback verdict: also FALSIFIED, 1/3
seeds (2026-07-13).** 48.0→39.1→30.3mm, checkpoint-resumed per stage
(1000 iters/stage, full 4096-env population every stage, seeds
42/123/7). Stage-1 discovery check (48mm, instrumented eval): 0/8,
**8/8**, 0/8 — 1/3 seeds discover, exactly reproducing the bisect's own
48mm anchor (rung 2: 1/3, same seed). Stage-3 verdict (30.3mm,
instrumented eval): 0/8, **8/8**, 0/8 — same 1/3, below the >=2/3 bar.
All 9 training runs healthy (zero NaN, VF loss bounded). Mechanism
reading: unlike the primary's dilution failure, the transfer mechanism
itself works cleanly here — seed 123's discovered grasp carried
undegraded through both anneal stages (`position_error` ~0.11→0.099→
0.102, monotonic improvement, 8/8 sustained lift at both the 48mm and
final 30.3mm checks). The bottleneck is the 1/3 base discovery rate at
48mm itself; neither curriculum variant creates NEW discovery for
seeds that never find a grasp — they only propagate discovery when it
already happened. Both pre-authorized size-curriculum arms are now
falsified; per the spec's own next-step, shape itself (grasp-strategy/
reward changes) needs a new spec/research pass rather than further
object-scale curriculum variants on this asset. Seed 123's 8/8
30.3mm result is the project's first confirmed d20 lift+carry policy
at the real target size — a reusable baseline for that next
investigation. Full verdict + ops notes appended to
`docs/superpowers/specs/2026-07-13-size-curriculum-design.md`; task
report `.superpowers/sdd/task-staged-anneal-report.md`.

**d4 edge-grasp rung 0: seeded trials FALSIFIED at the implementation
layer, grasp-mechanism hypothesis untested (2026-07-13).** Spec
`docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`, full
trial data `.superpowers/sdd/task-d4-rung0-trials-report.md`. Tasks 0-1
(desk check + implementation — shape-general opposite-edge-pair grasp
geometry, `tasks/franka/antipodal_edge_grasp.py`, 17 unit tests) passed
review clean. Tasks 2-3 (5 seeded trials, seeds 42/123/7/1000/2026): 0/5
— every trial failed identically at `stage2_descend_d4` (tilted-axis
descent to grasp height never converged within budget, final residual
26.6-40.1mm vs a 5mm tolerance), so **no trial ever closed the gripper**
— zero grasp attempts, zero ejection/z-gain data, the die never
perturbed in any trial. This does not falsify the edge-grasp mechanism
itself (the spec's own falsification condition requires a *converged*
tilted approach before an ejection counts as evidence) — it's an
unresolved IK-reachability gap in the scripted controller. Diagnostic
signature points at implementation, not mechanism: 100% reproducible
non-convergence, a consistent ~13mm z-bias present already at the end
of stage 1 in all 5 trials regardless of which edge-pair/wrist-yaw was
selected, and orientation error growing during the failed stage-2
attempt in 3/5 trials despite starting stage 2 already converged.
Contact-force instrumentation (`d4_leftfinger_contact`/
`d4_rightfinger_contact` `ContactSensorCfg`s, filtered to `Die_d4`,
added this task per the spec's own flagged phi-regime-verification need)
is implemented and regression-clean but never exercised (no trial
reached closure). Non-d4 regression guard: code path confirmed
byte-identical (`git diff -w`); d20 smoke (seed 42, run twice,
byte-identical both times) came back FAIL but traced directly to a
pre-existing, seed-specific detector-vs-GT offset (8.4mm, exceeds the
squeeze-out margin) matching fragility already documented in
`kb/wiki/experiments/dice-pick-demo.md` before this task — not
attributed to the d4 work. **Recommendation flagged to Principal, not
unilaterally executed**: root-cause the stage2 IK non-convergence
(leading suspect: the tilted-approach waypoint/standoff math, or a
missing XY+Z refine fallback analogous to the non-d4 path's own) as a
scoped follow-up before treating rung 0 as falsified and climbing to
rung 1 (pad geometry) — rung 1 wouldn't address this failure mode
either, since it never reaches the point where pad geometry matters.

**d4 edge-grasp rung 0: FALSIFIED at reachability (2026-07-13 evening).**
The tilted opposite-edge grasp pose puts the lower jaw 3-15mm below the
table for every plausible opening/finger-geometry combination — PhysX
blocks it, DiffIK diverges fighting the contact (root-caused via exact
waypoint-arithmetic reconstruction + a failed finer-step-caps trial; the
geometry helper itself is nanometer-exact). Mechanism-level result: stock
Franka jaws cannot straddle a table-resting ~24mm tetrahedron along its
edge-pair axis. Ladder climbs to rung 1 (pad modification — V-groove /
compliant pad with the existing straight-down approach; Guo et al. 2017
grounding). New standing desk-check rule: grasp-geometry specs must check
swept jaw volume vs the support surface, not just contact friction cones.
Open ops item: d20-seed42 demo smoke now fails deterministically on clean
HEAD (detector-side, 8.4mm) despite passing 2026-07-11 — archaeology
pending. Full trail: spec's Rung-0 closure section.

**Cloud training pipeline re-verified end-to-end, zero preemptions
(2026-07-14/15).** Second full shakedown run, independent of attempt 3:
instance `rl-franka-shakedown` fell back to `us-west1-a`
(`us-central1-a/b/c` AND `us-east1-b/c/d` all
`ZONE_RESOURCE_POOL_EXHAUSTED` this time — worse stockout than
2026-07-13). ik-cube 4096-env training ran uninterrupted to 1500/1500
iterations in ~35min with **zero SPOT preemptions** (attempt 3 hit two)
— checkpoint-resume path unexercised this run. Total instance lifetime
54m47s. Completion verified directly from the downloaded tfevents file
(every scalar tag has exactly 1500 points, steps 0-1499), not from the
`model_1499.pt` checkpoint filename alone — the instance's tee'd stdout
log appearing to stop at iteration 1493 is a stdout-buffering artifact
at process exit, confirmed against the raw event data, not an early
stop. Artifacts:
`gs://rl-manipulation-hks-runs/cloud-shakedown/ik-cube/seed42/2026-07-15_01-52-15/`
(31 checkpoints + tfevents + manifest + params), git SHA `ab7c8ea6` at
ship time. New for budget-planning: real per-SKU GCP pricing pulled from
the Cloud Billing Catalog API shows the L4 GPU is a **separate SKU from
the `g2-standard-4` machine type, not bundled** — CPU+RAM $0.075/hr + L4
GPU $0.2862/hr (the dominant cost) = $0.361/hr combined, boot disk
$0.10/GiB-month; this run ≈$0.35 total. The GCP Billing console lags
real usage by hours (mid-run it showed only the ~$0.019 disk charge,
none of the ~$0.33 instance/GPU charge) and this project has no
BigQuery billing export configured, so duration × published-SKU-rate is
the only cost-estimation path available. Recipe updated:
`docs/cloud/franka-cloud-shakedown.md`.

**Standard-vs-jumbo d20 size correction (2026-07-15).** Web research
(user-confirmed) established that a real *standard* commercial d20 is
~20-22mm across, not 30.3mm — 30.3mm is itself a real, commonly-sold
**"jumbo"** d20 size (e.g. Twenty Sided Store's own "Jumbo Dice 30mm
D20" listing), not a mistake or edge case. Every rung in
`tasks/franka/dice_lift_joint_env_cfg.py` (Heavy/Big/Mid/Mixed) had
treated 30.3mm — the baked asset's default spawn size — as "the real
target size" to anneal a curriculum down toward. **This does not change
any existing asset-bisect/size-curriculum verdict** — the 30.3mm/48.0mm/
etc. results stand exactly as originally reported; it only corrects
what "the real/final target size" means for *future* d20 size-related
work. Added a new forward-facing class, `FrankaDieLiftJointStandardEnvCfg`
(+ `_PLAY` variant), inheriting from `FrankaDieLiftJointHeavyEnvCfg` (mass
pinned 0.216kg, unchanged) with `scale=(0.000727, 0.000727, 0.000727)` —
derived by fitting this file's own four existing rung constants
(0.001585/48.0mm, 0.001440/43.6mm, 0.001291/39.1mm, 0.001146/34.7mm) to
an average scale-per-mm ratio of 3.302305e-5, then 22mm × 3.302305e-5 =
0.000727. Live-verified via new diagnostic
`scripts/_diag_d20_standard_scale_check.py` (same no-physics headless-
SimulationApp bounding-box-read pattern as `_diag_die_scale_check.py`/
`_diag_dexcube_scale_check.py`): measured bbox at this scale is
**21.993mm** (delta -0.007mm from the 22.0mm target), well inside the
0.3mm tolerance — no adjustment needed. This class is not yet used by
any training run; it's a target for a future d20 grasp-strategy spec
(shape itself, per the size-curriculum verdict above, is the next thing
needing a direct attack).

**d4 edge-grasp rung 1 (V-notch fingertip fixture): UNTESTED — 0/5 seeded
trials reached the grasp mechanism, blocked entirely at perception
(2026-07-15).** Spec
`docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md`,
research `.superpowers/sdd/research-d4-rung1-pad-geometry.md` +
independent review `.superpowers/sdd/review-d4-rung1-pad-geometry.md`
(one citation-attribution error caught and corrected: the V-groove
precedent is Habibi/Sutera/Guastella/Muscato, *Robotics* 14(7):87, 2025,
not "Zhang et al. 2026" as the research pass first wrote it — the paper
and its numbers were real, only the author label was fabricated, per
this project's known citation-detail-fabrication risk). Built a rigid
110° V-notch fixture (compound convex-wall collider, fixed-jointed onto
both Franka fingertips unconditionally, not a d4-only branch), unifying
`dice_pick_demo.py`'s straight-down path across all 5 dice (rung 0's
tilted-axis d4 branch removed entirely). Task-review caught one
Critical, sim-crashing bug before any cloud run (missing
`activate_contact_sensors=True` on the fixture's spawn config — fixed
and re-verified against isaaclab source). On GCP cloud (SPOT
g2-standard-4 + L4, ~$1-2 total across 3 preemptions/1 zone stockout,
recovered via disk-snapshot cross-zone restore): all 5 d4 seeded trials
(42/123/7/1000/2026) failed identically at the detector step — zero
`d4`-class detections returned in any trial, `select_target_detection`
raising before any grasp attempt, so **the notch mechanism itself
remains completely untested, not falsified** (same category as rung
0's own outcome, different upstream cause). `sim.reset()` succeeded
cleanly in every trial — Task 1's fixed Critical bug holds under real
physics. Non-d4 regression guard: **3/4 clean PASS** (d8/d10/d12, all
matching/exceeding the kb baseline z-gain range, zero cross-die drift —
the unconditional-fixture North Star call does not regress the working
die types), **1/4 attributable FAIL** (d20, byte-identical reproduction
of `kb/wiki/experiments/dice-pick-demo.md`'s already-closed
seed-42 RTX-nondeterminism finding, predating this rung). **New finding
distinct from rung 1's own hypothesis**: 5/5 identical-shaped d4
detection misses across 5 different seeds/scene layouts is a more
systematic-looking signal than the kb's previously-documented
occasional per-seed offset noise — the one near-hit (seed 123) was
low-confidence (0.27-0.36) and displaced by a colocated higher-confidence
`d10` candidate. Reads as "d4 is a weak/marginal detection class for
this detector," not investigated further (out of this task's scope).
Full data: `.superpowers/sdd/task-2-report.md`. Open decision for
Principal: scope a ground-truth XY-bypass path to test the grasp
mechanism in isolation from perception (extending rung 0's own
GT-for-orientation isolation precedent to position), or treat the d4
detection weakness as its own prerequisite research question.

**d4 rung 1 (V-notch fixture): both parallel follow-ups complete —
mechanism FALSIFIED, detector weakness diagnosed (2026-07-15).** User
directed both open threads pursued in parallel.

*Detector investigation* (`.superpowers/sdd/research-d4-detector-weakness.md`):
d4 is NOT a weak class for this checkpoint (0.992-1.000 mAP50, among the
best), not meaningfully underrepresented in training data, not the
smallest die in this exact scene (larger than d8/d10, which detect fine
in the same frames), and confirmed fully visible/unoccluded in all 5
failing renders via direct 3D-to-pixel reprojection. Leading hypothesis
(explicitly flagged as untested extrapolation): under this scene's
degraded, near-textureless rendering, the detector may fall back on
residual 3D shape cues (facet edges, apex highlights, shading gradients)
to classify the other 4 dice, and d4's flat-face rest pose is the one
silhouette with none of those cues — a shape/silhouette-flatness
confound, sharper than but related to the kb's existing apparent-size
confound. Needs a controlled ablation to confirm, not attempted here.

*Ground-truth XY-bypass* (`docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md`'s
addendum): first bypass build had a real bug (`select_target_detection`
still raised on a total detection miss before the bypass branch ran, so
it never actually routed around d4's failure mode) - found live on a
cloud run, fixed (`9bc8820`, independently re-reviewed with an explicit
end-to-end trace), reran. **3/5 seeds genuinely reached the grasp
mechanism this time - 0/3 met the primary criterion.** Closure-window
lateral ejection 172.0mm/18.8mm/57.7mm (threshold ≤5mm), z-gain ~zero
in all 3, zero contact force in all 3, confirmed visually (frame
extraction from seed 42: gripper fully closed at the die's original
position, die sitting undisturbed several cm away - not a subtle
grasp-then-eject, the notch swept the die aside without ever engaging
it). **FALSIFIED as a fix for the d4 at the symmetric 110°
design's current dimensions/placement.** 2/5 seeds (123, 2026) hit an
unrelated, reproducible CUDA crash - root-caused to a hardcoded
contact-sensor buffer (`maxContactDataCount=4`, added in the rung-0
task) overflowing under denser real contact, corrupting a downstream
indexing op; flagged, not fixed. Cumulative rung-1 cloud spend across
all attempts: ~$2.3-3.3, well under the $15 cap. Open questions
(not resolved): whether the `xformOp` warning flagged earlier actually
indicates a real dynamic-collision-placement defect (only the fixture's
static position was ever verified, not its behavior under real
closure load); whether the contact-buffer crash correlates with denser
genuine engagement; whether 110°/~10mm/~4mm is a tuning problem or the
symmetric-notch-on-flat-jaws strategy itself is wrong. Full data:
`.superpowers/sdd/task-2-report.md`.

**Unified multi-die specialist-distillation experiment: underway (started
2026-07-16).** Spec
`docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`,
research `.superpowers/sdd/research-multi-die-unified-policy.md`, plan
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`.
Goal: a single RL policy that grasps a commanded die among {d8, d10, d12,
d20} (d4 deferred), via per-shape specialists distilled into one policy
(UniDexGrasp++'s GiGSL pattern), gated on the research's own finding that
uncurriculated multi-object clutter can collapse RL discovery
(DexSinGrasp, arXiv:2504.04516) — distractors/target-selection explicitly
deferred to a follow-on experiment. Tasks 0-1 (bake d8/d10/d12 physics
assets at real researched standard sizes — d8/d10=16.0mm, d12=18.0mm face-
to-face, weaker sourcing tier than d20's own single-citation correction;
add shape-class one-hot + Wadell-sphericity geometry-descriptor
observation terms) both complete, reviewed clean.

**Task 2 (train d8/d10/d12 specialists) result: 0/9 discovery (2026-07-16),
worse than d20's own 1/3-at-48mm asset-bisect baseline.** Cloud run
(~$1.97, 5.15hr across 1 manual-stop + 1 genuine SPOT-preemption
interruption, both recovered via checkpoint-resume with zero real
progress lost). Principal independently re-derived the raw per-step
height trajectories (not just the summary JSON) and found the reported
"~0.05m max height gain" figures are a measurement artifact — the eval
script's `max()` over a multi-episode recording window catches a spurious
height spike at an episode-reset boundary, not any policy-driven lift.
After correcting for this, the object is completely motionless
(byte-identical across all 8 parallel envs, at every checked timestep) for
the entire evaluation in every seed inspected — a *stronger* null result
than the raw report suggested (zero engagement, not "attempted but
unsustained"), though `franka_checkpoint_review.py`'s max-height-over-
window computation itself should be fixed before trusting its numbers on
a variant whose video spans a reset boundary. **Open, unresolved
confound (explicit user decision: proceed to Task 3 first, revisit
rather than block on this now):** d8/d10/d12 were trained directly at
their real ~16-18mm sizes, never at the 48mm cube-parity anchor
asset-bisect used to isolate shape from scale — so this 0/9 result
cannot yet be attributed to shape difficulty vs. these three objects
simply being too small at this Franka gripper's absolute scale,
independent of shape. A controlled 48mm-parity rerun for d8/d10/d12
(mirroring asset-bisect's own methodology) would resolve this and is the
natural next step if Task 3/4 don't resolve it first.

**Task 3 (d20 size-DR + geometry-feature retry) result: 0/120 discovery,
independently reverified (2026-07-18) — exactly reproduces the diluted-
population floor, does not clear the asset-bisect's undiluted 48mm
baseline.** Plan
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`
Task 3: `FrankaDieLiftJointRandomSizeEnvCfg`, 3 seeds (42/123/7) trained
against a per-env size population spanning {22.0, 28.5, 35.0, 41.5,
48.0}mm (`MultiAssetSpawnerCfg(random_choice=True)`, confirmed via direct
`isaaclab.sim.spawners` source read at Task 3 Step 1 to assign one size
per env once at scene-spawn time — the identical assignment mechanism as
the already-falsified `FrankaDieLiftJointMixedEnvCfg`, differing only in
random vs. round-robin pattern), with Task 1's geometry-descriptor
observation conditioning added as this retry's actual new variable.
Training/eval artifacts already existed in GCS
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-artifacts/joint-die-random-size/`,
all 15 seed×scale combinations) but the verdict was never independently
verified or written up until now. Downloaded and recomputed genuine
per-step height trajectories for all 15 (seed, scale) pairs — all 3
seeds × all 5 scales, 120 total eval envs (8 envs/eval) — restricting
analysis to the first-episode window to avoid the multi-episode
reset-boundary artifact `franka_checkpoint_review.py` was fixed forward
for at Task 3.5 (these Task 3 JSONs predate that fix and were never
regenerated). Result unchanged from the unfixed summary JSONs: **0/120
sustained lifts, at every scale, every seed:**

| scale (target mm)  | seed 42 | seed 123 | seed 7 |
|---------------------|---------|----------|--------|
| 22.0mm (0.000727)   | 0/8     | 0/8      | 0/8    |
| 28.5mm (0.000941)   | 0/8     | 0/8      | 0/8    |
| 35.0mm (0.001156)   | 0/8     | 0/8      | 0/8    |
| 41.5mm (0.001370)   | 0/8     | 0/8      | 0/8    |
| 48.0mm (0.001585)   | 0/8     | 0/8      | 0/8    |

No env's height gain ever sustains 25 consecutive steps above the 0.04m
threshold, even transiently, at any scale/seed — the reset-boundary
artifact never flips a verdict here either way. Separately identified a
second, smaller measurement quirk in these particular JSONs (distinct
from the reset-boundary one): every env's reported ~0.032-0.048m "max
height gain" is fully explained by the object's spawn-to-table free-fall
not finishing within the script's 10-step settle window (this env's
fall+settle actually takes ~18-20 steps) — after settling, every
inspected trajectory goes flat to within ~4e-8m for the remaining ~230
steps of the episode, i.e. genuinely motionless, not merely
"unsustained." Eval video reviewed directly (not just existence-checked):
extracted and inspected frames across the full episode (steps 0/50/100/
150/200/249/300/450) from the 48mm seed-42 video, plus mid-episode frames
from seed-123/seed-7 at 48mm. Seed 42's arm never descends toward the die
at all — it stays folded/elevated over the table for the entire episode
while the die sits undisturbed at its spawn position in every sampled
frame; seed 123/seed 7 show visibly different arm poses (reaching closer
to the table surface at the same timestamps) despite an identical 0/8
outcome. Comparison against existing baselines: the asset-bisect ladder's
own single-size, undiluted 48mm d20 population scored 1/3
(`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`); the
original size-curriculum's diluted 5-size population
(`FrankaDieLiftJointMixedEnvCfg`) scored 0/3 (0/8, 0/8, 0/8), attributed
there to ~5x dilution of the 48mm sub-population. This retry's 48mm
column reproduces that diluted floor exactly (0/3, 0/8/0/8/0/8) —
despite adding the geometry-descriptor conditioning Task 1 built
specifically to help the policy differentiate/adapt across sizes.
**Mechanism reading:** Task 3's population is per-env-fixed-at-spawn
across the same 5 sizes as the falsified `MixedEnvCfg` (confirmed
mechanism, not assumed), so its 48mm arm is itself a diluted (~1/5)
sub-population, not a single undiluted 48mm run — structurally identical
to the original size-curriculum's own 0/3 result. Reproducing that exact
floor is *consistent with* population dilution being (at least) a real
confound, and does not contradict that reading. But because Task 3 never
paired an actual single-size, undiluted-48mm arm with the geometry-
descriptor conditioning, this task's own data cannot distinguish
"dilution alone explains the null" from "shape/discoverability remains a
barrier even with geometry conditioning, independent of dilution" — both
hypotheses predict exactly the 0/3-at-diluted-48mm result observed here.
Task 3.5's own undiluted-48mm design (built to close exactly this gap)
was scoped to d8/d10/d12 only and was never re-run for d20 itself, so
this specific ambiguity remains technically open for the d20 case, even
though Task 3.5's own plan-doc framing treats the dilution explanation as
more settled than this task's data alone supports. Per the plan's own
Task 3 Step 5 instruction, this is reported as a clean numeric result
only — whether/how Task 4 (distillation) proceeds given zero working
specialists across d8/d10/d12/d20 to date is a controller decision, not
made here.

**Task 3.5 (48mm-parity check for d8/d10/d12) result: full 3x3 grid
complete (2026-07-19), one genuine partial positive found — d12 at
seed123.** d8-big's 3 seeds were trained/evaluated by a prior
desktop-dispatch agent in this same batch; d10-big and d12-big's 6 seeds
(42/123/7 each) were trained/evaluated by this task on GCP cloud (SPOT
g2-standard-4+L4, `docs/cloud/franka-cloud-shakedown.md` recipe), all at
1500 iterations, single-size undiluted 48mm populations, mass pinned
0.216kg, per-shape freshly-derived 48mm-targeting scale (not d20's
0.001585 constant, confirmed shape-specific per Task 0's own native-bbox
measurements). Full per-shape, per-seed discovery rate (envs with
sustained lift / 8 envs per seed):

| shape (48mm) | seed 42 | seed 123 | seed 7 | seeds-with-discovery |
|--------------|---------|----------|--------|----------------------|
| d8-big       | 0/8     | 0/8      | 0/8    | 0/3                  |
| d10-big      | 0/8     | 0/8      | 0/8    | 0/3                  |
| d12-big      | 0/8     | ~~4/8~~ **8/8** | 0/8    | 1/3          |

**d12-big seed123 corrected from 4/8 to 8/8 (2026-07-19 re-audit, see
"Task 3.5 re-audit" entry below)** — the number as originally measured in
this task (below) was itself a measurement artifact of the
settle-detection bug fixed in the d20-big-geom gate task; all other 8
cells in this grid were re-audited against the fixed method and confirmed
unchanged. The rest of this Task 3.5 entry is left as originally written
(the historical record of what was measured/reasoned at the time,
including the now-superseded "4/8, wedged/stuck" reading below) — see the
dedicated re-audit entry for the corrected numbers, root cause, and video
confirmation.

Compared against the asset-bisect ladder's own undiluted-48mm baselines
(`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`): cube 3/3
seeds (full 8/8 each), d20 1/3 seeds (seed 123, full 8/8). d8/d10 remain
completely null at 48mm parity — shape itself (not population dilution,
not absolute scale) is confirmed as a real barrier for these two shapes,
consistent with Task 2's original ~16-18mm finding. d12 shows a genuine
partial recovery: one seed (123, the same seed that discovered grasp in
the original d20 bisect) with 4 of 8 envs achieving sustained lift — a
real but *weaker* echo of d20's own 1/3-seed pattern (d20's lucky seed
got full 8/8 within-seed discovery; d12's lucky seed got half).

**d12-big seed123's 4 lifted envs (indices 1,5,6,7), reported height
gains (post-fix measurement): env1 +0.192m (max_z 0.213m), env5 +0.183m
(max_z 0.204m), env6 +0.188m (max_z 0.209m), env7 +0.065m (max_z
0.086m), each sustained for the full post-settle window observed (53/53
steps ≥ the 25-step/0.04m threshold).** Independently verified this is a
real result, not a third occurrence of the reset-boundary/settle-window
artifacts already fixed twice in this experiment (commit 977a748):
re-implemented the settle-detection and gain logic from scratch against
the raw `.npy` (not reusing `franka_checkpoint_review.py`'s own code) and
found the *shape* of the trajectory is a smooth, continuous rise starting
~step 40, reaching a stable plateau by ~step 90-115 and holding there
(±3-13mm jitter, no violent single-step jumps — max single-step delta
0.010m, inconsistent with a contact-explosion/launch artifact) through
the rest of the 249-step analysis window. The plateau height (env1-6:
~0.20-0.23m absolute) sits almost exactly inside `lift_env_cfg.py`'s own
goal-command z-range (`pos_z=(0.25, 0.5)`, `ObservationsCfg`/`CommandsCfg`
in `tasks/franka/lift_env_cfg.py:194`) — physically consistent with a
genuine grasp-lift-carry-toward-goal, not a glitch. One measurement
caveat found and NOT yet fixed (flagged to `BACKLOG.md`, did not block
this verdict): `_detect_settle_step`'s 5e-5m/15-step tolerance is tuned
for a motionless table-rested object and is too tight to ever match a
*held* object's natural grasp-contact jitter, so all 4 lifted envs report
`settle_step: -1` and fall back to the pre-fix free-fall-window-min
baseline (~0.021m) — this fallback happened to be numerically close to
the true table-rest height here (confirmed against the other 4 envs'
own directly-detected resting_z, 0.0175-0.0209m), so it did not change
the conclusion, but is not guaranteed to be harmless in general. Also
notable: the other 4 envs (0,2,3,4) are NOT simple table-rest nulls —
env0 is genuinely motionless at the table (0.0175m), but envs 2/3/4 sit
statically at unusually elevated, non-table heights (0.109m, 0.112m,
0.242m) with zero further gain — consistent with the die becoming
wedged/stuck (on the gripper or robot geometry) rather than either
resting on the table or being actively lifted; not investigated further
since it doesn't change the seed's own qualifying discovery count.
**Video-verification limitation, disclosed rather than papered over:**
`franka_checkpoint_review.py`'s camera is fixed on env_0 (the one
`FrankaLiftEnvCfg`/`ViewerCfg` docstring-documented framing, "so env_0's
whole arm...is in frame"); in this specific run env_0 was one of the
*non*-lifting envs, so the lifted envs (1,5,6,7) could not be directly
visually confirmed via video this time — the positive verdict rests on
the raw physics `root_pos_w` trajectory shape/timescale reasoning above,
not a video observation. Watched both this video (confirms env_0's own
null result: arm stays folded near the table the whole episode, matching
its 0/8-contributing data) and `joint-die-d10-big` seed42's video
(confirms genuine non-engagement: arm never descends toward the die,
same "folded/elevated over table" pattern as Task 3's own d20 seed42
null).

**Bug found and fixed during this task (both re-run to confirm, per this
repo's bug-handling discipline):**
1. **d8-big seed42/seed123's synced eval artifacts predated the
   episode-boundary/settle-window fix (commit 977a748), contradicting
   this task's own dispatch brief.** Verified directly via GCS object
   creation timestamps (seed42 json: 2026-07-19T02:27:47Z, seed123:
   02:40:52Z) vs. the fix commit's authored time (02:46:03Z, `-0400` ->
   UTC) — both predate the fix; only seed7's json (03:01:42Z) postdated
   it. Re-ran eval only (no retraining needed — checkpoints already
   existed in GCS) against the current fixed script for both seeds and
   re-synced; both reconfirmed 0/8 under the corrected measurement (no
   verdict changed, but the artifacts now match the schema/rigor of the
   rest of the grid).
2. **A genuine new cloud-infra bug: a SPOT preemption can truncate a
   checkpoint file mid-write, leaving a 0-byte `.pt` on disk that a naive
   "resume from highest iteration number" strategy would pick and fail
   to load (`EOFError: Ran out of input`).** Hit once (d10-big seed42's
   `model_1100.pt` truncated to 0 bytes by a preemption that landed
   exactly at a `save_interval=50` checkpoint write). Fixed the resume
   orchestration to skip any checkpoint under 100KB (a real checkpoint is
   ~1.27MB) before selecting a resume candidate, falling back to the next
   most-recent valid one (`model_1050.pt`); re-ran and confirmed the fix
   resumes correctly. Not yet folded into a shared/reusable script (this
   task's own one-off orchestration shell script) — worth carrying
   forward into `docs/cloud/dispatch-checklist.md`'s known-infra-gaps
   list if cloud SPOT training recurs (done, see that doc's own update).
3. **Operational, not a bug: 3 genuine SPOT preemptions in ~3 hours**
   (independently confirmed via `gcloud compute operations list`
   `compute.instances.preempted` system events each time, not
   manual/controller stops), a much higher rate than this project's
   prior cloud-shakedown history (2 preemptions per run at most). After
   the 3rd, switched the remaining 2 jobs (d12-big seed123/seed7) to
   **on-demand** provisioning (still the same instance type/zone-search
   logic, just `--provisioning-model` omitted) to stop losing wall-clock
   to repeated snapshot-recover-resume cycles — a pragmatic operational
   call, not a change to the plan's own methodology, well within the
   cost cap given on-demand's ~2x SPOT rate on a small remaining job
   count.

**Cost: ~$1.9-2.0 total for this task** (6 training runs + 8 eval runs +
1 re-eval pair), verified via real Cloud Billing Catalog API SKU rates
(not the doc's older estimate) — SPOT $0.361216/hr, on-demand
$0.706832/hr for g2-standard-4+1x L4 in `us-central1`/`us-east1`.
Instance-hours: 3 SPOT segments (task35 0.64hr, task35b 1.75hr, task35c
0.37hr = 2.76hr SPOT) + 1 on-demand segment (task35d 1.11hr) = compute
≈$1.78, plus small boot-disk/snapshot storage overhead (≈$0.1-0.2,
conservative upper bound). Well under the $12.04 remaining allotment for
this task plus Tasks 5/6 (cumulative across Tasks 2/3/3.5 now
≈$2.96+$2.0≈$4.96 of the original $15 cap). Full teardown verified via
`scripts/check_cloud_state.sh`: zero instances/disks/snapshots remain.

**Per this task's own Task 3.5 Step 7 instruction, this is reported as a
clean numeric result only — whether/how Task 4 (distillation) proceeds
given this grid (2 of 4 candidate shapes still fully null at 48mm parity,
1 partial, d20 itself still gated on Task 3's own open ambiguity) is a
controller decision, not made here.**

## Task 3.5 re-audit against the fixed settle-detection method (2026-07-19): d12-big seed123 corrected 4/8 → 8/8, d8/d10 confirmed unchanged

The d20-big-geom gate task above found and fixed a real bug in
`franka_checkpoint_review.py`'s settle-detection heuristic (the
`977a748` flatness-window scan, replaced with the current MIN-over-
early-window approach — see that task's own entry for the full
mechanism) that predates all of Task 3.5's own d8-big/d10-big/d12-big
runs. This left an explicit open risk, flagged in `BACKLOG.md`'s
"settle-detection flatness-window heuristic may have undercounted true
positives" entry: Task 3.5's reported grid was measured with the buggy
version and never re-checked. This task closes that gap.

**Method: pure offline reanalysis, no new GPU rollout.** Confirmed by
reading `franka_checkpoint_review.py`'s current source first (per this
task's own dispatch instruction) that everything downstream of
`height_history` (the raw per-step object-z array already saved to each
run's `heights_*.npy`) is plain numpy/torch math with no dependency on
Isaac Sim or the live env object — `episode_length_steps` (the one other
input the fixed logic needs) is also already recorded in each run's
existing summary JSON, itself unaffected by the settle-detection bug.
Downloaded all 9 already-synced `heights_*.npy`/`.json` pairs from
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-
artifacts/joint-die-{d8,d10,d12}-big/seed{42,123,7}/` and re-ran a
line-for-line numpy port of the current fixed logic
(`EARLY_SETTLE_START=10`, `EARLY_SETTLE_END=45`, MIN-over-window
resting_z, `SUSTAINED_LIFT_STEPS=25` @ `LIFT_HEIGHT_THRESHOLD_M=0.04`)
directly against them — committed as `scripts/_diag_settle_reaudit_task35.py`
(plain Python + numpy, no `isaaclab`/`torch` dependency, runs on the Pi
directly) for reproducibility/future re-audits.

**Result: 8 of 9 cells unchanged, all with large safety margin (max
height gain 0.003-0.009m against the 0.04m sustained-lift threshold —
not close calls).** d8-big and d10-big are confirmed 0/8 in all 3 seeds
each under the fixed method — matches Task 2's independent raw-data
confirmation that these shapes are genuinely motionless at this scale,
consistent with there being no "wrong later plateau" for the settle-bug
to lock onto when the object never moves. d12-big seed42/seed7 are also
confirmed 0/8 (max gains 0.003-0.007m, noise-floor level).

**d12-big seed123 changes from 4/8 to 8/8 — a real, substantive
correction, not a rounding/edge-case artifact.** Root cause, read
directly off the original (pre-fix) JSON's per-env `settle_step`/
`resting_z` fields: the old flatness scan mistook 3 of the 4
previously-"non-lifting" envs' own held-elevated plateau for their
*resting* state (env 2: `resting_z_m=0.1088` — its actual held height,
not the true ~0.0175m table-rest; env 3: `0.1116`; env 4: `0.2415`), so
their real lift-and-hold registered as "already resting, zero further
gain." The 4th ("env 0"), had an approximately-correct `resting_z`
(0.0175, because by coincidence this env's die had already fallen back
to the table by the time the scan locked onto its plateau) but the old
code's single *shared* `post_settle_start` across all 8 envs was
`max(settle_step across all envs)=196` — driven by envs 2/3/4's own
bad, late detections — which excluded env 0's entire rise-and-fall event
(steps ~40-115) from the analysis window entirely.

Re-derived and independently confirmed all 4 newly-corrected envs are
real physical lifts, not measurement noise: all 8 envs (not just the
originally-credited 1/5/6/7) show the *same* smooth, monotonic rise
starting within a 5-step window of each other (steps 38-43), reaching
their plateau by steps 54-71, with comparable max single-step deltas
(0.0135-0.0205m, no teleports) — i.e. one consistent policy behavior
firing near-identically across all 8 parallel envs of this seed, not 4
real lifts plus 4 unrelated artifacts. 7 of the 8 (envs 1-7) hold their
elevated position through to the end of the episode, matching the
already-confirmed pattern for envs 1/5/6/7. **Env 0 is qualitatively
different and worth flagging on its own: a genuine lift-then-drop, not
a lift-and-hold** — it rises smoothly to ~0.111m by step ~68, holds
(satisfying the 25-consecutive-step sustained-lift bar with margin,
37 steps), then descends smoothly (no teleport) back down to the true
table-rest height (0.0175m) by ~step 115 and stays there for the rest
of the episode. Directly video-confirmed (env_0 is the one env this
script's fixed camera actually shows): extracted frames at steps
10/40/50/65/82/95/110/150/249 from
`franka_checkpoint_review_joint-die-d12-big_model_1499-step-0.mp4` —
step 10 shows the die resting on the table next to the closed gripper;
step 65 (zoomed crop) shows a small white sphere elevated near/above the
gripper, consistent with the ~0.111m reading at that step; step 95 shows
the die back down near table level, consistent with the trajectory's
smooth descent back to ~0.021m by that point. This is a real, if
less-complete, positive — it meets the project's own operational
"sustained lift" bar (`mdp.object_is_lifted` threshold held ≥0.5s) even
though it doesn't carry to end-of-episode like the other 7.

**Corrected grid:**

| shape (48mm) | seed 42 | seed 123 | seed 7 | seeds-with-discovery |
|--------------|---------|----------|--------|----------------------|
| d8-big       | 0/8     | 0/8      | 0/8    | 0/3                  |
| d10-big      | 0/8     | 0/8      | 0/8    | 0/3                  |
| d12-big      | 0/8     | **8/8**  | 0/8    | 1/3                  |

**Decision-relevance, flagged explicitly rather than silently folded in:
this correction does NOT change which shapes show real discovery** — d8
and d10 remain fully, robustly null (0/3 seeds each, confirmed with wide
safety margin, not close calls) across every size regime tested to date
(Task 2's real ~16-18mm size, Task 3.5's 48mm parity); d12 remains the
only one of the two "deferred" candidates with any discovery, still in
exactly 1 of its 3 seeds. **`BACKLOG.md`'s "Task 4 scope decision"
(narrow to d12+d20, defer d8/d10) is unaffected at the shape-inclusion
level** — nothing here reopens the case for including d8/d10 in Task 4,
and nothing here removes d12 from consideration. What *does* change is
the characterization of d12's own specialist quality: it was reported as
a "weaker echo" of d20's pattern (half its lucky seed's envs vs. d20's
full 8/8); it is now a *matching* echo — d12's lucky seed (123) gets the
same full 8/8 within-seed completeness as d20's own lucky seeds, the
same clean pattern this project's asset-bisect/d20-big-geom lineage has
seen before. Since `BACKLOG.md` already earmarks this exact checkpoint
(`joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`) as one
of Task 4's two frozen specialist teachers, this is relevant context for
that task even though it doesn't change which shapes are in scope —
flagged here for the controller, not acted on unilaterally.

No new bugs found in the fixed settle-detection method itself or in this
task's own offline re-derivation script during this re-audit (cross-
checked step-for-step against the current `franka_checkpoint_review.py`
source, and the 8 unchanged cells all show wide safety margins rather
than borderline flips, which would be the more likely symptom of a
subtle bug in the re-derivation itself).

## d20-big-geom gate task: undiluted-48mm d20 retrain closes Task 3's
dilution ambiguity — result STRONGER than the falsifiable expectation
(2026-07-19)

Closes the open ambiguity flagged in Task 3's "0/120" entry above and in
`BACKLOG.md`'s "Task 4 scope decision" entry (2026-07-19): Task 3's own
d20 size-DR retry mixed 5 sizes at once (`random_choice=True`), diluting
the 48mm sub-population ~5x, so it never actually tested "d20 at a single
undiluted 48mm population with Task 1's geometry-descriptor conditioning."
This task does exactly that, mirroring Task 3.5's own d8/d10/d12-big
design, on GCP cloud (SPOT g2-standard-4+L4, `us-central1-a`, zero
preemptions this run), 3 seeds (42/123/7), 1500 iterations each,
`num_envs=4096` training / 8 eval envs, headless per the standing cloud
exception.

**No new env cfg class or `--variant` was added.** Direct source read
(`tasks/franka/dice_lift_joint_env_cfg.py`, `tasks/franka/lift_env_cfg.py`)
confirmed the existing `FrankaDieLiftJointBigEnvCfg`/`--variant
joint-die-big` (the asset-bisect rung-2 class, already wired into
`train_franka.py`/`franka_checkpoint_review.py`/`sync_run_to_gcs.py`)
already produces exactly the target config: Task 1's
`shape_class`/`geometry_descriptor` `ObsTerm`s were added unconditionally
to the shared base `ObservationsCfg.PolicyCfg` by commit `ec32bb0`
(2026-07-16), and `die_shape_class = "d20"` is set explicitly in
`FrankaDieLiftJointEnvCfg.__post_init__` (inherited unchanged through
`FrankaDieLiftJointHeavyEnvCfg` and `FrankaDieLiftJointBigEnvCfg`, neither
of which override it). The BACKLOG entry's caveat was about the old
*checkpoint* (trained 2026-07-12, before Task 1's code existed) predating
the schema, not about the env cfg *class* itself lacking it — retraining
today with the unchanged `joint-die-big` variant was sufficient to produce
a schema-compatible checkpoint. Avoids a functionally-duplicate
`FrankaDieLiftJointD20BigGeomEnvCfg` class that would have been
byte-for-byte behaviorally identical to the existing one.

**Per-seed discovery grid (envs with sustained lift / 8 envs per seed),
independently re-derived from raw `.npy`, not the summary JSON alone:**

| seed | sustained-lift envs |
|------|---------------------|
| 42   | 0/8                 |
| 123  | 8/8                 |
| 7    | 8/8                 |

**2/3 seeds fully discovering (8/8 within-seed), not the ~1/3 (likely
seed123 only) the task's own falsifiable expectation predicted going in.**
Reported exactly as observed, per this task's own explicit instruction not
to adjust the framing to match the a priori expectation. This is a
materially STRONGER result than the asset-bisect ladder's original
d20-at-48mm baseline (1/3 seeds, seed123, full 8/8) and than Task 3.5's
own d12-big echo as reported at the time (1/3 seeds, seed123, 4/8
partial — corrected to 8/8 by the "Task 3.5 re-audit" entry above; this
comparison predates that correction and is left as originally written)
— supports "population
dilution was Task 3's real confound" more decisively than the falsified
weaker form of that hypothesis: not only does undiluted-48mm d20 recover
discovery, it recovers it in *more* seeds and at *higher* per-seed
completeness than the pre-Task-1 baseline it's being compared against,
consistent with (though not strong enough evidence alone to conclude)
Task 1's geometry-descriptor conditioning itself helping rather than
merely being neutral.

**A genuine, serious measurement bug was found and fixed during this
task's own verification step** (per this repo's bug-handling discipline
and verification standard — raw-trajectory re-derivation is not optional
for positive results). The first eval pass (using
`franka_checkpoint_review.py` unchanged from `977a748`) reported seed123 at
only 1/8 and seed7 at 0/8 — direct inspection of the raw per-step `.npy`
found this was **wrong**: every env in both seeds showed a clean, smooth,
continuous spawn(0.052m)→table-rest(0.019-0.021m)→lift-to-goal-plateau
(0.26-0.46m, inside the 0.25-0.5m goal command z-range) trajectory, held
stable for the rest of the episode — a textbook successful lift+carry,
visually confirmed via the eval video too (arm reaches down to the table
by ~step 20, is fully extended upward and holding by ~step 90-150,
matching the height data). The bug: `_detect_settle_step`'s
flatness-window heuristic (a forward scan for the first 15-consecutive-
step window under a fixed range tolerance) is fundamentally unsuited to
this env cfg's fast, decisive grasps — the true table-rest phase is too
brief/noisy to reliably satisfy any single global tolerance (measured
window-range floor during genuine rest: ~0.0010-0.0024m across 24
env/seed combinations, both well above the original 5e-5m tolerance AND,
for some envs, above a first-attempt loosened 2e-3m), so the scan silently
walked past the true rest phase and locked onto a much later, fully-static
held plateau instead — with no warning printed, since a window *was*
found, just the wrong one. **Fixed by replacing the flatness-window
heuristic entirely** with a simpler, more robust, physically-grounded MIN
over a fixed early window (steps 10-45) — a grasp-driven ascent only moves
the object *up* from its true rest height, so MIN is correct regardless of
where within the window the low point falls. Verified directly: this
reproduces a clean, consistent ~0.019m resting_z for every genuinely-
resting env across all 3 seeds (vs. the flatness approach's spurious
0.26-0.46m readings for the same envs). See
`scripts/franka_checkpoint_review.py`'s own updated `EARLY_SETTLE_START`/
`EARLY_SETTLE_END` comment for the full derivation. **Open follow-up,
flagged to `BACKLOG.md`:** this same flatness-window mechanism (in one of
its two forms) was used, unquestioned, for Task 3.5's own already-reported
d8-big/d10-big/d12-big grid — those numbers were not re-audited here (out
of this task's scope) but may also undercount true positives, since the
old approach was never reliable for any env cfg checked so far, not just
this one. **Closed 2026-07-19 by the "Task 3.5 re-audit" entry above:**
d12-big seed123 was indeed undercounted (4/8 → corrected to 8/8);
d8-big/d10-big and d12-big's other 2 seeds were confirmed unchanged with
wide safety margin.

**Checkpoints for Task 4** (both fully valid, either usable as the frozen
d20 specialist teacher — seed123 kept as the nominal default per this
project's own recurring "seed123 is the lucky seed" convention, seed7 as
an equally-valid alternate given identical 8/8):
- `gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-big/seed123/2026-07-19_12-46-42/model_1499.pt`
- `gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-big/seed7/2026-07-19_13-17-02/model_1499.pt`

Eval artifacts (video/heights json/npy, corrected-measurement versions)
synced to
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-artifacts/joint-die-big/seed{42,123,7}/`.

**Cost: ~$0.91** (2.39hr instance uptime, SPOT g2-standard-4+L4, zero
preemptions, $0.361/hr compute + ~$0.05 boot-disk overhead), well under
the ~$12.04 remaining allotment. Full teardown verified via
`scripts/check_cloud_state.sh`: zero instances/disks/snapshots remain.

## Task 5 (real distillation run): BLOCKER RESOLVED, real run complete —
distilled policy 4/8 (d20) / 1/8 (d12), a real regression vs. each
specialist's own 8/8 baseline (2026-07-19)

**Supersedes the "BLOCKED" entry immediately below** (kept for history,
not deleted — the blocker it describes is the reason this task's
architecture changed). BACKLOG.md's controller decision "(b) single
mixed-population env" was implemented and the real distillation training
was run to completion; this entry is the actual outcome.

**Architecture fix implemented:** added
`FrankaDieLiftJointD12D20MixedEnvCfg`
(`tasks/franka/dice_lift_joint_env_cfg.py`) — ONE `ManagerBasedRLEnv`
splitting `num_envs` between d12/d20 via a deterministic round-robin
`MultiAssetSpawnerCfg(assets_cfg=[d12_cfg, d20_cfg], random_choice=False)`,
both shapes at their own already-verified 48mm-parity scale/mass
(`FrankaDieLiftJointD12BigEnvCfg`'s d12 constant, `FrankaDieLiftJointBigEnvCfg`'s
d20 constant), `scene.replicate_physics = False` per the same
`FrankaDieLiftJointMixedEnvCfg` gotcha. Extended
`object_shape_class_onehot`/`object_geometry_descriptor`
(`tasks/franka/mdp.py`) with a per-env-aware path
(`shape_class_onehot_per_env`/`geometry_descriptor_per_env`,
`tasks/franka/shape_observations.py`) that computes each env's own shape
class as `env_index % len(die_shape_classes_per_env)` — a pure function of
env index, no live USD/spawner-state query needed, exactly as the
controller decision predicted. **Directly verified against a real live
env before trusting it** (not just the source read):
`scripts/_diag_d12d20_mixed_env_check.py` built a real 8-env instance and
cross-checked the predicted round-robin against BOTH the live
`observation_manager`'s own computed `shape_class`/`geometry_descriptor`
values AND the live USD-authored per-env scale (ground truth, independent
of any of this task's own code) — all 8 envs matched the predicted
`env_index % 2` pattern exactly on both checks. `scripts/distill_specialists.py`'s
real-run driver (`build_real_mixed_env`) now builds this ONE env once for
the entire run and calls `collect_rollout` + `regress_on_pooled_batches`
directly each iteration — no more per-iteration env open/close, no more
two-envs-at-once at all.

**Two more real bugs found and fixed, neither ever previously exercised**
(the original two-envs design crashed before real GPU data ever reached
either code path):
- `mix_actions` (`tasks/franka/distillation.py`) built its Bernoulli
  `probs` tensor on the actions' own device (`cuda` for a real GPU run)
  but was called with a CPU `torch.Generator` — `torch.bernoulli` requires
  matching devices, so the first real-GPU smoke test crashed with
  `RuntimeError: Expected a 'cuda' device type for generator but found
  'cpu'`. Fixed the same way `pool_and_shuffle` already handles this split:
  sample on CPU (matching the generator), move the result to the actions'
  device.
- `scripts/distill_specialists.py`'s real-run branch constructed
  `MultiShapeTeacherRouter` with a 2-element `("d12", "d20")` shape-classes
  tuple (copied from `--dry-run`'s own non-faithful 2-shape stub layout)
  against the REAL env's 4-dim canonical one-hot
  (`tasks/franka/shape_observations.SHAPE_CLASSES`, d12/d20 at columns
  2/3) — would have silently produced a broken/`None` relabel (argmax
  values 2/3 never matched the router's own loop indices 0/1). Fixed by
  using the full canonical `SHAPE_CLASSES` tuple for the real run instead.
- Both fixes re-verified: 55/55 unit tests pass (28 Task 4 + 15 new
  per-env tests + 12 pre-existing shape-observation tests unchanged), a
  small real-GPU smoke test (`--num-envs 64 --num-iterations 3`) ran
  end-to-end with a decreasing BC loss curve before the full run was
  dispatched.

**Real run:** desktop GPU (confirmed AVAILABLE at dispatch,
`scripts/check_gpu_availability.sh`), non-headless, `--num-iterations 1500
--num-envs 4096` (default — same total env count as this project's
existing PPO convention, split ~2048/2048 between d12/d20 by the mixed
env's own round-robin; this halves each shape's own per-iteration sample
count vs. the original two-envs-side-by-side design's plan, which would
have given each shape its own full 4096 — an accepted, documented
tradeoff rather than risking an untested 8192-env single-env build).
Completed in ~27 minutes wall-clock (1500 iterations at ~1/s once the
scene finished building), mean BC loss dropping from 0.93 (iteration 0) to
~0.0003-0.0006 by the final iterations — the student's mean action closely
matches its routed teacher's mean action in an MSE sense by the end of
training.

**Real per-shape eval (`scripts/franka_checkpoint_review.py`, the SAME
variants/mechanism as each specialist's own 8/8 baseline, num_envs=8,
video + instrumented height JSON, undiluted 48mm — directly comparable,
not a new eval protocol):**

| shape | variant | distilled policy | specialist baseline |
|---|---|---|---|
| d20 | `joint-die-big` | **4/8** sustained lift | 8/8 |
| d12 | `joint-die-d12-big` | **1/8** sustained lift | 8/8 |

Spot-checked via extracted video frames (not just the JSON) — the failed
envs show physically sensible failed-grasp-attempt scenes (gripper near
but not on the die, die never lifted), not a broken render or crashed
scene, matching the JSON's own per-env `sustained_lift=False` calls.

**This is a real, honest negative/mixed result, not a task failure to
paper over.** Despite the BC loss converging to a very low value, real
behavioral discovery dropped substantially for BOTH shapes relative to
each frozen specialist — worse for d12 (1/8) than d20 (4/8). Plausible
(not yet investigated) explanations, offered as context for whoever picks
up Task 6, not a resolved diagnosis: (1) MSE-on-mean regression matching
two teachers' mean actions closely in aggregate does not guarantee the
student reproduces either teacher's actual CLOSED-LOOP behavior under
autonomous (beta=0) rollout — small per-step action deviations can compound
over a 250-step episode into a qualitatively different (and here, mostly
failing) trajectory, a known DAgger/BC failure mode distinct from anything
about this experiment's shape-mixing specifically; (2) compressing two
shapes' specialist behaviors into one shared 256/128/64 network may simply
be harder for d12's own grasp geometry than d20's (the d12 regression also
loses more relative to its own specialist than d20 does, consistent with
this); (3) the critic was never trained by this module's BC loss (by
design — Task 6's PPO fine-tune trains it from scratch), so it carries no
information here, but that shouldn't affect deterministic mean-action eval
behavior. This project's own prior evidence (Experiment 15, cited in
CLAUDE.md's Tier 1 process) already establishes that a converging loss/
reward scalar does not guarantee real behavior transfers — this is another
concrete instance of that same pattern, not a new phenomenon.

**Distilled checkpoint:**
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/distilled-d12-d20/seed42/2026-07-19_16-10-12/model_1499.pt`
(also `model_1449.pt`). Eval artifacts (video/heights json+npy) synced to
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-artifacts/distilled-d12-d20/{joint-die-big,joint-die-d12-big}/`.

**Cost:** desktop-only (GPU compute + the eval runs), $0 cloud compute
spend; negligible GCS storage/transfer cost for the checkpoint/artifact
upload. No new draw against the plan's $15 cumulative cap (cumulative
spend across Tasks 2/3/5/6 remains ~$3-5, unchanged by this task).

**Task 6 (RL fine-tune) does not proceed from this dispatch** — a
separate, later dispatch, per this task's own scope boundary. Whoever
picks it up should treat this 4/8 (d20, 1/8) d12 distilled starting point
as the ACTUAL starting point for PPO fine-tuning (not an 8/8-equivalent
warm start), and may want to weigh the plausible explanations above before
deciding whether to fine-tune this checkpoint directly or revisit the
distillation loss formulation first.

## Task 5 (real distillation run) attempted, BLOCKED on an Isaac Lab
architectural limit — no distilled checkpoint yet (2026-07-19) — HISTORICAL,
see the superseding entry immediately above for the real outcome

Dispatched the real (non-`--dry-run`) run of `scripts/distill_specialists.py`
against Task 4's two real frozen teachers (d20 `joint-die-big/seed123`, d12
`joint-die-d12-big/seed123`) on the desktop GPU. Two real bugs found under
real execution, neither caught by Task 4's own `--dry-run` (stub envs have
no notion of a simulation context):

**Bug 1, fixed and verified:** the real-run driver originally built BOTH
teacher envs before the DAgger loop, matching `tasks/franka/distillation.py`'s
own "two rollout environments run side by side" design — crashed
immediately with `RuntimeError: Simulation context already exists.` (Isaac
Lab's `SimulationContext` is a process-wide singleton; a second
`ManagerBasedRLEnv` cannot be constructed while a first is open). Fixed by
extracting a new `regress_on_pooled_batches` helper out of
`run_dagger_iteration`'s regression-step tail (identical logic, still
unit-tested) and rewriting the real-run driver to collect each shape's
rollout **sequentially** (open → `collect_rollout` → `close`, per shape,
per iteration) before calling the shared regression step on both
already-collected batches — never two envs open at once.
`run_dagger_iteration` itself is unchanged and still exercises the
simultaneous-stub-envs path for `--dry-run`/unit tests. Re-verified: 28/28
unit tests pass, `--dry-run`'s BC loss curve unchanged
(1.93→1.54→1.05 over 3 iterations).

**Bug 2, NOT fixable within this task — genuine architectural blocker:**
redispatching with the sequential-reopen fix, the job hung with zero log
output for 20+ minutes building the run's SECOND `ManagerBasedRLEnv` (the
first, d20's, built and closed cleanly in 8.5s) — CPU pinned ~104% on one
thread, all `carb.tasking*` workers idle, GPU 1% util, no error, no
progress. Independently confirmed with a minimal isolated repro
(`num_envs=16`, build→close→build, no distillation code at all): first env
1.44s, second env never returns (9m11s CPU time, zero output) — rules out
"just slow at num_envs=4096," this is inherent to reconstructing a
`ManagerBasedRLEnv` in-process after a prior `.close()`, in this Isaac Lab
installation. Both the real run and the repro were killed cleanly; full
teardown independently re-verified both times (`check_gpu_availability.sh`
AVAILABLE, `systemd-inhibit --list` clear of `rl-gpu-job`/
`rl-gpu-job-auto-detect`, `nvidia-smi --query-compute-apps` empty,
`/tmp/rl_isaac_sim.lock` free).

**Why this stops here rather than being worked around:** Task 4's own
foundational design ("two rollout environments side by side") is broken
under BOTH readings (simultaneous or sequential-reopen) in this real
installation. The two remaining fixes are both genuine new architecture,
not bug fixes — (a) two persistent per-shape Isaac Sim processes
exchanging student weights/rollout data via disk each iteration (keeps
Task 1's observation-schema contract untouched, but is new distributed-
training infra), or (b) one single mixed-population env splitting
`num_envs` between d12/d20 via the already-proven
`MultiAssetSpawnerCfg(random_choice=False)` mechanism
(`FrankaDieLiftJointMixedEnvCfg`'s own precedent), which needs
`object_shape_class_onehot`/`object_geometry_descriptor` extended from a
single per-cfg-constant broadcast to a per-env-aware read of the actually-
spawned asset — a real change to an established observation-term contract.
Flagged to the controller rather than picked unilaterally; full detail,
evidence, and this task's own (non-binding) lean toward option (b):
`BACKLOG.md`'s "Task 5 ... BLOCKED" entry.

**State:** no distilled checkpoint exists yet. Code changes committed:
`franka_checkpoint_review.py`'s `load_optimizer=False` fix (needed for the
eventual eval step, found while preparing for it, independent of the
blocker above) and the `regress_on_pooled_batches` refactor + (currently
non-functional, blocked) sequential-reopen real-run driver in
`scripts/distill_specialists.py`/`tasks/franka/distillation.py`. No cloud
spend (desktop-only, both dispatches torn down cleanly with zero waste).
Task 6 (RL fine-tune) cannot proceed until Task 5 actually produces a
checkpoint.

**RESOLVED 2026-07-19 — see the "Task 5 ... BLOCKER RESOLVED" entry above
this one for the real fix and real result.**

## Task 6 (RL fine-tune) + FINAL VERDICT: unified d12/d20 policy fully
recovers to 8/8 BOTH shapes — matches each frozen specialist exactly, no
gap (2026-07-19)

**This closes the unified-multi-die-specialist-distillation experiment.**
PPO-fine-tuned Task 5's distilled checkpoint (4/8 d20, 1/8 d12 pre-
fine-tune, a real BC/DAgger closed-loop-transfer regression) against
`FrankaDieLiftJointD12D20MixedEnvCfg` — the same one-env, deterministic-
round-robin d12/d20 mixed-population env Task 5's own distillation
training ran against.

**Two real bugs found and fixed before training, both re-verified before
trusting the real run:**
1. `scripts/train_franka.py` had never been wired for
   `FrankaDieLiftJointD12D20MixedEnvCfg` at all — Task 5 built this env
   directly in its own driver code, but the class was never added to this
   script's `--variant` choices. Added `--variant joint-die-d12-d20-mixed`.
2. `scripts/train_franka.py`'s `--checkpoint` resume path called
   `runner.load(args_cli.checkpoint)` with rsl_rl's default
   `load_optimizer=True` — unconditionally calls
   `self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])`,
   which would crash with a `KeyError` on Task 5's distilled checkpoint
   (`save_student_checkpoint` intentionally writes an EMPTY
   `optimizer_state_dict`, since a BC optimizer's Adam moments have
   nothing to do with PPO's — see Task 5's own entry). This is the exact
   same failure class already fixed in `franka_checkpoint_review.py`
   during Task 5. Fixed by adding a new `--policy_only_checkpoint` flag
   (passes `load_optimizer=False` when set, leaves the default
   `load_optimizer=True` behavior unchanged for a genuine same-run
   SPOT-preemption-recovery resume, where restoring Adam's own state is
   the whole point). Verified with a bounded 3-iteration smoke test
   (`--num_envs 64 --max_iterations 1502`) on the desktop before
   dispatching the real run — loaded cleanly, ran 3 iterations, no crash.
   Both fixes committed (`f836dae`) and pushed before the real run.
   `scripts/sync_run_to_gcs.py`'s `VARIANT_MAP` was also missing the new
   variant's log-dir-name mapping (`train_franka_jointdied12d20mixed` ->
   `joint-die-d12-d20-mixed`) — would have raised `KeyError` on the first
   sync attempt; fixed in the same pass (`a55d2c8`).

**Budget: 1500 PPO iterations** (this project's standard from-scratch
convention — used directly rather than guessing a smaller number, per
this task's own instruction to default to the established convention when
unsure of how much fine-tuning a distilled starting point needs).
Mechanically, this required `--max_iterations 2999`, not `1500`: Task 5's
`save_student_checkpoint` wrote its own DAgger loop's iteration counter
(1499) into the checkpoint's `"iter"` field in the same format
`rsl_rl.OnPolicyRunner.save()` uses, so `--checkpoint`'s existing
resume-arithmetic (`num_learning_iterations = max_iterations - resumed_at`)
carries that number forward as a bookkeeping/TensorBoard-step offset
(cosmetic only — this is a different training process, PPO fine-tuning a
BC-initialized policy, not a continuation of the DAgger run itself). The
smoke test confirmed this arithmetic directly: "Resumed from ... at
iteration 1499; running 3 more iteration(s) to reach the absolute target
1502." The real run used `--max_iterations 2999` (1499 + 1500) to get a
true 1500-iteration PPO budget, ending at `model_2998.pt`.

**Real run:** desktop GPU (confirmed AVAILABLE via
`scripts/check_gpu_availability.sh` before dispatch), non-headless
(`DISPLAY=:1`), `--num_envs 4096` (~2048/shape via the mixed env's own
round-robin, matching Task 5's own split), `flock -o`-guarded. Completed
in 27m54s wall-clock (iteration 1499->2998), ~1.1-1.2s/iteration
throughout, no preemptions/crashes (this is desktop, not cloud SPOT).

**Real per-shape eval** (`scripts/franka_checkpoint_review.py`, the
IDENTICAL mechanism/variants Task 5 and every specialist baseline used —
`joint-die-big` for d20, `joint-die-d12-big` for d12, `num_envs=8`,
undiluted 48mm, video + instrumented per-step height JSON):

| shape | variant | pre-fine-tune (Task 5) | **post-fine-tune (Task 6)** | specialist baseline | verdict |
|---|---|---|---|---|---|
| d20 | `joint-die-big` | 4/8 | **8/8** | 8/8 | **PASS — zero gap** |
| d12 | `joint-die-d12-big` | 1/8 | **8/8** | 8/8 | **PASS — zero gap** |

**Falsification check against the spec's pre-registered bar ("not
meaningfully below its own specialist"), reported exactly as observed per
this experiment's own no-softening discipline:** both shapes PASS, and
not narrowly — the fine-tuned unified policy matches each frozen
specialist's own 8/8 EXACTLY, a full recovery, not a partial one. RL
fine-tuning fully closed the real BC/DAgger closed-loop-transfer
regression Task 5 found and reported honestly rather than papering over.

**Not a measurement artifact — checked past the summary JSON, per this
experiment's own repeated verification discipline (Tasks 3/3.5's own
settle-detection bugs are exactly why this matters here):** every env's
`max_height_gain` is large and decisive (d20: 0.412-0.478m; d12:
0.386-0.427m — both roughly 10x the 0.04m lift threshold, not a marginal
call) and `max_consecutive_lifted_steps` is 211-219 out of a ~239-step
analysis window (the lift holds for essentially the entire episode, not a
brief threshold-crossing blip). Extracted and directly viewed real video
frames (not just the JSON) for env_0 of both shapes: a rest-pose frame
(step 5) shows the die genuinely sitting on the table before the arm
moves; a peak frame (step 57-62, matching each env's own recorded
argmax-height step) shows the arm in a visibly different, raised elbow
pose with a small object visibly gripped between the closed jaws —
physically consistent with a real grasp-lift, not a wedge/contact
artifact (Experiment 16's own precedent for why this check matters).

**Checkpoint:**
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-d20-mixed/seed42/2026-07-19_12-53-35/model_2998.pt`.
Eval artifacts (video/heights json+npy) synced to
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/eval-artifacts/finetuned-d12-d20/{joint-die-big,joint-die-d12-big}/`.

**Cost: $0 for this task** (desktop-only — both the fine-tune training
and both eval runs). **Cumulative cost across the whole experiment's
cloud-touching tasks (Tasks 2+3+3.5, the d20-big-geom gate task, Task 5,
Task 6) ≈ $5.87 of the plan's original $15 cap**: Tasks 2+3+3.5 ≈$4.96
(ROADMAP's Task 3.5 entry), the d20-big-geom gate task +$0.91 (this
ROADMAP's own entry above), Task 5 +$0 (desktop-only), Task 6 +$0
(desktop-only). Well under the cap; no controller notification needed.

**Full teardown verified**: `nvidia-smi --query-compute-apps` empty,
`tmux ls` shows no server running, `systemd-inhibit --list` clear of any
`rl-gpu-job*` guard, `scripts/check_gpu_availability.sh` reports
AVAILABLE — all checked directly after the run, not assumed.

**Code changes:** `scripts/train_franka.py` (`--variant
joint-die-d12-d20-mixed`, `--policy_only_checkpoint`, commit `f836dae`),
`scripts/sync_run_to_gcs.py` (`VARIANT_MAP` fix, commit `a55d2c8`). No
env-cfg or observation-schema changes needed — Task 5 already built
everything Task 6 needed to consume.

---

### FINAL VERDICT — unified-multi-die-specialist-distillation experiment (Tasks 0-6, 2026-07-16 -> 2026-07-19)

**The experiment's original 4-shape goal (d8/d10/d12/d20) narrowed to a
2-shape goal (d12/d20) partway through, on real evidence, not a scope
retreat of convenience:** d8 and d10 are genuinely, robustly null at
every size/geometry-conditioning combination tested (Task 2's real
~16-18mm size AND Task 3.5's 48mm-parity anchor, 0/9 and 0/6
respectively, wide safety margins, independently re-derived from raw
trajectories twice) — shape itself is a real barrier for these two
specific die geometries at the Franka gripper's absolute scale, not a
population-dilution or measurement confound. d12 and d20 both cleared an
undiluted-48mm-population bar (Task 3.5's re-audited d12 8/8, the
d20-big-geom gate task's 2/3-seeds-at-8/8) once Task 3's real confound —
population dilution across a randomized-size env, not shape — was
isolated and controlled for.

**With that narrowed 2-shape scope, the full specialist -> distill ->
RL-fine-tune pipeline (UniDexGrasp++'s GiGSL pattern) worked end to end,
including surfacing and recovering from a real, literature-predicted
failure mode along the way:** naive behavior-cloning/DAgger distillation
of two 8/8 frozen specialists converged to a very low imitation loss
(Task 5) but did NOT preserve real closed-loop discovery — 4/8 (d20),
1/8 (d12), a genuine regression this project reported honestly rather
than treating the converged loss curve as sufficient evidence (this
project's own Experiment 15 precedent: a converging shaped scalar does
not guarantee real behavior transfers). The RL fine-tune step Task 6
exists specifically to close (per GiGSL's own iterate-distillation-and-RL
design) did exactly that: **both shapes recovered to their frozen
specialists' own 8/8 EXACTLY, a full, not partial, recovery.**

**Bottom line:** a single unified policy that grasps-and-lifts either a
commanded d12 or d20 die, indistinguishable in closed-loop discovery rate
from two separate single-shape specialists, is real and checkpointed
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-d20-mixed/seed42/2026-07-19_12-53-35/model_2998.pt`).
d8/d10 remain open, unsolved shapes for a future experiment (not this
one) to pick up, with a documented, evidence-backed reason (real
shape-specific discoverability barrier, not a fixable pipeline defect) to
start from rather than re-litigating from scratch. Total cost across the
whole experiment: ≈$5.87 of the original $15 cloud-spend cap, roughly
61% under budget. No further work planned under this experiment; see
`kb/wiki/experiments/unified-multi-die-specialist-distillation.md` for
the compiled cross-referenced write-up.

---

## Task 6 (Stage D2) + FINAL VERDICT: target-selection-among-distractor-dice experiment PASSES — d12 8/8, d20 8/8 under 2-distractor clutter (2026-07-19)

Closing verdict for the target-selection-clutter experiment (plan:
`docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md`,
spec: `docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md`),
Experiment 2 of the multi-die RL arc, extending
[[unified-multi-die-specialist-distillation]]'s finished single-object
d12/d20 policy (`model_2998.pt`, 8/8 both shapes with exactly one die in
the scene) into a 3-die scene (1 commanded target + 2 distractor dice)
via a distractor-count curriculum (SO: 0 active -> D1: 1 -> D2: 2) plus a
new fixed-size zero-padded distractor-distance observation term
(DexSinGrasp's own `d_t^S` mechanism, arXiv:2504.04516), with the reward
function and target-identification mechanism left completely unchanged.
Full technical writeup: `kb/wiki/experiments/target-selection-clutter.md`.

**Pre-registered falsifiable hypothesis: PASSES for both shapes.**

| stage | active distractors | d12 | d20 | gate/bar | verdict |
|---|---|---|---|---|---|
| SO (original, from-scratch) | 0 (parked) | 0/8 | 0/8 | >=7/8 internal sanity gate | FAIL — confounded |
| SO (corrected, warm-started from `model_2998.pt`) | 0 (parked) | 8/8 | 7/8 | >=7/8 internal sanity gate | PASS |
| D1 | 1 (real) | 8/8 | 8/8 | intermediate data point | matches single-object 8/8 baseline |
| D2 (primary falsification check) | 2 (real, both slots) | **8/8** | **8/8** | >=6/8 (75%) primary bar | **PASS — NOT falsified** |

**Stage SO's original from-scratch attempt was a real, worth-recording
confound, not a dead end:** the 43-dim observation schema (the new
`distractor_distance_summary` term) is incompatible with the 41-dim
`model_2998.pt` checkpoint, forcing Stage SO to train fully from scratch
per the plan's own design. That from-scratch run scored 0/8 both shapes
— a "reach but never grasp" pattern indistinguishable from this
project's own long-documented d12/d20 cold-start grasp-discovery
difficulty (the exact barrier the specialist -> distill -> RL-fine-tune
pipeline exists to route around), confounding "did the new scene/
observation code break something" with "does plain from-scratch PPO ever
discover these grasps at all." The fix — a new weight-surgery script
(`scripts/extend_checkpoint_observation_dims.py`) extending
`model_2998.pt`'s 41-dim first-layer weights to 43 dims (41 columns
copied unchanged, 2 new always-zero-at-Stage-SO columns freshly
initialized) — is mathematically guaranteed lossless at Stage SO
specifically, verified bit-for-bit identical (0.0 max abs diff) against
the real checkpoint before any training spend (with a negative-control
check confirming the verification wasn't vacuous). The corrected,
warm-started Stage SO passed cleanly (d12 8/8, d20 7/8), confirming the
schema/scene extension itself was never broken.

**Stage D1 (1 real distractor) and Stage D2 (2 real distractors, the
target configuration) both cleared their bars with no discovery
degradation from the single-object baseline**, and Stage D2 — the
experiment's real primary result — matched it exactly (8/8 both shapes,
not just above the 6/8 floor). Instrumented numbers, not just the summary
fraction: D1 `max_height_gain` 365-457mm (d12) / 409-440mm (d20),
215-222/239 sustained-lift steps; D2 `max_height_gain` 308.3-480.8mm
(d12) / 334.3-481.2mm (d20), 216-223/249 (d12) and 217-222/249 (d20)
sustained-lift steps — all far above the 40mm lift threshold and holding
for the large majority of each analysis window at every stage, not a
brief threshold-crossing blip.

**Both D1 and D2 eval videos were downloaded and inspected frame-by-frame
(`ffmpeg`-extracted stills, not just the JSON), specifically checking
whether the policy ever grasped the WRONG die (a distractor instead of
the commanded target) — a distinct, more concerning failure mode than a
simple discovery-rate shortfall. No such episode was found in any
inspected frame, at either stage, for either shape**: distractor dice
remain visibly undisturbed in their fixed reset positions throughout
every inspected timestamp, while the gripper carries the target through a
large elevation arc. Also recorded as a structural (not just visual)
guarantee: the height instrumentation reads `scene["object"]`'s own root
position directly — `scene["object"]` is structurally always the
commanded target, a physically separate rigid body from either distractor
slot — so a wrong-die grasp could not by itself produce an inflated
`max_height_gain` reading for the target; the 8/8 metric alone already
rules out "lifted a distractor, left the target behind," and the video
check confirms the complementary case (no distractor disturbance
alongside the correct grasp).

**Execution/cost, including a real cross-workstream infra finding:** Task
6 (Stage D2) initially dispatched to cloud (desktop was BUSY at check
time) and hit a genuine SPOT preemption after install completed but
before training started; the restart attempt then hit real zone stockout,
and investigating surfaced that this project's GCP `GPUS_ALL_REGIONS`
quota is **1, project-wide** — already held by a different concurrent
Senior workstream's own live cloud instance (the exploration-bonus
experiment), blocking a second simultaneous cloud dispatch. Rather than
wait on or touch another workstream's resource, this task rechecked
desktop availability, found it had freed up, and switched to a fresh,
isolated desktop checkout (`~/projects/rl-target-selection-d2`,
deliberately not the shared `~/projects/rl` checkout there, which had a
different workstream's uncommitted changes) — training and both eval
runs completed on desktop at $0. **This is flagged to the controller as a
real, previously-undocumented constraint** on how many Senior workstreams
can use cloud GPU dispatch simultaneously under this project's
fan-out/parallel-ownership model — not resolved here (a Console-UI-only
quota increase request, same mechanism as the original grant).

**Total cost across Tasks 4/5/6: ≈$1.35 of the plan's $5 cap** (SO's two
attempts ≈$0.83 + D1 ≈$0.36 + D2 ≈$0.16 aborted-cloud, desktop-completed)
— well under, no controller notification needed. Full teardown verified
at every stage (`nvidia-smi --query-compute-apps` empty, no `tmux` server,
`systemd-inhibit --list` clear, `check_gpu_availability.sh` AVAILABLE for
desktop; `scripts/check_cloud_state.sh` clean of this task's own
resources for the aborted cloud attempt).

**Checkpoints:**
- SO (corrected, warm-started): `gs://rl-manipulation-hks-runs/target-selection-clutter-stageso-warmstart/joint-die-target-selection-so/seed42/2026-07-19_22-52-41/model_3297.pt`
- D1: `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d1/seed42/2026-07-19_23-58-28/model_4097.pt`
- **D2 (final, this experiment's own end state):** `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d2/seed42/2026-07-19_21-08-07/model_5096.pt`

**Bottom line:** a single unified policy that grasps-and-lifts a
commanded d12 or d20 die when 2 other dice (independently drawn from the
same 2-shape population) are simultaneously present in the scene, with no
degradation from the single-object 8/8 baseline and no evidence of ever
grasping the wrong entity, is real and checkpointed. Curriculum + a
fixed-size zero-padded distractor-distance observation term, transplanted
from DexSinGrasp's own state-based-teacher formulation despite that
paper's materially different setting (dexterous multi-finger hand,
heaped/occluding clutter, vs. this project's parallel-jaw gripper, flat
non-occluding tabletop), was sufficient on its own — no
distractor-avoidance reward term was needed, and none was attempted, per
this experiment's own scope. Since the primary bar passed cleanly, the
spec's own pre-registered falsification-escalation path (a Deep-Sets/
attention architecture over distractor state, or a distractor-avoidance
reward term) is not needed. Open follow-ons (not decided or started
here): d8/d10 as distractors or targets (still gated on those shapes ever
achieving real single-object discovery first), 3+ distractors or
heaped/occluding arrangements, and multi-seed replication of this
experiment's single-seed (seed42) result. No further work planned under
this experiment; see `kb/wiki/experiments/target-selection-clutter.md`
for the full compiled, cross-referenced write-up.

---

### d8/d10 demonstration-augmented warm-start (2026-07-19 -> 2026-07-20): H1 FALSIFIED both shapes

Tested H1 from `docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`:
DAPG-style behavior-cloning pretrain from one real scripted-grasp
demonstration trajectory per shape (captured via `dice_pick_demo.py`'s
own DiffIK grasp controller at the unified-multi-die-specialist-
distillation experiment's 48mm-parity anchor), warm-starting an otherwise
completely unchanged full 1500-iteration PPO fine-tune — a candidate fix
for that experiment's own robust d8/d10 0/24-both-shapes cold-start null.

Tasks 0-2 (re-verify the scripted grasp transfers to 48mm scale, build
`regress_on_paired_batches`/`tasks/franka/demo_action_mapping.py`/
`scripts/extract_demo_trajectory.py`/`scripts/bc_pretrain_demo_warmstart.py`)
completed cleanly — BC-pretrain converged to a clean loss plateau
(≈0.0007-0.0009) for both shapes and a bounded PPO-handoff smoke test
confirmed the resume mechanics.

**Task 3 (the real H1 run) hit real, previously-undiscovered cloud-dispatch
infra gaps**, all resolved in-task rather than worked around silently:
the demo/capture scene's 5-die visual mesh assets
(`vision/data/raw/dice_sets_v1/set_00013_*.usd` — note: `set_00013`, NOT
`set_00000`, a real distinction the first attempt at resolving this got
wrong before self-correcting) and the vision detector's model weights
were desktop-only and had never been shipped to any cloud instance before
in this project (every prior cloud run either used the asset-free
`ik-cube` variant or ran on the desktop directly); resolved via a narrow
read-only file copy of the 10 needed asset files plus a real bug fix in
`scripts/extract_demo_trajectory.py` (it called the vision detector
subprocess unconditionally even under `--gt-xy-bypass`, whose whole point
is not needing the detector — commit `bdb31b2`) so cloud captures for
both shapes never need vision infra at all. The `notch_fixture.usd`
fingertip-fixture asset (offline `pxr`-only build script,
`scripts/build_notch_fixture_asset.py`) needed a symlink to the cloud
instance's own pip-installed Isaac Sim extension paths rather than the
desktop's from-source layout. A project-wide `GPUS_ALL_REGIONS=1` GCP
quota (shared across every concurrent Senior cloud workstream) blocked
provisioning for ~40 real minutes until a sibling workstream's job
finished — the same quota-sharing constraint independently flagged by the
concurrent target-selection-clutter experiment's own Task 4-6 report.
SPOT preemptions clustered unusually severely (6 across this task's real
GPU-active time); recovered via checkpoint-resume each time, then
switched the remaining runs to on-demand provisioning (this project's own
documented precedent) after the 6th, via a snapshot-and-zone-migrate
cycle — zero further preemptions afterward. **A second real bug** was
found and fixed in `scripts/franka_checkpoint_review.py`: its output
filenames depended only on the checkpoint's basename, so 3 seeds of the
same shape training to the same iteration count (identical
`model_1519.pt`/`model_1517.pt` names in different run directories)
silently overwrote each other's eval video/heights.npy/summary.json; all
6 evals were re-run after the fix with genuinely distinct artifacts.

**Result: 0/8 sustained-lift discovery in every one of 6 runs (3 seeds x
2 shapes) — d8 0/24, d10 0/24. H1 is falsified for both shapes**, per the
spec's own falsification bar. Independently re-derived (a fresh
reimplementation of the settle-window/gain/sustained-lift logic, not a
re-run of the eval script) from the raw `heights_*.npy` for one seed per
shape and got byte-identical 0/8 both times; per-env `max_height_gain_m`
was small (≈8.8mm d8, ≈4.3-5.4mm d10, well under the 40mm threshold) and
essentially identical across all 8 parallel envs within each run — not a
policy that sometimes grasps, one that never learned any real
object-contingent behavior from the single-demonstration warm start.
Frame-by-frame video review (both re-derived seeds) confirms the gripper
approaches and hovers near the die in both shapes but never closes a real
grasp, consistent with the instrumented null (Experiment 16 precedent for
why the video check matters). No never-before-observed "partial"
(1/8-7/8) result was seen in any seed.

**Checkpoints:** `gs://rl-manipulation-hks-runs/d8-d10-demo-warmstart/joint-die-{d8,d10}-big/seed{42,123,7}/<timestamp>/model_{1519,1517}.pt`.
Eval artifacts: `gs://rl-manipulation-hks-runs/d8-d10-demo-warmstart/eval-artifacts/`.

**Cost: ≈$4 of the plan's $10 cloud-spend cap** (duration × published-SKU-
rate estimate, no exact billing export exists — this project's standing
methodology). Full teardown verified after every provisioning cycle.

**Bottom line:** BC-pretraining from one real demonstration per shape
(the original Task 2 design's pooled-5-trajectories approach — this run's
own captures produced 5 valid trajectories per shape, all pooled) does
not unlock grasp discovery for either d8 or d10 at the 48mm-parity
anchor. H2 (checkpoint warm-start from the d12 specialist) remains
pre-authorized as the next rung per the spec's own fallback design but
was **not started here** — reported back to the controller per the
plan's own stop-and-report instruction; see
`kb/wiki/experiments/d8-d10-demo-warmstart.md` for the full compiled
write-up.

### Exploration-bonus grasp discovery (2026-07-19 -> 2026-07-20): H1 SPLIT — mechanism fires (seed 123), lift still never completes (0/24)

Task 4 (real 3-seed/1500-iteration run) of
`docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-discovery-implementation.md`,
testing H1 (a GRM `D=1`, action-dependent potential-based exploration
bonus for gripper-closure attempts near the object —
`gripper_closure_attempt_bonus` + `gripper_closure_attempt_bonus_correction`,
Tasks 1-3 already built/committed: `10a9588`, `59b0246`, `e9bc14b`) against
d8's robust, independently-established 0/24 from-scratch null at the
48mm-parity anchor (`FrankaDieLiftJointD8BigEnvCfg`, Task 3.5).

**Both falsification bars, per seed, never averaged:**

| seed | mechanism-level (`frac_steps_raw_action_negative_near_object`) | behavioral (`envs_with_sustained_lift`) |
|------|------------------------------------------------------------------|------------------------------------------|
| 42   | 8/8 envs **null** (final checkpoint never got within 5cm of the object in any env, all episode) | 0/8 |
| 123  | 7/8 envs = **1.0** (every near-object step had a negative raw gripper action — the mechanism firing strongly), 1/8 null | 0/8 |
| 7    | 1/8 env = **0.0** (confirmed: got near, never attempted), 7/8 null | 0/8 |

Per the spec's own pre-registered rule, "H1's mechanism-level claim is
falsified only if this fraction is exactly `0.000` in all 3 seeds" —
seed 123's `1.0` result (a clear, repeated, nonzero value, not a rounding
artifact) means **the mechanism-level bar is not falsified; it fires**.
The behavioral bar is a clean 0/24 across all 3 seeds — `max_height_gain_m`
≈0.0088m for every env in every seed (essentially identical across all 24
env-seed combinations, the same physics-settle-noise magnitude seen in
the from-scratch baseline and the d8/d10 demo-warmstart null, well under
the 0.04m lift threshold).

**Overall verdict, per the spec's own explicit rule (falsified only if
BOTH bars fail across all 3 seeds): this is the SPLIT outcome, not a
falsification** — the exploration mechanism demonstrably works (in ≥1
seed, the policy reliably samples gripper-closure attempts specifically
when near the object, not at random), but this does not translate into
any completed lift. This is the first result in this project's history
that lands in this explicitly-anticipated third category rather than a
plain pass/fail.

**Independent verification** (per this project's standing practice):
re-derived both bars for seed 123 (the strongest mechanism signal) from
raw per-step arrays, not the summary JSON — `raw_arrays.npz`
(`raw_gripper_action`/`ee_object_distance`) reproduced the exact
per-env fractions (envs 0-3/5-7 = 1.0, env 4 = null) byte-for-byte; a
fresh, independent reimplementation of `franka_checkpoint_review.py`'s
post-`977a748` settle-detection logic against the raw `heights_*.npy`
reproduced 0/8 and the same `max_height_gain_m` values to displayed
precision. Frame-by-frame video review of seed 123 (frames 10/15: gripper
descending toward the die; frames 20/30: fingers visibly closed at
roughly the die's location, object occluded/not visible; frames 100/248:
die still resting on the table, ungrasped, gripper retracted to a resting
pose above and away from it) visually confirms the instrumented split —
a real closure attempt near the object that does not result in a lift,
not a video/instrumentation mismatch (Experiment 16 precedent for why
this check matters).

**No code bug found or fixed in this task** — Tasks 1-3's mechanism,
instrumentation, and wiring all behaved exactly as designed; the split
result is a genuine substantive finding about grasp mechanics, not a
defect in the exploration-bonus implementation. (Real, non-code infra
friction did occur: two SPOT preemptions during the eval phase, both
apparently from contention with the concurrently-running d8/d10-H2
workstream over this project's shared `GPUS_ALL_REGIONS=1` project-wide
quota — the same constraint already flagged by the target-selection-clutter
and d8/d10-demo-warmstart tasks. Both were real blocking-polled, not
guessed through; `--instance-termination-action=STOP` preserved the
instance's disk/venv/repo state across both, so recovery was a plain
restart + idempotent script re-run, no re-install and no training
re-run needed — all 3 seeds' 1500-iteration training runs themselves
completed in a single shot with zero preemptions.)

**Checkpoints:** `logs/train_franka_jointdied8bigexplorationbonus/{2026-07-20_09-52-47,2026-07-20_10-25-11,2026-07-20_10-57-38}/model_1499.pt`
(seeds 42/123/7 respectively; cloud-local, not synced to GCS — this task's
own dispatch did not run `sync_run_to_gcs.py`, only the 3 diagnostic/eval
scripts needed for the verdict itself).

**Cost: ≈$1.2 of the plan's $3 cap** (≈3.0hr real SPOT g2-standard-4+L4
instance-running time across 3 start/stop windows, at the recipe's
published $0.361/hr combined rate, plus ≈$0.07 disk — duration ×
published-SKU-rate estimate, this project's standing methodology, no
exact billing export exists). Full teardown verified
(`scripts/check_cloud_state.sh` shows zero resources belonging to this
task; the other workstream's own `rl-d8d10-h2-checkpoint-warmstart`
instance/disk, still present, is not this task's to touch).

**Bottom line and forward pointer:** per the spec's own explicit
characterization of this exact outcome, this points *away* from the pure
discoverability question this spec targeted and *toward* a downstream
grasp-mechanics problem — the policy now reliably finds and attempts the
right moment/place to close (seed 123), but closing there still doesn't
produce a lift. This is meaningfully different from Experiment 5's
AR4-era null (policy never even approached the object, "reach, grip,
freeze" — see [[experiment-05-potential-based-reward-shaping]]) and from
this arc's own from-scratch/demo-warmstart nulls (gripper hovers near the
object but the raw action never goes negative there at all): for the
first time on this shape, real closure attempts are being sampled exactly
where they should be, and still fail to lift. The honest next candidate
direction is grasp-mechanics-specific, not exploration-specific:
verifying actual finger-object contact/antipodal alignment at the moment
of closure (this project's Experiment 9/10 antipodal-grasp-quality axis,
not revisited since the Franka pivot), or the d8 die's own
geometry/mass/friction parameters at 48mm-parity scale, rather than a
further exploration- or warm-start-based fix — those are the two
already-tried axes (this experiment and d8/d10-demo-warmstart) that both
now show the same shape: getting the gripper to the right place/moment is
solvable, closing effectively once there is not. Not decided or started
here — a stop-and-report point back to the controller, per the spec's own
convention for a result outside its pre-authorized fallback chain (H2/H3
are only pre-authorized for a plain both-bars falsification, which this
is not). See `kb/wiki/experiments/exploration-bonus-grasp-discovery.md`
for the full compiled write-up.

---

### d8/d10 demonstration-augmented warm-start, H2 fallback (2026-07-20): PASSED both shapes — investigation closed

H1 (`ROADMAP.md`'s own entry above/`kb/wiki/experiments/d8-d10-demo-warmstart.md`)
had just cleanly falsified for both d8 and d10 (0/24 each). Per the
design spec's pre-authorized "one fallback rung, no new spec" convention,
executed H2 for both shapes: direct `scripts/train_franka.py --checkpoint
<d12 checkpoint> --variant joint-die-{d8,d10}-big --max_iterations 2999`
resume (full optimizer-state resume, no `--policy_only_checkpoint`) from
the already-converged, nearest-by-sphericity d12 specialist checkpoint
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`,
ψ=0.9286 vs. d8 ψ=0.8896/d10 ψ=0.8959) — no new pipeline code, exactly as
the spec specified. Pre-flight-verified the checkpoint directly
(`gsutil stat` + a real `torch.load` shape check, not trusted from the
spec text alone): `actor.0.weight (256, 41)`, `actor.6.weight (8, 64)` —
an exact match to d8/d10's own 41-dim observation / 8-dim action schema,
confirming clean drop-in compatibility.

**Result: d8 3/3 seeds full 8/8 (24/24 envs, a clean sweep — matching
cube's own perfect record and exceeding d12's 1/3/d20's 2/3 at this same
anchor). d10 1/3 seeds full 8/8 (seed7 only, 8/24 envs — the same "0 or
full-8/8-within-seed, never a spurious partial" pattern this project has
seen for every other shape's discovering seed).** Neither shape
falsified; both pass H2's bar cleanly. Independently re-derived from raw
`heights_*.npy` for d8 seed42, d10 seed7, and d10 seed42 (a from-scratch
reimplementation of the resting-z/gain/sustained-lift logic, not reusing
`franka_checkpoint_review.py`'s own code) — matched the tool's own numbers
exactly after fixing a self-caught bug in that reimplementation (an
initial pass wrongly included the object's pre-settle spawn-drop
transient in the lift-analysis window, producing a spuriously elevated
gain reading for a genuinely-null d10 seed; restricting the window to
start at `post_settle_start_step` per the tool's own documented
convention fixed it). Frame-by-frame video review (d8 seed42, d10 seed7,
d10 seed42-null) confirms a real gripper-closed-and-lifted posture change
for both positive seeds and continued non-engagement for the null,
matching the instrumented numbers.

**Real infra friction, same category as H1's Task 3:** the single
project-wide `GPUS_ALL_REGIONS=1` quota was contended twice by a
concurrent sibling workstream's own training batch (real blocking polls,
not worked around); two genuine SPOT preemptions (confirmed via `gcloud
compute operations list` preemption system events) recovered via
checkpoint-resume, after which the instance was switched from SPOT to
on-demand provisioning (`gcloud compute instances set-scheduling
--no-preemptible --provisioning-model=STANDARD` — note
`--maintenance-policy=MIGRATE` is rejected for GPU-attached instances,
`onHostMaintenance` must stay `TERMINATE`; folded into
`docs/cloud/dispatch-checklist.md`), after which zero further
preemptions occurred. Confirmed switching to on-demand does **not**
bypass the shared `GPUS_ALL_REGIONS` quota itself (only protects against
preemption once the GPU is actually acquired). Confirmed
`franka_checkpoint_review.py`'s output-filename collision fix (commit
`d5b9cd1`) was already present and held correctly under this task's own
same-basename-across-seeds scenario (6 genuinely distinct artifact sets
produced).

**Cost: ≈$3.44** (duration × published SPOT/on-demand SKU rates,
on-demand L4 rate $0.560/GPU-hr confirmed via the live Cloud Billing
Catalog API rather than assumed) — well under the plan's $10 cap; combined
with H1's own ≈$4, total experiment spend ≈$7.44. Full teardown verified
clean (`scripts/check_cloud_state.sh`).

**Checkpoints:** `gs://rl-manipulation-hks-runs/d8-d10-h2-checkpoint-warmstart/joint-die-{d8,d10}-big/seed{42,123,7}/<timestamp>/model_2998.pt`.
Eval artifacts: `gs://rl-manipulation-hks-runs/d8-d10-h2-checkpoint-warmstart/eval-artifacts/`.

**Bottom line: the d8/d10 grasp-discoverability investigation is closed
with a real positive resolution, not a third null.** From-scratch PPO
cannot discover the d8/d10 grasp with this project's standard recipe
(the original Task 3.5 null and H1's demonstration-BC-warm-start null
both confirm this), but PPO fine-tuned from a different, geometrically-nearest
shape's already-converged weights can — cleanly for d8, partially
(matching this project's own established discovery-rate pattern) for
d10. This is evidence the original null was a policy-initialization/
exploration problem specific to cold-start learning, not an intrinsic
physical or reward-design barrier: the same reward function, PPO
hyperparameters, and observation schema that never discovered the grasp
from scratch now discovers it reliably once seeded from nearby-shape
weights. `BACKLOG.md`'s "Task 4 scope decision" (which deferred d8/d10
from the specialist-distillation arc on the original from-scratch null)
can now be revisited with this positive result in hand — a decision for
Principal, not made here. See `kb/wiki/experiments/d8-d10-demo-warmstart.md`'s
"H2" section for the full compiled write-up.

---

### d8 antipodal/force-closure grasp-quality reward, dual action-space test (2026-07-20): H_joint FALSIFIED, H_taskspace CONFIRMED — exact AR4 Experiment 10→11 replay, transferred cross-platform for the first time

Spec: `docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md`.
Plan: `docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-quality-implementation.md`.
Follows directly from [[exploration-bonus-grasp-discovery]]'s own forward
pointer (mechanism-level closure attempts fire reliably, no lift ever
completes; next candidate axis named there: verify actual finger-object
contact/antipodal alignment at the moment of closure). Ported AR4's
`antipodal_grasp_bonus` (`tasks/ar4/mdp.py:902-940`, Experiments 9-11's own
mechanism) onto d8's independently-established 0/24 grasp-discoverability
null (`FrankaDieLiftJointD8BigEnvCfg`, 48mm-parity), refit to this scene's
real μ=0.5 friction (`antipodal_cos_threshold=-0.894427`, vs. AR4's own
μ=1.0 → `-0.7071`), and tested it as two separate, individually falsifiable
hypotheses per the AR4-era arc's own explicit precedent that joint-space and
task-space control produced qualitatively different outcomes for this exact
mechanism (Experiment 10: regresses to exact zero under joint-space;
Experiment 11: task-space/IK unlocks the first genuine sustained signal).
Both conditions run to completion unconditionally, 3 seeds/1500 iterations
each (Tasks 1-2 build/wire, committed `74bd058`/`de02c5d`; Task 3 = H_joint,
committed `f8f5fca`; Task 4 = H_taskspace, committed `cb39276`).

**H_joint (Condition A, joint-space `JointPositionActionCfg`, unchanged from
`FrankaDieLiftJointD8BigEnvCfg`'s own default): FALSIFIED, both bars, all 3
seeds.** Mechanism-level (`Episode_Reward/antipodal_grasp_quality`,
final-100-of-1500-iteration mean): **exactly `0.00000000` in all 3 seeds**
(42/123/7) — an exact numerical replay of Experiment 10's own AR4-era
outcome (`0.000000`), not merely "small." Confirmed live, not dead wiring:
direct re-derivation from the raw tfevents trace shows real, nonzero,
non-degenerate transient firing early-to-mid training in every seed (peak
0.0087 in seed 42) that decays to exact zero by the end — the
contact-sensor/reward pipeline works, the policy converges away from the
signal. Behavioral: clean **0/24** sustained-lift, `max_height_gain_m`
≈0.0088m uniformly across all envs/seeds — the same physics-settle-noise
signature as this env cfg's already-established baseline and the
immediately-prior exploration-bonus null. Frame-by-frame video review
(seed 42, the seed with the largest transient mechanism signal) shows the
arm reaching down and holding one static pose for the entire post-settle
episode, no visible closure or lift distinguishable from rest.

**H_taskspace (Condition B, task-space/relative-IK
`DifferentialInverseKinematicsActionCfg`, `FrankaLiftEnvCfg`'s own existing
"ik-cube" recipe re-asserted after `FrankaDieLiftJointEnvCfg`'s joint-space
override): CONFIRMED, not falsified — but genuinely heterogeneous across
seeds, not a clean win.**

| seed | mechanism-level (final-100-iter mean) | behavioral | `max_height_gain_m` |
|------|------|------|------|
| 42 | 0.00023876 (barely passes 1e-4, ~2.4x) | 0/8 | ≈0.0088 (no-grasp baseline) |
| 123 | 0.83944721 (passes overwhelmingly, sustained not transient) | **8/8 clean sweep** | 0.306-0.403 (all 8 envs) |
| 7 | 0.00000012 (fails) | 0/8 | ≈0.0088 (no-grasp baseline) |

Per the spec's own pre-registered rule (falsified only if both bars fail in
*all* 3 seeds): seed 123 alone contributes a full 8/8 behavioral result and
a durable, non-transient mechanism signal, so both bars are NOT falsified —
**H_taskspace is CONFIRMED**. Seed 123's positive result was independently,
physically verified beyond the standard video-review bar: a live diagnostic
(`env.scene["object"].data.root_pos_w` vs. `panda_hand`'s `body_pos_w`,
printed every step) shows the object tracking the hand frame at a constant
≈0.10m offset in X/Y/Z across ~120 consecutive steps while both rise
together from ≈0.10m to ≈0.46m — the rigid-body signature of a genuine held
grasp, not a contact-solver artifact or an indexing bug, resolving a case
where the eval video's own rest/peak frames were visually too subtle to
read confidently by eye alone. Experiment 11's own AR4-era critic-divergence
risk (`Loss/value_function` exploding to ~5e23) was watched for throughout
all 3 seeds and **not observed** — seed 123 showed a real, bounded elevated
period (max 5.65 around iteration 270) that plateaued and fully recovered,
ordinary PPO value-loss variance, not the AR4-era blowup; the pre-authorized
`clip_actions=5.0` contingency was not needed and was not built.

**Combined outcome-matrix classification (spec's own 5-row table): Row 2 —
H_joint falsified, H_taskspace confirmed.** This is, by the letter of the
pre-registered rule, **an exact replay of the AR4-era Experiment 10→11
pattern, now demonstrated to transfer across platforms**: action-space
precision — not the antipodal mechanism itself, not a reward-calibration
problem — is the real gate on whether a geometrically-correct force-closure
signal ever becomes learnable at all, and this same gate now shows up on a
different robot (Franka vs. AR4), a different object (d8 die vs. cube), and
a from-scratch pure-`torch`/`ContactSensorCfg` reimplementation of the
mechanism, not a copy-pasted artifact of AR4's own code. This is a real,
notable finding in its own right, independent of anything else in this
result: the specific action-space-dependent behavior of a classical
antipodal/force-closure reward, previously observed on exactly one platform
(AR4), now demonstrably transfers to a structurally different arm.

**But the row-2 classification should not be read as a clean win, and this
report does not overstate it as one.** Unlike Experiment 11's own single-seed
AR4 report (0.018815 final, nonzero in 91.6% of iterations — a robust
positive on the one seed run), this experiment's 3-seed design exposes real
heterogeneity the AR4-era arc never had occasion to observe under this exact
mechanism: 1 of 3 seeds (123) gets a full, clean 8/8 grasp+lift with a
strong, durable mechanism signal; 1 of 3 seeds (42) gets a marginal,
barely-above-threshold mechanism signal with zero behavioral payoff — the
same mechanism-fires/behavior-doesn't shape as the sibling exploration-bonus
SPLIT result, at the single-seed level; 1 of 3 seeds (7) gets nothing at
all, indistinguishable from H_joint's own uniform null. Task-space control
is confirmed necessary for this mechanism to ever produce a usable signal on
Franka/d8 (H_joint's hard zero vs. H_taskspace's real positive rate proves
that much unambiguously) but is not by itself sufficient for reliable
discovery — 2 of 3 seeds under the *same* task-space condition still failed
behaviorally. "Task-space + antipodal grasp-quality solves d8" would
overstate this result; "task-space control is necessary for the antipodal
mechanism to ever become learnable on Franka/d8, and sufficient for
discovery in at least one seed, but not yet a reliable from-scratch recipe
across seeds" is the accurate reading.

**Relation to the exploration-bonus SPLIT result's own forward pointer:**
[[exploration-bonus-grasp-discovery]] named "verifying actual finger-object
contact/antipodal alignment at the moment of closure" as the next candidate
direction after its own SPLIT (reliable closure attempts near the object,
0/24 lift, all under joint-space control — the same action space as this
experiment's H_joint condition). H_joint's result **complicates that forward
pointer taken literally, but confirms its underlying diagnosis once
combined with a second change**: adding the antipodal-quality signal under
the *same* joint-space control the exploration-bonus experiment used does
not merely fail to produce a lift (as exploration-bonus's own bonus term
did) — it fails to even make the antipodal signal learnable at all, a harder
failure than exploration-bonus's own SPLIT. The forward pointer's implicit
framing (grasp-quality/antipodal alignment as *the* missing ingredient,
addressable on its own) was incomplete — it required an action-space change
as a co-requisite, not identified as necessary until this experiment's
dual-condition design surfaced it. Once that second change (task-space
control) is added, the forward pointer's underlying diagnosis is vindicated
in seed 123 specifically: the exact mechanism the exploration-bonus result
pointed toward is what closes the gap between "reliable closure attempt"
and "completed lift," genuinely producing this project's first-ever
sustained-lift discovery on d8 driven by an antipodal/force-closure reward
term. The honest combined reading: the exploration-bonus forward pointer
correctly identified antipodal alignment as the mechanistically relevant
next axis, but the fix that actually worked needed both that reward term
*and* a switch to task-space control together — neither alone was sufficient
(joint-space + antipodal = hard zero here; task-space alone, with no
antipodal term, is untested by this experiment and a different question).

**Relation to d8's own H2 success ([[d8-d10-demo-warmstart]]): two
independently-valid, non-competing mechanisms, not one story.** d8 is now
solvable via at least two structurally different, independently-verified
routes: H2's checkpoint warm-start (from the converged d12 specialist,
**3/3 seeds, a clean 24/24 sweep**, using `FrankaDieLiftJointD8BigEnvCfg`'s
plain joint-space recipe with no antipodal reward term at all) and this
experiment's task-space+antipodal combination (**1/3 seeds, 8/24 total**).
These are not in tension and this report does not force them into a single
ranked "which mechanism is real" resolution — they answer genuinely
different questions:
- **H2 answers a practical/engineering question**: given this project's
  existing dice-family curriculum (a converged specialist checkpoint for a
  geometrically-similar shape already exists), how do you reliably get PPO
  to discover d8's grasp at all? Answer: seed initialization near an
  already-solved basin, with no reward or action-space change whatsoever.
  This is currently the more reliable, cheaper, production-ready recipe for
  d8 specifically — but it is a shape-family-specific bootstrapping trick:
  it presupposes another shape's specialist already exists and says nothing
  about *why* cold-start joint-space training fails, nor anything that
  obviously transfers to a genuinely new shape/arm with no nearby specialist
  to warm-start from.
- **This experiment answers a mechanistic question**: does a real,
  physically-grounded antipodal/force-closure signal matter for grasp
  discovery at all, and if so, under what conditions does it become
  learnable? Answer: yes, mechanistically — but it is gated on
  action-space precision exactly as the AR4-era arc found, and even once
  unlocked, is not yet seed-reliable from scratch. This result is weaker in
  practical seed-reliability than H2's warm-start, but it is the more
  fundamental explanation of *why* from-scratch joint-space training fails
  here (imprecise positioning prevents antipodal contact from ever becoming
  learnable, independent of whether a warm start is available), and it is
  further evidence in favor of task-space/Cartesian action formulations —
  the action-space family CLAUDE.md's own North Star already favors for
  cross-arm/cross-task generalization, on independent grounds unrelated to
  this specific result.

Neither mechanism supersedes the other. If the practical goal is "get d8
working today," H2's warm-start is the stronger, cheaper, more reliable
recipe. If the goal is "understand why the mechanism fails/succeeds, in a
way likely to transfer to a new shape or a new arm with no existing
specialist to warm-start from," this experiment's task-space+antipodal
result is the more diagnostic finding, action-space-family-general rather
than shape-family-specific — even though it is not yet reliable enough on
its own to be a recommended default. Both are real, independently-verified
findings from this project's own history; nothing about this report is
served by picking a winner between them.

**No code bug found in either condition.** Both dispatches surfaced no
defect in Tasks 1-2's mechanism, instrumentation, or wiring. One bug was
found and immediately corrected in an independent-verification
reimplementation itself (a window-slicing error reproducing a known settle-
detection failure class from this project's own history), not in production
code.

**Cost:** H_joint (Task 3) ≈$1.6, H_taskspace (Task 4) ≈$1.1, plus Tasks
1-2's smoke tests — running total ≈$3.2 of the plan's shared $6 cap, not
exceeded. Full teardown verified after every provisioning cycle.

**Honest next candidate direction:** this result is not the dispositive
"both falsified" row that would close Direction 1 (contact/antipodal grasp
verification) for Franka/d8 — it is a genuine, if heterogeneous,
confirmation that the mechanism matters and transfers cross-platform once
action-space precision is available. The next well-motivated step, if this
axis is revisited, is investigating *why* 2 of H_taskspace's 3 seeds still
fail even with both fixes present (task-space control + antipodal reward)
— e.g., whether the surviving failure mode is itself an exploration problem
(same shape as the exploration-bonus SPLIT, now one level further in), a
residual positioning-precision gap even under task-space/IK control, or
something else — rather than assuming task-space+antipodal is now a
solved recipe just because it produced this project's first positive
antipodal-driven d8 lift. Not decided or started here — this is Task 5's
closing report, per the plan's own explicit "do not start any new
experiment" constraint; a decision for Principal on whether/how to pursue
that residual-seed question is a separate, later call. See
`kb/wiki/experiments/d8-antipodal-grasp-quality.md` for the full
per-seed/per-condition evidence and independent-verification detail.

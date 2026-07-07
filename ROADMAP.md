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

## Direction

Isaac-Lab-based robotics RL, expanding beyond AR4 manipulation into other
tasks/robots, object detection/perception, and mobility. No committed
roadmap items beyond AR4 yet — this is a stated direction, not a scoped
backlog.

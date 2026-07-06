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

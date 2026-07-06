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
2. Shape classifier misclassifies cube/rectangular-prism as "sphere" against
   real depth data. Root-caused: `PLANARITY_RESIDUAL_THRESHOLD` (tuned on
   near-noiseless synthetic data) doesn't generalize to real sensor noise.
   Circularity looks more promising as the primary signal, but real
   tilt/plane-fit readings were also noisy on small, low-pixel-count real
   objects — may need more than a threshold nudge.
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

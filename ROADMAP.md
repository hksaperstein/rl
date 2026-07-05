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
   results: `reaching_sphere` converges to ~0.92-0.93, but
   `lifting_sphere`/`sphere_goal_tracking`/`sphere_reached_goal` stay at
   exactly 0.0000 for the entire run in both cases. Real eval video
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
     Lee et al.'s stage-specific reward decomposition
     (arXiv:2512.10235); **(c)** as a last resort, hierarchical/frozen
     reach-policy-then-grasp-policy training, with the caveat (flagged in
     review) that a reach-only policy may converge to a non-grasp-feasible
     approach pose, forcing joint retraining anyway.
   - Note: the discrete/continuous gripper-action recommendation from the
     initial literature pass turned out to be moot — this env's gripper
     action is already binary open/close
     (`tasks/ar4/env_cfg.py:48`, `BinaryJointPositionActionCfg`).
   - Per the plan, further iteration beyond the one bounded fallback is
     deferred here rather than continued as open-ended tuning.
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

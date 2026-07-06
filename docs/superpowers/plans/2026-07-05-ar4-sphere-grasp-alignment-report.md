# AR4 sphere grasp-alignment experiment — report

Implements the design in
`docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-alignment-design.md`: a
multiplicatively-gated `grasp_sphere_aligned` reward that only credits
gripper closure when the sphere is genuinely centered between two new
fingertip `FrameTransformer` target frames — the follow-up fix to the
grasp-bonus experiment's reward-hacking failure (gripper closed *beside* the
sphere, not around it; see
`docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-bonus-report.md`).

## Code changes

### `tasks/ar4/pickplace_env_cfg.py` diff

```diff
--- a/tasks/ar4/pickplace_env_cfg.py
+++ b/tasks/ar4/pickplace_env_cfg.py
@@ -25,7 +25,9 @@ from isaaclab.utils.configclass import configclass
 
 from isaaclab_tasks.manager_based.manipulation.lift import mdp
 
+from . import mdp as ar4_mdp
 from .env_cfg import ActionsCfg, Ar4SceneCfg
+from .robot_cfg import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS
 
 # Empirically-tuned offset (m) from the link_6 frame to the gripper's jaw
 # pinch point along link_6's local +Z axis (ee_link sits at this same frame
@@ -42,10 +44,18 @@ class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
     ee_frame: FrameTransformerCfg = FrameTransformerCfg(
         prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
         debug_vis=False,
         target_frames=[
             FrameTransformerCfg.FrameCfg(
                 prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                 name="end_effector",
                 offset=OffsetCfg(pos=_EE_OFFSET),
             ),
+            FrameTransformerCfg.FrameCfg(
+                prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
+                name="finger_left",
+            ),
+            FrameTransformerCfg.FrameCfg(
+                prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
+                name="finger_right",
+            ),
         ],
     )
@@ -160,6 +170,18 @@ class RewardsCfg:
         weight=5.0,
     )
 
+    grasp_sphere_aligned = RewTerm(
+        func=ar4_mdp.aligned_grasp_bonus,
+        weight=10.0,
+        params={
+            "centering_std": 0.01,
+            "open_joint_pos": GRIPPER_OPEN_POS,
+            "object_cfg": SceneEntityCfg("sphere"),
+            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
+            "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
+        },
+    )
+
     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
 
     joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
```

### New file: `tasks/ar4/mdp.py` (fresh — prior experiment's version was reverted, not merged)

```python
# tasks/ar4/mdp.py
"""AR4-specific MDP reward terms not covered by Isaac Lab's generic
manipulation/lift mdp module.

Import conventions mirror Isaac Lab's own
isaaclab_tasks/manager_based/manipulation/cabinet/mdp/rewards.py.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def aligned_grasp_bonus(
    env: ManagerBasedRLEnv,
    centering_std: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus for closing the gripper only when the object is
    centered between the two fingertip frames.

    Multiplicatively gates the closure term by an alignment score (per
    GRIT's r_h*alpha_h pattern, arXiv:2604.04138) instead of the prior
    experiment's additive/independently-satisfiable combination, which
    let the policy collect the closure reward without ever positioning
    the object between the jaws (see
    docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-bonus-report.md).
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    finger_left_pos = ee_frame.data.target_pos_w[..., 1, :]
    finger_right_pos = ee_frame.data.target_pos_w[..., 2, :]
    finger_midpoint = (finger_left_pos + finger_right_pos) / 2.0

    centering_dist = torch.norm(object.data.root_pos_w - finger_midpoint, dim=-1)
    alignment_score = 1.0 - torch.tanh(centering_dist / centering_std)

    gripper_joint_pos = env.scene[gripper_asset_cfg.name].data.joint_pos[:, gripper_asset_cfg.joint_ids]
    closure_amount = torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

    return alignment_score * closure_amount
```

No other reward term (`reaching_sphere`, `lifting_sphere`,
`sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`) was touched —
single-variable experiment as specified.

## Prim-path hypothesis: confirmed correct, no correction needed

The spec flagged the `gripper_jaw1_link`/`gripper_jaw2_link` prim-path
hypothesis as unconfirmed (three standalone introspection attempts had hung
in the sandbox). The smoke test (Task 2) used this exact hypothesis
unmodified and **passed on the first attempt** — `FrameTransformerCfg`
resolved both new target frames without any invalid-prim-path error, so the
pattern-matched sibling-of-`link_6` naming convention was right. No
correction cycle, no debug-print fallback, was needed.

## Task 2: smoke test

`isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless`

- Exit code: 0
- No `Traceback` in the log
- Fresh checkpoints produced: `logs/train/2026-07-05_21-09-51/model_0.pt`,
  `model_1.pt`

## Task 3: full training run (num_envs=4096, 1500 iterations)

Run dir: `logs/train/2026-07-05_21-10-06/` (event file
`events.out.tfevents.1783300213.home.126429.0`, final checkpoint
`model_1499.pt`). Wall time: 17m53s. Exit confirmed via process no longer
present in `ps` plus the log's final `Learning iteration 1499/1500` block
and saved `model_1499.pt`; no `Traceback` anywhere in the 48.7k-line log.

Scalar trajectories pulled directly from the TensorBoard event file (1500
steps logged, one per iteration):

| Scalar | First 5 values (it. 0-4) | Last 5 values (it. 1495-1499) | Max (iteration) |
|---|---|---|---|
| `Episode_Reward/grasp_sphere_aligned` | 0.0, 8.7e-10, 1.9e-8, 2.7e-10, 9.9e-8 | 0.00159, 0.00157, 0.00142, 0.00147, 0.00142 | 0.00207 (it. 1415) |
| `Episode_Reward/lifting_sphere` | 0.0 x5 | 0.0 x5 | 0.00265 (it. 199) |
| `Episode_Reward/reaching_sphere` | 0.00015, 0.0013, 0.0017, 0.0024, 0.0043 | 0.7043, 0.7033, 0.7010, 0.7032, 0.7053 | 0.7095 (it. 1437) |
| `Episode_Termination/sphere_reached_goal` | 0.0 x5 | 0.0 x5 | 0.0226 (it. 36) |

**Key result: `grasp_sphere_aligned` never took off.** It stays essentially
at noise level (max 0.00207, roughly 0.7% of the term's theoretical max
~0.284 from full closure with perfect alignment) for the entire 1500-iteration
run — a sharp contrast with the prior grasp-bonus experiment, where the
unglated term saturated near its max (~0.284-0.287) by iteration ~1300.
`lifting_sphere` and `sphere_reached_goal` show the same early-exploration-
noise-then-decay-to-zero pattern seen in all three prior runs (peaks at
iteration <200, then flat 0.0 for the remaining ~1300 iterations).
`reaching_sphere` converges to ~0.70-0.71, similar to (slightly lower than)
prior runs' ~0.89-0.93 range — plausibly because the policy now spends more
of its noise budget on gripper-closure/alignment exploration that yields
near-zero reward, without ever discovering the payoff.

## Task 4: eval + video verification (10 episodes)

Ran `scripts/eval_loop.py --checkpoint logs/train/2026-07-05_21-10-06/model_1499.pt --episodes 10 --headless`.
Produced 10 videos (`logs/videos/ar4_pickplace-step-{0,250,500,750,1000,1250,1500,1750,2000,2250}.mp4`),
249 frames each (~5s episodes).

Extracted start (~0.2s), mid (~2.5s), and end (~4.7s) frames for all 10
episodes via `ffmpeg`, cropped tight on the gripper+sphere region and
upscaled 2x, plus a denser 10-sample close-up sequence (t=0.2s through
4.9s) for episode `step-0` to check whether the arm moves at all after its
initial approach.

**Per-episode verdict: 0 of 10 episodes show a real grasp+lift.** All 10
episodes checked (start/mid/end for all 10; dense intermediate sampling on
`step-0`) follow the same pattern:

1. Arm reaches down toward the sphere during the first ~1s (gripper visibly
   open in early frames, sphere clearly visible next to the fingers).
2. By ~t=1.0s the arm settles into a fixed pose and **does not move again for
   the rest of the episode** — the dense 10-frame sequence on episode
   `step-0` shows byte-identical arm geometry from t=1.0s through t=4.9s.
3. The sphere disappears from camera view once the gripper settles — it is
   occluded behind/under the gripper body from this camera angle, consistent
   with the gripper resting at or near the sphere's ground position, not
   with the sphere being lifted (a lift would move it into clearer view
   above the gripper, not hide it further).
4. `lifting_sphere` staying flat at 0.0000 across the entire eval-checkpoint's
   training history (Task 3 data) independently confirms no height gain ever
   occurred — the occlusion is a camera-angle artifact of a stationary
   gripper, not evidence of a successful grasp.

No episode shows the sphere elevated, carried, or displaced toward the
target region (visible in the frame or otherwise).

Episode-by-episode: 0/10 (step-0, step-250, step-500, step-750, step-1000,
step-1250, step-1500, step-1750, step-2000, step-2250 all show the identical
reach-then-freeze pattern with no motion in the last ~4 seconds).

## New-reward-hacking check

The task instructions specifically asked whether the alignment gate
introduced some *other* degenerate shortcut (e.g. `lifting_sphere` success
correlating with a non-centered contact). **No such pattern exists here,
because `lifting_sphere` never fires at all** — there is no successful
lift to check for a degenerate correlation in. More directly on the
gating mechanism itself: `grasp_sphere_aligned` never climbed above ~0.002
(vs. a ~0.284 max), so the policy never found *any* way to collect this
reward reliably, let alone a hacked one. Given the term is a product of
`alignment_score` (bounded [0,1], collapses toward 0 unless the sphere is
within roughly a centimeter of the finger midpoint) and `closure_amount`
(bounded [0, 0.014] per finger, so max ~0.028 summed), both factors would
have to be pushed toward their maxima simultaneously for the term to
saturate — and since the policy's exploration noise never discovered a
combination that did so, there's no exploit to diagnose. The multiplicative
structure did what it was designed to do (structurally closing the "closure
near, not around" exploit path) — but it accomplished this partly by making
the reward landscape sparser, not just more correctly-shaped.

## Overall verdict

**The alignment gate is not reward-hacked, but the fix did not produce a
grasp+lift either — 0/10, unchanged from every prior attempt.** This is a
different failure signature than the previous experiment: instead of the
policy confidently exploiting a loose reward (`grasp_sphere` saturating
near its max while lift stayed at 0), here `grasp_sphere_aligned` stays
near noise level for the entire run. The evidence points at a re-emergence
of the **original exploration failure** identified in the very first sphere
experiment (`ROADMAP.md`'s "grasp/lift never emerges" entry): the policy
converges to a static reach-and-freeze pose within the first second of every
episode and never attempts a deliberate close-around-the-object motion at
all. The video evidence for this run (identical arm geometry from t=1.0s to
episode end, across all 10 episodes) is qualitatively the same "static open/
settled pose, no closing attempt" signature as the very first (pre-any-
grasp-reward) experiment, not the "closes but misses" signature of the
second experiment.

Read together with the two prior falsified hypotheses, this is now a
**third falsified dense-reward-shaping-only hypothesis**: (1) lift-weight
bump — no-op; (2) additive proximity+closure grasp bonus — reward-hacked;
(3) multiplicatively-gated alignment+closure grasp bonus — structurally
un-hackable but too sparse to ever be discovered by unguided exploration.
Making the reward correct (requiring true centering) came at the direct
cost of making it harder to stumble into during random exploration — the
1cm `centering_std` window is tight relative to the sphere's own ~9mm
radius and the fingers' small travel range, so a randomly-exploring policy
essentially never produces the joint (position, orientation, closure)
combination needed to get any signal from this term, whereas the previous
term's 4cm proximity threshold was loose enough to find by chance.

Per `superpowers:systematic-debugging` Phase 4.5 and the design doc's own
escalation ordering, this confirms the recommended next step is **no longer
a fourth reward-shaping-only tweak** — the spec's own fallback
(Isaac Lab's `ContactSensor`/`contact_forces` infrastructure, confirmed
available and installed) or a curriculum/staged-reward approach (reach-only
-> reach+close-gripper bonus with a *looser* discovery-friendly threshold
-> tightened alignment requirement introduced only after closure-near-object
is already a well-established behavior) are better next candidates than
further tuning of a single-shot dense term that must be correct and
discoverable simultaneously.

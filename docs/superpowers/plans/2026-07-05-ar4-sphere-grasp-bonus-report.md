# AR4 sphere grasp-bonus experiment — report

Implements the design in
`docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-bonus-design.md`: a dense
`grasp_sphere` reward term, adapted from Isaac Lab's `manipulation/cabinet`
task's `grasp_handle` reward, to give the policy credit for closing the
gripper near the sphere (the missing precursor behavior identified in
`ROADMAP.md`'s "grasp/lift never emerges" follow-up).

## Preliminary: worktree was stale

Before touching any code, `git log`/`git merge-base` showed this worktree's
branch (`worktree-sphere-grasp-bonus`) was rooted at commit `a8df301`, an
ancestor of `main` missing 9 commits — including the actual Cube→Sphere
retargeting (`79c9089`) and lift-weight bump (`d9e7a6c`) that this experiment
spec depends on. `tasks/ar4/pickplace_env_cfg.py` on disk still had
`reaching_cube`/`lifting_cube`/etc. Since the branch had zero commits of its
own beyond that ancestor point, I fast-forwarded it to `main`
(`git merge main --ff-only`), which brought in the sphere-based reward names
the spec and task instructions reference. No divergent work was lost — this
was a clean fast-forward.

## Code changes

### New file: `tasks/ar4/mdp.py`

```python
# tasks/ar4/mdp.py
"""Local MDP reward terms for the AR4 pick-and-place task that don't exist
in any of Isaac Lab's built-in `mdp` modules.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def grasp_object_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus for closing the gripper while near an object.

    Bootstraps grasp-attempt exploration: adapted from isaaclab_tasks'
    manipulation/cabinet task's grasp_handle reward (identical
    is_close * sum(open - current) pattern), generalized from a fixed
    cabinet-handle frame to any object_cfg/ee_frame_cfg pair.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[gripper_asset_cfg.name].data.joint_pos[:, gripper_asset_cfg.joint_ids]

    distance = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)
```

### `tasks/ar4/pickplace_env_cfg.py` diff

```diff
--- a/tasks/ar4/pickplace_env_cfg.py
+++ b/tasks/ar4/pickplace_env_cfg.py
@@ -26,7 +26,9 @@ from isaaclab.utils.configclass import configclass
 
 from isaaclab_tasks.manager_based.manipulation.lift import mdp
 
+from . import mdp as ar4_mdp
 from .env_cfg import ActionsCfg, Ar4SceneCfg
+from .robot_cfg import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS
 
 # Empirically-tuned offset (m) from the link_6 frame to the gripper's jaw
 # pinch point along link_6's local +Z axis (ee_link sits at this same frame
@@ -160,6 +162,18 @@ class RewardsCfg:
         weight=5.0,
     )
 
+    grasp_sphere = RewTerm(
+        func=ar4_mdp.grasp_object_bonus,
+        weight=10.0,
+        params={
+            "threshold": 0.04,
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

No other reward term (`reaching_sphere`, `lifting_sphere`, `sphere_goal_tracking`,
`sphere_goal_tracking_fine_grained`) was touched — single-variable experiment
as specified.

One environment note unrelated to the code change: this fresh worktree had no
generated USD assets (`assets/` is gitignored). The smoke test's first attempt
failed on a missing `assets/shapes/wedge.usd` — not a code bug. Ran
`scripts/build_asset.py` with `AR4_DESCRIPTION_PATH` pointed at the existing
`annin_ar4_description` checkout to regenerate the assets locally, then
re-ran the smoke test.

## Task 2: smoke test

`isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless`

- Exit code: 0
- No `Traceback` in the log
- Fresh checkpoints produced: `logs/train/2026-07-05_19-09-20/model_0.pt`,
  `model_1.pt`

Passed cleanly on the second attempt (first attempt's failure was the
missing-asset issue above, not a code bug).

## Task 3: full training run (num_envs=4096, 1500 iterations)

Run dir: `logs/train/2026-07-05_19-09-37/` (event file
`events.out.tfevents.1783292983.home.84239.0`, final checkpoint
`model_1499.pt`). Wall time ~14m47s.

Scalar trajectories (1500 steps logged, one per iteration):

| Scalar | First 5 values | Last 5 values (it. 1495-1499) | Max (iteration) |
|---|---|---|---|
| `Episode_Reward/grasp_sphere` | ~0, 1.5e-6, 5.5e-6, 8.1e-7, 6.7e-5 | 0.2848, 0.2846, 0.2846, 0.2846, 0.2844 | 0.2865 (it. 1311) |
| `Episode_Reward/lifting_sphere` | 0.0, 0.0, 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0, 0.0, 0.0 | 0.0019 (it. 10) |
| `Episode_Termination/sphere_reached_goal` | 0.0 x5 | 0.0 x5 | 0.0199 (it. 47) |
| `Train/mean_reward` | 0.003, 0.012, 0.015, 0.018, 0.046 | 5.90, 5.90, 5.86, 5.88, 5.87 | 5.95 (it. 1255) |
| `Episode_Reward/reaching_sphere` (context) | ~0.0003 -> 0.0085 | 0.894, 0.894, 0.891, 0.892, 0.894 | 0.904 (it. 1252) |
| `Episode_Reward/sphere_goal_tracking*` | 0.0 x5 | 0.0 x5 | ~0 (noise only) |

**Key result:** `grasp_sphere` climbed from ~0 and **saturated near its
theoretical max (~0.284-0.287) well before the end of training**, staying
there for the rest of the run — i.e. the policy fully and reliably learned
to close the gripper whenever near the sphere; the new reward term did
exactly what it was designed to do, mechanically. However
`lifting_sphere` and `sphere_reached_goal` **never moved off zero in any
sustained way** — both show only tiny early-exploration blips (0.0019 at
iteration 10, 0.0199 at iteration 47) that decay to exactly 0.0 and stay
there for the remaining ~1450 iterations. `Train/mean_reward` plateaus at
~5.9, consistent with `reaching_sphere` (~0.89) + `grasp_sphere` (~0.28) +
small penalties and nothing else — no lift/goal-tracking reward is ever
collected.

## Task 4: eval + video verification (10 episodes)

Ran `scripts/eval_loop.py --checkpoint logs/train/2026-07-05_19-09-37/model_1499.pt --episodes 10 --headless`.
Produced 10 videos (`logs/videos/ar4_pickplace-step-{0,250,500,750,1000,1250,1500,1750,2000,2250}.mp4`),
249 frames each (~5s episodes at the configured decimation).

Extracted start/mid/end frames (and additional intermediate frames for one
episode to check finger state specifically) via `ffmpeg`, cropped tight on
the gripper+sphere region, and visually inspected all 10 episodes (mid frame
for all 10, start+end frames for a representative subset, plus a 6-frame
close-up sequence on one episode to check finger open/close state over
time).

**Per-episode verdict: 0 of 10 episodes show a real grasp+lift.** All 10
follow an essentially identical pattern:
1. Arm approaches the sphere from its reset pose (gripper visibly open,
   spread fingers with a visible gap in early frames).
2. Gripper settles right next to/against the sphere and the finger gap
   closes (fingers together) — consistent with `grasp_sphere` reward being
   collected.
3. The sphere remains resting on the ground, immediately adjacent to the
   closed gripper, for the entire rest of the episode (mid-frame and
   end-frame positions are visually indistinguishable from each other in
   every episode checked).
4. No episode shows the sphere elevated, carried, or displaced toward the
   target region.

This is corroborated by `Metrics/object_pose/position_error` staying flat at
~0.32-0.33 for the whole run (no improvement) and by `sphere_goal_tracking*`
staying at 0.0.

## Reward-hacking check

The spec explicitly flagged this risk, and the evidence points at exactly
that failure mode, not a physical grip/slip failure: the closer look at
finger state (6-frame sequence within one episode) shows the fingers visibly
open with a gap while approaching (t~30, pre-contact) and then closed with no
visible gap once settled (t~60 onward) — but the sphere is positioned
*beside* the closed gripper's tip, not *between* the jaws. The
`grasp_object_bonus` reward as specified only checks distance from the EE
frame's origin to the object's root position (`threshold=0.04`) and gripper
joint closure — it has no notion of whether the object is actually enclosed
between the fingers. The policy discovered and fully exploited this: get the
EE frame origin within 4cm of the sphere center (which the existing
`reaching_sphere` term already drives it to do) and close the gripper fully,
collecting the `grasp_sphere` bonus every step, regardless of whether the
fingers are actually positioned to capture the sphere. There is no evidence
of chattering/oscillating open-close (the reward is flat/saturated, not
noisy), and no evidence of the sphere being nudged or dragged along the
ground — it stays put once contacted.

## Overall verdict

**Partial outcome, and not the hoped-for one: the grasp-bonus term is fully
learned but does not produce the target behavior.** It successfully solved
the sub-problem it was designed for (closing the gripper near the object —
`grasp_sphere` saturates at its max well before training ends), which
confirms the earlier diagnosis that lack of any closing-gripper credit was
part of the problem. But it did not close the actual gap: `lifting_sphere`
and `sphere_reached_goal` are indistinguishable from the pre-fix baseline and
lift-weight-bump runs (both exactly 0.0 after early noise decays). Per the
design doc's own framing, this is a second falsified hypothesis (after the
lift-weight bump) for a *dense-shaping-only* fix — the newly-added evidence
specifically implicates the grasp-bonus term's own position-only distance
check as insufficiently strict: it rewards proximity + closure without
requiring the object to be positioned between the jaws, so the policy
satisfies it without ever attempting a geometrically correct grasp.

This matches the literature review's original priority-(a) recommendation
(contact-based reward, e.g. `ContactSensorCfg` on the gripper fingers, or at
minimum a stricter geometric check — e.g. requiring the sphere to be between
the two finger positions, similar to the cabinet task's own
`align_grasp_around_handle`/`approach_gripper_handle` pattern rather than
just `grasp_handle`'s bare distance check) — per
`superpowers:systematic-debugging`'s Phase 4.5, this is grounds to escalate
rather than attempt a third dense-reward-only tweak.

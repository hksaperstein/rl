# Experiment 13: residual RL over a classical waypoint-seeking base controller

## Context

Experiment 12 fixed a real reward-rate bug (grasp-and-freeze incentive) but
the result was inconclusive: scalars mixed, and video inspection of 3/10
eval episodes showed no lift in any of them — one episode showed the arm
never even engaging the cube at all
(`docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md`,
ROADMAP.md). Eleven prior experiments (9-12, and the earlier sphere-task
saga) have all worked within the same paradigm: a flat, end-to-end policy
whose raw action *is* the entire per-step motor command, shaped toward the
full pick-and-place sequence purely through reward engineering. Per this
project's own standing practice (`CLAUDE.md`'s "generate genuinely new
directions, not just refinements" mandate), an inconclusive result after
this many reward-shaping attempts on the same paradigm is grounds to try a
structurally different approach, not another weight/threshold tweak.

## What's different about this attempt

This repo's environment already computes everything needed for a classical
per-step target during pick-and-place: `compute_path_waypoints` (event,
`tasks/ar4/mdp.py:372-422`) produces a 5-waypoint Cartesian path
(pregrasp/grasp/lift/transit/place) every episode reset, and
`env._path_waypoint_idx` already auto-advances monotonically as the
end-effector gets within `advance_tolerance` of the active waypoint (the
same mechanism `path_proximity_bonus` and `ik_guided_path_bonus` already
use for reward shaping). This is a fully classical, deterministic
geometric solution to *where the gripper should go next* — it has never
been used to actually drive the arm, only to shape a reward the policy has
to discover the equivalent of from scratch.

## Literature grounding

Two independently-verified sources support using this existing classical
path as a base controller rather than only a reward signal:

- **Silver et al., "Residual Policy Learning" (arXiv:1812.06298, verified
  via `ar5iv.labs.arxiv.org/html/1812.06298`)**: combines a base controller
  and a learned residual by plain additive superposition in the same
  action space, `π_θ(s) = π_base(s) + f_θ(s)`. Tested on 6 MuJoCo
  manipulation environments with partial observability/model
  misspecification; explicitly reports RPL "can perform long-horizon,
  sparse-reward tasks for which reinforcement learning alone fails" and
  "consistently and substantially improves on the initial controllers."
  Notes DDPG+HER was used, and flags a real caveat: an early-training
  performance dip from poor critic initialization relative to the actor,
  mitigated with a burn-in period — relevant here since this repo's own
  Experiment 11 saw a real critic-divergence bug under a similarly-new
  action term, so this is not a new risk class, just one to watch again.
  Also reports the gain is not universal (comparable to from-scratch RL on
  some of their 6 tasks) — the paper's own caveat this design should not
  overclaim past.
- **Johannink et al., "Residual Reinforcement Learning for Robot Control"
  (ICRA 2019, verified via search result abstracts)**: "decomposes the
  problem into a part solved by conventional feedback control and a
  residual solved with RL, with the final control policy being a
  superposition of both control signals" — demonstrated on a real-robot
  block-insertion task. This is the closer structural match to what's
  proposed below: a genuine classical feedback controller (not just a
  demonstration/imitation target) plus an RL correction on top.
- A third source (PMC10296071, Robosuite/Panda pick-and-place, SAC)
  supports the adjacent idea of staged task decomposition (scripted
  grasp-close between two learned reach legs), achieving 93.2% success vs.
  reported end-to-end baselines under 80% (and as low as <20% on harder
  variants) — corroborating evidence that *some* form of injecting
  structure beats pure end-to-end RL on this task class, even though the
  specific design below is residual-RL rather than full staging.

## Design

New file `tasks/ar4/pickplace_residual_env_cfg.py`
(`Ar4PickPlaceResidualEnvCfg`), additive/parallel to every other
`pickplace_*.py` file per this repo's established convention — reuses
`Ar4PickPlaceMirrorSceneCfg` (same scene) and Experiment 12's exact reward
weights (`path_proximity_bonus` 25.0, `gripper_schedule_bonus` 0.1,
`antipodal_grasp_bonus` 3.0, `stillness_penalty` 5.0, `action_rate`
-1e-4, `joint_vel` -1e-4 — unchanged from
`pickplace_taskspace_env_cfg.py`), isolating the action-space variable
specifically, the same way Experiment 11 isolated action space against
Experiment 10's reward/physics fixes.

**New action term**: `ResidualDifferentialIKAction`
(`ResidualDifferentialIKActionCfg`), in a new file
`tasks/ar4/residual_ik_action.py` (a class, not a bare reward/observation
function — doesn't belong in `mdp.py`, which is functions only; this keeps
`mdp.py`, already 650+ lines, from growing further). Subclasses Isaac
Lab's `DifferentialInverseKinematicsAction`
(`isaaclab/envs/mdp/actions/task_space_actions.py:31`), overriding only
`process_actions()`:

```python
def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    base_delta = self._compute_base_delta()  # (num_envs, 3), body frame
    self._processed_actions[:] = base_delta + self.raw_actions * self._scale
    if self.cfg.clip is not None:
        self._processed_actions = torch.clamp(
            self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        )
    ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
    self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
```

`_compute_base_delta()` (new private method) implements a bounded pursuit
step toward the currently-active waypoint, reusing the exact
waypoint-lookup and world-to-body-frame transform `ik_guided_path_bonus`
already performs (`tasks/ar4/mdp.py:474-513`):

```python
def _compute_base_delta(self) -> torch.Tensor:
    env = self._env
    if not hasattr(env, "_path_waypoints_w"):
        return torch.zeros(self.num_envs, 3, device=self.device)
    current_waypoint_w = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    root_pose_w = self._asset.data.root_pose_w
    target_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], current_waypoint_w)
    ee_pos_curr, _ = self._compute_frame_pose()
    direction = target_b - ee_pos_curr
    dist = torch.norm(direction, dim=-1, keepdim=True)
    step = torch.clamp(dist, max=_BASE_MAX_STEP)
    return direction / (dist + 1e-8) * step
```

`_BASE_MAX_STEP = 0.05` — deliberately identical to the existing
`scale=0.05` already used for the policy's own raw-action contribution, so
base and residual are comparably-sized "collaborators" (neither dominates
the other by construction), and bounded so the base controller never
overshoots a waypoint in one step (`torch.clamp(dist, max=...)`, not a
fixed-magnitude step). This is a plain proportional ("seek") controller,
not a second differential-IK solve — the existing `DifferentialIKController`
inside `DifferentialInverseKinematicsAction.apply_actions()` (unchanged,
not overridden) still does the actual joint-space conversion, exactly as
today; only the *Cartesian delta fed into it* changes from "raw policy
output only" to "base pursuit step + raw policy output."

`ResidualDifferentialIKActionCfg` subclasses
`isaaclab_mdp.DifferentialInverseKinematicsActionCfg` with
`class_type: type[ActionTerm] = ResidualDifferentialIKAction` as its only
override — same fields otherwise (`asset_name`, `joint_names`,
`body_name`, `body_offset`, `scale=0.05`, `controller` with
`use_relative_mode=True`, `ik_method="dls"`), all copied verbatim from
`pickplace_taskspace_env_cfg.py`'s `ActionsCfg`.

**Gripper action unchanged**: `gripper_position` stays a plain
`BinaryJointPositionActionCfg`, policy-controlled, not given a base
controller. Rationale: `gripper_schedule_bonus`'s own scalar (Experiment
11: final 0.079; Experiment 12: final 0.077) shows gripper-timing is
already being learned reasonably well — the diagnosed gap is specifically
in the *arm's* ability to execute the lift/carry/place motion, so the
residual treatment is scoped to the dimension where the evidence points,
not applied uniformly out of convenience.

**PPO runner config**: reuse `Ar4PickPlaceTaskspacePPORunnerCfg` unchanged
(`clip_actions=5.0`, already-verified fix from Experiment 11/12) — no new
PPO hyperparameters, isolating the action-term variable alone.

## What this does NOT change

- No reward function changes (Experiment 12's weights carry over exactly).
- No episode-length change (queued separately, deprioritized in
  Experiment 12's ROADMAP entry — the arm has substantial unused episode
  time already, so length isn't the diagnosed bottleneck).
- No change to `pickplace_taskspace_env_cfg.py`,
  `pickplace_mirror_env_cfg.py`, `pickplace_ik_guided_env_cfg.py`,
  `env_cfg.py`, or `objects_cfg.py` — purely additive, per this repo's
  established per-experiment-file convention.
- `apply_actions()` (the actual IK solve + joint command) is inherited
  unchanged from `DifferentialInverseKinematicsAction` — only
  `process_actions()` is overridden.

## Verification plan

Same sequence as Experiment 12 (diagnostic run gate, full run, video
inspection, ROADMAP record) — this repo's established pattern for every
experiment this session:

1. Syntax-check + a short smoke test (16 envs, 2 iterations, matching
   every prior experiment's smoke-test convention) verifying the new
   action term constructs without exceptions and produces the expected
   checkpoint files, before committing to a real-scale run.
2. 300-iteration diagnostic (`num_envs=4096`) checking: no traceback, and
   `Loss/value_function` stays bounded (the specific risk Silver et al.
   flag for a newly-introduced base-controller term, and the exact failure
   class Experiment 11 hit under a different new action term) — if this
   regresses, stop and report rather than proceeding, exactly as
   Experiment 12's Task 2 gate did.
3. Full 1500-iteration run + TensorBoard report, comparing final values
   against Experiment 12's exact numbers (`antipodal_grasp_bonus`
   0.012777, `path_proximity_bonus` 0.064421, `cube_reached_goal`
   0.010773, `stillness_penalty` -0.001857) — final-snapshot-vs-final-
   snapshot only, per this project's established correction protocol.
4. Eval + personal video inspection of multiple episodes (not just one,
   per the lesson from Experiment 12's own video review: with a ~1%
   success rate, a single episode isn't a representative sample) —
   specifically checking for genuine lift-to-height and carry-to-goal
   motion, which is the concrete behavioral bar Experiments 11 and 12 both
   failed to clear.
5. ROADMAP record regardless of outcome.

## Success criteria

Not "full pick-and-place solved" — the bar is the same one Experiment 12
didn't clear: observable lift-off-the-ground and carry-toward-goal motion
in a meaningful fraction of inspected eval episodes, and/or a
`cube_reached_goal` termination rate clearly above the ~1% baseline
established across Experiments 11-12. A result that still shows no lift
would be informative too — it would suggest the bottleneck is not
"exploration is too hard to find the geometric path" (which residual RL
over a correct path should directly fix) but something else entirely
(e.g., a genuine physical/contact-stability limit on maintaining grasp
during motion), narrowing the hypothesis space for whatever comes next.

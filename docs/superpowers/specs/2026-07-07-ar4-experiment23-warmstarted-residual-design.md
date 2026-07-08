# Experiment 23: residual RL over a classical waypoint controller, with literature-grounded warm-start

## Hypothesis

**Six consecutive experiments (17-22) have each tried a variant of the
same technique family — tweak the reward or the joint-space action
space, and hope pure PPO exploration discovers the compound grasp+lift
behavior — and each has narrowed *where* the failure occurs without
resolving it. Per this repo's own mandate to default to a structurally
different strategy after a string of nulls, this experiment revisits a
genuinely different paradigm already partially tried in this repo's
history: residual RL over a classical waypoint-pursuit base controller
(Experiment 13). That attempt produced a clean regression with a
precisely diagnosed cause — the residual was never warm-started, so an
untrained, effectively-random residual fought a committed classical
base-controller step every single step from iteration 0, exactly the
failure mode Johannink et al. 2019 warn against and explicitly solve
for. Implementing that specific, literature-grounded fix — ramping the
residual's authority from 0 to full over an initial training window,
rather than the full 1.0 authority from step 0 — combined with the
current, more-refined reward configuration (Experiment 22's exact
reward set, not Experiment 12's superseded weights) should let the
residual mechanism actually work as the literature describes, rather
than repeating Experiment 13's diagnosed implementation gap.**

Falsifiable: if `Episode_Reward/lifting_object` still stays at exactly
`0/1500` despite the warm-start being verified as actually ramping
(instrumented confirmation the residual's authority factor increases
from 0 to 1.0 over the configured window, not just asserted), this
specifically falsifies "the warm-start gap explains Experiment 13's
regression and blocks the residual mechanism from working" — the
classical-base-plus-residual paradigm itself would then be a more
fundamental non-fit for this task, not an implementation-gap problem,
pointing toward demonstration/imitation bootstrapping (which requires
either human teleoperation or a from-scratch expert-controller
pipeline, a larger undertaking) as the next candidate.

## Background research

Grounded in this repo's own prior verified evidence and citations
(already independently verified in Experiment 13's own design spec,
`docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md`,
re-cited here rather than re-verified from scratch, per this repo's
"don't over-research-loop" practice — these are the same citations,
not new claims):

- **Johannink et al., "Residual Reinforcement Learning for Robot
  Control" (ICRA 2019)**: "decomposes the problem into a part solved by
  conventional feedback control and a residual solved with RL... holds
  the residual fixed at zero for an initial period while training only
  the value function, allowing for a good estimate of the value of the
  base controller before learning begins." This experiment implements
  an engineering approximation of that idea — a gradual ramp of the
  residual's authority from 0 to 1.0 over an initial window, rather
  than a hard zero-then-full switch with value-function-only
  pretraining (which would require modifying rsl_rl's own PPO training
  loop internals to decouple policy/value updates, a much larger and
  more invasive change). The ramp is a genuine, if softer,
  implementation of the same underlying goal: an untrained residual
  should not immediately fight a committed base-controller step at full
  authority. Flagged explicitly as an approximation, not a literal
  reproduction, so this distinction isn't lost in a future retrospective.
- **Silver et al., "Residual Policy Learning" (arXiv:1812.06298)**:
  already-verified additive-superposition design (`π = π_base + f_θ`),
  the structural pattern `ResidualDifferentialIKAction` already
  implements, carried over unchanged from Experiment 13.
- **This repo's own Experiment 13 report**
  (`docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md`,
  ROADMAP.md): confirmed no critic divergence (`Loss/value_function` max
  0.17, healthier than the plain-action baseline it was compared
  against) — ruling out the specific failure class Experiment 11 hit
  under a different new action term. The regression was specifically a
  *behavioral* one (video evidence of ongoing instability, a
  `stillness_penalty` move in the wrong direction), consistent with the
  diagnosed warm-start gap, not a sign the base mechanism itself is
  broken.

## Design

**Reused unchanged**: `compute_path_waypoints` (`tasks/ar4/mdp.py:404`)
— already reads `env._target_pos_w` (set by `set_mirrored_goal`, the
current scene's own goal-randomization event), fully compatible with
the current `Ar4PickPlaceMirrorSceneCfg` lineage despite being written
for the older `pickplace_ik_guided_env_cfg.py`. No changes needed.

**New action term**, appended to `tasks/ar4/actions.py`:

```python
class WarmStartedResidualDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Residual RL over a classical waypoint-pursuit base controller,
    with a literature-grounded warm-start (Johannink et al. 2019):
    the residual's authority ramps linearly from 0 to 1.0 over
    cfg.warmup_steps environment steps, rather than contributing at
    full strength from iteration 0 - the specific gap Experiment 13's
    own diagnosed regression identified. Also performs the waypoint
    auto-advance side effect (env._path_waypoint_idx increments when
    the end-effector comes within cfg.advance_tolerance of the active
    waypoint) directly, rather than reusing ik_guided_path_bonus's
    bundled reward+advance logic - keeps this experiment's only new
    variable the action space, isolated from any reward change. See
    docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.
    """

    cfg: WarmStartedResidualDifferentialIKActionCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._step_count = 0

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
        step = torch.clamp(dist, max=self.cfg.base_max_step)

        # Waypoint auto-advance side effect (moved here from
        # ik_guided_path_bonus, which this experiment does not reuse).
        ee_pos_w = self._env.scene[self.cfg.ee_frame_cfg.name].data.target_pos_w[:, 0, :]
        dist_to_waypoint_w = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
        reached = dist_to_waypoint_w < self.cfg.advance_tolerance
        env._path_waypoint_idx = torch.where(
            reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
        )

        return direction / (dist + 1e-8) * step

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        base_delta = self._compute_base_delta()
        residual_authority = min(1.0, self._step_count / self.cfg.warmup_steps)
        self._processed_actions[:] = base_delta + residual_authority * self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        self._step_count += 1


@configclass
class WarmStartedResidualDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = WarmStartedResidualDifferentialIKAction
    ee_frame_cfg: SceneEntityCfg = MISSING
    base_max_step: float = 0.05
    advance_tolerance: float = MISSING
    warmup_steps: int = MISSING
```

`warmup_steps = 1200` (50 iterations × `num_steps_per_env=24`, confirmed
from `tasks/ar4/agents/rsl_rl_ppo_cfg.py` — not guessed): a small
fraction (~3.3%) of the full 1500-iteration/36,000-step training
budget, giving the critic real initial training time before the
residual starts influencing behavior, without consuming a large share
of the overall budget. `advance_tolerance = 0.03` (3cm, matching the
scale of `minimal_height`/other proximity thresholds already used
throughout this repo's reward functions, e.g. `antipodal_grasp_bonus`'s
`minimal_height=0.03`).

**New EventCfg**, adding `compute_path_waypoints` to Experiment 22's
existing events (`reset_all`, `reset_cube_position`, `randomize_goal`),
registered last (after both cube-position and goal reset, per the
function's own documented ordering requirement).

**New env cfg** `tasks/ar4/pickplace_warmresidual_env_cfg.py`
(`Ar4PickPlaceWarmResidualEnvCfg`): reuses Experiment 22's exact
`RewardsCfg`, `ObservationsCfg`, `TerminationsCfg`, `CurriculumCfg`
unchanged (isolating the action-space change as the only new variable
relative to the current best-performing reward lineage). Gripper action
reuses `MirroredGripperActionCfg` unchanged (Experiment 22's current
best gripper mechanism, not reverted to Experiment 13's plain gripper).
Arm action replaced with `WarmStartedResidualDifferentialIKActionCfg`.

**PPO runner config**: reuse `Ar4PickPlaceTaskspacePPORunnerCfg`
(task-space action family, matching the existing `--taskspace`/
`--residual`/`--reachskip`/`--baseproximity` selection condition in
`scripts/train.py` — this new flag must be added to that condition,
the exact mistake pattern this repo has repeatedly had to verify
against).

## What this does NOT change

No reward-function changes (Experiment 22's exact `RewardsCfg` carries
over). No change to the gripper mechanism (Experiment 22's
`MirroredGripperActionCfg` carries over unchanged). No change to
`tasks/ar4/robot_cfg.py` or `scripts/build_asset.py`. Does not modify
`compute_path_waypoints`, `ResidualDifferentialIKAction` (Experiment
13's original, kept as-is for its own historical record), or any
existing env cfg file — purely additive.

## Verification plan

Smoke test, then a dedicated **warm-start verification step** before
the standard diagnostic: an instrumented rollout confirming
`residual_authority` actually ramps from ~0 at step 0 to 1.0 by step
1200, not asserted from the formula alone (direct inspection of the
action term's own computed value each step, matching this session's
established pattern of verifying mechanism claims before trusting
downstream training results). Then: 300-iteration diagnostic (checking
`Loss/value_function` stays bounded — the specific risk both Silver et
al. and Experiment 11's own history flag for a new action term), full
1500-iteration run, TensorBoard report comparing against Experiment
22's exact final values, and the same Task-6-style instrumented contact
diagnostic used after Experiments 20-22.

## Success criteria

Primary: does `Episode_Reward/lifting_object`'s nonzero rate move off
Experiment 22's exact `0/1500` — prerequisite: the warm-start mechanism
itself must be verified actually ramping, otherwise a null result
would be uninterpretable (confounded by whether Experiment 13's
diagnosed gap was actually fixed). A null result with a verified-working
warm-start would specifically falsify "the warm-start gap explains
Experiment 13's regression" — narrowing rather than repeating the open
question, and pointing toward demonstration/imitation bootstrapping
(the remaining major untried technique family) as the next direction.

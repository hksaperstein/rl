# Experiment 14: reach-skip curriculum (arm starts at a computed pregrasp pose)

## Context

Three experiments in a row (11, 12, 13) diagnosed the same specific gap:
grasp-contact is reliably achievable (Experiment 11's antipodal grasp
bonus, sustained in 91-93% of iterations across 11-13), but the policy
never progresses to lift+carry+place. Experiment 12 (reward-rate fix) was
inconclusive. Experiment 13 (residual RL over a classical controller) was
a genuine regression, and a separate classical (zero-RL) demo built this
session independently confirmed that even a pure classical controller
driving straight-line Cartesian pursuit toward the cube can stall on
kinematic difficulties — two independent pieces of evidence that keep
landing on the same underlying problem class from different angles.

Per this project's standing mandate (CLAUDE.md: "after any string of
failed/null experiments, explicitly ask whether the next attempt is a
structurally different strategy or just another parameter tweak") — this
is the third non-improving result in the same broad family (flat
end-to-end policy, reward-shaped, differing only in action space or
reward weights). Time to pivot structurally, not tune again.

## Design

**Core idea**: every experiment so far makes the policy solve the entire
reach → grasp → lift → carry → place sequence from a fixed home pose,
every episode, even though the diagnosed gap has been specifically
*post*-grasp for three experiments running. Reach itself is not the
unsolved sub-problem — it's the well-trodden, reliably-learned part
(confirmed by `path_proximity_bonus`'s consistently high values and
`antipodal_grasp_bonus`'s consistent non-zero rate across Experiments
11-13). Removing it from what the policy has to (re-)discover every
episode reallocates the *entire* step budget and exploration effort to
the actually-unsolved sub-problem, without touching the reward function
or action space that reach+grasp already learned to use.

**Mechanism**: a new reset `EventTerm`,
`reset_arm_to_pregrasp_pose(env, env_ids, object_cfg, robot_cfg,
pregrasp_hover, gripper_tool_offset)`, added to `tasks/ar4/mdp.py`. Runs
*after* the cube's position is randomized (so it computes IK against the
cube's real, just-randomized position for this episode) and *before*
`compute_path_waypoints`. Reuses the exact live-`DifferentialIKController`
construction and gripper-tool-offset correction already proven in
`ik_guided_path_bonus` (`tasks/ar4/mdp.py:463-513`) — a **one-shot** solve
(not per-step, unlike `ik_guided_path_bonus`'s or
`ResidualDifferentialIKAction`'s live per-step use), computing the joint
configuration that places the gripper's pinch point at the pregrasp
waypoint (cube position + `pregrasp_hover` in z), then writes that
directly into the simulation via `Articulation.write_joint_position_to_sim`
(and `set_joint_position_target` to the same value, so the PD drive
doesn't immediately fight to move away from the teleported pose) —
bypassing the action manager entirely for this one-time initialization,
the same class of technique `reset_root_state_uniform`/`reset_scene_to_default`
already use for other reset-time state changes in this repo.

**Env cfg**: new file `tasks/ar4/pickplace_reachskip_env_cfg.py`
(`Ar4PickPlaceReachskipEnvCfg`), built on **Experiment 12's clean,
non-regressed baseline** — the plain `isaaclab_mdp.DifferentialInverseKinematicsActionCfg`
(not Experiment 13's `ResidualDifferentialIKActionCfg`, whose root cause
is not yet resolved) — reusing Experiment 12's exact reward weights
unchanged (`path_proximity_bonus` 25.0, `gripper_schedule_bonus` 0.1,
`antipodal_grasp_bonus` 3.0 at `-0.7071`, `stillness_penalty` 5.0,
`action_rate`/`joint_vel` -1e-4) and `Ar4PickPlaceTaskspacePPORunnerCfg`
(`clip_actions=5.0`) unchanged. This isolates the one new variable —
starting state — against the last known-good reward/action baseline,
rather than building on Experiment 13's unresolved regression.

**Event ordering** in the new `EventCfg`:
1. `reset_all` (unchanged).
2. `reset_cube_position` (unchanged, randomizes the cube).
3. **New**: `reset_arm_to_pregrasp_pose` — computes and writes the
   pregrasp joint configuration for *this episode's* randomized cube
   position.
4. `randomize_goal` (unchanged).
5. `compute_path_waypoints` (unchanged) — still computes the full
   5-waypoint pregrasp/grasp/lift/transit/place path; since the arm now
   starts already at (or very near) the pregrasp waypoint,
   `path_proximity_bonus`'s waypoint-advance logic will likely credit
   waypoint 0 almost immediately, naturally shifting the effective
   remaining problem to grasp→lift→carry→place without any reward
   function change.

**Episode length**: unchanged (`episode_length_s=5.0`, 250 steps) for
this experiment — isolate the starting-state variable alone. If this
experiment shows genuine progress but still runs out of time, episode
length becomes a well-motivated *follow-up* variable to test next,
now backed by evidence rather than the earlier, unsupported assumption.

## What this does NOT change

- No modification to `pickplace_taskspace_env_cfg.py`,
  `pickplace_residual_env_cfg.py`, `residual_ik_action.py`, or any
  existing reward function in `mdp.py` — purely additive (new event
  function appended, new env cfg file).
- Does not attempt to fix or retry Experiment 13's residual mechanism —
  that remains open, unresolved, and explicitly not built upon here.
- Does not touch the classical demo's singularity finding — a different,
  independent thread.

## Verification plan

Same sequence this project uses for every experiment: smoke test (does
the new reset event construct and run without exception — this is a new
kind of event, one-shot IK + direct joint-state write, not yet exercised
anywhere in this repo, so a real Isaac-Sim-level smoke test matters more
than usual here), diagnostic run (300 iter, checking `Loss/value_function`
stays bounded — a new event term touching joint state at reset is a new
surface area, worth the same scrutiny prior new mechanisms got), full run
(1500 iter) + report comparing against Experiment 12's exact final values,
multi-episode video inspection (per the lesson from Experiments 12-13:
one episode is not a representative sample at this task's success rate),
ROADMAP record regardless of outcome.

## Success criteria

Not "full pick-and-place solved" in one shot. The bar: does
`path_proximity_bonus`'s scalar data and/or eval video show the policy
reaching waypoint index ≥2 (lift) *more* than any prior experiment, and
does eval video show genuine lift-off-the-ground in a meaningfully larger
fraction of episodes than the ~0/3 seen in Experiments 12-13's video
samples. A null result here — the policy still doesn't progress past
grasp even with reach removed — would itself be highly informative: it
would rule out "the reach sub-problem is eating all the exploration
budget" as an explanation, and point toward something more fundamental in
the lift-carry-place mechanics themselves (e.g., real contact/grasp
stability during motion) as the next thing to investigate.

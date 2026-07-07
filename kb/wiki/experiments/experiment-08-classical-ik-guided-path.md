# Experiment 8: Classical-IK-guided path reward

**Object:** sphere, then cube (see below — this experiment spans the
sphere→cube pivot). Continuation of the sphere-era investigation, but the
result that actually completed training happened after a repo-wide object
switch.

## Hypothesis

A live classical-IK path-tracking reward — 5 Cartesian waypoints (pre-grasp,
grasp, lift, transit, place), each tracked via a fresh
`isaaclab.controllers.DifferentialIKController` solve against the real
physics state every step — plus a gripper-open/closed timing bonus, gives
the policy a much denser, more informative progress signal than the sparse
milestone/lift terms tried so far.

## What changed

New reward combining `ik_guided_path_bonus` (waypoint-tracking) and
`gripper_schedule_bonus` (open/closed timing), alongside the carried-over
`contact_grasp_bonus`/`stillness_penalty`. Code review before training
caught and fixed a real bug: the IK target used the raw `link_6` body pose
instead of the gripper's actual pinch point (the 3.6cm corrected
`_EE_OFFSET` from [[experiment-01-contact-sensor-grasp-reward]], larger than
the `advance_tolerance` used for waypoint progression) — corrected by
rotating the offset into world frame via `link_6`'s live orientation before
commanding IK.

Two attempts at the full 1500-iteration run failed to complete on the
sphere: the first falsely reported success from stale/misleading console
text while checkpoints showed it never got past ~2 iterations; the second
was deliberately killed by the user mid-run (~iteration 114/1500) to
redirect all experimentation to a cube instead of a sphere. **Repo-wide
pivot**: `pickplace_mirror_env_cfg.py` and `pickplace_ik_guided_env_cfg.py`
were converted from `SPHERE_CFG` to the existing, unmodified `CUBE_CFG`
(18mm cuboid) — every scene field, event term, observation term,
termination term, and `SceneEntityCfg` reference renamed from "sphere" to
"cube" throughout both files. The cube's own 18mm default size was kept
unchanged rather than applying an analogous shrink to
[[experiment-07-sphere-shrink]]'s 12mm hack, since that experiment already
showed object-size shrinking alone doesn't help. All future experiments in
this line use the cube.

After the pivot, the classical-IK-guided path run was retried on the cube.
The first retry hit a real physics crash mid-run (`PxRigidActor::detachShape`
at iteration 893/1500, the first genuine PhysX-level failure across all
prior sphere/mirror-scene runs) — the retry after that succeeded with no
recurrence, confirming a rare fluke rather than a systematic issue.

## Quantitative result

Full 1500-iteration cube training run completed
(`logs/train/2026-07-06_19-45-06/`). Final episode-cumulative scalars:
`ik_guided_path_bonus=0.1428`, `gripper_schedule_bonus=0.0142`,
`contact_grasp_bonus=16.7976`, `stillness_penalty=-0.2318`,
`cube_reached_goal=0.0072`.

## Qualitative video finding

Not directly video-inspected for this checkpoint — the decision to move
straight to a reward redesign (Experiment 9) was made from the TensorBoard
scalar evidence, judged sufficient on its own to show the same underlying
failure mode already characterized on video in prior experiments.

## Verdict

**Completed, and directly informative rather than a dead end** — but not
itself a grasp/lift success. Two senior-tier literature research passes
commissioned in parallel on this run's own data converged independently on
real, actionable problems: `contact_grasp_bonus` (16.80) outweighs
`ik_guided_path_bonus` (0.14) by **~118:1** in the actual trained policy's
behavior, and the reward checks bilateral contact-force *magnitude* only,
discarding the force-*direction* information `force_matrix_w` already
provides. See [[experiment-09-antipodal-grasp-bonus]] for the resulting
redesign.

## Related concepts

[[reward-rate-arithmetic]] — the 118:1 reward-dominance figure, this
concept's clearest quantitative confirmation. [[grasp-mechanics-antipodal-vs-magnitude]]
— the magnitude-only-vs-direction finding that motivates Experiment 9.
[[citation-verification-practice]] — the two parallel senior research
passes analyzing this run's data.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md`,
`docs/superpowers/specs/research/2026-07-06-rl-manipulation-senior-b.md`,
`docs/superpowers/specs/research/2026-07-06-classical-manipulation-senior-a.md`

# Experiment 17: grasp-verification-gated lift/goal-tracking reward

## Hypothesis

**Experiment 16's reward function allowed a "stage leakage" exploit: the
policy satisfies the lift/goal-tracking reward's height-only condition by
wedging the cube against its own wrist/gripper-housing geometry as it
moves, never engaging the gripper jaws, because nothing in the reward
checks whether a genuine grasp produced that height.** Adding a
grasp-verification gate — requiring genuine bilateral antipodal contact
force (this repo's own already-proven force-closure check) alongside the
existing height condition, before either `lifting_object` or
`object_goal_tracking` can fire — should close this specific exploit and
push the policy toward genuine gripper-mediated lift, without
reintroducing a separately-rewarded grasp bonus (preserving the design
principle both of Experiment 16's proven references still support: grasp
itself earns no direct reward, only what it enables).

Falsifiable: if this experiment's video still shows the cube elevated
without measurable jaw contact force (the same instrumented check used to
diagnose Experiment 16), the gate itself has a bug or the exploit has
another form (e.g. minimal, non-antipodal jaw contact) and the hypothesis
is wrong or incomplete.

## Background research

**Direct confirmation from Experiment 16's own root-cause investigation**
(ROADMAP.md's Experiment 16 correction, 2026-07-07): a fresh instrumented
rollout of Experiment 16's exact final checkpoint
(`logs/train/2026-07-07_14-40-53/model_1499.pt`) — logging
`gripper_jaw1_contact`/`gripper_jaw2_contact` force magnitudes
(`force_matrix_w`), gripper joint positions, and the cube's distance to
the jaw links vs. the wrist/gripper-base link, every step of a full
episode — found **zero jaw contact force at every one of 250 logged
steps**, gripper essentially fully open throughout the ~170-step "held"
period, and the cube consistently closer to the wrist (`link_6`)/
`gripper_base_link` (~0.023m) than to either jaw (~0.051-0.056m). This is
not a hypothesis this experiment needs to re-establish — it is the
already-confirmed root cause this experiment is designed to fix.

**Literature grounding for the general phenomenon and the specific fix
approach** (verified directly from source, not a secondhand summary — see
this session's citation-verification practice): Xu et al., "Stage-
Transition Dense Reward Modeling for Reinforcement Learning" (arXiv:
2606.31377, 2026), independently names and studies exactly this failure
mode under the term **"stage leakage"**: quoting the paper directly,
"removing this module leads to 'stage leakage,' where the reward signal
prematurely transitions to the next functional stage before the agent has
securely grasped the object. Consequently, the agent often moves toward
the target with an empty gripper or drops the object mid-air." Their
proposed fix is a "grasping regulation module" that gates the stage-index
transition (from grasp stage to moving/placing stages) on a verification
signal, and their ablation (Figure 5, Pick-Place and Coffee-Push tasks)
reports worse convergence and final success when the module is removed
— i.e. published evidence that this class of fix (grasp-gate a downstream
reward transition) has real effect, not just intuitive plausibility.

**One honest, load-bearing difference from that paper, stated explicitly
rather than glossed over**: their verification signal is a learned MLP
operating on visual features ("the module predicts a stable grasp from
visual features"), not a direct contact-force reading — the paper does
not specify contact force as their signal. This experiment uses a
different, arguably more directly grounded signal available in this
repo's own privileged-state training setup: the same bilateral antipodal
force-closure check (`antipodal_grasp_bonus`, `tasks/ar4/mdp.py`) already
implemented and used across Experiments 9-15, itself grounded in Nguyen
1988 ("Constructing Force-Closure Grasps") and Ponce & Faverjon 1991/1993
("On Computing Two-Finger Force-Closure Grasps of Curved 2D Objects") —
already-cited, already-proven classical grasp mechanics, not a new
mechanism. The STDR paper's contribution is cited here for the *general
principle* (gate downstream reward on grasp verification to prevent stage
leakage) and its *published evidence that removing such a gate hurts
performance*, not for its specific implementation, which this experiment
does not replicate.

## Design

New file `tasks/ar4/pickplace_graspgated_env_cfg.py`
(`Ar4PickPlaceGraspGatedEnvCfg`) — built directly on Experiment 16's
`Ar4PickPlaceProvenRecipeEnvCfg` structure (same scene, same plain
joint-space action, same curriculum, same observations/events/
terminations — all unchanged, reused via import where possible), with
only the `lifting_object` and `object_goal_tracking`/
`object_goal_tracking_fine_grained` reward terms replaced by
grasp-gated versions. **Weights are kept identical to Experiment 16**
(`lifting_object` 15.0, `object_goal_tracking` 16.0,
`object_goal_tracking_fine_grained` 5.0) — isolating the grasp-gate as
the only new variable against Experiment 16's exact baseline, per this
project's established one-variable-at-a-time practice.

**New shared gating helper**, appended to `tasks/ar4/mdp.py`:

```python
def genuine_grasp_and_lift(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
    minimal_height: float,
) -> torch.Tensor:
    """Shared gating condition for Experiment 17: the object is lifted
    ONLY if both the height condition AND a genuine bilateral antipodal
    grasp (reusing antipodal_grasp_bonus's own force-closure check, not
    reimplementing it) hold simultaneously - fixes Experiment 16's
    "stage leakage" exploit (Xu et al. 2026, arXiv:2606.31377), confirmed
    via direct contact-sensor instrumentation to have let the policy wedge
    the cube against its own wrist/gripper-housing geometry with zero jaw
    contact force. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height_ok = object.data.root_pos_w[:, 2] > minimal_height
    grasp_ok = antipodal_grasp_bonus(
        env, force_threshold, antipodal_cos_threshold, jaw1_contact_cfg, jaw2_contact_cfg
    ) > 0.5
    return (height_ok & grasp_ok).float()
```

**New lift reward**, appended to `tasks/ar4/mdp.py` (replaces the reused
`mdp.object_is_lifted` from Experiment 16 — that function has no grasp
check and cannot be parameterized to add one, so a local wrapper is
needed, not a modification of the reused Isaac Lab function):

```python
def lifting_object_grasp_gated(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
    minimal_height: float,
) -> torch.Tensor:
    """Same binary reward shape as isaaclab_tasks' object_is_lifted
    (1.0/0.0), but ONLY pays out when genuine_grasp_and_lift's stricter
    condition holds - see that function's docstring. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    return genuine_grasp_and_lift(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height
    )
```

**New goal-tracking reward**, appended to `tasks/ar4/mdp.py` (direct
adaptation of Experiment 16's own `mirrored_goal_distance_gated`, with
the height-only gate replaced by `genuine_grasp_and_lift`'s stricter
condition — not a modification of Experiment 16's function, which stays
unmodified per this project's additive-only convention):

```python
def mirrored_goal_distance_grasp_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
) -> torch.Tensor:
    """Same tanh-kernel goal-distance formula as
    mirrored_goal_distance_gated (Experiment 16), but gated on
    genuine_grasp_and_lift's height-AND-grasp condition instead of height
    alone. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(env._target_pos_w - object.data.root_pos_w, dim=-1)
    gate = genuine_grasp_and_lift(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height
    )
    return gate * (1.0 - torch.tanh(distance / std))
```

**`RewardsCfg`** (only the three terms above differ from Experiment 16;
`reaching_object`, `action_rate`, `joint_vel` unchanged):

```python
reaching_object = RewTerm(
    func=mdp.object_ee_distance,
    weight=1.0,
    params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
)

lifting_object = RewTerm(
    func=ar4_mdp.lifting_object_grasp_gated,
    weight=15.0,
    params={
        "object_cfg": SceneEntityCfg("cube"),
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        "force_threshold": 0.05,
        "antipodal_cos_threshold": -0.7071,
        "minimal_height": 0.03,
    },
)

object_goal_tracking = RewTerm(
    func=ar4_mdp.mirrored_goal_distance_grasp_gated,
    weight=16.0,
    params={
        "std": 0.3,
        "minimal_height": 0.03,
        "object_cfg": SceneEntityCfg("cube"),
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        "force_threshold": 0.05,
        "antipodal_cos_threshold": -0.7071,
    },
)

object_goal_tracking_fine_grained = RewTerm(
    func=ar4_mdp.mirrored_goal_distance_grasp_gated,
    weight=5.0,
    params={
        "std": 0.05,
        "minimal_height": 0.03,
        "object_cfg": SceneEntityCfg("cube"),
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        "force_threshold": 0.05,
        "antipodal_cos_threshold": -0.7071,
    },
)

action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
```

`ActionsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`,
`CurriculumCfg`, and the PPO runner cfg are all identical to Experiment
16's — copy verbatim, no changes.

## What this does NOT change

No modification to `tasks/ar4/pickplace_provenrecipe_env_cfg.py` or any
existing function in `tasks/ar4/mdp.py` (including `antipodal_grasp_bonus`
and `mirrored_goal_distance_gated`, both reused/called, not edited) —
purely additive (three new functions, one new env cfg file). Does not
touch the secondary mimic-joint-import finding from Experiment 16's
correction (flagged there as its own, separate, not-yet-investigated
issue).

## Verification plan

Same sequence as every Tier-1 experiment: smoke test, 300-iteration
diagnostic (`Loss/value_function` bounded), full 1500-iteration run +
report comparing `cube_reached_goal` against Experiment 16's final value
and Experiment 12's, multi-episode video inspection (≥3 of 10 episodes).
**Critically, this experiment's verification cannot stop at video alone —
Experiment 16's own mistake was trusting video for a mechanism claim.**
Task 6 (or an explicit post-plan controller step, matching this session's
established pattern) must re-run the same instrumented contact-force
rollout check used to root-cause Experiment 16
(gripper_jaw1_contact/gripper_jaw2_contact force magnitudes, gripper
joint positions, cube-to-jaw vs. cube-to-wrist distances, logged across a
full episode) against this experiment's own trained checkpoint, to
directly confirm or refute that genuine bilateral jaw contact is present
during any elevated-hold behavior this experiment produces, before
writing any "genuine grasp" claim into ROADMAP.md.

## Success criteria

Primary: does the instrumented contact-force check (not video alone) show
genuine bilateral antipodal contact force during any sustained elevated
hold, confirming the exploit is closed. Secondary: does eval video still
show the cube reaching an elevated state at all (a real risk: the grasp
gate could make the reward sparse enough that lift becomes much harder to
discover than in Experiment 16, potentially regressing behavior back
toward Experiments 1-15's "reach and freeze" pattern) — if lift becomes
much rarer or disappears entirely, that is itself an informative,
reportable result about the tradeoff between exploit-closure and
discoverability, not a silent failure to gloss over.

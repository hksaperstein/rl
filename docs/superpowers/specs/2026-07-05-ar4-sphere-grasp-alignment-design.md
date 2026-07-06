# AR4 sphere grasp-alignment reward design (follow-up to grasp-bonus)

## Problem

Per `ROADMAP.md`'s "grasp/lift never emerges" follow-up, the prior
`grasp_sphere` dense reward (`docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-bonus-design.md`)
successfully taught the policy to close the gripper near the sphere, but
this was reward hacking: the reward only checked EE-origin-to-object
distance + gripper closure, with no check that the object was actually
*between* the fingers. 0/10 eval episodes showed a real grasp+lift — the
gripper closed beside the sphere, not around it.

## Research (citation-verified)

Delegated real literature research (junior researcher + independent senior
citation review — see `docs/superpowers/specs/research/2026-07-05-grasp-alignment-literature-junior.md`
and `-senior-review.md`). After stripping fabricated/misapplied citations
(the senior review caught a fabricated `std=0.02m` numeric claim with no
basis in any cited paper, a Factory-paper citation for a "bilateral contact
gating" mechanism that isn't in that paper, and an AsymDex citation
misapplied from bimanual inter-hand coordination to single-gripper
fingertip geometry), what survives verbatim-confirmed:

- **GRIT** (arXiv:2604.04138): multiplicative reward gating
  (`r = r_h·α_h + r_o·α_o − r_pen`, constraint coefficients ∈ [0,1]
  attenuating reward under undesirable behavior) is a real, precisely-
  quoted pattern — directly applicable to gate the closure reward on an
  alignment condition, rather than the previous additive/independently-
  satisfiable combination that got exploited.
- **Isaac Lab's own `ContactSensor`/`contact_forces`** infrastructure is
  confirmed real and available (verified directly against the installed
  source, `isaaclab/sensors/contact_sensor/contact_sensor.py`,
  `isaaclab/envs/mdp/rewards.py:281`) — kept as the Tier-3 fallback if this
  attempt also fails, per the research's escalation ordering.
- The general "fingertip-relative geometry, not single EE-origin distance"
  idea is sound kinematics regardless of the misapplied citation
  (AsymDex) — it doesn't need literature backing to justify, it's a direct
  fix for the exact failure mode observed on video (object beside, not
  between, the fingers).

## Decision

Implement a **multiplicative alignment-gated** grasp reward: extend the
existing `ee_frame` `FrameTransformerCfg` (already in
`Ar4PickPlaceSceneCfg`) with two more `target_frames`, one per gripper
jaw link, then gate the closure reward by how well-centered the sphere is
between them — replacing the previous single-point EE-origin proximity
check entirely (single-variable follow-up experiment; the exploited
`grasp_sphere` term from the prior experiment was already reverted, not
carried forward).

Gripper jaw link names: `gripper_jaw1_link` / `gripper_jaw2_link`
(confirmed from the AR4 URDF — `/tmp/ar4_urdf_51xsp4_s/ar4_mk5.urdf`,
prismatic joints `gripper_jaw1_joint`/`gripper_jaw2_joint`, both children of
`gripper_base_link`). Prim paths hypothesized by pattern-matching the
existing `link_6` FrameTransformer target
(`{ENV_REGEX_NS}/Robot/root_joint/link_6`, a flat sibling under
`root_joint` despite being deep in the URDF chain — this repo's USD import
flattens the articulation's links to siblings rather than mirroring URDF
nesting). Three standalone attempts to confirm this via direct Python
introspection (`pxr.Usd.Stage.Open` on the raw USD, and later
`ManagerBasedRLEnv` construction to read `robot.data.body_names`) hung
indefinitely in this sandbox for unclear reasons unrelated to the
hypothesis itself (Kit app boot stalling, not a code error — confirmed via
`ps`/`nvidia-smi`, no GPU deadlock, just near-zero CPU progress over
several minutes). Rather than keep debugging a standalone introspection
script, this hypothesis is handed to the implementer subagent's own
smoke-test workflow to validate/correct: **`FrameTransformerCfg` raises a
clear "prim path does not exist" error at env-creation time if wrong**, so
this is a safe, fast-failing assumption, not a silent-failure risk.

## Design

```python
@configclass
class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
                name="finger_left",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
                name="finger_right",
            ),
        ],
    )
```

New reward function in `tasks/ar4/mdp.py` (replacing, not augmenting,
the previous experiment's `grasp_object_bonus` — that function and its
`grasp_sphere` RewTerm registration were already reverted out of
`pickplace_env_cfg.py`, so this is a clean single addition):

```python
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

Registered in `RewardsCfg`:

```python
grasp_sphere_aligned = RewTerm(
    func=ar4_mdp.aligned_grasp_bonus,
    weight=10.0,
    params={
        "centering_std": 0.01,
        "open_joint_pos": GRIPPER_OPEN_POS,
        "object_cfg": SceneEntityCfg("sphere"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
    },
)
```

Parameter reasoning:
- `centering_std=0.01` (1cm): this specific number is **an engineering
  estimate, not a literature-derived value** — the senior review explicitly
  flagged the prior experiment's "std=0.02m" claim as a fabricated citation
  and recommended treating any such number as a hypothesis to test
  empirically, not a validated finding. 1cm is roughly the sphere's radius
  (9mm) — tight enough that "object sitting beside one finger" (observed
  failure, likely several cm off-center) scores near-zero alignment, while
  still permitting a few mm of real grasp tolerance.
  Documented explicitly as a first-guess parameter, not a citation-backed
  constant.
- `weight=10.0`: unchanged from the reverted experiment — same
  reward-scale reasoning (raw term max ~0.028 from gripper stroke, so
  weight 10 gives a max contribution ~0.28, comparable to but smaller than
  `reaching_sphere`'s max 1.0/step).
- This is now **multiplicative**, not additive: if `alignment_score≈0`
  (object not centered between fingers), the whole term collapses to ~0
  regardless of closure — structurally closing off the previous exploit
  path, per GRIT's verified gating pattern.
- No change to `reaching_sphere`, `lifting_sphere`, `sphere_goal_tracking*`
  — single-variable experiment, same discipline as both prior rounds.

## Verification plan

Identical rigor to the grasp-bonus experiment: smoke test
(`--num_envs 16 --max_iterations 2`), full run (`--num_envs 4096`, 1500
iterations, monitor `Episode_Reward/grasp_sphere_aligned`,
`Episode_Reward/lifting_sphere`, `Episode_Termination/sphere_reached_goal`
via TensorBoard), then real eval (`--episodes 10`) with frame-extracted
video inspection of all 10 episodes — specifically checking whether the
sphere is now positioned between the jaws before closure (not just
whether the gripper closes), and whether it actually leaves the ground.
Decision gate: 8/10 episodes show a real grasp+lift.

If `FrameTransformerCfg` fails at env-creation with an invalid-prim-path
error, that confirms the `gripper_jaw1_link`/`gripper_jaw2_link` hypothesis
was wrong — the error message will name the actual invalid path, and the
implementer should re-derive the correct sibling name from that error
(e.g. by opening the same `ManagerBasedRLEnv` construction path within the
smoke test's own Python process and printing `robot.data.body_names`
before re-raising, rather than a separate standalone script) and retry
once before escalating back to the Principal.

If this also fails to move `lifting_sphere` off 0.0000, this is the third
falsified reward-shaping hypothesis — per `superpowers:systematic-debugging`
Phase 4.5, escalate to the verified-available `ContactSensor`
infrastructure (Tier 3) rather than a fourth reward-only tweak.

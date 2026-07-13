# Size-Curriculum (Mixed-Size Training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one policy across per-env die sizes 30.3-48.0mm and test
whether lift transfers to the 30.3mm envs — the asset-bisect's follow-up
per `docs/superpowers/specs/2026-07-13-size-curriculum-design.md`.

**Architecture:** New env cfg `FrankaDieLiftJointMixedEnvCfg` layered on
`FrankaDieLiftJointHeavyEnvCfg`, replacing the object's spawn cfg with a
`MultiAssetSpawnerCfg` (verified present in installed Isaac Lab:
`isaaclab/sim/spawners/wrappers/wrappers_cfg.py:16`) carrying 5
`UsdFileCfg` entries — same `_D20_USD`, scales {0.001585, 0.001440,
0.001291, 0.001146, 0.001000} = {48.0, 43.6, 39.1, 34.7, 30.3}mm —
mass_props 0.216kg on all. PLAY cfg is **all-30.3mm** (the verdict
probe). 3000 iterations per run (pre-registered, 2x bisect standard).

**Tech Stack:** existing train_franka.py / franka_checkpoint_review.py
variant pattern (reference commits 8aefe20, ad312a8, c8d5ba8).

## Global Constraints

- NEVER `--headless` locally; `DISPLAY=:1`; every Isaac launch wrapped
  `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 ..."`.
- GPU is shared with a YOLO training job tonight: before any Isaac
  launch, check `nvidia-smi` — if a python vision-training process holds
  the GPU with <4GB VRAM free, wait for it with a bounded until-loop
  (≤90 min) rather than contending.
- Decided values verbatim: scales above; mass 0.216; seeds 42/123/7;
  3000 iterations; verdict = per-seed instrumented eval on the
  all-30.3mm PLAY cfg, seed PASS = ≥6/8 envs sustained lift; experiment
  PASS = ≥2/3 seeds.
- Stuck-lock / teardown-hang procedures per CLAUDE.md. Verify via
  artifacts. Commit+push together, every time. Commit messages end:
  `Claude-Session: https://claude.ai/code/session_01BwZBx9ssmVf2PbTXrYDFEy`

---

### Task 1: Mixed-size env variant + smoke

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` (append),
  `scripts/train_franka.py`, `scripts/franka_checkpoint_review.py`,
  `scripts/_diag_object_mass_check.py` (variant wiring, pattern of
  commit c8d5ba8's joint-cube-baked additions)

**Interfaces:**
- Consumes: `FrankaDieLiftJointHeavyEnvCfg`, `_D20_USD` (same module);
  `MultiAssetSpawnerCfg` from `isaaclab.sim.spawners.wrappers`.
- Produces: `FrankaDieLiftJointMixedEnvCfg` / `..._PLAY` (PLAY =
  all-30.3mm single-size); variant `joint-die-mixed`, log suffix
  `_jointdiemixed`.

- [ ] **Step 1: Read the installed spawner source** —
  `isaaclab/sim/spawners/wrappers/wrappers.py` `spawn_multi_asset` and
  its cfg. Confirm: per-env asset choice semantics (`random_choice`
  True=random / False=round-robin — use **False** for deterministic
  ~819-envs-per-size split), and that `RigidObjectCfg.spawn` accepts a
  `MultiAssetSpawnerCfg`. Record findings in the report; if semantics
  differ from this plan's assumption, STOP and report NEEDS_CONTEXT.

- [ ] **Step 2: Append the env cfgs**

```python
@configclass
class FrankaDieLiftJointMixedEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Size-curriculum primary arm (docs/superpowers/specs/2026-07-13-size-curriculum-design.md):
    per-env die size varied across {48.0, 43.6, 39.1, 34.7, 30.3}mm
    (deterministic round-robin), mass pinned 0.216kg on every size.
    Everything else inherits from the heavy variant unchanged."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _scales = (0.001585, 0.001440, 0.001291, 0.001146, 0.001000)
        self.scene.object.spawn = MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    usd_path=_D20_USD,
                    scale=(s, s, s),
                    rigid_props=_D20_RIGID_PROPS,  # extracted constant, see note below
                    mass_props=MassPropertiesCfg(mass=0.216),
                )
                for s in _scales
            ],
            random_choice=False,
        )
```
Note: the parent's `rigid_props` block (solver iterations etc.) must be
replicated onto each `UsdFileCfg` — extract the existing literal
`RigidBodyPropertiesCfg(...)` from `FrankaDieLiftJointEnvCfg` into a
module-level `_D20_RIGID_PROPS` constant and reference it from BOTH the
base class and the mixed cfg (single source of truth; base behavior
byte-identical — verify via the joint-die smoke config in Step 5's
diff check). Add the `MultiAssetSpawnerCfg` import from
`isaaclab.sim.spawners.wrappers`.

```python
@configclass
class FrankaDieLiftJointMixedEnvCfg_PLAY(FrankaDieLiftJointMixedEnvCfg):
    """All-30.3mm eval probe (the spec's verdict measurement)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn = UsdFileCfg(
            usd_path=_D20_USD,
            scale=(0.001, 0.001, 0.001),
            rigid_props=_D20_RIGID_PROPS,
            mass_props=MassPropertiesCfg(mass=0.216),
        )
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

- [ ] **Step 3: Wire variant `joint-die-mixed`** (choices, dispatch
  branch, log suffix `_jointdiemixed`) into the three scripts, exactly
  parallel to c8d5ba8's pattern.

- [ ] **Step 4: Verification launches** (GPU-contention rule from
  Global Constraints applies): (a) `_diag_object_mass_check.py
  --variant joint-die-mixed` → both sampled envs read 0.216; (b) smoke
  `train_franka.py --variant joint-die-mixed --seed 42 --num_envs 16
  --max_iterations 2` → 2 iters, no Traceback, no object_dropping
  storm, and CONFIRM size variation on the live stage: with 16 envs and
  round-robin over 5 assets, env prim inspection (or the spawner's own
  startup print, or a one-off readout of object bounding boxes) must
  show ≥2 distinct scales — state the evidence in the report.

- [ ] **Step 5: Diff check + commit + push** — all existing variant
  paths byte-identical (the `_D20_RIGID_PROPS` extraction is the one
  allowed touch to the base class; verify it's value-identical);
  message "feat: joint-die-mixed size-curriculum variant (Task 1)".

---

### Task 2: Three-seed 3000-iteration runs + per-seed 30.3mm eval verdict

- [ ] **Step 1: Launch sequentially** (each ~65 min):
```bash
for SEED in 42 123 7; do
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die-mixed --seed $SEED --num_envs 4096 --max_iterations 3000 2>&1 | tee /tmp/mixed_seed$SEED.log"
done
```
- [ ] **Step 2: Per-seed training readout** (event files, all points):
  VF boundedness; all-env-mean position_error trajectory (expect a
  blend — this is NOT the verdict metric, note it as context only);
  lifting_object trend.
- [ ] **Step 3: Per-seed verdict eval** — for each seed's model_2999.pt:
  `franka_checkpoint_review.py --variant joint-die-mixed --checkpoint
  <ckpt> --num_envs 8` (PLAY cfg = all-30.3mm; video_length 500
  default). Seed PASS = ≥6/8 sustained lifts in heights json.
- [ ] **Step 4: Experiment verdict** (≥2/3 seeds) → report to
  controller BEFORE any fallback-arm work. PASS → Task 3. FAIL →
  controller decides on the staged-anneal fallback arm.

---

### Task 3 (on PASS): report + docs + video

Controller inspects the first passing seed's eval video (full arm +
table, 2 episodes — standing rule) and sends to user. Write
`docs/superpowers/plans/2026-07-13-size-curriculum-report.md`
(hypothesis verdict, per-seed numbers vs the bisect anchors 0/3@30.3
and 1/3@48), update ROADMAP + kb in the same pass, sync runs to GCS
(`scripts/sync_all_franka_runs.sh` — extend its log-root map with
`train_franka_jointdiemixed` → experiment `size-curriculum`), commit+push.

---

## Verification standard

Artifacts at every step: live mass + live size-variation evidence (not
cfg prose), full event files, heights json for verdicts,
controller-inspected video for the headline claim.

# Asset-Bisect Ladder (Rung 1: Mass) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether raising the baked d20's mass from 0.0100kg to the
DexCube's measured 0.216kg (everything else pinned) makes the joint-space
lift recipe discover grasping — rung 1 of the asset-bisect ladder in
`docs/superpowers/specs/2026-07-12-asset-bisect-design.md`.

**Architecture:** One new env-cfg subclass (`FrankaDieLiftJointHeavyEnvCfg`)
layered on the existing `FrankaDieLiftJointEnvCfg`, overriding only
`spawn.mass_props`. A `--seed` flag on the trainer enables the spec's
3-seed protocol. Verdict per the spec: 2/3 seeds with
`Metrics/object_pose/position_error` decisively below the ~0.216
do-nothing baseline.

**Tech Stack:** Isaac Lab ManagerBasedRLEnv + rsl_rl PPO (existing
`scripts/train_franka.py`), TensorBoard event_accumulator readouts,
`scripts/franka_checkpoint_review.py` for the instrumented eval.

## Global Constraints

- NEVER pass `--headless` / set `args_cli.headless` for Isaac-touching
  scripts; `DISPLAY=:1` is available (standing user instruction).
- Every Isaac launch: `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p ..."`
  (`-o` is mandatory — Omniverse Hub daemon inherits the lock fd
  otherwise; see CLAUDE.md Environment conventions).
- Stuck lock diagnosis: `lsof /tmp/rl_isaac_sim.lock`; an `Omniverse Hub`
  holder at 0% CPU with no live training process → `kill -TERM` it.
  Teardown hang (log shows completion line, near-idle GPU) → `kill -TERM`
  is safe, output is already on disk.
- Isaac startup can take 5-8 min or non-deterministically hang once;
  budget one retry.
- Verify via real artifacts (env.yaml, event files, printed live masses),
  never exit codes alone.
- All decided values verbatim from the spec: mass override **0.216** kg,
  seeds **42, 123, 7**, 1500 iterations, 4096 envs, verdict = 2/3 seeds.
- Commit messages end with:
  `Claude-Session: https://claude.ai/code/session_01BwZBx9ssmVf2PbTXrYDFEy`

---

### Task 1: Heavy-die env variant + --seed flag + smoke test

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` (append two classes)
- Modify: `scripts/train_franka.py` (variant choice, log root, --seed)
- Modify: `scripts/franka_checkpoint_review.py` (variant choice)
- Modify: `scripts/_diag_object_mass_check.py` (variant choice)

**Interfaces:**
- Consumes: `FrankaDieLiftJointEnvCfg` (existing, same module) and the
  baked `assets/dice/d20_physics.usd` (has MassAPI baked — Task 1 of the
  die-lift plan — so a `mass_props` override MODIFIES it rather than
  silently no-oping; that history is why Step 4 verifies the live mass).
- Produces: `FrankaDieLiftJointHeavyEnvCfg` / `FrankaDieLiftJointHeavyEnvCfg_PLAY`
  importable from `tasks.franka.dice_lift_joint_env_cfg`;
  `scripts/train_franka.py --variant joint-die-heavy --seed <int>`;
  log root `logs/train_franka_jointdieheavy/`.

- [ ] **Step 1: Append the heavy variant to `tasks/franka/dice_lift_joint_env_cfg.py`**

Add `MassPropertiesCfg` to the existing schemas_cfg import line:

```python
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
```

Append at end of file:

```python
@configclass
class FrankaDieLiftJointHeavyEnvCfg(FrankaDieLiftJointEnvCfg):
    """Asset-bisect rung 1 (docs/superpowers/specs/2026-07-12-asset-bisect-design.md):
    the d20 with its mass raised 0.0100kg -> 0.216kg (DexCube's measured
    live PhysX mass, scripts/_diag_object_mass_check.py 2026-07-12).
    Shape, 30.3mm size, friction, and the whole joint-space config stay
    pinned - mass is this rung's ONLY variable."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=0.216)


@configclass
class FrankaDieLiftJointHeavyEnvCfg_PLAY(FrankaDieLiftJointHeavyEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

- [ ] **Step 2: Wire `--variant joint-die-heavy` + `--seed` into `scripts/train_franka.py`**

In the `--variant` argparse choices, change to:

```python
    choices=["ik-cube", "joint-die", "joint-cube", "joint-die-heavy"],
```

and append to its help string:
`" joint-die-heavy: asset-bisect rung 1 - the d20 at DexCube's measured 0.216kg mass (docs/superpowers/specs/2026-07-12-asset-bisect-design.md)."`

Add a `--seed` argument next to the other parser args:

```python
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Override the PPO runner cfg's seed (asset-bisect 3-seed protocol; default: keep agent cfg's own).",
)
```

In `main()`, extend the variant dispatch:

```python
    elif args_cli.variant == "joint-die-heavy":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg

        env_cfg = FrankaDieLiftJointHeavyEnvCfg()
```

After `agent_cfg = FrankaLiftPPORunnerCfg()` and its device line, before
`env_cfg.seed = agent_cfg.seed`, add:

```python
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
```

(the existing `env_cfg.seed = agent_cfg.seed` line then propagates it).

Extend the log-suffix map:

```python
    _log_suffix = {
        "ik-cube": "",
        "joint-die": "_jointdie",
        "joint-cube": "_jointcube",
        "joint-die-heavy": "_jointdieheavy",
    }[args_cli.variant]
```

- [ ] **Step 3: Wire the variant into `scripts/franka_checkpoint_review.py` and `scripts/_diag_object_mass_check.py`**

`franka_checkpoint_review.py`: read the file first — it already has a
`--variant {ik-cube,joint-die,joint-cube}` switch (commit 02f3bd3). Add
`"joint-die-heavy"` to its choices and a branch importing
`FrankaDieLiftJointHeavyEnvCfg_PLAY` from
`tasks.franka.dice_lift_joint_env_cfg`, exactly parallel to the existing
joint-die branch. Keep every existing branch byte-identical.

`_diag_object_mass_check.py`: change

```python
parser.add_argument("--variant", choices=["joint-die", "joint-cube"], required=True)
```

to

```python
parser.add_argument("--variant", choices=["joint-die", "joint-cube", "joint-die-heavy"], required=True)
```

and extend its `main()` import branch:

```python
    if args_cli.variant == "joint-die":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg as Cfg
    elif args_cli.variant == "joint-die-heavy":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg as Cfg
    else:
        from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeLiftJointEnvCfg as Cfg
```

- [ ] **Step 4: Verify the live mass override (the spec's mandatory check)**

Run:
```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_object_mass_check.py --variant joint-die-heavy 2>&1 | tee /tmp/mass_heavy.log"
```
Expected in the log: `object masses (kg, per env): [[0.216...], [0.216...]]`.
If it still reads 0.01, the override silently no-oped — STOP, report
BLOCKED (do not improvise a different override mechanism; the controller
decides).

- [ ] **Step 5: Training smoke test**

Run:
```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die-heavy --seed 42 --num_envs 16 --max_iterations 2 2>&1 | tee /tmp/heavy_smoke.log"
```
Expected: 2 iterations complete, no Traceback, log dir under
`logs/train_franka_jointdieheavy/`, saved `params/env.yaml` shows
`mass_props` with `mass: 0.216` on the object and `seed: 42` in
`params/agent.yaml`. No `object_dropping` termination storm.

- [ ] **Step 6: Confirm the other variants are untouched**

Run: `git diff scripts/train_franka.py scripts/franka_checkpoint_review.py`
and confirm every hunk is additive (new choice/branch/suffix entries
only); the ik-cube/joint-die/joint-cube code paths must read identically.

- [ ] **Step 7: Commit**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py scripts/train_franka.py scripts/franka_checkpoint_review.py scripts/_diag_object_mass_check.py
git commit -m "feat: joint-die-heavy variant (0.216kg d20) + --seed flag (asset-bisect rung 1, Task 1)"
git push origin franka-panda-pivot
```

---

### Task 2: Rung-1 three-seed runs + verdict readout

**Files:** none created (runs + readout; tee logs + TensorBoard events
are the artifacts; readout numbers go into the report in Task 3).

**Interfaces:**
- Consumes: `--variant joint-die-heavy --seed <s>` from Task 1.
- Produces: three `logs/train_franka_jointdieheavy/<ts>/` runs (seeds
  42, 123, 7) and a per-seed scalar readout for the rung verdict.

- [ ] **Step 1: Launch the three runs sequentially** (each ~30 min; the
flock queue serializes them naturally — launch back-to-back):

```bash
for SEED in 42 123 7; do
  flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die-heavy --seed $SEED --num_envs 4096 --max_iterations 1500 2>&1 | tee /tmp/heavy_seed$SEED.log"
done
```
Confirm each run's `params/agent.yaml` records the intended seed before
trusting its label.

- [ ] **Step 2: Per-seed scalar readout** (all 1500 points, event file via
`/home/saps/IsaacLab/_isaac_sim/python.sh` + `event_accumulator`, NOT bare
python3 — established pattern). For each seed record:
`Loss/value_function` boundedness (points >1e3 count),
`Metrics/object_pose/position_error` (trajectory samples + last-100 mean
vs the 0.216 do-nothing baseline), `Episode_Reward/lifting_object`
(vs its ~0.12 spawn-artifact floor), `Episode_Reward/object_goal_tracking`.

- [ ] **Step 3: Rung verdict per the spec**

PASS iff ≥2 of 3 seeds show position_error last-100 mean decisively
below 0.216 AND trending down AND lifting_object clearly above the 0.12
floor. A single positive seed = FAIL (record it as such, note the seed).
Value-loss divergence in any seed = flag for the controller, do not
self-interpret.

- [ ] **Step 4: Report the verdict to the controller BEFORE any further
runs.** The controller decides: rung PASS → Task 3; rung FAIL → rung 2
(Task 4). Do not start either without that decision.

---

### Task 3 (only if rung 1 PASSES): Instrumented eval + video + docs

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-asset-bisect-report.md` (create)
- Modify: `ROADMAP.md`, `kb/wiki/experiments/joint-space-die-lift.md`
  (or a new kb page `kb/wiki/experiments/asset-bisect.md`)

- [ ] **Step 1: Eval on the strongest seed's final checkpoint**

```bash
flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/franka_checkpoint_review.py --variant joint-die-heavy --checkpoint logs/train_franka_jointdieheavy/<ts>/model_1499.pt --num_envs 8 --video_length 500 2>&1 | tee /tmp/heavy_eval.log"
```
VIDEO RULE (standing user instruction, strengthened 2026-07-12): the
frame must include the FULL arm AND the table, and the recording must
run LONGER than the event — `--video_length 500` (2 full episodes) so
every lift event has lead-in and aftermath on film; never a clip that
ends at the key moment.
Expected artifacts: whole-arm+table-framed mp4 (500 steps) +
`heights_*.json/npy` with sustained-lift counts. The CONTROLLER inspects
the video and makes the verdict call (implementer reports numbers +
paths only).

- [ ] **Step 2: Write the report** (`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`):
hypothesis verdict, per-seed numbers, eval instrumentation, video
finding, and the one authorized confirmation-run decision (spec: rung
N-1 isolation check) left explicitly to the controller.

- [ ] **Step 3: Update ROADMAP.md + kb in the same pass, commit, push**

```bash
git add docs/ ROADMAP.md kb/
git commit -m "docs: asset-bisect rung 1 (mass) verdict + report"
git push origin franka-panda-pivot
```

---

### Task 4 (only if rung 1 FAILS): Rung 2 — size at pinned mass

**Files:**
- Modify: `tasks/franka/dice_lift_joint_env_cfg.py` (append two classes)
- Modify: `scripts/train_franka.py`, `scripts/franka_checkpoint_review.py`,
  `scripts/_diag_object_mass_check.py` (one more variant choice each,
  exactly parallel to Task 1's steps)

Append:

```python
@configclass
class FrankaDieLiftJointBigEnvCfg(FrankaDieLiftJointHeavyEnvCfg):
    """Asset-bisect rung 2: d20 scaled 30.3mm -> 48.0mm (DexCube's
    measured effective size) with mass PINNED at 0.216kg by the inherited
    mass_props override - size is this rung's ONLY new variable (letting
    mass scale with volume would silently reintroduce rung 1's variable,
    per the spec)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.object.spawn.scale = (0.001585, 0.001585, 0.001585)


@configclass
class FrankaDieLiftJointBigEnvCfg_PLAY(FrankaDieLiftJointBigEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

Variant name `joint-die-big`, log suffix `_jointdiebig`. Then repeat
Task 1 Steps 4-7 (live mass must STILL read 0.216; also verify the
spawned die's visual size vs the gripper in the smoke run's viewport or
via the scale value in env.yaml) and Task 2's three-seed protocol
verbatim with `--variant joint-die-big`. Same verdict rule; report to
controller; rung 3 (shape) requires a new controller-authored task (it
needs a new baked asset, not just a cfg override).

---

## Verification standard (whole plan)

Real evidence over proxies at every step: saved env.yaml/agent.yaml for
config claims, live `get_masses()` for the mass claim, full event files
for metric claims, controller-inspected video for the behavioral claim.

# Joint-Space (No-IK) RL Die-Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a PPO policy that lifts a d20 die with direct joint-space
actions (no IK anywhere in the action path), per
`docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md`.

**Architecture:** Subclass the repo's existing `FrankaLiftEnvCfg`
(IK-action cube-lift, itself a verified stock-recipe reproduction) and
override exactly two things in `__post_init__` — the arm action
(`JointPositionActionCfg`, Isaac Lab's own joint_pos values) and the
object (a physics-baked copy of the d20 USD). Same subclass-override
pattern Isaac Lab's own `joint_pos_env_cfg.py` uses.

**Tech Stack:** Isaac Lab (ManagerBasedRLEnvCfg), rsl_rl PPO, USD/pxr.

## Global Constraints

- Every Isaac-Sim-touching command runs from the repo root as
  `flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p <script> <args> 2>&1 | tee /tmp/<name>.log"`.
  Only one Isaac process at a time; another thread may hold the lock —
  flock blocks until free, that is expected.
- Interactive/watchable runs are NEVER headless (a display exists,
  DISPLAY=:1). Asset-baking with `SimulationApp({"headless": True})` is
  permitted (repo precedent: `scripts/build_asset.py`,
  `scripts/_diag_die_scale_check.py` — batch asset processing, not a
  watchable run).
- Known Isaac ops failure modes: startup can hang silently (if the tee
  log hasn't grown within 10 minutes, kill and relaunch once); after a
  script's own completion line, Kit reliably hangs in teardown — kill the
  Kit PID (split-string pgrep pattern, e.g. `pat='pytho''n3'`, so it never
  matches your own command line), then any `Omniverse Hub` process still
  holding the lock (`fuser -v /tmp/rl_isaac_sim.lock`). Never use the
  Monitor tool to wait on runs; use run_in_background Bash with until-loop
  greps.
- Do NOT modify: `tasks/franka/lift_env_cfg.py` (only subclass it),
  `vision/data/raw/dice_sets_v1/*` (read-only source assets),
  `tasks/franka/agents/rsl_rl_ppo_cfg.py`'s existing class.
- No reward/observation/PPO changes of any kind — this experiment pins
  everything except action space and object.
- Commit to `franka-panda-pivot`, push after each task.

---

### Task 1: Bake the d20 physics asset

**Files:**
- Create: `scripts/bake_die_asset.py`
- Create (output, committed): `assets/dice/d20_physics.usd`

**Interfaces:**
- Produces: `assets/dice/d20_physics.usd` — default prim named `Object`
  (renamed from the source root prim), every `UsdGeom.Mesh` carrying
  `UsdPhysics.CollisionAPI` + `UsdPhysics.MeshCollisionAPI`
  (approximation `"convexHull"`), root carrying `UsdPhysics.RigidBodyAPI`
  + `PhysxSchema.PhysxRigidBodyAPI` + `UsdPhysics.MassAPI` (mass 0.01).
  Geometry still in source units (mm-as-m); consumers apply
  `scale=(0.001, 0.001, 0.001)` at spawn.

- [ ] **Step 1: Write `scripts/bake_die_asset.py`**

```python
"""One-off asset bake: copy a dice_sets_v1 die USD and write physics schemas
into the copy, so RL env cfgs can spawn it without runtime patching.

Headless SimulationApp is correct here (batch asset processing, not a
watchable run - same precedent as scripts/build_asset.py). Run under flock:

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \
      /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py \
      --die d20 2>&1 | tee /tmp/bake_die_asset.log"

The output USD keeps the source's mm-as-m geometry units; spawn-time
scale=(0.001,)*3 is the consumer's job (dice-demo-validated convention).
The default prim is renamed to 'Object' so the stock lift recipe's
SceneEntityCfg("object", body_names="Object") terms match unchanged.
"""

import argparse
import os
import shutil

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Bake physics schemas into a die USD copy.")
parser.add_argument("--die", default="d20", choices=["d4", "d8", "d10", "d12", "d20"])
parser.add_argument("--set", dest="set_name", default="set_00000")
parser.add_argument("--mass", type=float, default=0.01, help="kg, dice-demo value")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "vision", "data", "raw", "dice_sets_v1", f"{args.set_name}_{args.die}.usd")
OUT_DIR = os.path.join(REPO, "assets", "dice")
OUT = os.path.join(OUT_DIR, f"{args.die}_physics.usd")


def main() -> None:
    if not os.path.isfile(SRC):
        raise FileNotFoundError(SRC)
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copyfile(SRC, OUT)

    stage = Usd.Stage.Open(OUT)
    old_root = stage.GetDefaultPrim()
    if not old_root:
        raise RuntimeError(f"{OUT} has no default prim")

    # Rename default prim to 'Object' (stock lift recipe's expected body name).
    if old_root.GetName() != "Object":
        # Usd has no in-place rename; re-parent via a new Xform and move children.
        # Simpler robust approach: define /Object, move the old root's children
        # is error-prone - instead use Sdf-level namespace edit.
        from pxr import Sdf

        layer = stage.GetRootLayer()
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Rename(old_root.GetPath(), "Object"))
        if not layer.Apply(edit):
            raise RuntimeError("prim rename failed")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Object"))

    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(args.mass)

    mesh_count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr("convexHull")
            mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError("no UsdGeom.Mesh prims found - nothing baked")

    stage.GetRootLayer().Save()

    # Verify by re-opening fresh.
    check = Usd.Stage.Open(OUT)
    croot = check.GetDefaultPrim()
    assert croot.GetName() == "Object", croot.GetName()
    assert UsdPhysics.RigidBodyAPI(croot), "RigidBodyAPI missing after bake"
    assert UsdPhysics.MassAPI(croot), "MassAPI missing after bake"
    n = sum(
        1
        for p in Usd.PrimRange(croot)
        if p.IsA(UsdGeom.Mesh)
        and UsdPhysics.CollisionAPI(p)
        and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull"
    )
    assert n == mesh_count, (n, mesh_count)
    print(f"[BAKE] OK: {OUT} root='Object' meshes_with_convex_hull={n} mass={args.mass}kg")


main()
simulation_app.close()
```

- [ ] **Step 2: Run the bake under flock**

Run:
```bash
flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py --die d20 2>&1 | tee /tmp/bake_die_asset.log"
```
Expected final line: `[BAKE] OK: .../assets/dice/d20_physics.usd root='Object' meshes_with_convex_hull=1 mass=0.01kg`
(mesh count 1 expected — the dice-demo runs found exactly 1 mesh prim per
die; if it differs, report the actual count, the assert only requires ≥1
consistency.) If the Sdf rename approach fails on this file, fall back to
creating a wrapper stage: a new USD whose `/Object` Xform prim references
the source file, with schemas applied on `/Object` — record which path was
taken in the report.

- [ ] **Step 3: Record the die-vs-DexCube size delta (spec pre-check)**

Append to the report (numbers, not prose): d20 manifest `size_mm` (read
`vision/data/raw/dice_sets_v1/set_00000_d20.json`), measured bbox from the
dice-demo scale diagnostic (~30mm vertex-to-vertex after 0.001 scale), vs
the recipe's DexCube (`dex_cube_instanceable.usd` at `scale=0.8` — read its
extent from the USD if accessible, else cite Isaac Lab docs value) and this
repo's lift `init_state.pos` z (0.055) / `minimal_height` (0.04). State
whether any event/command range in the recipe encodes object size (from
reading `tasks/franka/lift_env_cfg.py`: `reset_object_position` pose_range
and `object_pose` command ranges are position-only — expected answer: no
size assumptions, only the init z drop height).

- [ ] **Step 4: Commit**

```bash
git add scripts/bake_die_asset.py assets/dice/d20_physics.usd
git commit -m "feat: bake physics-schema d20 asset for joint-space RL lift (Task 1)"
git push origin franka-panda-pivot
```

---

### Task 2: Env cfg subclass + train script variant flag + smoke test

**Files:**
- Create: `tasks/franka/dice_lift_joint_env_cfg.py`
- Modify: `scripts/train_franka.py` (add `--variant` flag; default
  behavior unchanged)

**Interfaces:**
- Consumes: `assets/dice/d20_physics.usd` from Task 1 (default prim
  `Object`, schemas baked, mm-as-m units).
- Produces: `FrankaDieLiftJointEnvCfg` and `FrankaDieLiftJointEnvCfg_PLAY`
  (importable from `tasks.franka.dice_lift_joint_env_cfg`);
  `scripts/train_franka.py --variant joint-die` trains it
  (`--variant ik-cube` = existing default).

- [ ] **Step 1: Write `tasks/franka/dice_lift_joint_env_cfg.py`**

```python
# tasks/franka/dice_lift_joint_env_cfg.py
"""Joint-space (no-IK) d20-die-lift variant of the Franka lift task.

Subclasses tasks/franka/lift_env_cfg.py's FrankaLiftEnvCfg and overrides
exactly two things (the experiment's two variables, per
docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md):

1. arm_action: DifferentialInverseKinematicsActionCfg (task-space IK) ->
   JointPositionActionCfg with scale=0.5, use_default_offset=True - the
   exact values of Isaac Lab's own validated joint_pos lift variant
   (isaaclab_tasks/.../lift/config/franka/joint_pos_env_cfg.py:34-36,
   read directly), which is the only lift variant Isaac Lab ships RL
   agent configs for (see the research doc). No IK anywhere.
2. object: DexCube -> physics-baked d20 die (assets/dice/d20_physics.usd,
   Task 1 of the plan; default prim 'Object' so the stock recipe's
   SceneEntityCfg("object", body_names="Object") terms match unchanged),
   spawn-time scale 0.001 (mm-as-m source units, dice-demo convention),
   same solver-iteration rigid props as the DexCube recipe.

Everything else (rewards, observations, commands, events, terminations,
curriculum, episode length, PPO cfg) inherits byte-identical from
FrankaLiftEnvCfg. Import only after an AppLauncher exists.
"""

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .lift_env_cfg import FrankaLiftEnvCfg

_D20_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "dice",
    "d20_physics.usd",
)


@configclass
class FrankaDieLiftJointEnvCfg(FrankaLiftEnvCfg):
    """d20 lift with direct joint-position arm actions (no IK)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Variable 1: joint-space arm action (exact Isaac Lab joint_pos values).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # gripper_action inherited unchanged (BinaryJointPositionActionCfg).

        # Variable 2: the d20 die replaces the DexCube.
        if not os.path.isfile(_D20_USD):
            raise FileNotFoundError(f"baked die asset missing - run scripts/bake_die_asset.py: {_D20_USD}")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=_D20_USD,
                scale=(0.001, 0.001, 0.001),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


@configclass
class FrankaDieLiftJointEnvCfg_PLAY(FrankaDieLiftJointEnvCfg):
    """Smaller, non-corrupted-observation variant for eval/play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

Note for the implementer: `rigid_props` here DOES take effect (unlike the
dice demo's runtime-patching situation) because Task 1 baked
`RigidBodyAPI` into the USD — `modify_rigid_body_properties` finds an
existing schema to modify. State in your report that you confirmed the
mechanism (read `isaaclab/sim/schemas/schemas.py` if unsure; the dice-demo
task-1 report documents it).

- [ ] **Step 2: Add `--variant` to `scripts/train_franka.py`**

Add to the argparse block (before `AppLauncher.add_app_launcher_args`):

```python
parser.add_argument(
    "--variant",
    choices=["ik-cube", "joint-die"],
    default="ik-cube",
    help=(
        "ik-cube: the existing stock-recipe cube-lift with relative-IK actions (default, unchanged). "
        "joint-die: d20-die lift with direct joint-position actions (no IK) - see "
        "docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md."
    ),
)
```

In `main()`, replace `env_cfg = FrankaLiftEnvCfg()` with:

```python
    if args_cli.variant == "joint-die":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg

        env_cfg = FrankaDieLiftJointEnvCfg()
    else:
        env_cfg = FrankaLiftEnvCfg()
```

And make the log root variant-specific so runs don't interleave (replace
the `log_dir = ...` line):

```python
    log_dir = os.path.join(
        LOG_ROOT if args_cli.variant == "ik-cube" else LOG_ROOT + "_jointdie",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
```

- [ ] **Step 3: Smoke test (bounded, non-headless, flock)**

Run:
```bash
flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die --num_envs 16 --max_iterations 2 2>&1 | tee /tmp/jointdie_smoke.log"
```
Expected in the log: env construction succeeds; the printed manager
summaries list the SAME reward/observation/termination/curriculum terms as
the ik-cube variant (6 reward terms incl. reaching_object/lifting_object);
the action manager shows `JointPositionAction` (NOT
DifferentialInverseKinematicsAction) with 7 arm joints + binary gripper;
2 iterations complete; no NaN/explosion in the printed losses.
Also verify the die settles: no `object_dropping` termination storm in the
episode stats (a die falling through the table would mass-trigger it —
that's the baked-collision failure signature; if seen, the Task 1 bake is
defective, stop and fix there).

- [ ] **Step 4: Confirm the ik-cube default is untouched**

Run: `git diff HEAD -- scripts/train_franka.py | grep -c "^-.*FrankaLiftEnvCfg()"`
plus a 1-iteration ik-cube launch ONLY IF any doubt exists from the diff
(default path reads identically; avoid burning an Isaac launch when the
diff is clearly additive).

- [ ] **Step 5: Commit**

```bash
git add tasks/franka/dice_lift_joint_env_cfg.py scripts/train_franka.py
git commit -m "feat: joint-space (no-IK) d20 die-lift env variant + --variant flag (Task 2)"
git push origin franka-panda-pivot
```

---

### Task 3: 300-iteration diagnostic run

**Files:** none created (run + report only; tee log + TensorBoard events
are the artifacts).

**Interfaces:**
- Consumes: `--variant joint-die` from Task 2.
- Produces: a `logs/train_franka_jointdie/<timestamp>/` run with
  TensorBoard events; the report names the authoritative success metric
  for Task 4's verdict.

- [ ] **Step 1: Launch the diagnostic**

```bash
flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die --num_envs 4096 --max_iterations 300 2>&1 | tee /tmp/jointdie_diag300.log"
```
(Expect roughly the ik-cube variant's wall-clock scale; if a prior thread
holds the lock, flock waits — that is normal, do not kill the holder
without applying the Global Constraints' hang-diagnosis procedure.)

- [ ] **Step 2: Read the TensorBoard scalars directly**

Use the established pattern (read the event file with
`tensorboard.backend.event_processing.event_accumulator` via
`/home/saps/IsaacLab/_isaac_sim/python.sh`, NOT bare python3). Record in
the report, with numbers: `Loss/value_function` (must stay bounded — the
Experiment 11 divergence check; a spike to >1e3 that never recovers is an
automatic stop-and-report), `Episode_Reward/reaching_object` trend,
`Episode_Reward/lifting_object` trend, and the exact name of the
episode-termination/success scalar this env logs (this becomes the
spec's authoritative metric — name it explicitly in the report BEFORE
Task 4 starts).

- [ ] **Step 3: Verdict gate for proceeding**

Proceed to Task 4 iff: value loss bounded AND `reaching_object` clearly
rising (the recipe's reach converges early — flat-at-noise reach after
300 iterations means something structural is wrong; stop and report
rather than launching the full run). `lifting_object` at ~0 after only
300 iterations is NOT a stop signal (the stock recipe's lift emerges
later; this gate only screens for divergence/dead-reach).

- [ ] **Step 4: Commit the report note + ledger line**

```bash
git add -A docs/ 2>/dev/null; git commit -m "docs: joint-die 300-iter diagnostic readout (Task 3)" --allow-empty
git push origin franka-panda-pivot
```
(The tee log and TensorBoard events stay untracked; the report file
carries the numbers.)

---

### Task 4: Full 1500-iteration run + eval video verdict

**Files:**
- Possibly modify: `scripts/franka_checkpoint_review.py` (add the same
  `--variant` switch importing `FrankaDieLiftJointEnvCfg_PLAY`, if the
  script hardcodes the PLAY cfg — read it first; keep the default
  behavior unchanged).

**Interfaces:**
- Consumes: Task 2's env variant, Task 3's named authoritative metric.
- Produces: the experiment verdict per the spec's success criteria.

- [ ] **Step 1: Launch the full run**

```bash
flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die --num_envs 4096 --max_iterations 1500 2>&1 | tee /tmp/jointdie_full1500.log"
```

- [ ] **Step 2: Full-run scalar readout**

Same event-file procedure as Task 3 Step 2, over all 1500 iterations:
value-loss boundedness (all points, not samples), reach/lift/goal-tracking
trends, and the authoritative success metric's trajectory. Numbers in the
report.

- [ ] **Step 3: 10-episode eval with video**

Read `scripts/franka_checkpoint_review.py` first; extend with `--variant
joint-die` (importing `FrankaDieLiftJointEnvCfg_PLAY`) if needed, keeping
its default untouched. Run it non-headless under flock on the final
checkpoint (`logs/train_franka_jointdie/<ts>/model_1499.pt`), capture
video/frames per that script's own conventions, and inspect the video
around grasp moments frame-by-frame (motion continuity — stills lie;
Experiment 16 precedent). Also record the instrumented check: object
height trajectory across eval episodes (the script's printed states or a
direct `root_pos_w` readout), not eyeball-only.

- [ ] **Step 4: Verdict per the spec**

PASS requires BOTH the metric criterion and the video criterion from the
spec's "Success criteria / verdict protocol". If FAILED: the spec
pre-authorizes exactly one fallback rung — swap the object back to the
recipe's DexCube (identical joint-space config otherwise; a one-line
`self.scene.object` revert in a `FrankaCubeLiftJointEnvCfg` subclass),
full run + video, then STOP and report both results (asset-vs-recipe
isolation). No other unauthorized iteration.

- [ ] **Step 5: Commit + final report**

```bash
git add -A tasks/ scripts/ docs/
git commit -m "feat/docs: joint-space die-lift full run + eval verdict (Task 4)"
git push origin franka-panda-pivot
```
Update `.superpowers/sdd/progress.md` (ledger) and write the run report
(`docs/superpowers/plans/2026-07-11-joint-space-die-lift-report.md`):
hypothesis verdict (supported/falsified), all numbers, video findings,
deviations.

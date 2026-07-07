# Experiment 19: Mimic-Joint Asset Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the AR4 gripper's confirmed mimic-joint mechanical defect (the two prismatic jaw joints move independently instead of as one physically-coupled unit), then re-run Experiment 18's exact reward configuration unchanged to isolate whether that defect alone explains three consecutive experiments' null result on `Episode_Reward/lifting_object` (`0/1500` in both Experiment 17 and Experiment 18).

**Architecture:** Two source-file changes (`scripts/build_asset.py` authors a real `PhysxMimicJointAPI` coupling post-import; `tasks/ar4/robot_cfg.py` removes jaw2's independent PD drive so it doesn't fight that coupling), verified with an instrumented rollout before spending any training compute, then an unmodified re-run of `Ar4PickPlacePregraspEnvCfg` (`--pregrasp`, already wired into `scripts/train.py`) through this repo's standard diagnostic + full-run verification sequence.

**Tech Stack:** Isaac Lab / Isaac Sim 107.3.26 (PhysX), USD (`pxr`), rsl_rl PPO.

## Global Constraints

- Always invoke Isaac Lab scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from the repo root — never plain `python`.
- Every subagent dispatched to run a real Isaac Sim job must be given the literal blocking poll command verbatim in its dispatch prompt — not told to "poll" in prose.
- Task 2's jaw-tracking acceptance check is a hard gate: if it fails, do not proceed to Task 3 or Task 4. Report BLOCKED and stop; this is not something to silently route around.
- No reward-function or action-space changes anywhere in this plan. The fix is scoped entirely to `scripts/build_asset.py` (asset authoring) and `tasks/ar4/robot_cfg.py` (actuator config).
- `assets/` is gitignored. Never `git add` anything under it — only the source files that produce it (`scripts/build_asset.py`, `tasks/ar4/robot_cfg.py`).
- No new CLI flag is needed: `--pregrasp` (already wired into `scripts/train.py`/`scripts/eval_loop.py`) loads `Ar4PickPlacePregraspEnvCfg` unchanged; the fix is transparent to it via `robot_cfg.py`'s asset-manifest resolution.

---

### Task 1: Fix the mimic-joint asset defect

**Files:**
- Modify: `scripts/build_asset.py`
- Modify: `tasks/ar4/robot_cfg.py`

**Interfaces:**
- Produces: a rebuilt `assets/ar4_mk5/ar4_mk5.usd` (gitignored output) where `gripper_jaw2_joint` carries a `PhysxMimicJointAPI` referencing `gripper_jaw1_joint` (gearing=1, offset=0), and `tasks/ar4/robot_cfg.py`'s `AR4_MK5_CFG.actuators` has two separate gripper entries (`gripper_jaw1` driven, `gripper_jaw2` passive) instead of one shared entry. Later tasks consume this rebuilt asset transparently (no code changes needed in any task-cfg file).

- [ ] **Step 1: Add the mimic-joint authoring function to `build_asset.py`**

Add this function after `_generate_wedge_usd` (before `def main():`):

```python
def _apply_gripper_mimic_joint(usd_path: str) -> None:
    """Author a PhysxMimicJointAPI on gripper_jaw2_joint, referencing
    gripper_jaw1_joint, so the two prismatic jaw joints move as one
    physically-coupled unit (gearing=1, offset=0) - matching the source
    URDF's <mimic joint="gripper_jaw1_joint" multiplier="1" offset="0"/>
    on gripper_jaw2_joint (ar_gripper_macro.xacro:89), which the URDF
    importer's parse_mimic=True flag does not reliably reproduce (confirmed
    empirically: Experiment 17's Task 6 diagnostic found gripper_jaw2_joint
    drifting 20% past its own commanded position under contact load, while
    gripper_jaw1_joint tracked exactly).

    Applied as a stage-editing post-process on the already-written USD
    (rather than relying further on the importer's own mimic handling,
    the exact mechanism under suspicion). Idempotent: safe to call on a
    USD that already has this API applied (replaces rather than
    duplicates it), so re-running build_asset.py repeatedly is safe.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    jaw1_prim = None
    jaw2_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "gripper_jaw1_joint":
            jaw1_prim = prim
        elif prim.GetName() == "gripper_jaw2_joint":
            jaw2_prim = prim
    if jaw1_prim is None or jaw2_prim is None:
        sys.exit(
            f"Could not find gripper_jaw1_joint/gripper_jaw2_joint prims in {usd_path} "
            f"(found jaw1={jaw1_prim}, jaw2={jaw2_prim}) - URDF joint naming may have changed."
        )

    if jaw2_prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotX"):
        jaw2_prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, "rotX")

    mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(jaw2_prim, UsdPhysics.Tokens.rotX)
    mimic_api.GetReferenceJointRel().AddTarget(jaw1_prim.GetPath())
    mimic_api.GetReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotX)
    mimic_api.GetGearingAttr().Set(1.0)
    mimic_api.GetOffsetAttr().Set(0.0)

    stage.GetRootLayer().Save()
    print(f"Applied PhysxMimicJointAPI to gripper_jaw2_joint (reference: gripper_jaw1_joint) in {usd_path}")
```

- [ ] **Step 2: Call it from `main()`, right after the USD is confirmed written**

In `main()`, immediately after this existing block:

```python
        manifest_path = os.path.join(USD_OUT_DIR, "usd_path.txt")
        with open(manifest_path, "w") as f:
            f.write(output_usd)

        print(f"AR4 mk5 USD asset written to: {output_usd}")
```

add:

```python
        _apply_gripper_mimic_joint(output_usd)
```

(before the existing `_generate_wedge_usd(WEDGE_USD_PATH)` call, which is unrelated and stays as-is).

- [ ] **Step 3: Split the gripper actuator config in `tasks/ar4/robot_cfg.py`**

Replace the current single `"gripper"` entry in `AR4_MK5_CFG`'s `actuators={...}` dict:

```python
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_jaw[12]_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=50.0,
            armature=1e-3,
        ),
```

with two separate entries:

```python
        "gripper_jaw1": ImplicitActuatorCfg(
            joint_names_expr=["gripper_jaw1_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=50.0,
            armature=1e-3,
        ),
        "gripper_jaw2": ImplicitActuatorCfg(
            joint_names_expr=["gripper_jaw2_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            armature=1e-3,
        ),
```

`gripper_jaw1` keeps the original stiffness/damping (it remains the real actuated DOF, matching the PhysX mimic-joint test suite's own reference-joint role, which is the only joint that gets an independent drive). `gripper_jaw2` drops to `stiffness=0.0, damping=0.0` — no independent drive at all, purely passive, position determined solely by the mimic constraint's gearing relationship to `gripper_jaw1`.

- [ ] **Step 4: Rebuild the asset**

```bash
export AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description
/home/saps/IsaacLab/isaaclab.sh -p scripts/build_asset.py
```

Expected: exits with code 0, prints `AR4 mk5 USD asset written to: .../assets/ar4_mk5/ar4_mk5.usd`, then `Applied PhysxMimicJointAPI to gripper_jaw2_joint (reference: gripper_jaw1_joint) in .../assets/ar4_mk5/ar4_mk5.usd`, then `Wedge (triangular prism) USD asset written to: ...`. Confirm freshness:

```bash
date; ls -la assets/ar4_mk5/ar4_mk5.usd assets/ar4_mk5/usd_path.txt
```

Expected: both files' modification times are within the last minute (freshly rebuilt).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_asset.py tasks/ar4/robot_cfg.py
git commit -m "Fix gripper mimic-joint defect: real PhysxMimicJointAPI coupling + passive jaw2 actuator"
```

(Do NOT `git add` anything under `assets/` — it is gitignored.)

---

### Task 2: Verify the fix with an instrumented rollout

**Files:**
- Create: `scripts/mimic_joint_verify.py`

**Interfaces:**
- Consumes: the rebuilt asset from Task 1 (loaded transparently via `tasks/ar4/pickplace_pregrasp_env_cfg.py`'s `Ar4PickPlacePregraspEnvCfg`, which pulls in `robot_cfg.py`'s `AR4_MK5_CFG`), and the existing Experiment 18 checkpoint at `/home/saps/projects/rl/logs/train/2026-07-07_16-38-01/model_1499.pt`.
- Produces: `.superpowers/sdd/task-2-report.md` stating PASS/FAIL against the acceptance threshold below. This is a hard gate for Task 3/4 — do not proceed if it reports FAIL.

- [ ] **Step 1: Write the verification script**

Create `scripts/mimic_joint_verify.py`:

```python
"""Experiment 19 Task 2: instrumented rollout verifying the mimic-joint
fix (Task 1) - do gripper_jaw1_joint and gripper_jaw2_joint now track
each other under contact load, not just at rest?

Adapted from the Experiment 17 Task 6 diagnostic pattern
(exp17_grasp_gate_diagnostic.py): loads the Experiment 18 checkpoint
(the same trained policy, frozen weights) and rolls it out against the
newly-rebuilt USD asset (Task 1's fix), logging both jaw joint positions
and their divergence every step, with running stats restricted to steps
with measurable jaw contact force (not just at rest, where both jaws
naturally agree regardless of whether the fix works).

Baseline to compare against: Experiment 17 Task 6 found gripper_jaw2_joint
drifting to 0.0168 while gripper_jaw1_joint stayed exactly at 0.0140 under
contact load - a 0.0028m (20% of the 0.014m travel range) divergence.
PASS threshold: max divergence during contact under 0.0014m (10% of the
travel range) - a clear, substantial improvement, not just noise.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/mimic_joint_verify.py \
        --checkpoint /home/saps/projects/rl/logs/train/2026-07-07_16-38-01/model_1499.pt --episodes 3
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 19 mimic-joint fix verification.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=3, help="Number of full episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_pregrasp_env_cfg import Ar4PickPlacePregraspEnvCfg  # noqa: E402

# Task 1's fix acceptance threshold: 10% of the 0.014m gripper travel
# range, well below Experiment 17 Task 6's measured 0.0028m (20%) divergence.
PASS_THRESHOLD_M = 0.0014
# Any force reading above this is treated as "real contact," not float noise
# at rest (Experiment 17 Task 6 found contact forces of 7-20N when real
# contact occurred, vs. exactly 0.00000 at rest).
CONTACT_FORCE_EPSILON = 1e-4


def main() -> None:
    env_cfg = Ar4PickPlacePregraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    robot = env.scene["robot"]
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]

    gripper_joint_ids, gripper_joint_names = robot.find_joints(["gripper_jaw1_joint", "gripper_jaw2_joint"])
    print(f"[SETUP] gripper joint ids={gripper_joint_ids} names={gripper_joint_names}")
    print(f"[SETUP] pass_threshold_m={PASS_THRESHOLD_M} contact_force_epsilon={CONTACT_FORCE_EPSILON}")

    stats = {
        "total_steps": 0,
        "contact_steps": 0,
        "max_jaw_pos_diff_at_rest": 0.0,
        "max_jaw_pos_diff_during_contact": 0.0,
        "sum_jaw_pos_diff_during_contact": 0.0,
        "max_jaw1_force": 0.0,
        "max_jaw2_force": 0.0,
    }

    obs = wrapped.get_observations()
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            print(f"[EPISODE {episode} START]")
            for step in range(250):
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)

                jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec).item()
                jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec).item()

                gripper_joint_pos = robot.data.joint_pos[0, gripper_joint_ids].tolist()
                jaw_pos_diff = abs(gripper_joint_pos[0] - gripper_joint_pos[1])

                in_contact = (jaw1_force_mag > CONTACT_FORCE_EPSILON) or (jaw2_force_mag > CONTACT_FORCE_EPSILON)

                stats["total_steps"] += 1
                stats["max_jaw1_force"] = max(stats["max_jaw1_force"], jaw1_force_mag)
                stats["max_jaw2_force"] = max(stats["max_jaw2_force"], jaw2_force_mag)
                if in_contact:
                    stats["contact_steps"] += 1
                    stats["max_jaw_pos_diff_during_contact"] = max(
                        stats["max_jaw_pos_diff_during_contact"], jaw_pos_diff
                    )
                    stats["sum_jaw_pos_diff_during_contact"] += jaw_pos_diff
                else:
                    stats["max_jaw_pos_diff_at_rest"] = max(stats["max_jaw_pos_diff_at_rest"], jaw_pos_diff)

                print(
                    f"[EP {episode} STEP {step:3d}] jaw1_force={jaw1_force_mag:.5f} jaw2_force={jaw2_force_mag:.5f} "
                    f"in_contact={int(in_contact)} jaw1_pos={gripper_joint_pos[0]:.5f} "
                    f"jaw2_pos={gripper_joint_pos[1]:.5f} jaw_pos_diff={jaw_pos_diff:.5f}"
                )

                if bool(dones[0]):
                    print(f"[EP {episode} STEP {step}] episode done (early termination), stopping episode")
                    break
            print(f"[EPISODE {episode} END]")

    mean_diff_during_contact = (
        stats["sum_jaw_pos_diff_during_contact"] / stats["contact_steps"] if stats["contact_steps"] > 0 else None
    )
    result = "PASS" if (
        stats["contact_steps"] > 0 and stats["max_jaw_pos_diff_during_contact"] < PASS_THRESHOLD_M
    ) else "FAIL"

    print(
        "[SUMMARY] "
        f"total_steps={stats['total_steps']} contact_steps={stats['contact_steps']} "
        f"max_jaw_pos_diff_at_rest={stats['max_jaw_pos_diff_at_rest']:.5f} "
        f"max_jaw_pos_diff_during_contact={stats['max_jaw_pos_diff_during_contact']:.5f} "
        f"mean_jaw_pos_diff_during_contact={mean_diff_during_contact} "
        f"max_jaw1_force={stats['max_jaw1_force']:.5f} max_jaw2_force={stats['max_jaw2_force']:.5f}"
    )
    print(f"[RESULT] {result} (threshold={PASS_THRESHOLD_M}m, contact_steps={stats['contact_steps']})")
    print("[DIAGNOSTIC COMPLETE]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run it and capture the log**

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/mimic_joint_verify.py \
    --checkpoint /home/saps/projects/rl/logs/train/2026-07-07_16-38-01/model_1499.pt --episodes 3 \
    2>&1 | tee /tmp/exp19_mimic_verify_stdout.log
```

Expected: script runs to completion (`[DIAGNOSTIC COMPLETE]` printed), no traceback. Find the `[SUMMARY]` and `[RESULT]` lines:

```bash
grep -E "^\[SUMMARY\]|^\[RESULT\]" /tmp/exp19_mimic_verify_stdout.log
```

- [ ] **Step 3: Write the gate report**

Write `.superpowers/sdd/task-2-report.md` stating: the exact `[SUMMARY]`/`[RESULT]` line contents; `max_jaw_pos_diff_during_contact` compared explicitly against Experiment 17 Task 6's recorded `0.0028` m (20% divergence) baseline, stating the percentage improvement; whether `contact_steps > 0` (if zero, this checkpoint produced no contact event at all in this rollout — report this plainly, it means the gate could not be exercised, treat as FAIL per the threshold check's own `contact_steps > 0` requirement, not a silent pass); and the final PASS/FAIL verdict, copied verbatim from the script's own `[RESULT]` line.

**If FAIL:** stop here. Do not proceed to Task 3 or Task 4. Report BLOCKED with the report file path — the controller (not this task) decides what happens next (e.g., whether `stiffness=0.0` was too aggressive, whether a nonzero-but-small stiffness is needed for solver stability, or whether the diagnosis needs revisiting).

**If PASS:** proceed to Task 3.

---

### Task 3: 300-iteration diagnostic run

**Files:**
- None (verification-only task, no commit).

**Interfaces:**
- Consumes: Task 1's fixed asset (transparently, via `--pregrasp`), Task 2's PASS verdict.
- Produces: a diagnostic-scale training run confirming no training-stability regression from the actuator change, before spending the full 1500-iteration budget.

- [ ] **Step 1: Launch**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless --max_iterations 300 > /tmp/exp19_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

`save_interval=50` (confirmed in `tasks/ar4/agents/rsl_rl_ppo_cfg.py`) with `max_iterations=300` produces checkpoints at 0, 50, 100, 150, 200, 250, and a final one at iteration 299 (rsl_rl always saves a final checkpoint at `num_learning_iterations - 1`, confirmed by Experiment 18's own 1500-iteration run producing a `model_1499.pt` final checkpoint outside the 50-interval pattern).

```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_pregrasp_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "diagnostic run complete"
```

Re-issue this exact blocking command across tool calls if a single call's own timeout is hit before the checkpoint appears.

- [ ] **Step 3: Check the three formal gates**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
grep -iE "traceback|error|exception" /tmp/exp19_diagnostic_stdout.log
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Loss/value_function', 'Episode_Reward/lifting_object', 'Episode_Reward/pregrasp_readiness']:
    vals = ea.Scalars(tag)
    nonzero = sum(1 for v in vals if v.value != 0.0)
    print(f'{tag}: points={len(vals)} first={vals[0].value} last={vals[-1].value} max={max(v.value for v in vals)} min={min(v.value for v in vals)} nonzero={nonzero}/{len(vals)}')
"
```

Expected: the `grep` for traceback/error/exception finds nothing; `Loss/value_function` stays small and bounded (comparable to Experiment 18's diagnostic range, `[0.000146, 0.0228]`), no sustained growth; `Episode_Reward/lifting_object`'s nonzero count is reported as-is (not expected to necessarily be nonzero yet at this diagnostic scale — report the actual number either way, this step is a training-stability check, not the experiment's falsifiable answer). Note the exact `Episode_Reward/pregrasp_readiness` trend too, confirming the reward config is genuinely unchanged from Experiment 18 (should still show real nonzero growth, matching Experiment 18's diagnostic result).

If any gate fails (traceback found, or `Loss/value_function` shows sustained unbounded growth): report BLOCKED, do not proceed to Task 4.

If clean: proceed to Task 4.

---

### Task 4: Full 1500-iteration run + report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment19-report.md`

**Interfaces:**
- Consumes: Task 3's clean diagnostic gate.
- Produces: the full-run scalar trajectories and the experiment's answer to its single falsifiable question.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless > /tmp/exp19_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_pregrasp_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```

Expect roughly 15-25 minutes of real wall-clock time (Experiment 18's equivalent run took ~14.3 minutes). Re-issue this exact blocking command across tool calls if a single call's own timeout is hit before the checkpoint appears.

Once found, confirm checkpoint integrity:

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
find "$LATEST" -name "model_*.pt" | wc -l
ls -la "${LATEST}"events.out.tfevents.*
```

Expected: 31 checkpoints (0, 50, ..., 1450, 1499), `model_1499.pt` exists, event file mtime matches the run's actual completion time.

- [ ] **Step 3: Extract full scalar trajectories**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
tags = ['Episode_Reward/reaching_object', 'Episode_Reward/pregrasp_readiness', 'Episode_Reward/lifting_object',
        'Episode_Reward/object_goal_tracking', 'Episode_Reward/object_goal_tracking_fine_grained',
        'Episode_Termination/cube_reached_goal', 'Loss/value_function']
for tag in tags:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(f'=== {tag} ===')
        print('  points:', len(vals), 'first:', vals[0].value, 'last:', vals[-1].value,
              'max:', max(v.value for v in vals), 'min:', min(v.value for v in vals),
              'nonzero:', nonzero, '/', len(vals))
        for i in range(0, len(vals), 150):
            print(f'  iteration={vals[i].step:4d}, value={vals[i].value:.6f}')
        print(f'  iteration={vals[-1].step:4d}, value={vals[-1].value:.6f}')
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-07-ar4-experiment19-report.md` following the exact structure of `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 7 tags above).

**Include a "Fix Verification Recap" section** near the top, summarizing Task 2's `.superpowers/sdd/task-2-report.md` result (the PASS verdict and the measured jaw-position divergence during contact, vs. Experiment 17 Task 6's 0.0028m baseline) — this report is meaningless without that context.

**Include a "Key Comparison" section** against Experiment 18's exact final value and Experiment 17's exact final value:
- Experiment 17 final `Episode_Termination/cube_reached_goal`: 0.002360
- Experiment 18 final `Episode_Termination/cube_reached_goal`: 0.003499

**The report must explicitly and separately answer two questions, each with the actual sampled trajectory as evidence:**

1. Does `Episode_Reward/pregrasp_readiness` still show real, growing nonzero occurrence across the run (confirming the reward configuration is genuinely unchanged from Experiment 18, not an accidental confound)? Compare directly against Experiment 18's exact trajectory (0.000166 → 1.268935, 1500/1500 nonzero).
2. Does `Episode_Reward/lifting_object`'s nonzero count move off exactly `0/1500` at any point in the run — this is the single specific, falsifiable question this experiment exists to answer. Report the exact nonzero count (e.g. "3/1500" or "0/1500"), not just a qualitative description.

**If `lifting_object` is still exactly `0/1500`:** state this plainly as the finding — a clean falsification of the mimic-joint-defect hypothesis (given Task 2 already confirmed the fix itself works). Per this project's established practice, do NOT perform a video-inspection step in this case — the scalar evidence is unambiguous.

**If `lifting_object` DOES show any nonzero occurrences:** do NOT draw a final success/failure conclusion from scalars alone. Explicitly flag in the report that video inspection is now warranted, to be done separately and personally by the controller (not delegated) before any success claim — per this project's Experiment 16 lesson that shaped/gated reward scalars going nonzero is not sufficient evidence of genuine behavior on its own.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment19-report.md
git commit -m "Record Experiment 19 training run: mimic-joint fix isolate re-run"
```

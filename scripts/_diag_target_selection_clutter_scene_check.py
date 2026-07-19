"""One-off diagnostic (Task 1, docs/superpowers/plans/2026-07-19-target-
selection-clutter-implementation.md Step 3): the spec's own two explicit
empirical risk-checks ("Explicit known-weak-points" section) before Stage D2's
config is trusted for real training. Extends
scripts/_diag_d12d20_mixed_env_check.py's exact pattern (build a real, small
live env, no training, cross-check predicted vs. actual live state) to the new
3-die FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg topology (target +
distractor_1 + distractor_2, all real/active - the densest of the 3 curriculum
stages, so also the hardest test of multi-spawner coexistence).

(a) Multi-spawner coexistence: confirm the target's own already-proven
    deterministic round-robin (env_index % 2, random_choice=False) still
    matches BOTH the live observation_manager output AND the live
    USD-authored scale, exactly like _diag_d12d20_mixed_env_check.py already
    verifies for the single-object mixed env - now with two ADDITIONAL
    MultiAssetSpawnerCfg-populated sibling RigidObjects
    (distractor_1/distractor_2, random_choice=True) also present on the same
    scene. Also confirms each distractor's own live scale is one of the two
    known d12/d20 values (spawner worked at all) and that BOTH shapes appear
    across the batch for BOTH distractor slots (random_choice=True is
    actually varying the assignment, not silently stuck on one asset).

(b) Spawn-and-settle: step physics for real (no training) until the scene
    settles, then verify per-env that (1) no entity (target or either
    distractor) ends up off the table / outside the arm's reachable
    workspace, and (2) no two entities' live post-settle positions come
    closer than a safe minimum separation - i.e. the disjoint-lane design
    (module-level comment block in tasks/franka/dice_lift_joint_env_cfg.py
    above FrankaDieLiftTargetSelectionSceneCfg) actually holds up once real
    physics (gravity, any settle-time rolling/tumbling) is applied, not just
    the reset-time nominal ranges.

Both (a) and (b) are reported explicitly PASS/FAIL, not just "ran without
crashing" - matching the plan's own explicit requirement.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_target_selection_clutter_scene_check.py"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Verify FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg's 3-die scene topology."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import torch  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sim.utils import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg  # noqa: E402
from tasks.franka.shape_observations import SHAPE_CLASSES, SHAPE_GEOMETRY_DESCRIPTORS  # noqa: E402

NUM_ENVS = 16
EXPECTED_TARGET_ORDER = ("d12", "d20")
SCALE_TO_SHAPE = {0.001476: "d12", 0.001585: "d20"}
SCALE_TOL = 1e-6

# (a) reachable-workspace bound, generous over CommandsCfg's own
# pos_x=(0.4,0.6)/pos_y=(-0.25,0.25) reference range plus settle-drift margin.
REACH_X = (0.30, 0.70)
REACH_Y = (-0.32, 0.32)
REACH_Z_MIN = -0.10  # well above the shared GroundPlaneCfg (-1.05) and the -0.9 parked height (irrelevant here, both
# distractors are ACTIVE in Stage D2 - this bound is for the 3 real on-table entities only)

# (b) minimum safe pairwise separation post-settle: design estimate is a 70mm
# edge-to-edge lane gap with a ~48mm worst-case combined bounding-diameter
# requirement (see the module-level comment above FrankaDieLiftTargetSelectionSceneCfg)
# - use a materially looser bound here (60mm) as the actual FAIL threshold, since lane
# CENTERS are ~185mm apart and this check is a coarse settle-time sanity net, not a
# re-derivation of the design margin itself.
MIN_SEPARATION_M = 0.06

SETTLE_STEPS = 150  # 150 * dt(0.01) * decimation(2) = 3.0s simulated - ample settle time


def _live_scale(stage, prim_path: str) -> float:
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"no prim at {prim_path}"
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s = op.Get()
            return float(s[0])
    raise AssertionError(f"no scale xformOp found on {prim_path}")


def main() -> None:
    cfg = FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    assert cfg.active_distractor_count == 2, f"expected active_distractor_count=2, got {cfg.active_distractor_count}"
    assert cfg.die_shape_classes_per_env == EXPECTED_TARGET_ORDER

    stage = get_current_stage()

    # ---- Check (a): multi-spawner coexistence ----
    term_names = list(env.observation_manager.active_terms["policy"])
    term_dims = env.observation_manager.group_obs_term_dim["policy"]
    per_term_dim = dict(zip(term_names, term_dims))
    offset = 0
    for name, dims in zip(term_names, term_dims):
        if name == "shape_class":
            shape_class_start = offset
            break
        offset += int(dims[0])
    else:
        raise AssertionError(f"'shape_class' term not found in {term_names}")
    geom_start = shape_class_start + per_term_dim["shape_class"][0]

    full_obs = env.observation_manager.compute()["policy"]
    onehot_block = full_obs[:, shape_class_start : shape_class_start + 4]
    geom_block = full_obs[:, geom_start : geom_start + 1]

    print("=== Check (a): multi-spawner coexistence (target round-robin + distractor spawns) ===")
    print(f"{'env':>3} | {'tgt pred':>8} | {'tgt obs':>7} | {'tgt geom':>8} | {'tgt scale':>9} | {'d1 shape':>8} | "
          f"{'d2 shape':>8} | OK?")
    a_ok = True
    d1_shapes_seen = set()
    d2_shapes_seen = set()
    for env_idx in range(NUM_ENVS):
        predicted_target = EXPECTED_TARGET_ORDER[env_idx % len(EXPECTED_TARGET_ORDER)]
        obs_shape_idx = int(onehot_block[env_idx].argmax().item())
        obs_shape = SHAPE_CLASSES[obs_shape_idx]
        obs_geom = geom_block[env_idx, 0].item()

        tgt_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Object")
        d1_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Distractor1")
        d2_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Distractor2")

        d1_shape = next((s for sc, s in SCALE_TO_SHAPE.items() if abs(d1_scale - sc) < SCALE_TOL), None)
        d2_shape = next((s for sc, s in SCALE_TO_SHAPE.items() if abs(d2_scale - sc) < SCALE_TOL), None)
        if d1_shape is not None:
            d1_shapes_seen.add(d1_shape)
        if d2_shape is not None:
            d2_shapes_seen.add(d2_shape)

        expected_tgt_scale = next(sc for sc, s in SCALE_TO_SHAPE.items() if s == predicted_target)
        expected_geom = SHAPE_GEOMETRY_DESCRIPTORS[predicted_target]

        ok = (
            obs_shape == predicted_target
            and abs(tgt_scale - expected_tgt_scale) < SCALE_TOL
            and abs(obs_geom - expected_geom) < 1e-4
            and d1_shape is not None
            and d2_shape is not None
        )
        a_ok = a_ok and ok
        print(
            f"{env_idx:>3} | {predicted_target:>8} | {obs_shape:>7} | {obs_geom:>8.6f} | {tgt_scale:>9.6f} | "
            f"{d1_shape or 'INVALID':>8} | {d2_shape or 'INVALID':>8} | {'OK' if ok else 'MISMATCH'}"
        )

    variety_ok = len(d1_shapes_seen) == 2 and len(d2_shapes_seen) == 2
    print(f"\ndistractor_1 shapes seen: {sorted(d1_shapes_seen)} | distractor_2 shapes seen: {sorted(d2_shapes_seen)}")
    print(f"Check (a) per-env target/distractor-spawn-validity: {'PASS' if a_ok else 'FAIL'}")
    print(f"Check (a) random_choice=True variety (both shapes appear on both distractor slots): "
          f"{'PASS' if variety_ok else 'FAIL'}")

    # ---- Check (b): spawn-and-settle ----
    print(f"\n=== Check (b): spawn-and-settle ({SETTLE_STEPS} steps) ===")
    zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(SETTLE_STEPS):
        env.step(zero_actions)

    obj_pos = env.scene["object"].data.root_pos_w.clone()
    d1_pos = env.scene["distractor_1"].data.root_pos_w.clone()
    d2_pos = env.scene["distractor_2"].data.root_pos_w.clone()
    env_origins = env.scene.env_origins

    b_ok = True
    print(f"{'env':>3} | {'obj local xyz':>24} | {'d1 local xyz':>24} | {'d2 local xyz':>24} | in-reach? | min-sep(m) | OK?")
    for env_idx in range(NUM_ENVS):
        origin = env_origins[env_idx]
        obj_local = (obj_pos[env_idx] - origin).tolist()
        d1_local = (d1_pos[env_idx] - origin).tolist()
        d2_local = (d2_pos[env_idx] - origin).tolist()

        def in_reach(p):
            return REACH_X[0] <= p[0] <= REACH_X[1] and REACH_Y[0] <= p[1] <= REACH_Y[1] and p[2] >= REACH_Z_MIN

        reach_ok = in_reach(obj_local) and in_reach(d1_local) and in_reach(d2_local)

        d_obj_d1 = ((obj_pos[env_idx] - d1_pos[env_idx]) ** 2).sum().sqrt().item()
        d_obj_d2 = ((obj_pos[env_idx] - d2_pos[env_idx]) ** 2).sum().sqrt().item()
        d_d1_d2 = ((d1_pos[env_idx] - d2_pos[env_idx]) ** 2).sum().sqrt().item()
        min_sep = min(d_obj_d1, d_obj_d2, d_d1_d2)
        sep_ok = min_sep >= MIN_SEPARATION_M

        ok = reach_ok and sep_ok
        b_ok = b_ok and ok
        fmt = lambda p: f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
        print(
            f"{env_idx:>3} | {fmt(obj_local):>24} | {fmt(d1_local):>24} | {fmt(d2_local):>24} | "
            f"{'yes' if reach_ok else 'NO':>9} | {min_sep:>10.4f} | {'OK' if ok else 'FAIL'}"
        )

    print(f"\nCheck (b) spawn-and-settle: {'PASS' if b_ok else 'FAIL'}")

    env.close()
    simulation_app.close()

    overall_ok = a_ok and variety_ok and b_ok
    print(f"\n=== OVERALL: {'PASS' if overall_ok else 'FAIL'} ===")
    if not overall_ok:
        raise SystemExit("Target-selection clutter scene diagnostic FAILED - see checks above.")


if __name__ == "__main__":
    main()

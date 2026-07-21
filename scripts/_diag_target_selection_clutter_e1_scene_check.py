"""One-off diagnostic (Task 2, docs/superpowers/plans/2026-07-21-target-
selection-clutter-e1-3distractors-implementation.md Step 2/3): the spec's
own required-not-optional live spawn-and-settle check ("Explicit
known-weak-points" section) before Stage E1's new 2x2-grid scene topology
is trusted for any training spend. Extends
scripts/_diag_target_selection_clutter_scene_check.py's exact pattern
(predicted-vs-live cross-check + spawn-and-settle, build a real small live
env, no training) from the K=2/3-die topology to the new
FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg 4-die topology (target +
distractor_1 + distractor_2 + distractor_3, all real/active).

(a) Multi-spawner coexistence: confirm the target's own already-proven
    deterministic round-robin (env_index % 2, random_choice=False) still
    matches BOTH the live observation_manager output AND the live
    USD-authored scale, exactly like the K=2 diagnostic already verifies -
    now with THREE additional MultiAssetSpawnerCfg-populated sibling
    RigidObjects (distractor_1/distractor_2/distractor_3, random_choice=True)
    also present on the same scene. Also confirms each distractor's own live
    scale is one of the two known d12/d20 values and that BOTH shapes appear
    across the batch for ALL THREE distractor slots.

    Also checks the observation space is exactly 44-dim (41 base +
    3 distractor_distance_summary_3 columns) - the live, 4-entity-scene
    confirmation Task 1 of this plan explicitly deferred to this diagnostic
    (Task 1's own local-only import check could not build a real env, since
    distractor_3 as a scene entity did not exist until this task).

(b) Spawn-and-settle: step physics for real (no training) until the scene
    settles, then verify per-env that (1) no entity (target or any of the 3
    distractors) ends up off the table / outside the arm's reachable
    workspace, and (2) no two entities' live post-settle positions come
    closer than a safe minimum separation (60mm) - i.e. the 2x2-grid design
    (module-level comment block in tasks/franka/dice_lift_joint_env_cfg.py
    above FrankaDieLiftTargetSelectionE1SceneCfg) actually holds up once
    real physics (gravity, any settle-time rolling/tumbling) is applied, not
    just the reset-time nominal ranges. Reports the actual measured minimum
    pairwise separation for same-row (A-B, C-D), same-column (A-C, B-D), and
    diagonal (A-D, B-C) pairs explicitly - the spec's own nominal estimates
    are 70mm/80mm/106mm respectively; this is the live check that confirms
    or refutes those design-time numbers, not a re-assertion of them.

Both (a) and (b) are reported explicitly PASS/FAIL, not just "ran without
crashing" - matching the K=2 diagnostic's own convention.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_target_selection_clutter_e1_scene_check.py"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Verify FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg's 4-die 2x2-grid scene topology."
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

from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg  # noqa: E402
from tasks.franka.shape_observations import SHAPE_CLASSES, SHAPE_GEOMETRY_DESCRIPTORS  # noqa: E402

NUM_ENVS = 16
EXPECTED_TARGET_ORDER = ("d12", "d20")
SCALE_TO_SHAPE = {0.001476: "d12", 0.001585: "d20"}
SCALE_TOL = 1e-6
EXPECTED_POLICY_OBS_DIM = 44

# (a) reachable-workspace bound, generous over the 2x2 grid's own nominal
# range (near row x in [0.40,0.46], far row x in [0.54,0.60], left column y
# in [-0.215,-0.035], right column y in [0.035,0.215]) plus settle-drift
# margin - reused verbatim from the K=2 diagnostic's own margin choice
# (there, ~0.10m over CommandsCfg's pos_x=(0.4,0.6) and ~0.07m over
# pos_y=(-0.25,0.25)), which already generously covers both grid rows/
# columns here too.
REACH_X = (0.30, 0.70)
REACH_Y = (-0.32, 0.32)
REACH_Z_MIN = -0.10  # well above the shared GroundPlaneCfg (-1.05); all 4 entities are ACTIVE in Stage E1.

# (b) minimum safe pairwise separation post-settle - same 60mm floor the K=2
# diagnostic already used, a coarse settle-time sanity net (not a
# re-derivation of the grid's own nominal 70mm/80mm/106mm design margins).
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
    cfg = FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    assert cfg.active_distractor_count == 3, f"expected active_distractor_count=3, got {cfg.active_distractor_count}"
    assert cfg.die_shape_classes_per_env == EXPECTED_TARGET_ORDER

    stage = get_current_stage()

    # ---- Observation-dim check (Task 1's own deferred live check) ----
    term_names = list(env.observation_manager.active_terms["policy"])
    term_dims = env.observation_manager.group_obs_term_dim["policy"]
    per_term_dim = dict(zip(term_names, term_dims))
    total_dim = sum(int(d[0]) for d in term_dims)
    dist_term_dim = int(per_term_dim["distractor_distance_summary"][0])

    print("=== Observation-dim check (Task 1 deferred live check) ===")
    print(f"policy terms: {term_names}")
    print(f"total policy obs dim: {total_dim} (expected {EXPECTED_POLICY_OBS_DIM})")
    print(f"distractor_distance_summary term dim: {dist_term_dim} (expected 3)")
    dim_ok = total_dim == EXPECTED_POLICY_OBS_DIM and dist_term_dim == 3
    print(f"Observation-dim check: {'PASS' if dim_ok else 'FAIL'}")

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

    # ---- Check (a): multi-spawner coexistence ----
    print("\n=== Check (a): multi-spawner coexistence (target round-robin + 3 distractor spawns) ===")
    print(f"{'env':>3} | {'tgt pred':>8} | {'tgt obs':>7} | {'tgt geom':>8} | {'tgt scale':>9} | {'d1 shape':>8} | "
          f"{'d2 shape':>8} | {'d3 shape':>8} | OK?")
    a_ok = True
    d1_shapes_seen = set()
    d2_shapes_seen = set()
    d3_shapes_seen = set()
    for env_idx in range(NUM_ENVS):
        predicted_target = EXPECTED_TARGET_ORDER[env_idx % len(EXPECTED_TARGET_ORDER)]
        obs_shape_idx = int(onehot_block[env_idx].argmax().item())
        obs_shape = SHAPE_CLASSES[obs_shape_idx]
        obs_geom = geom_block[env_idx, 0].item()

        tgt_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Object")
        d1_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Distractor1")
        d2_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Distractor2")
        d3_scale = _live_scale(stage, f"/World/envs/env_{env_idx}/Distractor3")

        d1_shape = next((s for sc, s in SCALE_TO_SHAPE.items() if abs(d1_scale - sc) < SCALE_TOL), None)
        d2_shape = next((s for sc, s in SCALE_TO_SHAPE.items() if abs(d2_scale - sc) < SCALE_TOL), None)
        d3_shape = next((s for sc, s in SCALE_TO_SHAPE.items() if abs(d3_scale - sc) < SCALE_TOL), None)
        if d1_shape is not None:
            d1_shapes_seen.add(d1_shape)
        if d2_shape is not None:
            d2_shapes_seen.add(d2_shape)
        if d3_shape is not None:
            d3_shapes_seen.add(d3_shape)

        expected_tgt_scale = next(sc for sc, s in SCALE_TO_SHAPE.items() if s == predicted_target)
        expected_geom = SHAPE_GEOMETRY_DESCRIPTORS[predicted_target]

        ok = (
            obs_shape == predicted_target
            and abs(tgt_scale - expected_tgt_scale) < SCALE_TOL
            and abs(obs_geom - expected_geom) < 1e-4
            and d1_shape is not None
            and d2_shape is not None
            and d3_shape is not None
        )
        a_ok = a_ok and ok
        print(
            f"{env_idx:>3} | {predicted_target:>8} | {obs_shape:>7} | {obs_geom:>8.6f} | {tgt_scale:>9.6f} | "
            f"{d1_shape or 'INVALID':>8} | {d2_shape or 'INVALID':>8} | {d3_shape or 'INVALID':>8} | "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    variety_ok = len(d1_shapes_seen) == 2 and len(d2_shapes_seen) == 2 and len(d3_shapes_seen) == 2
    print(
        f"\ndistractor_1 shapes seen: {sorted(d1_shapes_seen)} | distractor_2 shapes seen: {sorted(d2_shapes_seen)} | "
        f"distractor_3 shapes seen: {sorted(d3_shapes_seen)}"
    )
    print(f"Check (a) per-env target/distractor-spawn-validity: {'PASS' if a_ok else 'FAIL'}")
    print(f"Check (a) random_choice=True variety (both shapes appear on all 3 distractor slots): "
          f"{'PASS' if variety_ok else 'FAIL'}")

    # ---- Check (b): spawn-and-settle ----
    print(f"\n=== Check (b): spawn-and-settle ({SETTLE_STEPS} steps) ===")
    zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(SETTLE_STEPS):
        env.step(zero_actions)

    obj_pos = env.scene["object"].data.root_pos_w.clone()
    d1_pos = env.scene["distractor_1"].data.root_pos_w.clone()
    d2_pos = env.scene["distractor_2"].data.root_pos_w.clone()
    d3_pos = env.scene["distractor_3"].data.root_pos_w.clone()
    env_origins = env.scene.env_origins

    def dist(p, q, env_idx):
        return ((p[env_idx] - q[env_idx]) ** 2).sum().sqrt().item()

    b_ok = True
    same_row_seps = []  # A-B (object-d1), C-D (d2-d3)
    same_col_seps = []  # A-C (object-d2), B-D (d1-d3)
    diagonal_seps = []  # A-D (object-d3), B-C (d1-d2)

    print(
        f"{'env':>3} | {'obj local xyz':>24} | {'d1 local xyz':>24} | {'d2 local xyz':>24} | {'d3 local xyz':>24} | "
        f"in-reach? | min-sep(m) | OK?"
    )
    for env_idx in range(NUM_ENVS):
        origin = env_origins[env_idx]
        obj_local = (obj_pos[env_idx] - origin).tolist()
        d1_local = (d1_pos[env_idx] - origin).tolist()
        d2_local = (d2_pos[env_idx] - origin).tolist()
        d3_local = (d3_pos[env_idx] - origin).tolist()

        def in_reach(p):
            return REACH_X[0] <= p[0] <= REACH_X[1] and REACH_Y[0] <= p[1] <= REACH_Y[1] and p[2] >= REACH_Z_MIN

        reach_ok = in_reach(obj_local) and in_reach(d1_local) and in_reach(d2_local) and in_reach(d3_local)

        d_obj_d1 = dist(obj_pos, d1_pos, env_idx)  # A-B, same-row
        d_d2_d3 = dist(d2_pos, d3_pos, env_idx)  # C-D, same-row
        d_obj_d2 = dist(obj_pos, d2_pos, env_idx)  # A-C, same-column
        d_d1_d3 = dist(d1_pos, d3_pos, env_idx)  # B-D, same-column
        d_obj_d3 = dist(obj_pos, d3_pos, env_idx)  # A-D, diagonal
        d_d1_d2 = dist(d1_pos, d2_pos, env_idx)  # B-C, diagonal

        same_row_seps.extend([d_obj_d1, d_d2_d3])
        same_col_seps.extend([d_obj_d2, d_d1_d3])
        diagonal_seps.extend([d_obj_d3, d_d1_d2])

        min_sep = min(d_obj_d1, d_d2_d3, d_obj_d2, d_d1_d3, d_obj_d3, d_d1_d2)
        sep_ok = min_sep >= MIN_SEPARATION_M

        ok = reach_ok and sep_ok
        b_ok = b_ok and ok
        fmt = lambda p: f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
        print(
            f"{env_idx:>3} | {fmt(obj_local):>24} | {fmt(d1_local):>24} | {fmt(d2_local):>24} | {fmt(d3_local):>24} | "
            f"{'yes' if reach_ok else 'NO':>9} | {min_sep:>10.4f} | {'OK' if ok else 'FAIL'}"
        )

    print("\n=== Measured pairwise separations vs. spec's nominal design-time estimates ===")
    print(f"same-row (A-B, C-D):    min={min(same_row_seps):.4f}m  max={max(same_row_seps):.4f}m  (nominal 0.070m)")
    print(f"same-column (A-C, B-D): min={min(same_col_seps):.4f}m  max={max(same_col_seps):.4f}m  (nominal 0.080m)")
    print(f"diagonal (A-D, B-C):    min={min(diagonal_seps):.4f}m  max={max(diagonal_seps):.4f}m  (nominal 0.106m)")
    print(f"\nCheck (b) spawn-and-settle: {'PASS' if b_ok else 'FAIL'}")

    # Compute and print the OVERALL verdict BEFORE env.close()/simulation_app.close() -
    # found live (this task): simulation_app.close() appears to terminate the process
    # before any subsequent Python lines run (process exits fully, GPU idle, no
    # traceback - not the known Isaac-Sim teardown-HANG failure mode documented in
    # CLAUDE.md, which leaves the process alive; this is closer to close() itself
    # ending execution), so a print placed after it never reached the log in a real
    # cloud run of this script. Ordering the verdict first is a real, load-bearing fix,
    # not just defensive style.
    overall_ok = dim_ok and a_ok and variety_ok and b_ok
    print(f"\n=== OVERALL: {'PASS' if overall_ok else 'FAIL'} ===")
    overall_failed = not overall_ok

    env.close()
    simulation_app.close()

    if overall_failed:
        raise SystemExit("Target-selection clutter E1 scene diagnostic FAILED - see checks above.")


if __name__ == "__main__":
    main()

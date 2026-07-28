"""One-off selection script (2026-07-28, ar4-pedestal-ground-clearance-fix
task): picks 3 genuinely distinct GRASP points (different bearings, not just
different joint_1 sweeps of the identical Stage-A survivor) from the
now-non-empty pedestal-height-corrected sweep
(scripts/ar4_graspable_workspace.py, PEDESTAL_HEIGHT_M=0.040), then derives a
matching PREGRASP hover config for each via the SAME local-Gaussian-
perturbation-search method scripts/_ar4_pregrasp_search_roll_constrained.py
already established (joint_1 fixed at the GRASP point's own value, hover
height = GRASP_AT_HEIGHT + 0.05m, same tilt/roll/margin filters) - not
re-derived, just applied at the new pedestal height and to 3 new points
instead of re-typing the same method by hand for each.

Not part of the standing pipeline - a throwaway search/selection script,
whose output gets hardcoded into the real grasp+lift confirm script
(scripts/ar4_pedestal_grasp_confirm.py), same convention as the pre-pedestal
precedent scripts.
"""
import math
import os
import sys

import numpy as np

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL_ROOT)

from scripts.ar4_graspable_workspace import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRASP_AT_HEIGHT,
    JOINT_LIMITS_RAD,
    PEDESTAL_HEIGHT_M,
    ROLL_TOL_DEG,
    TILT_TOL_DEG,
    _evaluate,
    base_to_world,
    run_sweep,
)

HOVER_HEIGHT = GRASP_AT_HEIGHT + 0.05
HEIGHT_TOL_M = 0.002
MARGIN_MIN_RAD = 0.25


def search_pregrasp(grasp_q_deg, seed, n_total=2_000_000, chunk=200_000, sigma_deg=15.0, label=""):
    grasp_q_rad = [math.radians(d) for d in grasp_q_deg]
    rng = np.random.default_rng(seed)
    sigma_rad = math.radians(sigma_deg)

    seed_2to6 = grasp_q_rad[1:]
    pinch_pos_chunks, tilt_chunks, margins_chunks, roll_chunks = [], [], [], []
    joint_chunks = {n: [] for n in ARM_JOINT_NAMES}
    n_done = 0
    while n_done < n_total:
        n = min(chunk, n_total - n_done)
        joint_values = {"joint_1": np.full(n, grasp_q_rad[0])}
        for i, name in enumerate(ARM_JOINT_NAMES[1:]):
            lo, hi = JOINT_LIMITS_RAD[name]
            samples = seed_2to6[i] + rng.normal(0.0, sigma_rad, size=n)
            joint_values[name] = np.clip(samples, lo, hi)

        pinch_pos_c, tilt_deg_c, margins_c, roll_offset_deg_c, ground_clearance_c, fingertip_z_c = _evaluate(joint_values)
        # 2026-07-28 (same-day pedestal-fix correction): height now measured
        # on the REAL fingertip (fingertip_z_c), not the abstract pinch
        # point - see ar4_graspable_workspace.py's own GROUND_CLEARANCE_MIN_M-
        # section comment for the full story of why this matters.
        height_err_c = fingertip_z_c - HOVER_HEIGHT
        margin_min_c = margins_c.min(axis=1)
        mask_c = (
            (np.abs(height_err_c) <= HEIGHT_TOL_M)
            & (tilt_deg_c <= TILT_TOL_DEG)
            & (margin_min_c >= MARGIN_MIN_RAD)
            & (roll_offset_deg_c <= ROLL_TOL_DEG)
        )
        if mask_c.sum() > 0:
            pinch_pos_chunks.append(pinch_pos_c[mask_c])
            tilt_chunks.append(tilt_deg_c[mask_c])
            margins_chunks.append(margins_c[mask_c])
            roll_chunks.append(roll_offset_deg_c[mask_c])
            for name in ARM_JOINT_NAMES:
                joint_chunks[name].append(joint_values[name][mask_c])
        n_done += n

    n_survivors = sum(len(c) for c in tilt_chunks)
    print(f"  [{label} PREGRASP search] Survivors: {n_survivors} / {n_total}")
    if n_survivors == 0:
        print(f"  [{label} PREGRASP search] No survivors - widen sigma or tolerances.")
        return None

    pinch_pos = np.concatenate(pinch_pos_chunks)
    tilt_deg = np.concatenate(tilt_chunks)
    margins = np.concatenate(margins_chunks)
    roll_offset_deg = np.concatenate(roll_chunks)
    joint_values = {name: np.concatenate(joint_chunks[name]) for name in ARM_JOINT_NAMES}
    margin_min = margins.min(axis=1)

    grasp_single = {name: np.array([v]) for name, v in zip(ARM_JOINT_NAMES, grasp_q_rad)}
    grasp_pinch_pos, _, _, _, _, _ = _evaluate(grasp_single)
    world_xy_grasp = base_to_world(grasp_pinch_pos)[0, :2]

    world_xy = base_to_world(pinch_pos)[:, :2]
    xy_dist_mm = np.linalg.norm(world_xy - world_xy_grasp, axis=1) * 1000.0

    close_mask = xy_dist_mm <= 2.0  # mm
    candidate_pool = np.flatnonzero(close_mask) if close_mask.sum() > 0 else np.arange(n_survivors)
    pool_tilt = tilt_deg[candidate_pool]
    pool_margin = margin_min[candidate_pool]

    healthy_margin_mask = pool_margin >= math.radians(20.0)
    if healthy_margin_mask.sum() > 0:
        sub_pool = np.flatnonzero(healthy_margin_mask)
        best_in_sub = sub_pool[np.argmin(pool_tilt[sub_pool])]
        best_idx = candidate_pool[best_in_sub]
    else:
        best_idx = candidate_pool[np.argmin(pool_tilt)]

    chosen_joints_deg = [math.degrees(float(joint_values[name][best_idx])) for name in ARM_JOINT_NAMES]
    print(f"  [{label} PREGRASP] world(x,y)={world_xy[best_idx]} (grasp xy={world_xy_grasp}, dist={xy_dist_mm[best_idx]:.3f}mm)")
    print(f"  [{label} PREGRASP] tilt={tilt_deg[best_idx]:.2f}deg roll_offset={roll_offset_deg[best_idx]:.2f}deg min_margin={math.degrees(margin_min[best_idx]):.2f}deg")
    print(f"  [{label} PREGRASP_Q_DEG] = {chosen_joints_deg}")
    return chosen_joints_deg


def main():
    print(f"GRASP_AT_HEIGHT={GRASP_AT_HEIGHT:.4f}m (PEDESTAL_HEIGHT_M={PEDESTAL_HEIGHT_M:.3f}m), HOVER_HEIGHT={HOVER_HEIGHT:.4f}m")
    print("Running full sweep (8M Stage A + 145-step Stage B)...")
    result = run_sweep(n_stage_a=8_000_000, n_joint1_steps=145, seed=0, chunk_size=200_000)
    if result is None:
        print("EMPTY WORKSPACE even with pedestal - aborting point selection.")
        sys.exit(1)

    world_xy = result["world_xy"]
    margin_min = result["margin_min"]
    ground_clearance_m = result["ground_clearance_m"]
    fingertip_z_m = result["fingertip_z_m"]
    roll_offset_deg = result["roll_offset_deg"]
    tilt_deg = result["tilt_deg"]
    joint_values = result["joint_values"]
    bearing_deg = np.degrees(np.arctan2(world_xy[:, 1], world_xy[:, 0]))

    # 3 distinct target bearings, spread ~14deg apart around the scene's own
    # "straight ahead" world +Y direction (bearing=90deg) - same spread
    # convention as the pre-pedestal P0(~94)/P1(~108)/P2(~98) points, chosen
    # again here fresh (not reused) since the pedestal changes GRASP_AT_HEIGHT
    # and therefore the whole survivor population.
    target_bearings = {"Q0_bearing90": 90.0, "Q1_bearing104": 104.0, "Q2_bearing76": 76.0}
    window_deg = 6.0

    points = {}
    for label, target_bearing in target_bearings.items():
        bearing_err = np.abs(((bearing_deg - target_bearing + 180.0) % 360.0) - 180.0)
        window = window_deg
        mask = bearing_err <= window
        while mask.sum() == 0 and window < 30.0:
            window += 2.0
            mask = bearing_err <= window
        if mask.sum() == 0:
            print(f"[{label}] NO survivors near bearing={target_bearing}deg even at +/-30deg - skipping")
            continue
        pool = np.flatnonzero(mask)
        best_in_pool = pool[np.argmax(margin_min[pool])]
        grasp_q_deg = [math.degrees(float(joint_values[name][best_in_pool])) for name in ARM_JOINT_NAMES]
        print(f"\n=== {label} (target bearing={target_bearing}deg, window used=+/-{window:.0f}deg) ===")
        print(f"  world(x,y) = ({world_xy[best_in_pool, 0]:.4f}, {world_xy[best_in_pool, 1]:.4f})  bearing={bearing_deg[best_in_pool]:.1f}deg")
        print(f"  tilt={tilt_deg[best_in_pool]:.2f}deg roll_offset={roll_offset_deg[best_in_pool]:.2f}deg min_margin={math.degrees(margin_min[best_in_pool]):.2f}deg")
        print(f"  jaw1_origin_ground_clearance={ground_clearance_m[best_in_pool]*1000:.2f}mm  "
              f"REAL_fingertip_clearance_above_pedestal_top={(fingertip_z_m[best_in_pool] - PEDESTAL_HEIGHT_M)*1000:.2f}mm")
        print(f"  GRASP_Q_DEG = {grasp_q_deg}")
        pregrasp_q_deg = search_pregrasp(grasp_q_deg, seed=hash(label) % (2**31), label=label)
        points[label] = {
            "cube_xy": (float(world_xy[best_in_pool, 0]), float(world_xy[best_in_pool, 1])),
            "grasp_q_deg": grasp_q_deg,
            "pregrasp_q_deg": pregrasp_q_deg,
            "ground_clearance_mm": float(ground_clearance_m[best_in_pool] * 1000.0),
            "bearing_deg": float(bearing_deg[best_in_pool]),
        }

    print("\n\n=== FINAL POINTS (python dict, paste into confirm script) ===")
    import json
    print(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()

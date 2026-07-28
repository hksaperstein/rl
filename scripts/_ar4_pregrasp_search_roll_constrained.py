"""One-off local Gaussian perturbation search for a PREGRASP hover config
matching the NEW roll-constrained GRASP_Q from scripts/ar4_graspable_workspace.py
(2026-07-27, ar4-graspable-roll-constraint task). Reuses that script's own
batch_fk_link6/_evaluate machinery (not re-derived) - same method the
original (pre-roll-constraint) PREGRASP_Q search used per
scripts/ar4_graspable_workspace_confirm.py's own docstring: "local Gaussian
perturbation search around GRASP_Q's own joint_2-6 values, joint_1 held fixed
... hover height = GRASP_AT_HEIGHT + 0.05m", now with the SAME roll
constraint applied (a hover waypoint with the wrong jaw heading would
re-introduce exactly the bug this task is fixing, just one phase earlier).

Not part of the standing pipeline - a throwaway search script, output values
get hardcoded into ar4_graspable_workspace_confirm_numeric.py same as before.
"""
import math
import os
import sys

import numpy as np

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL_ROOT)

from scripts.ar4_graspable_workspace import (  # noqa: E402
    ARM_JOINT_NAMES,
    JOINT_LIMITS_RAD,
    ROLL_TOL_DEG,
    TILT_TOL_DEG,
    _evaluate,
    base_to_world,
)

GRASP_AT_HEIGHT = 0.0105
HOVER_HEIGHT = GRASP_AT_HEIGHT + 0.05
HEIGHT_TOL_M = 0.002
MARGIN_MIN_RAD = 0.25


def search_pregrasp(grasp_q_deg, seed=42, n_total=2_000_000, chunk=200_000, sigma_deg=15.0, label=""):
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

        pinch_pos_c, tilt_deg_c, margins_c, roll_offset_deg_c = _evaluate(joint_values)
        height_err_c = pinch_pos_c[:, 2] - HOVER_HEIGHT
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
    print(f"[{label}] Survivors: {n_survivors} / {n_total}")
    if n_survivors == 0:
        print(f"[{label}] No survivors - widen sigma or tolerances.")
        return None

    pinch_pos = np.concatenate(pinch_pos_chunks)
    tilt_deg = np.concatenate(tilt_chunks)
    margins = np.concatenate(margins_chunks)
    roll_offset_deg = np.concatenate(roll_chunks)
    joint_values = {name: np.concatenate(joint_chunks[name]) for name in ARM_JOINT_NAMES}
    height_err = pinch_pos[:, 2] - HOVER_HEIGHT
    margin_min = margins.min(axis=1)

    grasp_single = {name: np.array([v]) for name, v in zip(ARM_JOINT_NAMES, grasp_q_rad)}
    grasp_pinch_pos, _, _, _ = _evaluate(grasp_single)
    world_xy_grasp = base_to_world(grasp_pinch_pos)[0, :2]

    world_xy = base_to_world(pinch_pos)[:, :2]
    xy_dist_mm = np.linalg.norm(world_xy - world_xy_grasp, axis=1) * 1000.0

    close_mask = xy_dist_mm <= 2.0  # mm
    candidate_pool = np.flatnonzero(close_mask) if close_mask.sum() > 0 else np.arange(n_survivors)
    print(f"[{label}] Close-xy (<=2mm) candidate pool size: {len(candidate_pool)}")
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
    print(f"[{label}] world (x,y) = {world_xy[best_idx]}  (grasp world xy = {world_xy_grasp}, dist={xy_dist_mm[best_idx]:.4f}mm)")
    print(f"[{label}] height error vs hover target: {height_err[best_idx]*1000:.3f}mm")
    print(f"[{label}] tilt: {tilt_deg[best_idx]:.2f} deg  roll offset: {roll_offset_deg[best_idx]:.2f} deg (tol={ROLL_TOL_DEG})")
    print(f"[{label}] min joint margin: {math.degrees(margin_min[best_idx]):.2f} deg")
    print(f"[{label}] PREGRASP_Q_DEG = {chosen_joints_deg}")
    return chosen_joints_deg


if __name__ == "__main__":
    # The NEW roll-constrained recommended GRASP_Q, plus 2 additional
    # genuinely distinct Stage-A survivors (2026-07-27 robustness-validation
    # addition - different joint_2-6, not just a different joint_1 sweep of
    # the same underlying config) selected from the same re-swept region -
    # see this task's own report for full provenance of these 3 points.
    POINTS = {
        "P0_recommended": [-4.32433486449247, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994],
        "P1_bearing108": [-19.45950689021618, 62.35207729321578, 23.688773948380668, -75.37680902627864, 4.445222138626336, -86.48635560483278],
        "P2_bearing98": [-8.648669728984965, 62.53091830515946, 21.273536133775046, 91.73174064021492, -3.2675925657736515, -3.90406760984636],
    }
    for label, gq in POINTS.items():
        search_pregrasp(gq, label=label)
        print()

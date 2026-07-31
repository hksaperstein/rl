"""Pi-local (pure-FK, no Isaac Sim) design of an AR4 SIDE / HORIZONTAL
pure-friction grasp of the 15mm cube.

Root cause being addressed (kb ar4-vs-franka-root-cause-comparison.md,
2026-07-31 UPDATE): a CENTERED TOP-DOWN approach is collision-blocked -- the
gripper palm/body hits the cube TOP FACE ~9mm early; the shallow fingers
cannot straddle a 15mm cube from directly above. The arm/drives/tracking are
proven flawless in free space.

FIX designed here: a horizontal side grasp. The gripper is oriented so the
jaw-slide (closing) axis is HORIZONTAL and aligned with one pair of the cube's
vertical faces, the pads sit at the cube's vertical mid-height, and the
approach axis is horizontal with the palm pointing away from the cube -- so the
palm never needs to be above the cube.

This reuses tasks/ar4/fk_verification.py's vendor-URDF FK chain and the
measured pad-centroid offset (12.5mm along each jaw link's local -Y). Runs on
the toy_env venv (numpy).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL_ROOT)

from tasks.ar4.fk_verification import (  # noqa: E402
    DEFAULT_JOINT_TABLE,
    compute_link_pose_from_joint_values,
)

ARM = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-105.0, 105.0),
    "joint_6": (-180.0, 180.0),
}
PAD_LOCAL = np.array([0.0, -0.0125, 0.0])  # measured pad centroid in jaw-link local frame
# jaw collision mesh extent in its own mesh-prim-local z: [-0.018475, +0.015825]
JAW_MESH_LO, JAW_MESH_HI = -0.018475, 0.015825


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def base_to_world_pos(p):
    """robot base_link -> world (AR4 base rotated 180deg about Z)."""
    return np.array([-p[0], -p[1], p[2]])


def base_to_world_mat(R):
    Rz180 = np.diag([-1.0, -1.0, 1.0])
    return Rz180 @ R


def gripper_frame(qd):
    """Return world-frame gripper geometry at arm config qd (degrees)."""
    jv = {n: math.radians(v) for n, v in zip(ARM, qd)}
    jv["gripper_jaw1_joint"] = 0.014
    jv["gripper_jaw2_joint"] = 0.014
    p6, q6 = compute_link_pose_from_joint_values(jv, "link_6")
    p1, q1 = compute_link_pose_from_joint_values(jv, "gripper_jaw1_link")
    p2, q2 = compute_link_pose_from_joint_values(jv, "gripper_jaw2_link")
    R1, R2, R6 = quat_to_mat(q1), quat_to_mat(q2), quat_to_mat(q6)
    pad1 = p1 + R1 @ PAD_LOCAL
    pad2 = p2 + R2 @ PAD_LOCAL
    # to world
    W = dict(
        link6=base_to_world_pos(p6),
        R6=base_to_world_mat(R6),
        pad1=base_to_world_pos(pad1),
        pad2=base_to_world_pos(pad2),
        jaw1=base_to_world_pos(p1),
        jaw2=base_to_world_pos(p2),
    )
    W["pad_mid"] = (W["pad1"] + W["pad2"]) / 2.0
    return W


def pad_mid_world(qd):
    return gripper_frame(qd)["pad_mid"]


# --------------------------------------------------------------------------
# 1. Characterise the gripper at the proven near-vertical GRASP_Q
GRASP_Q_DEG = [-19.4595, 53.2382, 14.9828, -14.9277, 21.9517, 114.6585]
W = gripper_frame(GRASP_Q_DEG)
print("=== Gripper geometry at the proven near-vertical GRASP_Q (jaws open 14mm) ===")
print(f" link_6 world      = {np.round(W['link6'],4)}")
print(f" pad1 world        = {np.round(W['pad1'],4)}")
print(f" pad2 world        = {np.round(W['pad2'],4)}")
print(f" pad_mid world     = {np.round(W['pad_mid'],4)}")
sep = np.linalg.norm(W["pad1"] - W["pad2"]) * 1000
print(f" pad separation    = {sep:.2f} mm  (jaws open)")

# closing axis (between pads), approach axis (link6 -> pad_mid), in world
close_ax = (W["pad2"] - W["pad1"]); close_ax /= np.linalg.norm(close_ax)
appr_ax = (W["pad_mid"] - W["link6"]); appr_ax /= np.linalg.norm(appr_ax)
print(f" closing axis (world) = {np.round(close_ax,4)}")
print(f" approach axis(world) = {np.round(appr_ax,4)}  (link6 -> pad_mid)")
print(f" approach tilt from vertical(-Z) = {math.degrees(math.acos(np.clip(-appr_ax[2],-1,1))):.2f} deg")

# palm-to-fingertip depth: distance from link_6 origin to pad_mid along approach
palm_to_pad = np.dot(W["pad_mid"] - W["link6"], appr_ax)
print(f"\n=== PALM-TO-FINGERTIP DEPTH ===")
print(f" |link6 -> pad_mid| along approach = {palm_to_pad*1000:.2f} mm")
# The jaw collision mesh extends JAW_MESH_LO along its finger axis. Establish
# the fingertip vs the jaw-link origin and vs link_6.
# express these axes in link_6 local frame (constant regardless of arm pose)
R6 = W["R6"]
close_l6 = R6.T @ close_ax
appr_l6 = R6.T @ appr_ax
print(f" closing axis in link_6 local = {np.round(close_l6,4)}")
print(f" approach axis in link_6 local= {np.round(appr_l6,4)}")

# Cube geometry
CUBE = 0.015
CUBE_HALF = CUBE / 2.0
print(f"\n cube size={CUBE*1000:.0f}mm, half={CUBE_HALF*1000:.1f}mm")
print(f" For a TOP-DOWN grasp the palm sits {palm_to_pad*1000:.1f}mm above the pads; to reach")
print(f" mid-height the palm would be {(palm_to_pad-CUBE_HALF)*1000:.1f}mm above the cube-top -- the shortfall.")

# --------------------------------------------------------------------------
# 2. Design the SIDE grasp target orientation.
# Desired world basis for the gripper: closing axis -> world +X (grip the
# cube's +X/-X vertical faces), approach axis -> world +Y (approach from -Y,
# palm points -Y away from cube). The third axis completes right-handed.
# We want R6_world such that R6_world @ close_l6 = +X and R6_world @ appr_l6 = +Y.
target_close_w = np.array([1.0, 0.0, 0.0])
target_appr_w = np.array([0.0, 1.0, 0.0])


def solve_orientation(c_l6, a_l6, c_w, a_w):
    """Find world rotation R mapping local axes (c_l6,a_l6) -> world (c_w,a_w)."""
    # build orthonormal bases
    def basis(u, v):
        e1 = u / np.linalg.norm(u)
        e2 = v - np.dot(v, e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        return np.column_stack([e1, e2, e3])
    B_l6 = basis(c_l6, a_l6)
    B_w = basis(c_w, a_w)
    return B_w @ B_l6.T


R6_target = solve_orientation(close_l6, appr_l6, target_close_w, target_appr_w)
print("\n=== SIDE-GRASP target world orientation for link_6 ===")
print(f" R6_target @ close_l6 = {np.round(R6_target @ close_l6,3)} (want +X)")
print(f" R6_target @ appr_l6  = {np.round(R6_target @ appr_l6,3)} (want +Y)")


def make_orientation(az_deg, tilt_deg):
    """Build a target link_6 world orientation for a side grasp.
    approach axis = horizontal direction at azimuth az (0=+Y,90=+X,180=-Y,
    270=-X) tilted DOWN from horizontal by tilt_deg. closing axis = horizontal,
    perpendicular to the approach azimuth (so it lies on the cube's vertical
    faces)."""
    az = math.radians(az_deg)
    horiz = np.array([math.sin(az), math.cos(az), 0.0])  # az=0 -> +Y, az=90 -> +X
    down = np.array([0.0, 0.0, -1.0])
    t = math.radians(tilt_deg)
    appr_w = math.cos(t) * horiz + math.sin(t) * down
    appr_w /= np.linalg.norm(appr_w)
    # closing axis: horizontal, perpendicular to horiz
    close_w = np.array([math.cos(az), -math.sin(az), 0.0])
    return solve_orientation(close_l6, appr_l6, close_w, appr_w), close_w, appr_w

# --------------------------------------------------------------------------
# 3. IK: find arm config placing pad_mid at cube center with R6 == R6_target.
# Cube center world target. Choose pedestal + cube placement afterward; first
# find where the arm can comfortably reach a horizontal-gripper pose in front.
# Use DLS-IK on 6 joints. Position from pad_mid, orientation from link_6.


def rot_err(R_cur, R_tgt):
    Re = R_tgt @ R_cur.T
    ang = math.acos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    axis /= (np.linalg.norm(axis) + 1e-12)
    return axis * ang


def ik(target_pos, R_tgt, q0_deg, iters=150, Lrot=0.05):
    q = np.array([math.radians(v) for v in q0_deg])

    def measure(qq):
        qd = [math.degrees(v) for v in qq]
        Wc = gripper_frame(qd)
        pos = Wc["pad_mid"]
        oe = Lrot * rot_err(Wc["R6"], R_tgt)
        return np.concatenate([pos, oe]), Wc

    for _ in range(iters):
        m, Wc = measure(q)
        err = np.concatenate([target_pos - m[:3], -m[3:]])
        if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
            break
        J = np.zeros((6, 6))
        for j in range(6):
            dq = 1e-4
            qp = q.copy(); qp[j] += dq
            mp, _ = measure(qp)
            J[:, j] = (mp - m) / dq
        lam = 0.01
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        step = np.max(np.abs(dq))
        if step > math.radians(3):
            dq *= math.radians(3) / step
        q = q + dq
    qd = [math.degrees(v) for v in q]
    m, Wc = measure(q)
    return qd, m, Wc


# Try a series of candidate cube placements in front of the robot; pick one
# reachable with good joint-limit margin and a horizontal gripper.
print("\n=== IK search: sweep approach directions x tilt x cube position ===")
overall = []
for az in [0.0, 90.0, 180.0, 270.0]:
    for tilt in [0.0, 30.0, 45.0]:
        R_t, close_w, appr_w = make_orientation(az, tilt)
        best = None
        for cy in [0.28, 0.34]:
            for cx in [-0.12]:
                for cz in [0.09, 0.11]:
                    tgt = np.array([cx, cy, cz])
                    qd, m, Wc = ik(tgt, R_t, GRASP_Q_DEG, iters=70)
                    perr = np.linalg.norm(tgt - m[:3]) * 1000
                    oerr = math.degrees(np.linalg.norm(rot_err(Wc["R6"], R_t)))
                    if perr > 1.5 or oerr > 3.0:
                        continue
                    margins = [min(v - LIMITS_DEG[n][0], LIMITS_DEG[n][1] - v) for n, v in zip(ARM, qd)]
                    if min(margins) < 5.0:
                        continue
                    cand = dict(tgt=tgt, qd=qd, perr=perr, oerr=oerr, minm=min(margins),
                                margins=margins, Wc=Wc, close_w=close_w, appr_w=appr_w)
                    if best is None or cand["minm"] > best["minm"]:
                        best = cand
        tag = f"az={az:.0f} tilt={tilt:.0f}"
        if best is None:
            print(f" [{tag}] no reachable pose")
        else:
            b = best
            print(f" [{tag}] REACHABLE tgt={np.round(b['tgt'],3)} min_margin={b['minm']:.1f}deg "
                  f"perr={b['perr']:.2f}mm oerr={b['oerr']:.2f}deg")
            overall.append((az, tilt, b))

if overall:
    # pick the most horizontal (smallest tilt) with best margin
    overall.sort(key=lambda t: (t[1], -t[2]["minm"]))
    az, tilt, b = overall[0]
    Wc = b["Wc"]
    print("\n=== SELECTED SIDE-GRASP POSE ===")
    print(f" approach az={az:.0f} deg, tilt-from-horizontal={tilt:.0f} deg")
    print(f" closing axis world = {np.round(b['close_w'],3)}   approach axis world = {np.round(b['appr_w'],3)}")
    print(f" cube center target = {np.round(b['tgt'],4)}")
    print(f" GRASP_Q_DEG = {[round(v,4) for v in b['qd']]}")
    for n, v, mgn in zip(ARM, b["qd"], b["margins"]):
        lo, hi = LIMITS_DEG[n]
        print(f"   {n}: {v:+8.3f}deg limits[{lo},{hi}] margin={mgn:.1f}deg")
    print(f" pad1 world={np.round(Wc['pad1'],4)} pad2 world={np.round(Wc['pad2'],4)}")
    print(f" jaw1 link world={np.round(Wc['jaw1'],4)} jaw2 link world={np.round(Wc['jaw2'],4)}")
    print(f" link_6(palm) world={np.round(Wc['link6'],4)}")
    print(f" pad z's = {Wc['pad1'][2]*1000:.1f}mm, {Wc['pad2'][2]*1000:.1f}mm (== cube mid-height target)")
    # lowest gripper point (fingertip below pad center along -approach? use jaw links)
    lowest_z = min(Wc['pad1'][2], Wc['pad2'][2], Wc['jaw1'][2], Wc['jaw2'][2], Wc['link6'][2])
    print(f" lowest gripper origin z = {lowest_z*1000:.1f}mm  (pedestal top must be below this)")
else:
    print("\n NO reachable side-grasp pose found across all sampled orientations.")

# --------------------------------------------------------------------------
# 4. FINALIZE the pure-horizontal az=90 side-grasp poses at high precision.
print("\n" + "=" * 70)
print("FINALIZED PURE-HORIZONTAL SIDE-GRASP (az=90, tilt=0)")
print("=" * 70)
R_t, close_w, appr_w = make_orientation(90.0, 0.0)  # approach +X, close along Y
CUBE_CENTER = np.array([-0.12, 0.28, 0.09])
CUBE_HALF = 0.0075
PED_TOP = CUBE_CENTER[2] - CUBE_HALF  # 0.0825

def report_pose(name, tgt):
    qd, m, Wc = ik(tgt, R_t, GRASP_Q_DEG, iters=400)
    perr = np.linalg.norm(tgt - m[:3]) * 1000
    oerr = math.degrees(np.linalg.norm(rot_err(Wc["R6"], R_t)))
    margins = [min(v - LIMITS_DEG[n][0], LIMITS_DEG[n][1] - v) for n, v in zip(ARM, qd)]
    print(f"\n {name}: tgt={np.round(tgt,4)} perr={perr:.3f}mm oerr={oerr:.3f}deg minmargin={min(margins):.1f}deg")
    print(f"   Q_DEG = {[round(v,4) for v in qd]}")
    for n, v, mg in zip(ARM, qd, margins):
        print(f"     {n}: {v:+8.3f}  margin={mg:5.1f}")
    print(f"   pad1={np.round(Wc['pad1'],4)} pad2={np.round(Wc['pad2'],4)} link6={np.round(Wc['link6'],4)}")
    return qd, Wc

qg, Wg = report_pose("GRASP (pad_mid at cube center)", CUBE_CENTER)
qp, Wp = report_pose("PREGRASP (backed -X 35mm)", CUBE_CENTER - np.array([0.035, 0.0, 0.0]))
ql, Wl = report_pose("LIFT (up 80mm)", CUBE_CENTER + np.array([0.0, 0.0, 0.080]))

print("\n=== SCENE / CLEARANCE ===")
print(f" CUBE center world = {np.round(CUBE_CENTER,4)} (rests on pedestal)")
print(f" pedestal top z = {PED_TOP:.4f}  (cube half {CUBE_HALF*1000:.1f}mm below center)")
print(f" gripper at grasp: all origins ~z=0.09, i.e. {(0.09-PED_TOP)*1000:.1f}mm above pedestal top")
print(f" closing axis world = {np.round(close_w,3)} (grips cube +Y/-Y faces)")
print(f" approach axis world = {np.round(appr_w,3)} (link6 -> pads, palm on -X side)")
print(" -> palm is on the -X SIDE of the cube, never above it: sidesteps the top collision.")
print(" -> pedestal should be NARROW in Y (< 15mm) so pads at cube +/-Y faces (+/-7.5mm) clear it.")

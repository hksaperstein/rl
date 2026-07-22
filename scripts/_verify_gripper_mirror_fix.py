# scripts/_verify_gripper_mirror_fix.py
"""One-off, non-permanent verification script (ar4-franka-fixes-transfer
plan, Task 5 - controller-authorized gripper-mirror-sign bug fix,
2026-07-21; extended 2026-07-21 "later" pass to verify the follow-up
PhysxMimicJointAPI-removal fix): direct-joint-control check that
gripper_jaw2_joint now tracks gripper_jaw1_joint's MIRRORED (negated)
commanded position under real PD-actuator dynamics, rather than either
the identical-value pre-fix bug or the "pinned at a hard limit regardless
of target" behavior found in the first live dynamic rollout after the
sign fix alone (see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
2026-07-21 "later" UPDATE). That rollout implicated a physics-level
tug-of-war between gripper_jaw2_joint's PhysxMimicJointAPI spring
constraint and its independent ImplicitActuatorCfg PD actuator; the fix
(scripts/build_asset.py's new _remove_gripper_jaw2_mimic_constraint)
strips the mimic constraint entirely at asset-build time. This script's
job is to confirm, with real dynamic data (not just a static USD
inspection), that jaw2 now tracks its own PD target with a normal
convergence curve like jaw1's.

Drives PhysX directly (write_data_to_sim + sim.step), same pattern as
scripts/interactive_joint_control.py, using the
GRIPPER_OPEN_COMMAND_EXPR/GRIPPER_CLOSED_COMMAND_EXPR dicts
(tasks/ar4/robot_cfg.py) instead of a single shared scalar target for both
joints. Prints jaw1/jaw2 settled positions after reset (open), after
commanding close, and after commanding open again - real, direct evidence
(not a shaped metric or an eyeballed video) that jaw2 mirrors jaw1 rather
than overshooting/hitting its own hard limit.

Runs non-headless per this repo's standing convention (CLAUDE.md
"Environment conventions": don't set headless for Isaac-Sim-touching
scripts while the user wants to watch) - requires DISPLAY=:1 in the
launching environment.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_gripper_mirror_fix.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Verify gripper_jaw2_joint mirror-sign fix.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
)


def _settle(
    env, robot, gripper_cfg, arm_cfg, arm_hold_target, target_expr, jaw_body_ids, contact_sensors,
    n_steps=60, print_every=10, label="",
):
    target = torch.tensor(
        [[target_expr[name] for name in GRIPPER_JOINT_NAMES]], device=env.device
    )
    for i in range(n_steps):
        # DIAGNOSTIC FIX (added after the previous run showed gripper_jaw
        # world z falling from +0.47 to +0.20 over just 120 steps - the
        # WHOLE ARM sagging under gravity, since this script was only ever
        # commanding a joint-position target for the gripper joints, never
        # the arm joints, and the arm's own ImplicitActuatorCfg
        # (stiffness=40, damping=4, effort_limit_sim=20.0 - much weaker
        # than the gripper's stiffness=1000) apparently can't hold the
        # arm's own weight open-loop with no active per-step target).
        # An uncontrolled falling/swinging arm base injects real Coriolis/
        # base-acceleration coupling into the child gripper joints, which
        # would confound any conclusion about gripper_jaw2_joint's own
        # drive-sign/tracking behavior drawn from the earlier runs - hold
        # the arm at its own reset joint positions explicitly every step,
        # exactly like the gripper target, so this run actually isolates
        # the gripper's own dynamics.
        robot.set_joint_position_target(arm_hold_target, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(target, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        for sensor in contact_sensors:
            sensor.update(env.physics_dt, force_recompute=True)
        if i % print_every == 0 or i == n_steps - 1:
            pos = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
            jaw1_z, jaw2_z = robot.data.body_pos_w[0, jaw_body_ids, 2].cpu().tolist()
            jaw1_force = contact_sensors[0].data.net_forces_w[0, 0].norm().item()
            jaw2_force = contact_sensors[1].data.net_forces_w[0, 0].norm().item()
            arm_drift = (robot.data.joint_pos[0, arm_cfg.joint_ids] - arm_hold_target[0]).abs().max().item()
            print(
                f"  [{label} step {i:3d}] jaw1={pos[0]:+.5f}  jaw2={pos[1]:+.5f}  target={target.cpu().tolist()}  "
                f"jaw1_z={jaw1_z:+.4f}  jaw2_z={jaw2_z:+.4f}  jaw1_cube_force={jaw1_force:.4f}N  jaw2_cube_force={jaw2_force:.4f}N  "
                f"arm_max_drift={arm_drift:.5f}rad"
            )
    return robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1

    # DIAGNOSTIC FIX, round 2 (the round-1 fix - explicitly commanding the
    # arm to hold its own reset joint positions every step - produced
    # BYTE-FOR-BYTE IDENTICAL jaw trajectories to the un-held run, proving
    # it was a genuine no-op: Isaac Lab's own env.reset() already sets the
    # articulation's joint position targets to match the initial state, so
    # this script's explicit hold command was redundant, not missing. The
    # real finding: the arm's own actuator gains (stiffness=40, damping=4,
    # tasks/ar4/robot_cfg.py) are themselves too weak to resist gravity at
    # this arm's reset pose even WITH an active hold target - genuine
    # actuator underpower, not a missing-target bug. Test-local override
    # only (not touching the shared robot_cfg.py, which may rely on RL
    # continuously re-commanding a fresh action every control step in a way
    # this static single-target diagnostic never does): temporarily boost
    # the arm actuator's own stiffness/damping enough to hold position
    # statically, so this diagnostic's own arm base stays genuinely fixed
    # and doesn't inject Coriolis/base-acceleration coupling into the
    # gripper's own joints while sagging.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)

    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    contact_sensors = [env.scene["gripper_jaw1_contact"], env.scene["gripper_jaw2_contact"]]

    with torch.inference_mode():
        env.reset()

    # State 1: fresh reset - should already reflect the fixed init_state
    # (jaw1=+GRIPPER_OPEN_POS, jaw2=-GRIPPER_OPEN_POS).
    reset_pos = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()

    # Hold the arm at its own just-reset joint positions for the rest of
    # this script (see the _settle docstring-comment above for why this is
    # necessary now, not optional) - captured fresh from the real
    # post-reset state, not assumed to be all-zeros.
    arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].unsqueeze(0).clone()

    # DIAGNOSTIC (added after the run below showed an asymmetric CLOSE-vs-
    # OPEN signature inconsistent with a plain sign inversion): report
    # where the gripper jaws and the cube actually are in world space, and
    # the cube-filtered contact-force reading on each jaw, so a ground- or
    # cube-contact explanation for jaw2's asymmetric resistance can be
    # directly confirmed or ruled out, not just guessed at from the joint
    # trajectory alone. gripper_jaw1_contact/gripper_jaw2_contact are
    # filtered to the Cube prim specifically (see
    # tasks/ar4/pickplace_graspgoal_env_cfg.py), so a nonzero reading here
    # means real cube contact, not ground contact (ground isn't in either
    # sensor's filter list) - ground proximity is instead read off the raw
    # jaw link world z heights vs the ground plane at z=0.
    jaw1_z0, jaw2_z0 = robot.data.body_pos_w[0, jaw_body_ids, 2].cpu().tolist()
    cube_pos0 = env.scene["cube"].data.root_pos_w[0].cpu().tolist()
    print("=" * 70)
    print(f"[reset] gripper_jaw1_link z={jaw1_z0:+.4f}  gripper_jaw2_link z={jaw2_z0:+.4f}  (ground at z=0)")
    print(f"[reset] cube root_pos_w = {cube_pos0}")
    print("=" * 70)

    with torch.inference_mode():
        closed_pos = _settle(env, robot, gripper_cfg, arm_cfg, arm_hold_target, GRIPPER_CLOSED_COMMAND_EXPR, jaw_body_ids, contact_sensors, label="CLOSE")
        open_pos = _settle(env, robot, gripper_cfg, arm_cfg, arm_hold_target, GRIPPER_OPEN_COMMAND_EXPR, jaw_body_ids, contact_sensors, label="OPEN")

    print("=" * 70)
    print(f"GRIPPER_JOINT_NAMES = {GRIPPER_JOINT_NAMES}")
    print(f"GRIPPER_OPEN_COMMAND_EXPR   = {GRIPPER_OPEN_COMMAND_EXPR}")
    print(f"GRIPPER_CLOSED_COMMAND_EXPR = {GRIPPER_CLOSED_COMMAND_EXPR}")
    print("-" * 70)
    print(f"[reset, fixed init_state]   jaw1={reset_pos[0]:+.5f}  jaw2={reset_pos[1]:+.5f}  "
          f"jaw1+jaw2={reset_pos[0] + reset_pos[1]:+.5f}")
    print(f"[commanded CLOSED, settled] jaw1={closed_pos[0]:+.5f}  jaw2={closed_pos[1]:+.5f}  "
          f"jaw1+jaw2={closed_pos[0] + closed_pos[1]:+.5f}")
    print(f"[commanded OPEN, settled]   jaw1={open_pos[0]:+.5f}  jaw2={open_pos[1]:+.5f}  "
          f"jaw1+jaw2={open_pos[0] + open_pos[1]:+.5f}")
    print("-" * 70)
    mirror_ok = abs(open_pos[0] + open_pos[1]) < 0.002 and abs(closed_pos[0] + closed_pos[1]) < 0.002
    print(f"MIRROR CHECK (jaw1 ~= -jaw2 in both states, sum ~= 0): {'PASS' if mirror_ok else 'FAIL'}")
    print("=" * 70)

    # Diagnostic follow-up (added after the above FAILED even post-mimic-
    # removal): jaw2 moved substantially but consistently landed at the
    # OPPOSITE end from its own commanded target in both phases (target=0
    # -> stayed near -0.014; target=-0.014 -> moved to 0). That is the
    # signature of an inverted actuator-drive sign for this specific joint
    # (drive applies force as if tracking -target, not target), not of a
    # limit-pinning defect. Test this directly: hold jaw1 fixed and sweep
    # jaw2 to a MID-RANGE target (-0.007) where "converges toward -0.007"
    # vs. "moves toward the opposite endpoint (0)" unambiguously
    # distinguishes correct tracking from an inverted drive.
    print("DIAGNOSTIC: isolated jaw2 mid-range target sweep (jaw1 held fixed)")
    jaw1_hold = open_pos[0]
    mid_target_expr = {GRIPPER_JOINT_NAMES[0]: jaw1_hold, GRIPPER_JOINT_NAMES[1]: -0.007}
    with torch.inference_mode():
        mid_pos = _settle(env, robot, gripper_cfg, arm_cfg, arm_hold_target, mid_target_expr, jaw_body_ids, contact_sensors, label="MID(-0.007)")
    print(f"[commanded jaw2=-0.007, settled] jaw1={mid_pos[0]:+.5f}  jaw2={mid_pos[1]:+.5f}")
    print(
        "  -> jaw2 converged TOWARD -0.007 (correct tracking)"
        if abs(mid_pos[1] - (-0.007)) < 0.003
        else "  -> jaw2 did NOT converge toward -0.007 (still not tracking its own target correctly)"
    )
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()

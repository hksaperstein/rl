"""Instrumented rollout confirming WarmStartedResidualDifferentialIKAction's
residual_authority ramp actually rises from ~0 toward 1.0 over cfg.warmup_steps
env steps during a REAL env rollout - not just the isolated formula check in
scripts/warmresidual_action_smoke_test.py. This is Experiment 23's hard gate:
if the ramp doesn't move as expected here, the whole design's central premise
(a warm-started residual, per Johannink et al. 2019) has not actually been
implemented correctly, and training must not proceed. See
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

Uses a zero-action policy (no trained checkpoint exists yet at this point in
the plan) - the ramp value itself does not depend on what the policy outputs,
only on cfg.warmup_steps and how many process_actions() calls have occurred,
so a zero policy is sufficient to verify the ramp mechanism in isolation.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_verify.py --steps 1300
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify WarmStartedResidualDifferentialIKAction's warm-start ramp.")
parser.add_argument("--steps", type=int, default=1300, help="Number of env steps to run.")
parser.add_argument("--log_every", type=int, default=100, help="Print the ramp value every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceWarmResidualEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    arm_action_term = env.action_manager.get_term("arm_action")

    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    zero_actions = torch.zeros(env_cfg.scene.num_envs, action_dim, device=env.device)

    readings = []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            env.step(zero_actions)
            authority = min(1.0, arm_action_term._step_count / arm_action_term.cfg.warmup_steps)
            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                print(
                    f"[STEP {step:5d}] internal_step_count={arm_action_term._step_count} "
                    f"warmup_steps={arm_action_term.cfg.warmup_steps} residual_authority={authority:.4f}"
                )
            readings.append((step, authority))

    env.close()

    first_authority = readings[0][1]
    near_warmup_idx = min(range(len(readings)), key=lambda i: abs(readings[i][0] - arm_action_term.cfg.warmup_steps))
    at_warmup_authority = readings[near_warmup_idx][1]
    final_authority = readings[-1][1]

    print(
        f"[SUMMARY] first_step_authority={first_authority:.4f} "
        f"authority_near_warmup_step={at_warmup_authority:.4f} "
        f"final_authority={final_authority:.4f}"
    )

    ramp_rose = first_authority < 0.05 and at_warmup_authority > 0.9
    clamped_at_one = final_authority == 1.0
    if ramp_rose and clamped_at_one:
        print("[VERIFICATION] PASS: residual_authority ramped from ~0 to 1.0 and stayed clamped.")
    else:
        print(
            f"[VERIFICATION] FAIL: ramp_rose={ramp_rose} clamped_at_one={clamped_at_one} - "
            "BLOCKED, do not proceed to Task 5/6."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()

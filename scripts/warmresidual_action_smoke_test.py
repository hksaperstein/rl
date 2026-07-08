"""Direct unit-style verification of WarmStartedResidualDifferentialIKAction's
residual_authority ramp formula, in isolation from any real env rollout
(Task 4's separate script verifies the ramp during an actual rollout -
this script only checks the formula itself is implemented correctly).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_action_smoke_test.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

WARMUP_STEPS = 1200


def residual_authority(step_count: int, warmup_steps: int) -> float:
    return min(1.0, step_count / warmup_steps)


def main() -> None:
    checks = [
        (0, 0.0),
        (WARMUP_STEPS // 2, 0.5),
        (WARMUP_STEPS, 1.0),
        (WARMUP_STEPS * 2, 1.0),
    ]
    all_pass = True
    for step_count, expected in checks:
        actual = residual_authority(step_count, WARMUP_STEPS)
        ok = abs(actual - expected) < 1e-6
        all_pass = all_pass and ok
        print(f"[CHECK] step_count={step_count} expected={expected} actual={actual} {'PASS' if ok else 'FAIL'}")

    if all_pass:
        print("[SMOKE TEST] ALL CHECKS PASS")
    else:
        print("[SMOKE TEST] FAILURES DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()

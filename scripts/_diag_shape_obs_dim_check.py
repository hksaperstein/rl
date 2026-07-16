"""One-off diagnostic (Task 1 of docs/superpowers/plans/2026-07-16-unified-
multi-die-specialist-distillation.md, Step 5): confirms
FrankaDieLiftJointD8StandardEnvCfg_PLAY's observation space grows by exactly
4 + K (K=1, so 5) dims after wiring mdp.object_shape_class_onehot /
mdp.object_geometry_descriptor into ObservationsCfg.PolicyCfg, and nothing
else changes.

Uses isaaclab.app.AppLauncher (NOT a raw isaacsim.SimulationApp - the
project_check-working-pattern-before-new-script memory / this repo's own
scripts/train_franka.py boilerplate: plain SimulationApp doesn't configure
the Nucleus asset-root isaaclab_assets.robots.franka's FRANKA_PANDA_HIGH_PD_CFG
needs to resolve its USD path, so building a live ManagerBasedRLEnv with a
real Franka robot needs the AppLauncher path, unlike this repo's OTHER
_diag_* scripts which only read raw USD mesh data with no robot/env
involved at all). Run non-headless per CLAUDE.md's standing instruction (a
display is available); builds ONE tiny (num_envs=2) env, reads obs dims,
closes immediately - no training loop, nothing to watch beyond the env
window briefly opening/closing.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_shape_obs_dim_check.py"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check obs-space dim growth for the shape-observation terms.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8StandardEnvCfg_PLAY  # noqa: E402


def main() -> None:
    cfg = FrankaDieLiftJointD8StandardEnvCfg_PLAY()
    cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    term_dims = env.observation_manager.group_obs_term_dim["policy"]
    term_names = list(env.observation_manager.active_terms["policy"])
    per_term = dict(zip(term_names, term_dims))
    after_total = env.observation_manager.group_obs_dim["policy"]
    if isinstance(after_total, tuple):
        after_total = after_total[0]

    print("Per-term obs dims (policy group):")
    for name, dim in per_term.items():
        print(f"  {name}: {dim}")
    print(f"AFTER total (concatenated): {after_total}")

    assert per_term["shape_class"] == (4,), f"expected shape_class dim (4,), got {per_term['shape_class']}"
    assert per_term["geometry_descriptor"] == (1,), (
        f"expected geometry_descriptor dim (1,), got {per_term['geometry_descriptor']}"
    )

    new_terms_total = per_term["shape_class"][0] + per_term["geometry_descriptor"][0]
    before_total = after_total - new_terms_total
    print(f"BEFORE total (after_total - new_terms_total): {before_total}")
    print(f"new_terms_total (shape_class + geometry_descriptor): {new_terms_total}")

    assert new_terms_total == 5, f"expected exactly 4+K=5 new dims, got {new_terms_total}"

    # Sanity-check the actual values for the d8 specialist env: onehot index
    # 0 ("d8" is SHAPE_CLASSES[0]) should be 1.0, geometry descriptor should
    # match SHAPE_GEOMETRY_DESCRIPTORS["d8"].
    obs = env.observation_manager.compute()["policy"]
    shape_class_start = sum(per_term[n][0] for n in term_names[: term_names.index("shape_class")])
    shape_slice = obs[:, shape_class_start : shape_class_start + 4]
    geom_slice = obs[:, shape_class_start + 4 : shape_class_start + 5]
    print(f"shape_class slice (env 0): {shape_slice[0].tolist()}")
    print(f"geometry_descriptor slice (env 0): {geom_slice[0].tolist()}")
    assert torch.allclose(shape_slice[0], torch.tensor([1.0, 0.0, 0.0, 0.0], device=shape_slice.device), atol=1e-6)
    assert abs(geom_slice[0, 0].item() - 0.889647) < 1e-4

    print("PASS: obs space grows by exactly 4+K=5, values correct for d8 specialist.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

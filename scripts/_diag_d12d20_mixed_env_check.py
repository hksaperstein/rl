"""One-off diagnostic (Task 5, docs/superpowers/plans/2026-07-16-unified-
multi-die-specialist-distillation.md; BACKLOG.md's 2026-07-19 controller
decision "(b) single mixed-population env"): directly verifies, against a
REAL live Isaac Lab env (not just a source read), that
FrankaDieLiftJointD12D20MixedEnvCfg's deterministic round-robin per-env
shape assignment actually matches tasks/franka/mdp.py's
object_shape_class_onehot/object_geometry_descriptor's own
`index % len(die_shape_classes_per_env)` formula - i.e. that the pure-math
assumption (shape_observations.py's shape_class_onehot_per_env) and the
live spawned PhysX/USD reality genuinely agree, not just that the source
read of spawn_multi_asset says they should.

Two independent checks, cross-referenced against each other AND against the
predicted ("d12", "d20") round-robin:

1. Observation-side: env.observation_manager's own live `shape_class`/
   `geometry_descriptor` per-env values (read via a fresh env.reset(), the
   same live-computed values the real DAgger rollout will actually see).
2. Physics/USD-side: each env's own spawned Object prim's live authored
   scale (xformOp:scale) - MultiAssetSpawnerCfg's per-asset UsdFileCfg.scale
   is what actually reaches the PhysX stage, so a per-env scale of 0.001476
   (d12's 48mm-parity constant) vs 0.001585 (d20's) is an independent,
   ground-truth confirmation of which physical asset each env actually
   received - NOT derived from mdp.py/shape_observations.py at all, so it
   can't just be self-consistently wrong the same way source-only trust
   could be.

Both are asserted to match the predicted env_index % 2 pattern AND each
other, for every env in a small (num_envs=8) build.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_d12d20_mixed_env_check.py"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify FrankaDieLiftJointD12D20MixedEnvCfg's per-env shape assignment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sim.utils import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20MixedEnvCfg  # noqa: E402
from tasks.franka.shape_observations import SHAPE_GEOMETRY_DESCRIPTORS  # noqa: E402

NUM_ENVS = 8
EXPECTED_ORDER = ("d12", "d20")
EXPECTED_SCALE = {"d12": 0.001476, "d20": 0.001585}
SCALE_TOL = 1e-6


def main() -> None:
    cfg = FrankaDieLiftJointD12D20MixedEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset(), None

    assert cfg.die_shape_classes_per_env == EXPECTED_ORDER, (
        f"env cfg's die_shape_classes_per_env {cfg.die_shape_classes_per_env} != expected {EXPECTED_ORDER}"
    )

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

    stage = get_current_stage()

    print(f"{'env':>3} | {'predicted':>9} | {'obs argmax':>10} | {'geom desc':>9} | {'live scale':>10} | OK?")
    all_ok = True
    for env_idx in range(NUM_ENVS):
        predicted_shape = EXPECTED_ORDER[env_idx % len(EXPECTED_ORDER)]

        obs_shape_idx = int(onehot_block[env_idx].argmax().item())
        # SHAPE_CLASSES canonical order is (d8, d10, d12, d20) - map back to name.
        from tasks.franka.shape_observations import SHAPE_CLASSES

        obs_shape = SHAPE_CLASSES[obs_shape_idx]
        obs_geom = geom_block[env_idx, 0].item()

        prim_path = f"/World/envs/env_{env_idx}/Object"
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), f"no prim at {prim_path}"
        xformable = UsdGeom.Xformable(prim)
        live_scale = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                s = op.Get()
                live_scale = float(s[0])
                break
        assert live_scale is not None, f"no scale xformOp found on {prim_path}"

        expected_scale = EXPECTED_SCALE[predicted_shape]
        expected_geom = SHAPE_GEOMETRY_DESCRIPTORS[predicted_shape]

        ok = (
            obs_shape == predicted_shape
            and abs(live_scale - expected_scale) < SCALE_TOL
            and abs(obs_geom - expected_geom) < 1e-4
        )
        all_ok = all_ok and ok
        print(
            f"{env_idx:>3} | {predicted_shape:>9} | {obs_shape:>10} | {obs_geom:>9.6f} | {live_scale:>10.6f} | "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    env.close()
    simulation_app.close()

    if not all_ok:
        raise SystemExit("MISMATCH: predicted round-robin assignment does not match live obs/USD state - see table above")
    print("\nALL OK: predicted env_index % len(assets) round-robin matches BOTH the live observation_manager output "
          "AND the live USD-authored scale, for all envs.")


if __name__ == "__main__":
    main()

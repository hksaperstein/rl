"""Minimal test: just create Ar4EnvCfg and do one reset."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal env test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedEnv
from tasks.ar4.env_cfg import Ar4EnvCfg

try:
    print("[MINIMAL] Creating Ar4EnvCfg...")
    env_cfg = Ar4EnvCfg()
    env_cfg.sim.device = args_cli.device
    print("[MINIMAL] Config created")

    print("[MINIMAL] Creating ManagerBasedEnv...")
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[MINIMAL] Environment created")

    print("[MINIMAL] Calling env.reset()...")
    obs, info = env.reset()
    print("[MINIMAL] Reset complete, obs shape:", obs.shape if hasattr(obs, 'shape') else 'N/A')

    print("[MINIMAL] Calling env.step() with zeros...")
    action = torch.zeros(env.num_envs, 7, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    print("[MINIMAL] Step complete")

    print("[MINIMAL] SUCCESS!")
    env.close()

except Exception as e:
    print(f"[MINIMAL] ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

simulation_app.close()

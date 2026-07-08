"""Simple test to check if GraspDemoEnvCfg can be created."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test grasp demo env cfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedEnv

try:
    # Import the config
    print("[TEST] Attempting to import GraspDemoEnvCfg...")
    from scripts.grasp_demo import GraspDemoEnvCfg
    print("[TEST] GraspDemoEnvCfg imported successfully")

    # Create the config
    print("[TEST] Creating GraspDemoEnvCfg instance...")
    env_cfg = GraspDemoEnvCfg()
    env_cfg.sim.device = args_cli.device
    print("[TEST] GraspDemoEnvCfg instance created")

    # Create the environment
    print("[TEST] Creating ManagerBasedEnv...")
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[TEST] Environment created successfully!")
    print(f"[TEST] Num envs: {env.num_envs}")
    print(f"[TEST] Step dt: {env.step_dt}")

    # Test reset
    print("[TEST] Testing env.reset()...")
    obs, info = env.reset()
    print(f"[TEST] Reset successful, obs shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    # Test step
    print("[TEST] Testing env.step()...")
    action = torch.zeros(env.num_envs, 7, device=env.device)
    obs, _, terminated, truncated, info = env.step(action)
    print("[TEST] Step successful!")

    # Check sensors
    print("[TEST] Checking sensors...")
    print(f"[TEST] Has ee_frame: {'ee_frame' in env.scene._entity_names}")
    print(f"[TEST] Has gripper_jaw1_contact: {'gripper_jaw1_contact' in env.scene._entity_names}")
    print(f"[TEST] Has gripper_jaw2_contact: {'gripper_jaw2_contact' in env.scene._entity_names}")
    print(f"[TEST] Has perception_camera: {'perception_camera' in env.scene._entity_names}")

    env.close()
    print("[TEST] SUCCESS: All tests passed!")

except Exception as e:
    print(f"[TEST] ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

simulation_app.close()

"""One-off: compute the demo-camera's OpenGL-convention quaternion for a
raised, slightly-downward-angled eye/target pair, via Isaac Lab's own
create_rotation_matrix_from_view/quat_from_matrix (not hand-derived), same
convention tasks/ar4/graspgoal_democam_env_cfg.py's existing comment
describes using originally.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_compute_democam_quat.py
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402

EYE = (0.0, 1.03, 0.40)
TARGET = (0.0, 0.0, 0.12)

eyes = torch.tensor([EYE])
targets = torch.tensor([TARGET])
R = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
quat = quat_from_matrix(R)
print(f"EYE={EYE} TARGET={TARGET}")
print(f"QUAT_OPENGL = {tuple(quat[0].tolist())}")

simulation_app.close()

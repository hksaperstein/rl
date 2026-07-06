# AR4 Sphere Lift — Potential-Based Reward Shaping Report

## Task 2: Full 1500-iteration training run

**Training run directory:** `logs/train/2026-07-06_14-11-25/`

**TensorBoard scalar results:**

- `Episode_Reward/staged_potential_progress -> first: 0.003897153539583087 last: -0.11663519591093063 max: 0.003897153539583087 trajectory (10 samples): [0.003897, -0.065028, -0.067557, -0.07415, -0.085908, -0.0962, -0.103246, -0.103511, -0.103313, -0.109335]`
- `Episode_Termination/sphere_reached_goal -> first: 0.0 last: 0.0181070975959301 max: 0.0248819999396801 trajectory (10 samples): [0.0, 0.007294, 0.011515, 0.020152, 0.021891, 0.017008, 0.015798, 0.007874, 0.012909, 0.017314]`
- `Episode_Reward/action_rate -> first: -6.769504398107529e-05 last: -0.003035701811313629 max: -6.769504398107529e-05 trajectory (10 samples): [-6.8e-05, -0.001844, -0.002669, -0.002774, -0.002647, -0.002621, -0.002337, -0.0024, -0.002689, -0.002755]`
- `Episode_Reward/joint_vel -> first: -7.895268208812922e-05 last: -0.001706040813587606 max: -7.895268208812922e-05 trajectory (10 samples): [-7.9e-05, -0.001551, -0.00185, -0.001963, -0.001849, -0.001874, -0.001829, -0.001888, -0.001823, -0.001683]`

The `staged_potential_progress` reward shows a downward trajectory from a small initial value of 0.003897 to negative values by the end of training, with no sustained growth pattern, suggesting the potential-based term did not produce increasing reward signal through training.

## Task 3: Real eval + video inspection (decision gate)

(To be completed after eval video inspection)

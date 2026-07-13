# Why the Isaac Sim viewport freezes (diagnosed 2026-07-13)

Three mechanisms, only one a real bug. Evidence: direct source reading
of the installed Isaac Lab (`manager_based_env.py` step(),
`simulation_context.py` render modes) + today's process forensics.

## 1. Training: rhythmic freeze every PPO iteration (structural)

The Kit UI is serviced only inside `sim.render()`; Isaac Lab calls that
only inside `env.step()` during rollout (once per `render_interval` sim
steps — our cfgs set it to `decimation`, i.e. once per env-step). The
PPO learning update between rollouts (minibatch epochs, pure torch) and
logging/checkpoint writes never touch the UI loop → the window goes
dead for that fraction of EVERY iteration (~0.5-1.5s of each ~1.5-2.5s
cycle at 4096 envs) and the WM repeatedly flags "not responding."
Inherent to Isaac Lab's single-thread sim/UI scheduling; fixing it
means patching rsl_rl's runner to pump the app during updates — not
worth the throughput cost. Watch TensorBoard (:6006) for training
health, not the window.

## 2. Demos: synchronous perception subprocess (fixable, small)

`dice_pick_demo.py` blocks on the vision detector subprocess (model
load = seconds) with no Kit update running → guaranteed multi-second
viewport freeze at each detection. Fix when demo code is next touched:
poll the subprocess + pump `app.update()` in the wait loop.
(2026-07-13's 29-min frozen window was this mechanism with a
never-returning subprocess — missing .venv symlink in a worktree.)

## 3. Real hangs (rare, the actual bug)

Mid-training livelock (seen once: log+checkpoints frozen hours, ~3
cores spinning, GPU ~12%, SIGKILL required) and the documented
post-[DONE] teardown hang. The window is NOT a health signal in either
direction — the log-mtime heartbeat is (stall detector: >10min silent
log + live process → investigate CPU/GPU, then kill).

Related: [[size-curriculum]] (livelock incident record).

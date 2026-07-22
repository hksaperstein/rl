# Isaac Sim process management (the flock lock)

Full background for `CLAUDE.md`'s "Only one Isaac Sim process at a time"
rule. `CLAUDE.md` keeps the actionable command and diagnostic steps
(they get copy-pasted into dispatch prompts); this doc has the fuller
story of why each piece exists.

## Why `-o` is mandatory

2026-07-12 finding: without the `-o` flag, every child process of the
locked command inherits the lock's file descriptor, and Isaac Sim spawns
a detached long-lived **Omniverse Hub daemon** that keeps that fd open
forever — the lock then stays held even after the training process exits
cleanly, silently blocking every queued job. `-o` closes the fd before
exec so only the flock process itself holds the lock.

If a queued flock is stuck and `lsof /tmp/rl_isaac_sim.lock` shows the
holder is an `Omniverse Hub` process (0% CPU, no GPU compute apps, no
live python/kit training process), `kill -TERM` that Hub pid — it's a
relaunch-on-demand asset service, safe to kill when no Isaac app is
starting up.

## Why this matters: the polling incident

2026-07-09 finding: a Junior burned ~40 minutes and 72 tool calls
independently `ps aux`-polling in a sleep loop waiting for the GPU, while
another thread's *unlocked* process held it the whole time. `flock`
blocks natively (kernel-level mutex, zero polling) until the lock is
free, then runs, then releases automatically on exit — this is how
concurrent Senior threads under this repo's fan-out model should
coordinate GPU access instead.

## Known gap: a hung process still holds the lock

Isaac Sim has a known failure mode (hit repeatedly in this project) where
it hangs during its own Kit/extension shutdown teardown *after* the
script's actual work is already done and written to disk — the process
keeps holding the flock lock indefinitely, blocking every other queued
job with no indication anything is wrong.

If a queued job seems stuck for an unusually long time, check the
suspected holder's actual GPU/CPU activity (`nvidia-smi` for GPU
utilization, `ps` for CPU%), not just whether the process exists —
near-idle GPU/CPU with the process still alive and its log already
showing a completion/`[DONE]` line means it's hung in teardown, not doing
real work. `kill -TERM <pid>` is safe in that case (the real output was
already written before the hang) and releases the lock immediately for
the next queued job.

## When the lock isn't needed

Plain Isaac-Sim-free scripts (e.g. a `gymnasium`/`stable_baselines3` toy
prototype) don't need the lock at all and are the better choice when a
research question doesn't specifically require Isaac Sim's physics.

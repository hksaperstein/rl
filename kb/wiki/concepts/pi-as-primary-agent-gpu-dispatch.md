# Pi-as-primary-agent GPU dispatch (2026-07-18)

Infra concept, not an RL experiment. The Raspberry Pi became the primary
agent host (no GPU of its own); GPU-heavy work (Isaac Sim training/eval,
vision jobs) is routed elsewhere. Routing priority: **desktop first,
cloud fallback** — full write-up and exact mechanics in `CLAUDE.md`'s
"Pi-as-primary-agent GPU dispatch" section (kept there, not duplicated
here, since it's the operational contract future dispatches read
directly) and the design doc
`docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`.

Three scripts, in dispatch order: `scripts/check_desktop_gpu.sh` (low-level
probe, AVAILABLE/BUSY/UNKNOWN), `scripts/check_gpu_availability.sh`
(desktop-vs-cloud routing decision), `scripts/run_on_desktop_gpu.sh`
(the actual dispatch wrapper — `systemd-inhibit` + detached `tmux`).

**Why this matters for experiment history:** [[unified-multi-die-specialist-distillation]]'s
Task 3.5 is the first RL training task actually dispatched through this
system rather than GCP cloud — chosen specifically because this same task
had already hit real cloud infra friction (SPOT preemption, pip-cache
corruption, a `Linger=no` systemd default killing a detached install) on
a prior attempt, and the desktop already has Isaac Lab installed (no
from-scratch install window to fail mid-way). See `BACKLOG.md`'s "Task
3.5 execution backend" entry for the decision record.

**Known gap as of 2026-07-18:** the desktop has no passwordless sudo, so
the shutdown/sleep inhibitor initially degraded to idle-only for
SSH-dispatched jobs (polkit denies the full inhibitor to non-seated
sessions). A same-day follow-on effort (a separate agent running locally
on the desktop, with real sudo access) replaced the SSH-polling check
with an always-on HTTP status server (`scripts/gpu_status_server.py`)
that also applied the polkit fix directly — see `CLAUDE.md`'s current
"Known gaps" text for whether this is still open or has since been
closed for good.

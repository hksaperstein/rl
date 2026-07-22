# GPU dispatch runbook

Full mechanics for `CLAUDE.md`'s "Pi-as-primary-agent GPU dispatch"
rules. Read this before debugging a dispatch failure or modifying the
dispatch scripts; `CLAUDE.md` itself only keeps the routing rule and a
one-line purpose per script.

## The three scripts

1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe.
   Queries `GET http://home.local:8077/gpu-status` (a small always-on
   HTTP status server on the desktop, `scripts/gpu_status_server.py`,
   see `docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`)
   instead of SSHing to the desktop twice. Judgment: non-empty
   `compute_apps` OR a non-zero `rl_gpu_job_guard_count` (covers both the
   Pi-dispatched `rl-gpu-job` guard and the desktop's own auto-detect
   `rl-gpu-job-auto-detect` guard, added by the same server) means BUSY.
   No SSH fallback if the HTTP call fails — that's UNKNOWN. Requires `jq`
   on the Pi to parse the JSON response (`apt install jq` — not
   preinstalled on Raspberry Pi OS); if missing, the script fails safe to
   UNKNOWN (never a false AVAILABLE), but desktop routing will silently
   never fire until it's installed.
2. **`scripts/check_gpu_availability.sh`** — routing decision. Calls (1)
   and prints `TARGET=desktop` (exit 0) or `TARGET=cloud` (exit 1 if
   BUSY, exit 2 if UNKNOWN) plus a human-readable reason. Does not
   provision cloud infrastructure itself — on `TARGET=cloud`, follow the
   existing `docs/cloud/dispatch-checklist.md` recipe.
3. **`scripts/run_on_desktop_gpu.sh [--detach] <command...>`** — the
   actual dispatch wrapper. Pre-checks via (1) and refuses to proceed on
   BUSY/UNKNOWN. On AVAILABLE, ships the command to the desktop and runs
   it inside a detached `tmux` session (survives a Pi-side SSH
   disconnect — not just backgrounded over the SSH pipe) wrapped in a
   `systemd-inhibit --who=rl-gpu-job` guard (prevents the desktop
   suspending/shutting down mid-job — the concern this whole design
   exists for). Default mode blocks and streams the job's output live
   via a separate log-tailing SSH call, per this repo's blocking-over-
   check-in-early convention (see the cloud dispatch checklist);
   `--detach` dispatches and returns immediately, printing how to check
   on it later (`tmux attach`, `tail -f` the remote log, or
   `check_desktop_gpu.sh`). Exit codes: 0 = success, 1 = BUSY, 2 =
   UNKNOWN, 3 = usage error, 4 = dispatch mechanism itself failed
   (SSH/tmux/inhibitor setup — distinct from the job's own failure), 5 =
   the remote command ran and exited non-zero (exact code printed, not
   preserved as the wrapper's own exit code since arbitrary integers
   don't round-trip through shell exit status).

## GPU status server

`scripts/gpu_status_server.py`, added 2026-07-18: runs as a `systemd
--user` service (`~/.config/systemd/user/gpu-status-server.service`,
local machine config, not committed) on the desktop, started at boot via
`loginctl enable-linger saps` (already enabled on this account) +
`systemctl --user enable --now gpu-status-server.service`. Serves `GET
/gpu-status` on port 8077 (LAN-only, no auth — approved posture, read-only
endpoint on a trusted home network) with GPU telemetry plus the
`compute_apps`/`rl_gpu_job_guard_count` fields `check_desktop_gpu.sh`
actually judges availability on. Also runs the auto-detect shutdown-inhibitor
watchdog described below. See
`docs/superpowers/specs/2026-07-18-gpu-status-server-design.md` for the
full design and endpoint contract.

## Known gaps on the desktop side (as of 2026-07-18)

Both root causes: no passwordless sudo on the desktop, so neither is
fixable from an unattended SSH session — flagging here rather than
working around silently.

- **tmux isn't apt-installed.** Installed userspace-only instead:
  `apt-get download tmux libutempter0` (works without root) then
  `dpkg -x <deb> ~/.local/opt/tmux-extracted` (extraction, no root
  needed either) on the desktop, with a wrapper at
  `~/.local/bin/tmux` that sets `LD_LIBRARY_PATH` for
  `libutempter.so.0` before exec'ing the real binary. Non-interactive
  SSH commands don't source `~/.bashrc`, so `~/.local/bin` is never on
  `PATH` for these sessions — `run_on_desktop_gpu.sh` always invokes
  tmux by its full remote path, never bare `tmux`. If tmux is ever
  properly apt-installed by someone with root later, update
  `REMOTE_TMUX` in `run_on_desktop_gpu.sh` (or symlink `/usr/bin/tmux`
  into `~/.local/bin`).
- **Polkit fix applied 2026-07-18.**
  `/etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules` (root-owned, not
  committed to git) now grants user `saps` unconditional
  `org.freedesktop.login1.inhibit-*` access regardless of seat. Before
  this fix, polkit denied `systemd-inhibit --what=shutdown:sleep:idle`
  for non-seated/unattended sessions, silently degrading to idle-only
  protection (shutdown/sleep left unprotected); this fix removes that
  degradation for both this Pi-dispatch job guard and the desktop's own
  auto-detect watchdog
  (`scripts/gpu_status_server.py`'s `InhibitWatchdog`, holding a
  `rl-gpu-job-auto-detect` guard whenever `nvidia-smi` shows any active
  compute app, independent of whether the job was Pi-dispatched).
  **Open follow-up:** this was verified to work for an active login
  session; full proof for the unattended (linger, no-seat) case — the
  actual scenario this fix targets — requires observing the guard
  correctly acquire `shutdown:sleep` (not just `idle`) after a real
  reboot with no one logged in. Confirm this the next time this desktop
  reboots, and remove this note once observed.

## Live-test confirmation

All three scripts were tested live against the real desktop (not just
exit codes): a genuine `sleep`-based job was dispatched, and mid-run,
both `check_desktop_gpu.sh`/`check_gpu_availability.sh` and a raw
`systemd-inhibit --list` on the desktop were confirmed to show the
`rl-gpu-job` guard actually held.

# GPU Status Server — HTTP replacement for the Pi's SSH-based availability check

**Date:** 2026-07-18
**Status:** Approved (direct user decisions recorded inline below).
**Scope:** Infrastructure tooling for the existing Pi-as-primary-agent GPU
dispatch system (`CLAUDE.md` "Pi-as-primary-agent GPU dispatch", 2026-07-18).
Not a Tier 1/Tier 2 RL experiment — no falsifiable-hypothesis gate applies.

## Background

The Pi (primary agent host, no GPU of its own) already has a working
dispatch system, committed on `origin/main` the same day as this design:

- `scripts/check_desktop_gpu.sh` — SSHes to the desktop twice (once for
  `nvidia-smi --query-compute-apps`, once for `systemd-inhibit --list`
  filtered to a lock named `rl-gpu-job`) to decide AVAILABLE (exit 0) /
  BUSY (exit 1) / UNKNOWN (exit 2, unreachable or check itself failed).
- `scripts/check_gpu_availability.sh` — calls the above and prints a
  `TARGET=desktop`/`TARGET=cloud` routing line.
- `scripts/run_on_desktop_gpu.sh` — the actual dispatch wrapper: SSHes a
  command to the desktop, runs it in a detached `tmux` session guarded by
  `systemd-inhibit --who=rl-gpu-job`.

Two SSH round-trips per availability check is more than this needs. This
design replaces `check_desktop_gpu.sh`'s internals with one HTTP GET to a
small always-on status server on the desktop, and separately adds an
auto-detect shutdown-inhibitor so GPU usage started *directly on the
desktop* (not via Pi dispatch) also gets shutdown protection, which today
it does not.

`check_gpu_availability.sh` and `run_on_desktop_gpu.sh` are unchanged —
dispatch stays SSH-based. A status server must never become a remote-exec
endpoint; that boundary is intentional, not an oversight.

## Components

1. **`scripts/gpu_status_server.py`** (new) — Python stdlib only
   (`http.server` + `json`, no new dependency). Two responsibilities:
   - Serves `GET /gpu-status` (see contract below), querying `nvidia-smi`
     fresh on every request (cheap enough not to need caching).
   - Runs a background poll loop (interval: 10s) that watches
     `nvidia-smi --query-compute-apps`. On empty→non-empty transition,
     spawns `systemd-inhibit --what=shutdown:sleep --who=rl-gpu-job-auto-detect
     --why="GPU actively in use (auto-detected)" -- sleep infinity` and
     holds the child process handle. On non-empty→empty, terminates that
     child, releasing the lock. This is additive to (and independent of)
     the existing `rl-gpu-job` guard `run_on_desktop_gpu.sh` already
     acquires for Pi-dispatched jobs — both are visible separately in
     `systemd-inhibit --list`.
2. **`~/.config/systemd/user/gpu-status-server.service`** (new, not
   committed — installed on the desktop only) — `Restart=always`,
   `WantedBy=default.target`. Requires `loginctl enable-linger saps`
   (one-time, needs sudo) so it starts at boot with no login session.
3. **`/etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules`** (new, not
   committed, root-owned system file) — grants user `saps` unconditional
   `org.freedesktop.login1.inhibit-*` access regardless of seat. Required
   because a linger-started user service has no active login "seat," and
   polkit's default rules only grant inhibit actions to seated sessions —
   the exact gap already documented in CLAUDE.md for the Pi's SSH-dispatched
   jobs (which degrade to idle-only today). This fix removes that
   degradation for both the existing Pi-dispatch path and the new
   auto-detect watchdog. **User-decision record:** applying this was
   explicitly confirmed with the user (modifies system-wide policy, needs
   interactive sudo).
4. **`scripts/check_desktop_gpu.sh`** (rewrite) — replaces both SSH calls
   with one `curl` to `http://home.local:8077/gpu-status`. Same exit-code
   contract (0/1/2) and same AVAILABLE/BUSY judgment (non-empty
   `compute_apps` OR non-zero guard count → BUSY), so callers
   (`check_gpu_availability.sh`, `run_on_desktop_gpu.sh`) need no changes.
5. **`CLAUDE.md`** (edit) — update the "Pi-as-primary-agent GPU dispatch"
   section: describe the HTTP mechanism in place of the 2-SSH-call one,
   document the auto-detect guard and the polkit fix as applied (removing
   the "known gap" language once the fix is confirmed live), document the
   one-time desktop setup steps (linger + polkit rule + service install).
   This section update *is* the contract the Pi-side agent reads on its
   next pull — no separate docs file.

## Endpoint contract

`GET http://home.local:8077/gpu-status` →

```json
{
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "utilization_pct": 12,
  "memory_used_mb": 850,
  "memory_total_mb": 16384,
  "temperature_c": 47,
  "power_draw_w": 45.2,
  "compute_apps": [{"pid": 1234, "process_name": "python3", "used_memory_mb": 800}],
  "rl_gpu_job_guard_count": 0,
  "checked_at": "2026-07-18T17:32:00-04:00"
}
```

`compute_apps` and `rl_gpu_job_guard_count` are load-bearing (drive the
AVAILABLE/BUSY judgment, unchanged from today's SSH-based logic — guard
count is any `systemd-inhibit --list` entry whose `--who` starts with
`rl-gpu-job`, which covers both `rl-gpu-job` and the new
`rl-gpu-job-auto-detect`). The rest is telemetry for visibility, not
currently consumed by any script logic.

Binds `0.0.0.0:8077` — LAN-only exposure is acceptable per user decision
(trusted home network, read-only endpoint, no auth).

## Error handling

`check_desktop_gpu.sh`'s `curl` call: `--connect-timeout 3 --max-time 5
--retry 2 --retry-delay 1`. Any failure (timeout, connection refused,
non-200) → UNKNOWN, exit 2. **No SSH fallback** — direct user decision:
trust `Restart=always` + boot-start to keep the server up; a failed check
must never be read as available, matching the philosophy already in
`check_desktop_gpu.sh`'s existing UNKNOWN handling.

## Testing plan

1. `curl localhost:8077/gpu-status` on the desktop directly — verify JSON
   shape and that `compute_apps`/guard count match a live `nvidia-smi`/
   `systemd-inhibit --list` read taken at the same time.
2. `ssh pi@agent.local curl http://home.local:8077/gpu-status` — confirm
   LAN reachability from the Pi's actual vantage point.
3. Rewritten `check_desktop_gpu.sh` run from the Pi, both while idle
   (expect AVAILABLE) and while a real job is dispatched via the untouched
   `run_on_desktop_gpu.sh` (expect BUSY via the existing `rl-gpu-job` guard,
   confirming the HTTP path sees what the SSH path used to see directly).
4. Auto-detect guard: start a GPU-using process directly on the desktop
   (not via Pi dispatch — e.g. a manual `nvidia-smi`-visible job), confirm
   `rl-gpu-job-auto-detect` appears in `systemd-inhibit --list` within one
   poll interval, and disappears within one interval after the process
   ends.
5. Polkit fix: after installing the rule, re-run the
   `systemd-run --user --scope -- systemd-inhibit --what=shutdown:sleep:idle
   ...` probe (this design's discovery step used this to find the gap) and
   confirm it now succeeds where it previously would have been denied.

No pytest suite — this is small enough (~1 script, mostly stdlib) that the
live tests above are the appropriate verification level, per this repo's
"real evidence over proxies" standard; a unit-test harness would mostly be
mocking `subprocess` calls to `nvidia-smi`/`systemd-inhibit`, which adds
little over exercising the real thing on real hardware.

## Out of scope

- Auth/TLS on the HTTP endpoint (LAN-only, read-only, trusted network).
- Any change to `run_on_desktop_gpu.sh`'s dispatch mechanism (stays SSH).
- A general-purpose GPU dashboard/UI — this is a machine-readable status
  endpoint for scripts, not a human-facing page.

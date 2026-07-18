# GPU Status Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Pi's 2-SSH-round-trip GPU availability check with one
HTTP GET to a small always-on status server on this desktop, and add an
auto-detect shutdown-inhibitor so GPU usage started directly on the
desktop (not via Pi dispatch) also gets shutdown protection.

**Architecture:** `scripts/gpu_status_server.py` (Python stdlib only) runs
as a `systemd --user` service on this desktop, serving `GET /gpu-status`
and running a background poll loop that acquires/releases a
`systemd-inhibit` shutdown lock based on live `nvidia-smi` compute-app
state. `scripts/check_desktop_gpu.sh` (on the Pi side, same repo) is
rewritten to `curl` this endpoint instead of SSHing twice. A one-time
polkit rule (installed with sudo, not committed to git) removes a
seat-based permission gap that would otherwise silently degrade the
inhibitor to idle-only when this runs unattended at boot.

**Tech Stack:** Python 3 stdlib (`http.server`, `subprocess`, `json`,
`threading`), bash + `curl` + `jq`, systemd user services, polkit.

## Global Constraints

- Server binds `0.0.0.0:8077`, no auth — LAN-only exposure, approved by
  the user (trusted home network, read-only endpoint).
- No SSH fallback in `check_desktop_gpu.sh` — any HTTP failure means
  UNKNOWN (exit 2), never treated as available. Direct user decision.
- `check_gpu_availability.sh` and `run_on_desktop_gpu.sh` are **not**
  modified by this plan — dispatch stays SSH-based; the status server
  must never become a remote-exec endpoint.
- `nvidia-smi` field order/format verified live on this GPU (RTX 5070
  Ti): `--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits`
  → `NVIDIA GeForce RTX 5070 Ti, 4, 636, 16303, 31, 30.89`.
- The systemd user service file and the polkit rule file are **not**
  committed to git — they're local machine config, documented in
  CLAUDE.md as install steps/snippets (matching this repo's existing
  convention for the tmux/polkit workarounds already documented there).
- The Pi confirmed reachable at `pi@agent.local`; its repo clone lives at
  `~/projects/rl` there; the desktop is reachable from the Pi as SSH alias
  `desktop` / HTTP hostname `home.local` (mDNS, already proven working via
  the existing SSH alias).
- `loginctl show-user saps -p Linger` already returns `Linger=yes` on this
  desktop — no `loginctl enable-linger` step needed in this plan.

---

### Task 1: GPU status HTTP endpoint (no watchdog yet)

**Files:**
- Create: `scripts/gpu_status_server.py`

**Interfaces:**
- Produces: `query_compute_apps() -> list[dict]` (keys: `pid: int`,
  `process_name: str`, `used_memory_mb: int`), `query_gpu_telemetry() ->
  dict` (keys: `gpu_name: str`, `utilization_pct: int`, `memory_used_mb:
  int`, `memory_total_mb: int`, `temperature_c: int`, `power_draw_w:
  float`), `query_guard_count() -> int`, `build_status() -> dict` (merges
  the above plus `checked_at: str` ISO8601 and
  `rl_gpu_job_guard_count: int`). `PORT = 8077`, `GUARD_WHO =
  "rl-gpu-job-auto-detect"` module constants. Task 2 imports/extends this
  same file; Task 5's `check_desktop_gpu.sh` consumes the JSON shape
  `build_status()` produces over HTTP.

- [ ] **Step 1: Write the server script**

```python
#!/usr/bin/env python3
"""HTTP status server exposing this desktop's GPU state for the Pi's
availability check (replaces 2 SSH round-trips with one local read).
See docs/superpowers/specs/2026-07-18-gpu-status-server-design.md.
"""
import json
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8077
POLL_INTERVAL_SECONDS = 10
GUARD_WHO = "rl-gpu-job-auto-detect"


def query_compute_apps():
    """Returns list of {"pid", "process_name", "used_memory_mb"} dicts."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    apps = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, name, mem = [p.strip() for p in line.split(",")]
        apps.append({"pid": int(pid), "process_name": name, "used_memory_mb": int(mem)})
    return apps


def query_gpu_telemetry():
    """Returns the single-GPU telemetry fields for the status response."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    name, util, mem_used, mem_total, temp, power = [p.strip() for p in out.split(",")]
    return {
        "gpu_name": name,
        "utilization_pct": int(util),
        "memory_used_mb": int(mem_used),
        "memory_total_mb": int(mem_total),
        "temperature_c": int(temp),
        "power_draw_w": float(power),
    }


def query_guard_count():
    """Counts systemd-inhibit holders whose line mentions rl-gpu-job
    (covers both the Pi-dispatched `rl-gpu-job` guard and this server's
    own `rl-gpu-job-auto-detect` guard) -- same whole-line substring match
    already used and tested in the original SSH-based check_desktop_gpu.sh."""
    out = subprocess.run(
        ["systemd-inhibit", "--list", "--no-legend"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout
    return sum(1 for line in out.splitlines() if "rl-gpu-job" in line)


def build_status():
    apps = query_compute_apps()
    status = query_gpu_telemetry()
    status["compute_apps"] = apps
    status["rl_gpu_job_guard_count"] = query_guard_count()
    status["checked_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    return status


class GpuStatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/gpu-status":
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = json.dumps(build_status()).encode()
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # keep the systemd journal clean -- access logs add no value here


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), GpuStatusHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it fails predictably before nvidia-smi has anything to report on a dead port (sanity: script starts, doesn't crash on import)**

Run: `python3 -c "import ast; ast.parse(open('scripts/gpu_status_server.py').read())" && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 3: Start the server in the background and curl it locally**

Run:
```bash
python3 scripts/gpu_status_server.py &
SERVER_PID=$!
sleep 1
curl -s http://localhost:8077/gpu-status | python3 -m json.tool
```
Expected: valid JSON with keys `gpu_name`, `utilization_pct`,
`memory_used_mb`, `memory_total_mb`, `temperature_c`, `power_draw_w`,
`compute_apps` (a list, likely empty if GPU is idle), `rl_gpu_job_guard_count`
(an integer, likely 0), `checked_at` (an ISO8601 timestamp). Cross-check
`memory_total_mb` against a direct `nvidia-smi --query-gpu=memory.total
--format=csv,noheader,nounits` run at the same time -- values must match.

- [ ] **Step 4: Verify the 404 path and stop the server**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8077/anything-else`
Expected: `404`

Run: `kill "$SERVER_PID"`

- [ ] **Step 5: Commit**

```bash
git add scripts/gpu_status_server.py
git commit -m "feat: add GPU status HTTP server (endpoint only, no watchdog yet)"
```

---

### Task 2: Auto-detect shutdown-inhibitor watchdog

**Files:**
- Modify: `scripts/gpu_status_server.py` (append; do not change Task 1's
  functions/classes)

**Interfaces:**
- Consumes: `query_compute_apps()`, `GUARD_WHO`, `POLL_INTERVAL_SECONDS`
  from Task 1.
- Produces: `InhibitWatchdog` class with `.run_forever()` method (no
  return value, runs until process exit). `main()` from Task 1 is
  modified to start it in a background thread before serving.

- [ ] **Step 1: Add the watchdog class and wire it into `main()`**

Add near the top of the file (with the other imports):
```python
import threading
import time
```

Add after `query_guard_count()` (before `build_status`):
```python
class InhibitWatchdog:
    """Holds a systemd-inhibit shutdown:sleep lock while the GPU has any
    active compute app, independent of the Pi-dispatch `rl-gpu-job` guard
    that run_on_desktop_gpu.sh already acquires for its own jobs."""

    def __init__(self, poll_interval):
        self.poll_interval = poll_interval
        self._proc = None

    def _acquire(self):
        self._proc = subprocess.Popen([
            "systemd-inhibit", "--what=shutdown:sleep",
            f"--who={GUARD_WHO}",
            "--why=GPU actively in use (auto-detected)",
            "--", "sleep", "infinity",
        ])

    def _release(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def run_forever(self):
        while True:
            try:
                busy = len(query_compute_apps()) > 0
            except Exception:
                busy = False  # nvidia-smi hiccup: don't hold a stale lock on bad data
            holding = self._proc is not None and self._proc.poll() is None
            if busy and not holding:
                self._acquire()
            elif not busy and holding:
                self._release()
            time.sleep(self.poll_interval)
```

Replace `main()`:
```python
def main():
    watchdog = InhibitWatchdog(POLL_INTERVAL_SECONDS)
    threading.Thread(target=watchdog.run_forever, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), GpuStatusHandler)
    server.serve_forever()
```

- [ ] **Step 2: Syntax check**

Run: `python3 -c "import ast; ast.parse(open('scripts/gpu_status_server.py').read())" && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 3: Live test — trigger a real GPU compute app and watch the guard appear**

Run in the background:
```bash
python3 scripts/gpu_status_server.py &
SERVER_PID=$!
sleep 1
systemd-inhibit --list --no-legend | grep -c rl-gpu-job-auto-detect
```
Expected: `0` (nothing running on the GPU yet)

Run:
```bash
/home/saps/projects/rl/vision/.venv/bin/python -c "
import torch, time
x = torch.zeros(1000, 1000, device='cuda')
time.sleep(25)
" &
TORCH_PID=$!
sleep 12
curl -s http://localhost:8077/gpu-status | python3 -m json.tool
systemd-inhibit --list --no-legend | grep rl-gpu-job-auto-detect
```
Expected: the curl output's `compute_apps` list is non-empty (the torch
process's PID shows up), and the `systemd-inhibit --list` line for
`rl-gpu-job-auto-detect` is present (waited 12s to be safely past one
10s poll interval).

- [ ] **Step 4: Verify release after the GPU app exits**

Run:
```bash
wait "$TORCH_PID"
sleep 12
systemd-inhibit --list --no-legend | grep -c rl-gpu-job-auto-detect
curl -s http://localhost:8077/gpu-status | python3 -c "import json,sys; print(json.load(sys.stdin)['compute_apps'])"
```
Expected: `0` (guard released) and `[]` (no compute apps) — waited 12s
past the torch script's own exit to clear one more poll interval.

- [ ] **Step 5: Stop the manual server**

Run: `kill "$SERVER_PID"`

- [ ] **Step 6: Commit**

```bash
git add scripts/gpu_status_server.py
git commit -m "feat: add auto-detect shutdown-inhibitor watchdog to GPU status server"
```

---

### Task 3: Install as a systemd --user service, verify boot-start and crash-restart

**Files:**
- Create (local machine only, NOT committed to git):
  `~/.config/systemd/user/gpu-status-server.service`

**Interfaces:**
- Consumes: `scripts/gpu_status_server.py` from Tasks 1-2, at its
  absolute repo path.
- Produces: a running `gpu-status-server.service` unit that Task 5's live
  cross-host test depends on being up.

- [ ] **Step 1: Write the unit file**

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/gpu-status-server.service <<'EOF'
[Unit]
Description=GPU status HTTP server for Pi-dispatch availability checks
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/saps/projects/rl/scripts/gpu_status_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
```

- [ ] **Step 2: Enable and start it**

Run:
```bash
systemctl --user daemon-reload
systemctl --user enable --now gpu-status-server.service
systemctl --user is-enabled gpu-status-server.service
systemctl --user is-active gpu-status-server.service
```
Expected: `enabled` then `active`

- [ ] **Step 3: Verify it actually serves the endpoint**

Run: `curl -s http://localhost:8077/gpu-status | python3 -m json.tool`
Expected: same JSON shape as Task 1 Step 3.

- [ ] **Step 4: Verify crash-restart**

Run:
```bash
SVC_PID="$(systemctl --user show -p MainPID --value gpu-status-server.service)"
kill -9 "$SVC_PID"
sleep 6
systemctl --user is-active gpu-status-server.service
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8077/gpu-status
```
Expected: `active` and `200` — confirms `Restart=always` actually recovers
from a hard kill, not just a clean stop.

- [ ] **Step 5: Verify boot-start is enabled (linger already on, confirmed in Global Constraints)**

Run: `loginctl show-user saps -p Linger`
Expected: `Linger=yes` (already true; this step is a confirmation, not a
change — if it ever reads `no`, run `loginctl enable-linger saps`, which
this session already verified succeeds without needing sudo on this
account).

- [ ] **Step 6: No git commit for this task** — the unit file is local
machine config per the Global Constraints, not tracked in the repo. Task
6 documents its exact content in CLAUDE.md instead.

---

### Task 4: One-time polkit fix (needs the user's sudo password)

**Files:**
- Create (local machine only, root-owned, NOT committed to git):
  `/etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: unconditional `org.freedesktop.login1.inhibit-*` permission
  for user `saps`, which Task 2's watchdog and the existing
  Pi-dispatched `rl-gpu-job` guard both need to get real
  `shutdown:sleep` protection (not just `idle`) once this service is
  running unattended (no active login session/seat) — e.g. right after a
  reboot, before anyone logs in.

**This step needs your sudo password interactively — the agent's Bash
tool cannot relay a live password prompt, so run this yourself.**

- [ ] **Step 1 (you run this): install the rule**

Type this in your terminal (prefix with `!` if running inside this Claude
Code session):
```bash
sudo tee /etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules > /dev/null <<'EOF'
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.freedesktop.login1.inhibit-") == 0 &&
        subject.user == "saps") {
        return polkit.Result.YES;
    }
});
EOF
```
No restart needed — polkit picks up rule file changes automatically.

- [ ] **Step 2: Confirm the file exists with the right ownership (needs sudo to read the directory listing, since `/etc/polkit-1/rules.d/` is `root:polkitd`, mode 750)**

Run (you, with `!` prefix, or let the agent run it — this is a read-only
`ls`, safe either way):
```bash
sudo ls -la /etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules
```
Expected: the file listed, owned by `root`.

- [ ] **Step 3: Known limitation — full behavioral proof requires a real reboot**

This session already confirmed `systemd-inhibit --what=shutdown:sleep:idle`
is granted for the *current* interactive login session, both directly and
under `systemd-run --user --scope` — but that test can't distinguish
"granted because the rule works" from "granted because a login session
happens to be active right now" (the same reason the Pi's earlier SSH-based
probe, with no session/seat at all, hit "Access denied" before this fix).
The only conclusive test is behavior after a real logout/reboot, when the
`gpu-status-server` service is running under linger with no active
session. Do not claim this fix is behaviorally proven until that's been
observed once — note it as an open follow-up in CLAUDE.md (Task 6) rather
than asserting certainty now.

- [ ] **Step 4: No git commit for this task** — root-owned system file,
not tracked in the repo, per the Global Constraints.

---

### Task 5: Rewrite `check_desktop_gpu.sh` as an HTTP client

**Files:**
- Modify: `scripts/check_desktop_gpu.sh` (full rewrite; current version
  is 68 lines, SSH-based — see `git show HEAD:scripts/check_desktop_gpu.sh`
  for the version being replaced)

**Interfaces:**
- Consumes: `GET http://home.local:8077/gpu-status` (Tasks 1-3's server,
  must be reachable from the Pi over the LAN — confirmed resolvable via
  mDNS already, since the Pi's own `~/.ssh/config` alias `desktop` uses
  the same `home.local` hostname).
- Produces: unchanged exit-code contract (0=AVAILABLE, 1=BUSY,
  2=UNKNOWN) and unchanged stdout format (`AVAILABLE: ...` / `BUSY:` +
  app lines / `UNKNOWN: ...`) — `check_gpu_availability.sh` and
  `run_on_desktop_gpu.sh` call this script and parse its exit code only,
  so as long as the exit codes match, neither needs to change.

- [ ] **Step 1: Write the rewritten script**

```bash
#!/usr/bin/env bash
# Reports whether the desktop's GPU (saps@home.local, HTTP status server
# on port 8077) is free for a new job dispatched from this Pi.
#
# Queries scripts/gpu_status_server.py's /gpu-status endpoint over HTTP
# instead of SSHing to the desktop twice (nvidia-smi + systemd-inhibit
# --list) -- see docs/superpowers/specs/2026-07-18-gpu-status-server-design.md.
# Judgment call is unchanged from the SSH-based version: non-empty
# compute_apps OR a non-zero rl_gpu_job_guard_count (covers both the
# Pi-dispatched `rl-gpu-job` guard and the desktop's own auto-detect
# `rl-gpu-job-auto-detect` guard) means BUSY.
#
# Exit codes are distinct on purpose so a caller can branch correctly
# instead of treating every non-zero the same way:
#   0 = AVAILABLE (desktop reachable, GPU idle, no job guard held)
#   1 = BUSY       (desktop reachable, but GPU or job guard in use)
#   2 = UNKNOWN     (desktop unreachable or the check itself failed --
#                    NOT the same as available; caller must not treat
#                    this as a green light, fall back to cloud or stop)
#
# No SSH fallback if the HTTP server is down -- direct project decision
# (see the design doc): trust systemd Restart=always + boot-start to keep
# it up, and treat any failure to reach it as UNKNOWN rather than risking
# a false green light.
#
# Usage: scripts/check_desktop_gpu.sh
set -Eeuo pipefail

STATUS_URL="http://home.local:8077/gpu-status"

fail_unknown() {
  echo "UNKNOWN: $1" >&2
  exit 2
}

if ! RESPONSE="$(curl --silent --show-error --fail \
    --connect-timeout 3 --max-time 5 --retry 2 --retry-delay 1 \
    "$STATUS_URL" 2>&1)"; then
  fail_unknown "could not reach gpu status server at $STATUS_URL: $RESPONSE"
fi

if ! APPS_COUNT="$(echo "$RESPONSE" | jq '.compute_apps | length' 2>&1)"; then
  fail_unknown "malformed response from gpu status server: $APPS_COUNT"
fi

if ! GUARD_COUNT="$(echo "$RESPONSE" | jq '.rl_gpu_job_guard_count' 2>&1)"; then
  fail_unknown "malformed response from gpu status server: $GUARD_COUNT"
fi

if [ "$APPS_COUNT" -eq 0 ] && [ "$GUARD_COUNT" -eq 0 ]; then
  echo "AVAILABLE: desktop GPU idle, no rl-gpu-job guard active"
  exit 0
fi

echo "BUSY:"
echo "$RESPONSE" | jq -c '.compute_apps[]'
if [ "$GUARD_COUNT" -gt 0 ]; then
  echo "(rl-gpu-job shutdown guard is held -- a dispatched job is claiming the GPU even if nvidia-smi shows nothing yet)"
fi
exit 1
```

- [ ] **Step 2: Local sanity test (on the desktop, targeting itself)**

Run: `curl -s http://localhost:8077/gpu-status > /dev/null && echo "server reachable, script logic already exercised in Task 1-2 tests"`
(This just confirms the server this script depends on is up; the script
itself must be tested from the Pi's actual vantage point, see Step 3,
since it hardcodes `home.local`.)

- [ ] **Step 3: Isolated reachability check from the Pi's actual vantage point**

Run (from the desktop, over SSH to the Pi):
```bash
ssh pi@agent.local 'curl -s http://home.local:8077/gpu-status | python3 -m json.tool'
```
Expected: same JSON shape as Task 1 Step 3, fetched from the Pi's network
position — isolates "can the Pi reach the server at all" from the
script's own parsing logic, tested next.

- [ ] **Step 4: Test from the Pi, idle case**

Run (from the desktop, over SSH to the Pi, since the Pi has the same repo cloned at `~/projects/rl`):
```bash
ssh pi@agent.local 'cd ~/projects/rl && git pull --ff-only && bash scripts/check_desktop_gpu.sh; echo "exit:$?"'
```
Expected: `AVAILABLE: desktop GPU idle, no rl-gpu-job guard active` and `exit:0` (assuming GPU is actually idle at test time — confirm with a local `nvidia-smi` check first if unsure).

- [ ] **Step 5: Test from the Pi, busy case (real dispatched job)**

Run (from the desktop):
```bash
ssh pi@agent.local 'cd ~/projects/rl && scripts/run_on_desktop_gpu.sh --detach sleep 30'
sleep 3
ssh pi@agent.local 'cd ~/projects/rl && bash scripts/check_desktop_gpu.sh; echo "exit:$?"'
```
Expected: `BUSY:` output including the `rl-gpu-job shutdown guard is held`
line (the guard shows up even though `sleep 30` uses no GPU memory — this
confirms the HTTP path sees `rl_gpu_job_guard_count` correctly, the same
signal the old SSH-based version read via `systemd-inhibit --list`) and
`exit:1`.

- [ ] **Step 6: Test from the Pi, unreachable case**

Run (from the desktop):
```bash
systemctl --user stop gpu-status-server.service
ssh pi@agent.local 'cd ~/projects/rl && time bash scripts/check_desktop_gpu.sh; echo "exit:$?"'
systemctl --user start gpu-status-server.service
sleep 1
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8077/gpu-status
```
Expected: `UNKNOWN: could not reach gpu status server ...` and `exit:2`,
completing in well under the old SSH check's ~8s timeout (the retry
policy caps this well below that). Final curl confirms the server is
back up after the test.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_desktop_gpu.sh
git commit -m "refactor: check_desktop_gpu.sh queries HTTP status server instead of 2x SSH"
```

---

### Task 6: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md` (the "Pi-as-primary-agent GPU dispatch" section,
  added 2026-07-18 — replace its description of `check_desktop_gpu.sh`'s
  mechanism and its "Known gaps" polkit note)

**Interfaces:**
- Consumes: final state of all prior tasks (this is a documentation-only
  task, no code interfaces).
- Produces: the canonical description of this mechanism for any future
  agent (Pi-side or desktop-side) reading CLAUDE.md — this doc update,
  plus the rewritten `check_desktop_gpu.sh`, together constitute the full
  contract per the design doc.

- [ ] **Step 1: Read the current section**

Run: `grep -n "Pi-as-primary-agent GPU dispatch" -A 100 CLAUDE.md | head -120`

Locate the exact line range of the section (from its `## ` heading to the
next `## ` heading) to replace in Step 2.

- [ ] **Step 2: Replace the section**

Replace the `check_desktop_gpu.sh` bullet (numbered `1.` under "Three
scripts implement this") to describe the HTTP mechanism instead of the
2-SSH-call one, e.g.:

```markdown
1. **`scripts/check_desktop_gpu.sh`** — low-level availability probe.
   Queries `GET http://home.local:8077/gpu-status` (a small always-on
   HTTP status server on the desktop, `scripts/gpu_status_server.py`,
   see `docs/superpowers/specs/2026-07-18-gpu-status-server-design.md`)
   instead of SSHing to the desktop twice. Same judgment as before:
   non-empty `compute_apps` OR a non-zero `rl_gpu_job_guard_count`
   (covers both the Pi-dispatched `rl-gpu-job` guard and the desktop's
   own auto-detect `rl-gpu-job-auto-detect` guard, added by the same
   server) means BUSY. No SSH fallback if the HTTP call fails — that's
   UNKNOWN, same as before.
```

Replace the "Known gaps on the desktop side" section's polkit bullet
(the `systemd-inhibit --what=shutdown:sleep:idle is denied by polkit`
one) with:

```markdown
- **Polkit fix applied 2026-07-18.**
  `/etc/polkit-1/rules.d/49-rl-gpu-job-inhibit.rules` (root-owned, not
  committed to git) now grants user `saps` unconditional
  `org.freedesktop.login1.inhibit-*` access regardless of seat, removing
  the shutdown/sleep degradation described above for both this
  Pi-dispatch job guard and the desktop's own auto-detect watchdog
  (`scripts/gpu_status_server.py`'s `InhibitWatchdog`, holding a
  `rl-gpu-job-auto-detect` guard whenever `nvidia-smi` shows any active
  compute app, independent of whether the job was Pi-dispatched).
  **Open follow-up:** this was verified to work for an active login
  session; full proof for the unattended (linger, no-seat) case — the
  actual scenario this fix targets — requires observing the guard
  correctly acquire `shutdown:sleep` (not just `idle`) after a real
  reboot with no one logged in. Confirm this the next time this desktop
  reboots, and remove this note once observed.
```

Add a new bullet describing the server itself, after the numbered list of
three scripts (or wherever reads most naturally given the final section
structure):

```markdown
**GPU status server (`scripts/gpu_status_server.py`, added 2026-07-18):**
runs as a `systemd --user` service (`~/.config/systemd/user/gpu-status-server.service`,
local machine config, not committed) on the desktop, started at boot via
`loginctl enable-linger saps` (already enabled on this account) +
`systemctl --user enable --now gpu-status-server.service`. Serves `GET
/gpu-status` on port 8077 (LAN-only, no auth — approved posture, read-only
endpoint on a trusted home network) with GPU telemetry plus the
`compute_apps`/`rl_gpu_job_guard_count` fields `check_desktop_gpu.sh`
actually judges availability on. Also runs the auto-detect shutdown-inhibitor
watchdog described above. See
`docs/superpowers/specs/2026-07-18-gpu-status-server-design.md` for the
full design and endpoint contract.
```

- [ ] **Step 3: Verify the doc renders sensibly**

Run: `grep -n "Pi-as-primary-agent GPU dispatch" -A 100 CLAUDE.md | head -140`

Read through it once — confirm no leftover references to the old 2-SSH-call
mechanism remain, and the section reads coherently top to bottom.

- [ ] **Step 4: Commit (auto-pushes via the existing PostToolUse hook)**

```bash
git add CLAUDE.md
git commit -m "docs: update Pi-as-primary-agent GPU dispatch section for HTTP status server"
git log origin/main --oneline -1
```
Expected: the last line shows the new commit already on `origin/main`
(the repo's `PostToolUse` hook pushes automatically after any `git
commit`) — confirms the Pi-side agent will see this on its own next pull,
completing the "let the other agent know how to interact" requirement
from the original request.

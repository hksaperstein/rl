#!/usr/bin/env python3
"""HTTP status server exposing this desktop's GPU state for the Pi's
availability check (replaces 2 SSH round-trips with one local read).
See docs/superpowers/specs/2026-07-18-gpu-status-server-design.md.
"""
import atexit
import json
import signal
import subprocess
import sys
import threading
import time
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
    watchdog = InhibitWatchdog(POLL_INTERVAL_SECONDS)
    threading.Thread(target=watchdog.run_forever, daemon=True).start()

    # Release the held shutdown-inhibit lock on normal termination (SIGTERM
    # from systemctl stop/restart or a plain `kill`, SIGINT from Ctrl-C) so
    # a server crash/restart can't orphan the systemd-inhibit/sleep-infinity
    # child and permanently block shutdown. This cannot cover SIGKILL
    # (kill -9) -- that signal is uncatchable by any process, an inherent
    # limit, not something fixable here.
    atexit.register(watchdog._release)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))

    server = ThreadingHTTPServer(("0.0.0.0", PORT), GpuStatusHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

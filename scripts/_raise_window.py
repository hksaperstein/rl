"""Best-effort helper: repeatedly raise/activate an X11 window matching a
title substring for a fixed duration. Used by drive_joints_demo.py so the
Isaac Sim viewport actually comes to the front on desktops that don't
auto-focus new windows. Not part of the core sim-foundation pipeline.

Usage: python3 _raise_window.py "<title substring>" <duration_seconds>
"""

import subprocess
import sys
import time

from Xlib import X, display
from Xlib.protocol import event


def find_window(title_substr: str):
    out = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if title_substr in line and "mutter-x11-frames" in line:
            return int(line.strip().split()[0], 16)
    return None


def main() -> None:
    title_substr = sys.argv[1]
    duration = float(sys.argv[2])

    d = display.Display()
    root = d.screen().root
    net_active_window = d.intern_atom("_NET_ACTIVE_WINDOW")

    deadline = time.time() + duration
    win_id = None
    while time.time() < deadline:
        if win_id is None:
            win_id = find_window(title_substr)
        if win_id is not None:
            try:
                win = d.create_resource_object("window", win_id)
                ev = event.ClientMessage(
                    window=win, client_type=net_active_window, data=(32, [1, X.CurrentTime, 0, 0, 0])
                )
                root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
                win.configure(stack_mode=X.Above)
                d.sync()
            except Exception:
                win_id = None  # window closed or not found anymore, keep looking
        time.sleep(0.5)


if __name__ == "__main__":
    main()

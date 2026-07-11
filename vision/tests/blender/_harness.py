"""
Shared test runner for Blender-background test scripts.

Blender's `--background --python script.py` mode exits with code 0 even when
the script raises an uncaught exception (verified empirically against
Blender 5.1.2) — a bare `assert` at module scope will NOT fail the shell
command. Every Blender-dependent test script must call `run_and_report`
instead of relying on Python's default exception propagation.
"""
import sys
import traceback


def run_and_report(fn):
    try:
        fn()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)

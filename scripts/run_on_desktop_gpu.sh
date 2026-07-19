#!/usr/bin/env bash
# Dispatch an arbitrary command to run on the desktop's GPU (saps@home.local,
# ssh alias "desktop"), safely: pre-checks availability via
# check_desktop_gpu.sh, then runs the job on the desktop under BOTH a
# systemd-inhibit shutdown/sleep/idle guard (name "rl-gpu-job", the exact
# string check_desktop_gpu.sh greps for) AND inside a detached tmux session
# (so the job survives an SSH disconnect from the Pi side, not just a
# backgrounded pipe that dies with the SSH connection).
#
# Usage:
#   scripts/run_on_desktop_gpu.sh [--detach] <command> [args...]
#
#   --detach   Dispatch and return immediately once the remote job has
#              actually started (tmux session + inhibitor confirmed up),
#              without waiting for it to finish. Prints the session name
#              and log path so it can be checked on later. Default (no
#              flag) is to BLOCK and stream the job's output live until
#              it finishes -- this project's convention (CLAUDE.md /
#              docs/cloud/dispatch-checklist.md) is to block on
#              long-running dispatches rather than check in early, so
#              that is the default here.
#
# Example:
#   scripts/run_on_desktop_gpu.sh sleep 20
#   scripts/run_on_desktop_gpu.sh --detach some/long/training/script.sh
#
# IMPORTANT (tmux availability, 2026-07-18 finding): the desktop has no
# passwordless sudo, so `apt install tmux` cannot be scripted. tmux was
# installed userspace-only via:
#   apt-get download tmux libutempter0   (works without root -- fetches
#                                          .deb to cwd, no install step)
#   dpkg -x <deb> ~/.local/opt/tmux-extracted   (extract, no root needed)
# A wrapper at ~/.local/bin/tmux sets LD_LIBRARY_PATH to the extracted
# libutempter.so.0 and execs the real binary. Because non-interactive SSH
# commands don't source ~/.bashrc, ~/.local/bin is NOT on PATH for these
# sessions -- this script always invokes tmux by its full remote path
# ($REMOTE_HOME/.local/bin/tmux), never bare `tmux`. If tmux is ever
# properly apt-installed on the desktop later, that full path still
# resolves (dpkg-installed tmux would live at /usr/bin/tmux instead --
# in that case just update REMOTE_TMUX below, or symlink /usr/bin/tmux
# into ~/.local/bin).
#
# IMPORTANT (systemd-inhibit polkit gap, 2026-07-18 finding): on this
# desktop, `systemd-inhibit --what=shutdown:sleep:idle` gets "Failed to
# inhibit: Access denied" when invoked from a non-interactive SSH session
# (polkit's default rules only implicitly allow login1 inhibit actions
# for sessions with a seat / active local login -- an SSH session has
# neither, and there's no polkit auth agent to prompt interactively, and
# no passwordless sudo to add a rule granting it). `--what=idle` alone
# DOES succeed from SSH. The remote job script below therefore PROBES
# with a cheap `true` command first: if the full shutdown:sleep:idle
# inhibitor can be acquired, the real job uses it; if not, it falls back
# to idle-only and prints a loud WARNING into the job's own log (both
# locally and streamed to the Pi side) so a degraded run is never
# mistaken for a fully-protected one. Either way `--who=rl-gpu-job`
# still shows up in `systemd-inhibit --list`, so check_desktop_gpu.sh's
# BUSY detection (which only checks presence-by-name, not "what") is
# unaffected by the fallback.
#
# The one-time real fix (needs root on the desktop, not doable from this
# unattended SSH session -- see CLAUDE.md's "Pi-as-primary GPU dispatch"
# section for the exact polkit rule to install) would restore full
# shutdown/sleep protection; until that's done, treat shutdown/sleep as
# NOT actually blocked during dispatched jobs -- only idle-suspend is.
#
# Exit codes (deliberately distinct so a caller can branch correctly --
# same philosophy as check_desktop_gpu.sh's 0/1/2 scheme, extended here):
#   0 = success: desktop was available, job dispatched, remote command
#       exited 0 (or, with --detach, the job was successfully started).
#   1 = desktop BUSY per check_desktop_gpu.sh -- nothing was dispatched.
#   2 = desktop UNKNOWN (unreachable / check itself failed) per
#       check_desktop_gpu.sh -- nothing was dispatched. NOT a green
#       light; caller should fall back (e.g. to cloud) or stop.
#   3 = usage error (no command given).
#   4 = dispatch mechanism itself failed (SSH error setting up the
#       remote script/tmux session/inhibitor -- distinct from the job's
#       own exit code; the desktop WAS available, something else broke).
#   5 = the remote command itself ran and exited non-zero. The exact
#       remote exit code is printed to stderr; this wrapper always exits
#       5 in this case (shell exit codes don't reliably round-trip
#       arbitrary integers), so check the printed message for specifics.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ServerAliveInterval/CountMax: fail a stale connection (e.g. desktop
# rebooted mid-stream) within ~30s instead of hanging forever -- found
# for real 2026-07-18: a batch of these SSH streaming calls survived a
# desktop reboot as zombie connections, still "alive" per the local
# kernel with no data flowing, blocking their callers indefinitely.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
REMOTE_HOME="/home/saps"
REMOTE_TMUX="${REMOTE_HOME}/.local/bin/tmux"

DETACH=0
if [ "${1:-}" = "--detach" ]; then
  DETACH=1
  shift
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [--detach] <command> [args...]" >&2
  exit 3
fi

# --- 1. availability pre-check -------------------------------------------
# NOTE: must be an `if`, not a bare `VAR="$(cmd)"` assignment -- under
# `set -e`, a plain assignment statement whose command substitution exits
# non-zero (BUSY=1 / UNKNOWN=2) aborts the whole script right there before
# the informative messages below ever print (hit this for real: the
# refusal exit code was correct but the message was silently missing).
# Wrapping in `if` puts it in a tested context, which `set -e` exempts.
if CHECK_OUT="$("$SCRIPT_DIR/check_desktop_gpu.sh" 2>&1)"; then
  CHECK_RC=0
else
  CHECK_RC=$?
fi
if [ "$CHECK_RC" -eq 1 ]; then
  echo "REFUSING TO DISPATCH: desktop GPU is BUSY." >&2
  echo "$CHECK_OUT" >&2
  exit 1
elif [ "$CHECK_RC" -eq 2 ]; then
  echo "REFUSING TO DISPATCH: desktop availability is UNKNOWN (unreachable or check failed)." >&2
  echo "$CHECK_OUT" >&2
  echo "Not a green light -- fall back to cloud or stop, do not treat as available." >&2
  exit 2
elif [ "$CHECK_RC" -ne 0 ]; then
  echo "REFUSING TO DISPATCH: check_desktop_gpu.sh exited unexpected code $CHECK_RC." >&2
  echo "$CHECK_OUT" >&2
  exit 2
fi

# --- 2. build the remote job script locally, quoting the command safely --
SESSION="rl-gpu-job-$(date -u +%Y%m%dT%H%M%SZ)-$$"
REMOTE_SCRIPT="/tmp/${SESSION}.sh"
REMOTE_LOG="/tmp/${SESSION}.log"
DONE_MARKER="__RL_GPU_JOB_DONE__"

CMD_QUOTED=""
for arg in "$@"; do
  CMD_QUOTED+="$(printf '%q ' "$arg")"
done

LOCAL_TMP_SCRIPT="$(mktemp)"
# This trap is widened below (section 4) once STREAM_FIFO/SSH_TAIL_PID
# exist, to also reap the streaming ssh process -- EXIT traps replace
# rather than stack in bash, so only the widened one at the bottom
# actually runs; this one only fires if the script exits before that
# point (e.g. before dispatch even starts).
trap 'rm -f "$LOCAL_TMP_SCRIPT"' EXIT

cat > "$LOCAL_TMP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -o pipefail
# Probe whether the full shutdown:sleep:idle inhibitor can actually be
# acquired from this (likely non-seated SSH) session before running the
# real job -- see the 2026-07-18 polkit-access-denied note at the top of
# run_on_desktop_gpu.sh. The probe is a no-op ("true") so it can't buffer
# or delay real job output; only the fallback decision is made up front.
if systemd-inhibit --what=shutdown:sleep:idle --who=rl-gpu-job \\
    --why="RL GPU job probe (session ${SESSION})" true 2>/dev/null; then
  INHIBIT_WHAT="shutdown:sleep:idle"
else
  echo "WARNING: full shutdown:sleep:idle inhibitor was denied by polkit for this session -- falling back to idle-only. Shutdown/sleep are NOT blocked for this job. See CLAUDE.md 'rl-gpu-job inhibitor' notes for the one-time desktop-side polkit fix."
  INHIBIT_WHAT="idle"
fi
systemd-inhibit --what="\$INHIBIT_WHAT" --who=rl-gpu-job \\
  --why="RL GPU job dispatched from Pi (session ${SESSION})" -- \\
  ${CMD_QUOTED}
echo "${DONE_MARKER} exit=\$?"
EOF

echo "Dispatching to desktop: ${CMD_QUOTED}" >&2
echo "Session: ${SESSION}" >&2
echo "Remote log: ${REMOTE_LOG}" >&2

# --- 3. ship the script + launch it in a detached tmux session -----------
if ! ssh "${SSH_OPTS[@]}" desktop "cat > '${REMOTE_SCRIPT}'" < "$LOCAL_TMP_SCRIPT"; then
  echo "DISPATCH FAILED: could not copy job script to desktop." >&2
  exit 4
fi

if ! ssh "${SSH_OPTS[@]}" desktop \
    "${REMOTE_TMUX} new-session -d -s '${SESSION}' \"bash '${REMOTE_SCRIPT}' > '${REMOTE_LOG}' 2>&1\""; then
  echo "DISPATCH FAILED: could not start remote tmux session." >&2
  exit 4
fi

# Confirm the job actually started, rather than trusting the previous
# command's exit code alone. Poll for EITHER the tmux session being up OR
# the log file existing -- a fast-finishing command (e.g. `echo hi`) can
# complete and tear down its tmux session (tmux destroys a session when
# its pane's process exits) within a fraction of a second, faster than a
# single post-hoc has-session check can reliably catch (hit this for real
# testing this script against a plain `echo`). The log file existing is
# proof the script ran either way.
DISPATCH_CONFIRMED=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ssh "${SSH_OPTS[@]}" desktop \
      "${REMOTE_TMUX} has-session -t '${SESSION}' 2>/dev/null || test -e '${REMOTE_LOG}'"; then
    DISPATCH_CONFIRMED=1
    break
  fi
  sleep 0.5
done
if [ "$DISPATCH_CONFIRMED" -eq 0 ]; then
  echo "DISPATCH FAILED: neither tmux session '${SESSION}' nor its log came up on desktop." >&2
  exit 4
fi

echo "Dispatched: tmux session '${SESSION}' is running on desktop." >&2

if [ "$DETACH" -eq 1 ]; then
  echo "Detach mode: not waiting for completion." >&2
  echo "Check later with:" >&2
  echo "  ssh desktop ${REMOTE_TMUX} attach -t ${SESSION}    # attach live" >&2
  echo "  ssh desktop tail -f ${REMOTE_LOG}                  # tail the log" >&2
  echo "  scripts/check_desktop_gpu.sh                       # confirm still busy" >&2
  exit 0
fi

# --- 4. blocking mode: stream the remote log live until done -------------
# This runs as its own separate SSH invocation from the tmux session that's
# actually running the job. If THIS ssh/tail is interrupted (Ctrl-C, this
# wrapper's own timeout, Pi-side network drop), only the tail dies -- the
# remote job keeps running under tmux + systemd-inhibit, unaffected. That
# separation is the whole point of not just backgrounding over the raw SSH
# pipe.
#
# Uses an explicit background job + FIFO (not `< <(ssh ...)` process
# substitution) so the streaming ssh's PID is known and can be killed
# once the loop is done -- 2026-07-18 finding: process substitution's
# command is never reaped after the local `while read` loop `break`s on
# the DONE_MARKER, so every successful dispatch leaked one orphaned
# `ssh ... tail -f` process forever (confirmed: 13 accumulated over one
# session, several outliving a desktop reboot as zombie connections that
# hung any caller piping this script's output through another buffering
# command like `| tail -N`, since that command blocks for EOF that never
# comes while the leaked process still holds the pipe's write end open).
STREAM_FIFO="$(mktemp -u)"
mkfifo "$STREAM_FIFO"
trap 'rm -f "$LOCAL_TMP_SCRIPT" "$STREAM_FIFO"; [ -n "${SSH_TAIL_PID:-}" ] && kill "$SSH_TAIL_PID" 2>/dev/null' EXIT
ssh "${SSH_OPTS[@]}" desktop "tail -n +1 -f '${REMOTE_LOG}'" > "$STREAM_FIFO" &
SSH_TAIL_PID=$!

REMOTE_EXIT=""
while IFS= read -r line; do
  echo "$line"
  if [[ "$line" == "${DONE_MARKER} exit="* ]]; then
    REMOTE_EXIT="${line#${DONE_MARKER} exit=}"
    break
  fi
done < "$STREAM_FIFO"

kill "$SSH_TAIL_PID" 2>/dev/null
wait "$SSH_TAIL_PID" 2>/dev/null
SSH_TAIL_PID=""

if [ -z "$REMOTE_EXIT" ]; then
  echo "DISPATCH WARNING: log stream ended without a completion marker (SSH dropped?)." >&2
  echo "The job may still be running on desktop under tmux session '${SESSION}'." >&2
  echo "Check with: ssh desktop ${REMOTE_TMUX} attach -t ${SESSION}" >&2
  exit 4
fi

if [ "$REMOTE_EXIT" -eq 0 ]; then
  echo "Remote command succeeded (exit 0)." >&2
  exit 0
else
  echo "REMOTE COMMAND FAILED: exited with code ${REMOTE_EXIT} on desktop." >&2
  echo "Full log: ssh desktop cat ${REMOTE_LOG}" >&2
  exit 5
fi

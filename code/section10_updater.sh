#!/bin/bash
# Keeps notebook Section 10 (pass^2 test) up to date on its own, independent of any Claude session.
#   ./section10_updater.sh start    start the updater detached (every 30 s); exits by itself when both runs are done
#   ./section10_updater.sh status   show whether it is running and the last line of its log
#   ./section10_updater.sh stop     stop it
#   ./section10_updater.sh once     refresh the notebook one time and exit
# It stops while the Mac sleeps and resumes when it wakes (it re-reads everything from the pod).
cd "$(dirname "$0")" || exit 1
PY=./.venv-nb/bin/python
PID=section10_updater.pid
LOG=section10_updater.log

running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }

case "$1" in
  start)
    if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
    nohup "$PY" -u live_passk_section.py --interval 30 >> "$LOG" 2>&1 &
    echo $! > "$PID"
    disown 2>/dev/null
    echo "started (pid $(cat $PID)), log: $LOG"
    ;;
  stop)
    if running; then kill "$(cat "$PID")"; echo "stopped"; else echo "not running"; fi
    rm -f "$PID"
    ;;
  status)
    if running; then echo "running (pid $(cat $PID))"; else echo "not running"; fi
    tail -n 1 "$LOG" 2>/dev/null
    ;;
  once)
    "$PY" live_passk_section.py --once
    ;;
  *)
    echo "usage: $0 start|stop|status|once"
    exit 1
    ;;
esac

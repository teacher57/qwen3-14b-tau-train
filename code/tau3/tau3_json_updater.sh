#!/bin/bash
# Keeps full_run_results/tau3_run.json current, independent of any Claude session. Exits when the run finishes or fails.
#   ./tau3_json_updater.sh start | stop | status | once
cd "$(dirname "$0")" || exit 1
PID=updater.pid; LOG=updater.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
         nohup python3 -u update_tau3_json.py >> "$LOG" 2>&1 & echo $! > "$PID"; disown 2>/dev/null
         echo "started (pid $(cat $PID)), log: tau3/$LOG" ;;
  stop)  if running; then kill "$(cat "$PID")"; echo stopped; else echo "not running"; fi; rm -f "$PID" ;;
  status) if running; then echo "running (pid $(cat $PID))"; else echo "not running"; fi; tail -n 3 "$LOG" 2>/dev/null ;;
  once)  python3 update_tau3_json.py --once ;;
  *) echo "usage: $0 start|stop|status|once"; exit 1 ;;
esac

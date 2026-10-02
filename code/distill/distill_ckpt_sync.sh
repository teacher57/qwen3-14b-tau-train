#!/bin/bash
# Copies the SFT checkpoints to saves/distill-* as they appear, independent of any Claude session.
#   ./distill_ckpt_sync.sh start | stop | status
cd "$(dirname "$0")" || exit 1
PID=ckpt_sync.pid
LOG=ckpt_sync.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start)
    if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
    nohup python3 -u sync_distill_checkpoints.py --interval 60 >> "$LOG" 2>&1 &
    echo $! > "$PID"; disown 2>/dev/null
    echo "started (pid $(cat $PID)), log: teacher/$LOG" ;;
  stop)
    if running; then kill "$(cat "$PID")"; echo stopped; else echo "not running"; fi
    rm -f "$PID" ;;
  status)
    if running; then echo "running (pid $(cat $PID))"; else echo "not running"; fi
    tail -n 3 "$LOG" 2>/dev/null; ls -d ../saves/distill-* 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac

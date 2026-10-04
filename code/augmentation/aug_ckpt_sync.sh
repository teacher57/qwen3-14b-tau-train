#!/bin/bash
# Copies the augmented-SFT checkpoints to sft_aug/saves/aug-* as they appear, independent of any Claude session.  ./aug_ckpt_sync.sh start|stop|status
cd "$(dirname "$0")" || exit 1
PID=ckpt_sync.pid; LOG=ckpt_sync.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) running && { echo "already running (pid $(cat $PID))"; exit 0; }; nohup python3 -u sync_aug_checkpoints.py --interval 60 >> "$LOG" 2>&1 & echo $! > "$PID"; disown 2>/dev/null; echo "started (pid $(cat $PID)), log: sft_aug/$LOG" ;;
  stop) running && kill "$(cat "$PID")"; rm -f "$PID"; echo stopped ;;
  status) running && echo "running (pid $(cat $PID))" || echo "not running"; tail -n 3 "$LOG" 2>/dev/null; ls -d saves/aug-* 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status" ;;
esac

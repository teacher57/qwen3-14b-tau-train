#!/bin/bash
# stop training cleanly (no half-written checkpoint) if the disk is nearly full
while pgrep -f '^python3 /root/grpo_gentle_train.py' >/dev/null; do
  free=$(df --output=avail -m / | tail -1)
  if [ "$free" -lt 400 ]; then
    echo "disk free ${free}MB -> stopping trainer" > /root/STOPPED_DISK
    pkill -f '^python3 /root/grpo_gentle_train.py'
    break
  fi
  sleep 60
done

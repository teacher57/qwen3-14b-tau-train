#!/bin/bash
# Runs AFTER the main chain finishes (never run in parallel with it): the combo tasks the 14B does not reliably solve
# that were not in the first lists. Hard ones (never solved, 12) first, then the sometimes-solved ones (37).
source /root/.taubench_env
cd /root/tau-bench
while ! grep -q TEACHER_CHAIN_DONE /root/teacher.log 2>/dev/null; do sleep 30; done
python3 /root/teacher_rollouts.py /root/extra_never12.json 4 0.7 32 teacher32b 2 >> /root/teacher.log 2>&1
python3 /root/teacher_rollouts.py /root/extra_mixed37.json 4 0.7 32 teacher32b 2 >> /root/teacher.log 2>&1
echo TEACHER_ALL_DONE >> /root/teacher.log
